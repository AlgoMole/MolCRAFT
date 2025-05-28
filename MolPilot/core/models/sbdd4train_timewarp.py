# This file contains the failed attempt to implement a trainable timewarp model for SBDD4Train.

import copy
import numpy as np
import torch
import wandb

from time import time
from typing import Any

import pytorch_lightning as pl

from torch_scatter import scatter_mean, scatter_sum

from core.config.config import Config
from core.models.bfn4sbdd import BFN4SBDDScoreModel
from core.models.time_scheduler_mono import ConstrainedFunction, StrictlyMonotonicNN

import core.evaluation.utils.atom_num as atom_num
import core.utils.transforms as trans
import core.utils.reconstruct as reconstruct

from core.utils.train import get_optimizer, get_scheduler


def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode="protein"):
    if mode == "none":
        offset = 0.0
        pass
    elif mode == "protein":
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset

def log_gradient_scales_to_wandb(loss_name, loss, model, logger, log_prefix="grad/"):
    """
    Logs gradient norms of model parameters to WandB.
    
    Args:
        loss_name (str): The name of the current loss component.
        loss (torch.Tensor): The loss tensor to backpropagate.
        model (torch.nn.Module): The model containing parameters.
        logger (wandb or LightningLogger): Logger for recording metrics.
        log_prefix (str): Prefix for logged keys.
    """
    model.zero_grad()
    
    loss.backward(retain_graph=True)
    
    grad_norms = []
    total_grad_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm(2).item()
            grad_norms.append((name, grad_norm))
            total_grad_norm += grad_norm ** 2
    
    total_grad_norm = total_grad_norm ** 0.5
    
    wandb.log(
        {f"{log_prefix}grad_norm_{loss_name}": total_grad_norm},
    )

    # logger.log_metrics({f"{log_prefix}{loss_name}/total_grad_norm": total_grad_norm})

    # for each parameter
    # for name, norm in grad_norms:
    #     logger.log_metrics({f"{log_prefix}{loss_name}/{name}": norm})


def compute_perturbation_impact(perturb_name, loss, loss_original, model, sigma):
    impact = (loss - loss_original).abs().item()

    wandb.log(
        {f"{str(perturb_name)}_delta": impact},
    )


def low_discrepancy_sampler(num_samples, device):
    """
    inspired from the variational diffusion paper 
    """
    single_u = torch.rand((1,), device=device, requires_grad=False, dtype=torch.float64)
    return (single_u + torch.arange(0., 1., step = 1./num_samples, device=device, requires_grad=False)) % 1


class SBDD4Train(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config
        self.dynamics = BFN4SBDDScoreModel(**self.cfg.dynamics.todict())
        # [ time, h_t, pos_t, edge_index]
        self.train_losses = []
        self.save_hyperparameters(self.cfg.todict())
        self.time_records = np.zeros(6)
        self.log_time = False
        self.include_protein = 'zinc' not in self.cfg.data.path
        self.num_invalid_gradients = 0
        self.log_grad = False
        # self.timewarp_cdf = Timewarp(**self.cfg.timewarp.todict())
        self.mono_nn = StrictlyMonotonicNN(input_dim=1, hidden_dim=16, num_hidden_layers=2, output_dim=2)
        self.timewarp_cdf = ConstrainedFunction(self.mono_nn)
        self.time_scheduler = None

    def configure_timewarp(self, timewarp_cdf):
        self.timewarp_cdf = timewarp_cdf

    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    print(f'WARNING: NaN ({torch.isnan(param.grad).any()}) or Inf ({torch.isinf(param.grad).any()}) gradients encountered after calling backward for parameter {name}. Setting to zero.')
                    break

        if not valid_gradients:
            self.num_invalid_gradients += 1
            self.zero_grad()

    def forward(self, x):
        pass

    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        r"""Overrides the PyTorch Lightning backward step and adds the OOM check."""
        try:
            loss.backward(*args, **kwargs)
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print('| WARNING: ran OOM error, skipping batch. Exception:', str(e))
                for p in self.dynamics.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
            else:
                raise e

    def training_step(self, batch, batch_idx):
        t1 = time()
        protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand = (
            getattr(batch, "protein_pos", None),
            batch.protein_atom_feature.float() if hasattr(batch, "protein_atom_feature") else None,
            getattr(batch, "protein_element_batch", None),
            batch.ligand_pos,
            batch.ligand_atom_feature_full,
            batch.ligand_element_batch,
        )  # get the data from the batch
        # batch is a data object
        # protein_pos: [N_pro,3]
        # protein_v: [N_pro,27]
        # batch_protein: [N_pro]
        # ligand_pos: [N_lig,3]
        # ligand_v: [N_lig,13]
        # protein_element_batch: [N_protein]

        t2 = time()
        num_graphs = batch_ligand.max().item() + 1

        if protein_pos is not None:
            with torch.no_grad():
                if self.cfg.train.pos_noise_std > 0:
                    # add noise to protein_pos
                    protein_noise = torch.randn_like(protein_pos) * self.cfg.train.pos_noise_std
                    protein_pos = batch.protein_pos + protein_noise
                # random rotation as data aug
                if self.cfg.train.random_rot:
                    M = np.random.randn(3, 3)
                    Q, __ = np.linalg.qr(M)
                    Q = torch.from_numpy(Q.astype(np.float32)).to(ligand_pos.device)
                    protein_pos = protein_pos @ Q
                    ligand_pos = ligand_pos @ Q

            # !!!!!
            protein_pos, ligand_pos, _ = center_pos(
                protein_pos,
                ligand_pos,
                batch_protein,
                batch_ligand,
                mode=self.cfg.dynamics.center_pos_mode,
            )  # TODO: ugly
        else:
            _, ligand_pos, _ = center_pos(
                ligand_pos,
                ligand_pos,
                batch_ligand,
                batch_ligand,
                mode=self.cfg.dynamics.center_pos_mode,
            )
            perturb_offset = torch.rand(1) * self.cfg.data.normalizer_dict.pos
            perturb_offset = perturb_offset.to(ligand_pos.device)
            ligand_pos = ligand_pos + perturb_offset
        # gt_protein_pos = gt_protein_pos / self.cfg.data.normalizer

        t3 = time()
        
        u = torch.rand(
            [num_graphs, 1], dtype=ligand_pos.dtype, device=ligand_pos.device, requires_grad=True
        )  
        # u = torch.rand(1, device=ligand_pos.device, requires_grad=True).unsqueeze(0).repeat(num_graphs, 1)
        # different t for different molecules.
        # u = low_discrepancy_sampler(num_samples=num_graphs, device=ligand_pos.device).unsqueeze(1)

        # inverse transform to get t = F^{-1}(u)
        # by default, the sigmas are inverted to [sigma_min, sigma_max]
        # normalize sigmas to [0,1]
        # torch.autograd.set_detect_anomaly(True)

        # print('batch', batch_idx)
        sigmas = self.timewarp_cdf(u)
        # print('u', u.requires_grad, u.device)
        # print('sigmas', sigmas.requires_grad, sigmas.device)
        assert sigmas.shape == (num_graphs, self.timewarp_cdf.num_features)

        # clamp sigmas to [t_min,1]
        if not self.cfg.dynamics.use_discrete_t and not self.cfg.dynamics.destination_prediction:
            sigmas = torch.clamp(sigmas, min=self.dynamics.t_min)
        
        if not getattr(self.cfg.train, 'docking_only', False):
            t, t_pos = sigmas[:, 0].unsqueeze(1), sigmas[:, 1].unsqueeze(1)
            t, t_pos = 1 - t, 1 - t_pos
            # print('t', t.requires_grad, t.device)
        else:
            raise NotImplementedError("Docking only not supported")
            t, t_pos = torch.ones_like(sigmas[:, 0]).unsqueeze(1) - 1e-7, torch.rand_like(sigmas[:, 1]).unsqueeze(1)
            sigmas = torch.cat([1 - t, 1 - t_pos], dim=1)


        # if self.cfg.time_decoupled:
        #     t_pos = torch.rand(
        #         [num_graphs, 1], dtype=ligand_pos.dtype, device=ligand_pos.device
        #     )  # different t for different modalities
        #     if self.cfg.decouple_mode == 'triangle':
        #         # make [t, t_pos] form a triangle instead of a square [0, 1] x [0, 1]
        #         t_pos = t_pos * t # t_pos <= t
        #     elif self.cfg.decouple_mode == 'clip':
        #         t_pos = torch.clamp(t_pos, max=t)
        # else:
        #     t_pos = t

        t4 = time()
        try:
            losses = self.dynamics.loss_one_step(
                t,
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                ligand_pos=ligand_pos,
                ligand_v=ligand_v,
                batch_ligand=batch_ligand,
                ligand_bond_type=getattr(batch, "ligand_fc_bond_type"),
                ligand_bond_index=getattr(batch, "ligand_fc_bond_index"),
                batch_ligand_bond=getattr(batch, "ligand_fc_bond_type_batch"),
                include_protein=self.include_protein,
                t_pos=t_pos,
                log_grad=self.log_grad and hasattr(self.cfg.train, "log_gradient_scale_interval") and self.global_step % self.cfg.train.log_gradient_scale_interval == 0,
                batch_mean=True,
            )
 
            pos_loss, type_loss, bond_loss = (
                losses['closs'],
                losses['dloss'],
                losses['dloss_bond'],
            )

            bfn_losses = torch.mean(pos_loss + self.cfg.train.v_loss_weight * type_loss + self.cfg.train.bond_loss_weight * bond_loss)
        
            loss = bfn_losses

        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print(f"Skipping batch {batch_idx} due to CUDA OOM.")
                for p in self.dynamics.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                return None
            else:
                raise e

        t5 = time()
        self.log_dict(
            {
                'lr': self.get_last_lr(),
                'loss': loss.item(), 
                'loss_pos': pos_loss.mean().item(), 
                'loss_type': type_loss.mean().item(),
                'loss_bond': bond_loss.mean().item(),
                't_2D': t.mean().item(),
                't_3D': t_pos.mean().item(),
            },
            on_step=True,
            prog_bar=True,
            batch_size=self.cfg.train.batch_size,
        )

        # check if loss is finite, skip update if not
        if not torch.isfinite(loss):
            return None
        self.train_losses.append(loss.clone().detach().cpu())

        t0 = time()

        if self.log_time:
            self.time_records = np.vstack((self.time_records, [t0, t1, t2, t3, t4, t5]))
            print(f'step total time: {self.time_records[-1, 0] - self.time_records[-1, 1]}, batch size: {num_graphs}')
            print(f'\tpl call & data access: {self.time_records[-1, 1] - self.time_records[-2, 0]}')
            print(f'\tunwrap data: {self.time_records[-1, 2] - self.time_records[-1, 1]}')
            print(f'\tadd noise & center pos: {self.time_records[-1, 3] - self.time_records[-1, 2]}')
            print(f'\tsample t: {self.time_records[-1, 4] - self.time_records[-1, 3]}')
            print(f'\tget loss: {self.time_records[-1, 5] - self.time_records[-1, 4]}')
            print(f'\tlogging: {self.time_records[-1, 0] - self.time_records[-1, 5]}')
        return loss

    def validation_step(self, batch, batch_idx):
        if not hasattr(self.cfg.train, 'val_mode') or self.cfg.train.val_mode == 'sample':
            tmp_timewarp_cdf = self.timewarp_cdf
            self.timewarp_cdf = None
            outputs = self.shared_sampling_step(batch, batch_idx, sample_num_atoms='ref')
            self.timewarp_cdf = tmp_timewarp_cdf
            return outputs

        protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand = (
            getattr(batch, "protein_pos", None),
            batch.protein_atom_feature.float() if hasattr(batch, "protein_atom_feature") else None,
            getattr(batch, "protein_element_batch", None),
            batch.ligand_pos,
            batch.ligand_atom_feature_full,
            batch.ligand_element_batch,
        )  # get the data from the batch

        num_graphs = batch_ligand.max().item() + 1

        if protein_pos is not None:
            with torch.no_grad():
                if self.cfg.train.pos_noise_std > 0:
                    # add noise to protein_pos
                    protein_noise = torch.randn_like(protein_pos) * self.cfg.train.pos_noise_std
                    protein_pos = batch.protein_pos + protein_noise
                # random rotation as data aug
                if self.cfg.train.random_rot:
                    M = np.random.randn(3, 3)
                    Q, __ = np.linalg.qr(M)
                    Q = torch.from_numpy(Q.astype(np.float32)).to(ligand_pos.device)
                    protein_pos = protein_pos @ Q
                    ligand_pos = ligand_pos @ Q

            # !!!!!
            protein_pos, ligand_pos, _ = center_pos(
                protein_pos,
                ligand_pos,
                batch_protein,
                batch_ligand,
                mode=self.cfg.dynamics.center_pos_mode,
            )  # TODO: ugly
        else:
            _, ligand_pos, _ = center_pos(
                ligand_pos,
                ligand_pos,
                batch_ligand,
                batch_ligand,
                mode=self.cfg.dynamics.center_pos_mode,
            )
            perturb_offset = torch.rand(1) * self.cfg.data.normalizer_dict.pos
            perturb_offset = perturb_offset.to(ligand_pos.device)
            ligand_pos = ligand_pos + perturb_offset

        u = torch.rand(
            [num_graphs, 1], dtype=ligand_pos.dtype, device=ligand_pos.device
        )  # different t for different molecules.

        # inverse transform to get t = F^{-1}(u)
        # by default, the sigmas are inverted to [sigma_min, sigma_max]
        # normalize sigmas to [0,1]
        sigmas = self.timewarp_cdf(u).to(torch.float32)
        assert sigmas.shape == (num_graphs, self.timewarp_cdf.num_features)

        # clamp sigmas to [t_min,1]
        if not self.cfg.dynamics.use_discrete_t and not self.cfg.dynamics.destination_prediction:
            sigmas = torch.clamp(sigmas, min=self.dynamics.t_min)
        if not getattr(self.cfg.train, 'docking_only', False):
            t, t_pos = sigmas[:, 0].unsqueeze(1), sigmas[:, 1].unsqueeze(1)
            t, t_pos = 1 - t, 1 - t_pos
        else:
            raise NotImplementedError("Docking only not supported")
            t, t_pos = torch.ones_like(sigmas[:, 0]).unsqueeze(1) - 1e-7, torch.rand_like(sigmas[:, 1]).unsqueeze(1)
            sigmas = torch.cat([1 - t, 1 - t_pos], dim=1)


        try:
            # if True:
            with torch.no_grad():
                losses = self.dynamics.loss_one_step(
                    t,
                    protein_pos=protein_pos,
                    protein_v=protein_v,
                    batch_protein=batch_protein,
                    ligand_pos=ligand_pos,
                    ligand_v=ligand_v,
                    batch_ligand=batch_ligand,
                    ligand_bond_type=getattr(batch, "ligand_fc_bond_type"),
                    ligand_bond_index=getattr(batch, "ligand_fc_bond_index"),
                    batch_ligand_bond=getattr(batch, "ligand_fc_bond_type_batch"),
                    include_protein=self.include_protein,
                    t_pos=t_pos,
                    log_grad=self.log_grad and hasattr(self.cfg.train, "log_gradient_scale_interval") and self.global_step % self.cfg.train.log_gradient_scale_interval == 0,
                    batch_mean=False,
                    recon_loss=True,
                )
 
            pos_loss, type_loss, bond_loss, pos_loss_mse, type_loss_ce, bond_loss_ce = (
                losses['closs'],
                losses['dloss'],
                losses['dloss_bond'],
                losses['closs_mse'],
                losses['dloss_ce'],
                losses['dloss_bond_ce'],
            )

            bfn_losses = torch.mean(pos_loss + self.cfg.train.v_loss_weight * type_loss + self.cfg.train.bond_loss_weight * bond_loss)

            if getattr(self.cfg.train, 'freeze_dynamics', False):
                bfn_losses = torch.zeros_like(bfn_losses)

            loss = bfn_losses
            loss = loss.unsqueeze(0)

        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print(f"Skipping batch {batch_idx} due to CUDA OOM.")
                for p in self.dynamics.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                return None
            else:
                raise e

        return loss

    def test_step(self, batch, batch_idx):
        out_data_list = []
        for _ in range(self.cfg.evaluation.num_samples):
            out_data_list.extend(self.shared_sampling_step(batch, batch_idx, sample_num_atoms=self.cfg.evaluation.sample_num_atoms))
        return out_data_list

    def shared_sampling_step(self, batch, batch_idx, sample_num_atoms):
        # here we need to sample the molecules in the validation step
        
        protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand = (
            getattr(batch, "protein_pos", None),
            batch.protein_atom_feature.float() if hasattr(batch, "protein_atom_feature") else None,
            getattr(batch, "protein_element_batch", None),
            batch.ligand_pos,
            batch.ligand_atom_feature_full,
            batch.ligand_element_batch,
        )
        
        num_graphs = batch_ligand.max().item() + 1  # B
        n_nodes = batch_ligand.size(0)  # N_lig
        assert num_graphs == len(batch), f"num_graphs: {num_graphs} != len(batch): {len(batch)}"


        # move protein center to origin & ligand correspondingly
        if protein_pos is not None:
            protein_pos, ligand_pos, offset = center_pos(
                protein_pos,
                ligand_pos,
                batch_protein,
                batch_ligand,
                mode=self.cfg.dynamics.center_pos_mode,
            )  # TODO: ugly
        else:
            _, ligand_pos, offset = center_pos(
                torch.zeros_like(ligand_pos),
                ligand_pos,
                batch_ligand,
                batch_ligand,
                mode=self.cfg.dynamics.center_pos_mode,
            )

        # determine the number of atoms in the ligand
        if sample_num_atoms == 'prior':
            ligand_num_atoms = []
            ligand_fc_bond_indices = []
            ligand_num_edges = []
            for data_id in range(len(batch)):
                data = batch[data_id]
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy() * self.cfg.data.normalizer_dict.pos)
                n_atoms = atom_num.sample_atom_num(pocket_size).astype(int)
                ligand_num_atoms.append(n_atoms)

                # Add the computed bond index to the list
                full_dst = torch.repeat_interleave(torch.arange(n_atoms), n_atoms)
                full_src = torch.arange(n_atoms).repeat(n_atoms)
                mask = full_dst != full_src
                full_dst, full_src = full_dst[mask], full_src[mask]
                # Shift the indices to the correct position
                if len(ligand_num_atoms) > 1:
                    full_dst += sum(ligand_num_atoms[:-1])
                    full_src += sum(ligand_num_atoms[:-1])
                ligand_fc_bond_index = torch.stack([full_src, full_dst], dim=0)
                assert ligand_fc_bond_index.size(0) == 2 and ligand_fc_bond_index.size(1) == n_atoms * (n_atoms - 1)
                ligand_fc_bond_indices.append(ligand_fc_bond_index)
                ligand_num_edges.append(ligand_fc_bond_index.size(1))

            batch_ligand = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(ligand_num_atoms)).to(ligand_pos.device)
            ligand_num_atoms = torch.tensor(ligand_num_atoms, dtype=torch.long, device=ligand_pos.device)
            batch_ligand_bond = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(ligand_num_edges)).to(ligand_pos.device)
            ligand_fc_bond_index = torch.cat(ligand_fc_bond_indices, dim=1).to(ligand_pos.device).long()
            assert ligand_fc_bond_index.size(1) == sum(ligand_num_edges)

        elif sample_num_atoms == 'ref':
            batch_ligand = batch.ligand_element_batch
            ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).to(ligand_pos.device)
            if hasattr(batch, "ligand_fc_bond_index"):
                ligand_fc_bond_index = batch.ligand_fc_bond_index
                batch_ligand_bond = batch.ligand_fc_bond_type_batch
            else:
                ligand_fc_bond_index = None
                batch_ligand_bond = None
        else:
            raise ValueError(f"sample_num_atoms mode: {sample_num_atoms} not supported")
        ligand_cum_atoms = torch.cat([
            torch.tensor([0], dtype=torch.long, device=ligand_pos.device), 
            ligand_num_atoms.cumsum(dim=0)
        ])

        # Construct reversed u steps
        sample_steps = self.cfg.evaluation.sample_steps
        u_steps = torch.linspace(1, 0, sample_steps + 1, device=self.device, dtype=torch.float32)

        if self.timewarp_cdf is not None:
            raise ValueError("timewarp_cdf not supported in validation")
            # Warp u to t' (sigma)
            t_steps = self.timewarp_cdf(u_steps, invert=True).detach().to(torch.float32)
            t_steps = (t_steps - self.timewarp_cdf.sigma_min) / (self.timewarp_cdf.sigma_max - self.timewarp_cdf.sigma_min)
            # Reverse t' to get t = 1 - t'
            t_steps = 1 - t_steps
        elif self.time_scheduler is not None:
            # TODO
            t_steps = self.time_scheduler / self.time_scheduler.max()
            assert t_steps.shape == (sample_steps + 1, 2), f"t_steps: {t_steps.shape}"
        else:
            t_steps = 1 - u_steps
            t_steps = t_steps.unsqueeze(-1).repeat(1, 2)

        # TODO reuse for visualization and test
        # forward pass to get the ligand sample
        if not hasattr(self.cfg.evaluation, "docking_rmsd") or not self.cfg.evaluation.docking_rmsd:
            # ligand_com = scatter_mean(ligand_pos, batch_ligand, dim=0)
            # pos_grad_weight = getattr(self.cfg.evaluation, "pos_grad_weight", 0.0)
            theta_chain, sample_chain, y_chain = self.dynamics.sample(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
                ligand_bond_index=ligand_fc_bond_index,
                batch_ligand_bond=batch_ligand_bond,
                # n_nodes=n_nodes,
                sample_steps=self.cfg.evaluation.sample_steps,
                n_nodes=num_graphs,
                include_protein=self.include_protein,
                t_steps=t_steps
                # if is validating, disable timewarp_cdf
                # pos_grad_weight=pos_grad_weight,
                # ligand_com=ligand_com,
                # ligand_pos=ligand_pos,  # for debug only
            )
        else:
            theta_chain, sample_chain, y_chain = self.dynamics.sample(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
                ligand_bond_index=ligand_fc_bond_index,
                batch_ligand_bond=batch_ligand_bond,
                n_nodes=num_graphs,
                sample_steps=self.cfg.evaluation.sample_steps,
                # condition on the ligand type and bond type
                ligand_v=ligand_v,
                ligand_bond_type=getattr(batch, "ligand_fc_bond_type"),
                include_protein=self.include_protein,
                t_steps=t_steps
            )

        # restore ligand to original position
        final = sample_chain[-1]  # mu_pos_final, k_final, k_hat_final
        pred_pos, one_hot, pred_charge, pred_bond_pmf = (
            final[0] + offset[batch_ligand], 
            final[1], final[2], final[3]
        )

        # along with normalizer
        pred_pos = pred_pos * torch.tensor(
            self.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=ligand_pos.device
        )
        out_batch = copy.deepcopy(batch)
        if protein_pos is not None:
            out_batch.protein_pos = out_batch.protein_pos * torch.tensor(
                self.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=ligand_pos.device
            )

        pred_v = one_hot.argmax(dim=-1)
        if pred_charge is not None:
            pred_charge = pred_charge.argmax(dim=-1)  # [N_lig]
            assert pred_v.shape == pred_charge.shape, f"pred_v: {pred_v.shape}, pred_charge: {pred_charge.shape}"
        # TODO: ugly, should be done in metrics.py (but needs a way to make it compatible with pyg batch)
        pred_atom_type = trans.get_atomic_number_from_index(
            pred_v, mode=self.cfg.data.transform.ligand_atom_mode
        ) # List[int]

        # for visualization
        if self.cfg.data.transform.ligand_atom_mode == 'basic_PDB' or self.cfg.data.transform.ligand_atom_mode == 'basic_plus_charge_PDB':
            atom_type = [trans.MAP_ATOM_TYPE_ONLY_TO_INDEX_PDB[i] for i in pred_atom_type]
        else:
            atom_type = [trans.MAP_ATOM_TYPE_ONLY_TO_INDEX[i] for i in pred_atom_type]  # List[int]
        atom_type = torch.tensor(atom_type, dtype=torch.long, device=ligand_pos.device)  # [N_lig]

        # for reconstruction
        pred_aromatic = trans.is_aromatic_from_index(
            pred_v, mode=self.cfg.data.transform.ligand_atom_mode
        ) # List[bool]

        # for bond generation
        if self.dynamics.bond_bfn:
            pred_bond = pred_bond_pmf.argmax(dim=-1)  # [N_lig * N_lig]
            if self.dynamics.pred_connectivity:
                pred_connectivity = final[4].argmax(dim=-1) # 1 stands for connected
                pred_bond = pred_bond * pred_connectivity
            if self.dynamics.num_bond_classes == 6:
                pred_bond = (pred_bond / 2).ceil().long() # 0, 1, 2, 3, 4, 5 -> 0, 1, 1, 2, 2, 3
            ligand_bond_array = pred_bond.cpu().numpy()
            ligand_num_bonds = scatter_sum(torch.ones_like(batch_ligand_bond),
                                            batch_ligand_bond).tolist()
            cum_bonds = np.cumsum([0] + ligand_num_bonds)
            # remove the offset to get the bond index
            ligand_fc_bond_index = ligand_fc_bond_index - ligand_cum_atoms[batch_ligand_bond]
            ligand_bond_index_array = ligand_fc_bond_index.cpu().numpy()

        molist = []
        for i in range(num_graphs):
            try:
                if not self.dynamics.bond_bfn:
                    if self.cfg.evaluation.fix_bond or pred_aromatic is None:
                        mol = reconstruct.reconstruct_from_generated(
                            xyz=pred_pos[batch_ligand == i].cpu().numpy().astype(np.float64),
                            atomic_nums=pred_atom_type[ligand_cum_atoms[i]:ligand_cum_atoms[i + 1]],
                        )
                    else:
                        mol = reconstruct.reconstruct_from_generated(
                            xyz=pred_pos[batch_ligand == i].cpu().numpy().astype(np.float64),
                            atomic_nums=pred_atom_type[ligand_cum_atoms[i]:ligand_cum_atoms[i + 1]],
                            aromatic=pred_aromatic[ligand_cum_atoms[i]:ligand_cum_atoms[i + 1]],
                            basic_mode=False
                        )
                else:
                    # pred_bond_index = ligand_bond_index_array[:, cum_bonds[i]:cum_bonds[i + 1]] - ligand_cum_atoms[i].cpu().numpy()
                    pred_bond_index = ligand_bond_index_array[:, cum_bonds[i]:cum_bonds[i + 1]]
                    pred_bond_index = pred_bond_index.tolist()

                    # loop over the bonds, set the bond type according to the bond_pmf
                    # for atom i-j, set the bond_pmf to the (bond_pmf[j, i] + bond_pmf[i, j]) / 2
                    # pred_bond_i is of shape [N_lig * (N_lig - 1), 5]
                    # pred_bond_i = pred_bond_pmf[batch_ligand_bond == i]
                    # bond_matrix = torch.zeros(ligand_num_atoms[i], ligand_num_atoms[i], self.cfg.dynamics.net_config.num_bond_classes, device=pred_bond_i.device)
                    # for bond_id, bond_pmf in enumerate(pred_bond_i):
                    #     node_i, node_j = int(pred_bond_index[0][bond_id]), int(pred_bond_index[1][bond_id])
                    #     bond_matrix[node_i, node_j] = bond_pmf

                    # for bond_id, bond_pmf in enumerate(pred_bond_i):
                    #     node_i, node_j = int(pred_bond_index[0][bond_id]), int(pred_bond_index[1][bond_id])
                    #     pred_bond_i[bond_id] = (bond_pmf + bond_matrix[node_j, node_i]) / 2

                    # pred_bond_i = pred_bond_i.argmax(dim=-1)
                    # pred_bond_i = pred_bond_i.argmax(dim=-1)
                    # pred_bond_array = pred_bond_i.cpu().numpy()

                    pred_bond_array = ligand_bond_array[cum_bonds[i]:cum_bonds[i + 1]]
                    assert all([0 <= x < ligand_num_atoms[i] for x in pred_bond_index[0]]), f"pred_bond_index@{i}: {pred_bond_index}"
                    # assert all index is in the range of the ligand

                    # for charge generation
                    if hasattr(self.cfg.dynamics, "ligand_atom_charge_dim") and self.cfg.dynamics.ligand_atom_charge_dim > 0:
                        pred_charge_i = pred_charge[ligand_cum_atoms[i]:ligand_cum_atoms[i + 1]] - 1
                        pred_charge_i = pred_charge_i.int().cpu().tolist()
                    else:
                        pred_charge_i = None
                    if self.cfg.evaluation.fix_bond:
                        mol = reconstruct.reconstruct_from_generated_with_bond_aromatic(
                            xyz=pred_pos[batch_ligand == i].cpu().numpy().astype(np.float64),
                            atomic_nums=pred_atom_type[ligand_cum_atoms[i]:ligand_cum_atoms[i + 1]],
                            bond_index=pred_bond_index,
                            bond_type=pred_bond_array,
                            aromatic=pred_aromatic[ligand_cum_atoms[i]:ligand_cum_atoms[i + 1]],
                            charges=pred_charge_i,
                        )
                        # mol = reconstruct.reconstruct_from_generated_with_bond(
                        #     xyz=pred_pos[batch_ligand == i].cpu().numpy().astype(np.float64),
                        #     atomic_nums=pred_atom_type[ligand_cum_atoms[i]:ligand_cum_atoms[i + 1]],
                        #     bond_index=pred_bond_index,
                        #     # bond_type=ligand_bond_array[cum_bonds[i]:cum_bonds[i + 1]],
                        #     bond_type=pred_bond_array,
                        #     charges=pred_charge_i,
                        # )
                    else:
                        mol = reconstruct.reconstruct_from_generated_with_bond_basic(
                            xyz=pred_pos[batch_ligand == i].cpu().numpy().astype(np.float64),
                            atomic_nums=pred_atom_type[ligand_cum_atoms[i]:ligand_cum_atoms[i + 1]],
                            bond_index=pred_bond_index,
                            # bond_type=ligand_bond_array[cum_bonds[i]:cum_bonds[i + 1]],
                            bond_type=pred_bond_array,
                            charges=pred_charge_i,
                        )
            except reconstruct.MolReconsError:
                mol = None
            molist.append(mol)
        
        # add necessary dict to new pyg batch
        out_batch.x, out_batch.pos = atom_type, pred_pos
        out_batch.atom_type = torch.tensor(pred_atom_type, dtype=torch.long, device=ligand_pos.device)
        out_batch.mol = molist

        _slice_dict = {
            "x": ligand_cum_atoms,
            "pos": ligand_cum_atoms,
            "atom_type": ligand_cum_atoms,
            "mol": out_batch._slice_dict["ligand_filename"],
        }
        _inc_dict = {
            "x": out_batch._inc_dict["ligand_element"], # [0] * B, TODO: figure out what this is
            "pos": out_batch._inc_dict["ligand_pos"],
            "atom_type": out_batch._inc_dict["ligand_element"],
            "mol": out_batch._inc_dict["ligand_filename"],
        }

        if self.dynamics.bond_bfn:
            out_batch.bond = pred_bond
            _slice_dict["bond"] = cum_bonds
            _inc_dict["bond"] = out_batch._inc_dict["ligand_fc_bond_type"]
            out_batch.bond_index = ligand_fc_bond_index
            _slice_dict["bond_index"] = cum_bonds
            _inc_dict["bond_index"] = out_batch._inc_dict["ligand_fc_bond_type"]

        if self.cfg.data.transform.ligand_atom_mode == 'add_aromatic':
            out_batch.is_aromatic = torch.tensor(pred_aromatic, dtype=torch.long, device=ligand_pos.device)
            _slice_dict["is_aromatic"] = ligand_cum_atoms
            _inc_dict["is_aromatic"] = out_batch._inc_dict["ligand_element"]
        
        out_batch._inc_dict.update(_inc_dict)
        out_batch._slice_dict.update(_slice_dict)
        # move to cpu
        out_batch = out_batch.detach().cpu()
        out_data_list = out_batch.to_data_list()
        return out_data_list

    def on_train_epoch_end(self) -> None:
        if len(self.train_losses) == 0:
            epoch_loss = 0
        else:
            epoch_loss = torch.stack([x for x in self.train_losses]).mean()
        print(f"epoch_loss: {epoch_loss}")
        self.log(
            "epoch_loss",
            epoch_loss,
            batch_size=self.cfg.train.batch_size,
            sync_dist=True,
        )
        self.train_losses = []

    def configure_optimizers(self):
        self.optim = get_optimizer(self.cfg.train.optimizer, self)
        self.scheduler, self.get_last_lr = get_scheduler(self.cfg.train, self.optim)

        return {
            'optimizer': self.optim, 
            'lr_scheduler': self.scheduler,
        }
