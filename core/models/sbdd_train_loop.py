import copy
import numpy as np
import torch

from time import time
from torch.profiler import profile, record_function, ProfilerActivity

import pytorch_lightning as pl

from torch_scatter import scatter_mean, scatter_sum

from core.config.config import Config
from core.models.bfn4sbdd import BFN4SBDDScoreModel

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


class SBDDTrainLoop(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config
        self.dynamics = BFN4SBDDScoreModel(**self.cfg.dynamics.todict())
        # [ time, h_t, pos_t, edge_index]
        self.train_losses = []
        self.save_hyperparameters(self.cfg.todict())
        self.time_records = np.zeros(6)
        self.log_time = False

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        t1 = time()
        protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand = (
            batch.protein_pos,
            batch.protein_atom_feature.float(),
            batch.protein_element_batch,
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

        with torch.no_grad():
            # add noise to protein_pos
            protein_noise = torch.randn_like(protein_pos) * self.cfg.train.pos_noise_std
            gt_protein_pos = batch.protein_pos + protein_noise
            # random rotation as data aug
            if self.cfg.train.random_rot:
                M = np.random.randn(3, 3)
                Q, __ = np.linalg.qr(M)
                Q = torch.from_numpy(Q.astype(np.float32)).to(ligand_pos.device)
                gt_protein_pos = gt_protein_pos @ Q
                ligand_pos = ligand_pos @ Q

        num_graphs = batch_protein.max().item() + 1
        # !!!!!
        gt_protein_pos, ligand_pos, _ = center_pos(
            gt_protein_pos,
            ligand_pos,
            batch_protein,
            batch_ligand,
            mode=self.cfg.dynamics.center_pos_mode,
        )  # TODO: ugly

        # gt_protein_pos = gt_protein_pos / self.cfg.data.normalizer

        t3 = time()
        t = torch.rand(
            [num_graphs, 1], dtype=ligand_pos.dtype, device=ligand_pos.device
        ).index_select(
            0, batch_ligand
        )  # different t for different molecules.

        if not self.cfg.dynamics.use_discrete_t and not self.cfg.dynamics.destination_prediction:
            # t = torch.randint(0, 999, [num_graphs, 1], dtype=ligand_pos.dtype, device=ligand_pos.device).index_select(0, batch_ligand) #different t for different molecules.
            # t = t / 1000.0
            # else:
            t = torch.clamp(t, min=self.dynamics.t_min)  # clamp t to [t_min,1]

        t4 = time()
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
        )

        pos_loss, type_loss, bond_loss, charge_loss = (
            losses['closs'],
            losses['dloss'],
            losses['dloss_bond'],
            losses['discretized_loss'],
        )

        # here the discretised_loss is close for current version.

        loss = torch.mean(pos_loss + self.cfg.train.v_loss_weight * type_loss + self.cfg.train.bond_loss_weight * bond_loss + charge_loss)

        t5 = time()
        self.log_dict(
            {
                'lr': self.get_last_lr(),
                'loss': loss.item(), 
                'loss_pos': pos_loss.mean().item(), 
                'loss_type': type_loss.mean().item(),
                'loss_bond': bond_loss.mean().item(),
            },
            on_step=True,
            prog_bar=False,
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
        out_data_list = self.shared_sampling_step(batch, batch_idx, sample_num_atoms='ref', desc=f'Val')
        return out_data_list
    
    def test_step(self, batch, batch_idx):
        # TODO change order, samples of the same pocket should be together, reduce protein loading
        out_data_list = []
        n_samples = self.cfg.evaluation.num_samples
        for _ in range(n_samples):
            batch_output = self.shared_sampling_step(batch, batch_idx, sample_num_atoms=self.cfg.evaluation.sample_num_atoms, 
                                                     desc=f'Test-{_}/{n_samples}')
            # for idx, item in enumerate(batch_output):
            out_data_list.append(batch_output)
                
        out_data_list_reorder = []
        for i in range(len(out_data_list[0])):
            for j in range(len(out_data_list)):
                out_data_list_reorder.append(out_data_list[j][i])
        return out_data_list_reorder

    def shared_sampling_step(self, batch, batch_idx, sample_num_atoms, desc=''):
        # here we need to sample the molecules in the validation step
        protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand = (
            batch.protein_pos,
            batch.protein_atom_feature.float(),
            batch.protein_element_batch,
            batch.ligand_pos,
            batch.ligand_atom_feature_full,
            batch.ligand_element_batch,
        )
        
        num_graphs = batch_protein.max().item() + 1  # B
        n_nodes = batch_ligand.size(0)  # N_lig
        assert num_graphs == len(batch), f"num_graphs: {num_graphs} != len(batch): {len(batch)}"

        # move protein center to origin & ligand correspondingly
        protein_pos, ligand_pos, offset = center_pos(
            protein_pos,
            ligand_pos,
            batch_protein,
            batch_ligand,
            mode=self.cfg.dynamics.center_pos_mode,
        )  # TODO: ugly

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
            ligand_fc_bond_index = batch.ligand_fc_bond_index
            batch_ligand_bond = batch.ligand_fc_bond_type_batch
        else:
            raise ValueError(f"sample_num_atoms mode: {sample_num_atoms} not supported")
        ligand_cum_atoms = torch.cat([
            torch.tensor([0], dtype=torch.long, device=ligand_pos.device), 
            ligand_num_atoms.cumsum(dim=0)
        ])


        # TODO reuse for visualization and test
        # forward pass to get the ligand sample
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
            # ligand_pos=ligand_pos,  # for debug only
        )

        # restore ligand to original position
        final = sample_chain[-1]  # mu_pos_final, k_final, k_hat_final
        pred_pos, one_hot, pred_charge, pred_bond_onehot = (
            final[0] + offset[batch_ligand], 
            final[1], final[2], final[3]
        )

        # along with normalizer
        pred_pos = pred_pos * torch.tensor(
            self.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=ligand_pos.device
        )
        out_batch = copy.deepcopy(batch)
        out_batch.protein_pos = out_batch.protein_pos * torch.tensor(
            self.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=ligand_pos.device
        )

        pred_v = one_hot.argmax(dim=-1)
        # TODO: ugly, should be done in metrics.py (but needs a way to make it compatible with pyg batch)
        pred_atom_type = trans.get_atomic_number_from_index(
            pred_v, mode=self.cfg.data.transform.ligand_atom_mode
        ) # List[int]

        # for visualization
        atom_type = [trans.MAP_ATOM_TYPE_ONLY_TO_INDEX[i] for i in pred_atom_type]  # List[int]
        atom_type = torch.tensor(atom_type, dtype=torch.long, device=ligand_pos.device)  # [N_lig]

        # for reconstruction
        pred_aromatic = trans.is_aromatic_from_index(
            pred_v, mode=self.cfg.data.transform.ligand_atom_mode
        ) # List[bool]

        # print('[DEBUG]', num_graphs, len(ligand_cum_atoms))
        # for bond generation
        if self.dynamics.bond_bfn:
            pred_bond = pred_bond_onehot.argmax(dim=-1)  # [N_lig * N_lig]
            ligand_bond_array = pred_bond.cpu().numpy()
            ligand_bond_index_array = ligand_fc_bond_index.cpu().numpy()
            # cum_bonds = batch.ligand_fc_bond_type_ptr
            ligand_num_bonds = scatter_sum(torch.ones_like(batch_ligand_bond),
                                            batch_ligand_bond).tolist()
            cum_bonds = np.cumsum([0] + ligand_num_bonds)

        molist = []
        for i in range(num_graphs):
            try:
                if not self.dynamics.bond_bfn:
                    mol = reconstruct.reconstruct_from_generated(
                        xyz=pred_pos[batch_ligand == i].cpu().numpy().astype(np.float64),
                        atomic_nums=pred_atom_type[ligand_cum_atoms[i]:ligand_cum_atoms[i + 1]],
                        # aromatic=pred_aromatic[ligand_cum_atoms[i]:ligand_cum_atoms[i + 1]],
                    )
                else:
                    pred_bond_index = ligand_bond_index_array[:, cum_bonds[i]:cum_bonds[i + 1]] - ligand_cum_atoms[i].cpu().numpy()
                    pred_bond_index = pred_bond_index.tolist()
                    # assert all index is in the range of the ligand
                    assert all([0 <= x < ligand_num_atoms[i] for x in pred_bond_index[0]]), f"pred_bond_index@{i}: {pred_bond_index}"
                    mol = reconstruct.reconstruct_from_generated_with_bond(
                        xyz=pred_pos[batch_ligand == i].cpu().numpy().astype(np.float64),
                        atomic_nums=pred_atom_type[ligand_cum_atoms[i]:ligand_cum_atoms[i + 1]],
                        bond_index=pred_bond_index,
                        bond_type=ligand_bond_array[cum_bonds[i]:cum_bonds[i + 1]],
                    )
            except reconstruct.MolReconsError:
                mol = None
            molist.append(mol)

        # add necessary dict to new pyg batch
        out_batch.x, out_batch.pos = atom_type, pred_pos
        # out_batch.bond = pred_bond
        out_batch.atom_type = torch.tensor(pred_atom_type, dtype=torch.long, device=ligand_pos.device)
        out_batch.mol = molist

        # TODO: add slice dict for bond
        _slice_dict = {
            "x": ligand_cum_atoms,
            "pos": ligand_cum_atoms,
            "atom_type": ligand_cum_atoms,
            "mol": out_batch._slice_dict["ligand_filename"],
            # "bond": cum_bonds,
        }
        _inc_dict = {
            "x": out_batch._inc_dict["ligand_element"], # [0] * B, TODO: figure out what this is
            "pos": out_batch._inc_dict["ligand_pos"],
            "atom_type": out_batch._inc_dict["ligand_element"],
            "mol": out_batch._inc_dict["ligand_filename"],
            # "bond": out_batch._inc_dict["ligand_fc_bond_type"],
        }
        if self.cfg.data.transform.ligand_atom_mode == 'add_aromatic':
            out_batch.is_aromatic = torch.tensor(pred_aromatic, dtype=torch.long, device=ligand_pos.device)
            _slice_dict["is_aromatic"] = ligand_cum_atoms
            _inc_dict["is_aromatic"] = out_batch._inc_dict["ligand_element"]
        
        out_batch._inc_dict.update(_inc_dict)
        out_batch._slice_dict.update(_slice_dict)
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
        )
        self.train_losses = []

    def configure_optimizers(self):
        self.optim = get_optimizer(self.cfg.train.optimizer, self)
        self.scheduler, self.get_last_lr = get_scheduler(self.cfg.train, self.optim)

        return {
            'optimizer': self.optim, 
            'lr_scheduler': self.scheduler,
        }
