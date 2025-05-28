from absl import logging
import wandb

import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from core.config.config import Struct
from core.datasets.pl_data import get_batch_connectivity_matrix, get_batch_type_pmf_matrix
from core.models.common import compose_context, ShiftedSoftplus, GaussianSmearing
from core.models.bfn_base import BFNBase
from core.models.uni_transformer import UniTransformerO2TwoUpdateGeneral
from core.models.uni_transformer_edge import UniTransformerO2TwoUpdateGeneralBond

def compute_input_grad_norms(ligand_pos, ligand_type, ligand_bond_type, model, total_loss):
    model.zero_grad()
    
    total_loss.backward(retain_graph=True)
    
    grad_norms = {}
    for modality_name, modality_input in zip(
        ["ligand_pos", "ligand_bond_type", "ligand_atom_type"],
        [ligand_pos, ligand_bond_type, ligand_type],
    ):
        assert modality_input.requires_grad, f"{modality_name} input should require grad"
        grad_norm = modality_input.grad.norm(2).item()
        grad_norms[f"grad/{modality_name}"] = grad_norm

    wandb.log(grad_norms)
    model.zero_grad()

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RBF(nn.Module):
    def __init__(self, start, end, n_center):
        super().__init__()
        self.start = start
        self.end = end
        self.n_center = n_center
        self.centers = torch.linspace(start, end, n_center)
        self.width = (end - start) / n_center

    def forward(self, x):
        assert x.ndim >= 2
        out = (x - self.centers.to(x.device)) / self.width
        ret = torch.exp(-0.5 * out**2)
        return F.normalize(ret, dim=-1, p=1) * 2 - 1


class TimeEmbedLayer(nn.Module):
    def __init__(self, time_emb_mode, time_emb_dim):
        super().__init__()
        self.time_emb_mode = time_emb_mode
        self.time_emb_dim = time_emb_dim

        if self.time_emb_mode == "simple":
            assert self.time_emb_dim == 1
            self.time_emb = lambda x: x
        elif self.time_emb_mode == "sin":
            self.time_emb = nn.Sequential(
                SinusoidalPosEmb(self.time_emb_dim),
                nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                nn.GELU(),
                nn.Linear(self.time_emb_dim * 4, self.time_emb_dim),
            )
        elif self.time_emb_mode == "rbf":
            self.time_emb = RBF(0, 1, self.time_emb_dim)
        elif self.time_emb_mode == "rbfnn":
            self.time_emb = nn.Sequential(
                RBF(0, 1, self.time_emb_dim),
                nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                nn.GELU(),
                nn.Linear(self.time_emb_dim * 4, self.time_emb_dim),
            )
        else:
            raise NotImplementedError

    def forward(self, t):
        return self.time_emb(t)


class BFN4SBDDScoreModel(BFNBase):
    def __init__(
        self,
        # in_node_nf,
        # hidden_nf=64,
        net_config,
        protein_atom_feature_dim,
        ligand_atom_feature_dim,
        sigma1_coord,
        beta1,
        beta1_bond=None,
        beta1_charge=None,
        beta1_aromatic=None,
        ligand_atom_type_dim=None,
        ligand_atom_charge_dim=0,
        ligand_atom_aromatic_dim=0,
        device="cuda",
        condition_time=True,
        bond_net_type=None,
        use_discrete_t=False,
        discrete_steps=1000,
        t_min=0.0001,
        # no_diff_coord=False,
        node_indicator=True,
        # charge_discretised_loss = False
        time_emb_mode='simple',
        time_emb_dim=1,
        center_pos_mode='protein',
        pos_init_mode='zero',
        destination_prediction=False,
        sampling_strategy="vanilla",
        pred_given_all=False,
        num_atoms_max=65,
        pred_connectivity=False,
        self_condition=False,
    ):
        super(BFN4SBDDScoreModel, self).__init__()
        net_config = Struct(**net_config)
        self.config = net_config
        self.num_atoms_max = num_atoms_max
  
        self.hidden_dim = net_config.hidden_dim
        self.ligand_atom_feature_dim = ligand_atom_feature_dim
        if ligand_atom_type_dim is None:
            ligand_atom_type_dim = ligand_atom_feature_dim
        self.num_classes = ligand_atom_type_dim
        self.num_charge = ligand_atom_charge_dim
        self.num_aromatic = ligand_atom_aromatic_dim

        self.pred_given_all = pred_given_all

        if net_config.name == 'unio2net':
            self.unio2net = UniTransformerO2TwoUpdateGeneral(**net_config.todict())
            self.bond_bfn = False
        elif 'bond' in net_config.name:
            if net_config.name == 'unio2net_bond':
                self.unio2net = UniTransformerO2TwoUpdateGeneralBond(bond_net_type=bond_net_type, **net_config.todict())
            else:
                raise ValueError(net_config.name)

            self.num_bond_classes = self.config.num_bond_classes
            self.ligand_bond_emb = nn.Linear(self.num_bond_classes, self.hidden_dim)
            self.bond_bfn = True
            # include bond
            self.bond_net_type = bond_net_type
            self.distance_expansion = GaussianSmearing(0., 5., num_gaussians=self.config.num_r_gaussian, fix_offset=False)
            if self.bond_net_type == 'pre_att':
                bond_input_dim = self.config.num_r_gaussian + self.hidden_dim
            elif self.bond_net_type == 'flowmol':
                bond_input_dim = self.config.num_r_gaussian + 2 * self.ligand_atom_feature_dim
            elif self.bond_net_type == 'lin':
                bond_input_dim = self.hidden_dim
                # apply MLP over concatenated features
                if self.pred_given_all:
                    bond_input_dim *= net_config.num_layers + 1
            elif self.bond_net_type == 'semla':
                bond_input_dim = self.config.num_r_gaussian + 3 * self.hidden_dim
            elif self.bond_net_type == 'lin+x':
                bond_input_dim = self.hidden_dim + 1
            else:
                raise ValueError(self.bond_net_type)
            self.bond_inference = nn.Sequential(
                nn.Linear(bond_input_dim, self.hidden_dim),
                ShiftedSoftplus(),
                nn.Linear(self.hidden_dim, self.num_bond_classes)
            )
        else:
            raise NotImplementedError(net_config.name)

        self.node_indicator = node_indicator

        if self.node_indicator:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        # atom embedding
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)
        self.center_pos_mode = center_pos_mode  # ['none', 'protein']

        self.time_emb_mode = time_emb_mode
        # self.time_emb_dim = time_emb_dim
        # if self.time_emb_dim > 0:
        #     self.time_emb_layer = TimeEmbedLayer(self.time_emb_mode, self.time_emb_dim)
        if hasattr(self.unio2net, 'adaptive_norm') and self.unio2net.adaptive_norm:
            self.time_emb_dim = 0
        else:
            self.time_emb_dim = time_emb_dim
            if self.time_emb_dim > 0:
                self.time_emb_layer = TimeEmbedLayer(self.time_emb_mode, self.time_emb_dim)
        
        self.ligand_atom_emb = nn.Linear(
            ligand_atom_feature_dim + self.time_emb_dim, emb_dim
        )

        # self.refine_net_type = config.model_type
        # self.refine_net = get_refine_net(self.refine_net_type, config)
        if self.pred_given_all:
            type_input_dim = self.hidden_dim * (net_config.num_layers + 1)
        else:
            type_input_dim = self.hidden_dim

        self.v_inference = nn.Sequential(
            nn.Linear(type_input_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, self.ligand_atom_feature_dim),
        )  # [hidden to 13]

        self.device = device
        self._edges_dict = {}
        self.condition_time = condition_time
        self.sigma1_coord = torch.tensor(sigma1_coord, dtype=torch.float32, device=device)  # coordinate sigma1, a schedule for bfn
        self.beta1 = torch.tensor(beta1, dtype=torch.float32, device=device)  # type beta, a schedule for atom types.
        if beta1_bond is not None:
            self.beta1_bond = torch.tensor(beta1_bond, dtype=torch.float32, device=device)  # bond beta, a schedule for bond types.
        if beta1_charge is not None:
            self.beta1_charge = torch.tensor(beta1_charge, dtype=torch.float32, device=device)  # charge beta, a schedule for charge types.
        if beta1_aromatic is not None:
            self.beta1_aromatic = torch.tensor(beta1_aromatic, dtype=torch.float32, device=device)  # aromatic beta, a schedule for aromatic types.
        self.use_discrete_t = use_discrete_t  # whether to use discrete t
        self.discrete_steps = discrete_steps
        self.t_min = t_min
        self.pos_init_mode = pos_init_mode
        self.destination_prediction = destination_prediction
        self.sampling_strategy = sampling_strategy
        # self.no_diff_coord = no_diff_coord #whether the output minus the inputs for the graph neural networks.
        self.pred_connectivity = pred_connectivity
        if self.pred_connectivity:
            self.connectivity_inference = nn.Sequential(
                nn.Linear(self.config.num_r_gaussian + 2 * self.ligand_atom_feature_dim + self.num_bond_classes, self.hidden_dim),
                ShiftedSoftplus(),
                nn.Linear(self.hidden_dim, 2)
            )

        self.self_condition = self_condition

    # the same signature as interdependency_modeling
    def forward(self, time, protein_pos, protein_v, batch_protein, theta_h_t, mu_pos_t, theta_bond_t, ligand_bond_index, batch_ligand, batch_ligand_bond, gamma_coord, include_protein, return_all=False, fix_x=False, ligand_atom_mask=None, t_pos=None):
        theta_h_t = 2 * theta_h_t - 1  # from 1/K \in [0,1] to 2/K-1 \in [-1,1]
        init_ligand_v = theta_h_t
        if self.bond_bfn:
            E = self.num_bond_classes  # bond_atom_feature_dim
            theta_bond_t = 2 * theta_bond_t - 1  # from 1/K \in [0,1] to 2/K-1 \in [-1,1]
            if self.config.name == 'unio2net+bond' and self.num_atoms_max > 0:
                # unbatch theta_bond_t to [N, N, E]
                ligand_bond_matrix = get_batch_type_pmf_matrix(batch_ligand, ligand_bond_index, theta_bond_t, batch_ligand_bond, padding=True, num_atoms_max=self.num_atoms_max)
                init_ligand_bond = torch.cat([matrix.view(len(matrix), -1) for matrix in ligand_bond_matrix], dim=0).to(theta_bond_t.device)
                init_ligand_v = torch.cat([init_ligand_v, init_ligand_bond], -1)
        # time embedding [simple, sin, rbf, learn]
        if self.time_emb_dim > 0:
            time_emb = self.time_emb_layer(time)
            input_ligand_feat = torch.cat([init_ligand_v, time_emb], -1)
        else:
            input_ligand_feat = init_ligand_v

        if protein_pos is not None: 
            h_protein = self.protein_atom_emb(protein_v)  # [N_protein, self.hidden_dim - 1]

        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)  # [N_ligand, self.hidden_dim - 1]
        # init_ligand_h = input_ligand_feat # TODO: no embedding for ligand atoms, check whether this make sense.

        if self.node_indicator:
            if protein_pos is not None:
                h_protein = torch.cat(
                    [h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1
                )  # [N_ligand, self.hidden_dim ]
            init_ligand_h = torch.cat(
                [init_ligand_h, torch.ones(len(init_ligand_h), 1).to(init_ligand_h)], -1
            )  # [N_ligand, self.hidden_dim]

        if protein_pos is not None:
            h_all, pos_all, batch_all, mask_ligand, mask_ligand_atom, p_index_in_ctx, l_index_in_ctx = compose_context(
                h_protein=h_protein,
                h_ligand=init_ligand_h,
                pos_protein=protein_pos,
                pos_ligand=mu_pos_t,
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
                ligand_atom_mask=ligand_atom_mask,
            )
        else:
            h_all, pos_all, batch_all = init_ligand_h, mu_pos_t, batch_ligand
            mask_ligand = torch.ones([batch_ligand.size(0)], device=batch_ligand.device).bool()
        # get the context for the protein and ligand, while the ligand is h is noisy (h_t)/ pos is also the noise version. (pos_t)

        # ---------------------

        if ligand_bond_index is not None:
            if protein_pos is not None:
                bond_index_in_all = l_index_in_ctx[ligand_bond_index]
            else:
                bond_index_in_all = ligand_bond_index
        else:
            bond_index_in_all = None

        # time = 2 * time - 1
        node_time = time[batch_all].squeeze(-1)
        h_bond = None
        if self.bond_bfn and self.config.name != 'unio2net+bond' and self.config.name != 'unio2net_bond_hybrid':
            bond_time = time[batch_ligand_bond].squeeze(-1)
            # bond_type = F.one_hot(theta_bond_t, num_classes=self.num_bond_classes).float()
            bond_type = theta_bond_t
            h_bond = self.ligand_bond_emb(bond_type)
            if 'unio2net_bond_twist' in self.config.name:
                pos_time = t_pos[batch_all].squeeze(-1)
                outputs = self.unio2net(
                    h=h_all, x=pos_all, group_idx=None,
                    bond_index=bond_index_in_all, h_bond=h_bond,
                    mask_ligand=mask_ligand,
                    # mask_ligand_atom=mask_ligand_atom,  # dummy node is marked as 0
                    batch=batch_all,
                    node_time=node_time,
                    bond_time=bond_time,
                    include_protein=include_protein,
                    return_all=self.pred_given_all,
                    pos_time=pos_time
                )
            else:
                outputs = self.unio2net(
                    h=h_all, x=pos_all, group_idx=None,
                    bond_index=bond_index_in_all, h_bond=h_bond,
                    mask_ligand=mask_ligand,
                    # mask_ligand_atom=mask_ligand_atom,  # dummy node is marked as 0
                    batch=batch_all,
                    node_time=node_time,
                    bond_time=bond_time,
                    include_protein=include_protein,
                    return_all=self.pred_given_all
                )
        elif self.config.name == 'unio2net_bond_hybrid':
            bond_type = theta_bond_t
            h_bond = self.ligand_bond_emb(bond_type)
            outputs = self.unio2net(
                h=h_all, x=pos_all, 
                mask_ligand=mask_ligand,
                batch=batch_all,
                bond_index=bond_index_in_all, h_bond=h_bond,
                fix_x=fix_x,
                return_all=self.pred_given_all
            )
        else:
            outputs = self.unio2net(
                h_all, pos_all, mask_ligand, batch_all, fix_x=fix_x, return_all=self.pred_given_all
            )

        final_pos, final_h = (
            outputs["x"],
            outputs["h"],
        )  # shape of the pos and shape of h
        final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand]
        if self.pred_given_all:
            all_h = outputs['all_h']
            final_ligand_h = torch.cat([h[mask_ligand] for h in all_h], dim=-1)

        # 1. for continuous, network outputs eps_hat(θ, t)
        # Eq.(84): x_hat(θ, t) = μ / γ(t) − \sqrt{(1 − γ(t)) / γ(t)} * eps_hat(θ, t)
        if not self.destination_prediction:
            coord_pred = (
                mu_pos_t / gamma_coord
                - torch.sqrt((1 - gamma_coord) / gamma_coord) * final_ligand_pos
            )
            coord_pred = torch.where(
                time < self.t_min, torch.zeros_like(mu_pos_t), coord_pred
            )
        else:
            coord_pred = final_ligand_pos #add destination prediction. 

        final_ligand_v = self.v_inference(final_ligand_h)  # [N_ligand, ligand_atom_feature_dim]


        k_hat = torch.zeros_like(mu_pos_t)  # TODO: here we close the
        final_ligand_connectivity = None

        # bond inference input
        if self.bond_bfn:
            if self.bond_net_type != 'lin':
                src, dst = bond_index_in_all
                dist = torch.norm(final_pos[dst] - final_pos[src], p=2, dim=-1, keepdim=True)
                r_feat = self.distance_expansion(dist)
                if self.bond_net_type == 'pre_att':
                    hi, hj = final_h[dst], final_h[src]
                    bond_inf_input = torch.cat([r_feat, (hi + hj) / 2], -1)
                elif self.bond_net_type == 'flowmol':
                    src, dst = ligand_bond_index
                    vi, vj = final_ligand_v[src], final_ligand_v[dst]
                    bond_inf_input = torch.cat([r_feat, vi, vj], -1)
                elif self.bond_net_type == 'semla':
                    hi, hj = final_h[dst], final_h[src]
                    if h_bond == None:
                        bond_type = theta_bond_t
                        h_bond = self.ligand_bond_emb(bond_type)
                    bond_inf_input = torch.cat([r_feat, h_bond, hi, hj], -1)
                elif self.bond_net_type == 'lin+x':
                    bond_inf_input = torch.cat([outputs['h_bond'], dist], -1)
                else:
                    raise ValueError(self.bond_net_type)                
            elif self.bond_net_type == 'lin':
                if self.pred_given_all:
                    bond_inf_input = torch.cat(outputs['all_h_bond'], dim=-1)
                else:
                    bond_inf_input = outputs['h_bond']

            final_ligand_e = self.bond_inference(bond_inf_input)
            if self.pred_connectivity:
                src, dst = bond_index_in_all
                dist = torch.norm(final_pos[dst] - final_pos[src], p=2, dim=-1, keepdim=True)
                r_feat = self.distance_expansion(dist)
                # hi, hj = final_h[dst], final_h[src]
                src, dst = ligand_bond_index
                vi, vj = final_ligand_v[dst], final_ligand_v[src]
                bond_inf_input = torch.cat([r_feat, vi, vj, final_ligand_e], -1)
                final_ligand_connectivity = self.connectivity_inference(bond_inf_input)
        else:
            final_ligand_e = None


        if torch.isnan(final_ligand_h).any():
            print("final_ligand_h logits contain NaN")
        if torch.isinf(final_ligand_h).any():
            print("final_ligand_h logits contain Inf")
            
        return {
            'final_ligand_pos': final_ligand_pos,
            'final_ligand_v': final_ligand_v,
            'final_ligand_connectivity': final_ligand_connectivity,
            'final_h': final_h,
            'final_ligand_h': final_ligand_h,
            'final_ligand_e': final_ligand_e,
        }


    def interdependency_modeling(
        self,
        time,
        protein_pos,  # transform from the orginal BFN codebase
        protein_v,  # transform from
        batch_protein,  # index for protein
        theta_h_t,
        mu_pos_t,
        theta_bond_t,  # add bond
        ligand_bond_index,  # index for bond
        batch_ligand,  # index for ligand
        batch_ligand_bond, # index for bond
        gamma_coord,
        include_protein,
        return_all=False,  # legacy from targetdiff
        fix_x=False,
        ligand_atom_mask=None,
        t_pos=None,
    ):
        """
        Compute output distribution parameters for p_O (x' | θ; t) (x_hat or k^(d) logits).
        Draw output_sample = x' ~ p_O (x' | θ; t).
            continuous x ~ δ(x - x_hat(θ, t))
            discrete k^(d) ~ softmax(Ψ^(d)(θ, t))_k
        Args:
            time: [node_num x batch_size, 1] := [N_ligand, 1]
            protein_pos: [node_num x batch_size, 3] := [N_protein, 3]
            protein_v: [node_num x batch_size, protein_atom_feature_dim] := [N_protein, 27]
            batch_protein: [node_num x batch_size] := [N_protein]
            theta_h_t: [node_num x batch_size, atom_type] := [N_ligand, 13]
            mu_pos_t: [node_num x batch_size, 3] := [N_ligand, 3]
            batch_ligand: [node_num x batch_size] := [N_ligand]
            gamma_coord: [node_num x batch_size, 1] := [N_ligand, 1]
        """
        K = self.num_classes  # ligand_atom_feature_dim
        if self.bond_bfn:
            E = self.num_bond_classes  # bond_atom_feature_dim
        outputs = self.forward(
            time, protein_pos, protein_v, batch_protein, theta_h_t, mu_pos_t, theta_bond_t, ligand_bond_index, batch_ligand, batch_ligand_bond, gamma_coord, include_protein, return_all, fix_x, ligand_atom_mask, t_pos
        )

        coord_pred = outputs['final_ligand_pos']
        final_ligand_v = outputs['final_ligand_v']
        final_ligand_connectivity = outputs['final_ligand_connectivity']
        final_ligand_e = outputs['final_ligand_e']
        final_h = outputs['final_h']

        if self.self_condition:
            outputs = self.forward(
                time, protein_pos, protein_v, batch_protein, final_ligand_v, coord_pred, final_ligand_e, ligand_bond_index, batch_ligand, batch_ligand_bond, gamma_coord, include_protein, return_all, fix_x, ligand_atom_mask, t_pos
            )
            coord_pred = outputs['final_ligand_pos']
            final_ligand_v = outputs['final_ligand_v']
            final_ligand_connectivity = outputs['final_ligand_connectivity']
            final_ligand_e = outputs['final_ligand_e']
            final_h = outputs['final_h']

        # 2. for discrete, network outputs Ψ(θ, t)
        # take softmax will do
        final_ligand_type = final_ligand_v[:, :K]

        if K == 2:
            p0_1 = torch.sigmoid(final_ligand_type[:, 0]).view(-1, 1)  #
            p0_2 = 1 - p0_1
            p0_h = torch.cat((p0_1, p0_2), dim=-1)  #
        else:
            p0_h = torch.nn.functional.softmax(final_ligand_type, dim=-1)  # [N_ligand, 13]

        if self.num_charge != 0:
            final_ligand_charge = final_ligand_v[:, K:K + self.num_charge]
            if self.num_charge == 2:
                p0_sp2 = torch.sigmoid(final_ligand_charge[:, 0]).view(-1, 1)
                p0_sp3 = 1 - p0_sp2
                p0_charge = torch.cat((p0_sp2, p0_sp3), dim=-1)
            else:
                p0_charge = torch.nn.functional.softmax(final_ligand_charge, dim=-1)
        else:
            p0_charge = None

        if self.num_aromatic != 0:
            final_ligand_aromatic = final_ligand_v[:, K + self.num_charge:]
            if self.num_aromatic == 2:
                p0_aromatic = torch.sigmoid(final_ligand_aromatic[:, 0]).view(-1, 1)
                p0_non_aromatic = 1 - p0_aromatic
                p0_aromatic = torch.cat((p0_aromatic, p0_non_aromatic), dim=-1)
            else:
                p0_aromatic = torch.nn.functional.softmax(final_ligand_aromatic, dim=-1)
        else:
            p0_aromatic = None

        if self.bond_bfn:
            if E == 2:
                pE_1 = torch.sigmoid(final_ligand_e)
                pE_2 = 1 - pE_1
                pE_h = torch.cat((pE_1, pE_2), dim=-1)
            else:
                pE_h = torch.nn.functional.softmax(final_ligand_e, dim=-1)
            if torch.isnan(final_ligand_e.any()):
                print("final_ligand_e logits contain NaN")
            if torch.isinf(final_ligand_e).any():
                print("final_ligand_e logits contain Inf")
        else:
            pE_h = None

        """
        for discretised variable, we return p_o
        """
        # print ("k_hat",k_hat.shape)


        preds = {
            'pred_ligand_pos': coord_pred,
            'pred_ligand_v': p0_h,
            'pred_ligand_aromatic': p0_aromatic,
            'pred_ligand_charge': p0_charge,
            # 'pred_ligand_charge': k_hat, # discretized charge
            'pred_ligand_bond': pE_h,
            'final_ligand_connectivity': final_ligand_connectivity,
            'final_h': final_h,
        }

        return preds

    def reconstruction_loss_one_step(
        self,
        t,  # [N_ligand, 1]
        protein_pos,
        protein_v,
        batch_protein,
        ligand_pos,
        ligand_v,
        ligand_bond_type,
        ligand_bond_index,
        batch_ligand,
        batch_ligand_bond,
        include_protein,
        t_pos=None,
        recon_loss=False,
    ):
        # TODO: implement reconstruction loss (but do we really need it?)
        # N = self.discrete_steps
        # sigma1 = self.sigma1_coord
        
        # bak, self.use_discrete_t = self.use_discrete_t, True
        losses = self.loss_one_step(
            t=t,
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            ligand_pos=ligand_pos,
            ligand_v=ligand_v,
            ligand_bond_type=ligand_bond_type,
            ligand_bond_index=ligand_bond_index,
            batch_ligand=batch_ligand,
            batch_ligand_bond=batch_ligand_bond,
            include_protein=include_protein,
            t_pos=t_pos,
            recon_loss=recon_loss
        )

        # self.use_discrete_t = bak

        # pos_loss_weight = N * (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * t))
        # losses['closs'] = losses['closs'] / pos_loss_weight
        return losses

    def loss_one_step(
        self,
        t,  # [B, 1]
        protein_pos,
        protein_v,
        batch_protein,
        ligand_pos,
        ligand_v,
        ligand_bond_type,
        ligand_bond_index,
        batch_ligand,
        batch_ligand_bond,
        include_protein,
        t_pos=None,
        log_grad=False,
        perturb=None,
        batch_mean=True,
        recon_loss=False,
    ):
        K = self.num_classes
        ligand_type = ligand_v[:, 0]
        if ligand_type.max().item() >= K:
            print(f"Error: {ligand_type.max().item()} >= {K}")
        assert ligand_type.max().item() < K, f"Error: {ligand_type.max().item()} >= {K}"
        ligand_type = F.one_hot(ligand_type.long(), K).float()  # [N, K]

        KH = self.num_charge
        if KH != 0:
            ligand_charge = ligand_v[:, 1]
            assert ligand_charge.max().item() < KH, f"Error: {ligand_charge.max().item()} >= {KH}"
            ligand_charge = F.one_hot(ligand_charge, KH).float()
        
        KA = self.num_aromatic
        if KA != 0:
            ligand_aromatic = ligand_v[:, 2]
            assert ligand_aromatic.max().item() < KA, f"Error: {ligand_aromatic.max().item()} >= {KA}"
            ligand_aromatic = F.one_hot(ligand_aromatic, KA).float()

        if self.bond_bfn:
            E = self.num_bond_classes
            # get bond connectivity 
            # ligand_connectivity = F.one_hot((ligand_bond_type != 0).long(), 2).float()
            ligand_connectivity = (ligand_bond_type != 0).long()
            assert ligand_bond_type.max().item() < E, f"Error: {ligand_bond_type.max().item()} >= {E}"
            ligand_bond_type = F.one_hot(ligand_bond_type, E).float()  # [Nb, E]
        
        if log_grad:
            ligand_pos.requires_grad = True
            ligand_type.requires_grad = True
            if self.bond_bfn:
                ligand_bond_type.requires_grad = True

        if perturb is not None:
            perturb_pos, perturb_type, perturb_bond_type = perturb
            ligand_pos = ligand_pos + perturb_pos * torch.randn_like(ligand_pos)
            ligand_type = ligand_type + perturb_type * torch.randn_like(ligand_type)
            # renormalize to [0, 1]
            if self.bond_bfn:
                ligand_bond_type = ligand_bond_type + perturb_bond_type * torch.randn_like(ligand_bond_type)


        # 1. Bayesian Flow p_F(θ|x;t), obtain input parameters θ
        # discrete ~ N(y | β(t)(Ke_x−1), β(t)KI)
        theta = self.discrete_var_bayesian_update(
            t[batch_ligand], beta1=self.beta1, x=ligand_type, K=K
        )  # [N, K]

        if KH != 0:
            theta_charge = self.discrete_var_bayesian_update(
                t[batch_ligand], beta1=self.beta1_charge, x=ligand_charge, K=KH
            )
            theta = torch.cat([theta, theta_charge], dim=-1)

        if KA != 0:
            theta_aromatic = self.discrete_var_bayesian_update(
                t[batch_ligand], beta1=self.beta1_aromatic, x=ligand_aromatic, K=KA
            )
            theta = torch.cat([theta, theta_aromatic], dim=-1)

        if self.bond_bfn:
            theta_bond = self.discrete_var_bayesian_update(
                t[batch_ligand_bond], beta1=self.beta1_bond, x=ligand_bond_type, K=E
            )  # [Nb, E]
            # TODO: harmonic prior
            # continuous ~ N(μ | γ(t)x, γ(t)(1 − γ(t))I)
            # batch_connectivity_matrix = get_batch_connectivity_matrix(
            #     batch_ligand, ligand_bond_index, ligand_bond_type, batch_ligand_bond
            # )
            # mu_coord, gamma_coord = self.continuous_var_bayesian_update_diagonal(
            #     t[batch_ligand], sigma1=self.sigma1_coord, x=ligand_pos, connectivity_matrix=batch_connectivity_matrix
            # )

        else:
            theta_bond = None
            batch_ligand_bond = None

        # continuous ~ N(μ | γ(t)x, γ(t)(1 − γ(t))I)
        mu_coord, gamma_coord = self.continuous_var_bayesian_update(
            t_pos[batch_ligand], sigma1=self.sigma1_coord, x=ligand_pos
        )  # [N, 3], [N, 1]

        preds = self.interdependency_modeling(
            time=t[batch_ligand],
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            theta_h_t=theta,
            theta_bond_t=theta_bond,
            ligand_bond_index=ligand_bond_index,
            mu_pos_t=mu_coord,
            batch_ligand=batch_ligand,
            batch_ligand_bond=batch_ligand_bond,
            gamma_coord=gamma_coord,
            include_protein=include_protein,
            t_pos=t_pos[batch_ligand]
        )  # [N, 3], [N, K], [?]
        coord_pred, p0_h = ( 
            preds['pred_ligand_pos'], preds['pred_ligand_v']
        )
        if self.bond_bfn:
            bond_pred = preds['pred_ligand_bond']
            connectivity_pred = preds['final_ligand_connectivity']

        # if self.include_charge:
        #     k_c = torch.tensor(self.k_c).to(self.device).unsqueeze(-1).unsqueeze(0)
        #     k_hat = (k_hat * k_c).sum(dim=1)
        # average
        # print("x",x.shape,"p0_h",p0_h.shape,"k_hat",k_hat.shape,"charges",charges.shape,mu_charge.shape)

        # 3. Compute reweighted loss (previous [N,] now [B,])


        if not self.use_discrete_t:
            raise NotImplementedError("Continuous time not implemented yet")
            closs = self.ctime4continuous_loss(
                t=t_pos[batch_ligand],
                sigma1=self.sigma1_coord,
                x_pred=coord_pred,
                x=ligand_pos,
                segment_ids=batch_ligand,
            )  # [B,]
            dloss = self.ctime4discrete_loss(
                t=t[batch_ligand],
                beta1=self.beta1,
                one_hot_x=ligand_type,
                p_0=p0_h,
                K=K,
                segment_ids=batch_ligand,
            )  # [B,]

            # TODO: add bond loss
            dloss_bond = torch.zeros_like(dloss)
            dloss_charge = torch.zeros_like(dloss)
            dloss_aromatic = torch.zeros_like(dloss)
            dloss_connectivity = torch.zeros_like(dloss).mean()
            if self.bond_bfn:
                raise NotImplementedError("Bond loss not implemented yet")
                dloss_bond = self.ctime4discrete_loss(
                    t=t[batch_ligand_bond],
                    beta1=self.beta1_bond,
                    one_hot_x=ligand_bond_type,
                    p_0=bond_pred,
                    K=E,
                    segment_ids=batch_ligand_bond,
                )

        else:
            i = (t * self.discrete_steps).int() + 1  # discrete interval [1,N]
            i_pos = (t_pos * self.discrete_steps).int() + 1
            closs = self.dtime4continuous_loss(
                i=i_pos[batch_ligand],
                N=self.discrete_steps,
                sigma1=self.sigma1_coord,
                x_pred=coord_pred,
                x=ligand_pos,
                segment_ids=batch_ligand,
                batch_mean=batch_mean,
            )

            dloss = self.dtime4discrete_loss_prob(
                i=i[batch_ligand],
                N=self.discrete_steps,
                beta1=self.beta1,
                one_hot_x=ligand_type,
                p_0=p0_h,
                K=K,
                segment_ids=batch_ligand,
                batch_mean=batch_mean,
            )

            if KH != 0:
                dloss_charge = self.dtime4discrete_loss_prob(
                    i=i[batch_ligand],
                    N=self.discrete_steps,
                    beta1=self.beta1_charge,
                    one_hot_x=ligand_charge,
                    p_0=preds['pred_ligand_charge'],
                    K=KH,
                    segment_ids=batch_ligand,
                    batch_mean=batch_mean,
                )
            else:
                dloss_charge = torch.zeros_like(dloss)

            if KA != 0:
                dloss_aromatic = self.dtime4discrete_loss_prob(
                    i=i[batch_ligand],
                    N=self.discrete_steps,
                    beta1=self.beta1_aromatic,
                    one_hot_x=ligand_aromatic,
                    p_0=preds['pred_ligand_aromatic'],
                    K=KA,
                    segment_ids=batch_ligand,
                    batch_mean=batch_mean
                )
            else:
                dloss_aromatic = torch.zeros_like(dloss)

            # TODO: add bond loss from BFN2DGraph

            if self.bond_bfn:
                dloss_bond = self.dtime4discrete_loss_prob(
                    i=i[batch_ligand_bond],
                    N=self.discrete_steps,
                    beta1=self.beta1_bond,
                    one_hot_x=ligand_bond_type,
                    p_0=bond_pred,
                    K=self.num_bond_classes,
                    segment_ids=batch_ligand_bond,
                    batch_mean=batch_mean,
                )
                if self.pred_connectivity and connectivity_pred is not None:
                    dloss_connectivity = F.cross_entropy(connectivity_pred, ligand_connectivity)
                else:
                    dloss_connectivity = torch.zeros_like(dloss_bond)
            else:
                dloss_bond = torch.zeros_like(dloss)
                dloss_connectivity = torch.zeros_like(dloss) 


        discretized_loss = torch.zeros_like(closs)

        losses = {
            'closs': closs,
            'dloss': dloss,
            'dloss_bond': dloss_bond,
            'dloss_charge': dloss_charge,
            'dloss_aromatic': dloss_aromatic,
            'dloss_connectivity': dloss_connectivity,
            'discretized_loss': discretized_loss
        }

        if recon_loss:
            with torch.no_grad():
                i_pos = (t_pos * self.discrete_steps).int() + 1
                closs = scatter_mean(
                    ((coord_pred - ligand_pos) ** 2).sum(-1), batch_ligand, dim=0
                )  # [B,]
                dloss = scatter_mean(
                    F.cross_entropy(p0_h, ligand_type.argmax(dim=-1), reduction='none'), batch_ligand, dim=0
                ) # [B,]
                if self.bond_bfn:
                    dloss_bond = scatter_mean(
                        F.cross_entropy(bond_pred, ligand_bond_type.argmax(dim=-1), reduction='none'), batch_ligand_bond, dim=0
                    ) # [B,]
                dloss_charge = torch.zeros_like(dloss)
                dloss_aromatic = torch.zeros_like(dloss)
                dloss_connectivity = torch.zeros_like(dloss)
                losses.update({
                    'closs_mse': closs,
                    'dloss_ce': dloss,
                    'dloss_bond_ce': dloss_bond,
                })

                closs_cont = self.ctime4continuous_loss(
                    t=t_pos[batch_ligand],
                    sigma1=self.sigma1_coord,
                    x_pred=coord_pred,
                    x=ligand_pos,
                    segment_ids=batch_ligand,
                )
                dloss_cont = self.ctime4discrete_loss(
                    t=t[batch_ligand],
                    beta1=self.beta1,
                    one_hot_x=ligand_type,
                    p_0=p0_h,
                    K=K,
                    segment_ids=batch_ligand,
                )  # [B,]
                if self.bond_bfn:
                    dloss_cont_bond = self.ctime4discrete_loss(
                        t=t[batch_ligand_bond],
                        beta1=self.beta1_bond,
                        one_hot_x=ligand_bond_type,
                        p_0=bond_pred,
                        K=E,
                        segment_ids=batch_ligand_bond,
                    )
                else:
                    dloss_cont_bond = torch.zeros_like(dloss_cont)

                losses.update({
                    'closs_cont': closs_cont,
                    'dloss_cont': dloss_cont,
                    'dloss_bond_cont': dloss_cont_bond,
                })

        if log_grad:
            loss = closs + dloss + dloss_bond * 10
            loss = loss.mean()
            compute_input_grad_norms(ligand_pos, ligand_type, ligand_bond_type, self, loss)

        return losses

    def dock_loss_one_step(self, t, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, ligand_bond_type, ligand_bond_index, batch_ligand, batch_ligand_bond, include_protein):
        K = self.num_classes
        ligand_type = ligand_v[:, 0]
        ligand_type = F.one_hot(ligand_type, K).float()

        if self.bond_bfn:
            E = self.num_bond_classes
            # get bond connectivity 
            # ligand_connectivity = F.one_hot((ligand_bond_type != 0).long(), 2).float()
            assert ligand_bond_type.max().item() < E, f"Error: {ligand_bond_type.max().item()} >= {E}"
            ligand_bond_type = F.one_hot(ligand_bond_type, E).float()  # [Nb, E]

        # continuous ~ N(μ | γ(t)x, γ(t)(1 − γ(t))I)
        mu_coord, gamma_coord = self.continuous_var_bayesian_update(
            t[batch_ligand], sigma1=self.sigma1_coord, x=ligand_pos
        )

        preds = self.interdependency_modeling(
            time=t[batch_ligand],
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            theta_h_t=ligand_type,
            theta_bond_t=ligand_bond_type,
            ligand_bond_index=ligand_bond_index,
            mu_pos_t=mu_coord,
            batch_ligand=batch_ligand,
            batch_ligand_bond=batch_ligand_bond,
            gamma_coord=gamma_coord,
            include_protein=include_protein,
            t_pos=t[batch_ligand]
        )

        coord_pred = preds['pred_ligand_pos']

        # obtain loss

        if not self.use_discrete_t:
            loss = self.ctime4continuous_loss(
                t=t[batch_ligand],
                sigma1=self.sigma1_coord,
                x_pred=coord_pred,
                x=ligand_pos,
                segment_ids=batch_ligand,
            )  # [B,]
        else:
            i = (t * self.discrete_steps).int() + 1  # discrete interval [1,N]
            loss = self.dtime4continuous_loss(
                i=i[batch_ligand],
                N=self.discrete_steps,
                sigma1=self.sigma1_coord,
                x_pred=coord_pred,
                x=ligand_pos,
                segment_ids=batch_ligand,
            )

        return loss

    def sample(
        self,
        protein_pos,
        protein_v,
        batch_protein,
        batch_ligand,
        batch_ligand_bond,
        ligand_bond_index, 
        n_nodes,  # B
        include_protein,
        sample_steps=1000,
        desc='Val',
        ligand_pos=None,  # for debug
        ligand_v=None,  # for docking
        ligand_bond_type=None,  # for docking
        t_steps=None,
        # ligand_com=None,  # for generation prior
        # pos_grad_weight=0,
        t_power=1.0,
    ):
        """
        The function implements a sampling procedure for BFN
        Args:
            t: should be a scalar tensor or the shape of [node_num x batch_size, 1] := [N, 1]
            protein_pos: [node_num x batch_size, 3] := [N_protein, 3]
            protein_v: [N_protein, protein_atom_feature_dim] := [N_protein, 27]
            batch_ligand / protein: segment_ids for ligand / protein
        """

        # 1. Initialize prior input parameters θ for p_I(x | θ_0),
        # for continuous, θ_0 = N(0, I)
        # for discrete, θ_0 = 1/K ∈ [0,1]**(KD)
        K = self.num_classes
        KH = self.num_charge
        KA = self.num_aromatic

        if self.bond_bfn:
            E = self.num_bond_classes
        else:
            ligand_bond_type = None

        if ligand_v is not None:
            ligand_type = ligand_v[:, 0]
            ligand_type = F.one_hot(ligand_type.long(), K).float()
            if KH != 0:
                ligand_charge = ligand_v[:, 1]
                ligand_charge = F.one_hot(ligand_charge, KH).float()
            if KA != 0:
                ligand_aromatic = ligand_v[:, 2]
                ligand_aromatic = F.one_hot(ligand_aromatic, KA).float()
        else:
            ligand_type = None
            ligand_charge = None
            ligand_aromatic = None
        if self.bond_bfn and ligand_bond_type is not None:
            ligand_bond_type = F.one_hot(ligand_bond_type, E).float()

        if self.pos_init_mode == 'zero':
            mu_pos_t = torch.zeros((n_nodes, 3)).to(
                self.device
            )  # [N, 3] coordinates prior N(0, 1)
        elif self.pos_init_mode == 'randn':
            mu_pos_t = torch.randn((n_nodes, 3)).to(self.device)

        theta_h_t = (
            torch.ones((n_nodes, K)).to(self.device) / K
        )  # [N, K] discrete prior (uniform 1/K)
        
        if KH != 0:
            theta_charge_t = (
                torch.ones((n_nodes, KH)).to(self.device) / KH
            )
            theta_h_t = torch.cat([theta_h_t, theta_charge_t], dim=-1)
        else:
            theta_charge_t = None
        if KA != 0:
            theta_aromatic_t = (
                torch.ones((n_nodes, KA)).to(self.device) / KA
            )
            theta_h_t = torch.cat([theta_h_t, theta_aromatic_t], dim=-1)
            
        ro_coord = 1

        # TODO: borrow from BFN 2D Graph
        if self.bond_bfn:
            theta_bond_t = torch.ones((n_nodes, E)).to(self.device) / E
        else:
            theta_bond_t = None

        sample_traj = []
        theta_traj = []
        y_traj = []

        # TODO: debug
        mu_pos_t = mu_pos_t[batch_ligand]
        theta_h_t = theta_h_t[batch_ligand]
        if self.bond_bfn: theta_bond_t = theta_bond_t[batch_ligand_bond]
        if KH != 0: theta_charge_t = theta_charge_t[batch_ligand]
        if KA != 0: theta_aromatic_t = theta_aromatic_t[batch_ligand]

        t1 = torch.ones((n_nodes, 1)).to(self.device)
        # Construct reversed u steps
        u_steps = torch.linspace(1, 0, sample_steps + 1)

        if t_steps is None:
            t_steps = 1 - u_steps
            if t_power != 1.0:
                t_steps = t_steps ** t_power
            t_steps = t_steps.unsqueeze(-1).repeat(1, 2)

        t_steps = t_steps.to(dtype=torch.float32, device=self.device)

        print('t_steps', t_steps.shape, t_steps[::10, 1])

        # t_steps = torch.clamp(t_steps, min=self.t_min)
        
        for i in trange(1, sample_steps + 1, desc=f'{desc}-Sampling'):
            # t = torch.ones((n_nodes, 1)).to(self.device) * (i - 1) / sample_steps
            t = t_steps[i-1].repeat(n_nodes, 1).to(self.device)
            if not self.use_discrete_t and not self.destination_prediction:
                t = torch.clamp(t, min=self.t_min)

            t, t_pos = t[:, 0].unsqueeze(-1), t[:, 1].unsqueeze(-1)

            # Eq.(84): γ(t) = σ1^(2t)
            gamma_coord = 1 - torch.pow(self.sigma1_coord, 2 * t[batch_ligand])
            # gamma_charge = 1 - torch.pow(self.sigma1_charges, 2 * t)

            # 2. Compute output distribution parameters for p_O (x' | θ; t)
            # output_params = Ψ(θ,t), x_hat or e_hat for output distribution
            # continuous x ~ δ(x − x_hat(θ, t))
            # discrete k^(d) ~ softmax(Ψ^(d)(θ, t))_k

            ###### for docking ######
            if ligand_type is not None:
                theta_h_t = self.discrete_var_bayesian_update(
                    t[batch_ligand], beta1=self.beta1, x=ligand_type, K=K
                )
                if KH != 0:
                    theta_charge_t = self.discrete_var_bayesian_update(
                        t[batch_ligand], beta1=self.beta1_charge, x=ligand_charge, K=KH
                    )
                    theta_h_t = torch.cat([theta_h_t, theta_charge_t], dim=-1)
                if KA != 0:
                    theta_aromatic_t = self.discrete_var_bayesian_update(
                        t[batch_ligand], beta1=self.beta1_aromatic, x=ligand_aromatic, K=KA
                    )
                    theta_h_t = torch.cat([theta_h_t, theta_aromatic_t], dim=-1)
            if self.bond_bfn and ligand_bond_type is not None:
                theta_bond_t = self.discrete_var_bayesian_update(
                    t[batch_ligand_bond], beta1=self.beta1_bond, x=ligand_bond_type, K=E
                )

            if protein_pos is None:
                mu_pos_t = torch.zeros((n_nodes, 3)).to(self.device)

            preds = self.interdependency_modeling(
                time=t[batch_ligand] if ligand_v is None else t1[batch_ligand],
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
                batch_ligand_bond=batch_ligand_bond,
                theta_h_t=theta_h_t if ligand_v is None else ligand_type,
                # mu_pos_t=mu_coord_gt,  # fix mu pos guidance, type decoding
                # mu_pos_t=mu_pos_t if i > sample_steps/10 else mu_coord_gt,  # early guidance
                mu_pos_t=mu_pos_t,  # no guidance
                gamma_coord=gamma_coord,
                # TODO: add charge
                # mu_charge_t=mu_charge_t,
                # gamma_charge=gamma_charge,
                theta_bond_t=theta_bond_t if ligand_v is None else ligand_bond_type,
                ligand_bond_index=ligand_bond_index,
                include_protein=include_protein,
                t_pos=t_pos[batch_ligand] # TODO: check if different scheduling for t works
            )
            coord_pred, p0_h_pred, k_hat, pE_h_pred = (
                preds['pred_ligand_pos'],
                preds['pred_ligand_v'],
                preds['pred_ligand_charge'],
                preds['pred_ligand_bond'],
            )

            if KH != 0:
                theta_charge_t = theta_h_t[:, K:K + KH]
            else:
                theta_charge_t = None
            
            theta_h_t = theta_h_t[:, :K]

            # maintain theta_traj
            theta_traj.append((mu_pos_t, theta_h_t, theta_charge_t, pE_h_pred))
            # TODO delete the following condition
            if not torch.all(p0_h_pred.isfinite()):
                p0_h_pred = torch.where(
                    p0_h_pred.isfinite(), p0_h_pred, torch.zeros_like(p0_h_pred)
                )
                logging.warn("p0_h_pred is not finite")

            p0_h_pred = torch.clamp(p0_h_pred, min=1e-6)
            sample_pred = torch.distributions.Categorical(p0_h_pred).sample()
            sample_pred = F.one_hot(sample_pred, num_classes=K)

            if self.bond_bfn:
                if not torch.all(pE_h_pred.isfinite()):
                    pE_h_pred = torch.where(
                        pE_h_pred.isfinite(), pE_h_pred, torch.zeros_like(pE_h_pred)
                    )
                    logging.warn("pE_h_pred is not finite")
                
                pE_h_pred = torch.clamp(pE_h_pred, min=1e-6)
                sample_pred_bond = torch.distributions.Categorical(pE_h_pred).sample()
                sample_pred_bond = F.one_hot(sample_pred_bond, num_classes=E)
            else:
                sample_pred_bond = None

            if KH != 0:
                p0_charge_pred = preds['pred_ligand_charge']
                if not torch.all(p0_charge_pred.isfinite()):
                    p0_charge_pred = torch.where(
                        p0_charge_pred.isfinite(), p0_charge_pred, torch.zeros_like(p0_charge_pred)
                    )
                    logging.warn("p0_charge_pred is not finite")
                
                p0_charge_pred = torch.clamp(p0_charge_pred, min=1e-6)
                sample_pred_charge = torch.distributions.Categorical(p0_charge_pred).sample()
                sample_pred_charge = F.one_hot(sample_pred_charge, num_classes=KH)
            else:
                sample_pred_charge = None

            if KA != 0:
                p0_aromatic_pred = preds['pred_ligand_aromatic']
                if not torch.all(p0_aromatic_pred.isfinite()):
                    p0_aromatic_pred = torch.where(
                        p0_aromatic_pred.isfinite(), p0_aromatic_pred, torch.zeros_like(p0_aromatic_pred)
                    )
                    logging.warn("p0_aromatic_pred is not finite")
                
                p0_aromatic_pred = torch.clamp(p0_aromatic_pred, min=1e-6)
                sample_pred_aromatic = torch.distributions.Categorical(p0_aromatic_pred).sample()
                sample_pred_aromatic = F.one_hot(sample_pred_aromatic, num_classes=KA)
            else:
                sample_pred_aromatic = None
            
            # 3. Model sender distribution for sample y ~ p_S (y | x'; α)
            # Algorithm (3)
            # for continuous, y.shape == data.shape
            # Eq.(95) α_i = σ1 ** (−2i/n) * (1 − σ1 ** (2/n))
            alpha_coord = torch.pow(self.sigma1_coord, -2 * i / sample_steps) * (
                1 - torch.pow(self.sigma1_coord, 2 / sample_steps)
            )
            # Eq.(86): p_S (y | x'; α) = N(y | x', 1/α*I)
            # (meaning that y ∼ p_R(· | θ_{i−1}; t_{i−1}, α_i) — see Eq. 4)
            y_coord = coord_pred + torch.randn_like(coord_pred) * torch.sqrt(
                1 / alpha_coord
            )
            # Algorithm (9)
            # for discrete, y \in R^K, while data \in {1,K}, cf. Eq.(141)
            # where e_k is network output p0_h_pred
            # Eq.(193): α_i = β(1) * (2i − 1) / n**2
            alpha_h = self.beta1 * (2 * i - 1) / (sample_steps**2)
            k = torch.distributions.Categorical(probs=p0_h_pred).sample()
            e_k = F.one_hot(k, num_classes=K).float()
            # y ~ N(α(Ke_k − 1) , αKI)
            mean = alpha_h * (K * e_k - 1)
            var = alpha_h * K
            std = torch.full_like(mean, fill_value=var).sqrt()
            y_h = mean + std * torch.randn_like(e_k)

            if self.bond_bfn:
                alpha_E = self.beta1_bond * (2 * i - 1) / (sample_steps**2)
                k_bond = torch.distributions.Categorical(probs=pE_h_pred).sample()
                e_k_bond = F.one_hot(k_bond, num_classes=E).float()
                mean_bond = alpha_E * (E * e_k_bond - 1)
                var_bond = alpha_E * E
                std_bond = torch.full_like(mean_bond, fill_value=var_bond).sqrt()
                y_bond = mean_bond + std_bond * torch.randn_like(e_k_bond)
            else:
                y_bond = None
            
            if KH != 0:
                alpha_charge = self.beta1 * (2 * i - 1) / (sample_steps**2)
                k_charge = torch.distributions.Categorical(probs=p0_charge_pred).sample()
                e_k_charge = F.one_hot(k_charge, num_classes=KH).float()
                mean_charge = alpha_charge * (KH * e_k_charge - 1)
                var_charge = alpha_charge * KH
                std_charge = torch.full_like(mean_charge, fill_value=var_charge).sqrt()
                y_charge = mean_charge + std_charge * torch.randn_like(e_k_charge)
            else:
                y_charge = None
            
            if KA != 0:
                alpha_aromatic = self.beta1 * (2 * i - 1) / (sample_steps**2)
                k_aromatic = torch.distributions.Categorical(probs=p0_aromatic_pred).sample()
                e_k_aromatic = F.one_hot(k_aromatic, num_classes=KA).float()
                mean_aromatic = alpha_aromatic * (KA * e_k_aromatic - 1)
                var_aromatic = alpha_aromatic * KA
                std_aromatic = torch.full_like(mean_aromatic, fill_value=var_aromatic).sqrt()
                y_aromatic = mean_aromatic + std_aromatic * torch.randn_like(e_k_aromatic)

            y_traj.append((y_coord, y_h, y_charge, y_bond))
            

            if self.sampling_strategy == "vanilla":
                sample_traj.append((coord_pred, sample_pred, sample_pred_charge, sample_pred_bond))

                # 4. Bayesian update input parameters θ_i = h(θ_{i−1}, y) for p_I(x | θ_i; t_i)
                # for continuous, Eq.(49): ρi = ρ_{i−1} + α,
                # Eq.(50): μi = [μ_{i−1}ρ_{i−1} + yα] / ρi
                mu_pos_t = (ro_coord * mu_pos_t + alpha_coord * y_coord) / (
                    ro_coord + alpha_coord
                )
                ro_coord = ro_coord + alpha_coord

                # for discrete, Eq.(171): h(θi−1, y, α) := e**y * θ_{i−1} / \sum_{k=1}^K e**y_k (θ_{i−1})_k
                theta_prime = torch.exp(y_h) * theta_h_t  # e^y * θ_{i−1}
                theta_h_t = theta_prime / theta_prime.sum(dim=-1, keepdim=True)

                if self.bond_bfn:
                    theta_prime_bond = torch.exp(y_bond) * theta_bond_t
                    theta_bond_t = theta_prime_bond / theta_prime_bond.sum(dim=-1, keepdim=True)

                if KH != 0:
                    theta_charge_prime = torch.exp(y_charge) * theta_charge_t
                    theta_charge_t = theta_charge_prime / theta_charge_prime.sum(dim=-1, keepdim=True)
                    theta_h_t = torch.cat([theta_h_t, theta_charge_t], dim=-1)

                if KA != 0:
                    theta_aromatic_prime = torch.exp(y_aromatic) * theta_aromatic_t
                    theta_aromatic_t = theta_aromatic_prime / theta_aromatic_prime.sum(dim=-1, keepdim=True)
                    theta_h_t = torch.cat([theta_h_t, theta_aromatic_t], dim=-1)

            elif "end_back" in self.sampling_strategy:
                # t = torch.ones((n_nodes, 1)).to(self.device) * i  / sample_steps #next time step
                if i == sample_steps:
                    t_next = torch.ones((n_nodes, 2)).to(self.device)
                else:
                    t_next = t_steps[i].repeat(n_nodes, 1).to(self.device)
                t_dis_next, t_pos_next = t_next[:, 0].unsqueeze(-1), t_next[:, 1].unsqueeze(-1)
                if self.sampling_strategy == "end_back_pmf":
                    theta_h_t = self.discrete_var_bayesian_update(t_dis_next[batch_ligand], beta1=self.beta1, x=p0_h_pred, K=K)
                    if self.bond_bfn: theta_bond_t = self.discrete_var_bayesian_update(t_dis_next[batch_ligand_bond], beta1=self.beta1_bond, x=pE_h_pred, K=E)
                    if KH != 0:
                        theta_charge_t = self.discrete_var_bayesian_update(t_dis_next[batch_ligand], beta1=self.beta1_charge, x=p0_charge_pred, K=KH)
                        theta_h_t = torch.cat([theta_h_t, theta_charge_t], dim=-1)
                    if KA != 0:
                        theta_aromatic_t = self.discrete_var_bayesian_update(t_dis_next[batch_ligand], beta1=self.beta1_aromatic, x=p0_aromatic_pred, K=KA)
                        theta_h_t = torch.cat([theta_h_t, theta_aromatic_t], dim=-1)
                else:
                    raise NotImplementedError(f"sampling strategy {self.sampling_strategy} not implemented")
                mu_pos_t = self.continuous_var_bayesian_update(t_pos_next[batch_ligand], sigma1=self.sigma1_coord, x=coord_pred)[0]

                    
                sample_traj.append((coord_pred, sample_pred, sample_pred_charge, sample_pred_bond))

            else:
                raise NotImplementedError
            
        ###### for docking ######
        if ligand_type is not None:
            theta_h_t = self.discrete_var_bayesian_update(
                t[batch_ligand], beta1=self.beta1, x=ligand_type, K=K
            )
            if KH != 0:
                theta_charge_t = self.discrete_var_bayesian_update(
                    t[batch_ligand], beta1=self.beta1_charge, x=ligand_charge, K=KH
                )
                theta_h_t = torch.cat([theta_h_t, theta_charge_t], dim=-1)
            if KA != 0:
                theta_aromatic_t = self.discrete_var_bayesian_update(
                    t[batch_ligand], beta1=self.beta1_aromatic, x=ligand_aromatic, K=KA
                )
                theta_h_t = torch.cat([theta_h_t, theta_aromatic_t], dim=-1)

        if self.bond_bfn and ligand_bond_type is not None:
            theta_bond_t = self.discrete_var_bayesian_update(
                t[batch_ligand_bond], beta1=self.beta1_bond, x=ligand_bond_type, K=E
            )
        
        if protein_pos is None:
            mu_pos_t = torch.zeros((n_nodes, 3)).to(self.device)

        # 5. Compute final output distribution parameters for p_O (x' | θ; t)
        preds = self.interdependency_modeling(
            time=torch.ones((n_nodes, 1)).to(self.device)[batch_ligand],
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
            batch_ligand_bond=batch_ligand_bond,
            theta_h_t=theta_h_t if ligand_v is None else ligand_type,
            mu_pos_t=mu_pos_t,
            # mu_charge_t=mu_charge_t,
            gamma_coord=1 - self.sigma1_coord**2,  # γ(t) = 1 − (σ1**2) ** t
            # gamma_charge=1 - self.sigma1_charges**2,
            theta_bond_t=theta_bond_t if ligand_v is None else ligand_bond_type,
            ligand_bond_index=ligand_bond_index,
            include_protein=include_protein,
            t_pos=torch.ones((n_nodes, 1)).to(self.device)[batch_ligand],
        )

        mu_pos_final, p0_h_final, pC_h_final, pE_h_final = (
            preds['pred_ligand_pos'],
            preds['pred_ligand_v'],
            preds['pred_ligand_charge'],
            preds['pred_ligand_bond'],
        )

        if protein_pos is None:
            mu_pos_final = torch.zeros((n_nodes, 3)).to(self.device)

        if not torch.all(p0_h_final.isfinite()):
            p0_h_final = torch.where(
                p0_h_final.isfinite(), p0_h_final, torch.zeros_like(p0_h_final)
            )
            logging.warn("p0_h_pred is not finite")
        p0_h_final = torch.clamp(p0_h_final, min=1e-6)
        if self.bond_bfn:
            if not torch.all(pE_h_final.isfinite()):
                pE_h_final = torch.where(
                    pE_h_final.isfinite(), pE_h_final, torch.zeros_like(pE_h_final)
                )
                logging.warn("pE_h_pred is not finite")
            pE_h_final = torch.clamp(pE_h_final, min=1e-6)
        if KH != 0:
            if not torch.all(pC_h_final.isfinite()):
                pC_h_final = torch.where(
                    pC_h_final.isfinite(), pC_h_final, torch.zeros_like(pC_h_final)
                )
                logging.warn("pC_h_pred is not finite")
            pC_h_final = torch.clamp(pC_h_final, min=1e-6)

        ###### for docking ######
        if ligand_type is not None:
            p0_h_final = ligand_type
        if ligand_bond_type is not None:
            pE_h_final = ligand_bond_type

        if self.pred_connectivity:
            connectivity_pred = preds['final_ligand_connectivity']
            if ligand_bond_type is not None:
                # [0, 1] repeats num_bonds times
                connectivity_pred = torch.repeat_interleave(torch.tensor([[0, 1]]), len(ligand_bond_type), dim=0).to(self.device)
        else:
            connectivity_pred = None
        theta_traj.append((mu_pos_final, p0_h_final, pC_h_final, pE_h_final, connectivity_pred))

        # 6. Draw final sample from p_O (· | θ_n, 1)
        # Update: directly take the mode of categorical distribution (as done in BFN paper)
        sample_traj.append((mu_pos_final, p0_h_final, pC_h_final, pE_h_final, connectivity_pred))

        return theta_traj, sample_traj, y_traj
