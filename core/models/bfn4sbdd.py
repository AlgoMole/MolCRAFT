from absl import logging

import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config.config import Struct
from core.models.common import compose_context, ShiftedSoftplus, GaussianSmearing
from core.models.bfn_base import BFNBase
from core.models.uni_transformer import UniTransformerO2TwoUpdateGeneral
from core.models.uni_transformer_edge import UniTransformerO2TwoUpdateGeneralBond

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
        device="cuda",
        condition_time=True,
        sigma1_coord=0.02,
        beta1=3.0,
        beta1_bond=None,
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
        destination_prediction = False,
        sampling_strategy = "vanilla"
    ):
        super(BFN4SBDDScoreModel, self).__init__()
        # if include_charge:
        #     out_node_nf = in_node_nf + 2
        # else:
        #     out_node_nf = in_node_nf + 1
        net_config = Struct(**net_config)
        self.config = net_config
        
        self.hidden_dim = net_config.hidden_dim
        self.num_classes = ligand_atom_feature_dim
        self.node_indicator = node_indicator

        if net_config.name == 'unio2net':
            self.unio2net = UniTransformerO2TwoUpdateGeneral(**net_config.todict())
            self.bond_bfn = False
        elif net_config.name == 'unio2net_bond':
            self.unio2net = UniTransformerO2TwoUpdateGeneralBond(**net_config.todict())
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
            else:
                raise ValueError(self.bond_net_type)
            self.bond_inference = nn.Sequential(
                nn.Linear(bond_input_dim, self.hidden_dim),
                ShiftedSoftplus(),
                nn.Linear(self.hidden_dim, self.num_bond_classes)
            )            
        else:
            raise NotImplementedError

        if self.node_indicator:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        # atom embedding
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)
        self.center_pos_mode = center_pos_mode  # ['none', 'protein']

        self.time_emb_mode = time_emb_mode
        if net_config.name == 'unio2net_bond':
            self.time_emb_dim = 0
        else:
            self.time_emb_dim = time_emb_dim
            if self.time_emb_dim > 0:
                self.time_emb_layer = TimeEmbedLayer(self.time_emb_mode, self.time_emb_dim)
        self.ligand_atom_emb = nn.Linear(
            ligand_atom_feature_dim + self.time_emb_dim, emb_dim
        )

        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim),
        )  # [hidden to 13]


        self.device = device
        self._edges_dict = {}
        self.condition_time = condition_time
        self.sigma1_coord = torch.tensor(sigma1_coord, dtype=torch.float32)  # coordinate sigma1, a schedule for bfn
        self.beta1 = torch.tensor(beta1, dtype=torch.float32)  # type beta, a schedule for types.
        
        if beta1_bond is not None:
            self.beta1_bond = torch.tensor(beta1_bond, dtype=torch.float32)  # bond beta, a schedule for bond types.
 
        self.use_discrete_t = use_discrete_t  # whether to use discrete t
        self.discrete_steps = discrete_steps
        self.t_min = t_min
        self.pos_init_mode = pos_init_mode
        self.destination_prediction = destination_prediction
        self.sampling_strategy = sampling_strategy

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
        return_all=False,  # legacy from targetdiff
        fix_x=False,
        ligand_atom_mask=None,
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
        theta_h_t = 2 * theta_h_t - 1  # from 1/K \in [0,1] to 2/K-1 \in [-1,1]
        
        if self.bond_bfn:
            E = self.num_bond_classes  # bond_atom_feature_dim
            theta_bond_t = 2 * theta_bond_t - 1  # from 1/K \in [0,1] to 2/K-1 \in [-1,1]

        # ---------for targetdiff-----------
        batch_size = batch_protein.max().item() + 1
        # init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()
        init_ligand_v = theta_h_t
        # time embedding [simple, sin, rbf, learn]
        if self.time_emb_dim > 0:
            time_emb = self.time_emb_layer(time)
            input_ligand_feat = torch.cat([init_ligand_v, time_emb], -1)
        else:
            input_ligand_feat = init_ligand_v

        h_protein = self.protein_atom_emb(protein_v)  # [N_protein, self.hidden_dim - 1]
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)  # [N_ligand, self.hidden_dim - 1]
        # init_ligand_h = input_ligand_feat # TODO: no embedding for ligand atoms, check whether this make sense.

        if self.node_indicator:
            h_protein = torch.cat(
                [h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1
            )  # [N_ligand, self.hidden_dim ]
            init_ligand_h = torch.cat(
                [init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1
            )  # [N_ligand, self.hidden_dim]

        h_all, pos_all, batch_all, mask_ligand, mask_ligand_atom, p_index_in_ctx, l_index_in_ctx = compose_context(
            h_protein=h_protein,
            h_ligand=init_ligand_h,
            pos_protein=protein_pos,
            pos_ligand=mu_pos_t,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
            ligand_atom_mask=ligand_atom_mask,
        )
        # get the context for the protein and ligand, while the ligand is h is noisy (h_t)/ pos is also the noise version. (pos_t)

        # ---------------------

        # time = 2 * time - 1
        if ligand_bond_index is not None:
            if protein_pos is not None:
                bond_index_in_all = l_index_in_ctx[ligand_bond_index]
            else:
                bond_index_in_all = ligand_bond_index
        else:
            bond_index_in_all = None

        # time = 2 * time - 1
        node_time = time[batch_all].squeeze(-1)
        if self.bond_bfn:
            bond_time = time[batch_ligand_bond].squeeze(-1)
            # bond_type = F.one_hot(theta_bond_t, num_classes=self.num_bond_classes).float()
            bond_type = theta_bond_t
            h_bond = self.ligand_bond_emb(bond_type)
            outputs = self.unio2net(
                h=h_all, x=pos_all, group_idx=None,
                bond_index=bond_index_in_all, h_bond=h_bond,
                mask_ligand=mask_ligand,
                # mask_ligand_atom=mask_ligand_atom,  # dummy node is marked as 0
                batch=batch_all,
                node_time=node_time,
                bond_time=bond_time,
                include_protein=True,
                return_all=return_all,
            )

        else:
            outputs = self.unio2net(
                h_all, pos_all, mask_ligand, batch_all, fix_x=fix_x, return_all=return_all
            )

        final_pos, final_h = (
            outputs["x"],
            outputs["h"],
        )  # shape of the pos and shape of h
        final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand]
        final_ligand_v = self.v_inference(final_ligand_h)  # [N_ligand, 13]

        # TODO: think about equivariance for pos & center of mass
        # final_ligand_pos = final_ligand_pos - mu_pos_t  # model the delta

        # _, final_ligand_pos, _ = center_pos(
        #     protein_pos, final_ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode)

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

        k_hat = torch.zeros_like(mu_pos_t)  # TODO: here we close the

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
            elif self.bond_net_type == 'lin':
                bond_inf_input = outputs['h_bond']
            else:
                raise ValueError(self.bond_net_type)
            final_ligand_e = self.bond_inference(bond_inf_input)

        # if self.condition_time:
        #     # Slice off last dimension which represented time.
        #     h_final = h_final[:, :-1]

        # 2. for discrete, network outputs Ψ(θ, t)
        # take softmax will do
        if K == 2:
            p0_1 = torch.sigmoid(final_ligand_v)  #
            p0_2 = 1 - p0_1
            p0_h = torch.cat((p0_1, p0_2), dim=-1)  #
        else:
            p0_h = torch.nn.functional.softmax(final_ligand_v, dim=-1)  # [N_ligand, 13]

        if self.bond_bfn:
            if E == 2:
                pE_1 = torch.sigmoid(final_ligand_e)
                pE_2 = 1 - pE_1
                pE_h = torch.cat((pE_1, pE_2), dim=-1)
            else:
                pE_h = torch.nn.functional.softmax(final_ligand_e, dim=-1)
        else:
            pE_h = None

        """
        for discretised variable, we return p_o
        """
        # print ("k_hat",k_hat.shape)

        # preds = {
        #     'pred_ligand_pos': final_ligand_pos,
        #     'pred_ligand_v': final_ligand_v,
        #     'final_h': final_h,
        #     'final_ligand_h': final_ligand_h
        # }
        # if return_all:
        #     final_all_pos, final_all_h = outputs['all_x'], outputs['all_h']
        #     final_all_ligand_pos = [pos[mask_ligand] for pos in final_all_pos]
        #     final_all_ligand_v = [self.v_inference(h[mask_ligand]) for h in final_all_h]
        #     preds.update({
        #         'layer_pred_ligand_pos': final_all_ligand_pos,
        #         'layer_pred_ligand_v': final_all_ligand_v
        #     })

        # TODO: here the preds are reformated.
        # print(coord_pred.shape, p0_h.shape, k_hat.shape)

        preds = {
            'pred_ligand_pos': coord_pred,
            'pred_ligand_v': p0_h,
            'pred_ligand_charge': k_hat,
            'pred_ligand_bond': pE_h,
            'final_h': final_h,
            'final_ligand_h': final_ligand_h
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
    ):
        # TODO: implement reconstruction loss (but do we really need it?)
        return self.loss_one_step(
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
        )

    def loss_one_step(
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
    ):
        K = self.num_classes
        assert ligand_v.max().item() < K, f"Error: {ligand_v.max().item()} >= {K}"
        ligand_v = F.one_hot(ligand_v, K).float()  # [N, K]

        if self.bond_bfn:
            E = self.num_bond_classes
            assert ligand_bond_type.max().item() < E, f"Error: {ligand_bond_type.max().item()} >= {E}"
            ligand_bond_type = F.one_hot(ligand_bond_type, E).float()  # [Nb, E]
 
        # 1. Bayesian Flow p_F(θ|x;t), obtain input parameters θ
        # continuous ~ N(μ | γ(t)x, γ(t)(1 − γ(t))I)
        mu_coord, gamma_coord = self.continuous_var_bayesian_update(
            t[batch_ligand], sigma1=self.sigma1_coord, x=ligand_pos
        )  # [N, 3], [N, 1]

        # discrete ~ N(y | β(t)(Ke_x−1), β(t)KI)
        theta = self.discrete_var_bayesian_update(
            t[batch_ligand], beta1=self.beta1, x=ligand_v, K=K
        )  # [N, K]

        if self.bond_bfn:
            theta_bond = self.discrete_var_bayesian_update(
                t[batch_ligand_bond], beta1=self.beta1_bond, x=ligand_bond_type, K=E
            )  # [Nb, E]
        else:
            theta_bond = None
            batch_ligand_bond = None

        # 2. Compute output distribution parameters for p_O (x' | θ; t) (x_hat or k^(d) logits)
        # continuous x ~ δ(x − x_hat(θ, t))
        # discrete k^(d) ~ softmax(Ψ^(d)(θ, t))_k
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
        )  # [N, 3], [N, K], [?]
        coord_pred, p0_h, k_hat = ( 
            preds['pred_ligand_pos'], preds['pred_ligand_v'], preds['pred_ligand_charge']
        )

        if self.bond_bfn:
            bond_pred = preds['pred_ligand_bond']
        # if self.include_charge:
        #     k_c = torch.tensor(self.k_c).to(self.device).unsqueeze(-1).unsqueeze(0)
        #     k_hat = (k_hat * k_c).sum(dim=1)
        # average
        # print("x",x.shape,"p0_h",p0_h.shape,"k_hat",k_hat.shape,"charges",charges.shape,mu_charge.shape)

        # 3. Compute reweighted loss (previous [N,] now [B,])
        if not self.use_discrete_t:
            closs = self.ctime4continuous_loss(
                t=t[batch_ligand],
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
            raise NotImplementedError
        else:
            i = (t * self.discrete_steps).int() + 1  # discrete interval [1,N]
            closs = self.dtime4continuous_loss(
                i=i[batch_ligand],
                N=self.discrete_steps,
                sigma1=self.sigma1_coord,
                x_pred=coord_pred,
                x=ligand_pos,
                segment_ids=batch_ligand,
            )

            # closs = self.ctime4continuous_loss(
            #     t=t, sigma1=self.sigma1_coord, x_pred=coord_pred, x=ligand_pos, segment_ids=batch_ligand
            # )  # [B,]
            # dloss = self.ctime4discrete_loss(
            #     t=t, beta1=self.beta1, one_hot_x=ligand_v, p_0=p0_h, K=K, segment_ids=batch_ligand
            # )  # [B,]

            dloss = self.dtime4discrete_loss_prob(
                i=i[batch_ligand],
                N=self.discrete_steps,
                beta1=self.beta1,
                one_hot_x=ligand_v,
                p_0=p0_h,
                K=K,
                segment_ids=batch_ligand,
            )

            if self.bond_bfn:
                dloss_bond = self.dtime4discrete_loss_prob(
                    i=i[batch_ligand_bond],
                    N=self.discrete_steps,
                    beta1=self.beta1_bond,
                    one_hot_x=ligand_bond_type,
                    p_0=bond_pred,
                    K=self.num_bond_classes,
                    segment_ids=batch_ligand_bond,
                )
            else:
                dloss_bond = torch.zeros_like(dloss)

            # mixed loss
            # dloss = self.ctime4discrete_loss(
            #     t=t,
            #     beta1=self.beta1,
            #     one_hot_x=ligand_v,
            #     p_0=p0_h,
            #     K=K,
            #     segment_ids=batch_ligand,
            # )  # [B,]

        # closs = self.ctime4continuous_loss(
        #     t=t, sigma1=self.sigma1_coord, x_pred=coord_pred, x=ligand_pos
        # )
        # dloss = self.ctime4discrete_loss(
        #     t=t, beta1=self.beta1, one_hot_x=ligand_v, p_0=p0_h, K=K
        # )
        # TODO: check compatible with charge
        # if self.include_charge:
        #     if self.charge_discretised_loss:
        #         discretized_loss = self.ctime4discreteised_loss(
        #         t=t, sigma1=self.sigma1_charges, x_pred=k_hat, x=charges)
        #     else:
        #         discretized_loss = self.ctime4continuous_loss(
        #         t=t, sigma1=self.sigma1_charges, x_pred=k_hat, x=charges)
        # else:
        discretized_loss = torch.zeros_like(closs)

        losses = {
            'closs': closs,
            'dloss': dloss,
            'dloss_bond': dloss_bond,
            'discretized_loss': discretized_loss
        }

        return losses

    def sample(
        self,
        protein_pos,
        protein_v,
        batch_protein,
        batch_ligand,
        batch_ligand_bond,
        ligand_bond_index, 
        n_nodes,  # B
        sample_steps=1000,
        desc='Val',
        ligand_pos=None,  # for debug
        ligand_v=None,  # for docking
        ligand_bond_type=None,  # for docking
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
        if self.bond_bfn:
            E = self.num_bond_classes
        else:
            ligand_bond_type = None

        # TODO: no charges here
        # if self.include_charge:
        # if False:
        #     mu_pos_t = torch.zeros((n_nodes, 3)).to(
        #         self.device
        #     )  # [N, 4] coordinates prior
        #     mu_charge_t = torch.zeros((n_nodes, 1)).to(self.device)

        # else:
        #     mu_pos_t = torch.zeros((n_nodes, 3)).to(
        #         self.device
        #     )  # [N, 3] coordinates prior N(0, 1)
        #     mu_charge_t = None

        # TODO test
        if self.pos_init_mode == 'zero':
            mu_pos_t = torch.zeros((n_nodes, 3)).to(
                self.device
            )  # [N, 3] coordinates prior N(0, 1)
        elif self.pos_init_mode == 'randn':
            mu_pos_t = torch.randn((n_nodes, 3)).to(self.device)

        theta_h_t = (
            torch.ones((n_nodes, K)).to(self.device) / K
        )  # [N, K] discrete prior (uniform 1/K)
        ro_coord = 1
        ro_charge = 1
        
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
        for i in trange(1, sample_steps + 1, desc=f'{desc}'):
            t = torch.ones((n_nodes, 1)).to(self.device) * (i - 1) / sample_steps
            if not self.use_discrete_t and not self.destination_prediction:
                t = torch.clamp(t, min=self.t_min)

            # Eq.(84): γ(t) = σ1^(2t)
            gamma_coord = 1 - torch.pow(self.sigma1_coord, 2 * t)
            # gamma_charge = 1 - torch.pow(self.sigma1_charges, 2 * t)

            # 2. Compute output distribution parameters for p_O (x' | θ; t)
            # output_params = Ψ(θ,t), x_hat or e_hat for output distribution
            # continuous x ~ δ(x − x_hat(θ, t))
            # discrete k^(d) ~ softmax(Ψ^(d)(θ, t))_k

            # debug only
            # mu_coord_gt, gamma_coord_gt = self.continuous_var_bayesian_update(t, sigma1=self.sigma1_coord, x=ligand_pos)

            preds = self.interdependency_modeling(
                time=t[batch_ligand],
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
                batch_ligand_bond=batch_ligand_bond,
                theta_h_t=theta_h_t,
                # mu_pos_t=mu_coord_gt,  # fix mu pos guidance, type decoding
                # mu_pos_t=mu_pos_t if i > sample_steps/10 else mu_coord_gt,  # early guidance
                mu_pos_t=mu_pos_t,  # no guidance
                gamma_coord=gamma_coord,
                # TODO: add charge
                # mu_charge_t=mu_charge_t,
                # gamma_charge=gamma_charge,
                theta_bond_t=theta_bond_t,
                ligand_bond_index=ligand_bond_index,
            )
            coord_pred, p0_h_pred, k_hat, pE_h_pred = (
                preds['pred_ligand_pos'],
                preds['pred_ligand_v'],
                preds['pred_ligand_charge'],
                preds['pred_ligand_bond'],
            )


            # debug only
            # if i % (sample_steps // 10) == 0:
            #     print("\nt*1000", t[0,0].item()*1000, "gamma_coord", gamma_coord[0,0].item(), torch.sqrt(gamma_coord * (1 - gamma_coord))[0,0].item())
            #     print("ligand_pos", ligand_pos)
            #     print("mu_coord_gt", mu_coord_gt)
            #     assert torch.allclose(gamma_coord_gt, gamma_coord, atol=1e-4), "gamma_coord_gt != gamma_coord"
            #     print("mu_pos_t", mu_pos_t)
            #     print("coord_pred", coord_pred)

            # maintain theta_traj
            theta_traj.append((mu_pos_t, theta_h_t, k_hat, pE_h_pred))
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

            # if self.include_charge:
            #     sample_traj.append((coord_pred[:, :-1], sample_pred))
            # else:
            #     sample_traj.append((coord_pred, sample_pred))

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

            y_traj.append((y_coord, y_h, k_hat, y_bond))

            if self.sampling_strategy == "vanilla":
                sample_traj.append((coord_pred, sample_pred, k_hat, sample_pred_bond))

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

            elif "end_back" in self.sampling_strategy:
                t = torch.ones((n_nodes, 1)).to(self.device) * i  / sample_steps #next time step
                if self.sampling_strategy == "end_back":
                    theta_h_t = self.discrete_var_bayesian_update(t[batch_ligand], beta1=self.beta1, x=sample_pred, K=K)
                    if self.bond_bfn: theta_bond_t = self.discrete_var_bayesian_update(t[batch_ligand_bond], beta1=self.beta1_bond, x=sample_pred_bond, K=E)
                elif self.sampling_strategy == "end_back_pmf":
                    theta_h_t = self.discrete_var_bayesian_update(t[batch_ligand], beta1=self.beta1, x=p0_h_pred, K=K)
                    if self.bond_bfn: theta_bond_t = self.discrete_var_bayesian_update(t[batch_ligand_bond], beta1=self.beta1_bond, x=pE_h_pred, K=E)
                elif self.sampling_strategy == "end_back_mode":
                    p0_mode = torch.argmax(p0_h_pred, dim=-1)
                    mode_pred = F.one_hot(p0_mode, num_classes=K).float()
                    theta_h_t = self.discrete_var_bayesian_update(t[batch_ligand], beta1=self.beta1, x=mode_pred, K=K)
                
                    if self.bond_bfn: 
                        pE_mode = torch.argmax(pE_h_pred, dim=-1)
                        mode_pred_bond = F.one_hot(pE_mode, num_classes=E).float()
                        theta_bond_t = self.discrete_var_bayesian_update(t[batch_ligand_bond], beta1=self.beta1_bond, x=mode_pred_bond, K=E)
                
                else:
                    raise NotImplementedError(f"sampling strategy {self.sampling_strategy} not implemented")
                mu_pos_t = self.continuous_var_bayesian_update(t[batch_ligand], sigma1=self.sigma1_coord, x=coord_pred)[0]
                sample_traj.append((coord_pred, sample_pred, k_hat, sample_pred_bond))

                # if i % (sample_steps // 10) == 0:
                    # print(f"theta_h_{i}", theta_h_t)
                    # mu_pos_t size [N,3]
                    # print(f"mu_pos_{i}_min", mu_pos_t.min(dim=0).values.cpu().numpy(), "max", mu_pos_t.max(dim=0).values.cpu().numpy())
                    # log to wandb
                    # wandb.log({"mu_pos_t": mu_pos_t})


            else:
                raise NotImplementedError
            
            # update of the discretised variable
            # TODO: charge
            # if self.include_charge:
            if False:
                if not self.charge_discretised_loss:
                    # for continous like update
                    alpha_charge = torch.pow(
                        self.sigma1_charges, -2 * i / sample_steps
                    ) * (1 - torch.pow(self.sigma1_charges, 2 / sample_steps))
                    y_charge = k_hat + torch.randn_like(k_hat) * torch.sqrt(
                        1 / alpha_charge
                    )
                    mu_charge_t = (
                        ro_charge * mu_charge_t + alpha_charge * y_charge
                    ) / (ro_charge + alpha_charge)
                    ro_charge = ro_charge + alpha_charge
                else:
                    # for discretised update
                    alpha_charge = torch.pow(
                        self.sigma1_charges, -2 * i / sample_steps
                    ) * (1 - torch.pow(self.sigma1_charges, 2 / sample_steps))
                    discrete_output = k_hat
                    discrete_output = torch.transpose(discrete_output, 1, 2)
                    batch_size = discrete_output.shape[0]
                    discrete_output = discrete_output.reshape(
                        -1, discrete_output.shape[-1]
                    )
                    # print("discrete_output",discrete_output.shape)
                    if not torch.all(discrete_output.isfinite()):
                        discrete_output = torch.where(
                            discrete_output.isfinite(),
                            discrete_output,
                            torch.zeros_like(discrete_output),
                        )
                        logging.warn("discrete_output is not finite")
                    discrete_output = torch.clamp(discrete_output, min=1e-6)

                    categorical = dist.Categorical(probs=discrete_output)
                    sample_k = categorical.sample()
                    sample_k = sample_k.view(batch_size, -1) + 1
                    sample_k_c = (2 * sample_k - 1) / self.bins - 1
                    y_charge = sample_k_c + torch.randn_like(sample_k_c) * torch.sqrt(
                        1 / alpha_charge
                    )
                    mu_charge_t = (
                        ro_charge * mu_charge_t + alpha_charge * y_charge
                    ) / (ro_charge + alpha_charge)
                    ro_charge = ro_charge + alpha_charge

            # if self.include_charge:
            if False:
                if self.charge_discretised_loss:
                    sample_traj.append((coord_pred, sample_pred, sample_k))
                else:
                    k_hat = torch.clamp(k_hat, min=-1, max=1)
                    k_c = torch.tensor(self.k_c).to(self.device).unsqueeze(0)
                    k_hat = find_closet_index(k_hat, k_c)
                    sample_traj.append((coord_pred, sample_pred, k_hat))
            else:
                continue
                # sample_traj.append((coord_pred, sample_pred,k_hat))

        # 5. Compute final output distribution parameters for p_O (x' | θ; t)
        preds = self.interdependency_modeling(
            time=torch.ones((n_nodes, 1)).to(self.device)[batch_ligand],
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
            batch_ligand_bond=batch_ligand_bond,
            theta_h_t=theta_h_t,
            mu_pos_t=mu_pos_t,
            # mu_charge_t=mu_charge_t,
            gamma_coord=1 - self.sigma1_coord**2,  # γ(t) = 1 − (σ1**2) ** t
            # gamma_charge=1 - self.sigma1_charges**2,
            theta_bond_t=theta_bond_t,
            ligand_bond_index=ligand_bond_index,
        )

        mu_pos_final, p0_h_final, k_hat_final, pE_h_final = (
            preds['pred_ligand_pos'],
            preds['pred_ligand_v'],
            preds['pred_ligand_charge'],
            preds['pred_ligand_bond'],
        )
        # TODO delete the following condition
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

        theta_traj.append((mu_pos_final, p0_h_final, k_hat_final, pE_h_final))

        # 6. Draw final sample from p_O (· | θ_n, 1)
        # Update: directly take the mode of categorical distribution (as done in BFN paper)
        k_final = p0_h_final
        # k_final = torch.distributions.Categorical(p0_h_final).sample()
        # k_final = F.one_hot(k_final, num_classes=K)

        # if self.include_charge:
        if False:
            if self.charge_discretised_loss:
                discretised_output_final = k_hat_final  # [B,Bins,1]
                discretised_output_final = torch.transpose(
                    discretised_output_final, 1, 2
                )
                batch_size = discretised_output_final.shape[0]
                discretised_output_final = discretised_output_final.reshape(
                    -1, discretised_output_final.shape[-1]
                )
                # print("discrete_output",discrete_output.shape)

                if not torch.all(discretised_output_final.isfinite()):
                    discretised_output_final = torch.where(
                        discretised_output_final.isfinite(),
                        discretised_output_final,
                        torch.zeros_like(discretised_output_final),
                    )
                    logging.warn("discrete_output is not finite")
                discretised_output_final = torch.clamp(
                    discretised_output_final, min=1e-6
                )

                categorical = dist.Categorical(probs=discretised_output_final)
                sample_k = categorical.sample()
                sample_k_final = sample_k.view(
                    batch_size, -1
                )  # sample_k_final is the value lies in [0,8]

                sample_traj.append((mu_pos_final, k_final, sample_k_final))
            else:
                k_hat_final = torch.clamp(k_hat_final, min=-1, max=1)
                k_c = torch.tensor(self.k_c).to(self.device).unsqueeze(0)
                k_hat_final = find_closet_index(k_hat_final, k_c)
                sample_traj.append((mu_pos_final, k_final, k_hat_final))
        else:
            sample_traj.append((mu_pos_final, k_final, k_hat_final, pE_h_final))

        return theta_traj, sample_traj, y_traj
