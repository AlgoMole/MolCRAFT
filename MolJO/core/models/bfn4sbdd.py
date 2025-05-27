from absl import logging

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum

from core.config.config import Struct
from core.models.common import compose_context, ShiftedSoftplus, l2_norm
from core.models.bfn_base import BFNBase
from core.models.uni_transformer import UniTransformerO2TwoUpdateGeneral

from scipy.spatial import KDTree

def find_closest_unique_points(A, B):
    Na = A.shape[0]
    Nb = B.shape[0]

    # Step 1: Compute the distance matrix
    distances = np.linalg.norm(A[:, np.newaxis] - B[np.newaxis, :, :], axis=2)

    # Step 2: Create a list to hold the unique closest points
    unique_closest_points = []
    used_indices = []

    # Step 3: Iterate over each point in A to find the closest unique point in B
    for i in range(Na):
        # Get the indices of the sorted distances for the i-th point in A
        sorted_indices = np.argsort(distances[i])
        
        # Find the closest unique point in B
        for idx in sorted_indices:
            if idx not in used_indices:
                used_indices.append(idx)
                unique_closest_points.append(B[idx])
                break

    distances = distances[np.arange(Na), used_indices]
    return used_indices, distances


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
        sampling_strategy = "vanilla",
    ):
        super(BFN4SBDDScoreModel, self).__init__()
        # if include_charge:
        #     out_node_nf = in_node_nf + 2
        # else:
        #     out_node_nf = in_node_nf + 1
        net_config = Struct(**net_config)
        self.config = net_config

        if net_config.name == 'unio2net':
            self.unio2net = UniTransformerO2TwoUpdateGeneral(**net_config.todict())
        else:
            raise NotImplementedError
        
        self.hidden_dim = net_config.hidden_dim
        self.num_classes = ligand_atom_feature_dim

        self.node_indicator = node_indicator

        if self.node_indicator:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        # atom embedding
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)
        self.center_pos_mode = center_pos_mode  # ['none', 'protein']

        self.time_emb_mode = time_emb_mode
        self.time_emb_dim = time_emb_dim
        if self.time_emb_dim > 0:
            self.time_emb_layer = TimeEmbedLayer(self.time_emb_mode, self.time_emb_dim)
        self.ligand_atom_emb = nn.Linear(
            ligand_atom_feature_dim + self.time_emb_dim, emb_dim
        )

        # self.refine_net_type = config.model_type
        # self.refine_net = get_refine_net(self.refine_net_type, config)
        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim),
        )  # [hidden to 13]

        # self.egnn = EGNN(
        #     in_node_nf=in_node_nf + int(condition_time),  # +1 for time
        #     hidden_nf=hidden_nf,
        #     out_node_nf=out_node_nf, # need to predict the mean and variance of the charges for discretised data
        #     in_edge_nf=0,
        #     device=device,
        #     act_fn=act_fn,
        #     n_layers=n_layers,
        #     attention=attention,
        #     # normalize=True,
        #     tanh=tanh,
        # )
        # self.in_node_nf = in_node_nf

        self.device = device
        self._edges_dict = {}
        self.condition_time = condition_time
        self.sigma1_coord = torch.tensor(sigma1_coord, dtype=torch.float32)  # coordinate sigma1, a schedule for bfn
        self.beta1 = torch.tensor(beta1, dtype=torch.float32)  # type beta, a schedule for types.
        self.use_discrete_t = use_discrete_t  # whether to use discrete t
        self.discrete_steps = discrete_steps
        self.t_min = t_min
        self.pos_init_mode = pos_init_mode
        self.destination_prediction = destination_prediction
        self.sampling_strategy = sampling_strategy
        # self.include_charge = include_charge
        # self.no_diff_coord = no_diff_coord #whether the output minus the inputs for the graph neural networks.

    def interdependency_modeling(
        self,
        time,
        protein_pos,  # transform from the orginal BFN codebase
        protein_v,  # transform from
        batch_protein,  # index for protein
        theta_h_t,
        mu_pos_t,
        batch_ligand,  # index for ligand
        gamma_coord,
        return_all=False,  # legacy from targetdiff
        fix_x=False,
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

        h_all, pos_all, batch_all, mask_ligand = compose_context(
            h_protein=h_protein,
            h_ligand=init_ligand_h,
            pos_protein=protein_pos,
            pos_ligand=mu_pos_t,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )
        # get the context for the protein and ligand, while the ligand is h is noisy (h_t)/ pos is also the noise version. (pos_t)

        # ---------------------

        # time = 2 * time - 1
        outputs = self.unio2net(
            h_all, pos_all, mask_ligand, batch_all, return_all=return_all, fix_x=fix_x
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
            # TODO: temporary fix, overwrite the original behavior
            if "vanilla" not in self.sampling_strategy:
                p0_h = final_ligand_v
            else:
                p0_h = torch.nn.functional.softmax(final_ligand_v, dim=-1)  # [N_ligand, 13]
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
        return coord_pred, final_ligand_v, k_hat
        # return coord_pred, p0_h, k_hat

    def reconstruction_loss_one_step(
        self,
        t,  # [N_ligand, 1]
        protein_pos,
        protein_v,
        batch_protein,
        ligand_pos,
        ligand_v,
        batch_ligand,
    ):
        # TODO: implement reconstruction loss (but do we really need it?)
        return self.loss_one_step(
            t, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand
        )

    def loss_one_step(
        self,
        t,  # [N_ligand, 1]
        protein_pos,
        protein_v,
        batch_protein,
        ligand_pos,
        ligand_v,
        batch_ligand,
    ):
        K = self.num_classes
        if self.use_discrete_t:
            i = (t * self.discrete_steps).int() + 1
            t = torch.ones_like(t) * (i - 1) / self.discrete_steps

        # if self.include_charge:
        #     assert x.size(-1) == K + 1
        #     charges = x[:, -1:]
        #     x = x[:, :-1]
        #     mu_charge, gamma_charge = self.discreteised_var_bayesian_update(t, sigma1=self.sigma1_charges, x=charges)
        # else:
        #     mu_charge = None
        #     gamma_charge = None
        #     #pos = torch.cat([pos, charges], dim=-1)

        # print("loss",charges)
        # TODO: no charge here
        assert ligand_v.max().item() < K, f"Error: {ligand_v.max().item()} >= {K}"
        ligand_v = F.one_hot(ligand_v, K).float()  # [N, K]

        # 1. Bayesian Flow p_F(θ|x;t), obtain input parameters θ
        # continuous ~ N(μ | γ(t)x, γ(t)(1 − γ(t))I)
        mu_coord, gamma_coord = self.continuous_var_bayesian_update(
            t, sigma1=self.sigma1_coord, x=ligand_pos
        )  # [N, 3], [N, 1]

        # discrete ~ N(y | β(t)(Ke_x−1), β(t)KI)
        theta = self.discrete_var_bayesian_update(
            t, beta1=self.beta1, x=ligand_v, K=K
        )  # [N, K]

        # 2. Compute output distribution parameters for p_O (x' | θ; t) (x_hat or k^(d) logits)
        # continuous x ~ δ(x − x_hat(θ, t))
        # discrete k^(d) ~ softmax(Ψ^(d)(θ, t))_k
        coord_pred, p0_h, k_hat = self.interdependency_modeling(
            time=t,
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            theta_h_t=theta,
            mu_pos_t=mu_coord,
            batch_ligand=batch_ligand,
            gamma_coord=gamma_coord,
        )  # [N, 3], [N, K], [?]
        # if self.include_charge:
        #     k_c = torch.tensor(self.k_c).to(self.device).unsqueeze(-1).unsqueeze(0)
        #     k_hat = (k_hat * k_c).sum(dim=1)
        # average
        # print("x",x.shape,"p0_h",p0_h.shape,"k_hat",k_hat.shape,"charges",charges.shape,mu_charge.shape)

        # 3. Compute reweighted loss (previous [N,] now [B,])
        if not self.use_discrete_t:
            closs = self.ctime4continuous_loss(
                t=t,
                sigma1=self.sigma1_coord,
                x_pred=coord_pred,
                x=ligand_pos,
                segment_ids=batch_ligand,
            )  # [B,]
            dloss = self.ctime4discrete_loss(
                t=t,
                beta1=self.beta1,
                one_hot_x=ligand_v,
                p_0=p0_h,
                K=K,
                segment_ids=batch_ligand,
            )  # [B,]
        else:
            i = (t * self.discrete_steps).int() + 1  # discrete interval [1,N]
            closs = self.dtime4continuous_loss(
                i=i,
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
                i=i,
                N=self.discrete_steps,
                beta1=self.beta1,
                one_hot_x=ligand_v,
                p_0=p0_h,
                K=K,
                segment_ids=batch_ligand,
            )

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

        return closs, dloss, discretized_loss

    # algorithm for alignment
    def kabsch(
        self,
        A, 
        B,
    ):
        """Compute the optimal rotation and translation matrix using the Kabsch algorithm in PyTorch."""
        assert A.shape == B.shape, "A and B must have the same shape."
        
        # Compute centroids
        centroid_A = A.mean(dim=0)
        centroid_B = B.mean(dim=0)
        # centroid_A = torch.zeros_like(A.mean(dim=0))
        # centroid_B = torch.zeros_like(B.mean(dim=0))
        
        # Center the points
        A_centered = A - centroid_A
        B_centered = B - centroid_B
        
        # Compute covariance matrix
        H = A_centered.T @ B_centered
        
        # SVD for rotation matrix
        U, S, Vt = torch.linalg.svd(H)
        
        R = Vt.T @ U.T
        
        # Ensure a proper rotation matrix (det(R) should be 1)
        if torch.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = centroid_B - R @ centroid_A
        
        return R, t
    
    def calculate_rmsd(
        self,
        A,
        B,
    ):
        """Calculate the root-mean-square deviation (RMSD) between two sets of points in PyTorch."""
        diff = A - B
        return torch.sqrt(torch.mean(torch.sum(diff**2, dim=1)))
    
    def icp(
        self,
        A, 
        B, 
        max_iterations=50, 
        tolerance=1e-5,
    ):
        prev_error = float('inf')
        matched_indices = np.zeros(len(A), dtype=int)
        
        for i in range(max_iterations):
            # Find the unique nearest neighbors in B for each point in A
            # Step 1: Find initial nearest neighbors using KDTree
            indices, distances = find_closest_unique_points(A.cpu().numpy(), B.cpu().numpy())
            # unique_closest_points = torch.from_numpy(unique_closest_points).to(B.device)
            matched_B = B[indices]

            # distances = np.linalg.norm(A.cpu().numpy() - matched_B.cpu().numpy(), axis=1)

            # assert torch.allclose(matched_B, unique_closest_points), f"The matched points are not equal. {matched_B} != {unique_closest_points}"
            assert len(list(set(indices))) == len(A), f"The indices are not unique. {indices}"
            if len(matched_B) != len(A):
                print(indices)
                print(f"Unique indices: {indices}")
                # print(f"Matched points: {unique_closest_points}")
                raise ValueError(f"Not enough unique matches found. {len(matched_B)} != {len(A)}")

            R, t = self.kabsch(A, matched_B)
            
            # A = A @ R + t
            
            mean_error = torch.mean(torch.tensor(distances, dtype=torch.float32))
            
            if torch.abs(prev_error - mean_error) < tolerance:
                matched_indices = indices
                break
            
            prev_error = mean_error
            matched_indices = indices
        
        rmsd = self.calculate_rmsd(A, B[matched_indices])
        
        return R, t, A, matched_indices, rmsd

    def align(self, gt_pos, pos, gt_v, v, ligand_mask, batch_ligand, verbose=False):
        x_aligned = gt_pos.shape
        x_toalign = pos.shape
        batch_size = batch_ligand.max().item() + 1

        new_pos = torch.zeros_like(pos)
        new_v = torch.zeros_like(v)
        rmsd_list = []
        # make sure that the shape of aligned_matrix and toalign_matrix is equal
        if x_aligned[0] != len(batch_ligand):
            raise ValueError("The number of rows in the matrix is not uniform")
        else:
            num_splits = batch_size
            for i in range(num_splits):
                mask = (batch_ligand == i)
                aligned_i = gt_pos[mask]
                toalign_i = pos[mask]
                # store the new position and type
                new_pos[mask] = pos[mask].clone().detach()
                new_v[mask] = v[mask].clone().detach()
                # take out the masked ligand substructure for alignment
                mask_i = ligand_mask[mask]
                aligned_i_substructure = aligned_i[mask_i]

                # if the ligand substructure is all zeros, skip the alignment
                if aligned_i_substructure.sum() == 0:
                    continue

                # sanity check
                assert aligned_i[~mask_i].sum() == 0, f"The masked ligand substructure ({aligned_i[~mask_i].sum()}) is not zero"
                _, _, transformed, matched_indices, rmsd = self.icp(aligned_i_substructure, toalign_i)

                rmsd_list.append(rmsd)

                if verbose:
                    breakpoint()

                for i, idx in enumerate(matched_indices):
                    # new_pos[mask][idx, :] = transformed[i, :]
                    # new_v[mask][idx, :] = gt_v[mask][mask_i][i, :]
                    flat_idx = mask.nonzero(as_tuple=True)[0][idx]
                    new_pos[flat_idx] = transformed[i]
                    new_v[flat_idx] = gt_v[mask][mask_i][i]

                # assert each row in aligned_i lies in new_pos
                if not torch.allclose(new_pos[mask][matched_indices], transformed):
                    print(f"The gt_pos does not lie in new_pos, {new_pos[mask][matched_indices]} != {transformed}")
                    breakpoint()
                # assert each row in aligned_i lies in new_v
                if not torch.allclose(new_v[mask][matched_indices], gt_v[mask][mask_i]):
                    print(f"The gt_v does not lie in new_v, {new_v[mask][matched_indices]} != {gt_v[mask][mask_i]}")
                    breakpoint()

                if verbose:
                    breakpoint()

            new_pos = new_pos.to(self.device)
            new_v = new_v.to(self.device)

            return new_pos, new_v, torch.tensor(rmsd_list).mean()
        
    def set_mask_zero(self, ligand_mask, pos_gt):
        pos_gt[~ligand_mask,:] = 0
        return pos_gt

    def sample(
        self,
        protein_pos,
        protein_v,
        batch_protein,
        batch_ligand,
        n_nodes,  # B
        sample_steps=1000,
        desc='Val',
        ligand_pos=None,  # for debug
        ligand_v=None,  # for debug
        ligand_mask=None,  # for inpainting
        classifiers=None,
        guide_mode=None,
        pos_grad_weight=1.0,
        type_grad_weight=1.0,
        EPS=0.,
        W_CFG=0.,
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
        if ligand_v is not None:
            ligand_v = F.one_hot(ligand_v, K).float()


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

        sample_traj = []
        theta_traj = []
        y_traj = []

        # TODO: debug
        mu_pos_t = mu_pos_t[batch_ligand]
        theta_h_t = theta_h_t[batch_ligand]
        # y ~ N(α(Ke_k − 1) , αKI)
        if "sde" in self.sampling_strategy or "ode" in self.sampling_strategy:
            num_steps = sample_steps
            self.num_steps = num_steps
            self.K = self.num_classes

            eta = 1e-5
            self.steps = torch.flip(torch.arange(num_steps+1), [0])
            self.times = self.steps.to(torch.float64)/(num_steps) * (1 - eta)
            self.delta_t = (1 - eta) / num_steps

            # continuous coord
            # self.beta_s_coord  = self.sigma1_coord ** (-2 * (1 - self.times)) - 1
            # self.gamma_t_coord = 1 - self.sigma1_coord ** (2 * (1 - self.times))
            # self.alpha_t_coord = 1 - self.sigma1_coord ** (2 * (1 - self.times))
            self.beta_s_coord  = self.sigma1_coord ** (-2 * (self.times)) - 1
            self.gamma_t_coord = 1 - self.sigma1_coord ** (2 * (1 - self.times))
            self.alpha_t_coord = 1 - self.sigma1_coord ** (2 * (1 - self.times))
            self.sigma_t_coord = (self.alpha_t_coord * (1 - self.alpha_t_coord)).sqrt()
            self.lambda_t_coord = torch.log(self.alpha_t_coord) - torch.log(self.sigma_t_coord)

            self.f_t_coord = 2 * torch.log(self.sigma1_coord) * (1 - self.gamma_t_coord) / self.gamma_t_coord
            self.g_t_coord = (-2 * torch.log(self.sigma1_coord) * (1 - self.gamma_t_coord)) ** 0.5

            # discrete type
            # f g
            self.f_t_type = -2 / (1 - self.times)
            self.max_sqrt_beta = torch.sqrt(self.beta1)
            self.g_t_type = (2 * self.num_classes * (1 - self.times))**0.5 * self.max_sqrt_beta

            # beta alpha
            self.beta_t_type  = (self.max_sqrt_beta * (1 - self.times))**2
            self.alpha_t_type = 2 * (1 - self.times) * self.max_sqrt_beta**2

            beta_t = (self.max_sqrt_beta * eta)**2
            std_t = (self.num_classes * beta_t).sqrt()
            y_h = torch.randn_like(theta_h_t) * std_t

            min_variance = self.sigma1_coord.sqrt()
            gamma_t = 1 - min_variance**eta
            std_t = (gamma_t * (1 - gamma_t))**0.5
            mu_pos_t = torch.randn_like(mu_pos_t) * std_t

            coord_pred_last, p0_pred_last = None, None

        y_pred_last, y_pred = None, None
        eps_ode = torch.randn_like(theta_h_t)
        with tqdm(total=sample_steps, desc=f'{desc}-Sampling') as pbar:
            for i in range(1, sample_steps + 1):
                pbar.update(1)
                t = torch.ones((n_nodes, 1)).to(self.device) * (i - 1) / sample_steps
                if not self.use_discrete_t and not self.destination_prediction:
                    t = torch.clamp(t, min=self.t_min)

                t = t[batch_ligand]
                # Eq.(84): γ(t) = σ1^(2t)
                gamma_coord = 1 - torch.pow(self.sigma1_coord, 2 * t)
                # gamma_charge = 1 - torch.pow(self.sigma1_charges, 2 * t)

                # 2. Compute output distribution parameters for p_O (x' | θ; t)
                # output_params = Ψ(θ,t), x_hat or e_hat for output distribution
                # continuous x ~ δ(x − x_hat(θ, t))
                # discrete k^(d) ~ softmax(Ψ^(d)(θ, t))_k

                # debug only
                if ligand_mask is not None:
                    # print("********************************************")
                    # print(np.array(ligand_mask.cpu()))
                    # print(ligand_mask.shape)
                    # print("********************************************")
                    assert ligand_pos is not None and ligand_v is not None
                    assert ligand_pos.shape == mu_pos_t.shape, f"Shape mismatch {ligand_pos.shape} != {mu_pos_t.shape}"  # ref mode for inpainting
                    mu_coord_gt, gamma_coord_gt = self.continuous_var_bayesian_update(t, sigma1=self.sigma1_coord, x=ligand_pos)
                    theta_h_gt = self.discrete_var_bayesian_update(t, beta1=self.beta1, x=ligand_v, K=K)
                    # set where ligand_pos is all zeros to mu_pos_t
                    # mu_coord_gt = torch.where(ligand_pos == 0, mu_pos_t, mu_coord_gt)
                    # mu_coord_gt = torch.where(ligand_pos == 0, mu_pos_t, ligand_pos)
                    ligand_pos_mask = ligand_mask.unsqueeze(-1).expand_as(ligand_pos)
                    mu_pos_t_interpolate = EPS * mu_coord_gt + (1 - EPS) * mu_pos_t
                    mu_pos_t_interpolate[~ligand_mask] = 0
                    # assert mask correctly
                    # i.e. mu_pos_t_interpolate should be zero where ligand_pos is zero
                    assert torch.allclose(mu_pos_t_interpolate[~ligand_mask], torch.zeros_like(mu_pos_t_interpolate[~ligand_mask])), "mu_pos_t_interpolate is not zero where ligand_pos is zero"

                    # if i >= sample_steps // 3:
                    # if self.sampling_strategy == "vanilla_back_130" and i >= sample_steps * 2 // 3:
                    if False:
                        try:
                            mu_pos_t, theta_h_t, rmsd = self.align(
                                mu_pos_t_interpolate,
                                mu_pos_t,
                                theta_h_gt,
                                theta_h_t,
                                ligand_mask,
                                batch_ligand
                            )
                            # log rmsd to tqdm bar
                            pbar.set_postfix(rmsd=rmsd.item())
                        except:
                            ligand_pos_mask = ligand_mask.unsqueeze(-1).expand_as(ligand_pos)
                            mu_pos_t = torch.where(ligand_pos_mask, mu_coord_gt, mu_pos_t_interpolate)
                            ligand_v_mask = ligand_mask.unsqueeze(-1).expand_as(ligand_v)
                            theta_h_gt = self.discrete_var_bayesian_update(t, beta1=self.beta1, x=ligand_v, K=K)
                            theta_h_t = torch.where(ligand_v_mask, theta_h_gt, theta_h_t)
                    else:
                        ligand_pos_mask = ligand_mask.unsqueeze(-1).expand_as(ligand_pos)
                        mu_pos_t = torch.where(ligand_pos_mask, mu_coord_gt, mu_pos_t_interpolate)
                        ligand_v_mask = ligand_mask.unsqueeze(-1).expand_as(ligand_v)
                        theta_h_gt = self.discrete_var_bayesian_update(t, beta1=self.beta1, x=ligand_v, K=K)
                        theta_h_t = torch.where(ligand_v_mask, theta_h_gt, theta_h_t)

                
                coord_pred, new_y_pred, k_hat = self.interdependency_modeling(
                    time=t,
                    protein_pos=protein_pos,
                    protein_v=protein_v,
                    batch_protein=batch_protein,
                    batch_ligand=batch_ligand,
                    theta_h_t=theta_h_t,
                    # mu_pos_t=mu_coord_gt,  # fix mu pos guidance, type decoding
                    # mu_pos_t=mu_pos_t if i > sample_steps/10 else mu_coord_gt,  # early guidance
                    mu_pos_t=mu_pos_t,  # no guidance
                    gamma_coord=gamma_coord,
                    # TODO: add charge
                    # mu_charge_t=mu_charge_t,
                    # gamma_charge=gamma_charge,
                )

                # maintain theta_traj
                theta_traj.append((mu_pos_t, theta_h_t, k_hat))

                y_pred_last, y_pred = y_pred, new_y_pred
                p0_h_pred = F.softmax(new_y_pred, dim=-1)
                if not torch.all(p0_h_pred.isfinite()):
                    p0_h_pred = torch.where(
                        p0_h_pred.isfinite(), p0_h_pred, torch.zeros_like(p0_h_pred)
                    )
                    logging.warn("p0_h_pred is not finite in new_y_pred")

                p0_h_pred = torch.clamp(p0_h_pred, min=1e-6)
                sample_pred = torch.distributions.Categorical(p0_h_pred).sample()
                sample_pred = F.one_hot(sample_pred, num_classes=K)

                if ('sde' not in self.sampling_strategy and 'ode' not in self.sampling_strategy) or self.sampling_strategy == 'end_back_ode':
                    # TODO delete the following condition
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

                if "vanilla" in self.sampling_strategy:

                    if classifiers is not None:
                        input_types = [classifier.input_type for classifier in classifiers if classifier.prop_name != "self"]
                        assert len(set(input_types)) <= 1, f"input types are not consistent, {input_types}"
                    
                        # calculate gradients
                        mu_pos_t = mu_pos_t.clone().detach().requires_grad_(True)
                        theta_h_t = theta_h_t.clone().detach().requires_grad_(True)
                        coord_pred_0 = None
                        p0_h_pred_0 = None

                        type_grad_list = []
                        pos_grad_list = []
                        exp_list = []
                        for classifier in classifiers:
                            if classifier.prop_name == "self":
                                if W_CFG != 0:
                                    coord_pred_0, p0_h_pred_0, k_hat_0 = classifier.interdependency_modeling(
                                        time=t,
                                        protein_pos=protein_pos,
                                        protein_v=protein_v,
                                        batch_protein=batch_protein,
                                        batch_ligand=batch_ligand,
                                        theta_h_t=theta_h_t,
                                        mu_pos_t=mu_pos_t,
                                        gamma_coord=gamma_coord,
                                    )
                            elif type_grad_weight + pos_grad_weight > 0:
                                with torch.enable_grad():
                                    if classifier.input_type == "parameter":
                                        exp_pred, atom_prop = classifier.interdependency_modeling(
                                            time=t,
                                            protein_pos=protein_pos,
                                            protein_v=protein_v,
                                            batch_protein=batch_protein,
                                            batch_ligand=batch_ligand,
                                            theta_h_t=theta_h_t,
                                            mu_pos_t=mu_pos_t,
                                            gamma_coord=gamma_coord,
                                        )
                                    
                                        final_exp_pred_log = exp_pred.log()
                                        type_grad = torch.autograd.grad(final_exp_pred_log, theta_h_t, grad_outputs=torch.ones_like(exp_pred), retain_graph=True)[0]
                                        pos_grad = torch.autograd.grad(final_exp_pred_log, mu_pos_t, grad_outputs=torch.ones_like(exp_pred), retain_graph=True)[0]
                                    elif classifier.input_type == "data":
                                        raise NotImplementedError("data input type not implemented")
                                    else:
                                        raise NotImplementedError(f"input type {classifier.input_type} not implemented")

                                    type_grad_list.append(type_grad.detach())
                                    pos_grad_list.append(pos_grad.detach())
                                    exp_list.append((exp_pred.detach().cpu(), type_grad.detach().cpu(), pos_grad.detach().cpu()))

                        if len(type_grad_list) > 0:
                            type_grad = torch.stack(type_grad_list, dim=0).mean(dim=0)
                        else:
                            type_grad = torch.zeros_like(theta_h_t)
                        if len(pos_grad_list) > 0:
                            pos_grad = torch.stack(pos_grad_list, dim=0).mean(dim=0)
                        else:
                            pos_grad = torch.zeros_like(mu_pos_t)

                    else:
                        exp_list = []

                    # take the mean of N+1 samples to estimate E[y]
                    # L2-normalize the type_grad and pos_grad to be the same scale of mu and theta
                    # type_grad = type_grad / l2_norm(type_grad) # * l2_norm(theta_h_t)
                    # pos_grad = pos_grad / l2_norm(pos_grad) # * l2_norm(mu_pos_t)

                    N = 2
            
                    if "back" in self.sampling_strategy:
                        back_step = int(self.sampling_strategy.split("_")[-1])
                        # back_step = 0 means no correction
                        # back_step = 1 means one step correction (correct i-2)
                    else:
                        back_step = 0

                    if back_step > 0:
                        if i <= back_step:
                            t = torch.ones((n_nodes, 1)).to(self.device) * i  / sample_steps #next time step
                            t = t[batch_ligand]
                            # Eq.(77): p_F(θ|x;t) ~ N (μ | γ(t)x, γ(t)(1 − γ(t))I)
                            # γ(t)x += pos_grad * γ(t)(1 − γ(t)) <=> x += pos_grad * (1 - gamma_coord)
                            coord_pred = coord_pred + pos_grad_weight * pos_grad * (1 - gamma_coord)
                            # Eq.(78): p_F(θ|x;t) ~ N (y | β(t)(Ke_x−1), β(t)KI)
                            p0_h_pred = p0_h_pred + type_grad_weight * type_grad
                            if coord_pred_0 is not None:
                                coord_pred = coord_pred + W_CFG * (coord_pred_0 - coord_pred)
                                assert p0_h_pred_0 is not None
                                p0_h_pred = p0_h_pred + W_CFG * (p0_h_pred_0 - p0_h_pred)
                            mu_pos_t, _ = self.continuous_var_bayesian_update(t, sigma1=self.sigma1_coord, x=coord_pred)
                            theta_h_t = self.discrete_var_bayesian_update(t, beta1=self.beta1, x=p0_h_pred, K=K)
                            ro_coord = ro_coord + alpha_coord

                        else:
                            coord_pred = coord_pred + pos_grad_weight * pos_grad / (ro_coord + alpha_coord)
                            final_ligand_v = torch.log(p0_h_pred)
                            final_ligand_v = final_ligand_v + type_grad_weight * type_grad
                            if coord_pred_0 is not None:
                                coord_pred = coord_pred + W_CFG * (coord_pred_0 - coord_pred)
                                assert p0_h_pred_0 is not None
                                final_ligand_v0 = torch.log(p0_h_pred_0)
                                final_ligand_v = final_ligand_v + W_CFG * (final_ligand_v0 - final_ligand_v)
                            p0_h_pred = torch.nn.functional.softmax(final_ligand_v, dim=-1)

                            # 4. Bayesian update input parameters θ_i = h(θ_{i−1-l}, y, α_i+...+α_{i-l}) for p_I(x | θ_i; t_i)
                            # reset mu_pos_t and theta_h_t to step i-back_step
                            mu_pos_t_back = theta_traj[i-back_step-1][0]
                            theta_h_t_back = theta_traj[i-back_step-1][1]
                            # for continuous, α_sum = α_i + ... + α_{i-l+1} = β(t_i) - β(t_{i-l})
                            # = σ1 ** (−2i/n) - σ1 ** (−2(i-l)/n)) = σ1 ** (−2i/n) * (1 − σ1 ** 2l/n)
                            alpha_coord_sum = torch.pow(self.sigma1_coord, -2 * i / sample_steps) * (
                                1 - torch.pow(self.sigma1_coord, 2 * (back_step) / sample_steps)
                            )
                            alpha_coord_diff = torch.pow(self.sigma1_coord, -2 * i / sample_steps) - torch.pow(self.sigma1_coord, -2 * (i - back_step) / sample_steps)
                            assert torch.allclose(alpha_coord_sum, alpha_coord_diff, atol=1e-3), f"alpha_coord_sum != alpha_coord_diff, {alpha_coord_sum} != {alpha_coord_diff}.\nShould be {torch.pow(self.sigma1_coord, -2 * i / sample_steps)} - {torch.pow(self.sigma1_coord, -2 * (i - back_step) / sample_steps)}\nOr {torch.pow(self.sigma1_coord, -2 * i / sample_steps)} * (1 - {torch.pow(self.sigma1_coord, 2 * (back_step) / sample_steps)})"
                            
                            # for discrete, α_sum = α_i + ... + α_{i-l} = β(t_i) - β(t_{i-l})
                            # = β(t_i) - β(t_{i-l}) = β(1) * (i**2 − (i-l)**2 / n**2
                            alpha_h_sum = self.beta1 * (back_step) * (2 * i - back_step) / (sample_steps**2)
                            alpha_h_diff = self.beta1 * (i ** 2 - (i - back_step) ** 2) / (sample_steps ** 2)
                            assert torch.allclose(alpha_h_sum, alpha_h_diff), f"alpha_h_sum != alpha_h_diff, {alpha_h_sum} != {alpha_h_diff}."
                            
                            # ro_i = 1 + β(t_i) = σ1 ** (−2i/n)
                            # ro_i = ro_coord (ro_{i-1}) + alpha_coord (α_i)
                            # ro_{i-l} = 1 + β(t_{i-l}) = σ1 ** (−2(i-l)/n) 
                            ro_back_direct = torch.pow(self.sigma1_coord, (-2 * (i - back_step) / sample_steps))
                            ro_back_diff = ro_coord + alpha_coord - alpha_coord_sum

                            # ro_back = ro_coord  # actually ro_{i-1}
                            # assert ro_back.dtype == torch.float32, f"ro_back is not float, {ro_back.dtype}"
                            # i_back = i-1, i-2, ..., i-back_step+1
                            # for i_back in range(i-1, i-back_step, -1):
                            #     # Eq.(95) α_i = σ1 ** (−2i/n) * (1 − σ1 ** (2/n))
                            #     alpha_coord_back = torch.pow(self.sigma1_coord, -2 * i_back / sample_steps) * (
                            #         1 - torch.pow(self.sigma1_coord, 2 / sample_steps)
                            #     )

                            #     # for continuous, Eq.(49): ρi = ρ_{i−1} + α,
                            #     ro_back = ro_back - alpha_coord_back  # can update for i_back - 1 step

                            # set tolerance for float comparison
                            # assert torch.allclose(ro_back, ro_back_direct, atol=1e-4), f"ro_back != ro_back_direct, {ro_back} != {ro_back_direct}"
                            assert torch.allclose(ro_back_direct, ro_back_diff, atol=1e-3), f"ro_back_direct != ro_back_diff, {ro_back_direct} != {ro_back_diff}"

                            # new sample generation
                            if "sample_mean" == guide_mode:
                                y_coords = []
                                y_hs = []
                                for _ in range(N + 1):
                                    y_coord = coord_pred + torch.randn_like(coord_pred) * torch.sqrt(1 / alpha_coord_sum)
                                    k = torch.distributions.Categorical(probs=p0_h_pred).sample()
                                    e_k = F.one_hot(k, num_classes=K).float()
                                    mean = alpha_h_sum * (K * e_k - 1)
                                    var = alpha_h_sum * K
                                    std = torch.full_like(mean, fill_value=var).sqrt()
                                    y_h = mean + std * torch.randn_like(e_k)
                                    y_coords.append(y_coord)
                                    y_hs.append(y_h)
                                y_coord = torch.stack(y_coords, dim=0).mean(dim=0)
                                y_h = torch.stack(y_hs, dim=0).mean(dim=0)
                            elif "sample_mode" == guide_mode:
                                y_coord = coord_pred
                                y_h = p0_h_pred 
                            else:
                                y_coord = coord_pred + torch.randn_like(coord_pred) * torch.sqrt(1 / alpha_coord_sum)
                                k = torch.distributions.Categorical(probs=p0_h_pred).sample()
                                e_k = F.one_hot(k, num_classes=K).float()
                                mean = alpha_h_sum * (K * e_k - 1)
                                var = alpha_h_sum * K
                                std = torch.full_like(mean, fill_value=var).sqrt()
                                y_h = mean + std * torch.randn_like(e_k)

                            # Eq.(62): μi = [μ_{i−2}ρ_{i−2} + x(α_i + α_{i-1}] / ρi
                            mu_pos_t = (ro_back_diff * mu_pos_t_back + alpha_coord_sum * y_coord) / (
                                ro_coord + alpha_coord
                            )
                            ro_coord = ro_coord + alpha_coord
                            theta_prime = torch.exp(y_h) * theta_h_t_back
                            theta_h_t = theta_prime / theta_prime.sum(dim=-1, keepdim=True)
                        
                    else:
                        coord_pred = coord_pred + pos_grad_weight * pos_grad / (ro_coord + alpha_coord)
                        final_ligand_v = torch.log(p0_h_pred)
                        final_ligand_v = final_ligand_v + type_grad_weight * type_grad
                        if coord_pred_0 is not None:
                            coord_pred = coord_pred + W_CFG * (coord_pred_0 - coord_pred)
                            assert p0_h_pred_0 is not None
                            final_ligand_v0 = torch.log(p0_h_pred_0)
                            final_ligand_v = final_ligand_v + W_CFG * (final_ligand_v0 - final_ligand_v)
                        p0_h_pred = torch.nn.functional.softmax(final_ligand_v, dim=-1)

                        if "sample_mean" == guide_mode:
                            y_coords = [y_coord]
                            y_hs = [y_h]
                            for _ in range(N):
                                y_coord = coord_pred + torch.randn_like(coord_pred) * torch.sqrt(1 / alpha_coord)
                                k = torch.distributions.Categorical(probs=p0_h_pred).sample()
                                e_k = F.one_hot(k, num_classes=K).float()
                                mean = alpha_h * (K * e_k - 1)
                                std = torch.full_like(mean, fill_value=var).sqrt()
                                y_h = mean + std * torch.randn_like(e_k)
                                y_coords.append(y_coord)
                                y_hs.append(y_h)
                            y_coord = torch.stack(y_coords, dim=0).mean(dim=0)
                            y_h = torch.stack(y_hs, dim=0).mean(dim=0)
                        elif "sample_mode" == guide_mode:
                            y_coord = coord_pred
                            y_h = p0_h_pred
                        
                        # 4. Bayesian update input parameters θ_i = h(θ_{i−1}, y, α_i) for p_I(x | θ_i; t_i)
                        # for continuous, Eq.(49): ρi = ρ_{i−1} + α,
                        # Eq.(50): μi = [μ_{i−1}ρ_{i−1} + yα] / ρi
                        mu_pos_t = (ro_coord * mu_pos_t + alpha_coord * y_coord) / (
                            ro_coord + alpha_coord
                        )
                        ro_coord = ro_coord + alpha_coord

                        # for discrete, Eq.(171): h(θi−1, y, α) := e**y * θ_{i−1} / \sum_{k=1}^K e**y_k (θ_{i−1})_k
                        theta_prime = torch.exp(y_h) * theta_h_t  # e^y * θ_{i−1}
                        theta_h_t = theta_prime / theta_prime.sum(dim=-1, keepdim=True)

                    assert torch.allclose(ro_coord, self.sigma1_coord ** (-2 * i / sample_steps)), f"[flag] ro_coord + alpha_coord != sigma1 ** (-2i/n), {ro_coord + alpha_coord} != {self.sigma1_coord ** (-2 * i / sample_steps)}"
                    sample_traj.append((coord_pred, p0_h_pred, k_hat, exp_list))
                    y_traj.append((y_coord, y_h, k_hat))


                elif "end_back" in self.sampling_strategy:
                    y_traj.append((y_coord, y_h, k_hat))

                    t = torch.ones((n_nodes, 1)).to(self.device) * i  / sample_steps #next time step
                    t = t[batch_ligand]

                    if classifiers is not None and i > 1:
                        input_types = [classifier.input_type for classifier in classifiers if classifier.prop_name != "self"]
                        assert len(set(input_types)) <= 1, f"input types are not consistent, {input_types}"
                    
                        # calculate gradients
                        mu_pos_t = mu_pos_t.clone().detach().requires_grad_(True)
                        if guide_mode == "param_naive":
                            theta_h_t = theta_h_t.clone().detach().requires_grad_(True)
                        elif guide_mode == "param_logit":
                            y_pred_last = y_pred_last.clone().detach().requires_grad_(True)
                            with torch.enable_grad():
                                theta_h_t = F.softmax(y_pred_last, dim=-1)
                        elif guide_mode == 'param_logit_2':
                            y_pred = y_pred.clone().detach().requires_grad_(True)
                            with torch.enable_grad():
                                theta_h_t = F.softmax(y_pred, dim=-1)
                        else:
                            raise NotImplementedError(f"guide_mode {guide_mode} not implemented")

                        coord_pred_0 = None
                        p0_h_pred_0 = None

                        type_grad_list = []
                        pos_grad_list = []
                        exp_list = []
                        for classifier in classifiers:
                            if classifier.prop_name == "self":
                                if W_CFG != 0:
                                    coord_pred_0, p0_h_pred_0, k_hat_0 = classifier.interdependency_modeling(
                                        time=t,
                                        protein_pos=protein_pos,
                                        protein_v=protein_v,
                                        batch_protein=batch_protein,
                                        batch_ligand=batch_ligand,
                                        theta_h_t=theta_h_t,
                                        mu_pos_t=mu_pos_t,
                                        gamma_coord=gamma_coord,
                                    )
                            elif type_grad_weight != 0 or pos_grad_weight != 0:
                                with torch.enable_grad():
                                    if classifier.input_type == "parameter":
                                        exp_pred, atom_prop = classifier.interdependency_modeling(
                                            time=t,
                                            protein_pos=protein_pos,
                                            protein_v=protein_v,
                                            batch_protein=batch_protein,
                                            batch_ligand=batch_ligand,
                                            theta_h_t=theta_h_t,
                                            mu_pos_t=mu_pos_t,
                                            gamma_coord=gamma_coord,
                                        )
                                        final_exp_pred_log = exp_pred.log()
                                        if guide_mode == "param_naive":
                                            type_grad = torch.autograd.grad(final_exp_pred_log, theta_h_t, grad_outputs=torch.ones_like(exp_pred), retain_graph=True)[0]
                                        elif guide_mode == "param_logit":
                                            type_grad = torch.autograd.grad(final_exp_pred_log, y_pred_last, grad_outputs=torch.ones_like(exp_pred), retain_graph=True)[0]
                                        elif guide_mode == "param_logit_2":
                                            type_grad = torch.autograd.grad(final_exp_pred_log, y_pred, grad_outputs=torch.ones_like(exp_pred), retain_graph=True)[0]
                                        pos_grad = torch.autograd.grad(final_exp_pred_log, mu_pos_t, grad_outputs=torch.ones_like(exp_pred), retain_graph=True)[0]

                                    elif classifier.input_type == "data":
                                        raise NotImplementedError("data input type not implemented")
                                        # TODO: fix this
                                        exp_pred, atom_prop = classifier.interdependency_modeling(
                                            time=t,
                                            protein_pos=protein_pos,
                                            protein_v=protein_v,
                                            batch_protein=batch_protein,
                                            batch_ligand=batch_ligand,
                                            theta_h_t=p0_h_pred,
                                            mu_pos_t=coord_pred,
                                            gamma_coord=gamma_coord,
                                        )                                
                                        final_exp_pred_log = exp_pred.log()
                                        type_grad = torch.autograd.grad(final_exp_pred_log, p0_h_pred, grad_outputs=torch.ones_like(exp_pred), retain_graph=True)[0]
                                        pos_grad = torch.autograd.grad(final_exp_pred_log, coord_pred, grad_outputs=torch.ones_like(exp_pred), retain_graph=True)[0]
                                        # coord_pred, p0_h_pred, k_hat = self.interdependency_modeling(
                                        #     time=t,
                                        #     protein_pos=protein_pos,
                                        #     protein_v=protein_v,
                                        #     batch_protein=batch_protein,
                                        #     batch_ligand=batch_ligand,
                                        #     theta_h_t=theta_h_t,
                                        #     mu_pos_t=mu_pos_t,  # no guidance
                                        #     gamma_coord=gamma_coord,
                                        # )
                                        # exp_pred, atom_prop = classifier.interdependency_modeling(
                                        #     time=t,
                                        #     protein_pos=protein_pos,
                                        #     protein_v=protein_v,
                                        #     batch_protein=batch_protein,
                                        #     batch_ligand=batch_ligand,
                                        #     theta_h_t=p0_h_pred,
                                        #     mu_pos_t=coord_pred,
                                        #     gamma_coord=gamma_coord,
                                        # )
                                        # final_exp_pred_log = exp_pred.log()
                                        # type_grad = torch.autograd.grad(final_exp_pred_log, theta_h_t, grad_outputs=torch.ones_like(exp_pred), retain_graph=True)[0]
                                        # pos_grad = torch.autograd.grad(final_exp_pred_log, mu_pos_t, grad_outputs=torch.ones_like(exp_pred), retain_graph=True)[0]
                                    else:
                                        raise NotImplementedError(f"input type {classifier.input_type} not implemented")

                                    type_grad_list.append(type_grad.detach())
                                    pos_grad_list.append(pos_grad.detach())
                                    exp_list.append((exp_pred.detach().cpu(), type_grad.detach().cpu(), pos_grad.detach().cpu()))
                        if len(type_grad_list) > 0:
                            type_grad = torch.stack(type_grad_list, dim=0).mean(dim=0)
                        else:
                            type_grad = torch.zeros_like(theta_h_t)
                        if len(pos_grad_list) > 0:
                            pos_grad = torch.stack(pos_grad_list, dim=0).mean(dim=0)
                        else:
                            pos_grad = torch.zeros_like(mu_pos_t)
                        
                        # L2-normalize the type_grad and pos_grad to be the same scale of mu and theta
                        # type_grad = type_grad / l2_norm(type_grad) # * l2_norm(theta_h_t)
                        # pos_grad = pos_grad / l2_norm(pos_grad) # * l2_norm(mu_pos_t)
                        
                        # apply gradients in parameter space
                        # Eq.(77): p_F(θ|x;t) ~ N (μ | γ(t)x, γ(t)(1 − γ(t))I)
                        # γ(t)x += pos_grad * γ(t)(1 − γ(t)) <=> x += pos_grad * (1 - gamma_coord)
                        coord_pred = coord_pred + pos_grad_weight * pos_grad * (1 - gamma_coord)
                        # if coord_pred_0 is not None: coord_pred = coord_pred + W_CFG * (coord_pred_0 - coord_pred)
                        # Eq.(78): p_F(θ|x;t) ~ N (y | β(t)(Ke_x−1), β(t)KI)
                        p0_h_pred = p0_h_pred + type_grad_weight * type_grad
                        # if p0_h_pred_0 is not None: p0_h_pred = p0_h_pred + W_CFG * (p0_h_pred_0 - p0_h_pred)
                    else:
                        exp_list = [(torch.zeros((n_nodes, 1)), torch.zeros_like(theta_h_t), torch.zeros_like(mu_pos_t)) for _ in range(len(classifiers))]
                    if self.sampling_strategy == "end_back":
                        raise NotImplementedError("end_back not implemented")
                        theta_h_t = self.discrete_var_bayesian_update(t, beta1=self.beta1, x=sample_pred, K=K)
                    elif self.sampling_strategy == "end_back_pmf":
                        theta_h_t = self.discrete_var_bayesian_update(t, beta1=self.beta1, x=p0_h_pred, K=K)
                    elif self.sampling_strategy == "end_back_ode":
                        theta_h_t = self.discrete_var_bayesian_update(t, beta1=self.beta1, x=p0_h_pred, K=K, eps=eps_ode)
                    elif self.sampling_strategy == "end_back_mode":
                        raise NotImplementedError("end_back_mode not implemented")
                        p0_mode = torch.argmax(p0_h_pred, dim=-1)
                        mode_pred = F.one_hot(p0_mode, num_classes=K).float()
                        theta_h_t = self.discrete_var_bayesian_update(t, beta1=self.beta1, x=mode_pred, K=K)
                    else:
                        raise NotImplementedError(f"sampling strategy {self.sampling_strategy} not implemented")
                    mu_pos_t = self.continuous_var_bayesian_update(t, sigma1=self.sigma1_coord, x=coord_pred)[0]
                    ro_coord = ro_coord + alpha_coord
                    sample_traj.append((coord_pred, sample_pred, k_hat, exp_list))

                    # if i % (sample_steps // 10) == 0:
                        # print(f"theta_h_{i}", theta_h_t)
                        # mu_pos_t size [N,3]
                        # print(f"mu_pos_{i}_min", mu_pos_t.min(dim=0).values.cpu().numpy(), "max", mu_pos_t.max(dim=0).values.cpu().numpy())
                        # log to wandb
                        # wandb.log({"mu_pos_t": mu_pos_t})

                elif "sde" in self.sampling_strategy or "ode" in self.sampling_strategy:
                    # y_traj.append((y_coord, y_h, k_hat))
                    if classifiers is not None:
                        input_types = [classifier.input_type for classifier in classifiers if classifier.prop_name != "self"]
                        assert len(set(input_types)) <= 1, f"input types are not consistent, {input_types}"
                    
                        # calculate gradients
                        mu_pos_t = mu_pos_t.clone().detach().requires_grad_(True)
                        if guide_mode == "param_naive":
                            theta_h_t = theta_h_t.clone().detach().requires_grad_(True)
                        elif guide_mode == "param_logit":
                            y_h = y_h.clone().detach().requires_grad_(True)
                            with torch.enable_grad():
                                theta_h_t = F.softmax(y_h, dim=-1)
                        else:
                            raise NotImplementedError(f"guide_mode {guide_mode} not implemented")

                        type_grad_list = []
                        pos_grad_list = []
                        exp_list = []
                        for classifier in classifiers:
                            if type_grad_weight != 0 and pos_grad_weight != 0:
                                with torch.enable_grad():
                                    if classifier.input_type == "parameter":
                                        exp_pred, atom_prop = classifier.interdependency_modeling(
                                            time=t,
                                            protein_pos=protein_pos,
                                            protein_v=protein_v,
                                            batch_protein=batch_protein,
                                            batch_ligand=batch_ligand,
                                            theta_h_t=theta_h_t,
                                            mu_pos_t=mu_pos_t,
                                            gamma_coord=gamma_coord,
                                        )
                                        final_exp_pred_log = exp_pred.log()
                                        if guide_mode == "param_naive":
                                            type_grad = torch.autograd.grad(final_exp_pred_log, theta_h_t, grad_outputs=torch.ones_like(exp_pred), retain_graph=True)[0]
                                        elif guide_mode == "param_logit":
                                            type_grad = torch.autograd.grad(final_exp_pred_log, y_h, grad_outputs=torch.ones_like(exp_pred), create_graph=True)[0]
                                        pos_grad = torch.autograd.grad(final_exp_pred_log, mu_pos_t, grad_outputs=torch.ones_like(exp_pred), retain_graph=True)[0]

                                    else:
                                        raise NotImplementedError(f"input type {classifier.input_type} not implemented")

                                    type_grad_list.append(type_grad.detach())
                                    pos_grad_list.append(pos_grad.detach())
                                    exp_list.append((exp_pred.detach().cpu(), type_grad.detach().cpu(), pos_grad.detach().cpu()))
                        if len(type_grad_list) > 0:
                            type_grad = torch.stack(type_grad_list, dim=0).mean(dim=0)
                        else:
                            type_grad = torch.zeros_like(theta_h_t)
                        if len(pos_grad_list) > 0:
                            pos_grad = torch.stack(pos_grad_list, dim=0).mean(dim=0)
                        else:
                            pos_grad = torch.zeros_like(mu_pos_t)
                        
                        # L2-normalize the type_grad and pos_grad to be the same scale of mu and theta
                        # type_grad = type_grad / l2_norm(type_grad) # * l2_norm(theta_h_t)
                        # pos_grad = pos_grad / l2_norm(pos_grad) # * l2_norm(mu_pos_t)
                        
                        # apply gradients in parameter space
                        # Eq.(77): p_F(θ|x;t) ~ N (μ | γ(t)x, γ(t)(1 − γ(t))I)
                        # γ(t)x += pos_grad * γ(t)(1 − γ(t)) <=> x += pos_grad * (1 - gamma_coord)
                        coord_pred = coord_pred + pos_grad_weight * pos_grad * (1 - gamma_coord)
                        
                        # Eq.(78): p_F(θ|x;t) ~ N (y | β(t)(Ke_x−1), β(t)KI)
                        p0_h_pred = p0_h_pred + type_grad_weight * type_grad

                    else:
                        exp_list = []
                    
                    t = torch.ones((n_nodes, 1)).to(self.device) * i  / sample_steps #next time step
                    t = t[batch_ligand]
                    if torch.isnan(y_pred).any():
                        logging.warn(f"y_pred has nan values, {y_pred}")
                    if self.sampling_strategy == "sde":
                        mu_pos_t, _ = self.sde_euler_update_coord(mu_pos_t, coord_pred, i-1, last_drop=True)
                        y_h, _ = self.sde_euler_update_type(y_h, p0_h_pred, i-1, last_drop=True) # TODO: new_y_pred
                        theta_h_t = F.softmax(y_h, dim=-1)
                    elif self.sampling_strategy == "sde_multi":
                        mu_pos_t, coord_pred_last = self.sde_bfnsolver2_multi_step_update_coord(mu_pos_t, coord_pred, i-1, coord_pred_last, last_drop=True, coord_grad=pos_grad, weight=pos_grad_weight)
                        y_h, p0_pred_last = self.sde_bfnsolver2_multi_step_update_type(y_h, new_y_pred, i-1, p0_pred_last, last_drop=True, type_grad=type_grad, weight=type_grad_weight)
                        theta_h_t = F.softmax(y_h, dim=-1)
                    elif self.sampling_strategy == 'sdec_endt':
                        mu_pos_t, _ = self.sde_euler_update_coord(mu_pos_t, coord_pred, i-1, last_drop=True)
                        theta_h_t = self.discrete_var_bayesian_update(t, beta1=self.beta1, x=p0_h_pred, K=K)
                    elif self.sampling_strategy == 'sdet_endc':
                        mu_pos_t = self.continuous_var_bayesian_update(t, sigma1=self.sigma1_coord, x=coord_pred)[0]
                        y_h, _ = self.sde_euler_update_type(y_h, new_y_pred, i-1, last_drop=True)
                        theta_h_t = F.softmax(y_h, dim=-1)
                    elif self.sampling_strategy == 'endc_odet':
                        mu_pos_t = self.continuous_var_bayesian_update(t, sigma1=self.sigma1_coord, x=coord_pred)[0]
                        y_h, _ = self.ode_euler_update_type(y_h, new_y_pred, i-1, last_drop=True)
                        theta_h_t = F.softmax(y_h, dim=-1)
                    elif self.sampling_strategy == 'endc_ode_bfnt':
                        mu_pos_t = self.continuous_var_bayesian_update(t, sigma1=self.sigma1_coord, x=coord_pred)[0]
                        y_h, _ = self.ode_bfnsolver1_update_type(y_h, new_y_pred, i-1, last_drop=True)
                        theta_h_t = F.softmax(y_h, dim=-1)
                    elif self.sampling_strategy == 'ode':
                        mu_pos_t, _ = self.ode_bfnsolver1_update_coord(mu_pos_t, coord_pred, i-1, last_drop=True)
                        y_h, _ = self.ode_bfnsolver1_update_type(y_h, new_y_pred, i-1, last_drop=True)
                        theta_h_t = F.softmax(y_h, dim=-1)

                    # mu_pos_t = self.continuous_var_bayesian_update(t, sigma1=self.sigma1_coord, x=coord_pred)[0]
                    # theta_h_t = self.discrete_var_bayesian_update(t, beta1=self.beta1, x=p0_h_pred, K=K)

                    sample_traj.append((coord_pred, sample_pred, k_hat, exp_list))

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
        if ligand_mask is not None:
            assert ligand_pos is not None and ligand_v is not None
            assert ligand_pos.shape == mu_pos_t.shape, f"Shape mismatch {ligand_pos.shape} != {mu_pos_t.shape}"  # ref mode for inpainting
            ligand_pos_mask = ligand_mask.unsqueeze(-1).expand_as(ligand_pos)
            ligand_v_mask = ligand_mask.unsqueeze(-1).expand_as(ligand_v)
            mu_coord_gt = ligand_pos
            theta_h_gt = ligand_v
            mu_pos_t_interpolate = EPS * mu_coord_gt + (1 - EPS) * mu_pos_t
            mu_pos_t_interpolate = torch.where(ligand_pos_mask, mu_pos_t_interpolate, torch.zeros_like(ligand_pos))
            
            # directly replace the ligand position with the ground truth
            mu_pos_t = torch.where(ligand_pos_mask, ligand_pos, mu_pos_t_interpolate)
            theta_h_t = torch.where(ligand_v_mask, ligand_v, theta_h_t)

        mu_pos_final, p0_h_final, k_hat_final = self.interdependency_modeling(
            time=torch.ones((n_nodes, 1)).to(self.device)[batch_ligand],
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
            theta_h_t=theta_h_t,
            mu_pos_t=mu_pos_t,
            # mu_charge_t=mu_charge_t,
            gamma_coord=1 - self.sigma1_coord**2,  # γ(t) = 1 − (σ1**2) ** t
            # gamma_charge=1 - self.sigma1_charges**2,
        )

        if ligand_mask is not None:
            assert ligand_pos is not None and ligand_v is not None
            assert ligand_pos.shape == mu_pos_t.shape, f"Shape mismatch {ligand_pos.shape} != {mu_pos_t.shape}"  # ref mode for inpainting
            ligand_pos_mask = ligand_mask.unsqueeze(-1).expand_as(ligand_pos)
            ligand_v_mask = ligand_mask.unsqueeze(-1).expand_as(ligand_v)
            mu_coord_gt = torch.where(ligand_pos_mask, ligand_pos, torch.zeros_like(ligand_pos))
            theta_h_gt = ligand_v
            mu_pos_final_interpolate = EPS * mu_coord_gt + (1 - EPS) * mu_pos_final
            mu_pos_final = torch.where(ligand_pos_mask, ligand_pos, mu_pos_final_interpolate)
            p0_h_final = torch.where(ligand_v_mask, ligand_v, p0_h_final)

        # TODO delete the following condition
        if not torch.all(p0_h_final.isfinite()):
            p0_h_final = torch.where(
                p0_h_final.isfinite(), p0_h_final, torch.zeros_like(p0_h_final)
            )
            logging.warn("p0_h_pred is not finite")
        p0_h_final = torch.clamp(p0_h_final, min=1e-6)
        theta_traj.append((mu_pos_final, p0_h_final, k_hat_final))

        # 6. Draw final sample from p_O (· | θ_n, 1)
        # Update: directly take the mode of categorical distribution (as done in BFN paper)
        k_final = p0_h_final

        if classifiers is not None:
            input_types = [classifier.input_type for classifier in classifiers if classifier.prop_name != "self"]
            assert len(set(input_types)) <= 1, f"input types are not consistent, {input_types}"
        
            # calculate gradients
            mu_pos_t = mu_pos_t.clone().detach().requires_grad_(True)
            theta_h_t = theta_h_t.clone().detach().requires_grad_(True)
            coord_pred_0 = None
            p0_h_pred_0 = None

            type_grad_list = []
            pos_grad_list = []
            exp_list = []
            for classifier in classifiers:
                if classifier.prop_name == "self":
                    if W_CFG != 0:
                        coord_pred_0, p0_h_pred_0, k_hat_0 = classifier.interdependency_modeling(
                            time=t,
                            protein_pos=protein_pos,
                            protein_v=protein_v,
                            batch_protein=batch_protein,
                            batch_ligand=batch_ligand,
                            theta_h_t=theta_h_t,
                            mu_pos_t=mu_pos_t,
                            gamma_coord=gamma_coord,
                        )
                elif type_grad_weight + pos_grad_weight > 0:
                    with torch.enable_grad():
                        if classifier.input_type == "parameter":
                            exp_pred, atom_prop = classifier.interdependency_modeling(
                                time=t,
                                protein_pos=protein_pos,
                                protein_v=protein_v,
                                batch_protein=batch_protein,
                                batch_ligand=batch_ligand,
                                theta_h_t=theta_h_t,
                                mu_pos_t=mu_pos_t,
                                gamma_coord=gamma_coord,
                            )
                        
                            final_exp_pred_log = exp_pred.log()
                            type_grad = torch.autograd.grad(final_exp_pred_log, theta_h_t, grad_outputs=torch.ones_like(exp_pred), retain_graph=True)[0]
                            pos_grad = torch.autograd.grad(final_exp_pred_log, mu_pos_t, grad_outputs=torch.ones_like(exp_pred), retain_graph=True)[0]
                        elif classifier.input_type == "data":
                            raise NotImplementedError("data input type not implemented")
                        else:
                            raise NotImplementedError(f"input type {classifier.input_type} not implemented")

                        type_grad_list.append(type_grad.detach())
                        pos_grad_list.append(pos_grad.detach())
                        exp_list.append((exp_pred.detach().cpu(), type_grad.detach().cpu(), pos_grad.detach().cpu()))

            if len(type_grad_list) > 0:
                type_grad = torch.stack(type_grad_list, dim=0).mean(dim=0)
            else:
                type_grad = torch.zeros_like(theta_h_t)
            if len(pos_grad_list) > 0:
                pos_grad = torch.stack(pos_grad_list, dim=0).mean(dim=0)
            else:
                pos_grad = torch.zeros_like(mu_pos_t)

        else:
            exp_list = []


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
            sample_traj.append((mu_pos_final, k_final, k_hat_final, exp_list))

        return theta_traj, sample_traj, y_traj

    def sde_euler_update_coord(self, x_s, x0_pred, step, last_drop=False, coord_grad=None, weight=1.0):
        # step = i
        # 
        
        # x_s -> x_t
        # t  = torch.ones_like(x_s, device=x_s.device) * (1 - self.times[step])
        # noise predict and x0 predict
        # with torch.no_grad():
        #     noise_pred = self.unet(x_s, t).reshape(x_s.shape)
        alpha_t, sigma_t = self.alpha_t_coord[step], self.sigma_t_coord[step]
        # x0_pred = (x_s - sigma_t * noise_pred) / alpha_t
        
        # clip x0
        # x0_pred = x0_pred.clip(min=-1.0, max=1.0)
        mu0_pred = x0_pred * alpha_t

        # noise_pred = (x_s - x0_pred * alpha_t) / sigma_t
        noise_pred = (x_s - mu0_pred) / sigma_t

        beta_t, beta_s = self.beta_s_coord[step + 1], self.beta_s_coord[step]
        gamma_s = self.gamma_t_coord[step]
        f = self.f_t_coord[step]
        g = self.g_t_coord[step]
        noise = torch.randn_like(x_s, device=x_s.device)

        # gamma_coord = 1 - self.sigma1_coord ** (2 * step / self.num_steps)
        # assert torch.allclose(gamma_s.float(), gamma_coord.float(), atol=1e-4), f"[flag] gamma_s != 1 - sigma1 ** (2i/n), {gamma_s} != {gamma_coord}"
        # assert torch.allclose(gamma_s, alpha_t, atol=1e-4), f"[flag] gamma_s != alpha_t, {gamma_s} != {alpha_t}"
        # assert torch.allclose((gamma_s * (1-gamma_s)) ** 0.5, sigma_t, atol=1e-4), f"[flag] sigma_t != (gamma_s * (1-gamma_s)) ** 0.5, {sigma_t} != {(gamma_s * (1-gamma_s)) ** 0.5}"

        if last_drop == True and step == self.num_steps - 1:
            return x0_pred, x0_pred
        else:
            # bfn
            x_t = ((beta_t - beta_s) / ((beta_t + 1) * gamma_s) + (beta_s + 1)/(beta_t + 1)) * x_s - (beta_t - beta_s) / (beta_t + 1) * ((1-gamma_s)/gamma_s)**0.5 * noise_pred + (beta_t-beta_s)**0.5/(beta_t + 1) * noise
            # sde
            # neg_score_estimated = noise_pred / (gamma_s * (1-gamma_s))**0.5  # TODO: wrong scale
            # neg_score_estimated = noise_pred / (gamma_s * (1-gamma_s))
            # if weight != 0 and coord_grad is not None:
            #     neg_score_estimated -= weight * coord_grad
            # x_t = x_s - (f * x_s + g**2 * neg_score_estimated) * self.delta_t + g * self.delta_t**0.5 * noise
        return x_t, x0_pred

    def sde_euler_update_type(self, x_s, logits, step, last_drop=False, cate_samp=False, addi_step=False, type_grad=None, weight=1.0):
        # x_s -> x_t
        # t = torch.ones(x_s.size()[:-1], device=x_s.device) * (1 - self.times[step])

        g = self.g_t_type[step]

        noise = torch.randn_like(x_s, device=x_s.device)


        with torch.no_grad():
            # theta = F.softmax(x_s, -1)
            # logits = self.unet(theta, t)

            data_pred = F.softmax(logits, -1)

            # if weight != 0:
            #     data_pred = data_pred + weight * type_grad

            if cate_samp == True:
                categorical = torch.distributions.categorical(logits=logits, validate_args=False)
                data_pred = categorical.sample()
                data_pred = F.one_hot(data_pred.long(), self.num_classes)

            if last_drop == True and step == self.num_steps - 1:
                return logits, data_pred    

            else:
                x_t = x_s + g**2 * (data_pred - 1/self.num_classes) * self.delta_t + g * self.delta_t**0.5 * noise
                return x_t, data_pred

    def ode_bfnsolver1_update_coord(self, x_s, x0_pred, step, last_drop=False):
        # x_s -> x_t
        t = torch.ones_like(x_s, device=x_s.device) * (1 - self.times[step])
        # noise predict and x0 predict
        # with torch.no_grad():
        #     noise_pred = self.unet(x_s, t).reshape(x_s.shape)
        alpha_t, sigma_t = self.alpha_t_coord[step], self.sigma_t_coord[step]
        # x0_pred = (x_s - sigma_t * noise_pred) / alpha_t

        # clip x0
        # x0_pred = x0_pred.clip(min=-1.0, max=1.0)
        noise_pred = (x_s - x0_pred * alpha_t) / sigma_t

        # get schedule
        lambda_t, lambda_s = self.lambda_t_coord[step + 1], self.lambda_t_coord[step]
        alpha_t, alpha_s = self.alpha_t_coord[step + 1], self.alpha_t_coord[step]
        sigma_t, sigma_s = self.sigma_t_coord[step + 1], self.sigma_t_coord[step]
        h = lambda_t - lambda_s

        if last_drop == True and step == self.num_steps - 1:
            return x0_pred, x0_pred
        else:
            x_t = (alpha_t / alpha_s) * x_s - (sigma_t * (torch.exp(h) - 1.0)) * noise_pred

        return x_t, x0_pred

    def ode_euler_update_type(self, x_s, logits, step, last_drop=False, cate_samp=False, addi_step=False):
        # x_s -> x_t
        # t = torch.ones(x_s.size()[:-1], device=x_s.device) * (1 - self.times[step])

        f = self.f_t_type[step]
        g = self.g_t_type[step]
        beta_s = self.beta_t_type[step]

        with torch.no_grad():
            theta = F.softmax(x_s, -1)
            # logits = self.unet(theta, t)
            data_pred = F.softmax(logits, -1)
            if cate_samp == True:
                categorical = torch.distributions.categorical(logits=logits, validate_args=False)
                data_pred = categorical.sample()
                data_pred = F.one_hot(data_pred.long(), self.K)
            if last_drop == True and step == self.num_steps - 1:
                return logits, data_pred
            # elif addi_step == True and step == self.num_steps - 1:
            #     x_t = x_s - ((f + (g**2)/(2 * self.K * beta_s)) * x_s - 0.5 * g**2 *(data_pred -1/self.K)) * self.delta_t
            #     theta = F.softmax(x_t, -1)
            #     t = torch.ones(x_s.size()[:-1], device=x_s.device) * (1 - self.times[step+1])
            #     logits = self.unet(theta, t)
            #     data_pred = F.softmax(logits, -1)
            #     return logits, data_pred
            else:
                x_t = x_s - ((f + (g**2)/(2 * self.K * beta_s)) * x_s - 0.5 * g**2 *(data_pred -1/self.K)) * self.delta_t
                return x_t, data_pred

    def ode_bfnsolver1_update_type(self, x_s, logits, step, last_drop=False):
        # x_s -> x_t
        t = torch.ones(x_s.size()[:-1], device=x_s.device) * (1 - self.times[step])
        t_t, t_s = self.times[step + 1], self.times[step]
        c_t = self.K * self.max_sqrt_beta**2 * (1 - t_t)
        with torch.no_grad():
            theta = F.softmax(x_s, -1)
            # logits = self.unet(theta, t)
            data_pred = F.softmax(logits, -1)

            if last_drop == True and step == self.num_steps - 1:
                return logits, data_pred
            else:
                x_t = (1-t_t)/(1-t_s) * x_s +c_t * (t_t -t_s) * ( 1 / self.K - data_pred)
                return x_t, data_pred


    def sde_bfnsolver2_multi_step_update_coord(self, x_s0, x0_pred_s0, step, x0_pred_last=None, last_drop=False, coord_grad=None, weight=1.0):
        lambda_t, lambda_s0 = self.lambda_t_coord[step + 1], self.lambda_t_coord[step],
        alpha_t, alpha_s0 = self.alpha_t_coord[step + 1], self.alpha_t_coord[step]
        sigma_t, sigma_s0 = self.sigma_t_coord[step + 1], self.sigma_t_coord[step]
        h = lambda_t - lambda_s0

        # timestep_s0 = torch.ones_like(x_s0, device=x_s0.device) * (1 - self.times[step])
        # with torch.no_grad():
        #     noise_pred_s0 = self.unet(x_s0, timestep_s0).reshape(x_s0.shape)
        # x0_pred_s0 = (x_s0 - sigma_s0 * noise_pred_s0) / alpha_s0
        # x0_pred_s0 = x0_pred_s0.clip(-1, 1)

        if weight != 0:
            x0_pred_s0 = x0_pred_s0 + weight * coord_grad

        noise = torch.randn_like(x_s0, device=x_s0.device)
        if step == 0:
            x_t = (sigma_t / sigma_s0 * torch.exp(-h)) * x_s0 + \
                  alpha_t * (1 - torch.exp(-2.0 * h)) * x0_pred_s0 + sigma_t * torch.sqrt(
                1.0 - torch.exp(-2 * h)) * noise
            return x_t, x0_pred_s0
        elif last_drop == True and step == self.num_steps - 1:
            return x0_pred_s0, x0_pred_s0
        else:
            lambda_s1 = self.lambda_t_coord[step - 1]
            h_0 = lambda_s0 - lambda_s1  # 和paper相反，和diffusers一致
            r = h_0 / h
            D1 = (x0_pred_s0 - x0_pred_last) / r
            x_t = (sigma_t / sigma_s0 * torch.exp(-h)) * x_s0 + alpha_t * (1 - torch.exp(-2.0 * h)) * x0_pred_s0 \
                  + 0.5 * alpha_t * (1 - torch.exp(-2.0 * h)) * D1 + sigma_t * torch.sqrt(
                1.0 - torch.exp(-2.0 * h)) * noise
            return x_t, x0_pred_s0

    def sde_bfnsolver2_multi_step_update_type(self, x_s, logits, step, data_pred_last=None, last_drop=False, type_grad=None, weight=1.0):
        # t = torch.ones(x_s.size()[:-1], device=x_s.device) * (1 - self.times[step])
        t_t, t_s = self.times[step + 1], self.times[step]
        beta_s = self.max_sqrt_beta**2 * (1 - t_s)**2
        beta_t = self.max_sqrt_beta**2 * (1 - t_t)**2
        with torch.no_grad():
            theta = F.softmax(x_s, -1)
            # logits = self.unet(theta, t)
            if weight != 0:
                logits = logits + weight * type_grad

            data_pred_s = F.softmax(logits, -1)
            if step == 0:
                noise = torch.randn_like(x_s, device=x_s.device)
                x_t = x_s + (beta_t - beta_s) * (self.K * data_pred_s - 1)  + (self.K * (beta_t - beta_s))**0.5 * noise
                return x_t, data_pred_s
            elif last_drop == True and step == self.num_steps - 1:
                return logits, data_pred_s
            else:
                noise = torch.randn_like(x_s, device=x_s.device)
                t_r = self.times[step-1]
                D1 = (data_pred_last - data_pred_s)/(t_r - t_s)
                # x_t_ = x_s + (beta_t - beta_s) * (self.K * data_pred_s - 1)\
                #     + (2*self.K*self.max_sqrt_beta**2*( ((t_t**2)/2 - (t_t**3)/3) - ((t_s**2)/2-(t_s**3)/3 ) ) + t_s * self.K * (beta_t - beta_s)) * D1 \
                #         + (self.K * (beta_t - beta_s))**0.5 * noise

                x_t = x_s + (beta_t - beta_s) * (self.K * data_pred_s - 1) \
                    + 1/3 * self.K * self.max_sqrt_beta**2 * (t_t - t_s)**2 * (t_s + 2 * t_t -3) * D1 \
                    + (self.K * (beta_t - beta_s))**0.5 * noise
                return x_t, data_pred_s


class ClassifierScoreModel(BFNBase):
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
        sampling_strategy = "vanilla",
        prop_name = "affinity",
        input_type = "parameter",
    ):
        super(ClassifierScoreModel, self).__init__()
        net_config = Struct(**net_config)
        self.config = net_config

        if net_config.name == 'unio2net':
            self.unio2net = UniTransformerO2TwoUpdateGeneral(**net_config.todict())
        else:
            raise NotImplementedError
        
        self.hidden_dim = net_config.hidden_dim
        self.num_classes = ligand_atom_feature_dim

        self.node_indicator = node_indicator

        if self.node_indicator:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        # atom embedding
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)
        self.center_pos_mode = center_pos_mode  # ['none', 'protein']

        self.time_emb_mode = time_emb_mode
        self.time_emb_dim = time_emb_dim
        if self.time_emb_dim > 0:
            self.time_emb_layer = TimeEmbedLayer(self.time_emb_mode, self.time_emb_dim)
        self.ligand_atom_emb = nn.Linear(
            ligand_atom_feature_dim + self.time_emb_dim, emb_dim
        )

        self.expert_pred = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )  # [hidden to 1]

        self.device = device
        self._edges_dict = {}
        self.condition_time = condition_time
        self.sigma1_coord = torch.tensor(sigma1_coord, dtype=torch.float32)  # coordinate sigma1, a schedule for bfn
        self.beta1 = torch.tensor(beta1, dtype=torch.float32)  # type beta, a schedule for types.
        self.use_discrete_t = use_discrete_t  # whether to use discrete t
        self.discrete_steps = discrete_steps
        self.t_min = t_min
        self.pos_init_mode = pos_init_mode
        self.destination_prediction = destination_prediction
        self.sampling_strategy = sampling_strategy
        self.prop_type = prop_name
        self.input_type = input_type

    def interdependency_modeling(
        self,
        time,
        protein_pos,  # transform from the orginal BFN codebase
        protein_v,  # transform from
        batch_protein,  # index for protein
        theta_h_t,
        mu_pos_t,
        batch_ligand,  # index for ligand
        gamma_coord,
        return_all=False,  # legacy from targetdiff
        fix_x=False,
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

        theta_h_t = 2 * theta_h_t - 1  # from 1/K \in [0,1] to 2/K-1 \in [-1,1]

        init_ligand_v = theta_h_t
        # time embedding [simple, sin, rbf, learn]
        if self.time_emb_dim > 0:
            time_emb = self.time_emb_layer(time)
            input_ligand_feat = torch.cat([init_ligand_v, time_emb], -1)
        else:
            input_ligand_feat = init_ligand_v

        h_protein = self.protein_atom_emb(protein_v)  # [N_protein, self.hidden_dim - 1]
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)  # [N_ligand, self.hidden_dim - 1]

        if self.node_indicator:
            h_protein = torch.cat(
                [h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1
            )  # [N_ligand, self.hidden_dim ]
            init_ligand_h = torch.cat(
                [init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1
            )  # [N_ligand, self.hidden_dim]

        h_all, pos_all, batch_all, mask_ligand = compose_context(
            h_protein=h_protein,
            h_ligand=init_ligand_h,
            pos_protein=protein_pos,
            pos_ligand=mu_pos_t,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )
        # get the context for the protein and ligand, while the ligand is h is noisy (h_t)/ pos is also the noise version. (pos_t)

        # ---------------------

        # time = 2 * time - 1
        outputs = self.unio2net(
            h_all, pos_all, mask_ligand, batch_all, return_all=return_all, fix_x=fix_x
        )
        final_pos, final_h = (
            outputs["x"],
            outputs["h"],
        )  # shape of the pos and shape of h
        final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand]
        
        # apply the expert prediction
        # [N_ligand, 1]
        # if self.exp_input == 'complex':
        #     atom_prop = self.expert_pred(final_h)
        #     final_exp_pred = scatter_mean(atom_prop, batch_all, dim=0)
        # elif self.exp_input == 'ligand':
        atom_prop = self.expert_pred(final_ligand_h)
        final_exp_pred = scatter_mean(atom_prop, batch_ligand, dim=0)
        final_exp_pred = final_exp_pred.squeeze(-1) # [B]
        # else:
        #     atom_prop, final_exp_pred = None, None

        return final_exp_pred, atom_prop

    def reconstruction_loss_one_step(
        self,
        t,  # [N_ligand, 1]
        protein_pos,
        protein_v,
        batch_protein,
        ligand_pos,
        ligand_v,
        batch_ligand,
        prop,
    ):
        assert self.use_discrete_t
        i = (t * self.discrete_steps).int() + 1
        t = torch.ones_like(t) * (i - 1) / self.discrete_steps
        K = self.num_classes
        ligand_v = F.one_hot(ligand_v, K).float()  # [N, K]

        # TODO: implement reconstruction loss (but do we really need it?)
        if self.input_type == "data" or self.input_type == "parameter":

            # take self.input_type == "parameter":
            # 1. Bayesian Flow p_F(θ|x;t), obtain input parameters θ
            # continuous ~ N(μ | γ(t)x, γ(t)(1 − γ(t))I)
            mu_coord, gamma_coord = self.continuous_var_bayesian_update(
                t, sigma1=self.sigma1_coord, x=ligand_pos
            )  # [N, 3], [N, 1]

            # discrete ~ N(y | β(t)(Ke_x−1), β(t)KI)
            theta = self.discrete_var_bayesian_update(
                t, beta1=self.beta1, x=ligand_v, K=K
            )  # [N, K]

            # 2. Compute output distribution parameters for p_O (x' | θ; t) (x_hat or k^(d) logits)
            # continuous x ~ δ(x − x_hat(θ, t))
            # discrete k^(d) ~ softmax(Ψ^(d)(θ, t))_k
            final_exp_pred, atom_prop = self.interdependency_modeling(
                time=t,
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                theta_h_t=theta,
                mu_pos_t=mu_coord,
                batch_ligand=batch_ligand,
                gamma_coord=gamma_coord,
            )  # [N, 3], [N, K], [?]

        elif self.input_type == "sample":
            # 1. Model sender distribution for sample y ~ p_S (y | x'; α)
            # Algorithm (3)
            # for continuous, y.shape == data.shape
            # Eq.(95) α_i = σ1 ** (−2i/n) * (1 − σ1 ** (2/n))
            alpha_coord = torch.pow(self.sigma1_coord, -2 * i / self.discrete_steps) * (
                1 - torch.pow(self.sigma1_coord, 2 / self.discrete_steps)
            ) # (N, 1)
            y_coord = ligand_pos + torch.randn_like(ligand_pos) * torch.sqrt(
                    1 / alpha_coord
            ) # (N, 3)
            # Algorithm (9)
            # for discrete, y \in R^K, while data \in {1,K}, cf. Eq.(141)
            # where e_k is network output p0_h_pred
            # Eq.(193): α_i = β(1) * (2i − 1) / n**2
            alpha_h = self.beta1 * (t**2) # (N, 1)
            # y ~ N(α(Ke_k − 1) , αKI)
            mean = alpha_h * (K * ligand_v - 1) # (N, K)
            std = torch.sqrt(alpha_h * K) # (N, 1)
            y_h = mean + torch.randn_like(ligand_v) * std # (N, K)

            # y_h = (y_h + 1) / 2
  
            final_exp_pred, atom_prop = self.interdependency_modeling(
                time=t,
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                theta_h_t=y_h,
                mu_pos_t=y_coord,
                batch_ligand=batch_ligand,
                gamma_coord=None,
            )  # [N, 3], [N, K], [?]
            

        # 3. Compute reweighted loss (previous [N,] now [B,])
        assert final_exp_pred is not None, "final_exp_pred is None"

        # absolute value
        exp_loss = torch.abs(final_exp_pred - prop)
        # if self.prop_type == "affinity":
        #     exp_loss *= 16 
        # elif self.prop_type == "qed_norm":
        #     # normalize: [0, 1] to [0.01, 0.95]
        #     # un-normalize
        #     exp_loss *=  0.94
        # elif self.prop_type == "sa_norm":
        #     # normalize: [0, 1] to [0.17, 1]
        #     # un-normalize
        #     exp_loss *= 0.83
        return exp_loss


    def loss_one_step(
        self,
        t,  # [N_ligand, 1]
        protein_pos,
        protein_v,
        batch_protein,
        ligand_pos,
        ligand_v,
        batch_ligand,
        prop,
    ):
        K = self.num_classes
        if self.use_discrete_t:
            i = (t * self.discrete_steps).int() + 1
            t = torch.ones_like(t) * (i - 1) / self.discrete_steps

        assert ligand_v.max().item() < K, f"Error: {ligand_v.max().item()} >= {K}"
        ligand_v = F.one_hot(ligand_v, K).float()  # [N, K]

        if self.input_type == "parameter":
            # 1. Bayesian Flow p_F(θ|x;t), obtain input parameters θ
            # continuous ~ N(μ | γ(t)x, γ(t)(1 − γ(t))I)
            mu_coord, gamma_coord = self.continuous_var_bayesian_update(
                t, sigma1=self.sigma1_coord, x=ligand_pos
            )  # [N, 3], [N, 1]

            # discrete ~ N(y | β(t)(Ke_x−1), β(t)KI)
            theta = self.discrete_var_bayesian_update(
                t, beta1=self.beta1, x=ligand_v, K=K
            )  # [N, K]

            # 2. Compute output distribution parameters for p_O (x' | θ; t) (x_hat or k^(d) logits)
            # continuous x ~ δ(x − x_hat(θ, t))
            # discrete k^(d) ~ softmax(Ψ^(d)(θ, t))_k
            final_exp_pred, atom_prop = self.interdependency_modeling(
                time=t,
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                theta_h_t=theta,
                mu_pos_t=mu_coord,
                batch_ligand=batch_ligand,
                gamma_coord=gamma_coord,
            )  # [N, 3], [N, K], [?]
        
        elif self.input_type == "sample":
            # 1. Model sender distribution for sample y ~ p_S (y | x'; α)
            # Algorithm (3)
            # for continuous, y.shape == data.shape
            # Eq.(95) α_i = σ1 ** (−2i/n) * (1 − σ1 ** (2/n))
            alpha_coord = torch.pow(self.sigma1_coord, -2 * i / self.discrete_steps) * (
                1 - torch.pow(self.sigma1_coord, 2 / self.discrete_steps)
            ) # (N, 1)
            y_coord = ligand_pos + torch.randn_like(ligand_pos) * torch.sqrt(
                    1 / alpha_coord
            ) # (N, 3)
            # Algorithm (9)
            # for discrete, y \in R^K, while data \in {1,K}, cf. Eq.(141)
            # where e_k is network output p0_h_pred
            # Eq.(193): α_i = β(1) * (2i − 1) / n**2
            alpha_h = self.beta1 * (t**2) # (N, 1)
            # y ~ N(α(Ke_k − 1) , αKI)
            mean = alpha_h * (K * ligand_v - 1) # (N, K)
            std = torch.sqrt(alpha_h * K) # (N, 1)
            y_h = mean + torch.randn_like(ligand_v) * std # (N, K)

            # y_h = (y_h + 1) / 2
  
            final_exp_pred, atom_prop = self.interdependency_modeling(
                time=t,
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                theta_h_t=y_h,
                mu_pos_t=y_coord,
                batch_ligand=batch_ligand,
                gamma_coord=None,
            )  # [N, 3], [N, K], [?]
            
        elif self.input_type == "data":
            final_exp_pred, atom_prop = self.interdependency_modeling(
                time=t,
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                theta_h_t=ligand_v,
                mu_pos_t=ligand_pos,
                batch_ligand=batch_ligand,
                gamma_coord=None,
            )
        else:
            raise NotImplementedError(f"input type {self.input_type} not implemented")

        # 3. Compute reweighted loss (previous [N,] now [B,])
        assert final_exp_pred is not None, "final_exp_pred is None"
        if final_exp_pred is not None:
            exp_loss = torch.abs(final_exp_pred - prop)
            # exp_loss = (final_exp_pred - prop).pow(2)
        else:
            exp_loss = torch.zeros_like(final_exp_pred)

        return exp_loss

