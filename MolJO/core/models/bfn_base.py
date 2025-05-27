import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torchdiffeq import odeint
from torch_scatter import scatter_mean, scatter_sum
import torch.distributions as dist

# from core.module.egnn_new import EGNN
import numpy as np


LOG2PI = np.log(2 * np.pi)


# def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode="protein"):
#     if mode == "none":
#         offset = 0.0
#         pass
#     elif mode == "protein":
#         offset = scatter_mean(protein_pos, batch_protein, dim=0)
#         protein_pos = protein_pos - offset[batch_protein]
#         ligand_pos = ligand_pos - offset[batch_ligand]
#     else:
#         raise NotImplementedError
#     return protein_pos, ligand_pos, offset


# def corrupt_t_pred(self, mu, t, gamma):
#     # if t < self.t_min:
#     #   return torch.zeros_like(mu)
#     # else:
#     # eps_pred = self.model()
#     t = torch.clamp(t, min=self.t_min)
#     # t = torch.ones((mu.size(0),1)).cuda() * t
#     eps_pred = self.model(mu, t)
#     x_pred = mu / gamma - torch.sqrt((1 - gamma) / gamma) * eps_pred
#     return x_pred


# def find_closet_index(x, y):
#     # x [B,1]
#     # y [1,K]
#     diff = torch.abs(y - x)
#     idx = torch.argmin(diff, dim=-1)

#     return idx


class BFNBase(nn.Module):
    # this is a general method which could be used for implement vector field in CNF or
    def __init__(self, *args, **kwargs):
        super(BFNBase, self).__init__(*args, **kwargs)

    # def zero_center_of_mass(self, x_pos, segment_ids):
    #     size = x_pos.size()
    #     assert len(size) == 2  # TODO check this
    #     seg_means = scatter_mean(x_pos, segment_ids, dim=0)
    #     mean_for_each_segment = seg_means.index_select(0, segment_ids)
    #     x = x_pos - mean_for_each_segment

    #     return x

    def get_k_params(self, bins):
        """
        function to get the k parameters for the discretised variable
        """
        # k = torch.ones_like(mu)
        # ones_ = torch.ones((mu.size()[1:])).cuda()
        # ones_ = ones_.unsqueeze(0)
        list_c = []
        list_l = []
        list_r = []
        for k in range(1, int(bins + 1)):
            # k = torch.cat([k,torch.ones_like(mu)*(i+1)],dim=1
            k_c = (2 * k - 1) / bins - 1
            k_l = k_c - 1 / bins
            k_r = k_c + 1 / bins
            list_c.append(k_c)
            list_l.append(k_l)
            list_r.append(k_r)
        # k_c = torch.cat(list_c,dim=0)
        # k_l = torch.cat(list_l,dim=0)
        # k_r = torch.cat(list_r,dim=0)

        return list_c, list_l, list_l

    def discretised_cdf(self, mu, sigma, x):
        """
        cdf function for the discretised variable
        """
        # in this case we use the discretised cdf for the discretised output function
        mu = mu.unsqueeze(1)
        sigma = sigma.unsqueeze(1)  # B,1,D

        f_ = 0.5 * (1 + torch.erf((x - mu) / (sigma * np.sqrt(2))))
        flag_upper = torch.ge(x, 1)
        flag_lower = torch.le(x, -1)
        f_ = torch.where(flag_upper, torch.ones_like(f_), f_)
        f_ = torch.where(flag_lower, torch.zeros_like(f_), f_)

        return f_

    def continuous_var_bayesian_update(self, t, sigma1, x):
        """
        x: [N, D]
        """
        """
        TODO: rename this function to bayesian flow
        """
        # Eq.(77): p_F(θ|x;t) ~ N (μ | γ(t)x, γ(t)(1 − γ(t))I)
        gamma = 1 - torch.pow(sigma1, 2 * t)  # [B]
        mu = gamma * x + torch.randn_like(x) * torch.sqrt(gamma * (1 - gamma))
        return mu, gamma

    def discrete_var_bayesian_update(self, t, beta1, x, K, eps=None):
        """
        x: [N, K]
        """
        # Eq.(182): β(t) = t**2 β(1)
        beta = beta1 * (t**2)  # (B,)

        # Eq.(185): p_F(θ|x;t) = E_{N(y | β(t)(Ke_x−1), β(t)KI)} δ (θ − softmax(y))
        # can be sampled by first drawing y ~ N(y | β(t)(Ke_x−1), β(t)KI)
        # then setting θ = softmax(y)
        one_hot_x = x  # (N, K)
        mean = beta * (K * one_hot_x - 1)
        std = (beta * K).sqrt()
        if eps is None:
            eps = torch.randn_like(mean)
        y = mean + std * eps
        theta = F.softmax(y, dim=-1)
        return theta

    def discreteised_var_bayesian_update(self, t, sigma1, x):
        """
        x: [N, D]
        Note, this is identical to the continuous_var_bayesian_update
        """
        gamma = 1 - torch.pow(sigma1, 2 * t)
        mu = gamma * x + torch.randn_like(x) * torch.sqrt(gamma * (1 - gamma))
        return mu, gamma

    def ctime4continuous_loss(self, t, sigma1, x_pred, x, segment_ids=None):
        # Eq.(101): L∞(x) = −ln(σ1) * E_{t∼U (0,1), p_F(θ|x;t)} [|x − x_hat(θ,t)|**2 / (σ_1**2)**t]
        if segment_ids is not None:
            loss = scatter_mean(
                torch.pow(sigma1, -2 * t.view(-1))
                * ((x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)),
                segment_ids,
                dim=0,
            )
        else:
            loss = torch.pow(sigma1, -2 * t.view(-1)) * (x_pred - x).view(
                x.shape[0], -1
            ).abs().pow(2).sum(dim=1)
        return -torch.log(sigma1) * loss

    def dtime4continuous_loss(self, i, N, sigma1, x_pred, x, segment_ids=None):
        # TODO not debuged yet
        weight = N * (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * i / N))
        # print(x_pred.shape, x.shape , i.shape,weight.shape)
        # print(segment_ids)
        if segment_ids is not None:
            loss = scatter_mean(
                weight.view(-1) * ((x_pred - x) ** 2).sum(-1), segment_ids, dim=0
            )
        else:
            loss = (
                N
                * (1 - torch.pow(sigma1, 2 / N))
                / (2 * torch.pow(sigma1, 2 * i / N))
                * (x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)
            )

        # print(loss.shape)
        return loss

    def ctime4discrete_loss(self, t, beta1, one_hot_x, p_0, K, segment_ids=None):
        # Eq.(205): L∞(x) = Kβ(1) E_{t∼U (0,1), p_F (θ|x,t)} [t|e_x − e_hat(θ, t)|**2,
        # where e_hat(θ, t) = (\sum_k p_O^(1) (k | θ; t)e_k, ..., \sum_k p_O^(D) (k | θ; t)e_k)
        e_x = one_hot_x  # [N, K]
        e_hat = p_0  # (N, K)
        assert e_x.size() == e_hat.size()
        if segment_ids is not None:
            L_infinity = scatter_mean(
                K * beta1 * t.view(-1) * ((e_x - e_hat) ** 2).sum(dim=-1),
                segment_ids,
                dim=0,
            )
        else:
            L_infinity = K * beta1 * t.view(-1) * ((e_x - e_hat) ** 2).sum(dim=-1)
        return L_infinity

    def dtime4discrete_loss_prob(
        self, i, N, beta1, one_hot_x, p_0, K, n_samples=200, segment_ids=None
    ):
        # this is based on the official implementation of BFN.
        # import pdb
        # pdb.set_trace()
        target_x = one_hot_x  # [D, K]
        e_hat = p_0  # (D,  K)
        alpha = beta1 * (2 * i - 1) / N**2  # [D]
        alpha = alpha.view(-1, 1) # [D, 1]
        classes = torch.arange(K, device=target_x.device).long().unsqueeze(0)  # [ 1, K]
        e_x = F.one_hot(classes.long(), K) #[1,K, K]
        # print(e_x.shape)
        receiver_components = dist.Independent(
            dist.Normal(
                alpha.unsqueeze(-1) * ((K * e_x) - 1), # [D K, K]
                (K * alpha.unsqueeze(-1)) ** 0.5, # [D, 1, 1]
            ),
            1,
        )  # [D,T, K, K]
        receiver_mix_distribution = dist.Categorical(probs=e_hat)  # [D, K]
        receiver_dist = dist.MixtureSameFamily(
            receiver_mix_distribution, receiver_components
        )  # [D, K]
        sender_dist = dist.Independent( dist.Normal(
            alpha* ((K * target_x) - 1), ((K * alpha) ** 0.5)
        ),1)  # [D, K]
        y = sender_dist.sample(torch.Size([n_samples])) 
        loss = N * (sender_dist.log_prob(y) - receiver_dist.log_prob(y)).mean(0).mean(
            -1, keepdims=True
        )
        # loss = (
        #         (sender_dist.log_prob(y) - receiver_dist.log_prob(y))
        #         .mean(0)
        #         .flatten(start_dim=1)
        #         .mean(1, keepdims=True)
        #     )
        # #
        return loss.mean()

    def dtime4discrete_loss(self, i, N, beta1, one_hot_x, p_0, K, segment_ids=None):
        # i in {1,n}
        # Algorithm 7 in BFN
        e_x = one_hot_x  # [D, K]
        e_hat = p_0  # (D, K)
        assert e_x.size() == e_hat.size()
        alpha = beta1 * (2 * i - 1) / N**2  # [D]

        # print(alpha.shape)
        mean_ = alpha * (K * e_x - 1)  # [D, K]
        std_ = torch.sqrt(alpha * K)  # [D,1] TODO check shape
        eps = torch.randn_like(mean_)  # [D,K,]
        y_ = mean_ + std_ * eps
        # modify this line:
        matrix_ek = torch.eye(K, K).unsqueeze(0).to(e_x.device)
        matrix_ek.repeat(alpha.size(0), 1, 1)  # [D,K,K]
        mean_matrix = alpha.unsqueeze(-1) * (K * matrix_ek - 1)  # [D,K,K]
        std_matrix = torch.sqrt(alpha * K).unsqueeze(-1)  #
        likelihood = (
            torch.exp(
                -((y_.unsqueeze(1).repeat(1, K, 1) - mean_matrix) ** 2)
                / (2 * std_matrix**2)
            )
            / (std_matrix * np.sqrt(2 * np.pi))
        ).prod(
            -1
        )  # [D,K]

        if segment_ids is not None:
            L_N = -scatter_mean(
                torch.log((likelihood * e_hat).sum(dim=-1)), segment_ids, dim=0
            )
        else:
            L_N = -torch.log((likelihood * e_hat).sum(dim=-1))  # [D]
        # print(L_N.shape)
        #
        return N * L_N

    def dtime4discrete_loss_gjj(self, i, N, beta1, one_hot_x, p_0, K, segment_ids=None):
        # i in {1,n}
        # Algorithm 7 in BFN
        e_x = one_hot_x  # [D, K]
        e_hat = p_0  # (D, K)
        assert e_x.size() == e_hat.size()
        alpha = beta1 * (2 * i - 1) / N**2  # [D]

        # print(alpha.shape)
        mean_ = alpha * (K * e_x - 1)  # [D, K]
        std_ = torch.sqrt(alpha * K)  # [D,1] TODO check shape
        eps = torch.randn_like(mean_)  # [D,K,]
        y_ = mean_ + std_ * eps  # [D, K]
        # modify this line:
        matrix_ek = torch.eye(K, K).to(e_x.device)  # [K, K]
        mean_matrix = K * matrix_ek - 1  # [K, K]
        std_matrix = torch.sqrt(alpha * K).unsqueeze(-1)  #
        _log_gaussians = (  # [D, K]
            (-0.5 * LOG2PI - torch.log(std_matrix))
            - (y_.unsqueeze(1) - mean_matrix) ** 2 / (2 * std_matrix**2)
        ).sum(-1)

        _inner_log_likelihood = torch.log(
            torch.sum(e_hat * torch.exp(_log_gaussians), dim=-1)
        )  # (D,)

        _inner_log_likelihood = torch.log(e_hat) + _log_gaussians  # [D, K]
        log_likelihood = torch.logsumexp(_inner_log_likelihood, dim=-1)  # [D]

        if segment_ids is not None:
            L_N = -scatter_mean(log_likelihood, segment_ids, dim=0)
        else:
            L_N = -log_likelihood.sum(dim=-1)  # [D]
        # print(L_N.shape)
        #
        return N * L_N

    def ctime4discreteised_loss(self, t, sigma1, x_pred, x, segment_ids=None):
        if segment_ids is not None:
            loss = scatter_sum(
                (x_pred - x).view(x.shape[0], -1).abs().pow(2), segment_ids, dim=0
            )
        else:
            raise NotImplementedError
            loss = (x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)
        return -torch.log(sigma1) * loss * torch.pow(sigma1, -2 * t.view(-1))

    def interdependency_modeling(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def loss_one_step(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError


# class BFN4MolEGNN(BFNBase):
#     def __init__(
#         self,
#         in_node_nf,
#         hidden_nf=64,
#         device="cuda",
#         act_fn=torch.nn.SiLU(),
#         n_layers=4,
#         attention=False,
#         condition_time=True,
#         tanh=False,
#         sigma1_coord=0.02,
#         sigma1_charges=0.02,
#         beta1=3.0,
#         K=5,
#         bins=9,
#         sample_steps=100,
#         t_min=0.0001,
#         include_charge=False,
#         no_diff_coord=False,
#         charge_discretised_loss=False,
#     ):
#         super(BFN4MolEGNN, self).__init__()
#         if include_charge:
#             out_node_nf = in_node_nf + 2
#         else:
#             out_node_nf = in_node_nf + 1

#         self.egnn = EGNN(
#             in_node_nf=in_node_nf + int(condition_time),  # +1 for time
#             hidden_nf=hidden_nf,
#             out_node_nf=out_node_nf,  # need to predict the mean and variance of the charges for discretised data
#             in_edge_nf=0,
#             device=device,
#             act_fn=act_fn,
#             n_layers=n_layers,
#             attention=attention,
#             # normalize=True,
#             tanh=tanh,
#         )
#         self.in_node_nf = in_node_nf

#         self.device = device
#         self._edges_dict = {}
#         self.condition_time = condition_time
#         self.sigma1_coord = torch.tensor(sigma1_coord, dtype=torch.float32)
#         self.beta1 = torch.tensor(beta1, dtype=torch.float32)
#         self.K = K  # number of classes for the discrete variable
#         self.sample_steps = sample_steps
#         self.t_min = t_min
#         self.include_charge = include_charge
#         self.no_diff_coord = no_diff_coord
#         self.k_c, self.k_l, self.k_r = self.get_k_params(bins)
#         self.sigma1_charges = torch.tensor(sigma1_charges, dtype=torch.float32)
#         self.charge_discretised_loss = charge_discretised_loss
#         self.bins = torch.tensor(bins, dtype=torch.float32)

#         # print(in_node_nf + int(condition_time),in_node_nf + int(condition_time)+1)

#     def interdependency_modeling(
#         self,
#         time,
#         theta_h_t,
#         mu_pos_t,
#         gamma_coord,
#         gamma_charge,
#         edge_index,
#         mu_charge_t=None,
#         edge_attr=None,
#         segment_ids=None,
#     ):
#         """
#         Args:
#             time: should be a scalar tensor or the shape of [node_num x batch_size, 1]
#             h_state: [node_num x batch_size, in_node_nf]
#             coord_state: [node_num x batch_size, 3]
#             edge_index: [2, edge_num]
#             edge_attr: [edge_num, 1] / None
#         """

#         K = self.K

#         theta_h_t = 2 * theta_h_t - 1
#         # time = 2 * time - 1
#         if self.condition_time:
#             if np.prod(time.size()) == 1:
#                 # t is the same for all elements in batch.
#                 h_time = torch.empty_like(theta_h_t[:, 0:1]).fill_(time.item())
#             else:
#                 h_time = time
#             h = torch.cat([theta_h_t, h_time], dim=1)

#         if self.include_charge:
#             h = torch.cat([h, mu_charge_t], dim=-1)  # concat the charge to the input
#             mu_pos_t_in = mu_pos_t
#         else:
#             mu_pos_t_in = mu_pos_t
#         # print("mu_pos_t_in", mu_pos_t_in.shape, "h", h.shape)
#         h_final, coord_final = self.egnn(h, mu_pos_t_in, edge_index, edge_attr)
#         # here we want the last two dimensions of h_final is mu_eps and ln_sigma_eps
#         # h_final = [atom_types, charges_mu,charge_sigma, t]

#         if self.no_diff_coord:
#             eps_coord_pred = coord_final
#         else:
#             eps_coord_pred = coord_final - mu_pos_t_in

#         eps_coord_pred = self.zero_center_of_mass(eps_coord_pred, segment_ids)

#         if self.include_charge:
#             # eps_coord_pred = torch.cat([eps_coord_pred, h_final[:, -1:]], dim=-1)
#             mu_charge_eps = h_final[:, -3].unsqueeze(-1)
#             log_sigma_eps = h_final[:, -2].unsqueeze(
#                 -1
#             )  # always predict two for the charges
#             h_final = h_final[
#                 :, :-3
#             ]  # cat the last 3 dims, for charge/eps, charge/sigma, time
#             # mu_charge_eps = h_final[:, -2]
#         else:
#             h_final = h_final[:, :-1]  # cut the last dimension which represented time.
#         coord_pred = (
#             mu_pos_t / gamma_coord
#             - torch.sqrt((1 - gamma_coord) / gamma_coord) * eps_coord_pred
#         )
#         if self.include_charge:
#             if self.charge_discretised_loss:
#                 sigma_charge_eps = torch.exp(log_sigma_eps)
#                 mu_charge_x = (
#                     mu_charge_t / gamma_charge
#                     - torch.sqrt((1 - gamma_charge) / gamma_charge) * mu_charge_eps
#                 )
#                 sigma_charge_x = (
#                     torch.sqrt((1 - gamma_charge) / gamma_charge) * sigma_charge_eps
#                 )
#                 k_r = torch.tensor(self.k_r).to(self.device).unsqueeze(-1).unsqueeze(0)
#                 k_l = torch.tensor(self.k_l).to(self.device).unsqueeze(-1).unsqueeze(0)
#                 # k_c = torch.tensor(self.k_c).to(self.device).unsqueeze(-1).unsqueeze(0)
#                 # print("k_r",k_r.shape,"mu_charge_x",mu_charge_x.shape)
#                 p_o = self.discretised_cdf(
#                     mu_charge_x, sigma_charge_x, k_r
#                 ) - self.discretised_cdf(mu_charge_x, sigma_charge_x, k_l)
#                 k_hat = p_o
#                 # (p_o * k_c).sum(dim=1)
#             else:
#                 """
#                 charge is taken as the continous variable.
#                 the sigma is just not trained and fixed. And the previous mu is considered as the eps
#                 """
#                 # print("badddddddd")
#                 k_hat = (
#                     mu_charge_t / gamma_charge
#                     - torch.sqrt((1 - gamma_charge) / gamma_charge) * mu_charge_eps
#                 )
#         else:
#             k_hat = torch.zeros_like(mu_pos_t)

#         # if self.condition_time:
#         #     # Slice off last dimension which represented time.
#         #     h_final = h_final[:, :-1]
#         if K == 2:
#             p0_1 = torch.sigmoid(h_final)  #
#             p0_2 = 1 - p0_1
#             p0_h = torch.cat((p0_1, p0_2), dim=-1)  #
#         else:
#             p0_h = torch.nn.functional.softmax(h_final, dim=-1)
#         """
#         for discretised variable, we return p_o
#         """
#         # print ("k_hat",k_hat.shape)

#         return coord_pred, p0_h, k_hat

#     def loss_one_step(
#         self,
#         t,
#         x,
#         pos,
#         edge_index,
#         edge_attr=None,
#         segment_ids=None,
#     ):
#         K = self.K

#         if self.include_charge:
#             assert x.size(-1) == K + 1
#             charges = x[:, -1:]
#             x = x[:, :-1]
#             mu_charge, gamma_charge = self.discreteised_var_bayesian_update(
#                 t, sigma1=self.sigma1_charges, x=charges
#             )
#         else:
#             mu_charge = None
#             gamma_charge = None
#             # pos = torch.cat([pos, charges], dim=-1)

#         # print("loss",charges)
#         # print ("x------",x.shape,"pos------",pos.shape)

#         mu_coord, gamma_coord = self.continuous_var_bayesian_update(
#             t, sigma1=self.sigma1_coord, x=pos
#         )
#         theta = self.discrete_var_bayesian_update(t, beta1=self.beta1, x=x, K=K)
#         # if self.include_charge:

#         coord_pred, p0_h, k_hat = self.interdependency_modeling(
#             t,
#             theta_h_t=theta,
#             mu_pos_t=mu_coord,
#             mu_charge_t=mu_charge,
#             gamma_coord=gamma_coord,
#             gamma_charge=gamma_charge,
#             edge_index=edge_index,
#             edge_attr=edge_attr,
#             segment_ids=segment_ids,
#         )
#         if self.include_charge and self.charge_discretised_loss:
#             k_c = torch.tensor(self.k_c).to(self.device).unsqueeze(-1).unsqueeze(0)
#             k_hat = (k_hat * k_c).sum(dim=1)
#             # average
#         # print("x",x.shape,"p0_h",p0_h.shape,"k_hat",k_hat.shape,"charges",charges.shape,mu_charge.shape)

#         closs = self.ctime4continuous_loss(
#             t=t, sigma1=self.sigma1_coord, x_pred=coord_pred, x=pos
#         )
#         dloss = self.ctime4discrete_loss(
#             t=t, beta1=self.beta1, one_hot_x=x, p_0=p0_h, K=K
#         )
#         if self.include_charge:
#             if self.charge_discretised_loss:
#                 discretized_loss = self.ctime4discreteised_loss(
#                     t=t, sigma1=self.sigma1_charges, x_pred=k_hat, x=charges
#                 )
#             else:
#                 discretized_loss = self.ctime4continuous_loss(
#                     t=t, sigma1=self.sigma1_charges, x_pred=k_hat, x=charges
#                 )
#         else:
#             discretized_loss = torch.zeros_like(closs)

#         return closs, dloss, discretized_loss

#     def forward(
#         self, n_nodes, edge_index, sample_steps=None, edge_attr=None, segment_ids=None
#     ):
#         """
#         The function implements a sampling procedure for BFN
#         Args:
#             t: should be a scalar tensor or the shape of [node_num x batch_size, 1, note here we use a single t
#             theta_t: [node_num x batch_size, atom_type]
#             mu_t: [node_num x batch_size, 3]
#             edge_index: [2, edge_num]
#             edge_attr: [edge_num, 1] / None
#         """
#         if self.include_charge:
#             mu_pos_t = torch.zeros((n_nodes, 3)).to(
#                 self.device
#             )  # [N, 4] coordinates prior
#             mu_charge_t = torch.zeros((n_nodes, 1)).to(self.device)

#         else:
#             mu_pos_t = torch.zeros((n_nodes, 3)).to(
#                 self.device
#             )  # [N, 3] coordinates prior
#             mu_charge_t = None

#         theta_h_t = (
#             torch.ones((n_nodes, self.K)).to(self.device) / self.K
#         )  # [N, K] discrete prior
#         ro_coord = 1
#         ro_charge = 1

#         if sample_steps is None:
#             sample_steps = self.sample_steps
#         sample_traj = []
#         theta_traj = []
#         for i in range(1, sample_steps + 1):
#             t = torch.ones((n_nodes, 1)).to(self.device) * (i - 1) / sample_steps
#             t = torch.clamp(t, min=self.t_min)
#             gamma_coord = 1 - torch.pow(self.sigma1_coord, 2 * t)
#             gamma_charge = 1 - torch.pow(self.sigma1_charges, 2 * t)

#             coord_pred, p0_h_pred, k_hat = self.interdependency_modeling(
#                 time=t,
#                 theta_h_t=theta_h_t,
#                 mu_pos_t=mu_pos_t,
#                 mu_charge_t=mu_charge_t,
#                 gamma_coord=gamma_coord,
#                 gamma_charge=gamma_charge,
#                 edge_index=edge_index,
#                 edge_attr=edge_attr,
#                 segment_ids=segment_ids,
#             )
#             # p0_h_pred
#             theta_traj.append((coord_pred, p0_h_pred, k_hat))
#             # maintain theta_traj
#             # TODO delete the following condition
#             if not torch.all(p0_h_pred.isfinite()):
#                 p0_h_pred = torch.where(
#                     p0_h_pred.isfinite(), p0_h_pred, torch.zeros_like(p0_h_pred)
#                 )
#                 logging.warn("p0_h_pred is not finite")
#             p0_h_pred = torch.clamp(p0_h_pred, min=1e-6)
#             sample_pred = torch.distributions.Categorical(p0_h_pred).sample()
#             sample_pred = F.one_hot(sample_pred, num_classes=self.K)

#             # if self.include_charge:
#             #     sample_traj.append((coord_pred[:, :-1], sample_pred))
#             # else:
#             #     sample_traj.append((coord_pred, sample_pred))

#             # [B]
#             alpha_coord = torch.pow(self.sigma1_coord, -2 * i / sample_steps) * (
#                 1 - torch.pow(self.sigma1_coord, 2 / sample_steps)
#             )
#             y_coord = coord_pred + torch.randn_like(coord_pred) * torch.sqrt(
#                 1 / alpha_coord
#             )
#             mu_pos_t = (ro_coord * mu_pos_t + alpha_coord * y_coord) / (
#                 ro_coord + alpha_coord
#             )
#             ro_coord = ro_coord + alpha_coord

#             # update of the discretised variable
#             if self.include_charge:
#                 if not self.charge_discretised_loss:
#                     # for continous like update
#                     alpha_charge = torch.pow(
#                         self.sigma1_charges, -2 * i / sample_steps
#                     ) * (1 - torch.pow(self.sigma1_charges, 2 / sample_steps))
#                     y_charge = k_hat + torch.randn_like(k_hat) * torch.sqrt(
#                         1 / alpha_charge
#                     )
#                     mu_charge_t = (
#                         ro_charge * mu_charge_t + alpha_charge * y_charge
#                     ) / (ro_charge + alpha_charge)
#                     ro_charge = ro_charge + alpha_charge
#                 else:
#                     # for discretised update
#                     alpha_charge = torch.pow(
#                         self.sigma1_charges, -2 * i / sample_steps
#                     ) * (1 - torch.pow(self.sigma1_charges, 2 / sample_steps))
#                     discrete_output = k_hat
#                     discrete_output = torch.transpose(discrete_output, 1, 2)
#                     batch_size = discrete_output.shape[0]
#                     discrete_output = discrete_output.reshape(
#                         -1, discrete_output.shape[-1]
#                     )
#                     # print("discrete_output",discrete_output.shape)
#                     if not torch.all(discrete_output.isfinite()):
#                         discrete_output = torch.where(
#                             discrete_output.isfinite(),
#                             discrete_output,
#                             torch.zeros_like(discrete_output),
#                         )
#                         logging.warn("discrete_output is not finite")
#                     discrete_output = torch.clamp(discrete_output, min=1e-6)

#                     categorical = dist.Categorical(probs=discrete_output)
#                     sample_k = categorical.sample()
#                     sample_k = sample_k.view(batch_size, -1) + 1
#                     sample_k_c = (2 * sample_k - 1) / self.bins - 1
#                     y_charge = sample_k_c + torch.randn_like(sample_k_c) * torch.sqrt(
#                         1 / alpha_charge
#                     )
#                     mu_charge_t = (
#                         ro_charge * mu_charge_t + alpha_charge * y_charge
#                     ) / (ro_charge + alpha_charge)
#                     ro_charge = ro_charge + alpha_charge

#             if self.include_charge:
#                 if self.charge_discretised_loss:
#                     sample_traj.append((coord_pred, sample_pred, sample_k))
#                 else:
#                     k_hat = torch.clamp(k_hat, min=-1, max=1)
#                     k_c = torch.tensor(self.k_c).to(self.device).unsqueeze(0)
#                     k_hat = find_closet_index(k_hat, k_c)
#                     sample_traj.append((coord_pred, sample_pred, k_hat))
#             else:
#                 sample_traj.append((coord_pred, sample_pred, k_hat))

#                 # print("k_hat",k_hat.shape)

#             # update of the discrete variable
#             k = torch.distributions.Categorical(probs=p0_h_pred).sample()
#             alpha_h = self.beta1 * (2 * i - 1) / (sample_steps**2)

#             e_k = F.one_hot(k, num_classes=self.K).float()  #

#             mean = alpha_h * (self.K * e_k - 1)
#             var = alpha_h * self.K
#             std = torch.full_like(mean, fill_value=var).sqrt()

#             y_h = mean + std * torch.randn_like(e_k)

#             theta_prime = torch.exp(y_h) * theta_h_t

#             theta_h_t = theta_prime / theta_prime.sum(dim=-1, keepdim=True)

#         mu_pos_final, p0_h_final, k_hat_final = self.interdependency_modeling(
#             time=torch.ones((n_nodes, 1)).to(self.device),
#             theta_h_t=theta_h_t,
#             mu_pos_t=mu_pos_t,
#             mu_charge_t=mu_charge_t,
#             gamma_coord=1 - self.sigma1_coord**2,
#             gamma_charge=1 - self.sigma1_charges**2,
#             edge_index=edge_index,
#             edge_attr=edge_attr,
#             segment_ids=segment_ids,
#         )
#         # TODO delete the following condition
#         if not torch.all(p0_h_final.isfinite()):
#             p0_h_final = torch.where(
#                 p0_h_final.isfinite(), p0_h_final, torch.zeros_like(p0_h_final)
#             )
#             logging.warn("p0_h_pred is not finite")
#         p0_h_final = torch.clamp(p0_h_final, min=1e-6)
#         # traj.append(mu_pos_final, p0_h_final)
#         theta_traj.append((mu_pos_final, p0_h_final, k_hat_final))
#         k_final = torch.distributions.Categorical(p0_h_final).sample()
#         k_final = F.one_hot(k_final, num_classes=self.K)
#         # if self.include_charge:
#         # sample_traj.append((mu_pos_final[:, :-1], k_final))
#         # else:

#         # print("k_hat_final",k_hat_final.shape)
#         if self.include_charge:
#             if self.charge_discretised_loss:
#                 discretised_output_final = k_hat_final  # [B,Bins,1]
#                 discretised_output_final = torch.transpose(
#                     discretised_output_final, 1, 2
#                 )
#                 batch_size = discretised_output_final.shape[0]
#                 discretised_output_final = discretised_output_final.reshape(
#                     -1, discretised_output_final.shape[-1]
#                 )
#                 # print("discrete_output",discrete_output.shape)

#                 if not torch.all(discretised_output_final.isfinite()):
#                     discretised_output_final = torch.where(
#                         discretised_output_final.isfinite(),
#                         discretised_output_final,
#                         torch.zeros_like(discretised_output_final),
#                     )
#                     logging.warn("discrete_output is not finite")
#                 discretised_output_final = torch.clamp(
#                     discretised_output_final, min=1e-6
#                 )

#                 categorical = dist.Categorical(probs=discretised_output_final)
#                 sample_k = categorical.sample()
#                 sample_k_final = sample_k.view(
#                     batch_size, -1
#                 )  # sample_k_final is the value lies in [0,8]

#                 sample_traj.append((mu_pos_final, k_final, sample_k_final))
#             else:
#                 k_hat_final = torch.clamp(k_hat_final, min=-1, max=1)
#                 k_c = torch.tensor(self.k_c).to(self.device).unsqueeze(0)
#                 k_hat_final = find_closet_index(k_hat_final, k_c)
#                 sample_traj.append((mu_pos_final, k_final, k_hat_final))
#         else:
#             sample_traj.append((mu_pos_final, k_final, k_hat_final))

#         return theta_traj, sample_traj
