import math
from typing import Union, Tuple, Optional
from torch_geometric.typing import PairTensor, Adj, OptTensor
from torch.nn import Linear
import torch.nn.functional as F
import torch
from torch import nn
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax


class Trans_Layer(MessagePassing):
    """The version for involving the edge feature. Multiply Msg. Without FFN and norm."""

    _alpha: OptTensor

    def __init__(self, x_channels: int, out_channels: int,
                 heads: int = 1, dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(Trans_Layer, self).__init__(node_dim=0, **kwargs)

        self.x_channels = x_channels
        self.in_channels = in_channels = x_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_key = Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_query = Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_value = Linear(in_channels, heads * out_channels, bias=bias)

        self.lin_edge0 = Linear(edge_dim, heads * out_channels, bias=False)
        self.lin_edge1 = Linear(edge_dim, heads * out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_edge0.reset_parameters()
        self.lin_edge1.reset_parameters()

    def forward(self, x: OptTensor,
                edge_index: Adj,
                edge_attr: OptTensor = None
                ) -> Tensor:
        """"""

        H, C = self.heads, self.out_channels

        x_feat = x
        query = self.lin_query(x_feat).view(-1, H, C)
        key = self.lin_key(x_feat).view(-1, H, C)
        value = self.lin_value(x_feat).view(-1, H, C)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out_x = self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr, size=None)

        out_x = out_x.view(-1, self.heads * self.out_channels)

        return out_x

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tuple[Tensor, Tensor]:

        edge_attn = self.lin_edge0(edge_attr).view(-1, self.heads, self.out_channels)
        edge_attn = torch.tanh(edge_attn)
        alpha = (query_i * key_j * edge_attn).sum(dim=-1) / math.sqrt(self.out_channels)

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # node feature message
        msg = value_j
        msg = msg * torch.tanh(self.lin_edge1(edge_attr).view(-1, self.heads, self.out_channels))
        msg = msg * alpha.view(-1, self.heads, 1)

        return msg

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class TransMixLayer(MessagePassing):
    """The version for involving the edge feature. Multiply Msg. Without FFN and norm.
    Extra attention heads from adjacency matrix."""

    _alpha: OptTensor

    def __init__(self, x_channels: int, out_channels: int, extra_heads: int = 2,
                 heads: int = 4, dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, inf: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.x_channels = x_channels
        self.in_channels = in_channels = x_channels
        self.out_channels = out_channels
        self.heads = heads
        self.extra_heads = extra_heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.sub_heads = sub_heads = heads - extra_heads
        self.sub_channels = sub_channels = (heads * out_channels) // sub_heads
        self.set_inf = inf

        self.lin_key = Linear(in_channels, sub_heads * sub_channels, bias=bias)
        self.lin_query = Linear(in_channels, sub_heads * sub_channels, bias=bias)
        self.lin_value = Linear(in_channels, heads * out_channels, bias=bias)

        self.lin_edge0 = Linear(edge_dim, sub_heads * sub_channels, bias=False)
        self.lin_edge1 = Linear(edge_dim, heads * out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_edge0.reset_parameters()
        self.lin_edge1.reset_parameters()

    def forward(self, x: OptTensor,
                edge_index: Adj,
                edge_attr: OptTensor,
                extra_heads: OptTensor
                ) -> Tensor:
        """"""
        H, E, C, = self.heads, self.sub_heads, self.out_channels

        # expand the extra heads
        cur_extra_heads = extra_heads.size(-1)
        if cur_extra_heads != self.extra_heads:
            n_expand = self.extra_heads // cur_extra_heads
            extra_heads = extra_heads.unsqueeze(-1).repeat(1, 1, n_expand)
            extra_heads = extra_heads.reshape(-1, self.extra_heads)

        x_feat = x
        query = self.lin_query(x_feat).reshape(-1, E, self.sub_channels)
        key = self.lin_key(x_feat).reshape(-1, E, self.sub_channels)
        value = self.lin_value(x_feat).reshape(-1, H, C)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out_x = self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr,
                               extra_heads=extra_heads, size=None)

        out_x = out_x.view(-1, self.heads * self.out_channels)

        return out_x

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor,
                extra_heads: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tuple[Tensor, Tensor]:

        edge_attn = self.lin_edge0(edge_attr).view(-1, self.sub_heads, self.sub_channels)
        edge_attn = torch.tanh(edge_attn)
        alpha = (query_i * key_j * edge_attn).sum(dim=-1) / math.sqrt(self.out_channels)

        # set 0 to -inf/1e-10 in extra_heads
        if self.set_inf:
            extra_inf_heads = extra_heads.clone()
            # extra_inf_heads[extra_inf_heads==0.] = -float('inf')
            extra_inf_heads[extra_inf_heads == 0.] = -1e10
            alpha = torch.cat([extra_inf_heads, alpha], dim=-1)
        else:
            alpha = torch.cat([extra_heads, alpha], dim=-1)

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # node feature message
        msg = value_j
        msg = msg * torch.tanh(self.lin_edge1(edge_attr).view(-1, self.heads, self.out_channels))
        msg = msg * alpha.view(-1, self.heads, 1)

        return msg

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class EdgeGateTransLayer(MessagePassing):
    """The version of edge feature gating."""

    _alpha: OptTensor

    def __init__(self, x_channels: int, out_channels: int,
                 heads: int = 1, dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(EdgeGateTransLayer, self).__init__(node_dim=0, **kwargs)

        self.x_channels = x_channels
        self.in_channels = in_channels = x_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_key = Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_query = Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_value = Linear(in_channels, heads * out_channels, bias=bias)

        self.lin_edge0 = Linear(edge_dim, heads * out_channels, bias=False)
        self.lin_edge1 = Linear(edge_dim, heads * out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_edge0.reset_parameters()
        self.lin_edge1.reset_parameters()

    def forward(self, x: OptTensor,
                edge_index: Adj,
                edge_attr: OptTensor = None
                ) -> Tensor:
        """"""

        H, C = self.heads, self.out_channels

        x_feat = x
        query = self.lin_query(x_feat).view(-1, H, C)
        key = self.lin_key(x_feat).view(-1, H, C)
        value = self.lin_value(x_feat).view(-1, H, C)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out_x = self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr, size=None)

        out_x = out_x.view(-1, self.heads * self.out_channels)

        return out_x

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tuple[Tensor, Tensor]:

        edge_attn = self.lin_edge0(edge_attr).view(-1, self.heads, self.out_channels)
        edge_attn = torch.tanh(edge_attn)
        alpha = (query_i * key_j * edge_attn).sum(dim=-1) / math.sqrt(self.out_channels)

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # node feature message
        msg = value_j
        msg = msg * torch.tanh(self.lin_edge1(edge_attr).view(-1, self.heads, self.out_channels))
        msg = msg * alpha.view(-1, self.heads, 1)

        return msg

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class LearnedSinusodialposEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb
    https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = x.unsqueeze(-1)
        freqs = x * self.weights.unsqueeze(0) * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    """Gaussian basis function layer for 3D distance features"""
    def __init__(self, K, *args, **kwargs):
        super().__init__()
        self.K = K - 1
        self.means = nn.Embedding(1, self.K)
        self.stds = nn.Embedding(1, self.K)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)

    def forward(self, x, *args, **kwargs):
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return torch.cat([x, gaussian(x, mean, std).type_as(self.means.weight)], dim=-1)


class CondGaussianLayer(nn.Module):
    """Gaussian basis function layer for 3D distance features, with time embedding condition"""
    def __init__(self, K, time_dim):
        super().__init__()
        self.K = K - 1
        self.means = nn.Embedding(1, self.K)
        self.stds = nn.Embedding(1, self.K)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 2)
        )
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)

    def forward(self, x, time_emb=None):
        if time_emb is not None:
            scale, shift = self.time_mlp(time_emb).chunk(2, dim=1)
            x = x * (scale + 1) + shift
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return torch.cat([x, gaussian(x, mean, std).type_as(self.means.weight)], dim=-1)


class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale


class CondCoorsNorm(nn.Module):
    def __init__(self, time_dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 1)
        )

    def forward(self, coors, time_emb):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        scale = self.time_emb(time_emb)
        return normed_coors * scale