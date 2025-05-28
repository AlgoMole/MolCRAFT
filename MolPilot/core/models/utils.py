import torch
_MODELS = {}


def register_model(cls=None, *, name=None):
    """"A decorator for registering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f"Already registerd model")
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def create_model(config):
    model = _MODELS[config.model.name](config)
    model = model.to(config.device)
    model = torch.nn.DataParallel(model)
    return model


############### Utility Functions ###############
# def remove_mean(x):
#     mean = torch.mean(x, dim=1, keepdim=True)
#     x = x - mean
#     return x


# def sample_symmetric_edge_feature_noise(n_samples, n_nodes, edge_ch, edge_mask):
#     """sample symmetric normal noise for edge feature."""
#     z_edge = torch.randn((n_samples, edge_ch, n_nodes, n_nodes), device=edge_mask.device)
#     z_edge = torch.tril(z_edge, -1)
#     z_edge = z_edge + z_edge.transpose(-1, -2)
#     z_edge = z_edge.permute(0, 2, 3, 1) * edge_mask.reshape(n_samples, n_nodes, n_nodes, 1)
#     return z_edge


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff


def coord2diff_adj(x, edge_index, spatial_th=2.):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
    with torch.no_grad():
        adj_spatial = radial.clone()
        adj_spatial[adj_spatial <= spatial_th] = 1.
        adj_spatial[adj_spatial > spatial_th] = 0.
    return radial, adj_spatial


def coord2dist(x, edge_index):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
    return radial


# def to_dense_edge_attr(edge_index, edge_attr, edge_final, bs, n_nodes):
#     edge_idx1, edge_idx2 = edge_index
#     idx0 = torch.div(edge_idx1, n_nodes, rounding_mode='floor')
#     idx1 = edge_idx1 - idx0 * n_nodes
#     idx2 = edge_idx2 - idx0 * n_nodes
#     idx = idx0 * n_nodes * n_nodes + idx1 * n_nodes + idx2
#     idx = idx.unsqueeze(-1).expand(edge_attr.size())
#     edge_final.scatter_add_(0, idx, edge_attr)
#     return edge_final.reshape(bs, n_nodes, n_nodes, -1)


# @torch.no_grad()
# def get_rw_feat(k_step, dense_adj):
#     """Compute k_step Random Walk for given dense adjacency matrix."""

#     rw_list = []
#     deg = dense_adj.sum(-1, keepdims=True)
#     AD = dense_adj / (deg + 1e-8)
#     rw_list.append(AD)

#     for _ in range(k_step):
#         rw = torch.bmm(rw_list[-1], AD)
#         rw_list.append(rw)
#     rw_map = torch.stack(rw_list[1:], dim=1)  # [B, k_step, N, N]

#     # rw_landing = torch.diagonal(rw_map, offset=0, dim1=2, dim2=3)  # [B, k_step, N]
#     # rw_landing = rw_landing.permute(0, 2, 1)  # [B, N, rw_depth]

#     # get the shortest path distance indices
#     tmp_rw = rw_map.sort(dim=1)[0]
#     spd_ind = (tmp_rw <= 0).sum(dim=1)  # [B, N, N]

#     spd_onehot = torch.nn.functional.one_hot(spd_ind, num_classes=k_step+1).to(torch.float)  # [B, N, N, kstep]
#     # spd_onehot = spd_onehot.permute(0, 3, 1, 2)

#     # return rw_landing, spd_onehot
#     return spd_onehot