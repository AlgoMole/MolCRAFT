"""Utils for sampling size of a molecule of a given protein pocket."""

import numpy as np
from scipy import spatial as sc_spatial




def get_space_size(pocket_3d_pos):
    aa_dist = sc_spatial.distance.pdist(pocket_3d_pos, metric='euclidean')
    aa_dist = np.sort(aa_dist)[::-1]
    return np.median(aa_dist[:10])


def _get_bin_idx(space_size, bounds):
    for i in range(len(bounds)):
        if bounds[i] > space_size:
            return i
    return len(bounds)


def sample_atom_num(space_size, db_path=None):
    if db_path is None or 'crossdock' in db_path:
        from core.evaluation.utils.atom_num_config import CONFIG
    elif 'PDBLigAug' in db_path:
        from core.evaluation.utils.atom_num_config_PDBLigAug import CONFIG
    else:
        # raise ValueError(f'Unknown db_path: {db_path}')
        from core.evaluation.utils.atom_num_config import CONFIG
    bin_idx = _get_bin_idx(space_size, CONFIG['bounds'])
    num_atom_list, prob_list = CONFIG['bins'][bin_idx]
    return np.random.choice(num_atom_list, p=prob_list)
