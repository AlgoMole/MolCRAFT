import torch
import torch.nn.functional as F
import numpy as np

from core.datasets.pl_data import ProteinLigandData
from core.datasets import utils as utils_data

AROMATIC_FEAT_MAP_IDX = utils_data.ATOM_FAMILIES_ID['Aromatic']

MAP_BOND_INDEX_TO_TYPE = utils_data.KEKULIZED_INDEX_TO_INDEX

# only atomic number 1, 6, 7, 8, 9, 15, 16, 17 exist
MAP_ATOM_TYPE_FULL_TO_INDEX = {
    (1, 'S', False): 0,
    (6, 'SP', False): 1,
    (6, 'SP2', False): 2,
    (6, 'SP2', True): 3,
    (6, 'SP3', False): 4,
    (7, 'SP', False): 5,
    (7, 'SP2', False): 6,
    (7, 'SP2', True): 7,
    (7, 'SP3', False): 8,
    (8, 'SP2', False): 9,
    (8, 'SP2', True): 10,
    (8, 'SP3', False): 11,
    (9, 'SP3', False): 12,
    (15, 'SP2', False): 13,
    (15, 'SP2', True): 14,
    (15, 'SP3', False): 15,
    (15, 'SP3D', False): 16,
    (16, 'SP2', False): 17,
    (16, 'SP2', True): 18,
    (16, 'SP3', False): 19,
    (16, 'SP3D', False): 20,
    (16, 'SP3D2', False): 21,
    (17, 'SP3', False): 22
}

MAP_ATOM_TYPE_ONLY_TO_INDEX = {
    1: 0,   # H
    6: 1,   # C
    7: 2,   # N
    8: 3,   # O
    9: 4,   # F
    15: 5,  # P
    16: 6,  # S
    17: 7,  # Cl
}

# MAP_ATOM_TYPE_ONLY_TO_INDEX_PDB = {
#     5: 0, # B
#     6: 1,   # C
#     7: 2,   # N
#     8: 3,   # O
#     9: 4,   # F
#     15: 5,  # P
#     16: 6,  # S
#     17: 7,  # Cl
#     35: 8,  # Br
#     53: 9,  # I
# }

MAP_ATOM_TYPE_ONLY_TO_INDEX_PDB = {
    6: 0,   # C
    7: 1,   # N
    8: 2,   # O
    9: 3,   # F
    15: 4,  # P
    16: 5,  # S
    17: 6,  # Cl
    35: 7,  # Br
}

MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {
    (1, False): 0,
    (6, False): 1,
    (6, True): 2,
    (7, False): 3,
    (7, True): 4,
    (8, False): 5,
    (8, True): 6,
    (9, False): 7,
    (15, False): 8,
    (15, True): 9,
    (16, False): 10,
    (16, True): 11,
    (17, False): 12
}

# pdb   aromatic_counter {(6, False): 803564, (8, False): 264143, (7, False): 198073, (6, True): 1371671, (7, True): 209316, (8, True): 8567, (9, False): 72259, (16, False): 21263, (17, False): 26781, (16, True): 11529, (35, False): 3359, (15, False): 2051, (1, False): 998, (53, False): 852, (14, False): 92, (5, False): 57} 
# pdb   full_counter {(6, 'SP3', False): 672520, (6, 'SP2', False): 120223, (8, 'SP2', False): 218819, (6, 'SP', False): 10821, (7, 'SP', False): 6030, (6, 'SP2', True): 1371671, (7, 'SP2', False): 149454, (7, 'SP2', True): 209316, (8, 'SP3', False): 45324, (8, 'SP2', True): 8567, (9, 'SP3', False): 72259, (16, 'SP2', False): 496, (16, 'SP3', False): 20745, (17, 'SP3', False): 26781, (16, 'SP2', True): 11529, (35, 'SP3', False): 3359, (7, 'SP3', False): 42589, (53, 'SP3', False): 852, (15, 'SP3', False): 2051, (5, 'SP3', False): 3, (5, 'SP2', False): 54, (1, 'S', False): 998, (14, 'SP3', False): 92, (16, 'SP3D2', False): 22}

MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX = {
    (1, False): 0,  # 998
    (6, False): 1,  # 803564
    (6, True): 2,   # 1371671
    (7, False): 3,  # 198073
    (7, True): 4,   # 209316
    (8, False): 5,  # 264143
    (8, True): 6,   # 8567
    (9, False): 7,  # 72259 
    (15, False): 8, # 2051
    (16, False): 9, # 21263
    (16, True): 10, # 11529
    (17, False): 11, # 26781
    (35, False): 12, # 3359
}

MAP_ATOM_TYPE_AROMATIC_PDB_ALL_TO_INDEX = {
    (1, False): 0,  # 998
    (6, False): 1,  # 803564
    (6, True): 2,   # 1371671
    (7, False): 3,  # 198073
    (7, True): 4,   # 209316
    (8, False): 5,  # 264143
    (8, True): 6,   # 8567
    (9, False): 7,  # 72259
    (14, False): 8, # 92 -------
    (15, False): 9, # 2051
    (16, False): 10, # 21263
    (16, True): 11, # 11529
    (17, False): 12, # 26781
    (35, False): 13, # 3359
    (53, False): 14, # 852 -----
}

# MAP_ATOM_TYPE_AROMATIC_FULL_TO_INDEX = {
#     (1, False): 0, # 0
#     (3, False): 1, # 5
#     (5, False): 2, # 333
#     (6, False): 3, # 1233508
#     (6, True): 4, # 1435006
#     (7, False): 5, # 262989
#     (7, True): 6, # 207743
#     (8, False): 7, # 622803
#     (8, True): 8, # 8984
#     (9, False): 9, # 52635
#     (12, False): 10, # 45
#     (13, False): 11, # 4
#     (14, False): 12, # 92
#     (15, False): 13, # 40252
#     (15, True): 14, # 1
#     (16, False): 15, # 28827
#     (16, True): 16, # 12789
#     (17, False): 17, # 25978
#     (21, False): 18, # 1
#     (23, False): 19, # 51
#     (24, False): 20, # 5
#     (26, False): 21, # 40
#     (33, False): 22, # 2
#     (34, False): 23, # 48
#     (34, True): 24, # 7
#     (35, False): 25, # 7524
#     (42, False): 26, # 80
#     (44, False): 27, # 53
#     (50, False): 28, # 12
#     (53, False): 29, # 1607
#     (74, False): 30, # 18
#     (79, False): 31, # 4
#     (80, False): 32, # 1
# }

MAP_INDEX_TO_ATOM_TYPE_ONLY = {v: k for k, v in MAP_ATOM_TYPE_ONLY_TO_INDEX.items()}
MAP_INDEX_TO_ATOM_TYPE_ONLY_PDB = {v: k for k, v in MAP_ATOM_TYPE_ONLY_TO_INDEX_PDB.items()}
MAP_INDEX_TO_ATOM_TYPE_AROMATIC = {v: k for k, v in MAP_ATOM_TYPE_AROMATIC_TO_INDEX.items()}
MAP_INDEX_TO_ATOM_TYPE_FULL = {v: k for k, v in MAP_ATOM_TYPE_FULL_TO_INDEX.items()}
MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB = {v: k for k, v in MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX.items()}
# MAP_INDEX_TO_ATOM_TYPE_AROMATIC_FULL = {v: k for k, v in MAP_ATOM_TYPE_AROMATIC_FULL_TO_INDEX.items()}

def get_atomic_number_from_index(index, mode):
    if mode == 'basic' or mode == 'basic_plus_charge' or mode == 'basic_plus_aromatic' or mode == 'basic_plus_full':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_ONLY[i] for i in index.tolist()]
    elif mode == 'basic_PDB' or mode == 'basic_plus_charge_PDB':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_ONLY_PDB[i] for i in index.tolist()]
    elif mode == 'add_aromatic' or mode == 'add_aromatic_plus_charge':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][0] for i in index.tolist()]
    elif mode == 'add_aromatic_PDB':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB[i][0] for i in index.tolist()]
    # elif mode == 'add_aromatic_full':
    #     atomic_number = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC_FULL[i][0] for i in index.tolist()]
    elif mode == 'full':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_FULL[i][0] for i in index.tolist()]
    else:
        raise ValueError
    return atomic_number


def is_aromatic_from_index(index, mode):
    if mode == 'add_aromatic' or mode == 'add_aromatic_plus_charge':
        is_aromatic = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][1] for i in index.tolist()]
    elif mode == 'add_aromatic_PDB':
        is_aromatic = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB[i][1] for i in index.tolist()]
    # elif mode == 'add_aromatic_full':
    #     is_aromatic = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC_FULL[i][1] for i in index.tolist()]
    elif mode == 'full':
        is_aromatic = [MAP_INDEX_TO_ATOM_TYPE_FULL[i][2] for i in index.tolist()]
    elif mode == 'basic' or mode == 'basic_plus_aromatic' or mode == 'basic_plus_full' or mode == 'basic_plus_charge' or mode == 'basic_PDB' or mode == 'basic_plus_charge_PDB':
        is_aromatic = None
    else:
        raise ValueError
    return is_aromatic


def get_hybridization_from_index(index, mode):
    if mode == 'full':
        hybridization = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][1] for i in index.tolist()]
    else:
        raise ValueError
    return hybridization


def get_index(atom_num, hybridization, is_aromatic, mode):
    if mode == 'basic' or mode == 'basic_plus_aromatic' or mode == 'basic_plus_full' or mode == 'basic_plus_charge':
        return MAP_ATOM_TYPE_ONLY_TO_INDEX[int(atom_num)]
    elif mode == 'basic_PDB' or mode == 'basic_plus_charge_PDB':
        return MAP_ATOM_TYPE_ONLY_TO_INDEX_PDB[int(atom_num)]
    elif mode == 'add_aromatic' or mode == 'add_aromatic_plus_charge':
        # self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17])  # H, C, N, O, F, P, S, Cl
        if (int(atom_num), bool(is_aromatic)) in MAP_ATOM_TYPE_AROMATIC_TO_INDEX:
            return MAP_ATOM_TYPE_AROMATIC_TO_INDEX[int(atom_num), bool(is_aromatic)]
        else:
            print(int(atom_num), bool(is_aromatic))
            return MAP_ATOM_TYPE_AROMATIC_TO_INDEX[(1, False)]
    elif mode == 'add_aromatic_PDB':
        # self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17, 35])  # H, C, N, O, F, P, S, Cl, Br
        if (int(atom_num), bool(is_aromatic)) in MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX:
            return MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX[int(atom_num), bool(is_aromatic)]
        else:
            print(int(atom_num), bool(is_aromatic), 'to', 35, False)
            raise ValueError
    # elif mode == 'add_aromatic_full':
        # return MAP_ATOM_TYPE_AROMATIC_FULL_TO_INDEX[int(atom_num), bool(is_aromatic)]
    elif mode == 'full':
        return MAP_ATOM_TYPE_FULL_TO_INDEX[(int(atom_num), str(hybridization), bool(is_aromatic))]
    else:
        raise NotImplementedError


class FeaturizeProteinAtom(object):

    def __init__(self):
        super().__init__()
        self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16, 34])  # H, C, N, O, S, Se
        self.max_num_aa = 20

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + self.max_num_aa + 1

    def __call__(self, data: ProteinLigandData):
        element = data.protein_element.view(-1, 1) == self.atomic_numbers.view(1, -1)  # (N_atoms, N_elements)
        amino_acid = F.one_hot(data.protein_atom_to_aa_type, num_classes=self.max_num_aa)
        is_backbone = data.protein_is_backbone.view(-1, 1).long()
        # TODO: is_backbone is 0/1 values, not sure the feature is treated as categorical, if so, change to 2-hot
        x = torch.cat([element, amino_acid, is_backbone], dim=-1)
        data.protein_atom_feature = x
        return data


class FeaturizeLigandAtom(object):

    def __init__(self, mode='basic'):
        super().__init__()
        assert mode in ['basic', 'basic_PDB', 'basic_plus_charge_PDB', 'basic_plus_charge', 'basic_plus_aromatic', 'basic_plus_full', 'add_aromatic', 'full', 'add_aromatic_full', 'add_aromatic_PDB', 'add_aromatic_plus_charge']
        self.mode = mode

    @property
    def feature_dim(self):
        if 'plus' not in self.mode:
            return self.type_feature_dim
        elif self.mode == 'basic_plus_aromatic':
            return self.type_feature_dim + self.aromatic_feature_dim
        elif self.mode == 'basic_plus_full':
            return self.type_feature_dim + self.aromatic_feature_dim + self.charge_feature_dim
        elif self.mode == 'basic_plus_charge_PDB' or self.mode == 'basic_plus_charge' or self.mode == 'add_aromatic_plus_charge':
            return self.type_feature_dim + self.charge_feature_dim
        else:
            raise NotImplementedError(self.mode)

    @property
    def type_feature_dim(self):
        if self.mode == 'basic' or self.mode == 'basic_plus_charge' or self.mode == 'basic_plus_aromatic' or self.mode == 'basic_plus_full':
            return len(MAP_ATOM_TYPE_ONLY_TO_INDEX)
        elif self.mode == 'basic_PDB' or self.mode == 'basic_plus_charge_PDB':
            return len(MAP_ATOM_TYPE_ONLY_TO_INDEX_PDB)
        elif self.mode == 'add_aromatic' or self.mode == 'add_aromatic_plus_charge':
            return len(MAP_ATOM_TYPE_AROMATIC_TO_INDEX)
        elif self.mode == 'add_aromatic_PDB':
            return len(MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX)
        # elif self.mode == 'add_aromatic_full':
        #     return len(MAP_ATOM_TYPE_AROMATIC_FULL_TO_INDEX)
        elif self.mode == 'full':
            return len(MAP_ATOM_TYPE_FULL_TO_INDEX)
        else:
            raise NotImplementedError

    @property
    def aromatic_feature_dim(self):
        if self.mode == 'basic_plus_aromatic' or self.mode == 'basic_plus_full':
            return 2
        return 0
    
    @property
    def hybridization_feature_dim(self):
        if self.mode == 'basic_plus_full':
            return len(utils_data.HYBRIDIZATION_TYPE)
        return 0

    @property
    def charge_feature_dim(self):
        if self.mode == 'basic_plus_charge_PDB' or self.mode == 'basic_plus_charge' or self.mode == 'add_aromatic_plus_charge':
            return 3  # 0, 1, -1
        return 0

    # adapted from https://github.com/Dunni3/FlowMol/blob/main/analysis/molecule_builder.py#L92
    def compute_valencies(self, data):
        """Compute the valencies of every atom in the molecule. Returns a tensor of shape (num_atoms,)."""
        assert hasattr(data, 'ligand_bond_index')
        n_atoms = data.ligand_pos.size(0)
        bond_matrix = torch.zeros(n_atoms, n_atoms).long()
        src, dst = data.ligand_bond_index
        bond_matrix[src, dst] = data.ligand_bond_type

        adj = torch.zeros((n_atoms, n_atoms))
        adjusted_bond_types = bond_matrix.clone()
        adjusted_bond_types[adjusted_bond_types == 4] = 1.5
        adjusted_bond_types = adjusted_bond_types.float()
        adj[src, dst] = adjusted_bond_types
        valencies = torch.sum(adj, dim=-1).long()
        return valencies

    def __call__(self, data: ProteinLigandData):
        element_list = data.ligand_element
        hybridization_list = data.ligand_hybridization
        if hasattr(data, 'ligand_charge'):
            charge_list = data.ligand_charge
        else:
            charge_list = torch.zeros(len(element_list))
        aromatic_list = [v[AROMATIC_FEAT_MAP_IDX] for v in data.ligand_atom_feature]

        x = [get_index(e, h, a, self.mode) for e, h, a in zip(element_list, hybridization_list, aromatic_list)]
        x = torch.tensor(x)
        # valencies = self.compute_valencies(data)
        # x_hybrid = torch.tensor([utils_data.HYBRIDIZATION_TYPE.index(h) for h in hybridization_list])
        x_charge = torch.tensor([c + 1 for c in charge_list])
        x_aromatic = torch.tensor(aromatic_list)
        data.ligand_atom_feature_full = torch.cat(
            [x.view(-1, 1), x_charge.view(-1, 1), x_aromatic.view(-1, 1)], dim=-1
        )
        return data


# class FeaturizeLigandBond(object):

#     def __init__(self):
#         super().__init__()

#     def __call__(self, data: ProteinLigandData):
#         print(data.ligand_bond_type)
#         print(data.ligand_element)
#         print(len(data.ligand_element))
#         data.ligand_bond_feature = F.one_hot(data.ligand_bond_type - 1, num_classes=len(utils_data.BOND_TYPES))
#         return data


class FeaturizeLigandBond(object):

    def __init__(self, mode='fc', set_bond_type=True):
        super().__init__()
        self.mode = mode
        self.set_bond_type = set_bond_type

    def __call__(self, data: ProteinLigandData):
        n_atoms = len(data.ligand_element)  # only ligand atom mask is reset in beta prior sampling
        full_dst = torch.repeat_interleave(torch.arange(n_atoms), n_atoms)
        full_src = torch.arange(n_atoms).repeat(n_atoms)
        mask = full_dst != full_src
        full_dst, full_src = full_dst[mask], full_src[mask]
        data.ligand_fc_bond_index = torch.stack([full_src, full_dst], dim=0)
        assert data.ligand_fc_bond_index.size(0) == 2


        if hasattr(data, 'ligand_bond_index') and self.set_bond_type:
            n_atoms = len(data.ligand_element)
            bond_matrix = torch.zeros(n_atoms, n_atoms).long()
            src, dst = data.ligand_bond_index
            bond_matrix[src, dst] = data.ligand_bond_type
            # assert data.ligand_bond_type.max() < 5, data.ligand_bond_type.max()
            if self.mode == 'divide':
                bond_matrix = (bond_matrix.float() / 2).ceil().long() # 0, 1, 2, 3, 4, 5 -> 0, 1, 1, 2, 2, 3
            data.ligand_fc_bond_type = bond_matrix[data.ligand_fc_bond_index[0], data.ligand_fc_bond_index[1]]
        return data


class RandomRotation(object):

    def __init__(self):
        super().__init__()

    def __call__(self,  data: ProteinLigandData):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        Q = torch.from_numpy(Q.astype(np.float32))
        data.ligand_pos = data.ligand_pos @ Q
        data.protein_pos = data.protein_pos @ Q
        return data
