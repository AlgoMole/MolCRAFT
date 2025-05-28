from rdkit import Chem
from rdkit.Chem.QED import qed

import torch
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
import pickle

from core.datasets.pl_data import ProteinLigandData
from core.datasets import utils as utils_data
from core.evaluation.utils.sascorer import compute_sa_score
from core.evaluation.utils.scoring_func import obey_lipinski

AROMATIC_FEAT_MAP_IDX = utils_data.ATOM_FAMILIES_ID['Aromatic']

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
    1: 0,
    6: 1,
    7: 2,
    8: 3,
    9: 4,
    15: 5,
    16: 6,
    17: 7,
    35: 8,
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
MAP_INDEX_TO_ATOM_TYPE_AROMATIC = {v: k for k, v in MAP_ATOM_TYPE_AROMATIC_TO_INDEX.items()}
MAP_INDEX_TO_ATOM_TYPE_FULL = {v: k for k, v in MAP_ATOM_TYPE_FULL_TO_INDEX.items()}
MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB = {v: k for k, v in MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX.items()}
# MAP_INDEX_TO_ATOM_TYPE_AROMATIC_FULL = {v: k for k, v in MAP_ATOM_TYPE_AROMATIC_FULL_TO_INDEX.items()}

def get_atomic_number_from_index(index, mode):
    if mode == 'basic':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_ONLY[i] for i in index.tolist()]
    elif mode == 'add_aromatic':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][0] for i in index.tolist()]
    elif mode == 'add_aromatic_pdb':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB[i][0] for i in index.tolist()]
    # elif mode == 'add_aromatic_full':
    #     atomic_number = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC_FULL[i][0] for i in index.tolist()]
    elif mode == 'full':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_FULL[i][0] for i in index.tolist()]
    else:
        raise ValueError
    return atomic_number


def is_aromatic_from_index(index, mode):
    if mode == 'add_aromatic':
        is_aromatic = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][1] for i in index.tolist()]
    elif mode == 'add_aromatic_pdb':
        is_aromatic = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC_PDB[i][1] for i in index.tolist()]
    # elif mode == 'add_aromatic_full':
    #     is_aromatic = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC_FULL[i][1] for i in index.tolist()]
    elif mode == 'full':
        is_aromatic = [MAP_INDEX_TO_ATOM_TYPE_FULL[i][2] for i in index.tolist()]
    elif mode == 'basic':
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
    if mode == 'basic':
        return MAP_ATOM_TYPE_ONLY_TO_INDEX[int(atom_num)]
    elif mode == 'add_aromatic':
        # self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17])  # H, C, N, O, F, P, S, Cl
        if (int(atom_num), bool(is_aromatic)) in MAP_ATOM_TYPE_AROMATIC_TO_INDEX:
            return MAP_ATOM_TYPE_AROMATIC_TO_INDEX[int(atom_num), bool(is_aromatic)]
        else:
            print(int(atom_num), bool(is_aromatic))
            return MAP_ATOM_TYPE_AROMATIC_TO_INDEX[(1, False)]
    elif mode == 'add_aromatic_pdb':
        # self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17, 35])  # H, C, N, O, F, P, S, Cl, Br
        if (int(atom_num), bool(is_aromatic)) in MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX:
            return MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX[int(atom_num), bool(is_aromatic)]
        else:
            print(int(atom_num), bool(is_aromatic), 'to', 35, False)
            return MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX[(35, False)]
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
        assert mode in ['basic', 'add_aromatic', 'full', 'add_aromatic_full', 'add_aromatic_pdb']
        self.mode = mode

    @property
    def feature_dim(self):
        if self.mode == 'basic':
            return len(MAP_ATOM_TYPE_ONLY_TO_INDEX)
        elif self.mode == 'add_aromatic':
            return len(MAP_ATOM_TYPE_AROMATIC_TO_INDEX)
        elif self.mode == 'add_aromatic_pdb':
            return len(MAP_ATOM_TYPE_AROMATIC_PDB_TO_INDEX)
        # elif self.mode == 'add_aromatic_full':
        #     return len(MAP_ATOM_TYPE_AROMATIC_FULL_TO_INDEX)
        elif self.mode == 'full':
            return len(MAP_ATOM_TYPE_FULL_TO_INDEX)
        else:
            raise NotImplementedError

    def __call__(self, data: ProteinLigandData):
        element_list = data.ligand_element
        hybridization_list = data.ligand_hybridization
        aromatic_list = [v[AROMATIC_FEAT_MAP_IDX] for v in data.ligand_atom_feature]

        x = [get_index(e, h, a, self.mode) for e, h, a in zip(element_list, hybridization_list, aromatic_list)]
        x = torch.tensor(x)
        data.ligand_atom_feature_full = x
        return data


class FeaturizeLigandBond(object):

    def __init__(self):
        super().__init__()

    def __call__(self, data: ProteinLigandData):
        data.ligand_bond_feature = F.one_hot(data.ligand_bond_type - 1, num_classes=len(utils_data.BOND_TYPES))
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


class NormalizeVina(object):
    def __init__(self, mode='pl'):
        super().__init__()
        self.mode = mode
        if 'pl' in mode:
            self.max_v = 0
            self.min_v = -16
        elif mode == 'pdbbind':
            self.max_v = 16
            self.min_v = 0
        else:
            raise ValueError
    
    def _trans(self, vina_score):
        if 'pl' in self.mode:
            return (self.max_v - np.clip(vina_score, self.min_v, self.max_v)) / (self.max_v - self.min_v)
        elif self.mode == 'pdbbind':
            return np.clip(vina_score, self.min_v, self.max_v) / (self.max_v - self.min_v)
        else:
            raise ValueError

    def __call__(self,  data: ProteinLigandData):
        data.affinity = self._trans(data.affinity)
        return data

class AddMolProp(object):

    def __init__(self):
        super().__init__()
        self.max_sa = 1.0
        self.min_sa = 0.17
        self.max_qed = 0.95
        self.min_qed = 0.01

    def __call__(self, data: ProteinLigandData):
        smi = data.ligand_smiles
        mol = Chem.MolFromSmiles(smi)
        data.qed = qed(mol)
        data.qed_norm = (qed(mol) - self.min_qed) / (self.max_qed - self.min_qed)
        data.sa = compute_sa_score(mol)
        data.sa_norm = (compute_sa_score(mol) - self.min_sa) / (self.max_sa - self.min_sa)
        data.lipinski = obey_lipinski(mol)
        data.lipinski_norm = data.lipinski / 5
        return data

class LoadInteraction(object):

    def __init__(self, interaction_path, interaction_types):
        super().__init__()
        self.interaction_path = os.path.join(os.path.dirname(interaction_path), 'in_cross')
        self.interaction_types = interaction_types


    def __call__(self, data: ProteinLigandData):
        protein_filename = data.protein_filename
        interaction_path = os.path.join(self.interaction_path, protein_filename.split('/')[-1].replace('.pdb', '_pv_interactions.csv'))
        if not os.path.exists(interaction_path):
            print(f'No interaction file found for {protein_filename}')
            int_counts = {}
        else:
            interactions = pd.read_csv(interaction_path)
            int_counts = interactions['Type'].value_counts()
        # Counter({'HPhob': 4842452, 'PiEdge': 63919, 'PiFace': 41029, 'PiCat': 20951, 'HAccep nn': 15272, 'HAccep cn': 4871, 'XBond': 3197, 'HDonor nn': 2590, 'HDonor cn': 938, 'Salt': 373, 'HAccep nc': 20, 'HAccep cc': 3})
        # Change HAccep cn to HAccep etc

        for int_type in self.interaction_types:
            if int_type == 'HAccep':
                count = int_counts.get('HAccep nn', 0) + int_counts.get('HAccep cn', 0) + int_counts.get('HAccep nc', 0) + int_counts.get('HAccep cc', 0)
            elif int_type == 'HDonor':
                count = int_counts.get('HDonor nn', 0) + int_counts.get('HDonor cn', 0)
            elif int_type == 'Pi':
                count = int_counts.get('PiEdge', 0) + int_counts.get('PiFace', 0) + int_counts.get('PiCat', 0)
            else:
                count = int_counts.get(int_type, 0)
            count = int(count)
            assert count >= 0 and type(count) == int, f'Error with {int_type}: {count}, {type(count)}'
            setattr(data, f'{int_type}', int(count))
        return data


class AddScaffoldMask(object):

    def __init__(self, mask_path, change_scaffold):
        super().__init__()
        self.mask_path = mask_path
        with open(mask_path, 'rb') as f:
            self.scaffold_mask = pickle.load(f)
        self.name2mask = {
            item[0]: item[1] for item in self.scaffold_mask
        }
        self.change_scaffold = change_scaffold

    def __call__(self, data: ProteinLigandData):
        ligand_filename = data.ligand_filename
        scaffold_mask = self.name2mask.get(ligand_filename, None)
        if scaffold_mask is None:
            print(f'No scaffold mask found for {ligand_filename}, using zeros shaped {data.ligand_element.shape}')
            scaffold_mask = torch.zeros_like(data.ligand_element, dtype=torch.bool)
        if self.change_scaffold:
            scaffold_mask = ~scaffold_mask
        data.ligand_mask = scaffold_mask
        return data

