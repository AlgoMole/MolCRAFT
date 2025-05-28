import torch
import torch_scatter
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

FOLLOW_BATCH = ('protein_element', 'ligand_element', 'ligand_fc_bond_type',)

common_keys = [
                'protein_pos', 'protein_atom_feature', 'protein_element', 'protein_atom_to_aa_type', 'protein_is_backbone',
                'ligand_pos', 'ligand_atom_feature_full', 'ligand_element', 'ligand_hybridization', 'ligand_atom_feature',
                'protein_filename', 'ligand_filename', 'ligand_smiles', 'affinity', 'ligand_charge',
                'ligand_fc_bond_index', 'ligand_fc_bond_type', 'ligand_bond_index', 'ligand_bond_type',
            ]

class ProteinLigandData(Data):

    def __init__(self, *args, **kwargs):
        new_kwargs = {}
        for key, value in kwargs.items():
            if key in common_keys:
                new_kwargs[key] = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
        kwargs = new_kwargs
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, **kwargs):
        instance = ProteinLigandData(**kwargs)

        if protein_dict is not None:
            for key, item in protein_dict.items():
                instance['protein_' + key] = item

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance['ligand_' + key] = item

        instance['ligand_nbh_list'] = {i.item(): [j.item() for k, j in enumerate(instance.ligand_bond_index[1])
                                                  if instance.ligand_bond_index[0, k].item() == i]
                                       for i in instance.ligand_bond_index[0]}
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'ligand_bond_index' or key == 'ligand_fc_bond_index':
            return self['ligand_element'].size(0)
        else:
            return super().__inc__(key, value)


class ProteinLigandDataLoader(DataLoader):

    def __init__(
            self,
            dataset,
            batch_size=1,
            shuffle=False,
            follow_batch=FOLLOW_BATCH,
            **kwargs
    ):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, follow_batch=follow_batch, **kwargs)


def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output


def get_batch_connectivity_matrix(ligand_batch, ligand_bond_index, ligand_bond_type, ligand_bond_batch):
    batch_ligand_size = torch_scatter.segment_coo(
        torch.ones_like(ligand_batch),
        ligand_batch,
        reduce='sum',
    )
    batch_index_offset = torch.cumsum(batch_ligand_size, 0) - batch_ligand_size
    batch_size = len(batch_index_offset)
    batch_connectivity_matrix = []
    for batch_index in range(batch_size):
        start_index, end_index = ligand_bond_index[:, ligand_bond_batch == batch_index]
        start_index -= batch_index_offset[batch_index]
        end_index -= batch_index_offset[batch_index]
        bond_type = ligand_bond_type[ligand_bond_batch == batch_index]
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.
        connectivity_matrix = torch.zeros(batch_ligand_size[batch_index], batch_ligand_size[batch_index],
                                          dtype=torch.int)
        for s, e, t in zip(start_index, end_index, bond_type):
            connectivity_matrix[s, e] = connectivity_matrix[e, s] = t
        batch_connectivity_matrix.append(connectivity_matrix)
    return batch_connectivity_matrix

def get_batch_type_pmf_matrix(ligand_batch, ligand_bond_index, ligand_bond_type_pmf, ligand_bond_batch, padding=False, num_atoms_max=65):
    batch_ligand_size = torch_scatter.segment_coo(
        torch.ones_like(ligand_batch),
        ligand_batch,
        reduce='sum',
    )
    batch_index_offset = torch.cumsum(batch_ligand_size, 0) - batch_ligand_size
    batch_size = len(batch_index_offset)
    batch_connectivity_matrix = []
    E = ligand_bond_type_pmf.size(-1)
    max_N = num_atoms_max  # Maximum number of atoms in a ligand (magic number for crossdock)
    for batch_index in range(batch_size):
        start_index, end_index = ligand_bond_index[:, ligand_bond_batch == batch_index]
        start_index -= batch_index_offset[batch_index]
        end_index -= batch_index_offset[batch_index]
        bond_type_pmf = ligand_bond_type_pmf[ligand_bond_batch == batch_index]
        # NxNxE connectivity matrix where each bond_type_pmf represents a vector of float densities over bond type (0: none, 1: single, 2: double, 3: triple, 4: aromatic).
        if padding:
            connectivity_matrix = torch.zeros(
               (batch_ligand_size[batch_index], max_N, E), dtype=torch.float 
            )
        else:
            connectivity_matrix = torch.zeros(
                (batch_ligand_size[batch_index], batch_ligand_size[batch_index], E), dtype=torch.float
            )
        for s, e, t in zip(start_index, end_index, bond_type_pmf):
            connectivity_matrix[s, e] = connectivity_matrix[e, s] = t
        batch_connectivity_matrix.append(connectivity_matrix)
    return batch_connectivity_matrix
