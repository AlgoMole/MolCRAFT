from rdkit import Chem
import numpy as np
import torch
import pickle as pkl
import os
import core.evaluation.utils.bond_analyze as bond_analyze
from torch_geometric.data import Data
from scipy.spatial.distance import cdist
import torch
import glob
import random
import contextlib

import sys, io
from absl import logging
import time
import tempfile


@contextlib.contextmanager
def timing(msg: str):
    print("Started %s", msg)
    tic = time.time()
    yield
    toc = time.time()
    print("Finished %s in %.3f seconds", msg, toc - tic)


@contextlib.contextmanager
def supress_stdout():
    in_memory_file = tempfile.SpooledTemporaryFile()
    # suppress stdout
    orig_stdout_fno = os.dup(sys.stdout.fileno())
    os.dup2(in_memory_file.fileno(), 1)
    # suppress stderr
    orig_stderr_fno = os.dup(sys.stderr.fileno())
    os.dup2(in_memory_file.fileno(), 2)
    try:
        yield
    finally:
        os.fsync(in_memory_file)
        os.dup2(orig_stdout_fno, 1)  # restore stdout
        os.dup2(orig_stderr_fno, 2)  # restore stderr
        in_memory_file.seek(0)
        outputs = in_memory_file.read().decode("utf-8").strip()
        if outputs:
            logging.info(outputs)
        in_memory_file.close()


"""
Atom cloud adding bonds and to graphs/smiles
"""

bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def convert_atomcloud_to_mol_smiles(
    positions, atom_type, atom_decoder, type_one_hot=True, single_bond=False
):
    """
    Convert an atom cloud to a SMILES string
    :param positions: (n_atoms, 3) tensor
    :par
    :return:
    """
    assert len(positions.shape) == 2
    positions = positions

    # TODO: compatibility check
    if type_one_hot:
        assert len(atom_type.shape) == 2
        assert atom_type.shape[1] == len(atom_decoder), atom_type.shape
        atom_type = torch.argmax(atom_type, dim=1)
    else:
        assert len(atom_type.shape) == 1
        # TODO: make atom_decoder list-like so as to avoid conversion
        atom_type = atom_type.detach().cpu().numpy()
    assert len(positions.shape) == 2

    mol = build_molecule(
        positions,
        atom_type,
        atom_decoder=atom_decoder,
        single_bond=single_bond,
    )
    smiles = mol2smiles(mol)
    return mol, smiles


def mol2smiles(mol):
    with supress_stdout():  # TODO: check
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return Chem.MolToSmiles(mol)


def build_molecule(
    positions, atom_type, atom_decoder, type_is_one_hot=False, single_bond=False
):
    X, A, E = build_xae_molecule(positions, atom_type, atom_decoder, single_bond)
    mol = Chem.RWMol()
    for atom in X:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(
            bond[0].item(), bond[1].item(), bond_dict[E[bond[0], bond[1]].item()]
        )
    return mol


def build_xae_molecule(positions, atom_type, atom_decoder, single_bond=False):
    """Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
    args:
    positions: N x 3  (already masked to keep final number nodes)
    atom_types: N
    returns:
    X: N         (int)
    A: N x N     (bool)                  (binary adjacency matrix)
    E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
    """
    assert len(positions.shape) == 2
    assert len(atom_type.shape) == 1
    n = positions.shape[0]
    X = atom_type
    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)
    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    for i in range(n):
        for j in range(i):
            try:
                pair = sorted([atom_type[i], atom_type[j]])
            except:
                print(atom_type)
                raise ValueError("atom_type error")
            order = bond_analyze.get_bond_order(
                atom_decoder[pair[0]],
                atom_decoder[pair[1]],
                dists[i, j],
                single_bond=single_bond,
                # check_exists=True
            )
            if order > 0:
                # Warning: the graph should be DIRECTED
                A[i, j] = 1
                E[i, j] = order
    return X, A, E


def check_stability(
    positions,
    atom_type,
    # atom_decoder,
    # type_one_hot=True,
    single_bond=False,
    debug=False,
    with_h=False,
):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    # TODO: TargetDiff
    # if type_one_hot:
    #     assert len(atom_type.shape) == 2, atom_type.shape
    #     assert atom_type.shape[1] == len(atom_decoder), atom_type.shape
    #     atom_type = torch.argmax(atom_type, dim=1)
    # else:
    #     assert len(atom_type.shape) == 1, atom_type.shape
    #     # TODO: make atom_decoder list-like so as to avoid conversion
    #     atom_type = atom_type.detach().cpu().numpy()

    positions = torch.Tensor(positions)
    distances = torch.cdist(positions, positions, p=2)
    ptable = Chem.GetPeriodicTable()
    atom_type = [ptable.GetElementSymbol(int(t)) for t in atom_type]

    # from scipy.spatial.distance import pdist
    # distances = pdist(positions, metric='euclidean')

    num_atoms = positions.shape[0]

    nr_bonds = np.zeros(num_atoms, dtype="int")

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = distances[i, j]
            # atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[atom_type[j]]
            atom1, atom2 = atom_type[i], atom_type[j]
            order = bond_analyze.get_bond_order(
                atom1, atom2, dist, single_bond=single_bond
            )
            nr_bonds[i] += order
            nr_bonds[j] += order
    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        possible_bonds = bond_analyze.allowed_bonds[atom_type_i]
        # hydrogen added, no more implicit bonds
        if with_h:
            if type(possible_bonds) == int:
                is_stable = possible_bonds == nr_bonds_i
            else:
                is_stable = nr_bonds_i in possible_bonds
        # implicit bonds exist, stable means <= max bonds
        else:
            if type(possible_bonds) != int:
                possible_bonds = max(possible_bonds)
            is_stable = (possible_bonds >= nr_bonds_i > 0)
        if not is_stable and debug:
            print(
                "Invalid bonds for molecule %s with %d bonds"
                % (atom_type_i, nr_bonds_i)
            )
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == num_atoms
    return molecule_stable, nr_stable_bonds, num_atoms


"""
save molecule and load molecules
"""


# TODO save sdf
def save_mol_list(
    path, molecule_list, index2atom, name="molecule", type_one_hot=True,  # self.cfg.dataset.atom_decoder
):
    # note t
    try:
        os.makedirs(path)
    except OSError:
        pass
    # molecule_list is a list of torch_geometry data.
    for id_ in range(len(molecule_list)):
        f = open(path + "/" + name + "_" + "%03d.txt" % (id_), "w")
        f.write(
            "%d\n\n" % (len(molecule_list[id_]["x"]))
        )  # write the number of atoms in the very top.
        if type_one_hot:
            atoms = torch.argmax(molecule_list[id_]["x"], dim=1)  # get atom type
        else:
            atoms = molecule_list[id_]["x"]
        n_atoms = int(molecule_list[id_]["x"].shape[0])
        pos_center = molecule_list[id_]["pos"].mean(dim=0)
        for atom_i in range(n_atoms):
            atom = atoms[atom_i]
            atom = index2atom[atom]
            f.write(
                "%s %.9f %.9f %.9f\n"
                % (
                    atom,
                    molecule_list[id_]["pos"][atom_i, 0] - pos_center[0],
                    molecule_list[id_]["pos"][atom_i, 1] - pos_center[1],
                    molecule_list[id_]["pos"][atom_i, 2] - pos_center[2],
                )
            )
        f.close()


def load_mol_file(file, index2atom):
    index2atom = {atom: index for index, atom in enumerate(index2atom)}
    # note t
    with open(file, encoding="utf8") as f:
        n_atoms = int(f.readline())
        one_hot = torch.zeros(n_atoms, len(index2atom))
        # charges = torch.zeros(n_atoms, 1)
        positions = torch.zeros(n_atoms, 3)
        f.readline()
        atoms = f.readlines()
        for i in range(n_atoms):
            atom = atoms[i].split(" ")
            atom_type = atom[0]
            one_hot[i, index2atom[atom_type]] = 1
            position = torch.Tensor([float(e) for e in atom[1:]])
            positions[i, :] = position
        return positions, one_hot


def load_xyz_files(path, shuffle=True):
    files = glob.glob(path + "/*.txt")
    if shuffle:
        random.shuffle(files)
    return files
