import collections
from typing import Tuple, Sequence, Dict, Optional

import numpy as np
from scipy import spatial as sci_spatial
import matplotlib.pyplot as plt

from core.evaluation.utils.eval_bond_length import _bond_str, _atom_str
import core.datasets.utils as utils_data

from rdkit import Chem
from rdkit.Chem import rdMolTransforms

# BondType = Tuple[int, int, int]  # (atomic_num, atomic_num, bond_type)
BondAngleType = Tuple[int, int, int, int, int]
BondAngleData = Tuple[BondAngleType, float]  # (angle_type, bond_angle)
BondAngleProfile = Dict[BondAngleType, np.ndarray]  # angle_type -> empirical distribution


BOND_ANGLE_TYPES = frozenset((
    (6, 1, 6, 1, 6), #	CCC	521	18.1%	59.52230720788234	135.50315793532704	0.759808507274447
    (6, 4, 6, 4, 6), #	C:C:C	460	16.0%	101.54806405949785	127.54928623790771	0.2600122217840986
    (6, 1, 6, 1, 8), #	CCO	274	9.5%	57.19735111082594	136.5409407542893	0.7934358964346336
))

BOND_ANGLE_BINS = np.arange(100, 140.01, 0.25)

def _format_angle_type(angle_type: BondAngleType) -> BondAngleType:
    atom1, bond1_type, atom2, bond2_type, atom3 = angle_type
    if atom1 > atom3:
        atom1, atom3 = atom3, atom1
        bond1_type, bond2_type = bond2_type, bond1_type
    return atom1, bond1_type, atom2, bond2_type, atom3


def _angle_type_str(angle_type: BondAngleType) -> str:
    atom1, bond1_type, atom2, bond2_type, atom3 = angle_type
    atom1, atom2, atom3 = _atom_str(atom1), _atom_str(atom2), _atom_str(atom3)
    bond1_str = _bond_str(bond1_type)
    bond2_str = _bond_str(bond2_type)
    return f'{atom1}{bond1_str}{atom2}{bond2_str}{atom3}'


def bond_angle_from_mol(mol):
    # collect all bond angles from rdkit mol
    bond_angles = []
    for bond1 in mol.GetBonds():
        atom1 = bond1.GetBeginAtom()
        atom2 = bond1.GetEndAtom()
        for bond2 in atom2.GetBonds():
            atom3 = bond2.GetOtherAtom(atom2)
            if atom3.GetIdx() == atom1.GetIdx():
                continue

            try:
                angle = rdMolTransforms.GetAngleDeg(mol.GetConformer(), 
                    atom1.GetIdx(), atom2.GetIdx(), atom3.GetIdx())
                atom1_num = atom1.GetAtomicNum()
                atom2_num = atom2.GetAtomicNum()
                atom3_num = atom3.GetAtomicNum()
                bond1_type = utils_data.BOND_TYPES[bond1.GetBondType()]
                bond2_type = utils_data.BOND_TYPES[bond2.GetBondType()]

                angle_type = (atom1_num, bond1_type, atom2_num, bond2_type, atom3_num)
                angle_type = _format_angle_type(angle_type)
                bond_angles.append((angle_type, angle))
            except ValueError as e:
                print(e)
                print('Error in bond angle calculation')
    return bond_angles


def get_distribution(angles: Sequence[float], bins=BOND_ANGLE_BINS) -> np.ndarray:
    return np.histogram(angles, bins=bins, density=True)[0]


def get_bond_angles(bond_angles: Sequence[BondAngleData]) -> BondAngleProfile:
    bond_angle_profile = collections.defaultdict(list)
    for angle_type, angle in bond_angles:
        angle_type = _format_angle_type(angle_type)
        bond_angle_profile[angle_type].append(angle)
    return bond_angle_profile


def get_bond_angle_profile(bond_angles: Sequence[BondAngleData]) -> BondAngleProfile:
    bond_angle_profile = get_bond_angles(bond_angles)
    bond_angle_distribution = {k: get_distribution(v) 
        for k, v in bond_angle_profile.items() if k in BOND_ANGLE_TYPES}
    return bond_angle_distribution


def eval_bond_angle_profile(ref_bond_angle_profile: BondAngleProfile, 
                            bond_angle_profile: BondAngleProfile) -> Dict[str, Optional[float]]:
    metrics = {}

    # Jensen-Shannon distances
    for angle_type in BOND_ANGLE_TYPES:
        if angle_type in bond_angle_profile:
            metrics[f'JSD_{_angle_type_str(angle_type)}'] = sci_spatial.distance.jensenshannon(
                ref_bond_angle_profile[angle_type], bond_angle_profile[angle_type])

    return metrics


