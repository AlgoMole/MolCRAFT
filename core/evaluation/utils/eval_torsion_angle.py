import collections
from typing import Tuple, Sequence, Dict, Optional

import numpy as np
from scipy import spatial as sci_spatial
import matplotlib.pyplot as plt

from core.evaluation.utils.eval_bond_length import BondType, _format_bond_type
import core.datasets.utils as utils_data
from core.evaluation.utils.eval_bond_length import _bond_str, _atom_str

from rdkit import Chem
from rdkit.Chem import rdMolTransforms

# BondType = Tuple[int, int, int]  # (atomic_num, atomic_num, bond_type)
# AngleType = Tuple[int, int, int, int, int]
TorsionType = Tuple[int, int, int, int, int, int, int]
TorsionData = Tuple[TorsionType, float]  # (torsion_type, torsion_angle)
TorsionProfile = Dict[TorsionType, np.ndarray]  # torsion_type -> empirical distribution


TORSION_ANGLE_TYPES = frozenset((
    (6, 1, 6, 1, 6, 1, 6), #	CCCC	486	13.7%	-179.92344102773612	179.9786281079837	3.599020691357198
    (6, 4, 6, 4, 6, 4, 6), #	C:C:C:C	443	12.5%	-179.97199569192466	179.98934915798304	3.599613448499077
    # (6, 1, 6, 1, 8, 1, 6), #	CCOC	189	5.3%	-177.62581170798194	179.81337162625871	3.574391833342406
    # (6, 1, 6, 1, 6, 1, 8), #	CCCO	177	5.0%	-179.67547425994576	179.97332118060115	3.596487954405469
    # (6, 1, 6, 1, 7, 1, 6), #	CCNC	167	4.7%	-179.97370952625153	179.87219385011082	3.5984590337636235
    # (6, 4, 6, 4, 7, 4, 6), #	C:C:N:C	166	4.7%	-179.91810961554717	179.97902253499925	3.5989713215054646
    # (6, 4, 6, 4, 6, 4, 7), #	C:C:C:N	111	3.1%	-179.9906883334351	179.9979409714953	3.5998862930493036
    # (6, 4, 7, 4, 6, 4, 7), #	C:N:C:N	103	2.9%	-179.99133735677788	179.98458043739564	3.599759177941735
    # (6, 1, 6, 1, 6, 1, 7), #	CCCN	102	2.9%	-179.807320362162	179.86965908118225	3.5967697944334422
))

TORSION_ANGLE_BINS = np.arange(-180, 181, 3)


def _format_torsion_type(torsion_type: TorsionType) -> TorsionType:
    atom1, bond1_type, atom2, bond2_type, atom3, bond3_type, atom4 = torsion_type
    reverse = False
    if atom1 > atom4:
        atom1, atom2, atom3, atom4 = atom4, atom3, atom2, atom1
        bond1_type, bond3_type = bond3_type, bond1_type
        reverse = True
    elif atom1 == atom4:
        if atom2 > atom3:
            atom1, atom2, atom3, atom4 = atom4, atom3, atom2, atom1
            bond1_type, bond3_type = bond3_type, bond1_type
            reverse = True
    return (atom1, bond1_type, atom2, bond2_type, atom3, bond3_type, atom4), reverse


def _torsion_type_str(torsion_type: TorsionType) -> str:
    assert len(torsion_type) == 7
    atom1, bond1_type, atom2, bond2_type, atom3, bond3_type, atom4 = torsion_type
    bond1_str = _bond_str(bond1_type)
    bond2_str = _bond_str(bond2_type)
    bond3_str = _bond_str(bond3_type)
    return f'{_atom_str(atom1)}{bond1_str}{_atom_str(atom2)}{bond2_str}{_atom_str(atom3)}{bond3_str}{_atom_str(atom4)}'


def torsion_angle_from_mol(mol):
    # collect all torsion angles from rdkit mol
    torsion_angles = []
    for bond1 in mol.GetBonds():
        atom1 = bond1.GetBeginAtom()
        atom2 = bond1.GetEndAtom()
        for bond2 in atom2.GetBonds():
            atom3 = bond2.GetOtherAtom(atom2)
            if atom3.GetIdx() == atom1.GetIdx():
                continue
            for bond3 in atom3.GetBonds():
                atom4 = bond3.GetOtherAtom(atom3)
                if atom4.GetIdx() == atom2.GetIdx():
                    continue

                try:
                    angle = rdMolTransforms.GetDihedralDeg(mol.GetConformer(), 
                        atom1.GetIdx(), atom2.GetIdx(), atom3.GetIdx(), atom4.GetIdx())
                    atom1_num = atom1.GetAtomicNum()
                    atom2_num = atom2.GetAtomicNum()
                    atom3_num = atom3.GetAtomicNum()
                    atom4_num = atom4.GetAtomicNum()
                    bond1_type = utils_data.BOND_TYPES[bond1.GetBondType()]
                    bond2_type = utils_data.BOND_TYPES[bond2.GetBondType()]
                    bond3_type = utils_data.BOND_TYPES[bond3.GetBondType()]

                    torsion_type = (atom1_num, bond1_type, atom2_num, bond2_type, atom3_num, bond3_type, atom4_num)
                    torsion_type, reverse = _format_torsion_type(torsion_type)
                    if reverse:
                        angle = -angle

                    torsion_angles.append((
                        (atom1_num, bond1_type, atom2_num, bond2_type, atom3_num, bond3_type, atom4_num), angle))
                except ValueError as e:
                    print(e)
                    print('Error in bond angle calculation')
    return torsion_angles


def get_torsion_angles(torsion_angles: Sequence[TorsionData]) -> TorsionProfile:
    torsion_angle_profile = collections.defaultdict(list)
    for torsion_type, angle in torsion_angles:
        torsion_type, _ = _format_torsion_type(torsion_type)
        torsion_angle_profile[torsion_type].append(angle)
    return torsion_angle_profile
    

def get_distribution(torsion_angles: Sequence[float], bins: Sequence[float]=TORSION_ANGLE_BINS) -> np.ndarray:
    return np.histogram(torsion_angles, bins=bins, density=True)[0]


def get_torsion_angle_profile(torsion_angle: Sequence[TorsionData]) -> TorsionProfile:
    torsion_angle_profile = get_torsion_angles(torsion_angle)
    torsion_angle_distribution = {k: get_distribution(v) 
        for k, v in torsion_angle_profile.items() if k in TORSION_ANGLE_TYPES}
    return torsion_angle_distribution


def eval_torsion_angle_profile(ref_torsion_angle_profile: TorsionProfile, torsion_angle_profile: TorsionProfile) -> Dict[str, Optional[float]]:
    metrics = {}

    # Jensen-Shannon distances
    for torsion_type in TORSION_ANGLE_TYPES:
        if torsion_type in torsion_angle_profile:
            ref_profile = ref_torsion_angle_profile[torsion_type]
            profile = torsion_angle_profile[torsion_type]
            metrics[f'JSD_{_torsion_type_str(torsion_type)}'] = sci_spatial.distance.jensenshannon(ref_profile, profile)
        
    return metrics
