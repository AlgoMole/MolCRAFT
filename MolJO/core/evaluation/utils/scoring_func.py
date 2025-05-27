from collections import Counter
from copy import deepcopy

import numpy as np
from scipy.spatial.distance import cosine as cos_distance

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
from rdkit.Chem.FilterCatalog import *
from rdkit.Chem.QED import qed
from rdkit.Chem.Scaffolds import MurckoScaffold


from tqdm import tqdm
from core.evaluation.utils.sascorer import compute_sa_score


def is_pains(mol):
    params_pain = FilterCatalogParams()
    params_pain.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    catalog_pain = FilterCatalog(params_pain)
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    entry = catalog_pain.GetFirstMatch(mol)
    if entry is None:
        return False
    else:
        return True


def obey_lipinski(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    logp = get_logp(mol)
    rule_4 = (logp >= -2) & (logp <= 5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])


def get_basic(mol):
    n_atoms = len(mol.GetAtoms())
    n_bonds = len(mol.GetBonds())
    n_rings = len(Chem.GetSymmSSSR(mol))
    weight = Descriptors.ExactMolWt(mol)
    return n_atoms, n_bonds, n_rings, weight


def get_rdkit_rmsd(mol, n_conf=20, random_seed=42):
    """
    Calculate the alignment of generated mol and rdkit predicted mol
    Return the rmsd (max, min, median) of the `n_conf` rdkit conformers
    """
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    mol3d = Chem.AddHs(mol)
    rmsd_list = []
    # predict 3d
    try:
        confIds = AllChem.EmbedMultipleConfs(mol3d, n_conf, randomSeed=random_seed)
        for confId in confIds:
            AllChem.UFFOptimizeMolecule(mol3d, confId=confId)
            rmsd = Chem.rdMolAlign.GetBestRMS(mol, mol3d, refId=confId)
            rmsd_list.append(rmsd)
        # mol3d = Chem.RemoveHs(mol3d)
        rmsd_list = np.array(rmsd_list)
        return [np.max(rmsd_list), np.min(rmsd_list), np.median(rmsd_list)]
    except:
        return [np.nan, np.nan, np.nan]


def get_logp(mol):
    return Crippen.MolLogP(mol)


def get_chem(mol):
    qed_score = qed(mol)
    sa_score = compute_sa_score(mol)
    # TODO ring_size will cause error below, fix later
    # File "/sharefs/kevin/project/BFN4SBDD/core/callbacks/validation_callback.py", line 90, in on_validation_epoch_end
    #     out_metrics = self.metric.evaluate(self.outputs)
    # File "/sharefs/kevin/project/BFN4SBDD/core/evaluation/metrics.py", line 197, in evaluate
    #     metrics[k2 + '_mean'] = np.mean(k_list)
    # File "/opt/conda/lib/python3.9/site-packages/numpy/core/fromnumeric.py", line 3504, in mean
    #     return _methods._mean(a, axis=axis, dtype=dtype,
    # File "/opt/conda/lib/python3.9/site-packages/numpy/core/_methods.py", line 131, in _mean
    #     ret = ret / rcount
    # TypeError: unsupported operand type(s) for /: 'Counter' and 'int'

    # logp_score = get_logp(mol)
    # lipinski_score = obey_lipinski(mol)
    # ring_info = mol.GetRingInfo()
    # ring_size = Counter([len(r) for r in ring_info.AtomRings()])
    # hacc_score = Lipinski.NumHAcceptors(mol)
    # hdon_score = Lipinski.NumHDonors(mol)

    return {
        'qed': qed_score,
        'sa': sa_score,
        # 'logp': logp_score,
        # 'lipinski': lipinski_score,
        # 'ring_size': ring_size
    }


def get_molecule_force_field(mol, conf_id=None, force_field='mmff', **kwargs):
    """
    Get a force field for a molecule.
    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    conf_id : int, optional
        ID of the conformer to associate with the force field.
    force_field : str, optional
        Force Field name.
    kwargs : dict, optional
        Keyword arguments for force field constructor.
    """
    if force_field == 'uff':
        ff = AllChem.UFFGetMoleculeForceField(
            mol, confId=conf_id, **kwargs)
    elif force_field.startswith('mmff'):
        AllChem.MMFFSanitizeMolecule(mol)
        mmff_props = AllChem.MMFFGetMoleculeProperties(
            mol, mmffVariant=force_field)
        ff = AllChem.MMFFGetMoleculeForceField(
            mol, mmff_props, confId=conf_id, **kwargs)
    else:
        raise ValueError("Invalid force_field {}".format(force_field))
    return ff


def get_conformer_energies(mol, force_field='mmff'):
    """
    Calculate conformer energies.
    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    force_field : str, optional
        Force Field name.
    Returns
    -------
    energies : array_like
        Minimized conformer energies.
    """
    energies = []
    for conf in mol.GetConformers():
        ff = get_molecule_force_field(mol, conf_id=conf.GetId(), force_field=force_field)
        energy = ff.CalcEnergy()
        energies.append(energy)
    energies = np.asarray(energies, dtype=float)
    return energies


def tanimoto_sim(mol, ref):
    fp1 = Chem.RDKFingerprint(ref)
    fp2 = Chem.RDKFingerprint(mol)
    return DataStructs.TanimotoSimilarity(fp1,fp2)

def tanimoto_dis(mol, ref):
    return 1 - tanimoto_sim(mol, ref)

def tanimoto_dis_N_to_1(mols, ref):
    sim = [tanimoto_dis(m, ref) for m in mols]
    return sim

def compute_diversity(results):
    diversity = []
    for res in tqdm(results, desc='pocket'):
        pocket_results = [r for r in res if r['mol'] is not None]

        mols = [r['mol'] for r in pocket_results]
        for j in range(len(mols)):
            tmp = tanimoto_dis_N_to_1(mols, mols[j])
            tmp.pop(j)
            diversity += tmp
    diversity = np.array(diversity)
    print('[Diversity] Avg: %.4f | Med: %.4f ' % (np.mean(diversity), np.median(diversity)))
    return diversity

def compute_similarity(results, refs):
    similarity = []
    for i, res in tqdm(enumerate(results), total=len(results), desc='pocket sim'):
        pocket_results = [r for r in res if r['mol'] is not None]

        mols = [r['mol'] for r in pocket_results]
        for j in range(len(mols)):
            tmp = tanimoto_sim(mols[j], refs[i])
            similarity.append(tmp)
    similarity = np.array(similarity)
    print('[Similarity] Avg: %.4f | Med: %.4f ' % (np.mean(similarity), np.median(similarity)))
    return similarity

def compute_novelty(results, refs):
    novelty = []
    for i, res in tqdm(enumerate(results), total=len(results), desc='pocket nov'):
        pocket_results = [r for r in res if r['mol'] is not None]

        mols = [r['mol'] for r in pocket_results]
        smis = [Chem.MolToSmiles(m) for m in mols]
        ref = Chem.MolToSmiles(refs[i])
        for j in range(len(mols)):
            smi = smis[j]
            novelty.append(1 if smi != ref else 0)
    novelty = np.array(novelty)
    print('[Novelty] Avg: %.4f | Med: %.4f ' % (np.mean(novelty), np.median(novelty)))
    return novelty

def compute_uniqueness(results):
    unique = []
    smis = []
    for res in tqdm(results, desc='pocket uniq'):
        pocket_results = [r for r in res if r['mol'] is not None]

        mols = [r['mol'] for r in pocket_results]
        if len(mols) == 0: continue
        smis.extend([Chem.MolToSmiles(m) for m in mols])
    unique = len(set(smis)) / len(smis)
    print('[Uniqueness] Avg: %.4f ' % (np.mean(unique)))
    return unique

def compute_scaffold_similarity(results, refs, min_rings=2):
    scaffold_similarity = []
    for i, res in tqdm(enumerate(results), total=len(results), desc='pocket scaffold sim'):
        pocket_results = [r for r in res if r['mol'] is not None]

        mols = [r['mol'] for r in pocket_results]
        scaffold_mol = compute_scaffolds(mols, min_rings=min_rings)
        scaffold_ref = compute_scaffolds([refs[i]], min_rings=min_rings)
        if len(scaffold_mol) == 0 or len(scaffold_ref) == 0:
            if len(scaffold_mol) == 0:
                print('No scaffold found for ref', i, pocket_results[0]['ligand_filename'])
            if len(scaffold_ref) == 0:
                print('No scaffold found for mols', i, pocket_results[0]['ligand_filename'])
            continue
        scaffold_similarity.append(cos_similarity(scaffold_ref, scaffold_mol))
    scaffold_similarity = np.array(scaffold_similarity)
    print('[Scaffold Similarity] Avg: %.4f | Med: %.4f ' % (np.mean(scaffold_similarity), np.median(scaffold_similarity)))
    return scaffold_similarity


def compute_scaffolds(mol_list, n_jobs=1, min_rings=2):
    """
    Extracts a scafold from a molecule in a form of a canonic SMILES
    """
    def compute_scaffold(mol, min_rings=2):
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        except (ValueError, RuntimeError):
            print('Failed to compute scaffold for mol', mol)
            return None
        n_rings = scaffold.GetRingInfo().NumRings()
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        if scaffold_smiles == '' or n_rings < min_rings:
            print('Failed to compute scaffold for scaffold smi', scaffold_smiles, n_rings, min_rings)
            return None
        return scaffold_smiles

    scaffolds = Counter(
        [compute_scaffold(mol, min_rings=min_rings) for mol in mol_list])
    if None in scaffolds:
        scaffolds.pop(None)
    return scaffolds


def cos_similarity(ref_counts, gen_counts):
    """
    Computes cosine similarity between
     dictionaries of form {name: count}. Non-present
     elements are considered zero:

     sim = <r, g> / ||r|| / ||g||
    """
    if len(ref_counts) == 0 or len(gen_counts) == 0:
        return np.nan
    keys = np.unique(list(ref_counts.keys()) + list(gen_counts.keys()))
    ref_vec = np.array([ref_counts.get(k, 0) for k in keys])
    gen_vec = np.array([gen_counts.get(k, 0) for k in keys])
    return 1 - cos_distance(ref_vec, gen_vec)

