# We implement the evaluation metric in this file.
from rdkit import Chem
from torch_geometric.data import Data
from core.evaluation.utils import scoring_func

from core.evaluation.utils import (
    check_stability,
    convert_atomcloud_to_mol_smiles,
    mol2smiles,
)
from core.evaluation.docking_qvina import QVinaDockingTask
from core.evaluation.docking_vina import VinaDockingTask
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np
import os
from posecheck import PoseCheck
from copy import deepcopy


class ModelResults:
    def __init__(self, name, full_name, results: list[dict]=None):
        self.name = name
        self.full_name = full_name
        self.results = results
    
    def __len__(self):
        return len(self.results)
    
    def __getitem__(self, idx):
        return self.results[idx]

    @property
    def smiles_list(self):
        return np.array([x['smiles'] for x in self.results])
    
    @property
    def complete_list(self):
        return np.array([x['complete'] for x in self.results])

    @property
    def validity_list(self):
        return np.array([x['validity'] for x in self.results])

    @property
    def center_change_list(self):
        return np.array([x['center_change'] for x in self.results])
    
    @property
    def mol_pos_range_list(self):
        return np.array([x['mol_pos_range'] for x in self.results])

    @property
    def atom_num_list(self):
        return np.array([x['mol'].GetNumAtoms() for x in self.results])

    @property
    def qed_list(self):
        return np.array([(x['chem_results']['qed'] if 'chem_results' in x else np.nan) for x in self.results])
    
    @property
    def sa_list(self):
        return np.array([(x['chem_results']['sa'] if 'chem_results' in x else np.nan) for x in self.results])
    
    @property
    def logp_list(self):
        return np.array([(x['chem_results']['logp'] if 'chem_results' in x else np.nan) for x in self.results])
    
    @property
    def lipinski_list(self):
        return np.array([(x['chem_results']['lipinski'] if 'chem_results' in x else np.nan) for x in self.results])

    @property
    def vina_score_list(self):
        return np.array([(x['vina']['score_only'][0]['affinity'] if 'vina' in x else np.nan) for x in self.results])
    
    @property
    def vina_min_list(self):
        return np.array([(x['vina']['minimize'][0]['affinity'] if 'vina' in x else np.nan) for x in self.results])
    
    @property
    def vina_dock_list(self):
        return np.array([(x['vina']['dock'][0]['affinity'] if 'vina' in x else np.nan) for x in self.results])

    @property
    def strain_list(self):
        return np.array([(x['pose_check']['strain'] if 'pose_check' in x else np.nan) for x in self.results])

    @property
    def clash_list(self):
        return np.array([(x['pose_check']['clash'] if 'pose_check' in x else np.nan) for x in self.results])


class CondMolGenMetric(object):
    def __init__(
        self, atom_decoder, atom_enc_mode, type_one_hot, single_bond, docking_config
    ):
        self.atom_decoder = atom_decoder
        self.atom_enc_mode = atom_enc_mode
        self.type_one_hot = type_one_hot
        self.single_bond = single_bond
        self.docking_config = docking_config

    def compute_stability(self, generated: list[dict]):
        n_samples = len(generated)
        molecule_stable = 0
        nr_stable_bonds = 0
        n_atoms = 0
        for data in generated:
            positions = data['pred_pos']
            atom_type = data['pred_v']
            
            stability_results = check_stability(
                positions=positions,
                atom_type=atom_type,
                # type_one_hot=self.type_one_hot,
                # atom_decoder=self.atom_decoder,
                single_bond=self.single_bond,
            )
            
            molecule_stable += int(stability_results[0])
            nr_stable_bonds += int(stability_results[1])
            n_atoms += int(stability_results[2])

        # stability
        fraction_mol_stable = molecule_stable / float(n_samples)
        fraction_atm_stable = nr_stable_bonds / float(n_atoms)
        stability_dict = {
            "mol_stable": fraction_mol_stable,
            "atm_stable": fraction_atm_stable,
        }
        return stability_dict

    def compute_chem_results(self, generated: list[dict]):
        # chem_list = []
        pc = None
        last_protein_fn = None

        for item in tqdm(generated, total=len(generated), desc="Chem eval"):
            mol = item['mol']

            try:
                ligand_filename = item['ligand_filename']
                # pos = item['pos'].cpu().numpy().astype('float64')
                pos = item['pred_pos']

                # qed, logp, sa, lipinski, etc
                chem_results = scoring_func.get_chem(mol)
                chem_results['atom_num'] = mol.GetNumAtoms()
                item['chem_results'] = chem_results
            except Exception as e:
                print(f'[CHEM FAIL] {e}')

            try:
                # docking
                if self.docking_config is not None:
                    if self.docking_config.mode == 'qvina':
                        raise NotImplementedError("QVina is not supported in this version.")
                        vina_task = QVinaDockingTask.from_generated_mol(
                            mol, ligand_filename, pos=pos, protein_root=self.docking_config.protein_root)
                        vina_results = {
                            'qvina': vina_task.run_sync()[0]['affinity']
                        }
                    elif self.docking_config.mode in ['vina_score', 'vina_dock']:
                        vina_task = VinaDockingTask.from_generated_mol(
                            mol, ligand_filename, pos=pos, protein_root=self.docking_config.protein_root)
                        score_only_results = vina_task.run(mode='score_only', exhaustiveness=self.docking_config.exhaustiveness)
                        minimize_results = vina_task.run(mode='minimize', exhaustiveness=self.docking_config.exhaustiveness)
                        vina_results = {
                            'score_only': score_only_results,
                            'minimize': minimize_results,
                        }
                        if self.docking_config.mode == 'vina_dock':
                            docking_results = vina_task.run(mode='dock', exhaustiveness=self.docking_config.exhaustiveness)
                            vina_results['dock'] = docking_results
                        # pose_check_results = vina_task.run_pose_check()
                    else:
                        raise NotImplementedError(f"Unknown docking mode: {self.docking_config.mode}")
                    item['vina'] = vina_results
                    # chem_results.update(pose_check_results)
            except Exception as e:
                print(f'[VINA FAIL] {e}')

            try:
                protein_fn = os.path.join(
                    self.docking_config.protein_root,
                    os.path.dirname(ligand_filename),
                    os.path.basename(ligand_filename)[:10] + '.pdb'
                )
                if protein_fn != last_protein_fn:
                    del pc
                    pc = PoseCheck()
                    pc.load_protein_from_pdb(protein_fn)
                    last_protein_fn = protein_fn
                pc.load_ligands_from_mols([mol])
                strain = pc.calculate_strain_energy()[0]
                clash = pc.calculate_clashes()[0]
                item['pose_check'] = {
                    'strain': strain,
                    'clash': clash,
                }
            except Exception as e:
                print(f'[POSE CHECK FAIL] {e}')

    def evaluate(self, generated: list[dict], bad_case_dir: str = None):
        """generated: list of pairs 
        (positions: n x 3, atom_types: n x K [int] if type_one_hot else n [int])
        the positions and atom types should already be masked."""

        # generated = 
        # eval
        stability_dict = self.compute_stability(generated)
        # valid, recon_dict = self.compute_recon_success(generated)
        # geo_dict = self.compute_geometry(generated)

        self.compute_chem_results(generated)  # TargetDiff reconstruction

        metrics = {
            # **recon_dict,
            **stability_dict,
            # **geo_dict,
        }

        def stat1(arr, name):
            n_total = len(arr)
            isnan = np.isnan(arr)
            n_isnan = isnan.sum()
            arr2 = arr[~isnan]
            return {
                f'{name}_fail': n_isnan / n_total,
                f'{name}_mean': np.mean(arr2)
            }
        results = ModelResults('bfn', 'molcraft', generated)

        metrics.update(stat1(results.qed_list, 'qed'))
        metrics.update(stat1(results.sa_list, 'sa'))

        def save_bad_case(idx, res):
            if bad_case_dir is None: return
            mol = res['mol']
            ligand_filename = res["ligand_filename"]
            atom_num = res['chem_results']['atom_num']
            center_change = res['center_change']
            mol_pos_range = res['mol_pos_range']
            qed = res['chem_results']['qed']
            sa = res['chem_results']['sa']
            lipinski = res['chem_results']['lipinski']
            if 'vina' in res:
                vina_score = res['vina']['score_only'][0]['affinity']
                vina_min = res['vina']['minimize'][0]['affinity']
            else:
                vina_score = np.nan
                vina_min = np.nan
            # vina_dock = res['vina']['dock'][0]['affinity']
            if 'pose_check' in res:
                strain = res['pose_check']['strain']
                clash = res['pose_check']['clash']
            else:
                strain = np.nan
                clash = np.nan
            mol.SetProp('_Name', ligand_filename)
            mol.SetProp('atom_num', str(atom_num))
            mol.SetProp('center_change', str(center_change))
            mol.SetProp('mol_pos_range', str(mol_pos_range))
            mol.SetProp('qed', str(qed))
            mol.SetProp('sa', str(sa))
            mol.SetProp('lipinski', str(lipinski))
            mol.SetProp('vina_score', str(vina_score))
            mol.SetProp('vina_min', str(vina_min))
            mol.SetProp('strain', str(strain))
            mol.SetProp('clash', str(clash))
            with Chem.SDWriter(os.path.join(bad_case_dir, f'{idx}.sdf')) as w:
                w.write(mol)

        pos_vina_msg = {}
        no_vina_msg = {}
        for idx, res in enumerate(results):
            ligand_filename = res["ligand_filename"]
            try:
                vina_score = res['vina']['score_only'][0]['affinity']
                vina_min = res['vina']['minimize'][0]['affinity']
                if vina_score > 0 or vina_min > 0:
                    if ligand_filename not in pos_vina_msg:
                        pos_vina_msg[ligand_filename] = ''
                    
                    _ = deepcopy(res)
                    del _['pred_pos'], _['pred_v'], _['is_aromatic'], _['mol']
                    _['vina'] = {
                        'vina_score': vina_score,
                        'vina_minimize': vina_min,
                    }
                    
                    pos_vina_msg[ligand_filename] += f'{idx} {_}\n'
                    save_bad_case(idx, res)
            except Exception as e:
                if ligand_filename not in no_vina_msg:
                    no_vina_msg[ligand_filename] = []
                no_vina_msg[ligand_filename].append(idx)
        if len(pos_vina_msg):
            for k, v in pos_vina_msg.items():
                print(f'[POS VINA] ligand_fn = {k}, n_ligand = {len(v)}')
                print(f'[POS VINA] ligand index = {v}')
        if len(no_vina_msg):
            for k, v in no_vina_msg.items():
                print(f'[NO VINA] ligand_fn = {k}, n_ligand = {len(v)}')
                print(f'[NO VINA] ligand index = {v}')                

        def stat2(arr, name):
            n_total = len(arr)
            isnan = np.isnan(arr)
            n_isnan = isnan.sum()
            arr2 = arr[~isnan]
            return {
                f'{name}_fail': n_isnan / n_total,
                f'{name}_mean': np.mean(arr2),
                f'{name}_median': np.median(arr2),
                f'{name}_neg_mean': np.mean(arr2[arr2 < 0]),
                f'{name}_neg_ratio': (arr2 < 0).sum() / len(arr2),
            }

        if 'vina' in results[0]:
            vina_score_list = results.vina_score_list
            metrics.update(stat2(vina_score_list, 'vina_score'))
            if 'minimize' in results[0]['vina']:
                vina_min_list = results.vina_min_list
                metrics.update(stat2(vina_min_list, 'vina_minimize'))
            if 'dock' in results[0]['vina']:
                vina_dock_list = results.vina_dock_list
                metrics.update(stat2(vina_dock_list, 'vina_dock'))

        def stat3(arr, name):
            n_total = len(arr)
            isnan = np.isnan(arr)
            n_isnan = isnan.sum()
            arr2 = arr[~isnan]
            perc = np.percentile(arr2, [25, 50, 75])
            return {
                f'{name}_fail': n_isnan / n_total,
                f'{name}_25': perc[0],
                f'{name}_50': perc[1],
                f'{name}_75': perc[2],
            }

        metrics.update(stat1(results.clash_list, 'clash'))
        metrics.update(stat3(results.strain_list, 'strain'))
    
        return metrics
