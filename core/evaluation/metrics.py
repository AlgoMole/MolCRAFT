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


class CondMolGenMetric(object):
    def __init__(
        self, atom_decoder, atom_enc_mode, type_one_hot, single_bond, docking_config
    ):
        self.atom_decoder = atom_decoder
        self.atom_enc_mode = atom_enc_mode
        self.type_one_hot = type_one_hot
        self.single_bond = single_bond
        self.docking_config = docking_config

    def compute_stability(self, generated: List[Data]):
        n_samples = len(generated)
        molecule_stable = 0
        nr_stable_bonds = 0
        n_atoms = 0
        for data in generated:
            positions = data.pos
            atom_type = data.x
            
            stability_results = check_stability(
                positions=positions,
                atom_type=atom_type,
                type_one_hot=self.type_one_hot,
                atom_decoder=self.atom_decoder,
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

    def compute_recon_success(self, generated: List[Data]):
        """generated: list of couples (positions, atom_types)"""
        valid, complete = [], []

        for graph in generated:
            try:
                mol = graph.mol
            except:
                continue
            smiles = mol2smiles(mol)
            if smiles is not None:
                data = {"mol": mol, "smiles": smiles, "ligand_filename": graph.ligand_filename}
                if '.' in smiles:
                    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    smiles = mol2smiles(largest_mol)
                    if smiles is not None:
                        valid.append(data)
                else:
                    valid.append(data)
                    complete.append(data)

        recon_dict = {
            "validity": len(valid) / len(generated),  # valid
            "completeness": len(complete) / len(generated),
        }
        return valid, recon_dict

    def compute_chem_results(self, generated: List[Dict]):
        chem_list = []
        for graph in tqdm(generated, total=len(generated), desc="Chem eval"):
            chem_results = {}
            try:
                mol = graph.mol
            except:
                chem_list.append(chem_results)
                continue
            try:
                mol = graph.mol
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                if len(mol_frags) > 1:
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    if self.docking_config is not None and self.docking_config.mode == 'vina_dock':
                        prefix = "Dock testing"
                    else:
                        prefix = "Validating"
                    # if largest_mol.GetNumAtoms() < 0.8 * mol.GetNumAtoms():
                    #     print(f"{prefix}: {graph.ligand_filename} has {len(mol_frags)} fragments. Atom number: {largest_mol.GetNumAtoms()} (originally {mol.GetNumAtoms()}). Skipped evaluation.")
                    #     chem_list.append(chem_results)
                    #     continue
                    # else:
                    #     print(f"{prefix}: {graph.ligand_filename} has {len(mol_frags)} fragments. Atom number: {largest_mol.GetNumAtoms()} (originally {mol.GetNumAtoms()}). Skipped evaluation.") # Using largest fragment for evaluation.")
                    #     chem_list.append(chem_results)
                    #     mol = largest_mol
                    #     continue
                    print(f"{prefix}: {graph.ligand_filename} has {len(mol_frags)} fragments. Atom number: {largest_mol.GetNumAtoms()} (originally {mol.GetNumAtoms()}). Skipped evaluation.")
                    chem_list.append(chem_results)
                    continue
                    
                ligand_filename = graph.ligand_filename
                pos = graph.pos.cpu().numpy().astype('float64')

                # qed, logp, sa, lipinski, ring size, etc
                chem_results = scoring_func.get_chem(mol)
                chem_results['atom_num'] = mol.GetNumAtoms()
                
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
                            'vina_score': score_only_results[0]['affinity'],
                            'vina_minimize': minimize_results[0]['affinity'],
                        }
                        if self.docking_config.mode == 'vina_dock':
                            docking_results = vina_task.run(mode='dock', exhaustiveness=self.docking_config.exhaustiveness)
                            vina_results['vina_dock'] = docking_results[0]['affinity']
                            vina_results['pose'] = docking_results[0]['pose']
                        # pose_check_results = vina_task.run_pose_check()
                    else:
                        raise NotImplementedError(f"Unknown docking mode: {self.docking_config.mode}")
                    chem_results.update(vina_results)
                    # chem_results.update(pose_check_results)
            except Exception as e:
                print(e)

            chem_list.append(chem_results)

        return chem_list

    def compute_geometry(self, generated: List[Data]):
        geo_list = []

        for graph in generated:
            positions = graph.pos
            mol_center = positions.mean(dim=0)
            protein_center = graph.protein_pos.mean(dim=0)
            geo_list.append({
                'center_change': (mol_center - protein_center).norm().item(),
                'mol_pos_range':(positions.max(dim=0)[0] - positions.min(dim=0)[0]).norm().item()
            })

        geo_dict = {
            k: np.mean([d[k] for d in geo_list])
            for k in geo_list[0].keys()
        }
        return geo_dict

    def evaluate(self, generated: List[Data], raw_evaluation=None):
        if raw_evaluation is None:
            raw_evaluation = self.compute_raw_evaluation(generated)

        metrics = {}
        for k, v in raw_evaluation.items():
            if isinstance(v, float):
                metrics[k] = v
            elif isinstance(v, list):
                if len(v) == 0: continue
                # calc median while excluding None
                chem_list = [v2 for v2 in v if v2 is not None]
                chem_keys = list(set([k for d in chem_list for k in d.keys()]))
                for k2 in chem_keys:
                    if 'pose' in k2: continue
                    k_list = [d[k2] for d in chem_list if k2 in d]
                    if 'vina' not in k2:
                        metrics[k2 + '_mean'] = np.mean(k_list)
                    else:
                        # calc mean while excluding positive values
                        metrics[k2 + '_median'] = np.median(k_list)
                        metrics[k2 + '_mean'] = np.mean(k_list)
                        k_list_filtered = [v2 for v2 in k_list if v2 < 0]
                        metrics[k2 + '_mean_filter'] = np.mean(k_list_filtered)
                        metrics[k2 + '_eval_success'] = len(k_list_filtered) / len(generated)
            else:
                raise ValueError(f"Unknown type of {k}: {type(v)}")
        
        return metrics

    def compute_raw_evaluation(self, generated: List[Data]):
        """generated: list of pairs 
        (positions: n x 3, atom_types: n x K [int] if type_one_hot else n [int])
        the positions and atom types should already be masked."""

        # eval
        stability_dict = self.compute_stability(generated)
        valid, recon_dict = self.compute_recon_success(generated)
        geo_dict = self.compute_geometry(generated)

        chem_list = self.compute_chem_results(generated)  # TargetDiff reconstruction

        return {
            **recon_dict,
            **stability_dict,
            **geo_dict,
            'chem': chem_list,
        }
