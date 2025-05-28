# We implement the evaluation metric in this file.
import os
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from torch_geometric.data import Data
from core.utils.reconstruct import reconstruct_from_generated
from core.evaluation.utils import scoring_func

from core.evaluation.utils import (
    check_stability,
    convert_atomcloud_to_mol_smiles,
    mol2smiles,
    remove_all_hs,
)
from core.evaluation.utils.eval_rmsd import (
    get_rmsd_between_mols,
    get_rmsd_between_mol_pdbqt,
)
from core.evaluation.docking_qvina import QVinaDockingTask
from core.evaluation.docking_vina import VinaDockingTask
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np
from posecheck import PoseCheck
from posecheck.utils.chem import remove_radicals
from posebusters import PoseBusters

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 

class CondMolGenMetric(object):
    def __init__(
        self, atom_decoder, atom_enc_mode, type_one_hot, single_bond, docking_config, dataset_smiles_set=[]
    ):
        self.atom_decoder = atom_decoder
        self.atom_enc_mode = atom_enc_mode
        self.type_one_hot = type_one_hot
        self.single_bond = single_bond  # TODO: check TargetDiff default to False (but cause more than 1 double bonds)
        self.docking_config = docking_config
        self.dataset_smiles_set = dataset_smiles_set

    def compute_uniqueness(self, complete):
        """complete: list of SMILES strings."""
        return list(set(complete)), len(set(complete)) / len(complete)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        for smiles in unique:
            if smiles not in self.dataset_smiles_set:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

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

            # TODO: add more TargetDiff metrics
            # pair_dist_results = eval_bond_length.pair_distance_from_pos_v(positions, atom_type)

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
            # mol, smiles = convert_atomcloud_to_mol_smiles(
            #     positions=graph.pos,
            #     atom_type=graph.x,
            #     atom_decoder=self.atom_decoder,
            #     type_one_hot=self.type_one_hot,
            #     single_bond=self.single_bond,
            # )
            # if smiles is not None:
            #     mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
            #     largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            #     smiles = mol2smiles(largest_mol)
            #     if smiles is not None:
            #         data = {"mol": mol, "smiles": smiles, "ligand_filename": graph.ligand_filename}
            #         valid.append(data)
            #         if len(mol_frags) == 1:
            #             complete.append(data)

            # if 'mol' not in graph.keys(): continue
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
                    complete.append(smiles)

        recon_dict = {
            "validity": len(valid) / len(generated),  # valid
            "completeness": len(complete) / len(generated),
        }
        return complete, recon_dict

    def compute_chem_results(self, generated: List[Dict]):
        chem_list = []
        pc = PoseCheck()
        pb = PoseBusters(config="dock")
        for graph in tqdm(generated, total=len(generated), desc="Chem eval"):
            chem_results = {}
            try:
                mol = graph.mol
                smiles = mol2smiles(mol)
                if smiles is None or '.' in smiles:
                    chem_list.append(chem_results)
                    continue
            except:
                chem_list.append(chem_results)
                continue
            try:
                # mol = graph["mol"]
                # ligand_filename = graph["ligand_filename"]
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
                    
                pos = graph.pos.cpu().numpy().astype('float64')

                # qed, logp, sa, lipinski, ring size, etc
                chem_results = scoring_func.get_chem(mol)
                chem_results['atom_num'] = mol.GetNumAtoms()
                
                # docking
                if self.docking_config is not None:
                    ligand_filename = graph.ligand_filename
                    protein_filename = graph.protein_filename
                    if self.docking_config.mode == 'qvina':
                        raise NotImplementedError("QVina is not supported in this version.")
                        vina_task = QVinaDockingTask.from_generated_mol(
                            mol, ligand_filename, pos=pos, protein_root=self.docking_config.protein_root)
                        vina_results = {
                            'qvina': vina_task.run_sync()[0]['affinity']
                        }
                    elif self.docking_config.mode in ['vina_score', 'vina_dock']:
                        vina_task = VinaDockingTask.from_generated_mol(
                            mol, ligand_filename=ligand_filename, protein_filename=protein_filename, pos=pos, protein_root=self.docking_config.protein_root)
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
                            vina_results['rmsd'] = get_rmsd_between_mol_pdbqt(mol, vina_results['pose'])
                        # pose_check_results = vina_task.run_pose_check()
                    else:
                        raise NotImplementedError(f"Unknown docking mode: {self.docking_config.mode}")
                    chem_results.update(vina_results)
                    # chem_results.update(pose_check_results)
            except Exception as e:
                print(e)

            # try:
            #     mol = remove_radicals(mol)
            #     pc.load_ligands_from_mols([mol])
            #     strain = pc.calculate_strain_energy()[0]
            #     chem_results['strain'] = strain
            # except Exception as e:
            #     print(e)

            try:
                mol = remove_radicals(mol)
                ligand_fn = graph.ligand_filename
                protein_fn = os.path.join(
                    os.path.dirname(ligand_filename),
                    os.path.basename(ligand_filename)[:10] + '.pdb' if 'ligand' not in ligand_filename  # PDBId_Chain_rec.pdb
                        else os.path.basename(ligand_filename).replace('_ligand.sdf', '_protein.pdb').replace('_ligand.mol2', '_protein.pdb')
                )
                protein_path = os.path.join(self.docking_config.protein_root, protein_fn)
                ligand_path = os.path.join(self.docking_config.protein_root, ligand_fn)
                df = pb.bust([mol], ligand_path, protein_path)
                # update the chem_results with the results from PoseBusters
                # add a prefix 'pb' to the keys to avoid conflicts
                df.columns = ['pb_' + c for c in df.columns]
                chem_results.update(df.iloc[0].to_dict())
                # add a rate of success for PoseBusters all columns equal True
                pb_valid = df.iloc[0].all()
                chem_results['pb_valid'] = pb_valid
            except Exception as e:
                print(e)

            chem_list.append(chem_results)

        return chem_list

    def compute_geometry(self, generated: List[Data]):
        geo_list = []

        for graph in generated:
            positions = graph.pos
            mol_center = positions.mean(dim=0)
            if 'protein_pos' in graph:
                protein_center = graph.protein_pos.mean(dim=0)
            else:
                protein_center = 0.0
            geo_item = {
                'center_change': (mol_center - protein_center).norm().item(),
                'mol_pos_range':(positions.max(dim=0)[0] - positions.min(dim=0)[0]).norm().item()
            }
            if 'bond' in graph:
                bonds = graph.bond
                geo_item['bond_mean'] = (bonds != 0).sum().item() / len(positions)
            geo_list.append(geo_item)

        geo_dict = {
            k: np.mean([d[k] for d in geo_list])
            for k in geo_list[0].keys()
        }
        return geo_dict

    def compute_unique_scaffolds(self, generated: List[Data]):

        scaffolds = set()
        num_mols = 0
        for graph in generated:
            try:
                mol = graph.mol
                if mol:
                    num_mols += 1
                    # Generate the Bemis-Murcko scaffold as a SMILES string
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    scaffold_smiles = Chem.MolToSmiles(scaffold)
                    scaffolds.add(scaffold_smiles)
            except:
                continue

        if num_mols == 0:
            ratio = 0
        else:
            ratio = len(scaffolds) / num_mols
        
        return {'uniq_scaffold_mean': ratio}

    def evaluate(self, generated: List[Data], raw_evaluation=None):
        if raw_evaluation is None:
            raw_evaluation = self.compute_raw_evaluation(generated)

        metrics = {}
        for k, v in raw_evaluation.items():
            if isinstance(v, float) or isinstance(v, int):
                metrics[k] = v
            elif isinstance(v, list):
                if len(v) == 0: continue
                # calc median while excluding None
                chem_list = [v2 for v2 in v if v2 is not None]
                chem_keys = list(set([k for d in chem_list for k in d.keys()]))
                for k2 in chem_keys:
                    if 'pose' in k2: continue
                    k_list = [d[k2] for d in chem_list if k2 in d]
                    if k2 == 'strain' or k2 == 'rmsd':
                        k_list = np.array(k_list).astype('float64')
                        n_total = len(k_list)
                        isnan = np.isnan(k_list)
                        n_isnan = isnan.sum()
                        arr2 = k_list[~isnan]
                        if len(arr2) == 0:
                            perc = [np.nan, np.nan, np.nan]
                        else:
                            perc = np.percentile(arr2, [25, 50, 75])
                        metrics.update({
                            f'{k2}_fail': n_isnan / n_total,
                            f'{k2}_25': perc[0],
                            f'{k2}_50': perc[1],
                            f'{k2}_75': perc[2],
                        })
                        if k2 == 'rmsd':
                            n_rmsd_lt2 = (arr2 < 2).sum()
                            metrics['rmsd<2'] = n_rmsd_lt2 / n_total
                    elif 'pb' in k2: # TODO: simplify this to only store pb_valid
                        metrics[k2 + '_mean'] = np.mean(k_list)
                    elif 'vina' not in k2: # qed, sa
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

    def compute_raw_evaluation(self, generated: List[Data], skip_chem=False):
        """generated: list of pairs 
        (positions: n x 3, atom_types: n x K [int] if type_one_hot else n [int])
        the positions and atom types should already be masked."""

        # eval
        stability_dict = self.compute_stability(generated)
        complete, recon_dict = self.compute_recon_success(generated)
        if len(complete) > 0:
            unique, unique_ratio = self.compute_uniqueness(complete)
            if self.dataset_smiles_set is not None:
                novel, novel_ratio = self.compute_novelty(unique)
            else:
                novel, novel_ratio = [], 0.0
        else:
            unique, unique_ratio = [], 0.0
            novel, novel_ratio = [], 0.0
        geo_dict = self.compute_geometry(generated)
        diversity_dict = self.compute_unique_scaffolds(generated)

        if skip_chem:
            chem_list = []
        else:
            chem_list = self.compute_chem_results(generated)  # TargetDiff reconstruction


        # TODO: add success rate
        # (QED > 0.25, SA > 0.59, Vina Dock < −8.18) as Success Rate
        return {
            **recon_dict,
            **stability_dict,
            **geo_dict,
            **diversity_dict,
            'chem': chem_list,
            'uniqueness': unique_ratio,
            'novelty': novel_ratio,
        }


class RMSDMetric(CondMolGenMetric):
    # calculate RMSD
    def __init__(self, atom_decoder, atom_enc_mode, type_one_hot, single_bond, protein_root):
        super().__init__(atom_decoder, atom_enc_mode, type_one_hot, single_bond, None)
        self.rmsd_threshold = 2.0
        self.protein_root = protein_root

    def compute_raw_evaluation(self, generated: List[Data], skip_chem=False):
        rmsd_list = []
        for graph in generated:
            ligand_filename = os.path.join(self.protein_root, graph.ligand_filename)
            ligand_rdmol = Chem.MolFromMolFile(ligand_filename)
            ligand_rdmol = remove_all_hs(ligand_rdmol)

            # get mol
            try:
                mol = graph.mol
            except:
                pos, atom_type = graph.pos, graph.atom_type
                xyz = pos.cpu().numpy().astype('float64')
                atomic_nums = atom_type.int().cpu().numpy().tolist()
                n_atoms = len(atomic_nums)

                # copy the ligand rdmol but update the coordinates
                rd_mol = Chem.RWMol(ligand_rdmol)
                rd_conf = rd_mol.GetConformer()
                assert rd_conf.GetNumAtoms() == n_atoms

                # add atoms and coordinates
                for i, atom in enumerate(atomic_nums):
                    # assert the atom index is correct
                    assert atom == rd_mol.GetAtomWithIdx(i).GetAtomicNum()
                    rd_coords = Geometry.Point3D(*xyz[i])
                    rd_conf.SetAtomPosition(i, rd_coords)

                # convert to rdmol
                mol = rd_mol.GetMol()
            
            mol = remove_all_hs(mol)
            # assign bond based on rdmol
            # mol = AllChem.AssignBondOrdersFromTemplate(ligand_rdmol, mol)

            # get ligand pos and rmsd
            rmsd = get_rmsd_between_mols(mol, ligand_rdmol)
            rmsd_list.append(rmsd)

        # eval
        if not skip_chem:
            other_results = super().compute_raw_evaluation(generated)
            
            # add rmsd to 'chem' in other_results
            chem_list = other_results['chem']
            for i, chem in enumerate(chem_list):
                chem['rmsd'] = rmsd_list[i]
        else:
            other_results = {}
            other_results['chem'] = []
            for rmsd in rmsd_list:
                other_results['chem'].append({'rmsd': rmsd})

        # TODO: add success rate
        # (QED > 0.25, SA > 0.59, Vina Dock < −8.18) as Success Rate
        return other_results


    # def evaluate(self, generated: List[Data], raw_evaluation=None):
    #     if raw_evaluation is None:
    #         raw_evaluation = self.compute_raw_evaluation(generated)

    #     rmsd_list = raw_evaluation["rmsd"]
    #     other_metrics = raw_evaluation["other"]
    #     return {
    #         "rmsd": np.mean(rmsd_list),
    #         "rmsd_success": len([r for r in rmsd_list if r < self.rmsd_threshold]) / len(rmsd_list),
    #         **other_metrics,
    #     }