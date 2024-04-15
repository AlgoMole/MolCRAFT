import copy
import collections
import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
from rdkit.Chem import PeriodicTable
from rdkit.Chem.Lipinski import RotatableBondSmarts
import scipy
from scipy import spatial as sci_spatial
import torch
from tqdm.auto import tqdm
# import seaborn as sns
from copy import deepcopy

ptable = Chem.GetPeriodicTable()

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  

from core.evaluation.utils import eval_bond_length, scoring_func, similarity, eval_bond_angle, eval_torsion_angle
from functools import cached_property 
import json
from collections import Counter


VINA_MEAN = -6.29181914893617
VINA_STD = 3.1398515446772386
LOG_STRAIN_MEAN = 4.632015623614187
LOG_STRAIN_STD = 1.3689898819384352


class ModelResults:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.pc_path = path.replace('.pt', '_pose_checked.pt')

    def load_vina_docked(self):        
        if os.path.exists(self.path):
            results = torch.load(self.path)
            if type(results[0]) is list:
                self.flat_results = [item for sublist in results for item in sublist]
            else:
                self.flat_results = results
            print(f'{self.name} loaded {len(self.flat_results)} vina docked results')

    def load_pose_checked(self):
        if os.path.exists(self.pc_path):
            self.flat_results = torch.load(self.pc_path)
            print(f'{self.name} loaded {len(self.flat_results)} pose checked results')

    def load_vina_docked_sdf(self, sdf_dir):
        flat_results = []
        n_vina = 0
        for i in tqdm(range(10000), desc=self.name):
            fn = os.path.join(sdf_dir, f'{i}.sdf')
            if not os.path.exists(fn): continue

            # use sdf supplier
            suppl = Chem.SDMolSupplier(fn, removeHs=False)
            mol = suppl[0]

            if mol is None or not mol.HasProp('vina_score'): continue

            n_vina += 1
            smiles = Chem.MolToSmiles(mol)
            complete = '.' not in smiles
            mol.SetProp('complete', str(complete))

            ligand_filename = mol.GetProp('_Name')
            vina_score = float(mol.GetProp('vina_score'))
            vina_min = float(mol.GetProp('vina_minimize'))
            if mol.HasProp('vina_dock'):
                vina_dock = float(mol.GetProp('vina_dock'))
            else:
                vina_dock = 10

            chem_dict = scoring_func.get_chem(mol)

            item = {
                'mol': mol,
                'smiles': smiles,
                'complete': complete,
                'ligand_filename': ligand_filename,
                'vina': {'score_only': [{'affinity': vina_score}], 'minimize': [{'affinity': vina_min}], 'dock': [{'affinity': vina_dock}]},
                'chem_results': chem_dict,
            }
            flat_results.append(item)

        self.flat_results = flat_results
        print(f'{self.name} loaded {n_vina} dina docked results')
        print(f'{self.name} has {len(self.flat_results)} to dump')
        torch.save(self.flat_results, self.path)

    def load_pose_checked_sdf(self, sdf_dir):
        flat_results = []
        n_isnan = 0
        n_strain = 0
        if hasattr(self, 'flat_results'):
            max_idx = len(self.flat_results)
        else:
            max_idx = 10000

        for i in tqdm(range(max_idx), desc=self.name):
            fn = os.path.join(sdf_dir, f'{i}.sdf')
            if not os.path.exists(fn): 
                if hasattr(self, 'flat_results'):
                    smiles = self.flat_results[i]['smiles']
                    complete = '.' not in smiles
                    self.flat_results[i]['complete'] = complete
                    self.flat_results[i]['posecheck'] = {'clash': np.nan, 'strain': np.nan}
                continue

            # use sdf supplier
            suppl = Chem.SDMolSupplier(fn, removeHs=False)
            mol = suppl[0]

            # if mol is None or not mol.HasProp('vina_score'): continue
            
            smiles = Chem.MolToSmiles(mol)
            complete = '.' not in smiles
            mol.SetProp('complete', str(complete))

            ligand_filename = mol.GetProp('_Name')
            vina_score = float(mol.GetProp('vina_score'))
            vina_min = float(mol.GetProp('vina_minimize'))
            if mol.HasProp('vina_dock'):
                vina_dock = float(mol.GetProp('vina_dock'))
            else:
                vina_dock = 10
            if mol.HasProp('clash'):
                clash = float(mol.GetProp('clash'))
            else:
                clash = np.nan
            if mol.HasProp('strain'):
                strain = float(mol.GetProp('strain'))
            else:
                strain = np.nan

            if np.isnan(strain):
                n_isnan += 1
            n_strain += 1
            chem_dict = scoring_func.get_chem(mol)

            if hasattr(self, 'flat_results'):
                for key in ['mol', 'smiles', 'ligand_filename', 'vina', 'chem_results']:
                    assert key in self.flat_results[i], f'{key} not in item {i}'

                self.flat_results[i]['complete'] = complete
                self.flat_results[i]['posecheck'] = {'clash': clash, 'strain': strain}
            else:                
                item = {
                    'mol': mol,
                    'smiles': smiles,
                    'complete': complete,
                    'ligand_filename': ligand_filename,
                    'vina': {'score_only': [{'affinity': vina_score}], 'minimize': [{'affinity': vina_min}], 'dock': [{'affinity': vina_dock}]},
                    'posecheck': {'clash': clash, 'strain': strain},
                    'chem_results': chem_dict,
                }
                flat_results.append(item)

        print(f'{self.name} loaded {n_strain} pose checked results, {n_isnan} is nan')
        if not hasattr(self, 'flat_results'):
            self.flat_results = flat_results
        print(f'{self.name} has {len(self.flat_results)} to dump')
        torch.save(self.flat_results, self.pc_path)

    def load_affinity(self, aff_fn):
        flat_results = []
        aff = pkl.load(open(aff_fn, 'rb'))
        for k, v in aff.items():
            ligand_filename = k
            vina_score = v['vina']
            clash = v['clash']
            strain = v['strain']
            if np.isnan(strain): continue
            # print(ligand_filename, vina_score, clash, strain)
            # break
                
            flat_results.append({
                'ligand_filename': ligand_filename,
                'vina': {'score_only': [{'affinity': vina_score}]},
                'posecheck': {'clash': clash, 'strain': strain},
            })

        print(f'{self.name} load {len(flat_results)} results')
        self.flat_results = flat_results

    @property
    def complete_list(self):
        return np.array([x['complete'] for x in self.flat_results])

    def remove_incomplete(self):
        print(f'{self.name} has {len(self.flat_results)} ligands in total')
        flat_results = []
        for res in self.flat_results:
            if res['complete']:
                flat_results.append(res)
        self.flat_results = flat_results
        print(f'{self.name} has {len(flat_results)} complete ligands left')

    @property
    def atom_num_list(self):
        return np.array([x['mol'].GetNumAtoms() for x in self.flat_results])

    @property
    def qed_list(self):
        return np.array([x['chem_results']['qed'] for x in self.flat_results])
    
    @property
    def sa_list(self):
        return np.array([x['chem_results']['sa'] for x in self.flat_results])
    
    @property
    def vina_score_list(self):
        return np.array([x['vina']['score_only'][0]['affinity'] for x in self.flat_results])
    
    @property
    def vina_min_list(self):
        return np.array([x['vina']['minimize'][0]['affinity'] for x in self.flat_results])
    
    @property
    def vina_dock_list(self):
        return np.array([x['vina']['dock'][0]['affinity'] for x in self.flat_results])

    @property
    def smiles_list(self):
        return np.array([x['smiles'] for x in self.flat_results])

    @property
    def clash_list(self):
        clash_list = []
        for res in self.flat_results:
            if 'posecheck' in res:
                clash_list.append(res['posecheck']['clash'])
            else:
                clash_list.append(np.nan)
        # return np.array([x['posecheck']['clash'] for x in self.flat_results])
        return np.array(clash_list)
    
    @property
    def strain_list(self):
        strain_list = []
        for res in self.flat_results:
            if 'posecheck' in res:
                strain_list.append(res['posecheck']['strain'])
            else:
                strain_list.append(np.nan)
        return np.array(strain_list)
        # return np.array([x['posecheck']['strain'] for x in self.flat_results])

    # @cached_property
    # def all_atom_distance(self):
    #     all_atom_distance = []
    #     for res in self.flat_results:
    #         mol = res['mol']
    #         mol = Chem.RemoveAllHs(mol)
    #         atom_coord = mol.GetConformer().GetPositions()
    #         dist = sci_spatial.distance.pdist(atom_coord, 'euclidean')
    #         all_atom_distance += dist.tolist()
    #     return np.array(all_atom_distance)
    
    # @cached_property
    # def all_c_c_distance(self):
    #     c_c_distance_list = []
    #     for res in self.flat_results:
    #         mol = res['mol']
    #         mol = Chem.RemoveAllHs(mol)
    #         for bond_type, dist in eval_bond_length.bond_distance_from_mol(mol):
    #             if bond_type[:2] == (6, 6):
    #                 c_c_distance_list.append(dist)
    #     return np.array(c_c_distance_list)

    @cached_property
    def bond_length_profile(self):
        bond_lengths = []
        for res in self.flat_results:
            mol = res['mol']
            mol = Chem.RemoveAllHs(mol)
            bond_lengths += eval_bond_length.bond_distance_from_mol(mol)
        return eval_bond_length.get_bond_length_profile(bond_lengths)

    @cached_property
    def bond_angle_profile(self):
        bond_angles = []
        for res in self.flat_results:
            mol = res['mol']
            mol = Chem.RemoveAllHs(mol)
            bond_angles += eval_bond_angle.bond_angle_from_mol(mol)
        return eval_bond_angle.get_bond_angle_profile(bond_angles)

    @cached_property
    def torsion_angle_profile(self):
        torsion_angles = []
        for res in self.flat_results:
            mol = res['mol']
            mol = Chem.RemoveAllHs(mol)
            torsion_angles += eval_torsion_angle.torsion_angle_from_mol(mol)
        return eval_torsion_angle.get_torsion_angle_profile(torsion_angles)

    @cached_property
    def ring_size_profile(self):
        ring_sizes = []
        uniq_ring_sizes = set()
        n_4 = 0
        for idx, res in enumerate(self.flat_results):
            mol = res['mol']
            _info = mol.GetRingInfo()
            _sizes = [len(r) for r in _info.AtomRings()]
            if 5 in _sizes:
                n_4 += 1
                # print(f'{model.name}, {idx}, {_sizes}, {res["ligand_filename"]}')
            if len(_sizes):
                uniq_ring_sizes.update(_sizes)
                # if max(_sizes) > 6:
                #     print(f'{model.name}, {idx}, {_sizes}, {res["ligand_filename"]}')
            ring_sizes.append(Counter([s for s in _sizes]))
        # print(n_4, len(ring_sizes))
        ring_size_profile = {}
        # for _size in sorted(uniq_ring_sizes):
        for _size in range(3, 10):
            n_mol = 0
            for counter in ring_sizes:
                # if _size in counter:
                #     n_mol += 1
                n_mol += counter[_size]
            # print(f'{model.name}, ring size = {_size}, ratio = {n_mol/len(ring_sizes)*100:f}')
            ring_size_profile[_size] = n_mol / sum([sum(list(_sizes.values())) for _sizes in ring_sizes])
        return ring_size_profile

    def get_diversity(self):
        diversity_dict = {
            'reference': (0, 0),
            'ar': (0.6965362931884093, 0.7400241837968561),
            'pocket2mol': (0.6945941268424666, 0.7350279676447774),
            'targetdiff': (0.7203867166412661, 0.7251640256537968),
            'decompdiff': (0.6818027291238488, 0.6877551020408164),
            'decompdiff_ref': (0.7328448683279085, 0.7612853863810253),
            'bfn': (0.7215627551051631, 0.7338661865710594),
            'bfn_complete': (0.7175153934690491, 0.7224242248333038),
            'flag': (0, 0),
        }
        if self.name in diversity_dict:
            return diversity_dict[self.name]
        else:
            # return (0, 0)
        
            agg_results = [[] for _ in range(100)]
            for res in self.flat_results:
                ligand_filename = res['ligand_filename']
                idx = ref_fns.index(ligand_filename)
                agg_results[idx].append(res)

            # smiles_list = self.smiles_list
            diversity_list = scoring_func.compute_diversity(agg_results)
            mean, median = np.mean(diversity_list), np.median(diversity_list)
            print(f'{self.name}, diversity: mean = {mean}, median = {median}')
        return mean, median

    def is_vina_success(self, factor=3):
        vina_score_list = np.array(self.vina_score_list)
        min_val, max_val = VINA_MEAN - factor * VINA_STD, VINA_MEAN + factor * VINA_STD
        is_outlier = (vina_score_list < min_val) | (vina_score_list > max_val)
        return ~is_outlier
    
    def is_strain_success(self, factor=3):
        strain_list = np.array(self.strain_list)
        is_outlier = strain_list <= 0
        is_outlier |= np.isnan(strain_list)
        min_val, max_val = LOG_STRAIN_MEAN - factor * LOG_STRAIN_STD, LOG_STRAIN_MEAN + factor * LOG_STRAIN_STD
        for idx, strain in enumerate(strain_list):
            if is_outlier[idx]: continue
            strain = np.log(strain)
            if (strain < min_val) or (strain > max_val):
                is_outlier[idx] = True
        return ~is_outlier

    def is_dcmpdiff_success(self, qed_list, sa_list, vina_dock_list):
        return (qed_list > 0.25) & (sa_list > 0.59) & (vina_dock_list < -8.18)    

    def get_metrics(self, factor=3, flags=None):
        assert flags in [None, 'vina', 'strain', 'both']

        metric_dict = {}
        if flags is None:
            is_success = np.ones(len(self.flat_results), dtype=bool)
        else:
            is_vina_success = self.is_vina_success(factor=factor)
            is_strain_success = self.is_strain_success(factor=factor)
            is_both_success = is_vina_success & is_strain_success

            is_success = eval(f'is_{flags}_success')
        
        metric_dict[f'outlier_rate'] = 1 - np.mean(is_success)

        atom_num_list = self.atom_num_list[is_success]
        metric_dict[f'avg_size'] = np.mean(atom_num_list)

        strain_list = self.strain_list[is_success]
        strain_list = strain_list[~np.isnan(strain_list)]
        if len(strain_list):
            metric_dict[f'strain_percentile'] = np.percentile(strain_list, [25, 50, 75])
        else:
            metric_dict[f'strain_percentile'] = (np.nan, np.nan, np.nan)

        clash_list = self.clash_list[is_success]
        clash_list = clash_list[~np.isnan(clash_list)]
        if len(clash_list):
            metric_dict[f'clash_mean/median'] = (np.mean(clash_list), np.median(clash_list))
        else:
            metric_dict[f'clash_mean/median'] = (np.nan, np.nan)

        metric_dict[f'diversity_mean/median'] = self.get_diversity()
        metric_dict[f'ring_size'] = list(self.ring_size_profile.values())

        for metric in ['vina_score', 'vina_min', 'vina_dock', 'qed', 'sa']:
            data = getattr(self, f'{metric}_list')
            data = np.array(data)
            data = data[is_success]

            mean, median = np.mean(data), np.median(data)
            metric_dict[f'{metric}_mean/median'] = (mean, median)

        vina_dock_list = np.array(self.vina_dock_list)[is_success]
        qed_list = np.array(self.qed_list)[is_success]
        sa_list = np.array(self.sa_list)[is_success]
        is_dcmpdiff_success = self.is_dcmpdiff_success(qed_list, sa_list, vina_dock_list)
        metric_dict[f'success_rate'] = np.sum(is_dcmpdiff_success) / len(self.flat_results)

        vina_score =  metric_dict[f'vina_score_mean/median']
        vina_min =  metric_dict[f'vina_min_mean/median']
        vina_dock =  metric_dict[f'vina_dock_mean/median']
        outlier_rate =  metric_dict[f'outlier_rate']
        qed =  metric_dict[f'qed_mean/median']
        sa =  metric_dict[f'sa_mean/median']
        clash =  metric_dict[f'clash_mean/median']
        diversity = metric_dict[f'diversity_mean/median']
        avg_size =  metric_dict[f'avg_size']
        strain_percentile =  metric_dict[f'strain_percentile']
        success_rate =  metric_dict[f'success_rate']
        ring_size = metric_dict['ring_size']

        def float_1_value(val):
            return f'{val[0]:.2f}'

        def float_2_value(val):
            return f'{val[0]:.2f} & {val[1]:.2f}'
        
        def float_more_value(val):
            return ' & '.join([f'{v:.1f}' for v in val])
        
        def percent_more_value(val):
            return ' & '.join([f'\t{v*100:.2f}\\%' for v in val])

        format_str_1 = f'{float_2_value(vina_score)} & {float_2_value(vina_min)} & {float_2_value(vina_dock)} & ' + \
            f'{float_1_value(qed)} & {float_1_value(sa)} & ' + \
            f'{float_1_value(diversity)} & {avg_size:.1f} & {float_more_value(strain_percentile)} \\\\'
        
        format_str_2 = f'{float_2_value(clash)} & {percent_more_value(ring_size)} \\\\'

        return metric_dict, format_str_1, format_str_2

    @classmethod
    def agg_print_results(cls, models, factor=3, flags=None):
        # collect metrics in each model and create dataframe
        metric_dict = collections.defaultdict(list)
        format_str_1_list, format_str_2_list = [], []
        for model in models:
            _metrics, _format_str_1, _format_str_2 = model.get_metrics(factor=factor, flags=flags)
            for k, v in _metrics.items():
                metric_dict[k].append(v)
            format_str_1_list.append(f'{model.name}\t\t& {_format_str_1}')
            format_str_2_list.append(f'{model.name}\t\t& {_format_str_2}')

        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('display.max_columns', None)
        df = pd.DataFrame(metric_dict, index=[model.name for model in models])
        df.to_csv('agg.csv')
        # change row and column of df
        # df = df.T
        print(df)

        print('\n\n')
        print('\n'.join(format_str_1_list))
        print('\n\n')
        print('\n'.join(format_str_2_list))


def ttest(model_1, model_2):
    # ttest
    print(f'\n\nTTEST {model_1.name} & {model_2.name}')
    for metric in ['vina_score', 'vina_min', 'atom_num']:
        data_1 = getattr(model_1, f'{metric}_list')
        data_2 = getattr(model_2, f'{metric}_list')
        data_1 = np.array(data_1)
        data_2 = np.array(data_2)
        data_1 = data_1[~np.isnan(data_1)]
        data_2 = data_2[~np.isnan(data_2)]
        t, p = scipy.stats.ttest_ind(data_1, data_2)
        print(f'{metric}: t = {t}, p = {p}')


result_dir = '/sharefs/share/sbdd_data/all_results'

ref_path = os.path.join(result_dir, 'crossdocked_test_vina_docked.pt')
tg_path = os.path.join(result_dir, 'targetdiff_vina_docked.pt')
p2m_path = os.path.join(result_dir, 'pocket2mol_vina_docked.pt')
bfn_path = os.path.join(result_dir, 'bfn_mols_v10_vina_docked.pt')
train_path = os.path.join(result_dir, 'train_vina_docked.pt')
dcmp_path = os.path.join(result_dir, '4-decompdiff_docked.pt')
dcmp_ref_path = os.path.join(result_dir, '4-decompdiff_ref_docked.pt')
# dcmp_ref_sdf_path = os.path.join(result_dir, 'dcmp_ref_out_sdfs')
ar_path = os.path.join(result_dir, 'ar_vina_docked.pt')
cvae_path = os.path.join(result_dir, 'cvae_vina_docked.pt')
flag_path = os.path.join('.', 'flag_vina_docked.pt')


ref = ModelResults('reference', ref_path)
ref.load_vina_docked()
ref_fns = [x['ligand_filename'] for x in ref.flat_results]


if __name__ == '__main__':
    cvae = ModelResults('cvae', cvae_path)
    ar = ModelResults('ar', ar_path)
    p2m = ModelResults('pocket2mol', p2m_path)
    flag = ModelResults('flag', flag_path)
    tg = ModelResults('targetdiff', tg_path)
    dcmp = ModelResults('decompdiff', dcmp_path)
    dcmp_ref = ModelResults('decompdiff_ref', dcmp_ref_path)
    bfn = ModelResults('bfn', bfn_path)

    # train = ModelResults('train', train_path)
    # train.load_affinity('../sbdd_workspace/pc_tmp/affinity_pose_check.pkl')
    # train.get_dist()

    ref.load_pose_checked()
    cvae.load_vina_docked()
    ar.load_pose_checked()
    p2m.load_pose_checked()
    tg.load_pose_checked()
    dcmp.load_pose_checked()
    dcmp_ref.load_pose_checked()
    bfn.load_pose_checked()
    flag.load_pose_checked()

    models = [ref, ar, p2m, tg, dcmp, dcmp_ref, bfn, flag]

    # ref.ring_size_profile
    print('\n\nRAW EVALUATION')
    ModelResults.agg_print_results(models, flags=None)
