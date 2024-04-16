import copy
import collections
import os
import pickle as pkl
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
from rdkit.Chem import PeriodicTable
from rdkit.Chem.Lipinski import RotatableBondSmarts
from rdkit.Geometry.rdGeometry import Point3D
import scipy
from scipy import spatial as sci_spatial
import torch
from tqdm.auto import tqdm
# import seaborn as sns
from copy import deepcopy

ptable = Chem.GetPeriodicTable()

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  

from core.evaluation.utils import eval_bond_length, scoring_func, similarity, eval_bond_angle, eval_torsion_angle, eval_rmsd, mol2smiles
from functools import cached_property 
import json
from collections import Counter
from pose_check_train_set import CDTrainSet
from core.utils.misc import DisjointSet


VINA_MEAN = -6.29181914893617
VINA_STD = 3.1398515446772386
LOG_STRAIN_MEAN = 4.632015623614187
LOG_STRAIN_STD = 1.3689898819384352
LARGE_RING = 7
LARGEST_FUSED_RING = 7


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
        else:
            print(f'{self.name} pose checked results not found')
    
    @property
    def complete_list(self):
        return np.array([x['complete'] for x in self.flat_results])

    @property
    def validity_list(self):
        return np.array([x['validity'] for x in self.flat_results])

    def remove_incomplete(self):
        print(f'{self.name} has {len(self.flat_results)} ligands in total')
        flat_results = []
        for res in self.flat_results:
            if 'complete' in res:
                if res['complete']:
                    flat_results.append(res)
            else:
                complete = '.' not in res['smiles']
                res['complete'] = complete
                if complete:
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
    def logp_list(self):
        return np.array([x['chem_results']['logp'] for x in self.flat_results])
    
    @property
    def lipinski_list(self):
        return np.array([x['chem_results']['lipinski'] for x in self.flat_results])

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
            if 'pose_check' in res and 'clash' in res['pose_check']:
                clash_list.append(res['pose_check']['clash'])
            else:
                clash_list.append(np.nan)
        return np.array(clash_list)
    
    @property
    def strain_list(self):
        strain_list = []
        for res in self.flat_results:
            if 'pose_check' in res and 'strain' in res['pose_check']:
                strain_list.append(res['pose_check']['strain'])
            else:
                strain_list.append(np.nan)
        return np.array(strain_list)

    @property
    def hb_donor_list(self):
        hb_donor_list = []
        for res in self.flat_results:
            if 'pose_check' in res and 'hb_donor' in res['pose_check']:
                hb_donor_list.append(res['pose_check']['hb_donor'])
            else:
                hb_donor_list.append(np.nan)
        return np.array(hb_donor_list)

    @property
    def hb_acceptor_list(self):
        hb_acceptor_list = []
        for res in self.flat_results:
            if 'pose_check' in res and 'hb_acceptor' in res['pose_check']:
                hb_acceptor_list.append(res['pose_check']['hb_acceptor'])
            else:
                hb_acceptor_list.append(np.nan)
        return np.array(hb_acceptor_list)

    @property
    def vdw_list(self):
        vdw_list = []
        for res in self.flat_results:
            if 'pose_check' in res and 'vdw' in res['pose_check']:
                vdw_list.append(res['pose_check']['vdw'])
            else:
                vdw_list.append(np.nan)
        return np.array(vdw_list)
    
    @property
    def hydrophobic_list(self):
        hydrophobic_list = []
        for res in self.flat_results:
            if 'pose_check' in res and 'hydrophobic' in res['pose_check']:
                hydrophobic_list.append(res['pose_check']['hydrophobic'])
            else:
                hydrophobic_list.append(np.nan)
        return np.array(hydrophobic_list)

    @property
    def rmsd_list(self):
        rmsd_list = []
        for res in self.flat_results:
            if 'rmsd' in res:
                rmsd_list.append(res['rmsd'])
            else:
                rmsd_list.append(np.nan)
        return np.array(rmsd_list)

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
        _largest_ring = 0
        _largest_fused_ring = 0
        mol_ring_sizes = np.zeros((len(self.flat_results), 100))
        mol_ring_no = []
        fused_ring_sizes = []
        mol_fused_rings_sizes = np.zeros((len(self.flat_results), 100))

        for idx, res in enumerate(tqdm(self.flat_results, ncols=120, desc=self.name.strip())):
            mol = res['mol']
            # Assuming mol is your molecule
            mol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(mol)

            _info = mol.GetRingInfo()
            _sizes = [len(r) for r in _info.AtomRings()]
            if len(_sizes):
                _largest_ring = max(_largest_ring, max(_sizes))
            ring_sizes.extend(_sizes)
            for s in np.unique(_sizes):
                mol_ring_sizes[idx, s] = 1
            mol_ring_no.append(len(_sizes))
            
            # check if two rings share the same edge
            disjoint_set = DisjointSet(len(_sizes))
            for i in range(len(_sizes)):
                ring_i = _info.AtomRings()[i]
                for j in range(i+1, len(_sizes)):
                    ring_j = _info.AtomRings()[j]
                    if len(set(ring_i).intersection(ring_j)) > 1:
                        # print(ring_i, ring_j)
                        disjoint_set.union(i, j)
            # print(disjoint_set.parent)
            parents = {}
            for i in range(len(_sizes)):
                p_i = disjoint_set.find(i)
                if p_i not in parents:
                    parents[p_i] = set()
                parents[p_i].add(i)
            # print(parents)
            fused_rings = []
            for s in parents.values():
                fused_rings.append(len(s))
            # print(fused_rings)
            if len(fused_rings):
                _largest_fused_ring = max(_largest_fused_ring, max(fused_rings))
                # print(idx, max(fused_rings))
                # if max(fused_rings) > 5:
                #     with Chem.SDWriter(f'kevin_tmp/fused_{idx}.sdf') as w:
                #         w.write(mol)
            fused_ring_sizes.extend(fused_rings)
            for s in np.unique(fused_rings):
                mol_fused_rings_sizes[idx, s] = 1

        print(f'{self.name}, largest fused ring = {_largest_fused_ring}')

        ring_sizes = np.array(ring_sizes)
        ring_size_ratio = {LARGE_RING: 0}
        mol_ring_ratio = {LARGE_RING: 0}
        ring_size_cnt = Counter(ring_sizes)

        for s in range(3, _largest_ring + 1):
            if s not in ring_size_cnt:
                ring_size_cnt[s] = 0
            
            if s < LARGE_RING:
                ring_size_ratio[s] = ring_size_cnt[s] / len(ring_sizes)
                mol_ring_ratio[s] = mol_ring_sizes[:, s].mean()
            else:
                ring_size_ratio[LARGE_RING] += ring_size_cnt[s] / len(ring_sizes)
                mol_ring_ratio[LARGE_RING] = max(mol_ring_ratio[LARGE_RING], mol_ring_sizes[:, s].mean())
        ring_size_ratio = dict(sorted(ring_size_ratio.items()))
        mol_ring_ratio = dict(sorted(mol_ring_ratio.items()))

        fused_ring_size_ratio = {LARGEST_FUSED_RING: 0}
        mol_fused_ring_ratio = {LARGEST_FUSED_RING: 0}
        fused_ring_size_cnt = Counter(fused_ring_sizes)
        for s in range(1, _largest_fused_ring + 1):
            if s not in fused_ring_size_cnt:
                fused_ring_size_cnt[s] = 0

            if s < LARGEST_FUSED_RING:
                fused_ring_size_ratio[s] = fused_ring_size_cnt[s] / len(fused_ring_sizes)
                mol_fused_ring_ratio[s] = mol_fused_rings_sizes[:, s].mean()
            else:
                fused_ring_size_ratio[LARGEST_FUSED_RING] += fused_ring_size_cnt[s] / len(ring_sizes)
                mol_fused_ring_ratio[LARGEST_FUSED_RING] = max(mol_fused_ring_ratio[LARGEST_FUSED_RING], mol_fused_rings_sizes[:, s].mean())

        fused_ring_size_ratio = dict(sorted(fused_ring_size_ratio.items()))
        mol_fused_ring_ratio = dict(sorted(mol_fused_ring_ratio.items()))

        return ring_size_ratio, mol_ring_ratio, np.array(mol_ring_no), fused_ring_size_ratio, mol_fused_ring_ratio

    def get_diversity(self):
        diversity_dict = {
            # 'reference': (0, 0),
            'ar': (0.6965362931884093, 0.7400241837968561),
            'pocket2mol': (0.6945941268424666, 0.7350279676447774),
            # 'flag': (0, 0),
            'targetdiff': (0.7203867166412661, 0.7251640256537968),
            'decompdiff': (0.6818027291238488, 0.6877551020408164),
            'decompdiff_ref': (0.7328448683279085, 0.7612853863810253),
            'binddm': (0.7465581790657266, 0.7513812154696132),
            'ipdiff': (0.7364082966678656, 0.7448818897637794),
            'bfn': (0, 0),
            'bfn_p2m': (0.7352802713395735, 0.7661596958174905),
            'bfn_dcmpo': (0.6079372290600982, 0.610759493670886),
            'ds_joint': (0.7578263886130838, 0.7633892885691447),
            'ds_cond': (0.7281506824792066, 0.7370721048798252),
        }
        if self.name.strip() in diversity_dict:
            return diversity_dict[self.name.strip()]
        else:        
            agg_results = [[] for _ in range(100)]
            for res in tqdm(self.flat_results, desc=self.name.strip()):
                ligand_filename = res['ligand_filename']
                idx = ref_fns.index(ligand_filename)
                agg_results[idx].append(res)

            # smiles_list = self.smiles_list
            diversity_list = scoring_func.compute_diversity(agg_results)
            mean, median = np.mean(diversity_list), np.median(diversity_list)
            print(f'{self.name}, diversity: mean = {mean}, median = {median}')
        return mean, median
    
    def is_binding_success(self, ref_vina=-2.48955, ref_strain=835.9925, ref_rmsd=2):
        is_success = np.zeros_like(self.vina_score_list, dtype=bool)
        for idx, (vina_score, strain, rmsd) in enumerate(zip(self.vina_score_list, self.strain_list, self.rmsd_list)):
            if np.isnan(vina_score) or np.isnan(strain):
                continue

            if not np.isnan(ref_rmsd):
                if np.isnan(rmsd):
                    continue
                elif (vina_score < ref_vina) and (strain < ref_strain) and (rmsd < ref_rmsd):
                    is_success[idx] = True
            else:
                if (vina_score < ref_vina) and (strain < ref_strain):
                    is_success[idx] = True

        return is_success

    def is_dcmpdiff_success(self, qed_list, sa_list, vina_dock_list):
        is_success = np.zeros_like(self.vina_score_list, dtype=bool)
        for idx, (qed, sa, vina_dock) in enumerate(zip(qed_list, sa_list, vina_dock_list)):
            if np.isnan(qed) or np.isnan(sa) or np.isnan(vina_dock):
                continue

            if (qed > 0.25) and (sa > 0.59) and (vina_dock < -8.18):
                is_success[idx] = True
        
        return is_success

    def get_metrics(self):
        metric_dict = {}

        atom_num_list = self.atom_num_list
        metric_dict[f'avg_size'] = np.mean(atom_num_list)

        strain_list = self.strain_list
        strain_list = strain_list[~np.isnan(strain_list)]
        if len(strain_list):
            metric_dict[f'strain_percentile'] = np.percentile(strain_list, [25, 50, 75])
        else:
            metric_dict[f'strain_percentile'] = (np.nan, np.nan, np.nan)

        clash_list = self.clash_list
        clash_list = clash_list[~np.isnan(clash_list)]
        if len(clash_list):
            metric_dict[f'clash_mean/median'] = (np.mean(clash_list), np.median(clash_list))
        else:
            metric_dict[f'clash_mean/median'] = (np.nan, np.nan)

        rmsd_list = self.rmsd_list
        rmsd_list = rmsd_list[~np.isnan(rmsd_list)]
        if len(rmsd_list):
            metric_dict[f'rmsd'] = np.mean(rmsd_list < 2)
        else:
            metric_dict[f'rmsd'] = 0
        # metric_dict[f'diversity_mean/median'] = self.get_diversity()
        # metric_dict[f'ring_size'] = list(self.ring_size_profile.values())

        bf_w_rmsd = self.is_binding_success(ref_rmsd=2)
        bf_wo_rmsd = self.is_binding_success(ref_rmsd=np.nan)
        metric_dict['bf_w_rmsd'] = bf_w_rmsd.mean()
        metric_dict['bf_wo_rmsd'] = bf_wo_rmsd.mean()

        for metric in ['vina_score', 'vina_min', 'vina_dock', 'qed', 'sa']:
            data = getattr(self, f'{metric}_list')
            data = np.array(data)

            mean, median = np.mean(data), np.median(data)
            metric_dict[f'{metric}_mean/median'] = (mean, median)

        # vina_dock_list = np.array(self.vina_dock_list)
        # qed_list = np.array(self.qed_list)
        # sa_list = np.array(self.sa_list)
        # is_dcmpdiff_success = self.is_dcmpdiff_success(qed_list, sa_list, vina_dock_list)
        # metric_dict[f'success_rate'] = np.sum(is_dcmpdiff_success) / len(self.flat_results)

        vina_score =  metric_dict[f'vina_score_mean/median']
        vina_min =  metric_dict[f'vina_min_mean/median']
        vina_dock =  metric_dict[f'vina_dock_mean/median']

        strain_percentile =  metric_dict[f'strain_percentile']
        clash =  metric_dict[f'clash_mean/median']
        rmsd = metric_dict[f'rmsd']

        qed =  metric_dict[f'qed_mean/median']
        sa =  metric_dict[f'sa_mean/median']
        diversity = -1
        # diversity = metric_dict[f'diversity_mean/median']

        bf_w_rmsd = metric_dict['bf_w_rmsd']
        bf_wo_rmsd = metric_dict['bf_wo_rmsd']
        avg_size =  metric_dict[f'avg_size']

        def float_1_value(val):
            return f'{val:.2f}'

        def float_2_value(val):
            return f'{val[0]:.2f} & {val[1]:.2f}'
        
        def float_more_value(val):
            return ' & '.join([f'{v:12.1f}' for v in val])
        
        def percent_1_value(val):
            return f'\t{val*100:4.2f}\\%'

        format_str = f'{float_2_value(vina_score)} & {float_2_value(vina_min)} & {float_2_value(vina_dock)} & ' +\
            f'{float_more_value(strain_percentile)} & {float_2_value(clash)} & {percent_1_value(rmsd)} & ' +\
            f'{float_2_value(sa)} & {float_2_value(qed)} & {float_1_value(diversity)} & ' +\
            f'{percent_1_value(bf_w_rmsd)} & {percent_1_value(bf_wo_rmsd)} & {float_1_value(avg_size)} \\\\'

        return metric_dict, format_str

    @classmethod
    def agg_print_results(cls, models):
        # collect metrics in each model and create dataframe
        metric_dict = collections.defaultdict(list)
        format_str_list = []
        for model in models:
            _metrics, _format_str = model.get_metrics()
            for k, v in _metrics.items():
                metric_dict[k].append(v)
            format_str_list.append(f'{model.name}\t\t& {_format_str}')

        # pd.set_option('display.expand_frame_repr', False)
        # pd.set_option('display.max_columns', None)
        df = pd.DataFrame(metric_dict, index=[model.name for model in models])
        # df.to_csv('agg.csv')
        # change row and column of df
        # df = df.T
        print(df)

        print('\n\n')
        print('\n'.join(format_str_list))


result_dir = './samples'

ref_path = os.path.join(result_dir, 'crossdocked_test_vina_docked.pt')
train_path = os.path.join(result_dir, 'crossdocked_train_vina_docked.pt')

ref = ModelResults('reference', ref_path)
ref.load_vina_docked()
ref_fns = [x['ligand_filename'] for x in ref.flat_results]

# cvae_path = os.path.join(result_dir, 'cvae_vina_docked.pt')
ar_path = os.path.join(result_dir, 'ar_vina_docked.pt')
p2m_path = os.path.join(result_dir, 'pocket2mol_vina_docked.pt')
flag_path = os.path.join(result_dir, 'flag_vina_docked.pt')

tg_path = os.path.join(result_dir, 'targetdiff_vina_docked.pt')
dcmp_path = os.path.join(result_dir, 'decompdiff_vina_docked.pt')
dcmp_ref_path = os.path.join(result_dir, 'decompdiff_ref_vina_docked.pt')

ds_joint_path = os.path.join(result_dir, 'diffsbdd_joint_vina_docked.pt')
ds_cond_path = os.path.join(result_dir, 'diffsbdd_cond_vina_docked.pt')

bd_path = os.path.join(result_dir, 'binddm_vina_docked.pt')
ip_path = os.path.join(result_dir, 'ipdiff_vina_docked.pt')

bfn_path = os.path.join(result_dir, 'bfn_vina_docked.pt')
bfn_p2m_path = os.path.join(result_dir, 'bfn_p2m_vina_docked.pt')
bfn_dcmpo_path = os.path.join(result_dir, 'bfn_dcmpo_vina_docked.pt')

if __name__ == '__main__':
    train = ModelResults('train\t', train_path)

    # cvae = ModelResults('cvae\t', cvae_path)
    ar = ModelResults('ar\t', ar_path)
    p2m = ModelResults('pocket2mol', p2m_path)
    flag = ModelResults('flag\t', flag_path)
    
    tg = ModelResults('targetdiff', tg_path)
    dcmp = ModelResults('decompdiff', dcmp_path)
    dcmp_ref = ModelResults('decompdiff_ref', dcmp_ref_path)
    
    ds_joint = ModelResults('diffsbdd_joint', ds_joint_path)
    ds_cond = ModelResults('diffsbdd_cond', ds_cond_path)

    bd = ModelResults('binddm\t', bd_path)
    ip = ModelResults('ipdiff\t', ip_path)
    
    bfn = ModelResults('bfn\t', bfn_path)
    bfn_p2m = ModelResults('bfn_p2m\t', bfn_p2m_path)
    bfn_dcmpo = ModelResults('bfn_dcmpo', bfn_dcmpo_path)

    bfn_p2m.load_pose_checked()
    bfn_p2m.remove_incomplete()
    qed_list = bfn_p2m.qed_list
    sa_list = bfn_p2m.sa_list
    vina_dock_list = bfn_p2m.vina_dock_list
    dcmp_success = bfn_p2m.is_dcmpdiff_success(qed_list, sa_list, vina_dock_list)
    print(dcmp_success.mean())

    # bfn_p2m.load_pose_checked()
    # bfn_dcmpo.load_pose_checked()
    # bf = bfn_p2m.is_binding_success()
    # print(np.mean(bf))
    # bf = bfn_dcmpo.is_binding_success()
    # print(np.mean(bf))
    # exit(0)

    # model = bfn_dcmpo
    # # model.load_vina_docked()
    # # model.load_pose_checked_sdf('./my_results/ds_cond_out_sdfs')
    # model.load_pose_checked()
    # model.remove_incomplete()

    # # div = model.get_diversity()
    # # print(model.name, 'div', div)
    # # exit(0)

    # # res = model.flat_results[0]
    # # for k, v in res.items():
    # #     print(k, v)
    # # exit(0)

    # vina_score_list = model.vina_score_list
    # vina_score_list = vina_score_list[~np.isnan(vina_score_list)]
    # vina_score_avg, vina_score_med = np.mean(vina_score_list), np.median(vina_score_list)
    # print(f'vina score, {vina_score_avg:.2f}, {vina_score_med:.2f}')
    # vina_min_list = model.vina_min_list
    # vina_min_avg, vina_min_med = np.mean(vina_min_list), np.median(vina_min_list)
    # print(f'vina min, {vina_min_avg:.2f}, {vina_min_med:.2f}')
    # vina_dock_list = model.vina_dock_list
    # vina_dock_avg, vina_dock_med = np.mean(vina_dock_list), np.median(vina_dock_list)
    # print(f'vina dock, {vina_dock_avg:.2f}, {vina_dock_med:.2f}')
    # strain_list = model.strain_list
    # strain_list = strain_list[~np.isnan(strain_list)]
    # strain_25, strain_50, strain_75 = np.percentile(strain_list, [25, 50, 75])
    # print(f'strain, {strain_25:.0f}, {strain_50:.0f}, {strain_75:.0f}')
    # clash_list = model.clash_list
    # clash_list = clash_list[~np.isnan(clash_list)]
    # clash = np.mean(clash_list)
    # print(f'clash, {clash:.2f}')
    # rmsd_list = model.rmsd_list
    # rmsd_list = rmsd_list[~np.isnan(rmsd_list)]
    # rmsd = (rmsd_list < 2).sum() / len(model.flat_results)
    # print(f'rmsd, {rmsd*100:.1f}')
    # sa_list = model.sa_list
    # sa = np.mean(sa_list)
    # print(f'sa, {sa:.2f}')
    # qed_list = model.qed_list
    # qed = np.mean(qed_list)
    # print(f'qed, {qed:.2f}')
    # lipinski_list = model.lipinski_list
    # lip_mean, lip_std = np.mean(lipinski_list), np.std(lipinski_list)
    # print(f'lipinski, {lip_mean:.2f} +- {lip_std:.2f}')
    # bf = model.is_binding_success()
    # bf = bf.mean()
    # print(f'bf, {bf*100:.1f}')
    # avg_size = np.mean(model.atom_num_list)
    # print(f'avg_size, {avg_size:.1f}')
    # exit(0)

    models = [bfn_p2m, bfn_dcmpo, ds_cond, ds_joint, bd]
    for model in models:
        model.load_pose_checked()
        model.remove_incomplete()

    for model in models:
        donor_list = model.hb_donor_list
        acceptor_list = model.hb_acceptor_list
        vdw_list = model.vdw_list
        hydrophobic_list = model.hydrophobic_list

        # strain_list = model.strain_list
        # print(f'{np.mean(np.isnan(strain_list))} strain is nan, {np.mean(np.isnan(donor_list))} donor is nan')
        donor_list = donor_list[~np.isnan(donor_list)]
        acceptor_list = acceptor_list[~np.isnan(acceptor_list)]
        vdw_list = vdw_list[~np.isnan(vdw_list)]
        hydrophobic_list = hydrophobic_list[~np.isnan(hydrophobic_list)]
        print(f'{model.name}, hb_donor, {donor_list.mean():.2f}, hb_acceptor, {acceptor_list.mean():.2f}, vdw, {vdw_list.mean():.2f}, hydrophobic, {hydrophobic_list.mean():.2f}')
    exit(0)

    models = [ref, train, ar, p2m, flag, ds_cond, ds_joint, tg, dcmp, dcmp_ref, bd, ip, bfn, bfn_p2m, bfn_dcmpo]
    for model in models:
        model.load_pose_checked()
        model.remove_incomplete()
    
    # exit(0)

    ring_sizes = range(3, LARGE_RING + 1) 
    fused_ring_sizes = range(1, LARGEST_FUSED_RING + 1)   
    all_ring_size_ratio = []
    all_mol_ring_ratio = []
    all_ring_no = []
    all_fused_ring_ratio = []
    all_mol_fused_ring_ratio = []
    for model in models:
        ring_size_ratio, mol_ring_ratio, mol_ring_no, fused_ring_ratio, mol_fused_ring_ratio = model.ring_size_profile
        tmp, tmp2 = [], []
        for s in ring_sizes:
            tmp.append(f'{ring_size_ratio[s] * 100:.1f}')
            tmp2.append(f'{mol_ring_ratio[s] * 100:.1f}')
        tmp3, tmp4 = [], []
        for s in fused_ring_sizes:
            tmp3.append(f'{fused_ring_ratio[s] * 100:.1f}')
            tmp4.append(f'{mol_fused_ring_ratio[s] * 100:.1f}')
        all_ring_size_ratio.append(model.name + '\t' + '\t'.join(tmp))
        all_mol_ring_ratio.append(model.name + '\t' + '\t'.join(tmp2))
        all_ring_no.append(model.name + '\t' + f'{np.mean(mol_ring_no):.1f}\t{np.median(mol_ring_no):.1f}')
        all_fused_ring_ratio.append(model.name + '\t' + '\t'.join(tmp3))
        all_mol_fused_ring_ratio.append(model.name + '\t' + '\t'.join(tmp4))

    print('n-member ring % \in all rings')
    print('\n'.join(all_ring_size_ratio))
    print('mol% \contain n-member ring')
    print('\n'.join(all_mol_ring_ratio))
    print('each mol contains no. rings')
    print('\n'.join(all_ring_no))
    print('rings% \are n-fused rings')
    print('\n'.join(all_fused_ring_ratio))
    print('mol% \contains n-fused rings')
    print('\n'.join(all_mol_fused_ring_ratio))

    exit(0)

    # for model in models:
    #     is_binding_success = model.is_binding_success(ref_rmsd=2)
    #     print(f'{model.name} binding success rate: {is_binding_success.mean() * 100:.1f}')

    # for model in models:
    #     is_binding_success = model.is_binding_success(ref_rmsd=1e5)
    #     print(f'{model.name} binding success rate w/o rmsd: {is_binding_success.mean() * 100:.1f}')
    # exit(0)

    # train = ModelResults('train', train_path)
    # train.load_affinity('../my_results/affinity_pose_check.pkl')
    # train.get_dist()

    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_columns', None)

    # ref.ring_size_profile
    print('\n\nEVALUATION')
    ModelResults.agg_print_results(models)
    exit(0)

    # bond length
    print('\n\nBOND LENGTH')
    metric_dict = collections.defaultdict(list)     
    for model in models[1:]:
        _metrics = eval_bond_length.eval_bond_length_profile(
            ref.bond_length_profile, model.bond_length_profile)

        for k, v in _metrics.items():
            metric_dict[k].append(v)

    df = pd.DataFrame(metric_dict, index=[model.name for model in models[1:]])
    print(df)

    # bond angle
    print('\n\nBOND ANGLE')
    metric_dict = collections.defaultdict(list)     
    for model in models[1:]:
        _metrics = eval_bond_angle.eval_bond_angle_profile(
            ref.bond_angle_profile, model.bond_angle_profile)
        
        for k, v in _metrics.items():
            metric_dict[k].append(v)

    df = pd.DataFrame(metric_dict, index=[model.name for model in models[1:]])
    print(df)

    # torsion angle
    print('\n\nTORSION ANGLE')
    metric_dict = collections.defaultdict(list)
    for model in models[1:]:
        _metrics = eval_torsion_angle.eval_torsion_angle_profile(
            ref.torsion_angle_profile, model.torsion_angle_profile)

        for k, v in _metrics.items():
            metric_dict[k].append(v)
    df = pd.DataFrame(metric_dict, index=[model.name for model in models[1:]])
    print(df)