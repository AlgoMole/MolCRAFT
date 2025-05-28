import argparse
from os.path import join
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from copy import deepcopy

from core.utils import misc
from core.evaluation.utils import scoring_func
from core.evaluation.docking_qvina import QVinaDockingTask
from core.evaluation.docking_vina import VinaDockingTask
from multiprocessing import Pool
from functools import partial
from pebble import ProcessPool
import glob, os

RDLogger.DisableLog('rdApp.*')

def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')


def print_ring_ratio(all_ring_sizes, logger):
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')


def evaluate_one_file(file, eval_args):
    # if isinstance(file, str):
        # mol = Chem.MolFromMolFile(file, sanitize=False, removeHs=False)
    pose_check_results = {}
    if isinstance(file, Chem.Mol):
        mol = file
        ligand_fn = mol.GetProp('_Name')
    elif isinstance(file, str):
        suppl = Chem.SDMolSupplier(file, sanitize=False, removeHs=False)
        mol = suppl[0]
        ligand_fn = mol.GetProp('_Name')
        if mol.HasProp('strain'):
            strain = float(mol.GetProp('strain'))
            pose_check_results['strain'] = strain
        if mol.HasProp('clash'):
            clash = float(mol.GetProp('clash'))
            pose_check_results['clash'] = clash
    else:
        raise ValueError('file must be either a string or a rdkit.Chem.Mol object')
    smiles = Chem.MolToSmiles(mol)
    if mol is None or '.' in smiles:
        return None

    # chemical and docking check
    chem_results = scoring_func.get_chem(mol)
    if eval_args.docking_mode == 'qvina':
        vina_task = QVinaDockingTask.from_generated_mol(
            mol, ligand_fn, protein_root=eval_args.protein_root)
        vina_results = vina_task.run_sync()
    elif eval_args.docking_mode in ['vina_score', 'vina_dock']:
        vina_task = VinaDockingTask.from_generated_mol(
            mol, ligand_fn, protein_root=eval_args.protein_root, 
            pos=mol.GetConformer(0).GetPositions())
        score_only_results = vina_task.run(mode='score_only', exhaustiveness=eval_args.exhaustiveness)
        minimize_results = vina_task.run(mode='minimize', exhaustiveness=eval_args.exhaustiveness)
        vina_results = {
            'score_only': score_only_results,
            'minimize': minimize_results
        }
        if eval_args.docking_mode == 'vina_dock':
            docking_results = vina_task.run(mode='dock', exhaustiveness=eval_args.exhaustiveness)
            vina_results['dock'] = docking_results
    else:
        vina_results = None

    return {
        'mol': mol,
        'smiles': smiles,
        'ligand_filename': file,
        'chem_results': chem_results,
        'vina': vina_results,
        'pose_check': pose_check_results
    }   


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--mol_dir', type=str, default='./data')
    parser.add_argument('--outdir', type=str, default='outputs')
    parser.add_argument('--protein_root', type=str, default='./data/test_set')
    parser.add_argument('--docking_mode', type=str, default='vina_dock', choices=['qvina', 'vina_score', 'vina_dock', 'none'])
    parser.add_argument('--exhaustiveness', type=int, default=16)
    parser.add_argument('--sequential', default=False, action='store_true')

    eval_args, unparsed_args = parser.parse_known_args()

    Path(eval_args.outdir).mkdir(parents=True, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=eval_args.outdir)

    if 'FLAG' in eval_args.mol_dir:
        files = glob.glob(join(eval_args.mol_dir, '*/*.sdf'))
    else:
        files = glob.glob(join(eval_args.mol_dir, '*.sdf'))

    print(len(files), files[0])
    results = []

    if eval_args.sequential:
        molist = []
        for idx, file in tqdm(enumerate(files), total=len(files), desc='Load mols'):
            # use sdf supplier
            suppl = Chem.SDMolSupplier(file, sanitize=False, removeHs=False)
            mol = suppl[0]
            if mol is None: continue
            smiles = Chem.MolToSmiles(mol)
            if '.' in smiles: continue
            if not mol.HasProp('original_path'):
                if not mol.HasProp('_Name') or '.sdf' not in mol.GetProp('_Name'):
                    parent_path = Path(file).parent
                    # get the pocket_info.txt under parent_path
                    pocket_info_path = parent_path / 'pocket_info.txt'
                    with open(pocket_info_path, 'r') as f:
                        pocket_info = f.read().strip()
                        assert '.pdb' in pocket_info, f'pocket_info.txt should contain the pdb file name, but got {pocket_info}'
                    mol.SetProp('_Name', pocket_info)
                    mol.SetProp('ligand_filename', str(idx))
                else:
                    basepath = Path(file).stem
                    mol.SetProp('ligand_filename', basepath)
                mol.SetProp('original_path', file)
            if 'FLAG' in eval_args.mol_dir:
                try:
                    AllChem.UFFOptimizeMolecule(mol)
                except:
                    continue
            molist.append(mol)
        
        print(len(molist), molist[0].GetProp('_Name'), molist[0].GetProp('ligand_filename'))

        for mol in tqdm(molist, desc='Docking'):
            try:
                if os.path.exists(join(eval_args.outdir, f'{mol.GetProp("ligand_filename")}.sdf')):
                    print(f'{mol.GetProp("ligand_filename")}.sdf exists, skip')
                    continue
                result = evaluate_one_file(mol, eval_args)
                if result is None: continue
                # set vina score, vina min, vina dock
                if eval_args.docking_mode == 'vina_dock':
                    vina_dock = result['vina']['dock'][0]['affinity']
                    mol.SetProp('vina_dock', str(vina_dock))
                vina_score = result['vina']['score_only'][0]['affinity']
                vina_min = result['vina']['minimize'][0]['affinity']
                mol.SetProp('vina_score', str(vina_score))
                mol.SetProp('vina_minimize', str(vina_min))
                # write to sdf
                writer = Chem.SDWriter(join(eval_args.outdir, f'{mol.GetProp("ligand_filename")}.sdf'))
                writer.write(mol)
                results.append(result)
            except Exception as e:
                print(e)

    else:
        # set up progress bar
        progress_bar = tqdm(total=len(files), desc='Docking')

        # set up multiprocessing
        with ProcessPool(max_workers=4) as pool:
            future = pool.map(evaluate_one_file, files, [eval_args] * len(files), timeout=1000.0)
            iterator = future.result()
            while True:
                try:
                    result = next(iterator)
                    if result is not None:
                        # get mol and write to sdf
                        mol = result['mol']
                        file = result['ligand_filename']
                        idx = int(file.split('/')[-1].split('.')[0])

                        # set vina score, vina min, vina dock
                        if eval_args.docking_mode == 'vina_dock':
                            vina_dock = result['vina']['dock'][0]['affinity']
                            mol.SetProp('vina_dock', str(vina_dock))
                        vina_score = result['vina']['score_only'][0]['affinity']
                        vina_min = result['vina']['minimize'][0]['affinity']
                        mol.SetProp('vina_score', str(vina_score))
                        mol.SetProp('vina_min', str(vina_min))

                        # set pose check
                        pose_check_results = result['pose_check']
                        if pose_check_results:
                            for k, v in pose_check_results.items():
                                mol.SetProp(k, str(v))

                        # write to sdf
                        writer = Chem.SDWriter(join(eval_args.outdir, f'{idx}.sdf'))
                        writer.write(mol)
                        results.append(result)
                        progress_bar.update()
                except StopIteration:
                    break
                except Exception as e:
                    print(e)
    
    torch.save(results, join(eval_args.outdir, "eval_all.pt"))

    num_samples = len(files)
    n_complete = len(results)
    logger.info(f'Evaluate done! {num_samples} samples in total.')

    fraction_complete = n_complete / num_samples
    validity_dict = {
        'complete': fraction_complete
    }
    print_dict(validity_dict, logger)

    logger.info('Number of evaluated mols: %d' % (len(results)))

    qed = [r['chem_results']['qed'] for r in results]
    sa = [r['chem_results']['sa'] for r in results]
    logger.info('QED:   Mean: %.3f Median: %.3f' % (np.mean(qed), np.median(qed)))
    logger.info('SA:    Mean: %.3f Median: %.3f' % (np.mean(sa), np.median(sa)))
    if eval_args.docking_mode == 'qvina':
        vina = [r['vina'][0]['affinity'] for r in results]
        logger.info('Vina:  Mean: %.3f Median: %.3f' % (np.mean(vina), np.median(vina)))
    elif eval_args.docking_mode in ['vina_dock', 'vina_score']:
        vina_score_only = [r['vina']['score_only'][0]['affinity'] for r in results]
        vina_min = [r['vina']['minimize'][0]['affinity'] for r in results]
        logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
        logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))
        if eval_args.docking_mode == 'vina_dock':
            vina_dock = [r['vina']['dock'][0]['affinity'] for r in results]
            logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vina_dock), np.median(vina_dock)))

    # check ring distribution
    print_ring_ratio([r['chem_results']['ring_size'] for r in results], logger)

if __name__ == "__main__":
    main()