from typing import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import os.path as osp
import functools
import sys
from collections import defaultdict
from pprint import pprint

import torch
from tqdm import tqdm
from posecheck.utils.chem import remove_radicals
from posebusters import PoseBusters
import fire

from eval.aio import (
    _listdir, get_mol, suppress_output,
    compute_buster, compute_vina_score, compute_strain,
    get_rmsd_between_mol_pdbqt,
)

def update_worker(pt:str, fn:Callable):
    try:
        mol_pt = torch.load(pt)
        if fn(mol_pt):
            torch.save(mol_pt, pt)
        return None
    except Exception as e:
        return {
            "error": e,
            "fn": pt[:-2]+"sdf",
        }

def update_mol(outputs:str, update_fn:Callable, cache:str="fail.pt", workers:int=1):
    """Update INDEX.pt in outputs folder.
    Args:
        outputs: the root folder follow defined sample structure
        update_fn: the function to update the mol_dict inplace
        workers: the number of workers. Default is 1 as sequential
    """
    failures = {}
    if workers == 1:
        mapper = map
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        mapper = pool.map
    tasks = [
        pt
        for pocket in _listdir(outputs, True, osp.isdir)
        for pt in _listdir(osp.join(pocket, "sdf"), True, lambda f: f.endswith(".pt"))
    ]

    results = mapper(functools.partial(update_worker, fn=update_fn), tasks)

    failures = {
        result["fn"]: result["error"]
        for result in tqdm(results, file=sys.stdout, total=len(tasks))
        if result
    }

    if workers != 1:
        pool.shutdown()

    print(f"Failures: {len(failures)}")
    # compute failure statistics
    failure_stats = {
        "type": defaultdict(int),
        "protein": defaultdict(int),
    }
    for key, val in failures.items():
        protein = key.split('/')[-3]
        val_type = type(val)
        failure_stats["protein"][protein] += 1
        failure_stats["type"][val_type] += 1
    pprint(failure_stats)
    if cache:
        torch.save({
            "cases": failures,
            "stats": failure_stats,
        }, osp.join(outputs, cache))

def update_cli(outputs:str, update_fn:str, cache:str=None, workers:int=1):
    updater = globals()[update_fn]
    update_mol(outputs, updater, cache, workers)

def get_tripair(mol_dict:dict):
    sdf_fn = mol_dict["ligand_filename"] # ../Benchmark/{model}/{outputs}/{protein}/sdf/INDEX.pt
    paths = sdf_fn.split('/')
    protein_name = paths[-3]
    # ../Benchmark/testset/{protein}/{protein}_pocket10A.pdb
    pair_root = osp.join(paths[0], paths[1], "testset", protein_name)
    protein_filename = osp.join(pair_root, f"{protein_name}_protein.pdb")
    ligand_filename = osp.join(pair_root, f"{protein_name}_ligand.sdf")
    return sdf_fn, protein_filename, ligand_filename

def update_vina(mol_dict:dict):
    # see INDEX_TEMPLATE
    if "vina" not in mol_dict or mol_dict["vina"]["score"] == 0.0:
        # redock
        sdf, protein, ligand = get_tripair(mol_dict)
        mol_dict["vina"] = compute_vina_score(protein, sdf)
        mol_dict["vina"]["dock_pose_pdbqt"] = get_rmsd_between_mol_pdbqt(get_mol(sdf), mol_dict["vina"]["pose"])
        return True
    return False

@suppress_output()
def update_rmsd(mol_dict:dict):
    sdf, protein, ligand = get_tripair(mol_dict)
    mol_dict["vina"]["dock_pose_pdbqt"] = get_rmsd_between_mol_pdbqt(get_mol(sdf), mol_dict["vina"]["pose"])
    return True

def update_pose(mol_dict:dict):
    # see INDEX_TEMPLATE
    # repose
    if "posebuster" in mol_dict \
        and "pb_volume_overlap_with_protein" in mol_dict["posebuster"] \
        and isinstance(mol_dict["posebuster"]["pb_volume_overlap_with_protein"], bool):
        return False
    sdf, protein, ligand = get_tripair(mol_dict)

    gen_mol = get_mol(sdf)
    gen_mol = remove_radicals(gen_mol)
    buster_result = compute_buster(gen_mol, PoseBusters("dock"), ligand, protein)
    mol_dict.update(buster_result)
    mol_dict["strain"] = compute_strain(gen_mol)
    
    return True

def update_strain(mol_dict:dict):
    sdf, protein, ligand = get_tripair(mol_dict)

    gen_mol = get_mol(sdf)
    gen_mol = remove_radicals(gen_mol)
    mol_dict["strain"] = compute_strain(gen_mol)
    
    return True

if __name__ == "__main__":
    fire.Fire(update_cli)