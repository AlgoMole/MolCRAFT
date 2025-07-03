from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List
import os.path as osp
import functools
import os
from collections import defaultdict
from pprint import pprint
from contextlib import contextmanager

import torch
import numpy as np
from tqdm import tqdm
import fire
from rdkit import Chem

from eval.aio import _listdir, noexcept

quanterify = functools.partial(np.percentile, q=[25, 50, 75])

data_config = {
    "AR": "outputs_pocket10A",
    "Pocket2Mol": "outputs_pocket10A",
    "TargetDiff": "outputs_ref10A",
    "DecompDiff": "outputs_ref10A",
    "MolCRAFT": "outputs_ref10A",
    "Ours": "outputs_ref10A",
}

@contextmanager
def mp_map(workers:int):
    if workers == 1:
        yield map
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        yield pool.map
        pool.shutdown()

def tanimoto_dis_N_to_1(mols: list, ref):
    sim = [1 - Chem.DataStructs.TanimotoSimilarity(m, ref) for m in mols]
    return sim

def compute_pocket_diversity(results: list):
    divs = []
    for i, mol in enumerate(results):
        tmp = tanimoto_dis_N_to_1(results, mol)
        tmp.pop(i)
        divs.extend(tmp)
    return divs

def compute_diversity(results: List[list], workers = 1):
    # results: List[List[Fingerprint]]
    diversity: List[float] = []

    with mp_map(workers) as mapper:
        for divs in tqdm(mapper(compute_pocket_diversity, results), "Computing diversity", total=len(results)):
            diversity += divs

    return {
        "avg": np.mean(diversity),
        "med": np.median(diversity)
    }


@noexcept({})
def get_meta(pt:str) -> dict:
    """Get posebuster result from output file.
    Args:
        output: the output file path
    """
    data = torch.load(pt)
    return data

@noexcept([])
def get_mols(sdfs: List[str]):
    results = []
    for sdf in sdfs:
        mol = Chem.MolFromMolFile(sdf)
        if mol is not None:
            results.append(Chem.RDKFingerprint(mol)) # Time costing
    return results

def stats_fetch(outputs:str, workers:int=1, file:str="pt", max_size:int=-1):
    """Get posebuster results from outputs folder.
    Args:
        outputs: the root folder follow defined sample structure
        workers: the number of workers. Default is 1 as sequential
    Return:
        "pt": list[dict]
        "sdf": list[list[Fingerprint]]
    """
    assert file in ["pt", "sdf"], "INDEX.pt or INDEX.sdf only!"
    results = []

    tasks = []
    for pocket in _listdir(outputs, True, osp.isdir):
        if not osp.exists(osp.join(pocket, "sdf")): continue
        if len(osp.basename(pocket)) != 8: continue
        pkt_result = []
        for pt in sorted(_listdir(osp.join(pocket, "sdf"), True, lambda x: x.endswith('.'+file))):
            pkt_result.append(pt)
        if max_size > 0 and max_size < len(pkt_result):
            pkt_result = pkt_result[:max_size]
        if file == "pt":
            tasks.extend(pkt_result)
        elif file == "sdf":
            tasks.append(pkt_result)

    if file == "pt":
        getter = get_meta
    elif file == "sdf":
        getter = get_mols

    with mp_map(workers) as mapper:
        for pkt in tqdm(mapper(getter, tasks), f"{file} loading" , total=len(tasks)):
            results.append(pkt)

    return results

def stats_run(metas:List[dict], getter:Callable, passer:Callable, calculator:Callable):
    """Run statistics on the given metas.
    Args:
        metas: the metas to run statistics on
        getter: the function to get the value from the meta
        passer: the function to filter the meta
        calculator: the function to calculate the statistics
    """
    items = filter(passer, map(getter, metas))

    stat_dict = {}
    stat_nan = {}

    items = list(items)

    for pb in tqdm(items, "stats", total=len(items)):
        for k,v in pb.items():
            if isinstance(v, float) and np.isnan(v):
                if k not in stat_nan:
                    stat_nan[k] = 0
                stat_nan[k] += 1
            else:
                if k not in stat_dict:
                    stat_dict[k] = []
                stat_dict[k].append(v)

    rate = {
        k: calculator(v)
        for k,v in stat_dict.items()
    }
    rate["count"] = {
        "raw": len(metas),
        "indeed": len(items),
    }
    return rate, stat_nan

def stats_show(stats:dict, stat_nan:dict, key:str=None):
    count = stats.pop("count")
    items = count["indeed"]
    if key:
        print(f"{key}:")
    prefix = '\t' if key else ''
    for k, v in stats.items():
        nan_str = f" (nan: {stat_nan[k]}/{items}|{stat_nan[k]/items:.3f})" if k in stat_nan else ""
        if isinstance(v, float):
            print(f"{prefix}{k}: {v:.4f} {nan_str}")
        else:
            print(f"{prefix}{k}: {v} {nan_str}")

def stats_show_batch(stats:dict):
    for k, v in stats.items():
        stats_show(v["stats"], v["nan"], k)

def stats_show_table(stats_all:dict):
    # banner = '\t'.join(stats.keys())
    header = ["model", "pb_valid", "vina score", "vina min", "vina dock", "rmsd<2A", "qed", "sa", "atoms"]
    table = ['\t'.join(header)]
    for model, stats in stats_all.items():
        # if not header:
        #     header = '\t'.join(stats.keys())
        #     table.append(header)
        content = [model]
        for field, data in stats.items():
            stat = data["stats"]
            stat.pop("count")
            for v in stat.values():
                content.append(f"{v:.3f}")
        table.append('\t'.join(content))
    print('\n'.join(table))


def process(
        outputs:str, workers:int, savept:str, fetch_type:str,
        getter:Callable, passer:Callable=None, calculator:Callable=np.mean
    ):
    metas = stats_fetch(outputs, workers, fetch_type)
    data, nand = stats_run(metas, getter, passer, calculator)
    stats_show(data, nand)

    if savept is None:
        save_file = None
    elif '/'  in savept:
        save_file = savept
    else:
        save_file = osp.join(outputs, savept)

    if save_file:
        torch.save({
            "stats": data,
            "nan": nand,
        }, save_file)


#region getter
def mol_getter(mols:list):
    return {
        "div": mols
    }
@noexcept({})
def rmsd_getter(meta:dict):
    # rmsd = meta.get("vina")
    if "vina" in meta:
        rmsd = meta["vina"]["dock_pose_pdbqt"]
    else:
        rmsd = np.nan

    return {
        "rmsd2": rmsd < 2,
        "pb_valid": meta["pb_valid"],
        "rmsd2 & pb": rmsd < 2 and meta["pb_valid"],
    }

mol_keys = ['mol_pred_loaded', 'mol_cond_loaded', 'sanitization', 'inchi_convertible', 'all_atoms_connected', 'bond_lengths', 'bond_angles', 'internal_steric_clash', 'aromatic_ring_flatness', 'double_bond_flatness', 'internal_energy']
dock_keys = ['protein-ligand_maximum_distance', 'minimum_distance_to_protein', 'minimum_distance_to_organic_cofactors', 'minimum_distance_to_inorganic_cofactors', 'minimum_distance_to_waters', 'volume_overlap_with_protein', 'volume_overlap_with_organic_cofactors', 'volume_overlap_with_inorganic_cofactors', 'volume_overlap_with_waters']
def pb_intra(meta:dict):
    pb = meta.get("posebuster", {})
    if pb:
        pb = {
            k[3:]: v
            for k,v in pb.items()
        }

        pb["valid"] = all(v for k,v in pb.items() if k in mol_keys)
    return pb

def pb_getter(meta:dict):
    pb = meta.get("posebuster", {})
    if pb:
        pb = {
            k[3:]: v
            for k,v in pb.items()
        }
        pb["valid"] = all(pb.values())
    return pb

@noexcept({})
def vina_getter(meta:dict):
    vina = meta["vina"]
    keys = ["score", "minimize", "dock"]
    return {
        key: vina[key]
        for key in keys
    }

@noexcept({})
def vina_nan_getter(meta:dict):
    vina = meta["vina"]
    keys = ["score", "minimize", "dock"]
    return {
        key: vina[key] if vina[key] < 0 else np.nan
        for key in keys
    }

@noexcept({})
def vina_zero_getter(meta:dict):
    vina = meta["vina"]
    keys = ["score", "minimize", "dock"]
    return {
        key: vina[key] if vina[key] < 0 else 0
        for key in keys
    }

@noexcept({})
def lbe_getter(meta:dict):
    vina_dock = meta["vina"]["dock"]

    return {
        "lbe": vina_dock / meta["atoms"],
        "lbe_nan": vina_dock / meta["atoms"] if vina_dock < 0 else np.nan,
        "lbe_zero": vina_dock / meta["atoms"] if vina_dock < 0 else 0
    }

@noexcept({})
def field_getter(meta:dict, field:str):
    value = meta[field]
    if isinstance(value, dict):
        return value
    else:
        return {field: value}
nac = defaultdict(int)
def pb_error(meta:dict):
    pb = meta.get("posebuster", {})
    if pb and not isinstance(pb["pb_volume_overlap_with_protein"], bool):
        nac[type(pb["pb_volume_overlap_with_protein"])] += 1
        return {}
    return pb
#endregion getter

def vina_process(outputs:str, workers:int, savept:str=None):
    metas = stats_fetch(outputs, workers)

    results = {}

    for k, v in {
        "avg": np.mean,
        "med": np.median,
    }.items():
        data, nand = stats_run(metas, vina_getter, None, v)
        results[k] = {
            "stats": data,
            "nan": nand,
        }
        print(f"vina|{k}:")
    stats_show_batch(results)

    if savept is None:
        return
    elif '/'  in savept:
        save_file = savept
    else:
        save_file = osp.join(outputs, savept)

    torch.save(results, save_file)

def stats_all(outputs:str, workers:int, savept:str=None):
    metas = stats_fetch(outputs, workers)
    mols: List[List[Chem.Mol]] = stats_fetch(outputs, workers, "sdf")

    results = { "diversity": compute_diversity(mols) }

    # vina
    for k, v in {
        "avg": np.mean,
        "med": np.median,
    }.items():
        data, nand = stats_run(metas, vina_getter, None, v)
        results[f"vina/{k}"] = {
            "stats": data,
            "nan": nand,
        }
    
    # others
    for key in ["strain", "qed", "sa", "atoms", "posebuster", "rmsd"]:
        # getter 
        if key == "posebuster":
            getter = pb_getter
        elif key == "rmsd":
            getter = rmsd_getter
        else:
            getter = functools.partial(field_getter, field=key)
        
        if key == "strain":
            calc = quanterify
        else:
            calc = np.mean

        data, nand = stats_run(metas, getter, None, calc)
        results[key] = {
            "stats": data,
            "nan": nand,
        }

    stats_show_batch(results)

    if savept is None:
        return
    elif '/'  in savept:
        save_file = savept
    else:
        save_file = osp.join(outputs, savept)
    torch.save(results, save_file)


def stats_table(outputs_root:str, workers:int, savept:str=None, max_size:int=-1):
    data_results = {}
    def pb_valid(meta:dict):
        pb = meta.get("posebuster", {})
        if pb:
            pb = {
                k[3:]: v
                for k,v in pb.items()
            }
            pb["valid"] = all(pb.values())
        return { "pb_valid": pb.get("valid", .0) }
    @noexcept({})
    def rmsd(meta:dict):
        rmsd = meta["vina"]["dock_pose_pdbqt"] if "vina" in meta else np.nan
        return { "rmsd2": rmsd < 2 }
    for k,v in data_config.items():
        metas = stats_fetch(osp.join(outputs_root, k, v), workers, max_size=max_size)
        results = {}


        for key in ["posebuster", "vina", "rmsd", "qed", "sa", "atoms"]:
            # getter 
            if key == "posebuster":
                getter = pb_valid
            elif key == "rmsd":
                getter = rmsd
            elif key == "vina":
                getter = vina_getter
            else:
                getter = functools.partial(field_getter, field=key)

            calc = np.mean

            data, nand = stats_run(metas, getter, None, calc)
            results[key] = {
                "stats": data,
                "nan": nand,
            }
        data_results[k] = results
    stats_show_table(data_results)

def cap_filter(item: dict, size=10):
    return len(item["div"]) >= size

def cli_hack(outputs:str, workers:int, savept:str, field:str, max_size:int):
    if field == "vina":
        vina_process(outputs, workers, savept)
    elif field == "all":
        stats_all(outputs, workers, savept)
    elif field == "table":
        stats_table(outputs, workers, savept, max_size)
    else:
        return
    exit()


def cli(outputs:str, savept:str=None, workers:int=1, type:str="pt", field:str=None, get_fn:str=None, pass_fn:str=None, max_size:int=-1):
    assert field or get_fn, "Either `field` or `get_fn` must be specified."
    assert not (field and get_fn), "Only one of `field` or `get_fn` can be specified."
    calc = np.mean

    if field:
        cli_hack(outputs, workers, savept, field, max_size)

        if field == "strain":
            calc = quanterify
        getter = functools.partial(field_getter, field=field)
    else:
        getter = globals()[get_fn]
        if type == "sdf":
            calc = compute_diversity
    passer = globals()[pass_fn] if pass_fn else None

    process(outputs, workers, savept, type, getter, passer, calc)



if __name__ == "__main__":
    fire.Fire(cli)
# pprint(nac)