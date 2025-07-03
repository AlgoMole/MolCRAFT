"""source files:
core/evaluation/metrics.py#L199-224
test/eval_dock_vina.py
test/eval_buster.py

format: core/callback/validation_callback.py#1019

targetdiff/../evaluation_diffusion.py
foreach SDF:
{
    "ligand_filename",
    "vina": {"score", "minimize", "dock", "dock_pose_pdbqt"},
    "posebuster": {...},
    qed,
    sa,
    ...
}
"""


"""
save strategy:

pocket10A for prior sampling
ref10A for reference sampling

*sampling result*:
outputs_pocket10A
- sample.yml
- 6T88_MWQ
    - log.txt # logger file
    - sample.pt
    - sdf
        - 000.sdf
        - xxx.sdf
        - 089.sdf # the last one
- other_proteins
"""


"""
*eval result*:
outputs_pocket10A
- eval_stats.pt
- 6T88_MWQ
    - eval.pt # eval stats for this pocket
    - sdf
        - 000.pt # eval result for this mol
        - 089.pt # the last one
- other_proteins

For index.pt, see #INDEX_TEMPLATE

For eval.pt, see #EVAL_TEMPLATE

For eval_stats.pt, see #compute_stats
"""

import os, glob, time, functools, sys
import os.path as osp
from pprint import pprint
from typing import List, Union, Callable
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from collections import defaultdict

from rdkit import Chem, RDLogger
from rdkit.Chem import Mol, Crippen
from rdkit.Chem.QED import qed
from posecheck import PoseCheck
from posecheck.utils.chem import remove_radicals
from posecheck.utils.strain import calculate_strain_energy
from posebusters import PoseBusters
import torch
import numpy as np
# from pebble import ProcessPool
from tqdm import tqdm
import fire

from core.evaluation.utils.sascorer import compute_sa_score
from core.evaluation.utils.eval_rmsd import get_rmsd_between_mol_pdbqt
from core.evaluation.docking_qvina import QVinaDockingTask
from core.evaluation.docking_vina import VinaDockingTask
from eval.validate import validate_dict_format, create_from_template

RDLogger.DisableLog('rdApp.*')
# global vars from env #
FORCE_EVAL = bool(os.getenv("FORCE_EVAL", 0))
PROFILE = bool(os.getenv("PROFILE", 0))
ERROR_RETRY = bool(os.getenv("ERROR_RETRY", 0))
# VINA_TIMEOUT = 40

INDEX_TEMPLATE = {
    "ligand_filename": str,
    "vina": {
        "score": float,
        "minimize": float,
        "dock": float,
        "pose": str,
        "dock_pose_pdbqt": (float, np.float64),
    },
    "strain": float,
    "pb_valid": np.bool_,
    "posebuster": dict,
    "qed": float,
    "sa": float,
    "atoms": int,
    # "logp": float,
}

EVAL_TEMPLATE = {
    "samples": list,
    "stats": {
        "count": int,
        "protein": str,
        "vina": {
            "score": {"avg": np.float64, "med": np.float64, "data": list},
            "minimize": {"avg": np.float64, "med": np.float64, "data": list},
            "dock": {"avg": np.float64, "med": np.float64, "data": list},
        },
        "strain": {
            "data": list,
            "pass": list, # not nan strain energy
            "nan": int,
        },
        "pb_valid": np.int64,
        "rmsd2": int,
        "rmsd_not_nan": int,
        "size": np.float64,
        "qed": np.float64,
        "sa": np.float64,
    },
    "elapse": float,
}

DEFAULT_VINA_RESULT = {
    "score": 0.0,
    "minimize": 0.0,
    "dock": 0.0,
    "pose": "TimeoutError",
    "dock_pose_pdbqt": np.nan,
}

def _query_time():
    if PROFILE:
        return time.time()
    return 0

def _read_file(file: str, bin: bool = False) -> Union[str, bytes]:
    mode = "rb" if bin else "r"
    with open(file, mode) as f:
        return f.read()

def _listdir(folder:str, with_dir:bool=False, cond_fn:Callable=None):
    result = []
    if with_dir:
        joiner = osp.join
    else:
        joiner = lambda _, f: f
    for f in os.listdir(folder):
        fd = joiner(folder, f)
        if cond_fn is None or cond_fn(fd):
            result.append(fd)
    return result

def exists(files:Union[List[str], str]):
    if isinstance(files, str):
        return osp.exists(files)
    return all(osp.exists(f) for f in files)

def suppress_output(stream="stdout", redir:str=None):
    if redir == None:
        redir = os.devnull
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            redir_fd = open(redir, 'w')
            if stream == "stdout":
                orig = sys.stdout
                sys.stdout = redir_fd
            elif stream == "stderr":
                orig = sys.stderr
                sys.stderr = redir_fd
            else:
                raise ValueError(f"Unknown stream: {stream}")
            result = func(*args, **kwargs)
            if stream == "stdout":
                sys.stdout = orig
            elif stream == "stderr":
                sys.stderr = orig
            redir_fd.close()
            return result
        return wrapper
    return decorator

def noexcept(except_value):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                result = except_value
            return result
        return wrapper
    return decorator

def meaningfulize(value, default, judger:Callable=None):
    if judger:
        return value if judger(value) else default
    return value if value else default

def get_mol(gen_fn: str) -> Mol:
    suppl = Chem.SDMolSupplier(gen_fn, sanitize=False, removeHs=False)
    gen_mol = suppl[0]
    gen_mol.UpdatePropertyCache()
    return gen_mol

#region components computation
def compute_chem(gen_fn: Union[str, Mol]):
    gen_mol = gen_fn if isinstance(gen_fn, Mol) else get_mol(gen_fn)
    return {
        "qed": qed(gen_mol),
        "sa": compute_sa_score(gen_mol),
        "atoms": gen_mol.GetNumAtoms(),
        "logp": Crippen.MolLogP(gen_mol),
    }

@suppress_output() # Finally you SHUT UP
def compute_rmsd(gen_mol:Mol, pose:str) -> float:
    # np.float64 or np.nan == float
    return get_rmsd_between_mol_pdbqt(gen_mol, pose)

def compute_vina_score(
    protein_fn: str,
    gen_fn: str,
    docking_mode: str = "vina_dock",
    exhaustiveness: int = 16,
    verbose: bool = False,
) -> dict:
    gen_mol = get_mol(gen_fn)

    if docking_mode == "qvina":
        vina_task = QVinaDockingTask(_read_file(protein_fn), gen_mol)
        # vina_results = vina_task.run_sync()
        raise NotImplementedError()
    elif docking_mode in ["vina_score", "vina_dock"]:
        vina_task = VinaDockingTask(
            protein_fn, gen_mol,
            tmp_dir="/tmp",
            pos=gen_mol.GetConformer(0).GetPositions(),
        )
        score_only_results = vina_task.run(
            mode="score_only", exhaustiveness=exhaustiveness
        )
        minimize_results = vina_task.run(mode="minimize", exhaustiveness=exhaustiveness)
        vina_results = {
            "score": score_only_results[0]["affinity"],
            "minimize": minimize_results[0]["affinity"],
        }
        if docking_mode == "vina_dock":
            docking_results = vina_task.run(mode="dock", exhaustiveness=exhaustiveness)
            vina_results["dock"] = docking_results[0]["affinity"]
            vina_results["pose"] = docking_results[0]["pose"]
            vina_results["dock_pose_pdbqt"] = compute_rmsd(gen_mol, vina_results["pose"]) # np.float64 or np.nan
    else:
        vina_results = None
    if verbose and all([vina_results[k] < 0 for k in vina_results if k != "pose"]):
        print(f"score:{vina_results['score']}", end=", ")
        print(f"min:{vina_results['min']}", end=", ")
        print(f"dock:{vina_results['dock']}")
    return vina_results


def compute_strain(gen_mols: Union[List[Mol], Mol]):
    if isinstance(gen_mols, Mol):
        return meaningfulize(calculate_strain_energy(gen_mols), np.nan)
    return [meaningfulize(calculate_strain_energy(mol), np.nan) for mol in gen_mols]


@suppress_output()
def compute_buster(
    gen_mols: Union[List[Mol], Mol], pb: PoseBusters,
    ref_mol: Union[Mol, str] = None, protein_fn:str = None,
):
    if isinstance(gen_mols, Mol):
        gen_mols = [gen_mols]
    df = pb.bust(gen_mols, ref_mol, protein_fn)
    df.columns = ["pb_" + c for c in df.columns]
    pb_dict = df.iloc[0].to_dict()
    return {
        "pb_valid": df.iloc[0].all(),
        "posebuster": pb_dict,
    }

#endregion components computation
def __app_paths(key:str, path:str):
    for path in sys.path:
        if key in path:
            return
    sys.path.append(path)

def retrive_sample_ratio(sample_pt):
    __app_paths("Benchmark", "../Benchmark")
    rate = torch.load(sample_pt)["rate"]
    return rate # {"count", "recon", "complete"}


def compute_metrics(
    protein_fn: str,
    ligand_fn: str,
    sdf_fn: str,
    docking_mode: str = "vina_dock",
    exhaustiveness: int = 16,
    save: bool = True,
) -> dict:
    ptf = sdf_fn[:-3] + "pt"
    if osp.exists(ptf) and not FORCE_EVAL:
        try:
            result = torch.load(ptf)
            if result and \
                (("error" in result and not ERROR_RETRY)  or \
                    validate_dict_format(result, INDEX_TEMPLATE, sdf_fn)):
                return result
        except EOFError as e: # OutOfDisk
            pass
    result = {"ligand_filename": sdf_fn}
    try:
        time_start = _query_time()
        gen_mol = get_mol(sdf_fn)
        ref_mol = get_mol(ligand_fn)
        time_get_mol = _query_time()
        vina_result = compute_vina_score(
            protein_fn, sdf_fn, docking_mode, exhaustiveness
        )
        result["vina"] = vina_result
        time_vina = _query_time()

        gen_mol = remove_radicals(gen_mol)
        buster_result = compute_buster(gen_mol, PoseBusters("dock"), ref_mol, protein_fn)
        result.update(buster_result)
        result["strain"] = compute_strain(gen_mol)
        time_pose = _query_time()

        chem_res = compute_chem(gen_mol)
        result.update(chem_res)
        time_chem = _query_time()
    except Exception as e:
        result["error"] = e
    
    if PROFILE and "error" not in result:
        result["time"] = {
            "all": time_chem - time_start,
            "mol": time_get_mol - time_start,
            "vina": time_vina - time_get_mol,
            "pose": time_pose - time_vina,
            "chem": time_chem - time_pose,
        }
    
    if save:
        torch.save(result, ptf)
    return result


def compute_pocket_stats(pockets: list, task:str):
    pocket_info = [v for v in pockets if "error" not in v]
    stats = {
        "count": len(pocket_info),
        "protein": task,
    }
    if len(pocket_info) == 0:
        stats_template = create_from_template(EVAL_TEMPLATE["stats"])
        stats_template.update(stats)
        return stats_template
    if PROFILE:
        try:
            time_stats = {}
            recorder = [v["time"] for v in pocket_info if "time" in v]
            for key in ["mol", "vina", "pose", "chem"]:
                if len(recorder) == 0:
                    time_stats[key] = -1
                else:
                    time_stats[key] = np.mean([t[key] for t in recorder])
            if len(recorder) != 0:
                print(time_stats)
            stats["time"] = time_stats
        except Exception as e:
            stats["time"] = e
    # count = len(pocket_info) # 100
    # vina
    vina_data = {
        key: [
            0.0 if np.isnan(v["vina"][key]) else v["vina"][key]
            for v in pocket_info
        ]
        for key in ["score", "minimize", "dock"]
    }
    vina_stats = {
        key: {
            "avg": np.mean(vina_data[key]),
            "med": np.median(vina_data[key]),
            "data": vina_data[key],
        }
        for key in ["score", "minimize", "dock"]
    }

    rmsd = [
        v["vina"]["dock_pose_pdbqt"]
        for v in pocket_info
        if not np.isnan(v["vina"]["dock_pose_pdbqt"])
    ]
    rmsd2 = int(sum([v < 2 for v in rmsd]))

    # strain 25% 50% 75%
    strain_data = [s["strain"] for s in pocket_info]
    strain_vs = [s for s in strain_data if not np.isnan(s)]

    pb_valid = sum([v["pb_valid"] for v in pocket_info])

    size_avg = np.mean([v["atoms"] for v in pocket_info])
    qed = np.mean([v["qed"] for v in pocket_info])
    sa = np.mean([v["sa"] for v in pocket_info])

    stats.update({
        "vina": vina_stats,
        "strain": {
            "data": strain_data,
            "pass": strain_vs,
            "nan": len(strain_data) - len(strain_vs),
        },
        "pb_valid": pb_valid,
        "rmsd2": rmsd2,
        "rmsd_not_nan": len(rmsd),
        "size": size_avg,
        "qed": qed,
        "sa": sa,
    })


    return stats

def compute_stats(pocket_stats:list):
    """Compute the statistics for a benchmark sampling.
    Args:
        pocket_stats: list[dict[str, any]]
    """
    mole_count = sum([v["count"] for v in pocket_stats])

    vina_data = [[], [], []]
    for idx, key in enumerate(["score", "minimize", "dock"]):
        for v in pocket_stats:
            vina_data[idx].extend(v["vina"][key]["data"])

    vina_stats = {}
    for idx, key in enumerate(["score", "minimize", "dock"]):
        vina_stats[key] = {
            "avg": np.mean(vina_data[idx]),
            "med": np.median(vina_data[idx]),
        }


    strain_vs = [
        d
        for v in pocket_stats
        for d in v["strain"]["pass"]
    ]
    strain_quan = np.percentile(strain_vs, [25, 50, 75])
    strain_nan = sum(v["strain"]["nan"] for v in pocket_stats)

    pb_valid = sum(v["pb_valid"] for v in pocket_stats)

    rmsd2 = sum(v["rmsd2"] for v in pocket_stats)
    rmsd_not_nan = sum(v["rmsd_not_nan"] for v in pocket_stats)

    size_avg = sum(v["size"]*v["count"] for v in pocket_stats) / mole_count
    qed_avg = sum(v["qed"]*v["count"] for v in pocket_stats) / mole_count
    sa_avg = sum(v["sa"]*v["count"] for v in pocket_stats) / mole_count

    return {
        "vina": vina_stats,
        "strain": {
            "quarter": list(strain_quan),
            "pass": len(strain_vs),
            "nan": strain_nan,
        },
        "pb_valid": pb_valid,
        "rmsd2": rmsd2,
        "rmsd_not_nan": rmsd_not_nan,
        "size": size_avg,
        "qed": qed_avg,
        "sa": sa_avg,
        "count": mole_count,
    }

@noexcept({})
def eval_for_pocket(
    protein_fn: str,
    ligand_fn: str,
    sdf_folder: str,
    docking_mode: str = "vina_dock",
    exhaustiveness: int = 16,
) -> dict:
    eval_pt = osp.join(osp.dirname(sdf_folder), "eval.pt")
    if osp.exists(eval_pt) and not FORCE_EVAL:
        cache = torch.load(eval_pt)
        if validate_dict_format(cache, EVAL_TEMPLATE, eval_pt):
            return cache["stats"]
        # return torch.load(eval_pt)["stats"] # dict[str, any]
    sdf_fns = [sdf for sdf in os.listdir(sdf_folder) if sdf.endswith("sdf")]
    pocket_results = []
    eval_task = protein_fn.split('/')[-2]
    start_time, tasks = time.time(), len(sdf_fns)

    for sdf in tqdm(sdf_fns, f"[{eval_task}]"):
        pocket_results.append(
            compute_metrics(
                protein_fn,
                ligand_fn,
                osp.join(sdf_folder, sdf),
                docking_mode,
                exhaustiveness,
            )
        )
        # med_time = time.time()
        # res_time = (med_time - start_time)/(idx+1)*tasks
        # print(f"{eval_task}: {idx+1}/{tasks} | {res_time/60:.2f} min remaining [spent {med_time - start_time:.2f} s]")
    pocket_stats = compute_pocket_stats(pocket_results, eval_task)
    torch.save(
        {
            "samples": pocket_results,
            "stats": pocket_stats,
            "elapse": time.time() - start_time, # secs
        },
        eval_pt,
    )
    return pocket_stats

def resolve_paths(pair_paths:list, sdf_root:str):
    pro_fns, lig_fns, gen_fns = [], [], []
    miss_fns = []
    for path in pair_paths:
        if not osp.isdir(path): continue
        suf = osp.basename(path)
        pro_fn = osp.join(path, f"{suf}_protein.pdb")
        lig_fn = osp.join(path, f"{suf}_ligand.sdf")
        gen_fn = osp.join(sdf_root, suf, "sdf")

        if exists((gen_fn, pro_fn, lig_fn)):
            pro_fns.append(pro_fn)
            lig_fns.append(lig_fn)
            gen_fns.append(gen_fn)
        else:
            miss_fns.append(suf)
    return pro_fns, lig_fns, gen_fns


def parallel(
        pair_root:str = "/sharefs/share/molcraft/posebusters_benchmark_180/",
        sdf_root:str = "/sharefs/pdliu/space/Benchmark/targetdiff/outputs_pocket10A/",
        workers:int=8
    ):
    pair_paths = glob.glob(f"{pair_root}/*")
    # suffix is the basename of pair_path
    # with ProcessPool(max_workers=workers) as pool:
    pro_fns, lig_fns, gen_fns = resolve_paths(pair_paths, sdf_root)
    pocket_stats = []
    progress = tqdm(desc="Evaluating", total=len(pro_fns), file=sys.stdout)
    if workers == 1:
        for result in map(eval_for_pocket, pro_fns, lig_fns, gen_fns):
            pocket_stats.append(result)
            progress.update()
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
                future = pool.map(
                    eval_for_pocket,
                    pro_fns, lig_fns, gen_fns,
                    # timeout=2100.0, # 35 min
                    )
                # iterator = future.result()
                # for result in iterator:
                for stat in future:
                    progress.update()
                    if stat:
                        pocket_stats.append(stat)
    progress.close()
    # stats all
    try:
        stats = compute_stats(pocket_stats)
        torch.save(stats, osp.join(sdf_root, "eval_stats.pt"))
        pprint(stats)
    except Exception as e:
        print("Error occurs when compute statistics.")
        pprint(e)
    
def demo(pkt:str="7XJN_NSD", base:str="targetdiff", mode:str="pkt"):
    demo_root = "/sharefs/share/molcraft/posebusters_benchmark_180/"
    output = "pocket" if mode == "pkt" else "ref"
    targetdiff = f"../Benchmark/{base}/outputs_{output}10A/"
    start = time.time()
    result = eval_for_pocket(
        osp.join(demo_root, pkt, f"{pkt}_protein.pdb"),
        osp.join(demo_root, pkt, f"{pkt}_ligand.sdf"),
        osp.join(targetdiff, pkt, "sdf"),
    )
    end = time.time()
    print(f"Time: {end - start}s")

def show_stats(evalpt:str):
    data = torch.load(evalpt)
    stats = data["stats"]
    timer = data["elapse"]
    protein = osp.basename(osp.dirname(evalpt))
    print(f"[{protein}] took {timer}s to get eval results")
    print(stats)



if __name__ == "__main__":
    fire.Fire({
        "pocket": eval_for_pocket,
        "run": parallel,
        "demo": demo,
        "stats": show_stats,
    })
