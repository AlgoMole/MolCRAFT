import os.path as osp

import fire

from eval.stats import cli

def main(sample_root:str, savept:str=None, workers:int=1, type:str="pt", field:str=None, get_fn:str=None, pass_fn:str=None):
    tasks = [
        "AR/outputs_pocket10A",
        "Pocket2Mol/outputs_pocket10A",
        "TargetDiff/outputs_ref10A",
        "DecompDiff/outputs_ref10A",
        "MolCRAFT/outputs_ref10A",
        "Ours/outputs_ref10A", # ref
        "Ours/scheduler_ref",
    ]
    for task in tasks:
        print(f"Processing {task}...")
        cli(osp.join(sample_root, task), savept, workers, type, field, get_fn, pass_fn)
        print("\n")

if __name__ == "__main__":
    fire.Fire(main)