import os
import os.path as osp
import shutil
import sys
import time

class SimpleProgressBar:
    def __init__(self, iterable, desc=None, total=None, ncols=50):
        self.iterable = iterable
        self.desc = desc if desc is not None else ""
        self.total = total if total is not None else len(iterable)
        self.ncols = ncols
        self.start_time = time.time()

    def __iter__(self):
        for i, item in enumerate(self.iterable):
            yield item
            self._update(i + 1)
        self.close()

    def _update(self, current):
        elapsed_time = time.time() - self.start_time
        progress = current / self.total
        bar_length = int(self.ncols * progress)
        bar = '#' * bar_length + '-' * (self.ncols - bar_length)
        percent = int(100 * progress)
        eta = elapsed_time / progress - elapsed_time if progress > 0 else 0
        sys.stdout.write(f"\r{self.desc} |{bar}| {percent}% {elapsed_time:.1f}s/{eta:.1f}s")
        sys.stdout.flush()

    def close(self):
        sys.stdout.write("\n")
        sys.stdout.flush()

# import fire

name_map = {
    "targetdiff": "TargetDiff",
    "3D-Generative-SBDD": "AR",
    "DecompDiff": "DecompDiff",
    "Pocket2Mol": "Pocket2Mol",
}

output_map = {
    "TargetDiff": ["outputs_pocket10A", "outputs_ref10A"],
    "DecompDiff": ["outputs_pocket10A", "outputs_ref10A"],
    "AR": ["outputs_pocket10A"],
    "Pocket2Mol": ["outputs_pocket10A"],
}

def aggresive_sample_pt(outputs:str, target:str):
    for model in os.listdir(outputs):
        if model not in name_map:
            continue
        model_tgt = name_map[model]
        for output in output_map[model_tgt]:
            out_dir = osp.join(outputs, model, output)
            tgt_dir = osp.join(target, model_tgt, output)
            os.makedirs(tgt_dir, exist_ok=True)

            # copy {out_dir}/*/sample.pt to {tgt_dir}/*.pt
            proteins = os.listdir(out_dir)
            for protein in SimpleProgressBar(proteins, f"{model_tgt}/{output}"):
                pp = osp.join(out_dir, protein)
                src_pt = osp.join(pp, "sample.pt")
                dst_pt = osp.join(tgt_dir, f"{protein}.pt")
                if not (osp.isdir(pp) and osp.exists(src_pt)):
                    continue
                if osp.exists(dst_pt):
                    continue
                shutil.copyfile(src_pt, dst_pt)

def aggresive_sample_sdf(outputs:str, target:str):
    for model in os.listdir(outputs):
        if model not in name_map:
            continue
        model_tgt = name_map[model]
        for output in output_map[model_tgt]:
            out_dir = osp.join(outputs, model, output)
            tgt_dir = osp.join(target, model_tgt, output)
            os.makedirs(tgt_dir, exist_ok=True)

            # copy {out_dir}/*/sdf to {tgt_dir}/*/
            proteins = os.listdir(out_dir)
            for protein in SimpleProgressBar(proteins, f"{model_tgt}/{output}"):
                pp = osp.join(out_dir, protein)
                src_sdf = osp.join(pp, "sdf")
                dst_sdf = osp.join(tgt_dir, protein)
                if not osp.isdir(pp):
                    continue
                if osp.exists(dst_sdf):
                    continue
                shutil.copytree(src_sdf, dst_sdf)

if __name__ == "__main__":
    # fire.Fire(aggresive_samples)
    # aggresive_sample_pt("../Benchmark", osp.expanduser("~/local/baselines/"))
    aggresive_sample_sdf("../Benchmark", osp.expanduser("~/local/baselines/"))
    # for i in SimpleProgressBar(range(100), 100):
    #     print(i)