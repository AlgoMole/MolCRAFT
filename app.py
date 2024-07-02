import glob
import numpy as np
import gradio as gr
from gradio_molecule3d import Molecule3D

import os


def load(value: str):
    full_pdb_path = value
    sdf_path = glob.glob(full_pdb_path.rstrip('.pdb') + '*.sdf')
    # print(sdf_path)
    assert len(sdf_path) == 1
    sdf_path = sdf_path[0]
    return [full_pdb_path, sdf_path]

pdb_files = glob.glob("./data/test_set/*/*.pdb")
pdb_files = [f for f in pdb_files if 'ligand' not in f and 'complex' not in f]
pdb_files = sorted(pdb_files)
pdb_files = [f for f in pdb_files if './data/test_set/' in f and '_tmp.pdb' not in f and '/pdb/' not in f]
pdb_files = [f for f in pdb_files if 'pdb/' not in f]
# print(pdb_files)
# exit(0)


reps = [
    {
      "model": 0,
      "style": "cartoon",
      "color": "whiteCarbon",
    },
    {
      "model": 1,
      "style": "stick",
      "color": "redCarbon",
      "residue_range": "",
    },
    # {
    #   "model": 2,
    #   "style": "stick",
    #   "color": "greenCarbon",
    #   "residue_range": "",
    # }
]

from sample_for_pocket_v2 import call, OUT_DIR, Metrics
# from rdkit import Chem
import json


def generate(value: str):
    protein_path, ligand_path = load(value)
    call(protein_path, ligand_path)
    
    out_fns = sorted(glob.glob(f'{OUT_DIR}/*.sdf'))
    return gr.update(choices=out_fns, value=out_fns[0])
    # return out_fns


def show(value: str, out_fn: str):
    protein_path, ligand_path = load(value)
    # sdf_mol = Chem.SDMolSupplier(out_fn, removeHs=False)[0]
    # # get all properties from sdf_mol
    # props = sdf_mol.GetPropsAsDict()

    return [protein_path, out_fn]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    

def evaluate(value: str, out_fn: str):
    protein_path, ligand_path = load(value)
    metrics = Metrics(protein_path, ligand_path, out_fn).evaluate()
    return json.dumps(metrics, indent=4, cls=NpEncoder)



with gr.Blocks() as demo:
    gr.Markdown("# MolCRAFT: Structure-Based Drug Design in Continuous Parameter Space [ICML 2024]")
    dropdown = gr.Dropdown(label="choose a pdb from CrossDocked test set:", choices=pdb_files, value=np.random.choice(pdb_files))
    ref_complex = Molecule3D(label="Protein Pocket & Reference Ligand", reps=reps)
    # out_ligand = Molecule3D(label='reference molecule', reps=reps)

    btn1 = gr.Button("visualize reference ligand in complex")
    btn1.click(load, inputs=dropdown, outputs=ref_complex)

    btn2 = gr.Button('generate')
    OUT_FILES = [f'./output/{i}.sdf' for i in range(10)]
    candidates = gr.Dropdown(label="choose a generated molecule:", choices=OUT_FILES, value=OUT_FILES[0], interactive=True)
    btn2.click(generate, inputs=[dropdown], outputs=[candidates])

    gen_complex = Molecule3D(label='Generated Molecule', reps=reps)
    btn3 = gr.Button('visualize generated ligand in complex (should run "generate" at first)')
    btn3.click(show, inputs=[dropdown, candidates], outputs=[gen_complex])

    metrics = gr.Textbox(label='metrics')
    btn4 = gr.Button('evaluate (this could be time consuming)')
    btn4.click(evaluate, inputs=[dropdown, candidates], outputs=[metrics])

    gr.Markdown(
"""
```
@article{qu2024molcraft,
  title={MolCRAFT: Structure-Based Drug Design in Continuous Parameter Space},
  author={Qu, Yanru and Qiu, Keyue and Song, Yuxuan and Gong, Jingjing and Han, Jiawei and Zheng, Mingyue and Zhou, Hao and Ma, Wei-Ying},
  journal={ICML 2024},
  year={2024}
}
@article{song2024unified,
  title={Unified Generative Modeling of 3D Molecules via Bayesian Flow Networks},
  author={Song, Yuxuan and Gong, Jingjing and Qu, Yanru and Zhou, Hao and Zheng, Mingyue and Liu, Jingjing and Ma, Wei-Ying},
  journal={ICLR 2024},
  year={2024}
}
```
"""
    )
    
if __name__ == '__main__':
    demo.launch(share=True)
