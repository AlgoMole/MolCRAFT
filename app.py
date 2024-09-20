import glob
import numpy as np
import gradio as gr
from gradio_molecule3d import Molecule3D

import os
import shutil


def copy_file_from_tmp_to_input(tmp_file_path):
    if not os.path.exists('./input'):
      os.makedirs('./input', exist_ok=True)
    if not os.path.exists('./output'):
      os.makedirs('./output', exist_ok=True)
    dst = os.path.join('./input', os.path.basename(tmp_file_path))
    shutil.copyfile(tmp_file_path, dst)
    return dst

def load(dropdown_value: str, upload_value: list):
    pdb_path = dropdown_value
    sdf_path = glob.glob(pdb_path.rstrip('.pdb') + '*.sdf')
    # print(sdf_path)
    assert len(sdf_path) == 1
    sdf_path = sdf_path[0]

    if upload_value is not None and type(upload_value) == list and len(upload_value) == 2:
      flag = False
      if upload_value[0].endswith('.pdb') and upload_value[1].endswith('.sdf'):
        tmp_pdb_path, tmp_sdf_path = upload_value
        flag = True
      elif upload_value[0].endswith('.sdf') and upload_value[1].endswith('.pdb'):
        tmp_sdf_path, tmp_pdb_path = upload_value
        flag = True
      if flag:
        print(upload_value)
        pdb_path = copy_file_from_tmp_to_input(tmp_pdb_path)
        sdf_path = copy_file_from_tmp_to_input(tmp_sdf_path)

    print(pdb_path)
    print(sdf_path)
    return [pdb_path, sdf_path]

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
      "color": "cyanCarbon",
      # "color": "Chain",
    },
    {
      "model": 1,
      "style": "stick",
      # "color": "redCarbon",
      "color": "PyMol",
      "residue_range": "",
    },
    # {
    #   "model": 2,
    #   "style": "stick",
    #   "color": "greenCarbon",
    #   "residue_range": "",
    # }
]

from sample_for_pocket import call, OUT_DIR, Metrics, NpEncoder
# from rdkit import Chem
import json


def generate(dropdown_value: str, upload_value: list):
    protein_path, ligand_path = load(dropdown_value, upload_value)
    call(protein_path, ligand_path)
    
    out_fns = sorted(glob.glob(f'{OUT_DIR}/*.sdf'))
    return gr.update(choices=out_fns, value=out_fns[0])
    # return out_fns


def show(dropdown_value: str, upload_value: list, out_fn: str):
    protein_path, ligand_path = load(dropdown_value, upload_value)
    # sdf_mol = Chem.SDMolSupplier(out_fn, removeHs=False)[0]
    # # get all properties from sdf_mol
    # props = sdf_mol.GetPropsAsDict()

    return [protein_path, out_fn]
    

def evaluate(dropdown_value: str, upload_value: list, out_fn: str):
    protein_path, ligand_path = load(dropdown_value, upload_value)
    metrics = Metrics(protein_path, ligand_path, out_fn).evaluate()
    return json.dumps(metrics, indent=4, cls=NpEncoder)



with gr.Blocks(
      title='MolCRAFT',
      css=".gradio-container, .gradio-container button {} footer {visibility: hidden}"
    ) as demo:
    gr.Markdown("# MolCRAFT: Structure-Based Drug Design in Continuous Parameter Space [ICML 2024]")
    with gr.Row():
        dropdown = gr.Dropdown(label="Option 1: choose a pdb from CrossDocked test set:", choices=pdb_files, value=np.random.choice(pdb_files))
        upload = gr.UploadButton("Option 2: upload two files that form an aligned complex (a protein pdb & a ligand sdf to clip the pocket)", file_count="multiple")
  
    # out_ligand = Molecule3D(label='reference molecule', reps=reps)

    btn1 = gr.Button("Visualize reference ligand in complex")
    ref_complex = Molecule3D(label="Protein Pocket & Reference Ligand", reps=reps)
    btn1.click(load, inputs=[dropdown, upload], outputs=ref_complex)

    btn2 = gr.Button('Generate 10 ligands (~30s)')
    OUT_FILES = [f'./output/{i}.sdf' for i in range(10)]
    candidates = gr.Dropdown(label="Choose a generated molecule:", choices=OUT_FILES, value=OUT_FILES[0], interactive=True)
    btn2.click(generate, inputs=[dropdown, upload], outputs=[candidates])

    btn3 = gr.Button('Visualize generated ligand in complex (run "generate" first)')
    gen_complex = Molecule3D(label='Generated Molecule', reps=reps)
    btn3.click(show, inputs=[dropdown, upload, candidates], outputs=[gen_complex])

    btn4 = gr.Button('Evaluate this generated ligand (1-2 minutes)')
    metrics = gr.Textbox(label='Metrics')
    btn4.click(evaluate, inputs=[dropdown, upload, candidates], outputs=[metrics])

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

    gr.Markdown("<a href='https://clustrmaps.com/site/1c0i3'  title='Visit tracker'><center><img src='//clustrmaps.com/map_v2.png?cl=ffffff&w=a&t=tt&d=TuKguAwDVF-AYoVCxeGN0dyAr5mp9qWMBD20OvyLtCc' width='30%'/></center></a>")
    
if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=10990, favicon_path="favicon.png")
