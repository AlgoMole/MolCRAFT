from rdkit import Chem
from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
# from torch_geometric.data import Data
from torch_scatter import scatter_mean
import numpy as np
# import torch
import os
from tqdm import tqdm
# import pickle as pkl
import json
import matplotlib
# import wandb
# import copy
# import glob
import shutil
import time

from core.evaluation.metrics import CondMolGenMetric
# from core.evaluation.utils import convert_atomcloud_to_mol_smiles, save_mol_list
# from core.evaluation.visualization import visualize, visualize_chain
# from core.utils import transforms as trans
# from core.evaluation.utils import timing
from core.callbacks.validation_callback import reconstruct_mol_and_filter_invalid

# this file contains the model which we used to visualize the

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# TODO: refactor and move center_pos (and that in train_bfn.py) into utils
def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein'):
    if mode == 'none':
        offset = 0.
        pass
    elif mode == 'protein':
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset


# add timestamp to the filename in the format of %Y-%m-%d-%H-%M-%S
timestr = time.strftime("%Y%m%d-%H%M%S")
OUT_DIR = f'./output-{timestr}'
# OUT_DIR = f'./output'
LAST_PROTEIN_FN = None


class DockingTestCallback(Callback):
    def __init__(self, dataset, atom_enc_mode, atom_decoder, atom_type_one_hot, single_bond, docking_config) -> None:
        super().__init__()
        self.dataset = dataset
        self.atom_enc_mode = atom_enc_mode
        self.atom_decoder = atom_decoder
        self.single_bond = single_bond
        self.type_one_hot = atom_type_one_hot
        self.docking_config = docking_config
        self.outputs = []
    
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        self.metric = CondMolGenMetric(
            atom_decoder=self.atom_decoder,
            atom_enc_mode=self.atom_enc_mode,
            type_one_hot=self.type_one_hot,
            single_bond=self.single_bond,
            docking_config=self.docking_config,
        )
    
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_test_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        self.outputs.extend(outputs)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_test_start(trainer, pl_module)
        self.outputs = []

    def on_test_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        super().on_test_epoch_end(trainer, pl_module)

        results, recon_dict = reconstruct_mol_and_filter_invalid(self.outputs)

        path = pl_module.cfg.accounting.test_outputs_dir
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

        if os.path.exists(OUT_DIR):
            shutil.rmtree(OUT_DIR)
        os.makedirs(OUT_DIR, exist_ok=True)

        for idx, res in enumerate(tqdm(results, desc="Chem eval")):
            
            try:
                mol = res['mol']
                ligand_filename = res['ligand_filename']
                mol.SetProp('_Name', ligand_filename)
            
                Chem.SanitizeMol(mol)
                smiles = Chem.MolToSmiles(mol)
                validity = smiles is not None
                complete = '.' not in smiles
            except:
                print('sanitize failed')
                continue

            if not validity or not complete:
                print('validity', validity, 'complete', complete)
                continue
                        
            # ligand_filename = graph.ligand_filename
            # ligand_dir = os.path.dirname(ligand_filename)
            # ligand_fn = os.path.basename(ligand_filename)
            # protein_fn = os.path.join(ligand_dir, ligand_fn[:10] + '.pdb')
    
            # print(json.dumps(chem_results, indent=4, cls=NpEncoder))

            out_fn = os.path.join(OUT_DIR, f'{idx}.sdf')
            with Chem.SDWriter(out_fn) as w:
                w.write(mol)

