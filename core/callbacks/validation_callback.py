from rdkit import Chem
from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch_geometric.data import Data
from torch_scatter import scatter_mean
import numpy as np
import torch
import os
import tqdm
import pickle as pkl
import json
import matplotlib
import wandb
import copy
import glob

from core.evaluation.metrics import CondMolGenMetric
from core.evaluation.utils import convert_atomcloud_to_mol_smiles, save_molist
from core.evaluation.visualization import visualize, visualize_chain
from core.utils import transforms as trans
from core.evaluation.utils import timing

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


# TODO merge with ReconValidationCallback
class CondMolGenValidationCallback(Callback):
    def __init__(self, dataset, atom_enc_mode, atom_decoder, atom_type_one_hot, single_bond, docking_config) -> None:
        super().__init__()
        self.dataset = dataset
        self.atom_enc_mode = atom_enc_mode
        self.atom_decoder = atom_decoder
        self.single_bond = single_bond
        self.type_one_hot = atom_type_one_hot
        self.docking_config = copy.deepcopy(docking_config)
        self.docking_config.mode = 'vina_score'
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

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        self.outputs.extend(outputs)  # num_samples * ([num_atoms_i, 3], [num_atoms_i, num_atom_types])

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_start(trainer, pl_module)
        self.outputs = []

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        path = pl_module.cfg.accounting.test_outputs_dir
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        if os.path.exists(os.path.join(path, 'val_outputs.pt')):
            outputs_num = len(glob.glob(os.path.join(path, 'val_outputs*.pt')))
            version = f'-v{outputs_num}'
        else:
            version = ''
        
        raw_evaluation = self.metric.compute_raw_evaluation(self.outputs)
        torch.save(self.outputs, os.path.join(path, f'val_outputs{version}.pt'))
        torch.save(raw_evaluation, os.path.join(path, f'val_raw_evaluation{version}.pt'))

        mol_path = os.path.join(path, f'mols{version}')
        os.makedirs(mol_path, exist_ok=True)
        for i, graph in enumerate(self.outputs):                                     
            if 'mol' not in graph: continue
            mol = graph.mol                                                     
            mol.SetProp('_Name', graph.ligand_filename)
            if raw_evaluation['chem'][i]: 
                mol.SetProp('vina_score', str(raw_evaluation['chem'][i]['vina_score']))   
                mol.SetProp('vina_minimize', str(raw_evaluation['chem'][i]['vina_minimize']))
            with Chem.SDWriter(os.path.join(mol_path, f'{i}.sdf')) as writer:
                writer.write(mol)

        out_metrics = self.metric.evaluate(self.outputs, raw_evaluation)
        out_metrics = {f'val/{k}': v for k, v in out_metrics.items()}
        pl_module.log_dict(out_metrics)
        print(json.dumps(out_metrics, indent=4))


class MolVisualizationCallback(Callback):
    # here the call back, we save the molecules and also draw the figures also to the wandb.
    def __init__(self, atom_decoder, colors_dic, radius_dic, type_one_hot=False) -> None:
        super().__init__()
        self.outputs = []
        self.named_chain_outputs = {}
        self.atom_decoder = atom_decoder
        self.colors_dic = colors_dic
        self.radius_dic = radius_dic
        self.type_one_hot = type_one_hot

    @torch.no_grad()
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        self.outputs.extend(outputs)
        pl_module.eval()
        if len(self.named_chain_outputs['y']) == 0 and pl_module.cfg.visual.visual_chain:
            # normalize the position
            pos_normalizer = torch.tensor(
                pl_module.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=batch.protein_pos.device
            )
            batch.protein_pos = batch.protein_pos / pos_normalizer
            batch.ligand_pos = batch.ligand_pos / pos_normalizer

            # prepare batch data
            protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand = (
                batch.protein_pos, 
                batch.protein_atom_feature.float(), 
                batch.protein_element_batch, 
                batch.ligand_pos,
                batch.ligand_atom_feature_full, 
                batch.ligand_element_batch
            )

            # move protein center to origin & ligand correspondingly
            protein_pos, ligand_pos, offset = center_pos(
                protein_pos, ligand_pos, batch_protein, batch_ligand, mode=pl_module.cfg.dynamics.center_pos_mode) #TODO: ugly 
            num_graphs = batch_protein.max().item() + 1
    
            theta_chain, sample_chain, y_chain = pl_module.dynamics.sample(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
                n_nodes=num_graphs,
                ligand_pos=ligand_pos, # for debug only
                sample_steps=pl_module.cfg.evaluation.sample_steps,
                desc='MolVis',
            )

            # restore the protein position
            batch.protein_pos = batch.protein_pos * pos_normalizer

            for chain, chain_name in zip([theta_chain, sample_chain, y_chain], ['theta', 'sample', 'y']):
                for i in range(len(chain)):
                    pred_pos = chain[i][0]
                    one_hot = chain[i][1]
                    out_batch = copy.deepcopy(batch)
                    # restore the ligand position (in chain)
                    pred_pos = pred_pos * pos_normalizer

                    atom_type = one_hot.argmax(dim=-1)
                    # TODO: ugly, should be done in metrics.py (but needs a way to make it compatible with pyg batch)
                    atom_type = trans.get_atomic_number_from_index(atom_type, mode=pl_module.cfg.data.transform.ligand_atom_mode)
                    atom_type = [trans.MAP_ATOM_TYPE_ONLY_TO_INDEX[i] for i in atom_type]
                    atom_type = torch.tensor(atom_type, dtype=torch.long, device=ligand_pos.device)
                    out_batch.x, out_batch.pos = atom_type, pred_pos
                    _slice_dict = {
                        "x": out_batch._slice_dict["ligand_element"],
                        "pos": out_batch._slice_dict["ligand_pos"],
                    }
                    _inc_dict = {"x": out_batch._inc_dict["ligand_element"], "pos": out_batch._inc_dict["ligand_pos"]}
                    out_batch._inc_dict.update(_inc_dict)
                    out_batch._slice_dict.update(_slice_dict)
                    
                    out_data_list = out_batch.to_data_list()
                    self.named_chain_outputs[chain_name].append(
                        out_data_list[0]
                    )  # always append the first sampled dtat

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_start(trainer, pl_module)
        self.outputs = []
        self.named_chain_outputs = {"theta": [], "sample": [], "y": []}

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        super().on_validation_epoch_end(trainer, pl_module)

        with timing('saving mol chain'):
            epoch = pl_module.current_epoch

            # save mols
            if pl_module.cfg.visual.save_mols:
                path = os.path.join(pl_module.cfg.accounting.generated_mol_dir, str(epoch))
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                # we save the figures here.
                save_molist(
                    path=path,
                    molecule_list=self.outputs,
                    index2atom=self.atom_decoder,
                    type_one_hot=self.type_one_hot,
                )
                if pl_module.cfg.visual.visual_nums > 0:
                    images = visualize(
                        path=path,
                        atom_decoder=self.atom_decoder,
                        color_dic=self.colors_dic,
                        radius_dic=self.radius_dic,
                        max_num=pl_module.cfg.visual.visual_nums,
                    )
                    # table = [[],[]]
                    table = []
                    for p_ in images:
                        im = plt.imread(p_)
                        table.append(wandb.Image(im))
                        # if len(table[0]) < 5:
                        #     table[0].append(wandb.Image(im))
                        # else:
                        #     table[1].append(wandb.Image(im))
                    # pl_module.logger.log_table(key="epoch {}".format(epoch),data=table,columns= ['1','2','3','4','5'])
                    pl_module.logger.log_image(key="epoch_{}".format(epoch), images=table)
                    # wandb.log()
                    # update to wandb
            
            # save chains
            if pl_module.cfg.visual.visual_chain:
                # we save the chains and visual the gif here.
                columns = list(self.named_chain_outputs.keys())
                chain_gifs = []

                table = wandb.Table(columns=columns)
                for chain_name in columns:     
                    chain_path = os.path.join(
                        pl_module.cfg.accounting.generated_mol_dir, str(epoch), f"{chain_name}_chain"
                    )

                    if not os.path.exists(chain_path):
                        os.makedirs(chain_path, exist_ok=True)

                    save_molist(
                        path=chain_path,
                        molecule_list=self.named_chain_outputs[chain_name],
                        index2atom=self.atom_decoder,
                        type_one_hot=self.type_one_hot,
                    )
                    # if pl_module.cfg.visual.visual_nums > 0:
                    gif_path = visualize_chain(
                        path=chain_path,
                        atom_decoder=self.atom_decoder,
                        color_dic=self.colors_dic,
                        radius_dic=self.radius_dic,
                        spheres_3d=False,
                    )
                    gifs = wandb.Video(gif_path)
                    chain_gifs.append(gifs)
                
                pl_module.logger.log_table(
                    key="epoch_{}".format(epoch), data=[chain_gifs], columns=columns
                )
    
    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.on_validation_start(trainer, pl_module)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.on_validation_epoch_end(trainer, pl_module)


class ReconValidationCallback(Callback):
    # compute the BFN reconstruction loss for validation dataloader.
    def __init__(self, val_freq) -> None:
        super().__init__()
        self.val_freq = val_freq

    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        super().on_train_batch_end(
            trainer, pl_module, outputs, batch, batch_idx
        )
        
        if trainer.global_step % self.val_freq == 0: 
            # perform a full validation
            with torch.no_grad():
                # switch to eval mode
                pl_module.dynamics.eval()
                sum_batches, sum_loss, sum_loss_pos, sum_loss_type = 0, 0., 0., 0.
                pos_normalizer = torch.tensor(
                    pl_module.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=batch.protein_pos.device
                )

                for batch in trainer.val_dataloaders:
                    # prepare batch data
                    batch = batch.to(pl_module.device)

                    batch.protein_pos = batch.protein_pos / pos_normalizer
                    batch.ligand_pos = batch.ligand_pos / pos_normalizer

                    protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand = (
                        batch.protein_pos, 
                        batch.protein_atom_feature.float(), 
                        batch.protein_element_batch, 
                        batch.ligand_pos,
                        batch.ligand_atom_feature_full, 
                        batch.ligand_element_batch
                    )
                    # move protein center to origin & ligand correspondingly
                    protein_pos, ligand_pos, offset = center_pos(
                        protein_pos, ligand_pos, batch_protein, batch_ligand, mode=pl_module.cfg.dynamics.center_pos_mode) #TODO: ugly 
                    num_graphs = batch_protein.max().item() + 1
                    sum_batches += num_graphs * (pl_module.cfg.dynamics.discrete_steps // 10)
                    
                    # sample a random timestep for reconstruction loss computation
                    for t in range(0, pl_module.cfg.dynamics.discrete_steps, 10):
                        t = torch.tensor(
                            [t / float(pl_module.cfg.dynamics.discrete_steps)], 
                            dtype=ligand_pos.dtype, device=ligand_pos.device
                        ).repeat(num_graphs, 1).index_select(
                            0, batch_ligand
                        )  # [num_graphs, 1]

                        if not pl_module.cfg.dynamics.use_discrete_t and not pl_module.cfg.dynamics.destination_prediction:
                            t = torch.clamp(t, min=pl_module.dynamics.t_min)  # clamp t to [t_min,1]

                        # compute bfn loss  # TODO: convert to reconstruction loss
                        c_loss, d_loss, discretised_loss = pl_module.dynamics.reconstruction_loss_one_step(
                            t,
                            protein_pos=protein_pos,
                            protein_v=protein_v,
                            batch_protein=batch_protein,
                            ligand_pos=ligand_pos,
                            ligand_v=ligand_v,
                            batch_ligand=batch_ligand,
                        )
                        loss = torch.mean(c_loss + pl_module.cfg.train.v_loss_weight * d_loss + discretised_loss)
                        sum_loss += float(loss) * num_graphs
                        sum_loss_pos += float(c_loss.sum())
                        sum_loss_type += float(d_loss.sum())

                # log the mean reconstruction loss
                pl_module.log_dict({
                        "val/recon_loss": sum_loss / sum_batches,
                        "val/recon_loss_pos": sum_loss_pos / sum_batches,
                        "val/recon_loss_type": sum_loss_type / sum_batches,
                    }, 
                    on_step=True,
                    prog_bar=True, 
                    batch_size=pl_module.cfg.train.batch_size,
                )
                # print(f"step {trainer.global_step}: recon_loss: {sum_loss / sum_batches:.4f}, recon_loss_pos: {sum_loss_pos / sum_batches:.4f}, recon_loss_type: {sum_loss_type / sum_batches:.4f}")
                pl_module.dynamics.train()

    def on_train_batch_start(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            batch: Any,
            batch_idx: int,
            unused: int = 0,
        ) -> None:
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)


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

        with timing('docking'):
            path = pl_module.cfg.accounting.test_outputs_dir
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            if os.path.exists(os.path.join(path, 'outputs.pt')):
                outputs_num = len(glob.glob(os.path.join(path, 'outputs*.pt')))
                version = f'-v{outputs_num}'
            else:
                version = ''
            
            torch.save(self.outputs, os.path.join(path, f'outputs{version}.pt'))
            raw_evaluation = self.metric.compute_raw_evaluation(self.outputs)
            torch.save(raw_evaluation, os.path.join(path, f'raw_evaluation{version}.pt'))

            mol_path = os.path.join(path, f'mols{version}')
            os.makedirs(mol_path, exist_ok=True)
            for i, graph in enumerate(self.outputs):                                     
                if 'mol' not in graph: continue
                mol = graph.mol                                                     
                mol.SetProp('_Name', graph.ligand_filename)
                if raw_evaluation['chem'][i]: 
                    mol.SetProp('vina_score', str(raw_evaluation['chem'][i]['vina_score']))   
                    mol.SetProp('vina_minimize', str(raw_evaluation['chem'][i]['vina_minimize']))
                    if 'vina_dock' in raw_evaluation['chem'][i]:
                        mol.SetProp('vina_dock', str(raw_evaluation['chem'][i]['vina_dock']))
                with Chem.SDWriter(os.path.join(mol_path, f'{i}.sdf')) as writer:
                    writer.write(mol)

            out_metrics = self.metric.evaluate(self.outputs, raw_evaluation)
            out_metrics = {f'test/{k}': v for k, v in out_metrics.items()}
            print(json.dumps(out_metrics, indent=4))
            pl_module.log_dict(out_metrics)
