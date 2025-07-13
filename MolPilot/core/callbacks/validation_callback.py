from rdkit import Chem
from rdkit.Chem import SanitizeFlags
from collections import defaultdict
from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.utilities import rank_zero_only
from torch_geometric.data import Data
from torch_scatter import scatter_mean
import numpy as np
import torch
import torch.nn.functional as F
import os
from tqdm.auto import tqdm
import pickle as pkl
import json
import matplotlib
import wandb
import copy
import glob
import time
from scipy.ndimage import gaussian_filter

from core.models.time_warp import Timewarp
from core.evaluation.metrics import CondMolGenMetric, RMSDMetric
from core.evaluation.utils import convert_atomcloud_to_mol_smiles, save_molist
from core.evaluation.visualization import visualize, visualize_chain
from core.utils import transforms as trans
from core.evaluation.utils import timing
from core.utils.reconstruct import reconstruct_from_generated, reconstruct_from_generated_with_bond, reconstruct_from_generated_with_bond_basic, MolReconsError

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

# Function to analyze sanitize passing rates
def analyze_sanitize_flags(mol):
    """Analyze sanitize passing rates for each sanitize flag.

    Args:
        mol (Chem.Mol): RDKit molecule object.

    Returns:
        dict: Passing rates for each sanitize flag.
    """
    sanitize_flags = [
        SanitizeFlags.SANITIZE_CLEANUP,
        SanitizeFlags.SANITIZE_PROPERTIES,
        SanitizeFlags.SANITIZE_SYMMRINGS,
        SanitizeFlags.SANITIZE_KEKULIZE,
        SanitizeFlags.SANITIZE_SETAROMATICITY,
        SanitizeFlags.SANITIZE_SETCONJUGATION,
        SanitizeFlags.SANITIZE_SETHYBRIDIZATION,
        SanitizeFlags.SANITIZE_CLEANUPCHIRALITY
    ]

    flag2name = {
        SanitizeFlags.SANITIZE_CLEANUP: 'SANITIZE_CLEANUP'.lower(),
        SanitizeFlags.SANITIZE_PROPERTIES: 'SANITIZE_PROPERTIES'.lower(),
        SanitizeFlags.SANITIZE_SYMMRINGS: 'SANITIZE_SYMMRINGS'.lower(),
        SanitizeFlags.SANITIZE_KEKULIZE: 'SANITIZE_KEKULIZE'.lower(),
        SanitizeFlags.SANITIZE_SETAROMATICITY: 'SANITIZE_SETAROMATICITY'.lower(),
        SanitizeFlags.SANITIZE_SETCONJUGATION: 'SANITIZE_SETCONJUGATION'.lower(),
        SanitizeFlags.SANITIZE_SETHYBRIDIZATION: 'SANITIZE_SETHYBRIDIZATION'.lower(),
        SanitizeFlags.SANITIZE_CLEANUPCHIRALITY: 'SANITIZE_CLEANUPCHIRALITY'.lower()
    }

    passing_rates = {}

    for flag in sanitize_flags:
        mol_copy = Chem.Mol(mol)
        try:
            Chem.SanitizeMol(mol_copy, sanitizeOps=flag)
            passing_rates[flag2name[flag]] = 1
        except Exception as e:
            passing_rates[flag2name[flag]] = 0

    return passing_rates

def reconstruct_mol_and_filter_invalid(out_list, bond_bfn=False):
    results = []
    n_recon, n_recon_arom, n_recon_bond = 0, 0, 0
    n_complete = {'mol_basic': 0, 'mol_arom': 0, 'mol_bond': 0}
    n_valid = {'mol_basic': 0, 'mol_arom': 0, 'mol_bond': 0}
    n_total = len(out_list)
    mol_pos_range_list = []
    valid_dict = {}

    for item in out_list:
        ligand_filename, pos, atom_type = item.ligand_filename, item.pos, item.atom_type
        if hasattr(item, 'is_aromatic'):
            is_aromatic = item.is_aromatic.cpu().numpy().astype('bool').tolist()
        else:
            is_aromatic = None

        if bond_bfn:
            bond_type = item.bond.int().cpu().numpy().tolist()
        else:
            bond_type = None
        
        pos = pos.cpu().numpy().astype('float64')
        atom_type = atom_type.int().cpu().numpy().tolist()
        # TODO turn off basic_mode = False to use predicted aromaticity
        # try:
        if True:
            try:
                mol_basic = reconstruct_from_generated(pos, atom_type, is_aromatic, basic_mode=True)
                n_recon += 1
            except MolReconsError:
                mol_basic = None
            try:
                mol_arom = reconstruct_from_generated(pos, atom_type, is_aromatic, basic_mode=False)
                n_recon_arom += 1
            except MolReconsError:
                mol_arom = None
            if bond_type is not None:
                bond_index = item.bond_index.int().cpu().numpy().tolist()
                # assert all non-negative
                assert all([i[0] >= 0 and i[1] >= 0 for i in bond_index]), bond_index
                assert all([i >= 0 for i in bond_type]), bond_type
                try:
                    mol_bond = reconstruct_from_generated_with_bond_basic(pos, atom_type, bond_index, bond_type, check_validity=False)
                    n_recon_bond += 1
                except MolReconsError:
                    mol_bond = None
            else:
                mol_bond = None

            mol_pos_range = np.linalg.norm(pos.max(axis=0)[0] - pos.min(axis=0)[0])

            res = {
                'mol_basic': mol_basic, 'mol_arom': mol_arom, 'mol_bond': mol_bond, 'ligand_filename': ligand_filename, 
                'pred_pos': pos, 'pred_v': atom_type, 'is_aromatic': is_aromatic, 'mol_pos_range': mol_pos_range,
            }
                
            if bond_type is not None:
                res['mol'] = mol_bond
                res.update({'bond_type': bond_type, 'bond_index': bond_index})
            else:
                res['mol'] = mol_arom
            mol_pos_range_list.append(mol_pos_range)

            for mol, mol_key in zip([mol_basic, mol_arom, mol_bond], ['mol_basic', 'mol_arom', 'mol_bond']):
                suffix = mol_key.replace('mol', '')
                res[mol_key] = mol
                try:
                    # if mol_key == 'mol_bond' and mol is not None:
                    #     # use rdkit to check the validity of the molecule
                    #     # and stat different types of sanity checks

                    #     sanitize_flags = analyze_sanitize_flags(mol)
                    #     for flag, passing_rate in sanitize_flags.items():
                    #         valid_dict[flag] = valid_dict.get(flag, 0) + passing_rate

                    smiles = Chem.MolToSmiles(mol)
                    mol = Chem.MolFromSmiles(smiles)
                    complete = smiles is not None and '.' not in smiles
                    validity = mol is not None

                    # count the number of complete and valid molecules
                    # according to mol_key
                    n_complete[f'mol{suffix}'] += int(complete)
                    n_valid[f'mol{suffix}'] += int(validity)

                    res[f'smiles{suffix}'] = smiles                    
                    res[f'complete{suffix}'] = complete
                    res[f'validity{suffix}'] = validity
                except:
                    res[f'smiles{suffix}'] = None
                    res[f'complete{suffix}'] = False
                    res[f'validity{suffix}'] = False
            results.append(res)
        # except Exception as e:
        #     raise(e)
        #     continue

    if bond_bfn and n_recon_bond > 0:
        for k, v in valid_dict.items():
            valid_dict[k] = v / n_recon_bond

    # compute the complete rate and validity rate
    for key in ['mol_basic', 'mol_arom', 'mol_bond']:
        valid_dict[key.replace('mol', 'valid')] = n_valid[key] / n_total
        valid_dict[key.replace('mol', 'complete')] = n_complete[key] / n_total

    # print(json.dumps(valid_dict, indent=4))

    return results, {
        'recon_success': n_recon / n_total,
        'recon_bond_success': n_recon_bond / n_total,
        **valid_dict,
        'completeness': valid_dict['complete_bond' if bond_bfn else 'complete_arom'],
        'mol_pos_range': np.mean(mol_pos_range_list),
    }

def plot_weights_and_bins(timewarp):
    # Get learned weights and bin sizes
    edges_t_left, edges_t_right, edges_u_left, edges_u_right, slopes = timewarp.get_bins(
        invert=False, normalize=True
    )
    
    # Compute bin centers
    bin_centers = (edges_t_left + edges_t_right) / 2
    bin_centers = bin_centers[0].detach().cpu().numpy()
    
    # Extract weights
    weights_t = timewarp.logits_t.softmax(dim=1)[0].detach().cpu().numpy()
    weights_u = timewarp.logits_u.softmax(dim=1)[0].detach().cpu().numpy()
    
    # Extract bin edges
    edges_t_left = edges_t_left[0].detach().cpu().numpy()
    edges_t_right = edges_t_right[0].detach().cpu().numpy()

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot weights
    plt.plot(bin_centers, weights_t, label="Weights (t)", marker="o", linestyle="-", color="blue")
    plt.plot(bin_centers, weights_u, label="Weights (u)", marker="o", linestyle="--", color="orange")
    
    # Add vertical lines for bin edges
    for edge in edges_t_left:
        plt.axvline(edge, color="gray", linestyle=":", alpha=0.7, label="Bin Edges" if edge == edges_t_left[0] else "")
    
    # Add labels and legend
    plt.title("Weights and Bin Sizes with Respect to Warped Time")
    plt.xlabel("Warped Time (t')")
    plt.ylabel("Weights")
    plt.legend()
    plt.grid(alpha=0.3)
    

# TODO merge with ReconValidationCallback
class CondMolGenValidationCallback(Callback):
    def __init__(self, dataset, atom_enc_mode, atom_decoder, atom_type_one_hot, single_bond, docking_config, dataset_smiles_set=[]) -> None:
        super().__init__()
        self.dataset = dataset
        self.atom_enc_mode = atom_enc_mode
        self.atom_decoder = atom_decoder
        self.single_bond = single_bond
        self.type_one_hot = atom_type_one_hot
        if docking_config is not None:
            self.docking_config = copy.deepcopy(docking_config)
            self.docking_config.mode = 'vina_score'
        else:
            self.docking_config = None
        self.outputs = []
        self.dataset_smiles_set = dataset_smiles_set

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        self.metric = CondMolGenMetric(
            atom_decoder=self.atom_decoder,
            atom_enc_mode=self.atom_enc_mode,
            type_one_hot=self.type_one_hot,
            single_bond=self.single_bond,
            docking_config=self.docking_config,
            dataset_smiles_set=self.dataset_smiles_set,
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
        
        raw_evaluation = self.metric.compute_raw_evaluation(self.outputs, skip_chem=True)
        
        out_metrics = self.metric.evaluate(self.outputs, raw_evaluation)
        out_metrics = {f'val/{k}': v for k, v in out_metrics.items()}
        pl_module.log_dict(out_metrics, on_step=False, on_epoch=True, sync_dist=True)
        if trainer.global_rank == 0:
            print(json.dumps(out_metrics, indent=4))
            # wandb.log(out_metrics, step=trainer.global_step)


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
        if len(self.named_chain_outputs['y']) == 0 and pl_module.cfg.visual.visual_chain and trainer.global_rank == 0:

            # prepare batch data
            protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand = (
                getattr(batch, "protein_pos", None),
                batch.protein_atom_feature.float() if hasattr(batch, "protein_atom_feature") else None,
                getattr(batch, "protein_element_batch", None),
                batch.ligand_pos,
                batch.ligand_element_batch
            )
            
            # normalize the position
            pos_normalizer = torch.tensor(
                pl_module.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=batch.ligand_pos.device
            )
            ligand_pos = ligand_pos / pos_normalizer
            if protein_pos is not None:
                protein_pos = protein_pos / pos_normalizer
                # move protein center to origin & ligand correspondingly
                protein_pos, ligand_pos, offset = center_pos(
                    protein_pos, ligand_pos, batch_protein, batch_ligand, mode=pl_module.cfg.dynamics.center_pos_mode) #TODO: ugly 
            else:
                ligand_pos = torch.zeros_like(ligand_pos)
            
            num_graphs = batch_ligand.max().item() + 1
    
            theta_chain, sample_chain, y_chain = pl_module.dynamics.sample(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
                n_nodes=num_graphs,
                ligand_bond_index=batch.ligand_fc_bond_index,
                batch_ligand_bond=batch.ligand_fc_bond_type_batch,
                ligand_pos=ligand_pos, # for debug only
                sample_steps=pl_module.cfg.evaluation.sample_steps,
                desc='MolVis',
                include_protein=pl_module.include_protein,
            )

            # restore the protein position
            if protein_pos is not None:
                protein_pos = protein_pos * pos_normalizer

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
            if pl_module.cfg.visual.save_mols and trainer.global_rank == 0:
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
            if pl_module.cfg.visual.visual_chain and trainer.global_rank == 0:
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
                    wandb.log({f"chain_{chain_name}": gifs})

                pl_module.logger.log_table(
                    key="epoch_chain_{}".format(epoch), data=[chain_gifs], columns=columns
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
                sum_batches, sum_loss, sum_loss_pos, sum_loss_type, sum_loss_bond = 0, 0., 0., 0., 0.
                sum_loss_charge, sum_loss_aromatic = 0., 0.
                sum_recon_loss_pos, sum_recon_loss_type, sum_recon_loss_bond, sum_recon_loss_charge = 0., 0., 0., 0.
                pos_normalizer = torch.tensor(
                    pl_module.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=batch.ligand_pos.device
                )

                for batch in trainer.val_dataloaders:
                    # prepare batch data
                    batch = batch.to(pl_module.device)

                    protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand = (
                        getattr(batch, "protein_pos", None),
                        batch.protein_atom_feature.float() if hasattr(batch, "protein_atom_feature") else None,
                        getattr(batch, "protein_element_batch", None),
                        batch.ligand_pos,
                        batch.ligand_atom_feature_full,
                        batch.ligand_element_batch,
                    )
                    ligand_pos = ligand_pos / pos_normalizer
                    
                    if protein_pos is not None:
                        protein_pos = protein_pos / pos_normalizer
                        # move protein center to origin & ligand correspondingly
                        protein_pos, ligand_pos, offset = center_pos(
                            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=pl_module.cfg.dynamics.center_pos_mode) #TODO: ugly 
                    else:
                        ligand_pos = torch.zeros_like(ligand_pos)
                    num_graphs = batch_ligand.max().item() + 1
                    sum_batches += num_graphs * 10
                    
                    # use timestep 1 for reconstruction loss computation
                    N = pl_module.dynamics.discrete_steps
                    sigma1 = pl_module.dynamics.sigma1_coord
                    K = pl_module.dynamics.num_classes
                    beta1 = pl_module.dynamics.beta1
                    t = torch.tensor(
                        [1], 
                        dtype=ligand_pos.dtype, device=ligand_pos.device
                    ).repeat(num_graphs, 1).index_select(
                        0, batch_ligand
                    )

                    losses = pl_module.dynamics.reconstruction_loss_one_step(
                        t,
                        protein_pos=protein_pos,
                        protein_v=protein_v,
                        batch_protein=batch_protein,
                        ligand_pos=ligand_pos,
                        ligand_v=ligand_v,
                        batch_ligand=batch_ligand,
                        ligand_bond_type=batch.ligand_fc_bond_type,
                        ligand_bond_index=batch.ligand_fc_bond_index,
                        batch_ligand_bond=batch.ligand_fc_bond_type_batch,
                        include_protein=pl_module.include_protein,
                    )

                    pos_loss, type_loss, bond_loss, charge_loss, aromatic_loss = (
                        losses['closs'],
                        losses['dloss'],
                        losses['dloss_bond'],
                        losses['dloss_charge'],
                        losses['dloss_aromatic'],
                    )
                    
                    if pl_module.cfg.dynamics.use_discrete_t:
                        pos_weight = N * (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2))
                        # type_weight = 1
                    else:
                        pos_weight = -torch.log(sigma1) * torch.pow(sigma1, -2)
                        # type_weight = K * beta1
                    pos_weight = pos_weight.item()
                    # type_weight = type_weight.item()
                    pos_loss = pos_loss / pos_weight
                    # type_loss = type_loss / type_weight

                    if protein_pos is None:
                        pos_loss = torch.zeros_like(pos_loss)

                    sum_recon_loss_pos += float(pos_loss.sum()) * 10
                    sum_recon_loss_type += float(type_loss.sum()) * 10
                    sum_recon_loss_bond += float(bond_loss.sum()) * 10
                    sum_recon_loss_charge += float(charge_loss.sum()) * 10
                    # sample a random timestep for val loss computation
                    # Construct reversed u steps
                    u_steps = torch.linspace(1, 0, N + 1, device=pl_module.device, dtype=torch.float32)

                    timewarp_cdf = pl_module.timewarp_cdf
                    if timewarp_cdf is not None:
                        # Warp u to t' (sigma)
                        t_steps = timewarp_cdf(u_steps, invert=True).detach().to(torch.float32)
                        t_steps = (t_steps - timewarp_cdf.sigma_min) / (timewarp_cdf.sigma_max - timewarp_cdf.sigma_min)
                        # Reverse t' to get t = 1 - t'
                        t_steps = 1 - t_steps
                    else:
                        t_steps = 1 - u_steps
                        t_steps = t_steps.unsqueeze(-1).repeat(1, 2)
            
                    for i in range(1, N+1, N // 10):
                        # t = torch.tensor(
                        #     [i / float(N)], 
                        #     dtype=ligand_pos.dtype, device=ligand_pos.device
                        # ).repeat(num_graphs, 1).index_select(
                        #     0, batch_ligand
                        # )  
                        # [num_graphs, 1]
                        t = t_steps[i-1].repeat(num_graphs, 1).to(self.device)

                        if not pl_module.cfg.dynamics.use_discrete_t and not pl_module.cfg.dynamics.destination_prediction:
                            t = torch.clamp(t, min=pl_module.dynamics.t_min)  # clamp t to [t_min,1]

                        t, t_pos = t[:, 0].unsqueeze(-1), t[:, 1].unsqueeze(-1)

                        # compute bfn loss  # TODO: convert to reconstruction loss
                        losses = pl_module.dynamics.reconstruction_loss_one_step(
                            t,
                            protein_pos=protein_pos,
                            protein_v=protein_v,
                            batch_protein=batch_protein,
                            ligand_pos=ligand_pos,
                            ligand_v=ligand_v,
                            batch_ligand=batch_ligand,
                            ligand_bond_type=batch.ligand_fc_bond_type,
                            ligand_bond_index=batch.ligand_fc_bond_index,
                            batch_ligand_bond=batch.ligand_fc_bond_type_batch,
                            include_protein=pl_module.include_protein,
                            t_pos=t_pos
                        )

                        pos_loss, type_loss, bond_loss, charge_loss, aromatic_loss, discretized_loss = (
                            losses['closs'],
                            losses['dloss'],
                            losses['dloss_bond'],
                            losses['dloss_charge'],
                            losses['dloss_aromatic'],
                            losses['discretized_loss'],
                        )

                        if protein_pos is None:
                            pos_loss = torch.zeros_like(pos_loss)

                        # here the discretised_loss is close for current version.

                        loss = torch.mean(pos_loss + pl_module.cfg.train.v_loss_weight * type_loss + 
                                          pl_module.cfg.train.bond_loss_weight * bond_loss + 
                                          charge_loss + aromatic_loss + charge_loss)

                        sum_loss += float(loss) * num_graphs
                        sum_loss_pos += float(pos_loss.sum())
                        sum_loss_type += float(type_loss.sum())
                        sum_loss_bond += float(bond_loss.sum())
                        sum_loss_charge += float(charge_loss.sum())
                        sum_loss_aromatic += float(aromatic_loss.sum())

                # log the mean reconstruction loss
                val_dict = {
                    "val/recon_loss": sum_loss / sum_batches,
                    "val/recon_loss_pos": sum_loss_pos / sum_batches,
                    "val/recon_loss_type": sum_loss_type / sum_batches,
                    "val/recon_loss_bond": sum_loss_bond / sum_batches,
                    "val/recon_loss_charge": sum_loss_charge / sum_batches,
                    "val/recon_loss_aromatic": sum_loss_aromatic / sum_batches,
                    "val/recon_loss_pos_step1": sum_recon_loss_pos / sum_batches,
                    "val/recon_loss_type_step1": sum_recon_loss_type / sum_batches,
                    "val/recon_loss_bond_step1": sum_recon_loss_bond / sum_batches,
                    "val/recon_loss_charge_step1": sum_recon_loss_charge / sum_batches,
                }
                pl_module.log_dict(val_dict, 
                    on_step=True,
                    prog_bar=True, 
                    sync_dist=True,
                    batch_size=pl_module.cfg.train.batch_size,
                )
                # if trainer.global_rank == 0:
                #     wandb.log(val_dict, step=trainer.global_step)
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

def interpolate_schedule(u_values, schedule, device):
    """
    Interpolates the schedule to find corresponding t_2D, t_3D for given u values.
    
    Args:
        u_values (torch.Tensor): Tensor of u values to query.
        schedule (torch.Tensor): Optimal schedule of shape [N, 2] with columns (t_2D, t_3D).
        device (torch.device): The device for tensors.

    Returns:
        t_2D(torch.Tensor): Interpolated t_2D values for each u.
        t_3D(torch.Tensor): Interpolated t_3D values for each u.
    """
    # Extract schedule values
    schedule = schedule.to(device)
    t_schedule = torch.linspace(0, 1, schedule.shape[0], device=device)

    # Find indices for interpolation
    indices = torch.searchsorted(t_schedule, u_values.flatten(), right=True)
    indices = torch.clamp(indices, 1, len(t_schedule) - 1)

    # Interpolation
    t_2D = torch.zeros_like(u_values)
    t_3D = torch.zeros_like(u_values)
    u_flat = u_values.flatten()
    idx_minus_1 = indices - 1

    t_2D = schedule[idx_minus_1, 0] + (u_flat - t_schedule[idx_minus_1]) * (schedule[indices, 0] - schedule[idx_minus_1, 0])
    t_3D = schedule[idx_minus_1, 1] + (u_flat - t_schedule[idx_minus_1]) * (schedule[indices, 1] - schedule[idx_minus_1, 1])

    t_2D = t_2D.view_as(u_values)
    t_3D = t_3D.view_as(u_values)

    return t_2D, t_3D

class TwistedReconValidationCallback(Callback):
    # compute the BFN reconstruction loss for validation dataloader.
    def __init__(self, val_freq, use_scheduler=False) -> None:
        super().__init__()
        self.val_freq = val_freq
        self.use_scheduler = use_scheduler

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
        if hasattr(pl_module.cfg.train, 'val_mode') and pl_module.cfg.train.val_mode == 'loss':
            return
        
        if trainer.global_step % self.val_freq == 0: 
            # perform a full validation
            with torch.no_grad():
                # switch to eval mode
                pl_module.dynamics.eval()
                pos_normalizer = torch.tensor(
                    pl_module.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=pl_module.device
                )

                sum_batches, sum_loss = 0, 0
                recon_loss_partial_modal = {}
                loss_partial_modal = {}
                
                # use timestep 1 for reconstruction loss computation
                N = pl_module.dynamics.discrete_steps
                sigma1 = pl_module.dynamics.sigma1_coord
                K = pl_module.dynamics.num_classes
                beta1 = pl_module.dynamics.beta1

                for batch in trainer.val_dataloaders:
                    # prepare batch data
                    batch = batch.to(pl_module.device)

                    protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand = (
                        getattr(batch, "protein_pos", None),
                        batch.protein_atom_feature.float() if hasattr(batch, "protein_atom_feature") else None,
                        getattr(batch, "protein_element_batch", None),
                        batch.ligand_pos,
                        batch.ligand_atom_feature_full,
                        batch.ligand_element_batch,
                    )
                    ligand_pos = ligand_pos / pos_normalizer
                
                    if protein_pos is not None:
                        protein_pos = protein_pos / pos_normalizer
                        # move protein center to origin & ligand correspondingly
                        protein_pos, ligand_pos, offset = center_pos(
                            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=pl_module.cfg.dynamics.center_pos_mode) #TODO: ugly 
                    else:
                        ligand_pos = torch.zeros_like(ligand_pos)
                    num_graphs = batch_ligand.max().item() + 1
                    sum_batches += num_graphs * 10

                    t0 = torch.tensor([0], dtype=ligand_pos.dtype, device=pl_module.device).repeat(num_graphs, 1)[batch_ligand]
                    t1 = torch.tensor([1], dtype=ligand_pos.dtype, device=pl_module.device).repeat(num_graphs, 1)[batch_ligand]

                    for (t_discrete, t_pos, key) in [(t0, t1, 'w/o 2D'), (t1, t0, 'w/o 3D'), (t1, t1, 'sync')]:
                        losses = pl_module.dynamics.reconstruction_loss_one_step(
                            t_discrete,
                            protein_pos=protein_pos,
                            protein_v=protein_v,
                            batch_protein=batch_protein,
                            ligand_pos=ligand_pos,
                            ligand_v=ligand_v,
                            batch_ligand=batch_ligand,
                            ligand_bond_type=batch.ligand_fc_bond_type,
                            ligand_bond_index=batch.ligand_fc_bond_index,
                            batch_ligand_bond=batch.ligand_fc_bond_type_batch,
                            include_protein=pl_module.include_protein,
                            t_pos=t_pos
                        )

                        pos_loss, type_loss, bond_loss, charge_loss, aromatic_loss = (
                            losses['closs'],
                            losses['dloss'],
                            losses['dloss_bond'],
                            losses['dloss_charge'],
                            losses['dloss_aromatic'],
                        )
                        
                        if pl_module.cfg.dynamics.use_discrete_t:
                            pos_weight = N * (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2))
                            # type_weight = 1
                        else:
                            pos_weight = -torch.log(sigma1) * torch.pow(sigma1, -2)
                            # type_weight = K * beta1
                        pos_weight = pos_weight.item()
                        # type_weight = type_weight.item()
                        pos_loss = pos_loss / pos_weight
                        # type_loss = type_loss / type_weight

                        if protein_pos is None:
                            pos_loss = torch.zeros_like(pos_loss)

                        if key not in recon_loss_partial_modal:
                            recon_loss_partial_modal[key] = {
                                'pos': 0.,
                                'type': 0.,
                                'bond': 0.,
                            }
                        
                        recon_loss_partial_modal[key]['pos'] += float(pos_loss.sum()) * 10
                        recon_loss_partial_modal[key]['type'] += float(type_loss.sum()) * 10
                        recon_loss_partial_modal[key]['bond'] += float(bond_loss.sum()) * 10

                    # sample a random timestep for val loss computation
                    # i = torch.randint(0, N, (1,)).item()

                    for i in range(0, N, N // 10):
                        u = torch.tensor([i / float(N)], dtype=ligand_pos.dtype, device=ligand_pos.device).repeat(num_graphs, 1)


                        if self.use_scheduler:
                            t, t_3d = interpolate_schedule(u, self.time_scheduler, device=u.device)
                        else:
                            t, t_3d = u, u


                        if not pl_module.cfg.dynamics.use_discrete_t and not pl_module.cfg.dynamics.destination_prediction:
                            t = torch.clamp(t, min=pl_module.dynamics.t_min)  # clamp t to [t_min,1]
                            t_3d = torch.clamp(t_3d, min=pl_module.dynamics.t_min)  # clamp t to [t_min,1]

                        # compute bfn loss  # TODO: convert to reconstruction loss
                        # ablate 2D
                        # for (t_discrete, t_pos, key) in [(t, t, 'sync')]:

                        for (t_discrete, t_pos, key) in [(t0, t_3d, 'w/o 2D'), (t1, t_3d, 'w/ 2D'), (t, t0, 'w/o 3D'), (t, t1, 'w/ 3D'), (t, t_3d, 'sync')]:
                            losses = pl_module.dynamics.reconstruction_loss_one_step(
                                t_discrete,
                                protein_pos=protein_pos,
                                protein_v=protein_v,
                                batch_protein=batch_protein,
                                ligand_pos=ligand_pos,
                                ligand_v=ligand_v,
                                batch_ligand=batch_ligand,
                                ligand_bond_type=batch.ligand_fc_bond_type,
                                ligand_bond_index=batch.ligand_fc_bond_index,
                                batch_ligand_bond=batch.ligand_fc_bond_type_batch,
                                include_protein=pl_module.include_protein,
                                t_pos=t_pos
                            )

                            pos_loss, type_loss, bond_loss, charge_loss, aromatic_loss = (
                                losses['closs'],
                                losses['dloss'],
                                losses['dloss_bond'],
                                losses['dloss_charge'],
                                losses['dloss_aromatic'],
                            )

                            if protein_pos is None:
                                pos_loss = torch.zeros_like(pos_loss)

                            # here the discretised_loss is close for current version.
                            loss = torch.mean(pos_loss + pl_module.cfg.train.v_loss_weight * type_loss + 
                                                pl_module.cfg.train.bond_loss_weight * bond_loss + 
                                                charge_loss + aromatic_loss + charge_loss)

                            if key not in loss_partial_modal:
                                loss_partial_modal[key] = {
                                    'pos': 0.,
                                    'type': 0.,
                                    'bond': 0.,
                                }
                            
                            loss_partial_modal[key]['pos'] += float(pos_loss.sum())
                            loss_partial_modal[key]['type'] += float(type_loss.sum())
                            loss_partial_modal[key]['bond'] += float(bond_loss.sum())

                            # here the discretised_loss is close for current version.
                            if key == 'sync':
                                loss = torch.mean(pos_loss + pl_module.cfg.train.v_loss_weight * type_loss + 
                                                pl_module.cfg.train.bond_loss_weight * bond_loss + 
                                                charge_loss + aromatic_loss + charge_loss)

                                sum_loss += float(loss) * num_graphs

                # access the loss
                sum_loss_pos = loss_partial_modal['sync']['pos']
                sum_loss_type = loss_partial_modal['sync']['type']
                sum_loss_bond = loss_partial_modal['sync']['bond']

                sum_loss_pos_given_2d = loss_partial_modal['w/ 2D']['pos']
                sum_loss_pos_wo_2d = loss_partial_modal['w/o 2D']['pos']
                sum_loss_type_given_3d = loss_partial_modal['w/ 3D']['type']
                sum_loss_bond_given_3d = loss_partial_modal['w/ 3D']['bond']
                sum_loss_type_wo_3d = loss_partial_modal['w/o 3D']['type']
                sum_loss_bond_wo_3d = loss_partial_modal['w/o 3D']['bond']

                sum_recon_loss_pos_wo_2d = recon_loss_partial_modal['w/o 2D']['pos']
                sum_recon_loss_type_wo_3d = recon_loss_partial_modal['w/o 3D']['type']
                sum_recon_loss_bond_wo_3d = recon_loss_partial_modal['w/o 3D']['bond']

                sum_recon_loss_pos = recon_loss_partial_modal['sync']['pos']
                sum_recon_loss_type = recon_loss_partial_modal['sync']['type']
                sum_recon_loss_bond = recon_loss_partial_modal['sync']['bond']

                # log the mean reconstruction loss
                val_dict = {
                    # likelihood averaged over t [0, 1)
                    "val/recon_loss": sum_loss / sum_batches,
                    "val/recon_loss_pos": sum_loss_pos / sum_batches,
                    "val/recon_loss_type": sum_loss_type / sum_batches,
                    "val/recon_loss_bond": sum_loss_bond / sum_batches,
                    # t=1 reconstruction loss (trivial)
                    "val/recon_loss_pos_step1": sum_recon_loss_pos / sum_batches,
                    "val/recon_loss_type_step1": sum_recon_loss_type / sum_batches,
                    "val/recon_loss_bond_step1": sum_recon_loss_bond / sum_batches,
                    # likelihood averaged over t, but given ground truth 2D / 3D
                    "val/recon_loss_pos_given_2d": sum_loss_pos_given_2d / sum_batches,
                    "val/recon_loss_type_given_3d": sum_loss_type_given_3d / sum_batches,
                    "val/recon_loss_bond_given_3d": sum_loss_bond_given_3d / sum_batches,
                    # likelihood averaged over t, but without ground truth 2D / 3D
                    "val/recon_loss_pos_wo_2d": sum_loss_pos_wo_2d / sum_batches,
                    "val/recon_loss_type_wo_3d": sum_loss_type_wo_3d / sum_batches,
                    "val/recon_loss_bond_wo_3d": sum_loss_bond_wo_3d / sum_batches,
                    # t=1 reconstruction loss without ground truth 2D / 3D
                    "val/recon_loss_pos_step1_wo_2d": sum_recon_loss_pos_wo_2d / sum_batches,
                    "val/recon_loss_type_step1_wo_3d": sum_recon_loss_type_wo_3d / sum_batches,
                    "val/recon_loss_bond_step1_wo_3d": sum_recon_loss_bond_wo_3d / sum_batches,
                }

                pl_module.log_dict(val_dict, 
                    on_step=True,
                    prog_bar=True, 
                    sync_dist=True,
                    batch_size=pl_module.cfg.train.batch_size,
                )

                if trainer.global_rank == 0:
                    if protein_pos is None:
                        # del the 3d related loss
                        for key in ['val/recon_loss_pos_given_2d', 'val/recon_loss_type_given_3d', 'val/recon_loss_bond_given_3d',
                                    'val/recon_loss_pos_wo_2d', 'val/recon_loss_type_wo_3d', 'val/recon_loss_bond_wo_3d',
                                    'val/recon_loss_pos_step1_wo_2d', 'val/recon_loss_type_step1_wo_3d', 'val/recon_loss_bond_step1_wo_3d',
                                    'val/recon_loss_pos_step1', 'val/recon_loss_pos']:
                            del val_dict[key]
                    print(json.dumps(val_dict, indent=4))
                # if trainer.global_rank == 0:
                #     wandb.log(val_dict, step=trainer.global_step)
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

        if not hasattr(pl_module.cfg.train, 'val_mode') or pl_module.cfg.train.val_mode == 'sample':
            results, recon_dict = reconstruct_mol_and_filter_invalid(self.outputs, pl_module.dynamics.bond_bfn)

            epoch = pl_module.current_epoch
            val_outputs_dir = pl_module.cfg.accounting.test_outputs_dir.replace('test_outputs', 'val_outputs')
            path = os.path.join(val_outputs_dir, f'epoch_{epoch}')
            os.makedirs(path, exist_ok=True)
            torch.save(results, os.path.join(path, f'generated.pt'))
            
            out_metrics = {}
            out_metrics.update(recon_dict)
            out_metrics = {f'val/{k}': v for k, v in out_metrics.items()}
            pl_module.log_dict(out_metrics)
            print(json.dumps(out_metrics, indent=4))

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.on_validation_start(trainer, pl_module)
        # super().on_test_start(trainer, pl_module)
        # self.outputs = []
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # self.path = os.path.join(pl_module.cfg.accounting.test_outputs_dir, timestr)
    
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        # super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        # self.outputs.extend(outputs)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.on_validation_epoch_end(trainer, pl_module)
        # super().on_test_epoch_end(trainer, pl_module)
        # results, recon_dict = reconstruct_mol_and_filter_invalid(self.outputs, pl_module.dynamics.bond_bfn)

        # path = os.path.join(self.path, f'generated.pt')
        # os.makedirs(path, exist_ok=True)
        # torch.save(results, os.path.join(path, f'generated.pt'))
        
        # out_metrics = {}
        # out_metrics.update(recon_dict)
        # out_metrics = {f'test/{k}': v for k, v in out_metrics.items()}
        # pl_module.log_dict(out_metrics)
        # print(json.dumps(out_metrics, indent=4))



class TwistedMonitorCallback(Callback):
    def __init__(self, val_freq=1) -> None:
        super().__init__()
        self.val_freq = val_freq

    def on_train_batch_start(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            batch: Any,
            batch_idx: int,
            unused: int = 0,
    ) -> None:
        super().on_train_batch_start(
            trainer, pl_module, batch, batch_idx
        )
        
        if False: 
            if not os.path.exists(pl_module.cfg.accounting.test_outputs_dir):
                os.makedirs(pl_module.cfg.accounting.test_outputs_dir, exist_ok=True)
            # perform a full validation
            with torch.no_grad():
                # switch to eval mode
                pl_module.dynamics.eval()
                pos_normalizer = torch.tensor(
                    pl_module.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=pl_module.device
                )

                sum_batches, sum_loss = 0, 0
                loss_partial_modal = {}
                
                # use timestep 1 for reconstruction loss computation
                N = pl_module.dynamics.discrete_steps
                sigma1 = pl_module.dynamics.sigma1_coord
                K = pl_module.dynamics.num_classes
                beta1 = pl_module.dynamics.beta1
                if hasattr(pl_module.dynamics, 'num_bond_classes'):
                    E = pl_module.dynamics.num_bond_classes
                    beta1_bond = pl_module.dynamics.beta1_bond
                else:
                    E = 0
                    beta1_bond = 0
                interval = 100

                for batch in trainer.val_dataloaders:
                    # prepare batch data
                    batch = batch.to(pl_module.device)

                    protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand = (
                        getattr(batch, "protein_pos", None),
                        batch.protein_atom_feature.float() if hasattr(batch, "protein_atom_feature") else None,
                        getattr(batch, "protein_element_batch", None),
                        batch.ligand_pos,
                        batch.ligand_atom_feature_full,
                        batch.ligand_element_batch,
                    )
                    ligand_pos = ligand_pos / pos_normalizer
                
                    if protein_pos is not None:
                        protein_pos = protein_pos / pos_normalizer
                        # move protein center to origin & ligand correspondingly
                        protein_pos, ligand_pos, offset = center_pos(
                            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=pl_module.cfg.dynamics.center_pos_mode) #TODO: ugly 
                    num_graphs = batch_ligand.max().item() + 1
                    sum_batches += num_graphs

                    t0 = torch.tensor([1 / float(N)], dtype=ligand_pos.dtype, device=pl_module.device).repeat(num_graphs, 1)[batch_ligand]
                    t1 = torch.tensor([1], dtype=ligand_pos.dtype, device=pl_module.device).repeat(num_graphs, 1)[batch_ligand]

                    # sample a random timestep for val loss computation
                    # i = torch.randint(0, N, (1,)).item()
                    for i in range(0, N, N // interval):
                        t = torch.tensor([(i+1) / float(N)], dtype=ligand_pos.dtype, device=ligand_pos.device).repeat(num_graphs, 1)[batch_ligand]

                        if not pl_module.cfg.dynamics.use_discrete_t and not pl_module.cfg.dynamics.destination_prediction:
                            t = torch.clamp(t, min=pl_module.dynamics.t_min)  # clamp t to [t_min,1]

                        # compute bfn loss  # TODO: convert to reconstruction loss
                        # ablate 2D
                        # for (t_discrete, t_pos, key) in [(t, t, 'sync')]:

                        for (t_discrete, t_pos, key) in [(t0, t, 'w/o 2D'), (t1, t, 'w/ 2D'), (t, t0, 'w/o 3D'), (t, t1, 'w/ 3D'), (t, t, 'sync')]:
                            losses = pl_module.dynamics.reconstruction_loss_one_step(
                                t_discrete,
                                protein_pos=protein_pos,
                                protein_v=protein_v,
                                batch_protein=batch_protein,
                                ligand_pos=ligand_pos,
                                ligand_v=ligand_v,
                                batch_ligand=batch_ligand,
                                ligand_bond_type=batch.ligand_fc_bond_type,
                                ligand_bond_index=batch.ligand_fc_bond_index,
                                batch_ligand_bond=batch.ligand_fc_bond_type_batch,
                                include_protein=pl_module.include_protein,
                                t_pos=t_pos,
                                recon_loss=True
                            )

                            pos_loss, type_loss, bond_loss, charge_loss, aromatic_loss = (
                                losses['closs'],
                                losses['dloss'],
                                losses['dloss_bond'],
                                losses['dloss_charge'],
                                losses['dloss_aromatic'],
                            )

                            loss_mse, loss_type_ce, loss_bond_ce, loss_pos_cont, loss_type_cont, loss_bond_cont = (
                                losses['closs_mse'],
                                losses['dloss_ce'],
                                losses['dloss_bond_ce'],
                                losses['closs_cont'],
                                losses['dloss_cont'],
                                losses['dloss_bond_cont'],
                            )

                            # here the discretised_loss is close for current version.
                            loss = torch.mean(pos_loss + pl_module.cfg.train.v_loss_weight * type_loss + 
                                                pl_module.cfg.train.bond_loss_weight * bond_loss + 
                                                charge_loss + aromatic_loss + charge_loss)

                            if key not in loss_partial_modal:
                                loss_partial_modal[key] = {
                                    'pos': [0] * interval,
                                    'type': [0] * interval,
                                    'bond': [0] * interval,
                                    'pos_cont': [0] * interval,
                                    'type_cont': [0] * interval,
                                    'bond_cont': [0] * interval,
                                    'pos_mse': [0] * interval,
                                    'type_ce': [0] * interval,
                                    'bond_ce': [0] * interval,
                                }

                            # normalize i to [0, interval]
                            idx = int(i / (N / interval))

                            assert idx < len(loss_partial_modal[key]['pos']), f"{i} {len(loss_partial_modal[key]['pos'])}"
                            
                            loss_partial_modal[key]['pos'][idx] += float(pos_loss.mean())
                            loss_partial_modal[key]['type'][idx] += float(type_loss.mean())
                            loss_partial_modal[key]['bond'][idx] += float(bond_loss.mean())
                            loss_partial_modal[key]['pos_cont'][idx] += float(loss_pos_cont.mean())
                            loss_partial_modal[key]['type_cont'][idx] += float(loss_type_cont.mean())
                            loss_partial_modal[key]['bond_cont'][idx] += float(loss_bond_cont.mean())
                            loss_partial_modal[key]['pos_mse'][idx] += float(loss_mse.mean())
                            loss_partial_modal[key]['type_ce'][idx] += float(loss_type_ce.mean())
                            loss_partial_modal[key]['bond_ce'][idx] += float(loss_bond_ce.mean())

                            # here the discretised_loss is close for current version.
                            if key == 'sync':
                                loss = torch.mean(pos_loss + pl_module.cfg.train.v_loss_weight * type_loss + 
                                                pl_module.cfg.train.bond_loss_weight * bond_loss + 
                                                charge_loss + aromatic_loss + charge_loss)

                                sum_loss += float(loss) * num_graphs

                # normalize the loss
                for key in loss_partial_modal:
                    for idx in range(interval):
                        # pos_weight = N * (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * idx / interval))
                        pos_weight = 1
                        loss_partial_modal[key]['pos'][idx] /= len(trainer.val_dataloaders) * pos_weight
                        loss_partial_modal[key]['type'][idx] /= len(trainer.val_dataloaders)
                        loss_partial_modal[key]['bond'][idx] /= len(trainer.val_dataloaders)

                torch.save(loss_partial_modal, os.path.join(pl_module.cfg.accounting.test_outputs_dir, f'loss_{trainer.global_step}.pt'))
                torch.save(loss_partial_modal, os.path.join('.', f'loss_latest.pt'))

                # access the loss
                sum_loss_pos = loss_partial_modal['sync']['pos']
                sum_loss_type = loss_partial_modal['sync']['type']
                sum_loss_bond = loss_partial_modal['sync']['bond']

                sum_loss_pos_given_2d = loss_partial_modal['w/ 2D']['pos']
                sum_loss_pos_wo_2d = loss_partial_modal['w/o 2D']['pos']
                sum_loss_type_given_3d = loss_partial_modal['w/ 3D']['type']
                sum_loss_bond_given_3d = loss_partial_modal['w/ 3D']['bond']
                sum_loss_type_wo_3d = loss_partial_modal['w/o 3D']['type']
                sum_loss_bond_wo_3d = loss_partial_modal['w/o 3D']['bond']

                # plot the loss w.r.t. t
                fig, ax = plt.subplots(3, 1, figsize=(10, 10))
                t = np.linspace(0, 1, N)
                for i, key in enumerate(['sync', 'w/ 2D', 'w/o 2D', 'w/ 3D', 'w/o 3D']):
                    if '3D' not in key:
                        ax[0].plot(t, loss_partial_modal[key]['pos'], label=key)
                    if '2D' not in key:
                        ax[1].plot(t, loss_partial_modal[key]['type'], label=key)
                        ax[2].plot(t, loss_partial_modal[key]['bond'], label=key)
        
                ax[0].set_title('Position Loss')
                ax[1].set_title('Type Loss')
                ax[2].set_title('Bond Loss')
                ax[0].legend()
                ax[1].legend()
                
                # log the plot to wandb
                wandb.log({"val/loss": wandb.Image(fig)}, step=trainer.global_step)

                # save the plot
                # plt.savefig(os.path.join(pl_module.cfg.accounting.test_outputs_dir, f'loss_{trainer.global_step}.png'))
                plt.savefig(f'loss_curve_{trainer.global_step}.png')
                plt.close()

                # if trainer.global_rank == 0:
                #     wandb.log(val_dict, step=trainer.global_step)
                # print(f"step {trainer.global_step}: recon_loss: {sum_loss / sum_batches:.4f}, recon_loss_pos: {sum_loss_pos / sum_batches:.4f}, recon_loss_type: {sum_loss_type / sum_batches:.4f}")
                pl_module.dynamics.train()

                self.loss_partial_modal = loss_partial_modal

                exit(0)

        if trainer.global_step % self.val_freq == 0:
            # plot the t - u curve
            save_dir = os.path.join(pl_module.cfg.accounting.logdir, f"val_outputs")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            if not os.path.exists(pl_module.cfg.accounting.test_outputs_dir):
                os.makedirs(pl_module.cfg.accounting.test_outputs_dir, exist_ok=True)
            with torch.no_grad():
                timewarp_cdf = pl_module.timewarp_cdf
                if timewarp_cdf is not None:
                    u_steps = torch.linspace(1, 0, pl_module.cfg.evaluation.sample_steps, device=pl_module.device)
                    if isinstance(timewarp_cdf, Timewarp):
                        t_steps = timewarp_cdf(u_steps.squeeze(), invert=True).detach().to(torch.float32)
                    else:
                        t_steps = timewarp_cdf(u_steps).detach().to(torch.float32)
                    # Reverse t' to get t = 1 - t'
                    t_steps = 1 - t_steps
                elif pl_module.time_scheduler is not None:
                    t_steps = pl_module.time_scheduler / pl_module.time_scheduler.max()
                else:
                    raise ValueError("No timewarp or time_scheduler found.")

                t_steps = torch.clamp(t_steps, min=pl_module.dynamics.t_min)

                # take 10 evenly spaced samples out of sample_steps
                plt.plot(t_steps[:, 0].cpu().numpy(), label='discrete')
                plt.plot(t_steps[:, 1].cpu().numpy(), label='continuous')
                plt.xlabel('u')
                plt.ylabel('t')
                plt.legend()
                plt.savefig(os.path.join(save_dir, f't_steps_{trainer.global_step}.png'))
                plt.close()
                image = wandb.Image(os.path.join(save_dir, f't_steps_{trainer.global_step}.png'))
                pl_module.logger.log_image(key="val/t_steps", images=[image])

        if False:
            with torch.no_grad():
                loss_partial_modal = {}
                true_loss_partial_modal = {}
                time_partial_modal = {}
                pos_normalizer = torch.tensor(
                    pl_module.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=pl_module.device
                )

                # use timestep 1 for reconstruction loss computation
                N = pl_module.dynamics.discrete_steps

                for batch in trainer.val_dataloaders:
                    # prepare batch data
                    batch = batch.to(pl_module.device)

                    protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand = (
                        getattr(batch, "protein_pos", None),
                        batch.protein_atom_feature.float() if hasattr(batch, "protein_atom_feature") else None,
                        getattr(batch, "protein_element_batch", None),
                        batch.ligand_pos,
                        batch.ligand_atom_feature_full,
                        batch.ligand_element_batch,
                    )
                    ligand_pos = ligand_pos / pos_normalizer
                
                    if protein_pos is not None:
                        protein_pos = protein_pos / pos_normalizer
                        # move protein center to origin & ligand correspondingly
                        protein_pos, ligand_pos, offset = center_pos(
                            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=pl_module.cfg.dynamics.center_pos_mode) #TODO: ugly 
                    num_graphs = batch_ligand.max().item() + 1

                    # sample a random timestep for val loss computation
                    # i = torch.randint(0, N, (1,)).item()
                    for i in range(0, N, N // 10):
                        u = torch.tensor([i / float(N)], dtype=ligand_pos.dtype, device=ligand_pos.device).repeat(num_graphs, 1)[batch_ligand]

                        if not pl_module.cfg.dynamics.use_discrete_t and not pl_module.cfg.dynamics.destination_prediction:
                            u = torch.clamp(u, min=pl_module.dynamics.t_min)  # clamp t to [t_min,1]

                        sigmas = pl_module.timewarp_cdf(u.squeeze(), invert = True).detach().to(torch.float32)
                        sigmas_norm = (sigmas - pl_module.timewarp_cdf.sigma_min) / (pl_module.timewarp_cdf.sigma_max - pl_module.timewarp_cdf.sigma_min)
                        
                        t, t_pos = sigmas_norm[:, 0].unsqueeze(1), sigmas_norm[:, 1].unsqueeze(1)
                        t, t_pos = 1 - t, 1 - t_pos

                        loss_estimated = pl_module.timewarp_cdf(sigmas)
                        loss_cat = loss_estimated[:, 0]
                        loss_cont = loss_estimated[:, 1]
                        if '3d' not in loss_partial_modal:
                            loss_partial_modal['3d'] = [0] * 10
                            loss_partial_modal['2d'] = [0] * 10
                            time_partial_modal['3d'] = [0] * 10
                            time_partial_modal['2d'] = [0] * 10
                            true_loss_partial_modal['3d'] = [0] * 10
                            true_loss_partial_modal['2d'] = [0] * 10
                        idx = int(i / (N / 10))
                        loss_partial_modal['3d'][idx] += float(loss_cat.mean())
                        loss_partial_modal['2d'][idx] += float(loss_cont.mean())
                        time_partial_modal['3d'][idx] += float(sigmas[:, 0].mean())
                        time_partial_modal['2d'][idx] += float(sigmas[:, 1].mean())

                        losses = pl_module.dynamics.reconstruction_loss_one_step(
                            t,
                            protein_pos=protein_pos,
                            protein_v=protein_v,
                            batch_protein=batch_protein,
                            ligand_pos=ligand_pos,
                            ligand_v=ligand_v,
                            batch_ligand=batch_ligand,
                            ligand_bond_type=batch.ligand_fc_bond_type,
                            ligand_bond_index=batch.ligand_fc_bond_index,
                            batch_ligand_bond=batch.ligand_fc_bond_type_batch,
                            include_protein=pl_module.include_protein,
                            t_pos=t_pos,
                            recon_loss=True
                        )

                        pos_loss, type_loss, bond_loss, charge_loss, aromatic_loss = (
                            losses['closs'],
                            losses['dloss'],
                            losses['dloss_bond'],
                            losses['dloss_charge'],
                            losses['dloss_aromatic'],
                        )

                        true_loss_partial_modal['3d'][idx] += float(pos_loss.mean())
                        true_loss_partial_modal['2d'][idx] += float(type_loss.mean() + bond_loss.mean())

                # normalize the loss
                for idx in range(10):
                    loss_partial_modal['3d'][idx] /= len(trainer.val_dataloaders)
                    loss_partial_modal['2d'][idx] /= len(trainer.val_dataloaders)
                    time_partial_modal['3d'][idx] /= len(trainer.val_dataloaders)
                    time_partial_modal['2d'][idx] /= len(trainer.val_dataloaders)
                    true_loss_partial_modal['3d'][idx] /= len(trainer.val_dataloaders)
                    true_loss_partial_modal['2d'][idx] /= len(trainer.val_dataloaders)

                # plot the loss w.r.t. t
                fig, ax = plt.subplots(2, 1, figsize=(10, 10))

                ax[0].plot(time_partial_modal['3d'], true_loss_partial_modal['3d'], label='True')
                ax[0].plot(time_partial_modal['3d'], loss_partial_modal['3d'], label='Estimated', linestyle='--')
                ax[1].plot(time_partial_modal['2d'], true_loss_partial_modal['2d'], label='True')
                ax[1].plot(torch.tensor(time_partial_modal['2d']), loss_partial_modal['2d'], label='Estimated', linestyle='--')

                ax[0].set_title('3D Loss')
                ax[1].set_title('2D Loss')
                ax[0].legend()
                ax[1].legend()
                # log the plot to wandb
                wandb.log({"val/loss": wandb.Image(fig)}, step=trainer.global_step)
                # save the plot
                plt.savefig(os.path.join(pl_module.cfg.accounting.test_outputs_dir, f'loss_step{trainer.global_step}.png'))
                plt.savefig(f'loss_step{trainer.global_step}.png')
                plt.close()


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

        if hasattr(pl_module.cfg.train, 'val_mode') and pl_module.cfg.train.val_mode == 'loss':
            recon_loss = torch.stack(self.outputs).mean().item()
            out_metrics = {
                'val/recon_loss': recon_loss,
            }
            pl_module.log_dict(out_metrics, on_step=False, on_epoch=True, sync_dist=True)

            print(json.dumps(out_metrics, indent=4))

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_test_start(trainer, pl_module)
        if not os.path.exists(pl_module.cfg.accounting.test_outputs_dir):
            os.makedirs(pl_module.cfg.accounting.test_outputs_dir, exist_ok=True)
        timewarp_cdf = pl_module.timewarp_cdf
        if timewarp_cdf:
            plot_weights_and_bins(timewarp_cdf)
            plt.savefig(os.path.join(pl_module.cfg.accounting.test_outputs_dir, f'weights_{trainer.global_step}.png'))
            # plt.savefig(f'weights_{trainer.global_step}.png')
            plt.close()
            image = wandb.Image(os.path.join(pl_module.cfg.accounting.test_outputs_dir, f'weights_{trainer.global_step}.png'))
            pl_module.logger.log_image(key="test/weights", images=[image])
            u_steps = torch.linspace(1, 0, pl_module.cfg.evaluation.sample_steps, device=pl_module.device)
            t_steps = timewarp_cdf(u_steps, invert=True).detach().to(torch.float32)
            t_steps = (t_steps - timewarp_cdf.sigma_min) / (timewarp_cdf.sigma_max - timewarp_cdf.sigma_min)
            # Reverse t' to get t = 1 - t'
            t_steps = 1 - t_steps
        elif pl_module.time_scheduler is not None:
            t_steps = pl_module.time_scheduler / pl_module.time_scheduler.max()
        else:
            t_steps = torch.linspace(0, 1, pl_module.cfg.evaluation.sample_steps, device=pl_module.device).unsqueeze(1).repeat(1, 2)

        t_steps = torch.clamp(t_steps, min=pl_module.dynamics.t_min)

        # take 10 evenly spaced samples out of sample_steps
        plt.plot(t_steps[:, 0].cpu().numpy(), label='discrete')
        plt.plot(t_steps[:, 1].cpu().numpy(), label='continuous')
        plt.xlabel('u')
        plt.ylabel('t')
        plt.legend()
        plt.savefig(os.path.join(pl_module.cfg.accounting.test_outputs_dir, f't_steps_{trainer.global_step}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        image = wandb.Image(os.path.join(pl_module.cfg.accounting.test_outputs_dir, f't_steps_{trainer.global_step}.png'))
        pl_module.logger.log_image(key="test/t_steps", images=[image])

def plot_surface(loss_partial_modal, ax, loss_key, title, zlabel, N, smooth=False):
    t = list(range(N))
    Z = np.array([[loss_partial_modal[i][j][loss_key] for j in range(N)] for i in range(N)])
    Z = Z.reshape(N, N)
    diagonal_z = np.array([loss_partial_modal[i][i][loss_key] for i in range(N)])

    X, Y = np.meshgrid(t, t, indexing='ij')
    # mask = X <= Y
    mask = None

    # clamp loss max
    Z = np.clip(Z, 0, 20)

    if smooth:
        Z = gaussian_filter(Z, sigma=1)
    
    if mask is None:
        ax.plot_surface(X, Y, Z, cmap='viridis')
    else:
        # Flatten and mask
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z.flatten()
        valid_indices = mask.flatten()

        # Filter valid points
        X_valid = X_flat[valid_indices]
        Y_valid = Y_flat[valid_indices]
        Z_valid = Z_flat[valid_indices]

        # Plot with trisurf for the valid region
        ax.plot_trisurf(X_valid, Y_valid, Z_valid, cmap='viridis')

    # Extract diagonal points (where X == Y)
    diagonal_x = np.diag(X)
    diagonal_y = np.diag(Y)
    
    # Plot the diagonal curve
    ax.plot(diagonal_x, diagonal_y, diagonal_z, color='red', linewidth=2, label='Diagonal Curve')

    # x ticks from 0 to 9
    # ax.set_xticks(t)
    # ax.set_yticks(t)
    ax.set_xlabel('t_2d')
    ax.set_ylabel('t_3d')
    ax.set_zlabel(zlabel)
    ax.set_title(title)


class LossSurfaceCallback(Callback):
    def __init__(self, val_freq=1) -> None:
        super().__init__()
        self.val_freq = val_freq
        self.batch_count = 0

    def on_train_batch_start(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            batch: Any,
            batch_idx: int,
            unused: int = 0,
    ) -> None:
        super().on_train_batch_start(
            trainer, pl_module, batch, batch_idx
        )
        
        if trainer.global_step == 0: 
            if not os.path.exists(pl_module.cfg.accounting.test_outputs_dir):
                os.makedirs(pl_module.cfg.accounting.test_outputs_dir, exist_ok=True)
            # perform a full validation
            with torch.no_grad():
                # switch to eval mode
                pl_module.dynamics.eval()
                pos_normalizer = torch.tensor(
                    pl_module.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=pl_module.device
                )

                sum_batches, sum_loss = 0, 0
                loss_partial_modal = {}

                # use timestep 1 for reconstruction loss computation
                N = pl_module.dynamics.discrete_steps
                sigma1 = pl_module.dynamics.sigma1_coord
                K = pl_module.dynamics.num_classes
                beta1 = pl_module.dynamics.beta1
                if hasattr(pl_module.dynamics, 'num_bond_classes'):
                    E = pl_module.dynamics.num_bond_classes
                    beta1_bond = pl_module.dynamics.beta1_bond
                else:
                    E = 0
                    beta1_bond = 0

                N = pl_module.cfg.evaluation.sample_steps
                progress_bar = tqdm(total=N * N)
                for i in range(N):
                    loss_partial_modal[i] = {}
                    for j in range(N):
                        loss_partial_modal[i][j] = {
                            'pos': 0,
                            'type': 0,
                            'bond': 0,
                            'pos_mse': 0,
                            'type_ce': 0,
                            'bond_ce': 0,
                            'pos_cont': 0,
                            'type_cont': 0,
                            'bond_cont': 0,
                            'loss': 0,
                        }
                
                mode = pl_module.cfg.evaluation.mode
                if mode == 'train':
                    loader = trainer.train_dataloader
                elif mode == 'val':
                    loader = trainer.val_dataloaders
                
                if os.path.exists(os.path.join(pl_module.cfg.accounting.test_outputs_dir, f'loss_grid_rect{N}_{mode}.pt')):
                    print('Already exists:', os.path.join(pl_module.cfg.accounting.test_outputs_dir, f'loss_grid_rect{N}_{mode}.pt'))
                    timestr = time.strftime("%Y%m%d-%H%M%S")
                    mode = f'{mode}_{timestr}'
                for batch in loader:
                    self.batch_count += 1

                    if self.batch_count > 1:
                        break

                    # prepare batch data
                    batch = batch.to(pl_module.device)

                    protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand = (
                        getattr(batch, "protein_pos", None),
                        batch.protein_atom_feature.float() if hasattr(batch, "protein_atom_feature") else None,
                        getattr(batch, "protein_element_batch", None),
                        batch.ligand_pos,
                        batch.ligand_atom_feature_full,
                        batch.ligand_element_batch,
                    )
                    ligand_pos = ligand_pos / pos_normalizer
                
                    if protein_pos is not None:
                        protein_pos = protein_pos / pos_normalizer
                        # move protein center to origin & ligand correspondingly
                        protein_pos, ligand_pos, offset = center_pos(
                            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=pl_module.cfg.dynamics.center_pos_mode) #TODO: ugly 
                    else:
                        ligand_pos = torch.zeros_like(ligand_pos)
                    num_graphs = batch_ligand.max().item() + 1
                    sum_batches += num_graphs


                    t0 = torch.tensor([0], dtype=ligand_pos.dtype, device=pl_module.device).repeat(num_graphs, 1)[batch_ligand]
                    t1 = torch.tensor([1], dtype=ligand_pos.dtype, device=pl_module.device).repeat(num_graphs, 1)[batch_ligand]

                    # sample a random timestep for val loss computation
                    # i = torch.randint(0, N, (1,)).item()
                    for t_dis in range(N):
                        t_discrete = torch.tensor([(t_dis + 1) / N], dtype=ligand_pos.dtype, device=ligand_pos.device).repeat(num_graphs, 1)

                        if not pl_module.cfg.dynamics.use_discrete_t and not pl_module.cfg.dynamics.destination_prediction:
                            t_discrete = torch.clamp(t_discrete, min=pl_module.dynamics.t_min)  # clamp t to [t_min,1]

                        # compute bfn loss  # TODO: convert to reconstruction loss
                        # ablate 2D
                        # for (t_discrete, t_pos, key) in [(t, t, 'sync')]:

                        for t_cont in range(N):
                            # if t_cont > t_dis: continue
                            t_pos = torch.tensor([(t_cont + 1) / N], dtype=ligand_pos.dtype, device=ligand_pos.device).repeat(num_graphs, 1)
                            losses = pl_module.dynamics.reconstruction_loss_one_step(
                                t_discrete,
                                protein_pos=protein_pos,
                                protein_v=protein_v,
                                batch_protein=batch_protein,
                                ligand_pos=ligand_pos,
                                ligand_v=ligand_v,
                                batch_ligand=batch_ligand,
                                ligand_bond_type=batch.ligand_fc_bond_type,
                                ligand_bond_index=batch.ligand_fc_bond_index,
                                batch_ligand_bond=batch.ligand_fc_bond_type_batch,
                                include_protein=pl_module.include_protein,
                                t_pos=t_pos,
                                recon_loss=True
                            )

                            pos_loss, type_loss, bond_loss = (
                                losses['closs'],
                                losses['dloss'],
                                losses['dloss_bond'],
                            )

                            loss_mse, loss_type_ce, loss_bond_ce, loss_pos_cont, loss_type_cont, loss_bond_cont = (
                                losses['closs_mse'],
                                losses['dloss_ce'],
                                losses['dloss_bond_ce'],
                                losses['closs_cont'],
                                losses['dloss_cont'],
                                losses['dloss_bond_cont'],
                            )

                            # here the discretised_loss is close for current version.
                            loss = torch.mean(pos_loss + pl_module.cfg.train.v_loss_weight * type_loss + 
                                                pl_module.cfg.train.bond_loss_weight * bond_loss)

                            assert t_dis in loss_partial_modal, f"{t_dis} {t_cont}"
                            assert t_cont in loss_partial_modal[t_dis], f"{t_dis} {t_cont}"
                            
                            loss_partial_modal[t_dis][t_cont]['pos'] += float(pos_loss.mean())
                            loss_partial_modal[t_dis][t_cont]['type'] += float(type_loss.mean())
                            loss_partial_modal[t_dis][t_cont]['bond'] += float(bond_loss.mean())
                            loss_partial_modal[t_dis][t_cont]['pos_mse'] += float(loss_mse.mean())
                            loss_partial_modal[t_dis][t_cont]['type_ce'] += float(loss_type_ce.mean())
                            loss_partial_modal[t_dis][t_cont]['bond_ce'] += float(loss_bond_ce.mean())
                            loss_partial_modal[t_dis][t_cont]['pos_cont'] += float(loss_pos_cont.mean())
                            loss_partial_modal[t_dis][t_cont]['type_cont'] += float(loss_type_cont.mean())
                            loss_partial_modal[t_dis][t_cont]['bond_cont'] += float(loss_bond_cont.mean())

                            # weighting_factor = t_discrete.view(-1) * beta1
        
                            # Compute the original MSE values
                            # dloss_cont_mse = loss_type_cont / weighting_factor
                            # dloss_bond_cont_mse = loss_bond_cont / weighting_factor
                            
                            # # Store the original MSE values back in the dictionary
                            # loss_partial_modal[t_dis][t_cont]['type_cont_mse'] += float(dloss_cont_mse.mean())
                            # loss_partial_modal[t_dis][t_cont]['bond_cont_mse'] += float(dloss_bond_cont_mse.mean())

                            # loss = torch.mean(loss_mse + dloss_cont_mse + dloss_bond_cont_mse)
                            loss_partial_modal[t_dis][t_cont]['loss'] += float(loss.mean())

                            progress_bar.update(1)

                # normalize the loss
                for key in loss_partial_modal:
                    for key2 in loss_partial_modal[key]:
                        loss_partial_modal[key][key2]['pos'] /= len(trainer.val_dataloaders)
                        loss_partial_modal[key][key2]['type'] /= len(trainer.val_dataloaders)
                        loss_partial_modal[key][key2]['bond'] /= len(trainer.val_dataloaders)
                        loss_partial_modal[key][key2]['pos_mse'] /= len(trainer.val_dataloaders)
                        loss_partial_modal[key][key2]['type_ce'] /= len(trainer.val_dataloaders)
                        loss_partial_modal[key][key2]['bond_ce'] /= len(trainer.val_dataloaders)
                        loss_partial_modal[key][key2]['pos_cont'] /= len(trainer.val_dataloaders)
                        loss_partial_modal[key][key2]['type_cont'] /= len(trainer.val_dataloaders)
                        loss_partial_modal[key][key2]['bond_cont'] /= len(trainer.val_dataloaders)
                        # loss_partial_modal[key][key2]['type_cont_mse'] /= len(trainer.val_dataloaders)
                        # loss_partial_modal[key][key2]['bond_cont_mse'] /= len(trainer.val_dataloaders)
                        loss_partial_modal[key][key2]['loss'] /= len(trainer.val_dataloaders)

                # save the loss
                torch.save(loss_partial_modal, f'loss_grid_rect{N}_{mode}.pt')
                torch.save(loss_partial_modal, os.path.join(pl_module.cfg.accounting.test_outputs_dir, f'loss_grid_rect{N}_{mode}.pt'))

                # plot the loss as a surface over 2D grid (one figure containing 3 subplots)
                fig, ax = plt.subplots(3, 1, figsize=(10, 10), subplot_kw={'projection': '3d'})

                plot_surface(loss_partial_modal, ax[0], 'pos', 'Position Loss', 'pos_loss', N)
                plot_surface(loss_partial_modal, ax[1], 'type', 'Type Loss', 'type_loss', N)
                plot_surface(loss_partial_modal, ax[2], 'bond', 'Bond Loss', 'bond_loss', N)
                plt.savefig(f'loss_single.png', bbox_inches='tight')
                plt.savefig(os.path.join(pl_module.cfg.accounting.test_outputs_dir, f'loss_{N}_{mode}.png'), bbox_inches='tight')

                # Plot another figure for the total loss
                fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
                plot_surface(loss_partial_modal, ax, 'loss', 'Total Loss', 'total_loss', N)
                plt.savefig(f'loss_total.png', bbox_inches='tight')
                plt.savefig(os.path.join(pl_module.cfg.accounting.test_outputs_dir, f'loss_total_{N}_{mode}.png'), bbox_inches='tight')

                # Plot the smoothed surface of the total loss
                fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
                plot_surface(loss_partial_modal, ax, 'loss', 'Total Loss Smoothed', 'total_loss', N, smooth=True)
                plt.savefig(f'loss_total_smoothed.png', bbox_inches='tight')
                plt.savefig(os.path.join(pl_module.cfg.accounting.test_outputs_dir, f'loss_total_smoothed_{N}_{mode}.png'), bbox_inches='tight')

                exit(0)


class LossValidationCallback(Callback):
    # compute the BFN reconstruction loss for validation dataloader.
    def __init__(self) -> None:
        super().__init__()
        self.total_samples = 0

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
        if not hasattr(pl_module.cfg.train, 'val_mode') or pl_module.cfg.train.val_mode == 'sample':
            return
        if outputs is not None:
            self.pos_losses.append(outputs['closs'])
            self.type_losses.append(outputs['dloss'])
            self.bond_losses.append(outputs['dloss_bond'])  # []
            if pl_module.cfg.time_decoupled:
                self.pos_losses_given_2d.append(outputs['closs_given_2d'])

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_start(trainer, pl_module)
        if not hasattr(pl_module.cfg.train, 'val_mode') or pl_module.cfg.train.val_mode == 'sample':
            return
        self.pos_losses = []
        self.type_losses = []
        self.bond_losses = []
        if pl_module.cfg.time_decoupled:
            self.pos_losses_given_2d = []

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        self.total_samples += len(pl_module.trainer.train_dataloader) * pl_module.cfg.train.batch_size
        if not hasattr(pl_module.cfg.train, 'val_mode') or pl_module.cfg.train.val_mode == 'sample':
            return

        pos_loss = torch.stack(self.pos_losses).mean().item()
        type_loss = torch.stack(self.type_losses).mean().item()
        bond_loss = torch.stack(self.bond_losses).mean().item()

        if not pl_module.include_protein:
            pos_loss = pos_loss * 0.0

        out_metrics = {
            'val/recon_loss_pos': pos_loss,
            'val/recon_loss_type': type_loss,
            'val/recon_loss_bond': bond_loss,
        }
        if pl_module.cfg.time_decoupled:
            if not pl_module.include_protein:
                pos_loss_given_2d = 0.0
            else:
                pos_loss_given_2d = torch.stack(self.pos_losses_given_2d).mean().item()
            out_metrics['val/recon_loss_pos_given_2d'] = pos_loss_given_2d

        out_metrics['val/recon_loss'] = pos_loss + pl_module.cfg.train.v_loss_weight * type_loss + \
                                        pl_module.cfg.train.bond_loss_weight * bond_loss

        pl_module.log_dict(out_metrics, on_step=False, on_epoch=True, sync_dist=True)
        if trainer.global_rank == 0:
            print(json.dumps(out_metrics, indent=4))

        return out_metrics


class ClassifierValidationCallback(Callback):
    # compute the property predictor MSE loss for validation dataloader.
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
                sum_batches, sum_loss = 0, 0.
                pos_normalizer = torch.tensor(
                    pl_module.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=batch.ligand_pos.device
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
                    try:
                        prop = getattr(batch, pl_module.dynamics_cfg.prop_name).float()  # [N_lig, 1]
                    except Exception as e:
                        print(f"Error: {e}")
                        prop = torch.tensor(getattr(batch, pl_module.dynamics_cfg.prop_name), dtype=torch.float32, device=ligand_pos.device)
                        print(prop, type(prop))
            
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
                        exp_loss = pl_module.dynamics.reconstruction_loss_one_step(
                            t,
                            protein_pos=protein_pos,
                            protein_v=protein_v,
                            batch_protein=batch_protein,
                            ligand_pos=ligand_pos,
                            ligand_v=ligand_v,
                            batch_ligand=batch_ligand,
                            prop=prop,
                        )
                        loss = torch.mean(exp_loss)
                        sum_loss += float(loss) * num_graphs

                # log the mean reconstruction loss
                pl_module.log_dict({
                        "val/recon_loss": sum_loss / sum_batches,
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
    def __init__(self, dataset, atom_enc_mode, atom_decoder, atom_type_one_hot, single_bond, docking_config, dataset_smiles_set=[], docking_rmsd=False,output_dir=None) -> None:
        super().__init__()
        self.dataset = dataset
        self.atom_enc_mode = atom_enc_mode
        self.atom_decoder = atom_decoder
        self.single_bond = single_bond
        self.type_one_hot = atom_type_one_hot
        self.docking_config = docking_config
        self.outputs = []
        self.output_dir = output_dir 
        self.dataset_smiles_set = dataset_smiles_set
        self.docking_rmsd = docking_rmsd
    
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        if self.docking_rmsd:
            self.metric = RMSDMetric(
                atom_decoder=self.atom_decoder,
                atom_enc_mode=self.atom_enc_mode,
                type_one_hot=self.type_one_hot,
                single_bond=self.single_bond,
                protein_root=self.docking_config.protein_root,
            )
        else:
            self.metric = CondMolGenMetric(
                atom_decoder=self.atom_decoder,
                atom_enc_mode=self.atom_enc_mode,
                type_one_hot=self.type_one_hot,
                single_bond=self.single_bond,
                docking_config=self.docking_config,
                dataset_smiles_set=self.dataset_smiles_set,
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
        mol_path = os.path.join(self.path, f'tmp.sdf')
        for i, graph in enumerate(outputs):                                     
            if 'mol' not in graph: continue
            mol = graph.mol                                                     
            mol.SetProp('_Name', graph.ligand_filename)
            with open(mol_path, 'a') as f:
                writer = Chem.SDWriter(f)
                writer.write(mol)
                writer.close()
        self.outputs.extend(outputs)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_test_start(trainer, pl_module)
        self.outputs = []
        path = pl_module.cfg.accounting.test_outputs_dir
        timestr = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(path, timestr)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        self.path = path
        # dump config
        pl_module.cfg.save2yaml(os.path.join(path, 'config.yaml'))


    def on_test_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        super().on_test_epoch_end(trainer, pl_module)

        if hasattr(pl_module.cfg, 'skip_eval') and pl_module.cfg.skip_eval:
            out_metrics = {}
        else:
            raw_evaluation = self.metric.compute_raw_evaluation(self.outputs, skip_chem=getattr(pl_module.cfg, 'skip_chem', False))
            out_metrics = self.metric.evaluate(self.outputs, raw_evaluation)
            out_metrics = {f'test/{k}': v for k, v in out_metrics.items()}
            # barrier
            # torch.distributed.barrier()
            pl_module.log_dict(out_metrics, sync_dist=True)
            
        # with timing('docking'):
        if trainer.global_rank == 0:
            
            path = self.path
            version = ''

            if hasattr(pl_module.cfg.evaluation, 'objective'):
                version += f'-{pl_module.cfg.evaluation.objective}'

            # torch.save(self.outputs, os.path.join(path, f'outputs{version}.pt'))
            # torch.save(raw_evaluation, os.path.join(path, f'raw_evaluation{version}.pt'))

            mol_path = os.path.join(path, f'mols{version}')
            os.makedirs(mol_path, exist_ok=True)
            results = []
            pb_valid_n_rmsd_lt2 = 0
            for i, graph in enumerate(self.outputs):                                     
                if 'mol' not in graph: continue
                mol = graph.mol                                                     
                mol.SetProp('_Name', graph.ligand_filename)
                item = {'mol': mol, 'ligand_filename': graph.ligand_filename}
                try:
                    if 'chem' in raw_evaluation and raw_evaluation['chem'][i]: 
                        item['vina'] = raw_evaluation['chem'][i]
                        if 'vina_score' in raw_evaluation['chem'][i]:
                            mol.SetProp('vina_score', str(raw_evaluation['chem'][i]['vina_score'])) 
                        if 'vina_minimize' in raw_evaluation['chem'][i]:
                            mol.SetProp('vina_minimize', str(raw_evaluation['chem'][i]['vina_minimize']))
                        if 'vina_dock' in raw_evaluation['chem'][i]:
                            mol.SetProp('vina_dock', str(raw_evaluation['chem'][i]['vina_dock']))
                        if 'rmsd' in raw_evaluation['chem'][i] and 'pb_valid' in raw_evaluation['chem'][i]:
                            mol.SetProp('rmsd', str(raw_evaluation['chem'][i]['rmsd']))
                            mol.SetProp('pb_valid', str(raw_evaluation['chem'][i]['pb_valid']))
                            if item['vina']['pb_valid'] and item['vina']['rmsd'] < 2:
                                pb_valid_n_rmsd_lt2 += 1
                except:
                    pass
                with Chem.SDWriter(os.path.join(mol_path, f'{i}.sdf')) as writer:
                    writer.write(mol)
                results.append(item)

            torch.save(results, os.path.join(path, 'evaluated.pt'))
            
            pb_dict = {'test/pb_valid & rmsd<2': pb_valid_n_rmsd_lt2 / len(self.outputs)}
            pl_module.log_dict(pb_dict)
            out_metrics.update(pb_dict)
            print(json.dumps(out_metrics, indent=4))


class DockingValidationCallback(Callback):
    def __init__(self, dataset, atom_enc_mode, atom_decoder, atom_type_one_hot, single_bond, docking_config, docking_rmsd=True) -> None:
        super().__init__()
        self.dataset = dataset
        self.atom_enc_mode = atom_enc_mode
        self.atom_decoder = atom_decoder
        self.single_bond = single_bond
        self.type_one_hot = atom_type_one_hot
        self.docking_config = docking_config
        self.outputs = []
        self.docking_rmsd = docking_rmsd
    
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        if self.docking_rmsd:
            self.metric = RMSDMetric(
                atom_decoder=self.atom_decoder,
                atom_enc_mode=self.atom_enc_mode,
                type_one_hot=self.type_one_hot,
                single_bond=self.single_bond,
                protein_root=self.docking_config.protein_root,
            )
        else:
            raise NotImplementedError('DockingValidationCallback only supports RMSD evaluation for now.')

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
        mol_path = os.path.join(self.path, f'tmp.sdf')
        for i, graph in enumerate(outputs):                                     
            if 'mol' not in graph: continue
            mol = graph.mol                                                     
            mol.SetProp('_Name', graph.ligand_filename)
            with open(mol_path, 'a') as f:
                writer = Chem.SDWriter(f)
                writer.write(mol)
                writer.close()
        self.outputs.extend(outputs)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_test_start(trainer, pl_module)
        self.outputs = []
        path = pl_module.cfg.accounting.test_outputs_dir
        timestr = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(path, timestr)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        self.path = path


    def on_test_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        super().on_test_epoch_end(trainer, pl_module)

        if hasattr(pl_module.cfg, 'skip_eval') and pl_module.cfg.skip_eval:
            out_metrics = {}
        else:
            raw_evaluation = self.metric.compute_raw_evaluation(self.outputs)
            out_metrics = self.metric.evaluate(self.outputs, raw_evaluation)
            out_metrics = {f'test/{k}': v for k, v in out_metrics.items()}
            # barrier
            # torch.distributed.barrier()
            pl_module.log_dict(out_metrics, sync_dist=True)
            
        # with timing('docking'):
        if trainer.global_rank == 0:
            
            path = self.path
            version = ''

            if hasattr(pl_module.cfg.evaluation, 'objective'):
                version += f'-{pl_module.cfg.evaluation.objective}'

            # dump config
            pl_module.cfg.save2yaml(os.path.join(path, 'config.yaml'))

            # torch.save(self.outputs, os.path.join(path, f'outputs{version}.pt'))
            # torch.save(raw_evaluation, os.path.join(path, f'raw_evaluation{version}.pt'))

            mol_path = os.path.join(path, f'mols{version}')
            os.makedirs(mol_path, exist_ok=True)
            results = []
            pb_valid_n_rmsd_lt2 = 0
            for i, graph in enumerate(self.outputs):                                     
                if 'mol' not in graph: continue
                mol = graph.mol                                                     
                mol.SetProp('_Name', graph.ligand_filename)
                item = {'mol': mol, 'ligand_filename': graph.ligand_filename}
                try:
                    if 'chem' in raw_evaluation and raw_evaluation['chem'][i]: 
                        item['vina'] = raw_evaluation['chem'][i]
                        if 'rmsd' in raw_evaluation['chem'][i] and 'pb_valid' in raw_evaluation['chem'][i]:
                            mol.SetProp('rmsd', str(raw_evaluation['chem'][i]['rmsd']))
                            mol.SetProp('pb_valid', str(raw_evaluation['chem'][i]['pb_valid']))
                            if item['vina']['pb_valid'] and item['vina']['rmsd'] < 2:
                                pb_valid_n_rmsd_lt2 += 1
                except:
                    pass
                with Chem.SDWriter(os.path.join(mol_path, f'{i}.sdf')) as writer:
                    writer.write(mol)
                results.append(item)

            torch.save(results, os.path.join(path, 'evaluated.pt'))
            
            pb_dict = {'test/pb_valid & rmsd<2': pb_valid_n_rmsd_lt2 / len(self.outputs)}
            pl_module.log_dict(pb_dict)
            out_metrics.update(pb_dict)
            print(json.dumps(out_metrics, indent=4))

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_start(trainer, pl_module)
        self.outputs = []

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.outputs.extend(outputs)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        if hasattr(pl_module.cfg, 'skip_eval') and pl_module.cfg.skip_eval:
            out_metrics = {}
        else:
            raw_evaluation = self.metric.compute_raw_evaluation(self.outputs, skip_chem=True)
            out_metrics = self.metric.evaluate(self.outputs, raw_evaluation)
            out_metrics = {f'val/{k}': v for k, v in out_metrics.items()}
            # barrier
            # torch.distributed.barrier()
            pl_module.log_dict(out_metrics, sync_dist=True)
            
        # with timing('docking'):
        if trainer.global_rank == 0:
            print(json.dumps(out_metrics, indent=4))
            pb_valid_n_rmsd_lt2 = 0
            for i, graph in enumerate(self.outputs):                                     
                if 'mol' not in graph: continue
                mol = graph.mol                                                     
                mol.SetProp('_Name', graph.ligand_filename)
                item = {'mol': mol, 'ligand_filename': graph.ligand_filename}
                try:
                    if 'chem' in raw_evaluation and raw_evaluation['chem'][i]: 
                        item['vina'] = raw_evaluation['chem'][i]
                        if 'rmsd' in raw_evaluation['chem'][i] and 'pb_valid' in raw_evaluation['chem'][i]:
                            mol.SetProp('rmsd', str(raw_evaluation['chem'][i]['rmsd']))
                            mol.SetProp('pb_valid', str(raw_evaluation['chem'][i]['pb_valid']))
                            if item['vina']['pb_valid'] and item['vina']['rmsd'] < 2:
                                pb_valid_n_rmsd_lt2 += 1
                except:
                    pass
            pb_dict = {'val/pb_valid & rmsd<2': pb_valid_n_rmsd_lt2 / len(self.outputs)}
            pl_module.log_dict(pb_dict)