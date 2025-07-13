import argparse
import os
import shutil

import torch

from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

import datetime, pytz

from core.config.config import Config, parse_config
from core.models.sbdd4train_timewarp import SBDD4Train
from core.callbacks.basic import RecoverCallback, GradientClip, NormalizerCallback, EMACallback
from core.callbacks.validation_callback import (
    CondMolGenValidationCallback,
    MolVisualizationCallback,
    ReconValidationCallback,
    DockingTestCallback,
)

import core.utils.transforms as trans
from core.datasets.utils import PDBProtein, parse_sdf_file
from core.datasets.pl_data import ProteinLigandData, torchify_dict
from core.datasets.pl_data import FOLLOW_BATCH

import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import seed_everything
from pytorch_lightning.profilers import SimpleProfiler, PyTorchProfiler

from absl import logging
import glob
import json
from rdkit import Chem
import numpy as np
from core.evaluation.utils import scoring_func
from core.evaluation.docking_vina import VinaDockingTask
from posecheck import PoseCheck

def get_dataloader_from_pdb(cfg):
    assert cfg.evaluation.protein_path is not None and cfg.evaluation.ligand_path is not None
    protein_fn, ligand_fn = cfg.evaluation.protein_path, cfg.evaluation.ligand_path

    # load protein and ligand
    kekulize = hasattr(cfg.dynamics.net_config, 'num_bond_classes') and cfg.dynamics.net_config.num_bond_classes == 4
    protein = PDBProtein(protein_fn)
    ligand_dict = parse_sdf_file(ligand_fn, kekulize=kekulize)
    pdb_block_pocket = protein.residues_to_pdb_block(
        protein.query_residues_ligand(ligand_dict, cfg.dynamics.net_config.r_max)
    )
    pocket = PDBProtein(pdb_block_pocket)
    pocket_dict = pocket.to_dict_atom()

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict=torchify_dict(ligand_dict),
    )
    data.protein_filename = protein_fn
    data.ligand_filename = ligand_fn

    # transform
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(cfg.data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]
    transform = Compose(transform_list)
    cfg.dynamics.protein_atom_feature_dim = protein_featurizer.feature_dim
    cfg.dynamics.ligand_atom_feature_dim = ligand_featurizer.feature_dim
    print(f"protein feature dim: {cfg.dynamics.protein_atom_feature_dim}, " +
            f"ligand feature dim: {cfg.dynamics.ligand_atom_feature_dim}")

    # dataloader
    collate_exclude_keys = ["ligand_nbh_list"]
    test_set = [transform(data)] * cfg.evaluation.num_samples
    cfg.evaluation.num_samples = 1
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    )

    if cfg.evaluation.docking_config is not None:
        cfg.evaluation.docking_config.protein_root = os.path.dirname(os.path.abspath(protein_fn))
        print(f"protein root: {cfg.evaluation.docking_config.protein_root}")

    return test_loader


def get_logger(cfg):
    os.makedirs(cfg.accounting.wandb_logdir, exist_ok=True)
    # TODO save code
    if cfg.wandb_resume_id is not None:
        wandb_logger = WandbLogger(
            id=cfg.wandb_resume_id,
            project=cfg.project_name,
            offline=cfg.no_wandb,
            save_dir=cfg.accounting.wandb_logdir,
            resume='must',
        )
    else: # start a new run
        wandb_logger = WandbLogger(
            name=f"{cfg.exp_name}_{cfg.revision}"
            + f'_{datetime.datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y-%m-%d-%H:%M:%S")}',
            project=cfg.project_name,
            offline=cfg.no_wandb,
            save_dir=cfg.accounting.wandb_logdir,
        )  # add wandb parameters
    return wandb_logger

class Metrics:
    def __init__(self, protein_fn, ref_ligand_fn, ligand_fn):
        self.protein_fn = protein_fn
        self.ref_ligand_fn = ref_ligand_fn
        self.ligand_fn = ligand_fn
        self.exhaustiveness = 16

    def vina_dock(self, mol):
        chem_results = {}
        try:
            chem_results.update(scoring_func.get_chem(mol))
            chem_results['atom_num'] = mol.GetNumAtoms()
            vina_task = VinaDockingTask.from_generated_mol(
                mol, ligand_filename=self.ref_ligand_fn, protein_filename=self.protein_fn)
            score_only_results = vina_task.run(mode='score_only', exhaustiveness=self.exhaustiveness)
            minimize_results = vina_task.run(mode='minimize', exhaustiveness=self.exhaustiveness)
            chem_results['vina_score'] = score_only_results[0]['affinity']
            chem_results['vina_minimize'] = minimize_results[0]['affinity']
            return chem_results
        except Exception as e:
            print(f"[WARN] Docking failed: {e}")
            return {}

    def pose_check(self, mol):
        pc = PoseCheck()
        try:
            pc.load_ligands_from_mols([mol])
            strain = pc.calculate_strain_energy()[0]
            return {'strain': strain}
        except:
            return {}

    def evaluate(self):
        mol = Chem.SDMolSupplier(self.ligand_fn, removeHs=False)[0]
        results = self.vina_dock(mol)
        results.update(self.pose_check(mol))
        return results


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def get_default_paths():
    
    default_ckpt = "/home/xthdhb/MolCRAFT/MolPilot/checkpoints/molpilot_epoch26-val_loss5.42-mol_stable0.48-complete0.83.ckpt"
    default_config = "/home/xthdhb/MolCRAFT/MolPilot/configs/molpilot_config.yaml"
    
    return {
        "ckpt_path": str(default_ckpt),
        "config_path": str(default_config)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_paths = get_default_paths()

    # meta
    parser.add_argument("--config_file", type=str, default=default_paths["config_path"])
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--revision", type=str, default="default")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb_resume_id", type=str, default=None)
    parser.add_argument('--empty_folder', action='store_true')
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default=default_paths["ckpt_path"])
    
    # global config
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--logging_level", type=str, default="warning")
    parser.add_argument("--val_freq", type=int, default=1000)

    # train data params
    parser.add_argument('--random_rot', action='store_true')
    parser.add_argument("--pos_noise_std", type=float, default=0)    
    parser.add_argument("--pos_normalizer", type=float, default=2.0)    
    parser.add_argument("--ligand_atom_mode", type=str, default="add_aromatic", choices=["basic", "basic_PDB", "basic_plus_charge_PDB", "add_aromatic", "add_aromatic_plus_charge", "basic_plus_aromatic", "basic_plus_full", "basic_plus_charge", "full"])
    parser.add_argument('--time_decoupled', action='store_true')
    parser.add_argument('--decouple_mode', type=str, default='none')
    
    # train params
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument('--v_loss_weight', type=float, default=1)
    parser.add_argument('--bond_loss_weight', type=float, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['cosine', 'plateau'])
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_grad_norm', type=str, default='Q')  # '8.0' for

    # bfn params
    parser.add_argument("--sigma1_coord", type=float, default=0.05)
    parser.add_argument("--beta1", type=float, default=1.0)
    parser.add_argument("--beta1_bond", type=float, default=1.0)
    parser.add_argument("--beta1_charge", type=float, default=1.5)
    parser.add_argument("--beta1_aromatic", type=float, default=3.0)
    # parser.add_argument("--no_diff_coord", type=eval, default=False)
    # parser.add_argument("--charge_discretised_loss", type=eval, default=False)
    parser.add_argument("--t_min", type=float, default=0.0001)
    parser.add_argument('--use_discrete_t', type=eval, default=True)
    parser.add_argument('--discrete_steps', type=int, default=1000)
    parser.add_argument('--destination_prediction', type=eval, default=True)
    parser.add_argument('--sampling_strategy', type=str, default='end_back_pmf') #vanilla or end_back

    # network params
    parser.add_argument(
        "--time_emb_mode", type=str, default="simple", choices=["simple", "sin", 'rbf', 'rbfnn']
    )
    parser.add_argument("--time_emb_dim", type=int, default=1)
    parser.add_argument('--pos_init_mode', type=str, default='zero', choices=['zero', 'randn'])
    parser.add_argument('--bond_net_type', type=str, default='lin', choices=['lin', 'pre_att', 'flowmol', 'semla', 'lin+x'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_blocks', type=int, default=1)
    parser.add_argument('--pred_given_all', action='store_true')
    parser.add_argument('--pred_connectivity', action='store_true')
    parser.add_argument('--self_condition', action='store_true')
    parser.add_argument('--adaptive_norm', type=eval, default=False)

    # eval params
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--sample_steps", type=int, default=100)
    parser.add_argument('--sample_num_atoms', type=str, default='ref', choices=['prior', 'ref'])
    parser.add_argument("--visual_chain", action="store_true")
    parser.add_argument("--best_ckpt", type=str, default="val_loss", choices=["mol_stable", "complete", "val_loss"])
    parser.add_argument("--fix_bond", action="store_true")
    parser.add_argument("--pos_grad_weight", type=float, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=100)
    parser.add_argument("--skip_chem", action="store_true")
    parser.add_argument("--t_power", type=float, default=1.0)
    parser.add_argument("--time_scheduler_path", type=str, default=None)
    parser.add_argument("--time_coef", type=float, default=1.0)
    parser.add_argument("--protein_path", type=str, default=None)
    parser.add_argument("--ligand_path", type=str, default=None)

    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated sdf files.")
    parser.add_argument("--res_path", type=str, required=True, help="Path to save results json file.")


    _args = parser.parse_args()
    cfg = Config(**_args.__dict__)
    seed_everything(cfg.seed)

    logging_level = {
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "fatal": logging.FATAL,
    }
    logging.set_verbosity(logging_level[cfg.logging_level])

    if cfg.empty_folder:
        shutil.rmtree(cfg.accounting.logdir)

    wandb_logger = get_logger(cfg)
        
    
    tr_cfg = Config(cfg.accounting.dump_config_path)

    cfg.dynamics = tr_cfg.dynamics
    tr_cfg.data.name = cfg.data.name
    tr_cfg.data.path = cfg.data.path
    if hasattr(cfg.data, 'split'):
        tr_cfg.data.split = cfg.data.split
    if hasattr(cfg.data, 'version'):
        tr_cfg.data.version = cfg.data.version
    elif hasattr(tr_cfg.data, 'version'):
        del tr_cfg.data.version
    tr_cfg.data.smiles_path = cfg.data.smiles_path
    tr_cfg.data.with_split = cfg.data.with_split
    cfg.data = tr_cfg.data

    test_loader = get_dataloader_from_pdb(cfg)

    wandb_logger.log_hyperparams(cfg.todict())
    print(f"The config of this process is:\n{cfg}")

    model = SBDD4Train(config=cfg)

    trainer = pl.Trainer(
        default_root_dir=cfg.accounting.logdir,
        max_epochs=cfg.train.epochs,
        check_val_every_n_epoch=cfg.train.ckpt_freq,
        devices=1,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        callbacks=[
            RecoverCallback(
                latest_ckpt=os.path.join(cfg.accounting.checkpoint_dir, "last.ckpt"),
                resume=cfg.train.resume,
                recover_trigger_loss=1e7,
            ),
            # TODO: add data normalizing back?
            NormalizerCallback(normalizer_dict=cfg.data.normalizer_dict),
            CondMolGenValidationCallback(
                dataset=None,  # TODO: implement CrossDockGen & NewBenchmark
                atom_decoder=cfg.data.atom_decoder,
                atom_enc_mode=cfg.data.transform.ligand_atom_mode,
                atom_type_one_hot=False,
                single_bond=True,
                docking_config=cfg.evaluation.docking_config,
            ),
            MolVisualizationCallback(
                atom_decoder=cfg.data.atom_decoder,
                colors_dic=cfg.data.colors_dic,
                radius_dic=cfg.data.radius_dic,
            ),
            ReconValidationCallback(
                val_freq=cfg.train.val_freq,
            ),
            DockingTestCallback(
                dataset=None,  # TODO: implement CrossDockGen & NewBenchmark
                atom_decoder=cfg.data.atom_decoder,
                atom_enc_mode=cfg.data.transform.ligand_atom_mode,
                atom_type_one_hot=False,
                single_bond=True,
                docking_config=cfg.evaluation.docking_config,
            ),
            ModelCheckpoint(
                monitor="val/recon_loss",
                mode="min",
                # every_n_train_steps=cfg.train.val_freq,
                # monitor="val/mol_stable",
                # mode="max",
                every_n_epochs=cfg.train.ckpt_freq,
                dirpath=cfg.accounting.checkpoint_dir,
                filename="epoch{epoch:02d}-val_loss{val/recon_loss:.2f}-mol_stable{val/mol_stable:.2f}-complete{val/completeness:.2f}",
                save_top_k=3,
                auto_insert_metric_name=False,
                save_last=True,
            ),
            EMACallback(decay=cfg.train.ema_decay, ema_device="cuda"),
            # DebugCallback(),
            # LearningRateMonitor(logging_interval='step'),
        ],
    )

    ckpts = glob.glob(os.path.join(cfg.accounting.checkpoint_dir, "*mol_stable*"))
    if not hasattr(cfg, "best_ckpt") or cfg.best_ckpt == "val_loss":
        best_ckpt = sorted(ckpts, key=lambda x: float(x.split("val_loss")[-1][:4]))[0]
    elif cfg.best_ckpt == "complete":
        ckpts = glob.glob(os.path.join(cfg.accounting.checkpoint_dir, "*complete*"))
        best_ckpt = sorted(ckpts, key=lambda x: float(x.split("complete")[-1][:4]))[-1]
    else:
        best_ckpt = sorted(ckpts, key=lambda x: float(x.split("mol_stable")[-1][:4]))[-1]
    print(f"Detected best_ckpt: {best_ckpt}")

    checkpoint = torch.load(best_ckpt)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    print(f"Loaded model from {best_ckpt}")
    if hasattr(cfg.evaluation, 'time_scheduler_path') and cfg.evaluation.time_scheduler_path is not None:
        if hasattr(model, 'timewarp_cdf'):
            model.timewarp_cdf = None
        time_scheduler = torch.load(cfg.evaluation.time_scheduler_path)
        time_scheduler = torch.from_numpy(time_scheduler)
        model.time_scheduler = time_scheduler
        print(f"Loaded time scheduler from {cfg.evaluation.time_scheduler_path}")

    cfg.evaluation.output_dir = _args.output_dir
    cfg.evaluation.res_path = _args.res_path

    if not hasattr(cfg.evaluation, 'fix_bond'):
        cfg.evaluation.fix_bond = False

    os.makedirs(os.path.dirname(cfg.evaluation.res_path), exist_ok=True)
    os.makedirs(cfg.evaluation.output_dir, exist_ok=True)

    trainer.test(model, dataloaders=test_loader)

    results = []
    for sdf_file in sorted(os.listdir(cfg.evaluation.output_dir)):
        if not sdf_file.endswith(".sdf"):
            continue
        sdf_path = os.path.join(cfg.evaluation.output_dir, sdf_file)
        print(f"[INFO] Evaluating {sdf_path}")
        metrics = Metrics(cfg.evaluation.protein_path, cfg.evaluation.ligand_path, sdf_path).evaluate()
        metrics['name'] = sdf_file
        results.append(metrics)

    os.makedirs(os.path.dirname(cfg.evaluation.res_path), exist_ok=True)
    with open(cfg.evaluation.res_path, "w") as f:
        json.dump(results, f, indent=4, cls=NpEncoder)
    print(f"[INFO] Results saved to {cfg.evaluation.res_path}")

