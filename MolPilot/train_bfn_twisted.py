import argparse
import os
import shutil

import torch

from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

import datetime, pytz
import wandb

from core.config.config import Config, parse_config
from core.models.sbdd4train import SBDD4Train
from core.callbacks.basic import RecoverCallback, GradientClip, NormalizerCallback #, EMACallback
from core.callbacks.ema import EMACallback
from core.callbacks.validation_callback import (
    CondMolGenValidationCallback,
    MolVisualizationCallback,
    TwistedReconValidationCallback,
    DockingTestCallback,
)

import core.utils.transforms as trans
from core.datasets import get_dataset
from core.datasets.pl_data import FOLLOW_BATCH

import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning import seed_everything
from pytorch_lightning.profilers import SimpleProfiler, PyTorchProfiler

from absl import logging
import glob
import pickle as pkl
from tqdm import tqdm


def get_dataloader(cfg):
    if cfg.data.name == 'pl_tr':
        dataset, subsets = get_dataset(config=cfg.data)
        train_set, val_set = subsets['train'], subsets['test']        
        cfg.dynamics.protein_atom_feature_dim = dataset.protein_atom_feature_dim
        cfg.dynamics.ligand_atom_feature_dim = dataset.ligand_atom_feature_dim
    else:
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
        cfg.dynamics.ligand_atom_type_dim = ligand_featurizer.type_feature_dim
        cfg.dynamics.ligand_atom_charge_dim = ligand_featurizer.charge_feature_dim
        cfg.dynamics.ligand_atom_aromatic_dim = ligand_featurizer.aromatic_feature_dim

        dataset, subsets = get_dataset(config=cfg.data, transform=transform)
        train_set, test_set = subsets['train'], subsets['test']

    if 'val' in subsets and len(subsets['val']) > 0:
        val_set = subsets['val']
    elif 'valid' in subsets and len(subsets['valid']) > 0:
        val_set = subsets['valid']
    else:
        val_set = test_set

    if len(train_set) == 0:
        assert cfg.test_only, "No training data found"
        train_set = val_set
    print(f"protein feature dim: {cfg.dynamics.protein_atom_feature_dim}, " +
          f"ligand feature dim: {cfg.dynamics.ligand_atom_feature_dim}")

    if cfg.train.ckpt_freq > 1:
        # repeat train set for ckpt_freq times
        train_set = torch.utils.data.ConcatDataset([train_set] * cfg.train.ckpt_freq)
        cfg.train.ckpt_freq = 1
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    # if using multiple GPUs, the batch size should be divided by the number of GPUs
    # and use DistributedSampler
    if torch.cuda.device_count() > 1:
        cfg.train.batch_size = cfg.train.batch_size // torch.cuda.device_count()

    dataset_smiles_set = compute_or_retrieve_dataset_smiles(train_set, cfg.data.smiles_path)

    # follow_batch = ['protein_element', 'ligand_element']
    collate_exclude_keys = ["ligand_nbh_list"]
    # size-1 debug set
    if cfg.debug:
        # debug_id = 9618 # 5000 (29)
        # debug_id = 29008 # 4000 (75)
        debug_id = 0
        debug_set = torch.utils.data.Subset(test_set, [debug_id] * 100) #[0] * 1600)
        debug_set_val = torch.utils.data.Subset(test_set, list(range(10)))
        debug_batch_val = next(iter(DataLoader(debug_set_val, batch_size=cfg.train.batch_size, shuffle=False)))
        train_loader = DataLoader(debug_set,
            batch_size=cfg.train.batch_size,
            shuffle=False,  # set shuffle to False 
            num_workers=cfg.train.num_workers,
            follow_batch=FOLLOW_BATCH,
            exclude_keys=collate_exclude_keys
        )
        val_loader = DataLoader(
            debug_set_val, 
            batch_size=cfg.evaluation.batch_size,
            shuffle=False,
            follow_batch=FOLLOW_BATCH, 
            exclude_keys=collate_exclude_keys
        )
    else:
        logging.info(f"Training: {len(train_set)} Validation: {len(val_set)}")
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.train.num_workers,
            follow_batch=FOLLOW_BATCH,
            exclude_keys=collate_exclude_keys,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.evaluation.batch_size,
            shuffle=False,
            follow_batch=FOLLOW_BATCH,
            exclude_keys=collate_exclude_keys
        )
    cfg.train.scheduler.max_iters = cfg.train.epochs * len(train_loader)

    return train_loader, val_loader, dataset_smiles_set


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
            name=f"{cfg.exp_name}_{str(cfg.revision)}"
            + f'_{datetime.datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y-%m-%d-%H:%M:%S")}',
            project=cfg.project_name,
            offline=cfg.no_wandb,
            save_dir=cfg.accounting.wandb_logdir,
        )  # add wandb parameters
    return wandb_logger

def compute_or_retrieve_dataset_smiles(
    dataset, save_path
):
    # create parent directory if it does not exist
    if save_path is None:
        return None
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    if not os.path.exists(save_path):
        all_smiles = []
        for data in tqdm(dataset, desc="SMILES"):
            all_smiles.append(data.ligand_smiles)
        print(f"Saving {len(all_smiles)} smiles to {save_path}")
        with open(save_path, "wb") as f:
            pkl.dump(all_smiles, f)
    else:
        print("Loading all smiles")
        with open(save_path, "rb") as f:
            all_smiles = pkl.load(f)
    smiles_set = set([s for s in all_smiles])
    return smiles_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # meta
    parser.add_argument("--config_file", type=str, default="configs/crossdock_train_test.yaml",)
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--revision", type=str, default="default")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb_resume_id", type=str, default=None)
    parser.add_argument('--empty_folder', action='store_true')
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default=None)
    
    # global config
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--logging_level", type=str, default="warning")

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
        
    if cfg.test_only:
        tr_cfg = Config(cfg.accounting.dump_config_path)
        tr_cfg.dynamics.net_config.num_blocks = cfg.dynamics.net_config.num_blocks
        
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
        if hasattr(tr_cfg, 'time_decoupled'):
            cfg.time_decoupled = tr_cfg.time_decoupled
        if hasattr(tr_cfg, 'decouple_mode'):
            cfg.decouple_mode = tr_cfg.decouple_mode

    elif hasattr(cfg, 'ckpt_path') and cfg.ckpt_path is not None:
        config_path = os.path.join('/'.join(cfg.ckpt_path.split('/')[:-2]), "config.yaml")
        assert os.path.exists(config_path), f"config file {config_path} not found"
        model_config = Config(config_path)
        cfg.dynamics = model_config.dynamics
        model_config.data.name = cfg.data.name
        if hasattr(cfg.data, 'version'):
            model_config.data.version = cfg.data.version
        model_config.data.path = cfg.data.path
        model_config.data.smiles_path = cfg.data.smiles_path
        model_config.data.with_split = cfg.data.with_split
        model_config.data.split = cfg.data.split
        cfg.data = model_config.data
        cfg.save2yaml(cfg.accounting.dump_config_path)
    else:
        cfg.save2yaml(cfg.accounting.dump_config_path)

    train_loader, val_loader, dataset_smiles_set = get_dataloader(cfg)
    
    wandb_logger.log_hyperparams(cfg.todict())
    print(f"The config of this process is:\n{cfg}")

    model = SBDD4Train(config=cfg)
    if cfg.train.resume:
        cfg.ckpt_path = os.path.join(cfg.accounting.checkpoint_dir, "last.ckpt")

    if hasattr(cfg, 'ckpt_path') and cfg.ckpt_path is not None:
        # model = SBDD4Train(config=model_config)
        checkpoint = torch.load(cfg.ckpt_path)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded model from {cfg.ckpt_path}")

    if hasattr(cfg.evaluation, "time_scheduler_path") and cfg.evaluation.time_scheduler_path is not None:
        time_scheduler = torch.load(cfg.evaluation.time_scheduler_path)
        time_scheduler = torch.from_numpy(time_scheduler).float()
        model.configure_time_scheduler(time_scheduler)
        print(f"Loaded time scheduler from {cfg.evaluation.time_scheduler_path}")


    callbacks = [
            RecoverCallback(
                latest_ckpt=os.path.join(cfg.accounting.checkpoint_dir, "last.ckpt"),
                resume=cfg.train.resume,
                recover_trigger_loss=1e7,
            ),
            # TODO: this seems a dynamic clip, turn to static?
            GradientClip(max_grad_norm=cfg.train.max_grad_norm),  # time consuming
            # TODO: add data normalizing back?
            NormalizerCallback(normalizer_dict=cfg.data.normalizer_dict),

            MolVisualizationCallback(
                # dataset=train_loader.loader.ds,
                # atom_enc_mode=cfg.data.transform.ligand_atom_mode,
                atom_decoder=cfg.data.atom_decoder,
                colors_dic=cfg.data.colors_dic,
                radius_dic=cfg.data.radius_dic,
            ),
            TwistedReconValidationCallback(
                val_freq=min(cfg.train.val_freq, len(train_loader)),
            ),
            CondMolGenValidationCallback(
                dataset=None,  # TODO: implement CrossDockGen & NewBenchmark
                # atom_decoder={1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl'},
                atom_decoder=cfg.data.atom_decoder,
                atom_enc_mode=cfg.data.transform.ligand_atom_mode,
                atom_type_one_hot=False,
                single_bond=False,
                docking_config=None if 'zinc' in cfg.data.path else cfg.evaluation.docking_config,
                dataset_smiles_set=dataset_smiles_set,
                # single_bond=cfg.evaluation.single_bond,  # TODO: check compatibility
            ),
            DockingTestCallback(
                dataset=None,  # TODO: implement CrossDockGen & NewBenchmark
                atom_decoder=cfg.data.atom_decoder,
                atom_enc_mode=cfg.data.transform.ligand_atom_mode,
                atom_type_one_hot=False,
                single_bond=False,
                docking_config=None if 'zinc' in cfg.data.path else cfg.evaluation.docking_config,
                dataset_smiles_set=dataset_smiles_set,
                docking_rmsd=getattr(cfg.evaluation, 'docking_rmsd', False),
            ),
            ModelCheckpoint(
                monitor="val/recon_loss",
                mode="min",
                # monitor="val/mol_stable",
                # mode="max",
                every_n_epochs=cfg.train.ckpt_freq,
                # every_n_train_steps=cfg.train.val_freq,
                dirpath=cfg.accounting.checkpoint_dir,
                filename="epoch{epoch:02d}-val_loss{val/recon_loss:.2f}-mol_stable{val/mol_stable:.2f}-complete{val/completeness:.2f}",
                save_top_k=3,
                auto_insert_metric_name=False,
                save_last=True,
            ),
            EMACallback(decay=cfg.train.ema_decay),
            EarlyStopping(monitor='val/recon_loss', mode='min', patience=cfg.train.scheduler.patience * 2),
            # EMACallback(decay=cfg.train.ema_decay, ema_device="cuda" if torch.cuda.is_available() else "cpu"),
            # DebugCallback(),
            # LearningRateMonitor(logging_interval='step'),
        ]
    

    trainer = pl.Trainer(
        default_root_dir=cfg.accounting.logdir,
        max_epochs=cfg.train.epochs,
        check_val_every_n_epoch=cfg.train.ckpt_freq,
        devices=1,
        # devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        strategy="auto",
        # accumulate_grad_batches=2,
        # overfit_batches=10,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        # overfit_batches=10,
        # gradient_clip_val=1.0,
        callbacks=callbacks,
    )
    # num_sanity_val_steps=2, overfit_batches=10, devices=1
    # trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=eval_loader)

    if not cfg.test_only:
        model.dynamics.sampling_strategy = 'vanilla'
        # if cfg.train.resume:
        #     trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path="last")
        # else:
        # wandb.watch(model.dynamics, log='all')
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        model.dynamics.sampling_strategy = cfg.dynamics.sampling_strategy
        if torch.cuda.device_count() > 1:
            trainer.devices = [0]
            callbacks.append(
                DockingTestCallback(
                    dataset=None,  # TODO: implement CrossDockGen & NewBenchmark
                    atom_decoder=cfg.data.atom_decoder,
                    atom_enc_mode=cfg.data.transform.ligand_atom_mode,
                    atom_type_one_hot=False,
                    single_bond=False,
                    docking_config=None if 'zinc' in cfg.data.path else cfg.evaluation.docking_config,
                    dataset_smiles_set=dataset_smiles_set,
                    docking_rmsd=getattr(cfg.evaluation, 'docking_rmsd', False),
                )
            )
            trainer.callbacks = callbacks
        trainer.test(model, dataloaders=val_loader, ckpt_path="best")
    else:
        ckpts = glob.glob(os.path.join(cfg.accounting.checkpoint_dir, "*val_loss*.ckpt"))
        if not hasattr(cfg, "best_ckpt") or cfg.best_ckpt == "val_loss":
            best_ckpt = sorted(ckpts, key=lambda x: float(x.split("val_loss")[-1][:4]))[0]
        elif cfg.best_ckpt == "complete":
            ckpts = glob.glob(os.path.join(cfg.accounting.checkpoint_dir, "*complete*"))
            best_ckpt = sorted(ckpts, key=lambda x: float(x.split("complete")[-1][:4]))[-1]
        else:
            best_ckpt = sorted(ckpts, key=lambda x: float(x.split("mol_stable")[-1][:4]))[-1]
        print(f"Detected best_ckpt: {best_ckpt} from {ckpts}")
        trainer.test(model, dataloaders=val_loader, ckpt_path=best_ckpt)
        # trainer.test(model, dataloaders=val_loader, ckpt_path="last")

