import argparse
import os
import shutil

import torch

from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

import datetime, pytz

from core.config.config import Config, parse_config
from core.models.train_loop import ClassifierTrainLoop
from core.callbacks.basic import RecoverCallback, GradientClip, NormalizerCallback, EMACallback
from core.callbacks.validation_callback import (
    ClassifierValidationCallback,
)

import core.utils.transforms as trans
from core.datasets import get_dataset
from core.datasets.pl_data import FOLLOW_BATCH

import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import seed_everything
from pytorch_lightning.profilers import SimpleProfiler, PyTorchProfiler

from absl import logging
import glob


def get_dataloader(cfg):
    if cfg.data.name == 'pl_tr':
        dataset, subsets = get_dataset(config=cfg.data, version='guided_v3')
        train_set, test_set = subsets['train'], subsets['test']        
        cfg.dynamics.protein_atom_feature_dim = dataset.protein_atom_feature_dim
        cfg.dynamics.ligand_atom_feature_dim = dataset.ligand_atom_feature_dim
    else:
        protein_featurizer = trans.FeaturizeProteinAtom()
        ligand_featurizer = trans.FeaturizeLigandAtom(cfg.data.transform.ligand_atom_mode)
        transform_list = [
            protein_featurizer,
            ligand_featurizer,
            # trans.FeaturizeLigandBond(),
            trans.NormalizeVina(),
            trans.AddMolProp(),
        ]
        transform = Compose(transform_list)
        cfg.dynamics.protein_atom_feature_dim = protein_featurizer.feature_dim
        cfg.dynamics.ligand_atom_feature_dim = ligand_featurizer.feature_dim
        dataset, subsets = get_dataset(config=cfg.data, transform=transform)
        train_set, test_set = subsets['train'], subsets['test']
    if 'val' in subsets and len(subsets['val']) > 0:
        val_set = subsets['val']
    else:
        val_set = test_set
    
    # TODO: temp fix for extremely long validation time
    val_set = test_set

    print(f"protein feature dim: {cfg.dynamics.protein_atom_feature_dim}, " +
          f"ligand feature dim: {cfg.dynamics.ligand_atom_feature_dim}")
    
    # follow_batch = ['protein_element', 'ligand_element']
    collate_exclude_keys = ["ligand_nbh_list"]
    # size-1 debug set
    if cfg.debug:
        debug_set = torch.utils.data.Subset(val_set, [0] * 1600)
        debug_set_val = torch.utils.data.Subset(val_set, [0] * 10)

        # get debug set from train set
        if cfg.test_only:
            print(f"debug test set: {len(debug_set)}")
            split = torch.load(cfg.data.split)
            debug_set_val = torch.utils.data.Subset(dataset, split['train'][:1000])

        # get debug set val data batch
        # debug_batch_val = next(iter(DataLoader(debug_set_val, batch_size=cfg.train.batch_size, shuffle=False)))
        # print(f"debug batch val: {debug_batch_val.ligand_filename}")
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
        test_loader = DataLoader(
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
        )
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.evaluation.batch_size,
            shuffle=False,
            follow_batch=FOLLOW_BATCH,
            exclude_keys=collate_exclude_keys
        )
        test_loader = DataLoader(
            test_set,
            batch_size=cfg.evaluation.batch_size,
            shuffle=False,
            follow_batch=FOLLOW_BATCH,
            exclude_keys=collate_exclude_keys
        )
    cfg.train.scheduler.max_iters = cfg.train.epochs * len(train_loader)

    return train_loader, val_loader, test_loader


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # meta
    parser.add_argument("--config_file", type=str, default="configs/train_prop.yaml",)
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--revision", type=str, default="default")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb_resume_id", type=str, default=None)
    parser.add_argument('--empty_folder', action='store_true')
    parser.add_argument("--test_only", action="store_true")
    
    # global config
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--logging_level", type=str, default="warning")

    # train data params
    parser.add_argument('--random_rot', action='store_true')
    parser.add_argument("--pos_noise_std", type=float, default=0)    
    parser.add_argument("--pos_normalizer", type=float, default=2.0)    
    parser.add_argument("--prop_name", type=str, default='affinity', 
                        choices=['affinity', 'qed', 'sa', 'qed_norm', 'sa_norm', 'lipinski_norm',
                                 'HDonor', 'HAccep', 'Pi', 'XBond', 'Salt'])
    
    # train params
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument('--v_loss_weight', type=float, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['cosine', 'plateau'])
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_grad_norm', type=str, default='Q')  # '8.0' for

    # bfn params
    parser.add_argument("--sigma1_coord", type=float, default=0.03)
    parser.add_argument("--beta1", type=float, default=1.5)
    # parser.add_argument("--no_diff_coord", type=eval, default=False)
    # parser.add_argument("--charge_discretised_loss", type=eval, default=False)
    parser.add_argument("--t_min", type=float, default=0.0001)
    parser.add_argument('--use_discrete_t', type=eval, default=True)
    parser.add_argument('--discrete_steps', type=int, default=1000)
    parser.add_argument('--destination_prediction', type=eval, default=True)
    parser.add_argument('--sampling_strategy', type=str, default='end_back_pmf') #vanilla or end_back

    parser.add_argument(
        "--time_emb_mode", type=str, default="simple", choices=["simple", "sin", 'rbf', 'rbfnn']
    )
    parser.add_argument("--time_emb_dim", type=int, default=1)
    parser.add_argument('--pos_init_mode', type=str, default='zero', choices=['zero', 'randn'])
    parser.add_argument('--num_layers', type=int, default=9)

    # eval params
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--sample_steps", type=int, default=100)
    parser.add_argument('--sample_num_atoms', type=str, default='prior', choices=['prior', 'ref'])
    parser.add_argument("--visual_chain", action="store_true")
    parser.add_argument("--last_ckpt", action="store_true")
    parser.add_argument("--docking_mode", type=str, default="vina_score", choices=['vina_score', 'vina_dock'])
    parser.add_argument("--save_traj", action="store_true")

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
        tr_cfg.test_only = cfg.test_only
        tr_cfg.evaluation = cfg.evaluation
        tr_cfg.visual = cfg.visual
        tr_cfg.accounting = cfg.accounting
        tr_cfg.dynamics.beta1 = cfg.dynamics.beta1
        tr_cfg.dynamics.sigma1_coord = cfg.dynamics.sigma1_coord
        tr_cfg.dynamics.sampling_strategy = cfg.dynamics.sampling_strategy
        tr_cfg.seed = cfg.seed
        tr_cfg.data.name = 'pl'
        tr_cfg.debug = cfg.debug
        cfg = tr_cfg
        if not hasattr(cfg.train, 'max_grad_norm'):
            cfg.train.max_grad_norm = 'Q'
    else:
        cfg.save2yaml(cfg.accounting.dump_config_path)

    train_loader, val_loader, test_loader = get_dataloader(cfg)
    wandb_logger.log_hyperparams(cfg.todict())
    print(f"The config of this process is:\n{cfg}")

    model = ClassifierTrainLoop(config=cfg)
    callbacks = [
        RecoverCallback(
            latest_ckpt=os.path.join(cfg.accounting.checkpoint_dir, "last.ckpt"),
            resume=cfg.train.resume,
            recover_trigger_loss=1e7,
        ),
        GradientClip(max_grad_norm=cfg.train.max_grad_norm),  # time consuming
        NormalizerCallback(normalizer_dict=cfg.data.normalizer_dict),
        ModelCheckpoint(
            monitor="val/recon_loss",
            every_n_epochs=cfg.train.ckpt_freq,
            dirpath=cfg.accounting.checkpoint_dir,
            filename="epoch{epoch:02d}-val_loss{val/recon_loss:.2f}",
            save_top_k=5,
            mode="min",
            auto_insert_metric_name=False,
            save_last=True,
        ),
        EMACallback(decay=cfg.train.ema_decay, ema_device="cuda"),
        ClassifierValidationCallback(
            val_freq=cfg.train.val_freq,
        ),
    ]

    trainer = pl.Trainer(
        default_root_dir=cfg.accounting.logdir,
        max_epochs=cfg.train.epochs,
        check_val_every_n_epoch=cfg.train.ckpt_freq,
        devices=1,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        # overfit_batches=10,
        # gradient_clip_val=1.0,
        # devices=1,
        callbacks=callbacks,
    )

    if cfg.test_only:
        trainer.test(model, dataloaders=test_loader)
    else:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(model, dataloaders=test_loader)