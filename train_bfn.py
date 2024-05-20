import argparse
import os
import shutil

import torch

from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

import datetime, pytz

from core.config.config import Config, parse_config
from core.models.sbdd_train_loop import SBDDTrainLoop
from core.callbacks.basic import RecoverCallback, GradientClip, NormalizerCallback, EMACallback
from core.callbacks.validation_callback import (
    ValidationCallback,
    VisualizeMolAndTrajCallback,
    DockingTestCallback,
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
        dataset, subsets = get_dataset(config=cfg.data)
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
    

    print(f"protein feature dim: {cfg.dynamics.protein_atom_feature_dim}, " +
          f"ligand feature dim: {cfg.dynamics.ligand_atom_feature_dim}")
    
    collate_exclude_keys = ["ligand_nbh_list"]
    # size-1 debug set
    if cfg.debug:
        debug_set = torch.utils.data.Subset(val_set, [0] * 800)
        debug_set_val = torch.utils.data.Subset(val_set, [0] * 10)
        cfg.train.val_freq = 100
        # get debug set val data batch
        debug_batch_val = next(iter(DataLoader(debug_set_val, batch_size=cfg.train.batch_size, shuffle=False)))
        print(f"debug batch val: {debug_batch_val.ligand_filename}")
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


def set_test_output_dir(cfg):
    path = cfg.accounting.test_outputs_dir
    version = 0
    while os.path.exists(path):
        version += 1
        path = cfg.accounting.test_outputs_dir + f'_v{version}'
    print(f'{cfg.accounting.test_outputs_dir} already exists, change test_output_dir to {path}')
    cfg.accounting.test_outputs_dir = path
    os.makedirs(cfg.accounting.test_outputs_dir, exist_ok=True)


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
    parser.add_argument("--config_file", type=str, default="configs/default.yaml",)
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
    parser.add_argument("--t_min", type=float, default=0.0001)
    parser.add_argument('--use_discrete_t', type=eval, default=True)
    parser.add_argument('--discrete_steps', type=int, default=1000)
    parser.add_argument('--destination_prediction', type=eval, default=True)
    parser.add_argument('--sampling_strategy', type=str, default='end_back_pmf', choices=['vanilla', 'end_back_pmf']) #vanilla or end_back

    parser.add_argument(
        "--time_emb_mode", type=str, default="simple", choices=["simple", "sin", 'rbf', 'rbfnn']
    )
    parser.add_argument("--time_emb_dim", type=int, default=1)
    parser.add_argument('--pos_init_mode', type=str, default='zero', choices=['zero', 'randn'])

    # eval params
    parser.add_argument('--ckpt_path', type=str, default='best', help='path to the checkpoint')
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--sample_steps", type=int, default=100)
    parser.add_argument('--sample_num_atoms', type=str, default='ref', choices=['prior', 'ref'])
    parser.add_argument("--visual_chain", action="store_true")
    parser.add_argument("--docking_mode", type=str, default="vina_score", choices=['vina_score', 'vina_dock'])

    _args = parser.parse_args()
    if _args.ckpt_path.lstrip('./') == 'checkpoints/last.ckpt':
        _args.exp_name = 'official'
        _args.revision = 'default'
    else:
        print('trying to automatically parse experiment folder...')
        try:
            *_, exp_name, revision, _, ckpt_fn = _args.ckpt_path.split('/')
            _args.exp_name = exp_name
            _args.revision = revision
            print(f'change log dir to **/{exp_name}/{revision}')
        except Exception as e:
            pass

    cfg = Config(**_args.__dict__)
    if not os.path.exists(cfg.accounting.logdir):
        os.makedirs(cfg.accounting.logdir, exist_ok=True)
        # shutil.copyfile('./configs/default.yaml', cfg.accounting.dump_config_path)
    set_test_output_dir(cfg)

    seed_everything(cfg.seed, workers=True)

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
        if os.path.exists(cfg.accounting.dump_config_path):
            # reload training config
            tr_cfg = Config(cfg.accounting.dump_config_path)
            tr_cfg.test_only = cfg.test_only
            tr_cfg.evaluation = cfg.evaluation
            tr_cfg.visual = cfg.visual
            tr_cfg.accounting = cfg.accounting
            # TODO: temporarily test different beta1 and sigma1_coord
            tr_cfg.dynamics.beta1 = cfg.dynamics.beta1
            tr_cfg.dynamics.sigma1_coord = cfg.dynamics.sigma1_coord
            tr_cfg.dynamics.sampling_strategy = cfg.dynamics.sampling_strategy
            if hasattr(cfg.dynamics, 'guide_mode'):
                tr_cfg.dynamics.guide_mode = cfg.dynamics.guide_mode
                tr_cfg.dynamics.objective = cfg.dynamics.objective
                tr_cfg.dynamics.pos_grad_weight = cfg.dynamics.pos_grad_weight
                tr_cfg.dynamics.type_grad_weight = cfg.dynamics.type_grad_weight
                if cfg.dynamics.guide_mode is not None:
                    tr_cfg.evaluation.batch_size = 4
            tr_cfg.data = cfg.data
            tr_cfg.seed = cfg.seed
            cfg = tr_cfg
        if not hasattr(cfg.train, 'max_grad_norm'):
            cfg.train.max_grad_norm = 'Q'
        cfg.save2yaml(os.path.join(cfg.accounting.test_outputs_dir, 'config.yaml'))
    else:
        cfg.save2yaml(cfg.accounting.dump_config_path)

    train_loader, val_loader, test_loader = get_dataloader(cfg)
    wandb_logger.log_hyperparams(cfg.todict())
    print(f"The config of this process is:\n{cfg}")

    model = SBDDTrainLoop(config=cfg)

    trainer = pl.Trainer(
        # deterministic=True,
        default_root_dir=cfg.accounting.logdir,
        max_epochs=cfg.train.epochs,
        check_val_every_n_epoch=cfg.train.ckpt_freq,
        devices=1,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        inference_mode=not cfg.test_only,
        callbacks=[
            RecoverCallback(
                latest_ckpt=os.path.join(cfg.accounting.checkpoint_dir, "last.ckpt"),
                resume=cfg.train.resume,
                recover_trigger_loss=1e7,
            ),
            GradientClip(max_grad_norm=cfg.train.max_grad_norm),  # time consuming
            NormalizerCallback(normalizer_dict=cfg.data.normalizer_dict),
            ValidationCallback(
                dataset=None,  # TODO: implement CrossDockGen & NewBenchmark
                atom_decoder=cfg.data.atom_decoder,
                atom_enc_mode=cfg.data.transform.ligand_atom_mode,
                atom_type_one_hot=False,
                single_bond=True,
                docking_config=cfg.evaluation.docking_config,
                val_freq=cfg.train.val_freq,
                # single_bond=cfg.evaluation.single_bond,  # TODO: check compatibility
            ),
            # VisualizeMolAndTrajCallback(
            #     atom_decoder=cfg.data.atom_decoder,
            #     colors_dic=cfg.data.colors_dic,
            #     radius_dic=cfg.data.radius_dic,
            # ),
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
                every_n_epochs=cfg.train.ckpt_freq,
                dirpath=cfg.accounting.checkpoint_dir,
                filename="epoch{epoch:02d}-val_loss{val/recon_loss:.2f}-mol_stable{val/mol_stable:.2f}-complete{val/completeness:.2f}-vina_score{val/vina_score_mean:.2f}",
                save_top_k=-1,
                mode="min",
                auto_insert_metric_name=False,
                save_last=True,
            ),
            EMACallback(decay=cfg.train.ema_decay, ema_device="cuda"),
        ],
    )

    # TODO split sample and evaluation

    if not cfg.test_only:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        # ckpt_path can be 'best', 'last', or a specific path
        trainer.test(model, dataloaders=test_loader, ckpt_path=cfg.evaluation.ckpt_path)
    else:
        trainer.test(model, dataloaders=test_loader, ckpt_path=cfg.evaluation.ckpt_path)

