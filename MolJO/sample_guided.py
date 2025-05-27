import argparse
import os
import shutil

import torch

from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

import datetime, pytz

from core.config.config import Config, parse_config
from core.models.train_loop import BFNTrainLoop, ClassifierTrainLoop
from core.callbacks.basic import RecoverCallback, GradientClip, NormalizerCallback, EMACallback
from core.callbacks.validation_callback import (
    CondMolGenValidationCallback,
    MolVisualizationCallback,
    ReconValidationCallback,
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
        if cfg.test_only:
            # transform_list.extend([
            #     trans.NormalizeVina(),
            #     trans.AddMolProp(),
            # ])
            if 'inpainting' in cfg.evaluation.sample_num_atoms:
                transform_list.append(trans.AddScaffoldMask('/sharefs/share/sbdd_data/Mask_cd_test.pkl', cfg.evaluation.change_scaffold))
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
    
    # follow_batch = ['protein_element', 'ligand_element']
    collate_exclude_keys = ["ligand_nbh_list"]
    # size-1 debug set
    if cfg.debug:
        debug_set = torch.utils.data.Subset(val_set, [0] * 1600)
        debug_set_val = torch.utils.data.Subset(val_set, [0] * 5)
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
        test_loader = DataLoader(
            test_set,
            batch_size=cfg.evaluation.batch_size,
            shuffle=False,
            follow_batch=FOLLOW_BATCH,
            exclude_keys=collate_exclude_keys
        )
        train_loader, val_loader = test_loader, test_loader
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
    parser.add_argument("--config_file", type=str, default="configs/test_opt.yaml",)
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--revision", type=str, default="default")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb_resume_id", type=str, default=None)
    parser.add_argument('--empty_folder', action='store_true')
    parser.add_argument("--test_only", type=eval, default=True)
    
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

    # eval params
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--sample_steps", type=int, default=200)
    parser.add_argument('--sample_num_atoms', type=str, default='prior', choices=['prior', 'ref', 'prior_ref', 'inpainting_ref', 'inpainting_prior'])
    parser.add_argument("--visual_chain", action="store_true")
    parser.add_argument("--last_ckpt", action="store_true")
    parser.add_argument("--docking_mode", type=str, default="vina_score", choices=['vina_score', 'vina_dock'])
    parser.add_argument("--save_traj", action="store_true")
    parser.add_argument("--objective", type=str, default='vina_sa', choices=[None, 'vina', 'qed', 'sa', 'lipinski', 'qed_sa', 'vina_qed', 'vina_sa', 'vina_qed_sa', 'vina_qed_sa_lipinski'])
    parser.add_argument("--guide_mode", type=str, default='param_naive', choices=[None, 'param_naive', 'param_logit', 'param_logit_2', 'sample_one', 'sample_mode', 'sample_mean', 'data_naive', 'data_logit'])
    parser.add_argument('--pos_grad_weight', type=float, default=50)
    parser.add_argument('--type_grad_weight', type=float, default=50)
    parser.add_argument('--change_scaffold', action='store_true')
    parser.add_argument("--interpolate_coef", type=float, default=0.)
    parser.add_argument("--classifier_layer", type=int, default=None)
    
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

    work_dir = '/sharefs/share/opt_data'
    # if cfg.test_only:
    #     assert os.path.exists(os.path.join(work_dir, 'pretrained'))
    #     tr_cfg = Config(os.path.join(work_dir, 'pretrained/config.yaml'))
    # assert cfg.test_only and not cfg.evaluation.last_ckpt
    # else:
    #     raise NotImplementedError("Training not supported yet")
    #     cfg.save2yaml(cfg.accounting.dump_config_path)

    if ('sde' in cfg.dynamics.sampling_strategy or 'ode' in cfg.dynamics.sampling_strategy) and cfg.dynamics.sampling_strategy != 'end_back_ode':
        best_ckpt = os.path.join(work_dir, 'pretrained/epoch28-cont.ckpt')
    else:
        # best_ckpt = glob.glob(os.path.join(work_dir, 'pretrained/epoch10*'))[-1]
        ckpts = glob.glob(os.path.join(cfg.accounting.checkpoint_dir, "*val_loss*.ckpt"))
        if not hasattr(cfg, "best_ckpt") or cfg.best_ckpt == "val_loss":
            best_ckpt = sorted(ckpts, key=lambda x: float(x.split("val_loss")[-1][:4]))[0]
        elif cfg.best_ckpt == "complete":
            ckpts = glob.glob(os.path.join(cfg.accounting.checkpoint_dir, "*complete*"))
            best_ckpt = sorted(ckpts, key=lambda x: float(x.split("complete")[-1][:4]))[-1]
        else:
            best_ckpt = sorted(ckpts, key=lambda x: float(x.split("mol_stable")[-1][:4]))[-1]
        print(f"Detected best_ckpt: {best_ckpt} from {ckpts}")

        config_path = os.path.join('/'.join(best_ckpt.split('/')[:-2]), "config.yaml")
        tr_cfg = Config(config_path)

        tr_cfg.test_only = cfg.test_only
        tr_cfg.debug = cfg.debug
        tr_cfg.evaluation = cfg.evaluation
        tr_cfg.visual = cfg.visual
        tr_cfg.accounting = cfg.accounting
        # TODO: temporarily test different beta1 and sigma1_coord
        tr_cfg.dynamics.beta1 = cfg.dynamics.beta1
        tr_cfg.dynamics.sigma1_coord = cfg.dynamics.sigma1_coord
        tr_cfg.dynamics.sampling_strategy = cfg.dynamics.sampling_strategy
        tr_cfg.seed = cfg.seed
        tr_cfg.data = cfg.data
        if 'pl_tr' == tr_cfg.data.name:
            tr_cfg.data.name = 'pl'
        if not hasattr(tr_cfg, 'classifier') and hasattr(cfg, 'classifier'):
            tr_cfg.classifier = cfg.classifier
        if cfg.evaluation.type_grad_weight + cfg.evaluation.pos_grad_weight == 0:
            tr_cfg.evaluation.batch_size = 100
        cfg = tr_cfg
        
    assert os.path.exists(best_ckpt)
    print(f"Detected best_ckpt: {best_ckpt}")


    wandb_logger.log_hyperparams(cfg.todict())
    train_loader, val_loader, test_loader = get_dataloader(cfg)
    print(f"The config of this process is:\n{cfg}")


    model = BFNTrainLoop(config=cfg)
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
        DockingTestCallback(
            dataset=None,  # TODO: implement CrossDockGen & NewBenchmark
            atom_decoder=cfg.data.atom_decoder,
            atom_enc_mode=cfg.data.transform.ligand_atom_mode,
            atom_type_one_hot=False,
            single_bond=True,
            docking_config=cfg.evaluation.docking_config,
        ),
        ModelCheckpoint(
            # monitor="val/recon_loss",
            # every_n_train_steps=cfg.train.val_freq,
            monitor="val/mol_stable",
            every_n_epochs=cfg.train.ckpt_freq,
            dirpath=cfg.accounting.checkpoint_dir,
            filename="epoch{epoch:02d}-val_loss{val/recon_loss:.2f}-mol_stable{val/mol_stable:.2f}-complete{val/completeness:.2f}",
            save_top_k=5,
            mode="max",
            auto_insert_metric_name=False,
            save_last=True,
        ),
        EMACallback(decay=cfg.train.ema_decay, ema_device="cuda"),
    ]

    if not cfg.evaluation.save_traj:
        callbacks.extend([
            CondMolGenValidationCallback(
                dataset=None,  # TODO: implement CrossDockGen & NewBenchmark
                # atom_decoder={1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl'},
                atom_decoder=cfg.data.atom_decoder,
                atom_enc_mode=cfg.data.transform.ligand_atom_mode,
                atom_type_one_hot=False,
                single_bond=True,
                docking_config=cfg.evaluation.docking_config,
                # single_bond=cfg.evaluation.single_bond,  # TODO: check compatibility
            ),
            MolVisualizationCallback(
                # dataset=train_loader.loader.ds,
                # atom_enc_mode=cfg.data.transform.ligand_atom_mode,
                atom_decoder=cfg.data.atom_decoder,
                colors_dic=cfg.data.colors_dic,
                radius_dic=cfg.data.radius_dic,
            ),
            ReconValidationCallback(
                val_freq=cfg.train.val_freq,
            ),
        ])

    trainer = pl.Trainer(
        default_root_dir=cfg.accounting.logdir,
        max_epochs=cfg.train.epochs,
        check_val_every_n_epoch=cfg.train.ckpt_freq,
        devices=1,
        # overfit_batches=10,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        inference_mode=not cfg.test_only,
        # overfit_batches=10,
        # gradient_clip_val=1.0,
        # devices=1,
        callbacks=callbacks,
    )
    # num_sanity_val_steps=2, overfit_batches=10, devices=1
    # trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=eval_loader)
    
    classifier_list = []
    postfix = ''
    if hasattr(cfg, 'classifier') and hasattr(cfg.classifier, 'input_type'):
        if cfg.classifier.input_type == 'sample':
            postfix = '_noisy'
        elif cfg.classifier.input_type == 'data':
            postfix = '_x0'

    if cfg.evaluation.classifier_layer is not None:
        postfix = f'{cfg.evaluation.classifier_layer}'
    
    print(f"Postfix: {postfix}")

    if cfg.evaluation.objective is not None:
        for objective in cfg.evaluation.objective.split('_'):
            if objective == 'vina':
                obj_cfg = Config(os.path.join(work_dir, f'pretrained/affinity{postfix}.yaml'))
                classifier_module = ClassifierTrainLoop.load_from_checkpoint(os.path.join(work_dir, f'pretrained/affinity{postfix}.ckpt'), config=obj_cfg)
            elif objective == 'qed':
                obj_cfg = Config(os.path.join(work_dir, f'pretrained/qed{postfix}.yaml'))
                classifier_module = ClassifierTrainLoop.load_from_checkpoint(os.path.join(work_dir, f'pretrained/qed{postfix}.ckpt'), config=obj_cfg)
            elif objective == 'sa':
                obj_cfg = Config(os.path.join(work_dir, f'pretrained/sa{postfix}.yaml'))
                classifier_module = ClassifierTrainLoop.load_from_checkpoint(os.path.join(work_dir, f'pretrained/sa{postfix}.ckpt'), config=obj_cfg)
            elif objective == 'lipinski':
                obj_cfg = Config(os.path.join(work_dir, f'pretrained/lipinski{postfix}.yaml'))
                classifier_module = ClassifierTrainLoop.load_from_checkpoint(os.path.join(work_dir, f'pretrained/lipinski{postfix}.ckpt'), config=cfg)
            else:
                raise NotImplementedError(f"Objective {objective} not supported")
            print(f"Loaded classifier for {objective}")
            print(obj_cfg.todict())
            classifier_module.eval()
            classifier_list.append(classifier_module.dynamics)
        model.configure_classifiers(classifier_list, cfg.evaluation.objective.split('_'), cfg.evaluation.guide_mode, cfg.evaluation.pos_grad_weight, cfg.evaluation.type_grad_weight)
    trainer.test(model, dataloaders=test_loader, ckpt_path=best_ckpt)

