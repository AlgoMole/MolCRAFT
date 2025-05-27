import argparse
import os
import pickle

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm
from torch_geometric.loader import DataLoader

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D


def data_likelihood_estimation_2d(model, batch, time_steps_cont, time_steps_disc, device='cuda:0'):
    """
    Compute the 2D grid of KL losses for continuous and discrete variables.

    Args:
        model: The model used for likelihood estimation.
        data: The input data.
        time_steps_cont: Time steps for continuous variables.
        time_steps_disc: Time steps for discrete variables.
        batch_size: Batch size for processing.
        device: Device to run the computation on.

    Returns:
        all_kl_pos_grid: 2D tensor of KL losses for continuous variables.
        all_kl_v_grid: 2D tensor of KL losses for discrete variables.
        sum_kl_pos: Total KL loss for continuous variables.
        sum_kl_v: Total KL loss for discrete variables.
    """
    num_timesteps_cont = len(time_steps_cont)
    num_timesteps_disc = len(time_steps_disc)

    # Initialize storage for KL losses
    all_kl_pos_grid = torch.zeros((num_timesteps_cont, num_timesteps_disc), device=device)
    all_kl_v_grid = torch.zeros((num_timesteps_cont, num_timesteps_disc), device=device)

    # Iterate over the 2D grid of time steps
    progress_bar = tqdm(total=num_timesteps_cont * num_timesteps_disc)
    for i, t_cont in enumerate(time_steps_cont):
        for j, t_disc in enumerate(time_steps_disc):
            # Perform likelihood estimation for the current time step pair
            kl_pos, kl_v = model.likelihood_estimation_decouple(
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,

                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                batch_ligand=batch.ligand_element_batch,

                time_step_cont=torch.tensor([t_cont] * len(batch), device=device),
                time_step_disc=torch.tensor([t_disc] * len(batch), device=device)
            )

            # Store the KL losses in the grid
            all_kl_pos_grid[i, j] = kl_pos.mean()
            all_kl_v_grid[i, j] = kl_v.mean()
            progress_bar.update(1)

    return all_kl_pos_grid.cpu(), all_kl_v_grid.cpu()


def data_likelihood_estimation(model, data, time_steps, batch_size=1, device='cuda:0'):
    num_timesteps = len(time_steps)
    num_batch = int(np.ceil(num_timesteps / batch_size))
    all_kl_pos, all_kl_v = [], []

    cur_i = 0
    # t in [T-1, ..., 0]
    for i in range(num_batch):
        n_data = batch_size if i < num_batch - 1 else num_timesteps - batch_size * (num_batch - 1)
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)
        time_step = time_steps[cur_i:cur_i + n_data]

        kl_pos, kl_v = model.likelihood_estimation(
            protein_pos=batch.protein_pos,
            protein_v=batch.protein_atom_feature.float(),
            batch_protein=batch.protein_element_batch,

            ligand_pos=batch.ligand_pos,
            ligand_v=batch.ligand_atom_feature_full,
            batch_ligand=batch.ligand_element_batch,

            time_step=time_step
        )
        all_kl_pos.append(kl_pos)
        all_kl_v.append(kl_v)
        cur_i += n_data

    # prior
    batch = Batch.from_data_list([data.clone() for _ in range(1)], follow_batch=FOLLOW_BATCH).to(device)
    time_step = torch.tensor([model.num_timesteps], device=device)
    kl_pos_prior, kl_v_prior = model.likelihood_estimation(
        protein_pos=batch.protein_pos,
        protein_v=batch.protein_atom_feature.float(),
        batch_protein=batch.protein_element_batch,

        ligand_pos=batch.ligand_pos,
        ligand_v=batch.ligand_atom_feature_full,
        batch_ligand=batch.ligand_element_batch,

        time_step=time_step
    )
    all_kl_pos, all_kl_v = torch.cat(all_kl_pos), torch.cat(all_kl_v)
    sum_kl_pos, sum_kl_v = model.num_timesteps * torch.mean(all_kl_pos), model.num_timesteps * torch.mean(all_kl_v)
    all_kl_pos, all_kl_v = torch.cat([all_kl_pos, kl_pos_prior]), torch.cat([all_kl_v, kl_v_prior])
    sum_kl_pos += kl_pos_prior[0]
    sum_kl_v += kl_v_prior[0]
    return all_kl_pos.cpu(), all_kl_v.cpu(), sum_kl_pos.item(), sum_kl_v.item()


def get_dataset_result(dset, affinity_info):
    valid_id = []
    for data_id in tqdm(range(len(dset)), desc='Filtering data'):
        data = dset[data_id]
        ligand_fn_key = data.ligand_filename[:-4]
        pk = affinity_info[ligand_fn_key]['pk']
        if pk > 0:
            valid_id.append(data_id)
    print(f'There are {len(valid_id)} examples with valid pK in total.')

    all_results = []
    for data_id in tqdm(valid_id, desc='Evaluating'):
        data = dset[data_id]
        # likelihoods
        time_steps = torch.tensor(list(range(0, 1000, 100)), device=args.device)
        all_kl_pos, all_kl_v, sum_kl_pos, sum_kl_v = data_likelihood_estimation(
            model, data, time_steps, batch_size=args.batch_size, device=args.device)
        kl = sum_kl_pos + sum_kl_v

        # embedding
        batch = Batch.from_data_list([data.clone() for _ in range(1)], follow_batch=FOLLOW_BATCH).to(args.device)
        preds = model.fetch_embedding(
            protein_pos=batch.protein_pos,
            protein_v=batch.protein_atom_feature.float(),
            batch_protein=batch.protein_element_batch,

            ligand_pos=batch.ligand_pos,
            ligand_v=batch.ligand_atom_feature_full,
            batch_ligand=batch.ligand_element_batch,
        )

        # gather results
        ligand_fn_key = data.ligand_filename[:-4]
        result = {
            'idx': data_id,
            **affinity_info[ligand_fn_key],
            'kl_pos': all_kl_pos,
            'kl_v': all_kl_v,
            'nll': kl,
            'pred_ligand_v': preds['pred_ligand_v'].cpu(),
            'final_h': preds['final_h'].cpu(),
            'final_ligand_h': preds['final_ligand_h'].cpu()
        }
        all_results.append(result)
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--config', type=str, default='configs/sampling_rect.yml')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--sample_steps', type=int, default=20)
    parser.add_argument('--result_path', type=str, default='./outputs_grid')

    args = parser.parse_args()

    logger = misc.get_logger('evaluate')

    # Load config
    config = misc.load_config(args.config)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    # Load checkpoint
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(ckpt['config'].data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]
    if ckpt['config'].data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    # Load dataset
    dataset, subsets = get_dataset(
        config=ckpt['config'].data,
        transform=transform
    )
    train_set, test_set = subsets['train'], subsets['test']
    logger.info(f'Successfully load the dataset (size: {len(test_set)})!')

    # Load model
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'], strict=False if 'train_config' in config.model else True)
    logger.info(f'Successfully load the model! {config.model.checkpoint}')

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=["ligand_nbh_list"]
    )

    all_kl_pos, all_kl_v = torch.zeros((args.sample_steps, args.sample_steps)), torch.zeros((args.sample_steps, args.sample_steps))
    for batch in test_loader:
        batch = batch.to(args.device)
        num_steps = ckpt['config'].model.num_diffusion_timesteps
        interval = ckpt['config'].model.num_diffusion_timesteps // args.sample_steps
        # list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        time_steps = torch.tensor(list(range(0, ckpt['config'].model.num_diffusion_timesteps, interval)), device=args.device) # shape (args.sample_steps,)
        print(time_steps.shape)
        kl_pos, kl_v = data_likelihood_estimation_2d(
            model, batch, time_steps, time_steps, device=args.device)
        assert all_kl_pos.shape == kl_pos.shape
        all_kl_pos += kl_pos
        all_kl_v += kl_v

    all_kl_pos /= len(test_loader)
    all_kl_v /= len(test_loader)

    # store results into the following format

    loss_partial_modal = []
    for i in range(args.sample_steps):
        loss_partial_modal.append([])
        for j in range(args.sample_steps):
            loss_partial_modal[i].append({
                'pos': all_kl_pos[i, j].item(),
                'type': all_kl_v[i, j].item(),
                'loss': all_kl_pos[i, j].item() + all_kl_v[i, j].item()
            })

    os.makedirs(args.result_path, exist_ok=True)
    torch.save(loss_partial_modal, os.path.join(args.result_path, f'loss_grid_rect{args.sample_steps}.pt'))
