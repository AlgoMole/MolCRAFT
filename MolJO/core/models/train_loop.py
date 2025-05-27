import copy
import numpy as np
import torch

from time import time
from torch.profiler import profile, record_function, ProfilerActivity

import pytorch_lightning as pl

from torch_scatter import scatter_mean, scatter_sum

from core.config.config import Config
from core.models.bfn4sbdd import BFN4SBDDScoreModel, ClassifierScoreModel

import core.evaluation.utils.atom_num as atom_num
import core.utils.transforms as trans
import core.utils.reconstruct as reconstruct

from core.utils.train import get_optimizer, get_scheduler
import os
from tqdm import trange, tqdm


# ligand_nums = []

def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode="protein"):
    if mode == "none":
        offset = 0.0
        pass
    elif mode == "protein":
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset


class BFNTrainLoop(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config
        self.dynamics = BFN4SBDDScoreModel(**self.cfg.dynamics.todict())
        # [ time, h_t, pos_t, edge_index]
        self.train_losses = []
        self.save_hyperparameters(self.cfg.todict())
        self.time_records = np.zeros(6)
        self.log_time = False
        self.classifiers = None
        self.objectives = None
        self.guide_mode = None
        self.pos_grad_weight = None
        self.type_grad_weight = None

    def configure_classifiers(self, classifiers, objectives, guide_mode, pos_grad_weight, type_grad_weight):
        self.classifiers = []
        for prop_name, classifier in zip(objectives, classifiers):
            classifier.prop_name = prop_name
            self.classifiers.append(classifier)
        self.classifiers = classifiers
        self.objectives = objectives
        self.guide_mode = guide_mode
        self.pos_grad_weight = pos_grad_weight
        self.type_grad_weight = type_grad_weight

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        t1 = time()
        protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand = (
            batch.protein_pos,
            batch.protein_atom_feature.float(),
            batch.protein_element_batch,
            batch.ligand_pos,
            batch.ligand_atom_feature_full,
            batch.ligand_element_batch,
        )  # get the data from the batch
        # batch is a data object
        # protein_pos: [N_pro,3]
        # protein_v: [N_pro,27]
        # batch_protein: [N_pro]
        # ligand_pos: [N_lig,3]
        # ligand_v: [N_lig,13]
        # protein_element_batch: [N_protein]

        t2 = time()

        with torch.no_grad():
            # add noise to protein_pos
            protein_noise = torch.randn_like(protein_pos) * self.cfg.train.pos_noise_std
            gt_protein_pos = batch.protein_pos + protein_noise
            # random rotation as data aug
            if self.cfg.train.random_rot:
                M = np.random.randn(3, 3)
                Q, __ = np.linalg.qr(M)
                Q = torch.from_numpy(Q.astype(np.float32)).to(ligand_pos.device)
                gt_protein_pos = gt_protein_pos @ Q
                ligand_pos = ligand_pos @ Q

        num_graphs = batch_protein.max().item() + 1
        # !!!!!
        gt_protein_pos, ligand_pos, _ = center_pos(
            gt_protein_pos,
            ligand_pos,
            batch_protein,
            batch_ligand,
            mode=self.cfg.dynamics.center_pos_mode,
        )  # TODO: ugly

        # gt_protein_pos = gt_protein_pos / self.cfg.data.normalizer

        t3 = time()
        t = torch.rand(
            [num_graphs, 1], dtype=ligand_pos.dtype, device=ligand_pos.device
        ).index_select(
            0, batch_ligand
        )  # different t for different molecules.

        if not self.cfg.dynamics.use_discrete_t and not self.cfg.dynamics.destination_prediction:
            # t = torch.randint(0, 999, [num_graphs, 1], dtype=ligand_pos.dtype, device=ligand_pos.device).index_select(0, batch_ligand) #different t for different molecules.
            # t = t / 1000.0
            # else:
            t = torch.clamp(t, min=self.dynamics.t_min)  # clamp t to [t_min,1]

        t4 = time()
        c_loss, d_loss, discretised_loss = self.dynamics.loss_one_step(
            t,
            protein_pos=gt_protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            ligand_pos=ligand_pos,
            ligand_v=ligand_v,
            batch_ligand=batch_ligand,
        )

        # here the discretised_loss is close for current version.

        loss = torch.mean(c_loss + self.cfg.train.v_loss_weight * d_loss + discretised_loss)

        t5 = time()
        self.log_dict(
            {
                'lr': self.get_last_lr(),
                'loss': loss.item(), 
                'loss_pos': c_loss.mean().item(), 
                'loss_type': d_loss.mean().item(),
                'loss_c_ratio': c_loss.mean().item() / loss.item(),
            },
            on_step=True,
            prog_bar=True,
            batch_size=self.cfg.train.batch_size,
        )

        # check if loss is finite, skip update if not
        if not torch.isfinite(loss):
            return None
        self.train_losses.append(loss.clone().detach().cpu())

        t0 = time()

        if self.log_time:
            self.time_records = np.vstack((self.time_records, [t0, t1, t2, t3, t4, t5]))
            print(f'step total time: {self.time_records[-1, 0] - self.time_records[-1, 1]}, batch size: {num_graphs}')
            print(f'\tpl call & data access: {self.time_records[-1, 1] - self.time_records[-2, 0]}')
            print(f'\tunwrap data: {self.time_records[-1, 2] - self.time_records[-1, 1]}')
            print(f'\tadd noise & center pos: {self.time_records[-1, 3] - self.time_records[-1, 2]}')
            print(f'\tsample t: {self.time_records[-1, 4] - self.time_records[-1, 3]}')
            print(f'\tget loss: {self.time_records[-1, 5] - self.time_records[-1, 4]}')
            print(f'\tlogging: {self.time_records[-1, 0] - self.time_records[-1, 5]}')
        return loss

    def validation_step(self, batch, batch_idx):
        return self.shared_sampling_step(batch, batch_idx, sample_num_atoms='ref')

    def test_step_original(self, batch, batch_idx):
        out_data_list = []
        # time_list = []
        for _ in range(self.cfg.evaluation.num_samples):
            # start = time()
            out_data_list.extend(self.shared_sampling_step(batch, batch_idx, sample_num_atoms=self.cfg.evaluation.sample_num_atoms))
            # end = time()
            # time_list.append(end - start)
            # print(f"sample time: {end - start}")
        # print(f"average sample time: {sum(time_list) / len(time_list)}")
        # print(f"std sample time: {np.std(time_list)}")
        return out_data_list

    def test_step_trajectory(self, batch, batch_idx):
        out_data_list = []
        all_results = {f'{name}_{key}': [] for name in ['sample', 'theta', 'y'] for key in ['pos', 'v', 'mol']}
        all_results['sample_exp'] = []

        # for _ in range(self.cfg.evaluation.num_samples):
        # only one sample for trajectory
        results = self.trajectory_step(batch, batch_idx, sample_num_atoms=self.cfg.evaluation.sample_num_atoms)
        for key, value in results.items():
            all_results[key].extend(value)

        # out data list is a list of dict {f'{name}_pos_traj': [num_steps, num_atoms, 3], 
        #        f'{name}_v_traj': [num_steps, num_atoms, 13],
        #        'sample_exp': [num_steps, num_classifiers, num_graphs]}
        all_pred_pos_traj = all_results['sample_pos']
        all_pred_v_traj = all_results['sample_v']
        all_pred_exps = all_results['sample_exp']
        assert len(all_pred_pos_traj) == len(all_pred_v_traj)
        for i in range(len(all_pred_pos_traj)):
            assert all_pred_pos_traj[i].shape[0] == all_pred_v_traj[i].shape[0], f"sample steps mismatch: {all_pred_pos_traj[i].shape[0]} != {all_pred_v_traj[i].shape[0]}"
            if self.classifiers is not None:
                assert len(all_pred_exps[i]) == all_pred_v_traj[i].shape[0], f"exp steps mismatch: {len(all_pred_exps[i])} != {all_pred_v_traj[i].shape[0]}"
                assert len(all_pred_exps[i][0]) == len(self.classifiers), f"exp classifiers mismatch: {len(all_pred_exps[i][0])} != {len(self.classifiers)}"
                exp_dict = {}
                for j in range(len(self.classifiers)):
                    exp_j = [all_pred_exps[i][k][j] for k in range(len(all_pred_exps[i]))]
                    exp_j_list = [exp_j[k][0] for k in range(len(exp_j))]
                    exp_j_type_grad = [exp_j[k][1] for k in range(len(exp_j))]
                    exp_j_pos_grad = [exp_j[k][2] for k in range(len(exp_j))]
                    exp_dict[self.objectives[j]] = torch.tensor(exp_j_list, dtype=torch.float32)
                    exp_dict[f'{self.objectives[j]}_type_grad'] = exp_j_type_grad
                    exp_dict[f'{self.objectives[j]}_pos_grad'] = exp_j_pos_grad
            else:
                exp_dict = {}

            out_data_list.append({
                'pos_traj': torch.tensor(all_pred_pos_traj[i], dtype=torch.float32),
                'v_traj': torch.tensor(all_pred_v_traj[i], dtype=torch.float32),
                **exp_dict,
                'theta_pos_traj': torch.tensor(all_results['theta_pos'][i], dtype=torch.float32),
                'theta_v_traj': torch.tensor(all_results['theta_v'][i], dtype=torch.float32),
                'y_pos_traj': torch.tensor(all_results['y_pos'][i], dtype=torch.float32),
                'y_v_traj': torch.tensor(all_results['y_v'][i], dtype=torch.float32),
                # 'mol_traj': all_results['sample_mol'][i],
            })
        return out_data_list

    def test_step(self, batch, batch_idx):
        if self.cfg.evaluation.save_traj:
            return self.test_step_trajectory(batch, batch_idx)
        else:
            return self.test_step_original(batch, batch_idx)

    def trajectory_step(self, batch, batch_idx, sample_num_atoms):
        protein_pos, protein_v, batch_protein, batch_ligand = (
            batch.protein_pos,
            batch.protein_atom_feature.float(),
            batch.protein_element_batch,
            batch.ligand_element_batch,
        )
        num_graphs = batch_protein.max().item() + 1

        protein_pos, _, offset = center_pos(
            protein_pos,
            protein_pos,
            batch_protein,
            batch_protein,
            mode=self.cfg.dynamics.center_pos_mode,
        )

        if sample_num_atoms == 'prior':
            ligand_num_atoms = []
            for data_id in range(len(batch)):
                data = batch[data_id]
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy() * self.cfg.data.normalizer_dict.pos)
                ligand_num_atoms.append(atom_num.sample_atom_num(pocket_size, self.cfg.data.path).astype(int))
            batch_ligand = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(ligand_num_atoms)).to(protein_pos.device)
            ligand_num_atoms = torch.tensor(ligand_num_atoms, dtype=torch.long, device=protein_pos.device)
        elif sample_num_atoms == 'ref':
            batch_ligand = batch.ligand_element_batch
            ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).to(protein_pos.device)
        else:
            raise ValueError(f"sample_num_atoms mode: {sample_num_atoms} not supported")
        ligand_cum_atoms = torch.cat([
            torch.tensor([0], dtype=torch.long, device=protein_pos.device), 
            ligand_num_atoms.cumsum(dim=0)
        ])

        theta_chain, sample_chain, y_chain = self.dynamics.sample(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
            sample_steps=self.cfg.evaluation.sample_steps,
            n_nodes=num_graphs,
            classifiers=self.classifiers,
            guide_mode=self.guide_mode,
            pos_grad_weight=self.pos_grad_weight,
            type_grad_weight=self.type_grad_weight,
            EPS=self.cfg.evaluation.interpolate_coef,
            W_CFG=self.cfg.evaluation.cfg_coef,
        )

        results = {}
        chains = {'theta': theta_chain, 'sample': sample_chain, 'y': y_chain}
        for name, chain in chains.items():
            all_step_pos = [[] for _ in range(num_graphs)]
            all_step_v = [[] for _ in range(num_graphs)]
            all_classifier_exp = [[] for _ in range(num_graphs)] if self.classifiers is not None else None
            molist = []
            for i_step, sample in tqdm(enumerate(chain), total=len(chain), desc=name):  # step_i
                # unbatch pos
                p = sample[0] + offset[batch_ligand]
                p = p * torch.tensor(
                    self.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=protein_pos.device
                )
                p_array = p.cpu().numpy().astype(np.float64)
                # unbatch type
                one_hot = sample[1]
                pred_pmf = one_hot.cpu().numpy()
                pred_v = one_hot.argmax(dim=-1)
                pred_v = pred_v.cpu().numpy()
                pred_atom_type = trans.get_atomic_number_from_index(
                    pred_v, mode=self.cfg.data.transform.ligand_atom_mode
                ) # List[int]

                pred_v = pred_pmf

                for k in range(num_graphs):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
                    all_step_v[k].append(pred_v[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
                    # try:
                    #     mol = reconstruct.reconstruct_from_generated(
                    #         p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]],
                    #         pred_atom_type[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]],
                    #         [False] * ligand_num_atoms[k].item(),
                    #         sanitize=False,
                    #     )
                    # except Exception as e:
                    #     print(f"Error in reconstructing molecule: {e}")
                    #     mol = None
                    # molist.append(mol)
                    if self.classifiers is not None and name == 'sample':
                        # sample[-1]: exp_list of List[num_classifiers, num_graphs]
                        assert len(sample[-1]) == len(self.classifiers), f"exp_list: {len(sample[-1])} != classifiers: {len(self.classifiers)}"
                        tup = []
                        for exp_item in sample[-1]: 
                            # exp_item in exp_list
                            # (exp_list: List[num_classifiers, num_graphs],
                            #  type_grad_list: List[num_classifiers, num_graphs],
                            #  pos_grad_list: List[num_classifiers, num_graphs])
                            tup_item = []
                            for exp in exp_item: # exp, type_grad, pos_grad in exp_list
                                if len(exp) != num_graphs:
                                    grad_k = exp[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]]
                                    tup_item.append(grad_k)
                                else:
                                    tup_item.append(exp[k]) # exp[k] is the exp for the k-th graph
                            tup.append(tup_item)
                        # tup = [(exp[k]) for exp in sample[-1]]
                        all_classifier_exp[k].append(tup)

            all_step_pos = [np.stack(step_pos) for step_pos in
                            all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
            all_step_v = [np.stack(step_v) for step_v in
                            all_step_v]  # num_samples * [num_steps, num_atoms_i, 13]
            
            results[f'{name}_pos'] = all_step_pos
            results[f'{name}_v'] = all_step_v
            results[f'{name}_mol'] = molist
            if name == 'sample' and self.classifiers is not None: results[f'{name}_exp'] = all_classifier_exp # List[num_steps, num_classifiers, num_graphs]

        return results


    def shared_sampling_step(self, batch, batch_idx, sample_num_atoms):
        # here we need to sample the molecules in the validation step
        protein_pos, protein_v, batch_protein, batch_ligand = (
            batch.protein_pos,
            batch.protein_atom_feature.float(),
            batch.protein_element_batch,
            batch.ligand_element_batch,
        )
        ligand_mask, ligand_pos, ligand_v = None, None, None

        num_graphs = batch_protein.max().item() + 1  # B
        assert num_graphs == len(batch), f"num_graphs: {num_graphs} != len(batch): {len(batch)}"

        # move protein center to origin & ligand correspondingly
        protein_pos, _, offset = center_pos(
            protein_pos,
            protein_pos,
            batch_protein,
            batch_protein,
            mode=self.cfg.dynamics.center_pos_mode,
        )

        # determine the number of atoms in the ligand
        if sample_num_atoms == 'prior':
            ligand_num_atoms = []
            for data_id in range(len(batch)):
                data = batch[data_id]
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy() * self.cfg.data.normalizer_dict.pos)
                ligand_num_atoms.append(atom_num.sample_atom_num(pocket_size, self.cfg.data.path).astype(int))
            batch_ligand = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(ligand_num_atoms)).to(protein_pos.device)
            ligand_num_atoms = torch.tensor(ligand_num_atoms, dtype=torch.long, device=protein_pos.device)
        elif sample_num_atoms == 'prior_ref':
            assert hasattr(batch, 'ligand_pos') and hasattr(batch, 'ligand_atom_feature_full')
            ligand_num_atoms = []
            while len(ligand_num_atoms) < len(batch):
                data_id = len(ligand_num_atoms)
                data = batch[data_id]
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy() * self.cfg.data.normalizer_dict.pos)
                atom_nums = atom_num.sample_atom_num(pocket_size, self.cfg.data.path).astype(int)
                if atom_nums < len(data.ligand_pos):
                    continue
                ligand_num_atoms.append(atom_nums)
            batch_ligand = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(ligand_num_atoms)).to(protein_pos.device)
            ligand_num_atoms = torch.tensor(ligand_num_atoms, dtype=torch.long, device=protein_pos.device)
        elif sample_num_atoms == 'ref' or sample_num_atoms == 'inpainting_ref':
            batch_ligand = batch.ligand_element_batch
            ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).to(protein_pos.device)
            if sample_num_atoms == 'inpainting_ref':
                assert hasattr(batch, 'ligand_pos') and hasattr(batch, 'ligand_atom_feature_full') and hasattr(batch, 'ligand_mask')
                ligand_mask = batch.ligand_mask
                ligand_pos = batch.ligand_pos
                ligand_v = batch.ligand_atom_feature_full
        elif sample_num_atoms == 'inpainting_prior':
            assert hasattr(batch, 'ligand_pos') and hasattr(batch, 'ligand_atom_feature_full') and hasattr(batch, 'ligand_mask')
            ligand_num_atoms = []
            ligand_mask, ligand_pos, ligand_v = [], [], []
            while len(ligand_num_atoms) < len(batch):
                data_id = len(ligand_num_atoms)
                data = batch[data_id]
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy() * self.cfg.data.normalizer_dict.pos)
                atom_nums = atom_num.sample_atom_num(pocket_size, self.cfg.data.path).astype(int)
                if atom_nums < len(data.ligand_pos):
                    continue
                ligand_num_atoms.append(atom_nums)

                # update ligand_pos and ligand_v, ligand_mask
                data.ligand_pos = torch.cat([data.ligand_pos, torch.zeros(atom_nums - len(data.ligand_pos), 3, device=data.ligand_pos.device)])
                data.ligand_atom_feature_full = torch.cat([data.ligand_atom_feature_full, torch.zeros(atom_nums - len(data.ligand_atom_feature_full), device=data.ligand_atom_feature_full.device)])
                data.ligand_mask = torch.cat([data.ligand_mask, torch.zeros(atom_nums - len(data.ligand_mask), dtype=torch.bool, device=data.ligand_mask.device)])
                batch[data_id] = data
                # print(f"sampled {atom_nums} atoms for ligand {data_id} in inpainting_prior mode")
                assert len(data.ligand_pos) == atom_nums and len(data.ligand_atom_feature_full) == atom_nums and len(data.ligand_mask) == atom_nums
                ligand_mask.append(data.ligand_mask)
                ligand_pos.append(data.ligand_pos)
                ligand_v.append(data.ligand_atom_feature_full)

            # print(f"ligand_num_atoms: {sum(ligand_num_atoms)}")
            batch_ligand = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(ligand_num_atoms)).to(protein_pos.device)
            ligand_num_atoms = torch.tensor(ligand_num_atoms, dtype=torch.long, device=protein_pos.device)
            ligand_mask = torch.cat(ligand_mask, dim=0).bool()
            ligand_pos = torch.cat(ligand_pos, dim=0)
            ligand_v = torch.cat(ligand_v, dim=0).long()
        else:
            raise ValueError(f"sample_num_atoms mode: {sample_num_atoms} not supported")
        ligand_cum_atoms = torch.cat([
            torch.tensor([0], dtype=torch.long, device=protein_pos.device), 
            ligand_num_atoms.cumsum(dim=0)
        ])
        n_nodes = batch_ligand.size(0)  # N_lig

        # ligand_nums.extend(ligand_num_atoms.tolist())
        # print(ligand_num_atoms, sum(ligand_nums) / len(ligand_nums))
        # return []

        if ligand_pos is not None:
            protein_pos, ligand_pos, offset = center_pos(
                batch.protein_pos,
                ligand_pos,
                batch_protein,
                batch_ligand,
                mode=self.cfg.dynamics.center_pos_mode,
            )

        # TODO reuse for visualization and test
        # forward pass to get the ligand sample
        theta_chain, sample_chain, y_chain = self.dynamics.sample(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
            # n_nodes=n_nodes,
            sample_steps=self.cfg.evaluation.sample_steps,
            n_nodes=num_graphs,
            classifiers=self.classifiers,
            guide_mode=self.guide_mode,
            pos_grad_weight=self.pos_grad_weight,
            type_grad_weight=self.type_grad_weight,
            ligand_pos=ligand_pos,  # for inpainting
            ligand_v=ligand_v,  # for inpainting
            ligand_mask=ligand_mask, 
            EPS=self.cfg.evaluation.interpolate_coef,
            W_CFG=self.cfg.evaluation.cfg_coef,
        )

        # restore ligand to original position
        final = sample_chain[-1]  # mu_pos_final, k_final, k_hat_final
        pred_pos, one_hot = final[0] + offset[batch_ligand], final[1]

        # along with normalizer
        pred_pos = pred_pos * torch.tensor(
            self.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=protein_pos.device
        )
        out_batch = copy.deepcopy(batch)
        out_batch.protein_pos = out_batch.protein_pos * torch.tensor(
            self.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=protein_pos.device
        )

        pred_v = one_hot.argmax(dim=-1)
        # TODO: ugly, should be done in metrics.py (but needs a way to make it compatible with pyg batch)
        pred_atom_type = trans.get_atomic_number_from_index(
            pred_v, mode=self.cfg.data.transform.ligand_atom_mode
        ) # List[int]

        # for visualization
        atom_type = [trans.MAP_ATOM_TYPE_ONLY_TO_INDEX[i] for i in pred_atom_type]  # List[int]
        atom_type = torch.tensor(atom_type, dtype=torch.long, device=protein_pos.device)  # [N_lig]

        # for reconstruction
        pred_aromatic = trans.is_aromatic_from_index(
            pred_v, mode=self.cfg.data.transform.ligand_atom_mode
        ) # List[bool]
        molist = []
        for i in range(num_graphs):
            # TODO: fix seg fault during reconstruction (caused by GetConformer().GetPositions())
            try:
                mol = reconstruct.reconstruct_from_generated(
                    pred_pos[batch_ligand == i].cpu().numpy().astype(np.float64),
                    pred_atom_type[ligand_cum_atoms[i]:ligand_cum_atoms[i + 1]],
                    pred_aromatic[ligand_cum_atoms[i]:ligand_cum_atoms[i + 1]],
                )
            except reconstruct.MolReconsError:
                mol = None
            molist.append(mol)
        
        # add necessary dict to new pyg batch
        out_batch.x, out_batch.pos = atom_type, pred_pos
        out_batch.atom_type = torch.tensor(pred_atom_type, dtype=torch.long, device=protein_pos.device)
        out_batch.is_aromatic = torch.tensor(pred_aromatic, dtype=torch.long, device=protein_pos.device)
        out_batch.mol = molist

        _slice_dict = {
            "x": ligand_cum_atoms,
            "pos": ligand_cum_atoms,
            "atom_type": ligand_cum_atoms,
            "is_aromatic": ligand_cum_atoms,
            "mol": out_batch._slice_dict["ligand_filename"],
        }
        _inc_dict = {
            "x": out_batch._inc_dict["ligand_element"], # [0] * B, TODO: figure out what this is
            "pos": out_batch._inc_dict["ligand_pos"],
            "atom_type": out_batch._inc_dict["ligand_element"],
            "is_aromatic": out_batch._inc_dict["ligand_element"],
            "mol": out_batch._inc_dict["ligand_filename"],
        }
        out_batch._inc_dict.update(_inc_dict)
        out_batch._slice_dict.update(_slice_dict)
        out_data_list = out_batch.to_data_list()
        return out_data_list

    def on_train_epoch_end(self) -> None:
        if len(self.train_losses) == 0:
            epoch_loss = 0
        else:
            epoch_loss = torch.stack([x for x in self.train_losses]).mean()
        print(f"epoch_loss: {epoch_loss}")
        self.log(
            "epoch_loss",
            epoch_loss,
            batch_size=self.cfg.train.batch_size,
        )
        self.train_losses = []

    def configure_optimizers(self):
        self.optim = get_optimizer(self.cfg.train.optimizer, self)
        self.scheduler, self.get_last_lr = get_scheduler(self.cfg.train, self.optim)

        return {
            'optimizer': self.optim, 
            'lr_scheduler': self.scheduler,
        }

class ClassifierTrainLoop(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config
        if hasattr(self.cfg, 'classifier'):
            self.dynamics = ClassifierScoreModel(**self.cfg.classifier.todict())
            self.dynamics_cfg = self.cfg.classifier
        else:
            print('No classifier config found, using default dynamics')
            self.dynamics = ClassifierScoreModel(**self.cfg.dynamics.todict())
            self.dynamics_cfg = self.cfg.dynamics

        # [ time, h_t, pos_t, edge_index]
        self.train_losses = []
        self.loss_traj = {i: [] for i in range(self.cfg.evaluation.sample_steps)}
        self.save_hyperparameters(self.cfg.todict())

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand = (
            batch.protein_pos,
            batch.protein_atom_feature.float(),
            batch.protein_element_batch,
            batch.ligand_pos,
            batch.ligand_atom_feature_full,
            batch.ligand_element_batch,
        )  # get the data from the batch
        # TODO: ugly fix for non-zero prop, update v4 dataset to have prop as float
        try:
            prop = getattr(batch, self.dynamics_cfg.prop_name).float()  # [N_lig, 1]
        except Exception as e:
            print(f"Error with {self.dynamics_cfg.prop_name}: {e}")
            print(getattr(batch, self.dynamics_cfg.prop_name), type(getattr(batch, self.dynamics_cfg.prop_name)))
            prop = torch.tensor(getattr(batch, self.dynamics_cfg.prop_name), dtype=torch.float32, device=ligand_pos.device)
            print(prop, type(prop))
        # batch is a data object
        # protein_pos: [N_pro,3]
        # protein_v: [N_pro,27]
        # batch_protein: [N_pro]
        # ligand_pos: [N_lig,3]
        # ligand_v: [N_lig,13]
        # protein_element_batch: [N_protein]

        with torch.no_grad():
            # add noise to protein_pos
            protein_noise = torch.randn_like(protein_pos) * self.cfg.train.pos_noise_std
            gt_protein_pos = batch.protein_pos + protein_noise
            # random rotation as data aug
            if self.cfg.train.random_rot:
                M = np.random.randn(3, 3)
                Q, __ = np.linalg.qr(M)
                Q = torch.from_numpy(Q.astype(np.float32)).to(ligand_pos.device)
                gt_protein_pos = gt_protein_pos @ Q
                ligand_pos = ligand_pos @ Q

        num_graphs = batch_protein.max().item() + 1
        gt_protein_pos, ligand_pos, _ = center_pos(
            gt_protein_pos,
            ligand_pos,
            batch_protein,
            batch_ligand,
            mode=self.dynamics_cfg.center_pos_mode,
        )

        t = torch.rand(
            [num_graphs, 1], dtype=ligand_pos.dtype, device=ligand_pos.device
        ).index_select(
            0, batch_ligand
        )  # different t for different molecules.

        if not self.dynamics_cfg.use_discrete_t and not self.dynamics_cfg.destination_prediction:
            # t = torch.randint(0, 999, [num_graphs, 1], dtype=ligand_pos.dtype, device=ligand_pos.device).index_select(0, batch_ligand) #different t for different molecules.
            # t = t / 1000.0
            # else:
            t = torch.clamp(t, min=self.dynamics.t_min)  # clamp t to [t_min,1]

        exp_loss = self.dynamics.loss_one_step(
            t,
            protein_pos=gt_protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            ligand_pos=ligand_pos,
            ligand_v=ligand_v,
            batch_ligand=batch_ligand,
            prop=prop,
        )

        # here the discretised_loss is close for current version.

        loss = torch.mean(exp_loss)

        self.log_dict(
            {
                'lr': self.get_last_lr(),
                'loss': loss.item(), 
            },
            on_step=True,
            prog_bar=True,
            batch_size=self.cfg.train.batch_size,
        )

        # check if loss is finite, skip update if not
        if not torch.isfinite(loss):
            return None
        self.train_losses.append(loss.clone().detach().cpu())

        return loss

    def on_train_epoch_end(self) -> None:
        if len(self.train_losses) == 0:
            epoch_loss = 0
        else:
            epoch_loss = torch.stack([x for x in self.train_losses]).mean()
        print(f"epoch_loss: {epoch_loss}")
        self.log(
            "epoch_loss",
            epoch_loss,
            batch_size=self.cfg.train.batch_size,
        )
        self.train_losses = []

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)
    
    def on_validation_epoch_end(self) -> None:
        return self.on_test_epoch_end()

    def test_step(self, batch, batch_idx):
        protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand = (
            batch.protein_pos,
            batch.protein_atom_feature.float(),
            batch.protein_element_batch,
            batch.ligand_pos,
            batch.ligand_atom_feature_full,
            batch.ligand_element_batch,
        )  # get the data from the batch
        # TODO: ugly fix for non-zero prop, update v4 dataset to have prop as float
        try:
            prop = getattr(batch, self.dynamics_cfg.prop_name).float()  # [N_lig, 1]
        except Exception as e:
            print(f"Error with {self.dynamics_cfg.prop_name}: {e}")
            print(getattr(batch, self.dynamics_cfg.prop_name), type(getattr(batch, self.dynamics_cfg.prop_name)))
            prop = torch.tensor(getattr(batch, self.dynamics_cfg.prop_name), dtype=torch.float32, device=ligand_pos.device)
            print(prop, type(prop))         

        num_graphs = batch_protein.max().item() + 1

        # sample a random timestep for reconstruction loss computation
        for i in trange(0, self.cfg.evaluation.sample_steps):
            t = torch.tensor(
                [i / float(self.cfg.evaluation.sample_steps)], 
                dtype=ligand_pos.dtype, device=ligand_pos.device
            ).repeat(num_graphs, 1).index_select(
                0, batch_ligand
            )  # [num_graphs, 1]

            if not self.dynamics_cfg.use_discrete_t and not self.dynamics_cfg.destination_prediction:
                t = torch.clamp(t, min=self.dynamics.t_min)  # clamp t to [t_min,1]

            # compute bfn loss  # TODO: convert to reconstruction loss
            exp_loss = self.dynamics.reconstruction_loss_one_step(
                t,
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                ligand_pos=ligand_pos,
                ligand_v=ligand_v,
                batch_ligand=batch_ligand,
                prop=prop,
            )

            self.loss_traj[i].extend(exp_loss.clone().detach().cpu().tolist())

    def on_test_epoch_end(self) -> None:
        if len(self.loss_traj) == 0:
            loss_traj = []
        else:
            loss_traj = [sum(self.loss_traj[i]) / len(self.loss_traj[i]) for i in range(self.cfg.evaluation.sample_steps)]
            if not os.path.exists(self.cfg.accounting.checkpoint_dir):
                os.makedirs(self.cfg.accounting.checkpoint_dir)
            torch.save(self.loss_traj, os.path.join(self.cfg.accounting.checkpoint_dir, 'loss_traj_dict.pt'))
            torch.save(loss_traj, os.path.join(self.cfg.accounting.checkpoint_dir, 'loss_traj.pt'))
        self.loss_traj = {i: [] for i in range(self.cfg.evaluation.sample_steps)}

    def configure_optimizers(self):
        self.optim = get_optimizer(self.cfg.train.optimizer, self)
        self.scheduler, self.get_last_lr = get_scheduler(self.cfg.train, self.optim)

        return {
            'optimizer': self.optim, 
            'lr_scheduler': self.scheduler,
        }
