import os
import torch
import time
from tqdm import tqdm, trange
from torch.utils.data import Subset, ConcatDataset
from core.datasets.pl_data import ProteinLigandData
from core.datasets.pl_pair_dataset import PocketLigandPairDataset, PocketLigandPairDatasetFeaturized, PocketLigandGeneratedPairDataset
from core.datasets.pdbbind import PDBBindDataset


def get_dataset(config, *args, **kwargs):
    name = config.name
    root = config.path
    ligand_atom_mode = config.transform.ligand_atom_mode
    version = getattr(config, 'version', 'final')
    kwargs['version'] = version
    if name == 'pl':
        dataset = PocketLigandPairDataset(root, *args, **kwargs)
    elif name == 'pl_dock':
        dataset = PocketLigandPairDataset(root, *args, **kwargs)
        pose_split = torch.load(config.split)
        train_set = Subset(dataset, indices=pose_split['train'])
        pb_dataset = PocketLigandPairDataset(config.path_dock, *args, **kwargs)
        pb_pose_split = torch.load(config.split_dock)
        pb_test_set = Subset(pb_dataset, indices=pb_pose_split['test'])

        return dataset, {"train": train_set, "test": pb_test_set}

    elif name == 'pl_tr':
        dataset = PocketLigandPairDatasetFeaturized(root, ligand_atom_mode=ligand_atom_mode,
                                                     *args, **kwargs)
        return dataset, {"train": dataset.train_data, "test": dataset.test_data}
    elif name == 'pl_dcmp':
        dataset = PocketLigandGeneratedPairDataset(root, *args, **kwargs)
        return dataset, {"train": dataset, "test": dataset}
    elif name == 'pdbbind':
        dataset = PDBBindDataset(root, *args, **kwargs)
    elif isinstance(name, int):
        # dataset = PocketLigandPairDataset('./data/crossdocked_pocket10', *args, **kwargs)
        dataset = PocketLigandPairDataset('./data/crossdocked_v1.1_rmsd1.0_pocket10', *args, **kwargs)
        pose_split = torch.load('./data/crossdocked_pocket10_pose_split_filtered.pt')
        test_set = Subset(dataset, indices=pose_split['test'])
        # train_set_part1 = Subset(dataset, indices=pose_split['train'])

        # if os.path.exists('./data/subset_pocket10_filtered_100k_v2.pt'):
        #     print('Loading cached dataset...')
        #     start = time.time()
        #     train_set = torch.load('./data/subset_pocket10_filtered_100k_v2.pt')
        #     print('Time:', time.time() - start)
        #     # transform = kwargs.get('transform')
        #     # train_set = [transform(ProteinLigandData(**(data.to_dict()))) for data in tqdm(train_set)]
        #     # torch.save(train_set, './data/subset_pocket10_filtered_100k_v2.pt')
        #     # print(len(train_set), train_set[0])
        # else:
        #     train_set = []
        #     transform = kwargs.get('transform', lambda x: x)
        #     for i in trange(2, 12):
        #         version = i * 100
        #         subset = torch.load(f'./data/subset_pocket10_filtered_{version}_10k.pt')
        #         train_set.extend(subset)
        #     train_set = [transform(ProteinLigandData(**data)) for data in tqdm(train_set)]
        #     torch.save(train_set, './data/subset_pocket10_filtered_100k.pt')

        train_set = []
        start = time.time()
        del kwargs['version']
        for i in trange(1, 16):
            version = f'{i*100}'
            try:
                dataset = PocketLigandPairDataset('/mnt/data/HelixDock-lmdb/_pocket10', version=version, *args, **kwargs)
                # pose_split = torch.load(f'./data/HelixDock_pose_split_qed0.3_aff-5_{version}_full_filtered.pt')
                pose_split = torch.load(f'/mnt/data/HelixDock_pose_split_filtered_qed0.3_aff-5_{version}_15k.pt')
                train_set.append(Subset(dataset, indices=pose_split['train']))
            except FileNotFoundError as e:
                print(f'Version {version} not found.', e)
            except Exception as e:
                print(f'Error in version {version}.', e)
        mid = time.time()
        print('Time for loading dataset:', mid - start)

        train_set = ConcatDataset(train_set)

        end = time.time()
        print('Time for Concat:', end - mid)

        return dataset, {"train": train_set, "test": test_set}
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)
    
    # print(config)

    if config.with_split:
        split = torch.load(config.split)
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset, {"train": dataset, "test": dataset}

