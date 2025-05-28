import torch
from torch.utils.data import Subset
from core.datasets.pl_pair_dataset import PocketLigandPairDataset, PocketLigandPairDatasetFeaturized, PocketLigandGeneratedPairDataset
from core.datasets.pdbbind import PDBBindDataset


def get_dataset(config, *args, **kwargs):
    name = config.name
    root = config.path
    ligand_atom_mode = config.transform.ligand_atom_mode
    if name == 'pl':
        dataset = PocketLigandPairDataset(root, *args, **kwargs)
    elif name == 'pl_tr':
        dataset = PocketLigandPairDatasetFeaturized(root, ligand_atom_mode=ligand_atom_mode,
                                                     *args, **kwargs)
        return dataset, {"train": dataset.train_data, "test": dataset.test_data}
    elif name == 'pl_dcmp':
        dataset = PocketLigandGeneratedPairDataset(root, *args, **kwargs)
        return dataset, {"train": dataset, "test": dataset}
    elif name == 'pdbbind':
        dataset = PDBBindDataset(root, *args, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)
    
    # print(config)

    if config.with_split:
        split = torch.load(config.split)
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset
