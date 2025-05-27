import os
import pickle
import lmdb
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import sys
from time import time

import torch
from torch_geometric.transforms import Compose

from core.datasets.utils import PDBProtein, parse_sdf_file, ATOM_FAMILIES_ID
from core.datasets.pl_data import ProteinLigandData, torchify_dict

import core.utils.transforms as trans


class DBReader:
    def __init__(self, path) -> None:
        self.path = path
        self.db = None
        self.keys = None

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def __del__(self):
        if self.db is not None:
            self._close_db()

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data = ProteinLigandData(**data)
        data.id = idx
        assert data.protein_pos.size(0) > 0
        return data
    

class PocketLigandPairDataset(Dataset):

    def __init__(self, raw_path, transform=None, version='final'):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')
        self.transform = transform
        self.reader = DBReader(self.processed_path)

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()
        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, *_) in enumerate(tqdm(index)):
                if pocket_fn is None: continue
                try:
                    # data_prefix = '/data/work/jiaqi/binding_affinity'
                    data_prefix = self.raw_path
                    pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
                    ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )
                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    data = data.to_dict()  # avoid torch_geometric version issue
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))
                    continue
        db.close()
    
    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        data = self.reader[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data


class PocketLigandGeneratedPairDataset(Dataset):

    def __init__(self, raw_path, transform=None, version='4-decompdiff'):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.generated_path = os.path.join('/sharefs/share/sbdd_data/all_results', f'{version}_docked_pose_checked.pt')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')
        self.transform = transform
        self.reader = DBReader(self.processed_path)

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()
        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.generated_path, 'rb') as f:
            results = torch.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            idx = -1
            for i, res in tqdm(enumerate(results), total=len(results)):
                if isinstance(res, dict):
                    res = [res]
                for r in res:
                    idx += 1
                    mol = r["mol"]
                    ligand_fn = r["ligand_filename"]
                    pocket_fn = os.path.join(
                        os.path.dirname(ligand_fn),
                        os.path.basename(ligand_fn)[:-4] + '_pocket10.pdb'
                    )

                    if pocket_fn is None: continue
                    try:
                        data_prefix = self.raw_path
                        pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
                        ligand_dict = parse_sdf_file(mol)
                        # ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
                        data = ProteinLigandData.from_protein_ligand_dicts(
                            protein_dict=torchify_dict(pocket_dict),
                            ligand_dict=torchify_dict(ligand_dict),
                        )
                        data.protein_filename = pocket_fn
                        data.ligand_filename = ligand_fn
                        data = data.to_dict()  # avoid torch_geometric version issue
                        txn.put(
                            key=str(idx).encode(),
                            value=pickle.dumps(data)
                        )
                    except Exception as e:
                        num_skipped += 1
                        print('Skipping (%d) %s' % (num_skipped, ligand_fn, ), e)
                        continue
        db.close()
    
    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        data = self.reader[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data


class PocketLigandPairDatasetFromComplex(Dataset):
    def __init__(self, raw_path, transform=None, version='final', radius=10.0):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        base_name = os.path.basename(self.raw_path)
        if 'pocket' in base_name:
            self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                               os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')
        else:
            self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                            os.path.basename(self.raw_path) + f'_pocket{radius}_processed_{version}.lmdb')
        self.transform = transform
        self.reader = DBReader(self.processed_path)

        self.radius = radius

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 50GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
            max_readers=256,
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        print('Processing data...', 'index', self.index_path, index[0])

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, *_) in enumerate(tqdm(index)):
                if pocket_fn is None: continue
                try:
                    data_prefix = self.raw_path
                    # clip pocket
                    ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
                    protein = PDBProtein(os.path.join(data_prefix, pocket_fn))
                    selected = protein.query_residues_ligand(ligand_dict, self.radius)
                    pdb_block_pocket = protein.residues_to_pdb_block(selected)
                    pocket_dict = PDBProtein(pdb_block_pocket).to_dict_atom()

                    # pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
                    # ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
                    
                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )
                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    data = data.to_dict()  # avoid torch_geometric version issue
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except Exception as e:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn, ), e)
                    with open('skipped.txt', 'a') as f:
                        f.write('Skip %s due to %s\n' % (ligand_fn, e))
                    continue
        db.close()

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        data = self.reader[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data
    

class PocketLigandPairDatasetFeaturized(Dataset):
    def __init__(self, raw_path, ligand_atom_mode, version='simple'):
        """
        in simple version, only these features are saved for better IO:
            protein_pos, protein_atom_feature, protein_element, 
            ligand_pos, ligand_atom_feature_full, ligand_element
        """
        self.raw_path = raw_path
        self.ligand_atom_mode = ligand_atom_mode
        self.version = version

        if version == 'simple':
            self.features_to_save = [
                'protein_pos', 'protein_atom_feature', 'protein_element', 
                'ligand_pos', 'ligand_atom_feature_full', 'ligand_element',
                'protein_filename', 'ligand_filename',
            ]
        else:
            raise NotImplementedError

        self.transformed_path = os.path.join(
            os.path.dirname(self.raw_path), os.path.basename(self.raw_path) + 
            f'_{ligand_atom_mode}_transformed_{version}.pt'
        )
        if not os.path.exists(self.transformed_path):
            print(f'{self.transformed_path} does not exist, begin transforming data')
            self._transform()
        else:
            print(f'reading data from {self.transformed_path}...')
            tic = time()
            tr_data = torch.load(self.transformed_path)
            toc = time()
            print(f'{toc - tic} elapsed')
            self.train_data, self.test_data = tr_data['train'], tr_data['test']
            self.protein_atom_feature_dim = tr_data['protein_atom_feature_dim']
            self.ligand_atom_feature_dim = tr_data['ligand_atom_feature_dim']
        
    def _transform(self):
        raw_dataset = PocketLigandPairDataset(self.raw_path, None, 'final')

        split_path = os.path.join(
            os.path.dirname(self.raw_path), 'crossdocked_pocket10_pose_split.pt',
        )
        split = torch.load(split_path)
        train_ids, test_ids = split['train'], split['test']
        print(f'train_size: {len(train_ids)}, test_size: {len(test_ids)}')

        protein_featurizer = trans.FeaturizeProteinAtom()
        ligand_featurizer = trans.FeaturizeLigandAtom(self.ligand_atom_mode)
        transform_list = [
            protein_featurizer,
            ligand_featurizer,
            # trans.FeaturizeLigandBond(),
        ]
        transform = Compose(transform_list)
        self.protein_atom_feature_dim = protein_featurizer.feature_dim
        self.ligand_atom_feature_dim = ligand_featurizer.feature_dim

        def _transform_subset(ids):
            data_list = []

            for idx in tqdm(ids):
                data = raw_dataset[idx]
                data = transform(data)
                tr_data = {}
                for k in self.features_to_save:
                    tr_data[k] = getattr(data, k)
                tr_data['id'] = idx
                tr_data = ProteinLigandData(**tr_data)
                data_list.append(tr_data)
            return data_list

        self.train_data = _transform_subset(train_ids)
        print(f'train_size: {len(self.train_data)}, {sys.getsizeof(self.train_data)}')
        self.test_data = _transform_subset(test_ids)
        print(f'test_size: {len(self.test_data)}, {sys.getsizeof(self.test_data)}')
        torch.save({
            'train': self.train_data, 'test': self.test_data,
            'protein_atom_feature_dim': self.protein_atom_feature_dim,
            'ligand_atom_feature_dim': self.ligand_atom_feature_dim,
        }, self.transformed_path)


if __name__ == '__main__':
    # original dataset
    dataset = PocketLigandPairDataset('./data/crossdocked_v1.1_rmsd1.0_pocket10')
    print(len(dataset), dataset[0])

    ############################################################

    # test DecompDiffDataset
    # dataset = PocketLigandGeneratedPairDataset('/sharefs/share/sbdd_data/crossdocked_pocket10')
    # print(len(dataset), dataset[0])

    ############################################################

    # test featurized dataset (GPU accelerated)
    # path = '/sharefs/share/sbdd_data/crossdocked_v1.1_rmsd1.0_pocket10'
    # ligand_atom_mode = 'add_aromatic'

    # dataset = PocketLigandPairDatasetFeaturized(path, ligand_atom_mode)
    # train_data, test_data = dataset.train_data, dataset.test_data
    # print(f'train_size: {len(train_data)}, {sys.getsizeof(train_data)}')
    # print(f'test_size: {len(test_data)}, {sys.getsizeof(test_data)}')
    # print(test_data[0], sys.getsizeof(test_data[0]))

    ############################################################

    # test featurization
    # find all atom types
    # atom_types = {(1, False): 0}

    # dataset = PocketLigandPairDataset(path, transform)
    # for i in tqdm(range(len(dataset))):
    #     data = dataset[i]
    #     element_list = data.ligand_element
    #     hybridization_list = data.ligand_hybridization
    #     aromatic_list = [v[trans.AROMATIC_FEAT_MAP_IDX] for v in data.ligand_atom_feature]

    #     types = [(e, a) for e, h, a in zip(element_list, hybridization_list, aromatic_list)]
    #     for t in types:
    #         t = (t[0].item(), bool(t[1].item()))
    #         if t not in atom_types:
    #             atom_types[t] = 0
    #         atom_types[t] += 1

    # idx = 0
    # for k in sorted(atom_types.keys()):
    #     print(f'{k}: {idx}, # {atom_types[k]}')
    #     idx += 1

    ############################################################
    
    # count atom types
    # type_counter, aromatic_counter, full_counter = {}, {}, {}
    # for i, data in enumerate(tqdm(dataset)):
    #     element_list = data.ligand_element
    #     hybridization_list = data.ligand_hybridization
    #     aromatic_list = [v[trans.AROMATIC_FEAT_MAP_IDX] for v in data.ligand_atom_feature]
    #     flag = False

    #     for atom_type in element_list:
    #         atom_type = int(atom_type.item())
    #         if atom_type not in type_counter:
    #             type_counter[atom_type] = 0
    #         type_counter[atom_type] += 1

    #     for e, a in zip(element_list, aromatic_list):
    #         e = int(e.item())
    #         a = bool(a.item())
    #         key = (e, a)
    #         if key not in aromatic_counter:
    #             aromatic_counter[key] = 0
    #         aromatic_counter[key] += 1

    #         if key not in trans.MAP_ATOM_TYPE_AROMATIC_TO_INDEX:
    #             flag = True

    #     for e, h, a in zip(element_list, hybridization_list, aromatic_list):
    #         e = int(e.item())
    #         a = bool(a.item())
    #         key = (e, h, a)
    #         if key not in full_counter:
    #             full_counter[key] = 0
    #         full_counter[key] += 1
        
    # print('type_counter', type_counter)
    # print('aromatic_counter', aromatic_counter)
    # print('full_counter', full_counter)

