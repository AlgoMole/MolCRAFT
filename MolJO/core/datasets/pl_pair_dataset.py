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
    def __init__(self, path, affinity_path=None) -> None:
        self.path = path
        self.affinity_path = affinity_path
        self.db = None
        self.keys = None
        self.affinity_info = None

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
        # if self.affinity_path is not None:
        #     if 'affinity' not in data:
        #         self._load_affinity_info()
        #         self._inject_affinity(key, data['ligand_filename'])
        #         data = pickle.loads(self.db.begin().get(key))
        data = ProteinLigandData(**data)
        data.id = idx
        assert data.protein_pos.size(0) > 0
        return data
    
    def _update(self, sid, affinity):
        if self.db is None:
            self._connect_db()
        txn = self.db.begin(write=True)
        data = pickle.loads(txn.get(sid))
        data.update({
            'affinity': affinity['vina'],
            'rmsd': affinity['rmsd'],
            'pk': affinity['pk'],
        })
        txn.put(
            key=sid,
            value=pickle.dumps(data)
        )
        txn.commit()

    def _load_affinity_info(self):
        if self.affinity_info is not None:
            return
        if os.path.exists(self.affinity_path):
            with open(self.affinity_path, 'rb') as f:
                affinity_info = pickle.load(f)
        else:
            raise FileNotFoundError(f'Affinity info not found at {self.affinity_path}')
            affinity_info = {}
            with open(self.raw_affinity_path, 'r') as f:
                for ln in tqdm(f.readlines()):
                    # <label> <pK> <RMSD to crystal> <Receptor> <Ligand> # <Autodock Vina score>
                    label, pk, rmsd, protein_fn, ligand_fn, vina = ln.split()
                    ligand_raw_fn = ligand_fn[:ligand_fn.rfind('.')]
                    affinity_info[ligand_raw_fn] = {
                        'label': float(label),
                        'rmsd': float(rmsd),
                        'pk': float(pk),
                        'vina': float(vina[1:])
                    }
            # save affinity info
            with open(self.affinity_path, 'wb') as f:
                pickle.dump(affinity_info, f)
        
        self.affinity_info = affinity_info

    def _inject_affinity(self, sid, ligand_path):
        if ligand_path[:-4] in self.affinity_info:
            affinity = self.affinity_info[ligand_path[:-4]]
            self._update(sid, affinity)
        else:
            raise AttributeError(f'affinity_info has no {ligand_path[:-4]}')



class PocketLigandPairDataset(Dataset):

    def __init__(self, raw_path, transform=None, version='final'):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')
        self.transform = transform
        affinity_path = os.path.join(os.path.dirname(self.raw_path), 'affinity_info_complete.pkl')
        if not os.path.exists(affinity_path):
            affinity_path = None
        self.reader = DBReader(self.processed_path, affinity_path)

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

    def __init__(self, raw_path, transform=None, version='decompdiff20'):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.generated_path = os.path.join('./data/all_results', f'decompdiff_vina_docked_pose_checked_v3.pt')
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
        ligand_cnt = {}
        with db.begin(write=True, buffers=True) as txn:
            idx = -1
            for i, res in tqdm(enumerate(results), total=len(results)):
                if isinstance(res, dict):
                    res = [res]
                for r in res:
                    idx += 1
                    mol = r["mol"]
                    ligand_fn = r["ligand_filename"]
                    ligand_cnt[ligand_fn] = ligand_cnt.get(ligand_fn, 0) + 1
                    if ligand_cnt[ligand_fn] > 20:
                        continue
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
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                            os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')
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
                    # data_prefix = '/data/work/jiaqi/binding_affinity'
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
    def __init__(self, raw_path, ligand_atom_mode, version='simple', split='crossdocked_pocket10_pose_split.pt'):
        """
        in simple version, only these features are saved for better IO:
            protein_pos, protein_atom_feature, protein_element, 
            ligand_pos, ligand_atom_feature_full, ligand_element
        """
        self.raw_path = raw_path
        self.ligand_atom_mode = ligand_atom_mode
        self.version = version
        self.pose_split = split

        if version == 'simple':
            self.features_to_save = [
                'protein_pos', 'protein_atom_feature', 'protein_element', 
                'ligand_pos', 'ligand_atom_feature_full', 'ligand_element',
                'protein_filename', 'ligand_filename',
            ]
        elif version == 'guided':
            self.features_to_save = [
                'protein_pos', 'protein_atom_feature', 'protein_element', 
                'ligand_pos', 'ligand_atom_feature_full', 'ligand_element',
                'protein_filename', 'ligand_filename', 
                'affinity', 'qed', 'sa',
            ]
        elif version == 'guided_v2':
            self.features_to_save = [
                'protein_pos', 'protein_atom_feature', 'protein_element', 
                'ligand_pos', 'ligand_atom_feature_full', 'ligand_element',
                'protein_filename', 'ligand_filename', 
                'affinity', 'qed', 'sa', 'qed_norm', 'sa_norm',
            ]
        elif version == 'guided_v3':
            self.features_to_save = [
                'protein_pos', 'protein_atom_feature', 'protein_element', 
                'ligand_pos', 'ligand_atom_feature_full', 'ligand_element',
                'protein_filename', 'ligand_filename', 
                'affinity', 'qed', 'sa', 'qed_norm', 'sa_norm',
                'lipinski', 'lipinski_norm',
            ]
        elif version == 'guided_v4':
            self.features_to_save = [
                'protein_pos', 'protein_atom_feature', 'protein_element', 
                'ligand_pos', 'ligand_atom_feature_full', 'ligand_element',
                'protein_filename', 'ligand_filename', 
                'affinity', 'qed', 'sa', 'qed_norm', 'sa_norm',
                'lipinski', 'lipinski_norm',
            ]
        elif version == 'guided_v5':
            self.features_to_save = [
                'protein_pos', 'protein_atom_feature', 'protein_element', 
                'ligand_pos', 'ligand_atom_feature_full', 'ligand_element',
                'protein_filename', 'ligand_filename', 
                'affinity', 'qed', 'sa', 'qed_norm', 'sa_norm',
                'lipinski', 'lipinski_norm',
                'ligand_mask',
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
        if 'PDBLigAug' in self.raw_path:
            raw_dataset = PocketLigandPairDatasetFromComplex(self.raw_path)
        else:
            raw_dataset = PocketLigandPairDataset(self.raw_path, None, 'final')

        split_path = os.path.join(
            os.path.dirname(self.raw_path), self.pose_split,
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
        if self.version is not None and 'guided' in self.version:
            transform_list.extend([
                trans.NormalizeVina(),
                trans.AddMolProp(),
            ])
            if 'guided_v4' in self.version:
                interaction_types = ['Pi', 'PiEdge', 'PiFace', 'PiCat', 'HAccep', 'HDonor', 'XBond', 'Salt']
                self.features_to_save.extend(interaction_types)
                transform_list.append(trans.LoadInteraction('./in_cross', interaction_types))
            elif 'guided_v5' in self.version:
                transform_list.append(trans.AddScaffoldMask('./Mask_cd_test.pkl'))
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
    # dataset = PocketLigandPairDataset('./data/crossdocked_pocket10')
    # print(len(dataset), dataset[0])

    ############################################################
    # test PocketLigandPairDatasetFromComplex
    # path = './data/PDBLigAug_v2'
    # dataset = PocketLigandPairDatasetFromComplex(path)
    # print(len(dataset), dataset[0])
    # allowed_elements = {1, 6, 7, 8, 9, 15, 16, 17, 35}
    # elements = {i: set() for i in range(90)}
    # for i, data in enumerate(tqdm(dataset, desc='Filter')):
    #     for e in data.ligand_element:
    #         elements[e.item()].add(i)

    # all_id = set(range(len(dataset)))
    # blocked_id = set().union(*[
    #     elements[i] for i in elements.keys() if i not in allowed_elements
    # ])

    # allowed_id = list(all_id - blocked_id)
    # print('Allowed: %d' % len(allowed_id))
    # with open(f'{path}/allowed_id_35.pkl', 'wb') as f:
    #     pickle.dump(allowed_id, f)

    # test featurized dataset (GPU accelerated)
    # ligand_atom_mode = 'add_aromatic_pdb'

    # dataset = PocketLigandPairDatasetFeaturized(path, ligand_atom_mode, split='PDBLigAug_v2_pose_split_35.pt')
    # train_data, test_data = dataset.train_data, dataset.test_data
    # print(f'train_size: {len(train_data)}, {sys.getsizeof(train_data)}')
    # print(f'test_size: {len(test_data)}, {sys.getsizeof(test_data)}')
    # print(test_data[0], sys.getsizeof(test_data[0]))
        
    ############################################################

    # test DecompDiffDataset
    dataset = PocketLigandGeneratedPairDataset('./data/crossdocked_pocket10')
    print(len(dataset), dataset[0])

    ############################################################

    # test featurized dataset (GPU accelerated)
    # path = './data/crossdocked_v1.1_rmsd1.0_pocket10'
    # ligand_atom_mode = 'add_aromatic'

    # dataset = PocketLigandPairDatasetFeaturized(path, ligand_atom_mode, version='guided_v5')
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

