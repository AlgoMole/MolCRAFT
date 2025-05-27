import os
import pickle
import lmdb
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import random
import sys
from time import time
import json
from multiprocessing import Pool
from functools import lru_cache

import torch
from torch_geometric.transforms import Compose
from rdkit import Chem

from core.datasets.utils import PDBProtein, PDBProteinNew, parse_sdf_file, ATOM_FAMILIES_ID
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
            max_readers=256,
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

        # if 'ligand_charge' not in data:
        #     try:
        #         self._inject_charge(key, data)
        #         data = pickle.loads(self.db.begin().get(key))
        #     except Exception as e:
        #         print(f'Failed to inject charge for {key}: {e}')
        #         return None

        data = ProteinLigandData(**data)
        # data.id = idx
        if hasattr(data, 'protein_pos'):
            assert data.protein_pos.size(0) > 0, f'Empty protein_pos: {data.ligand_filename}, {data.protein_pos.size()}'
        return data

    def _inject_charge(self, sid, data):
        # For PDBBind
        # data_prefix = './data/combine_set'
        # For Crossdock
        data_prefix = './data/crossdocked_pocket10'
        txn = self.db.begin(write=True)
        data = pickle.loads(txn.get(sid))

        ligand_fn = data["ligand_filename"]
        ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
        # assert the remaining keys are the same
        # print(data.keys())

        if data["ligand_element"].size(0) != len(ligand_dict['element']):
            update_dict = {}
            for key in data.keys():
                update_dict = {
                    f'ligand_{key}': data[key]
                }
            data.update(update_dict)
            data["ligand_smiles"] = ligand_dict['smiles']
            print(data, ligand_dict)
  
        data.update({
            'ligand_charge': ligand_dict['charge'],
        })
        txn.put(
            key=sid,
            value=pickle.dumps(data)
        )
        txn.commit()
    

class PocketLigandPairDataset(Dataset):

    def __init__(self, raw_path, transform=None, version='final'):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')
        self.transform = transform
        self.reader = DBReader(self.processed_path)

        self.add_hydrogen = False
        self.kekulize = version == 'kekulize'
        self.version = version

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data (add_hydrogen={self.add_hydrogen}, kekulize={self.kekulize})')
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
                    ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn), add_hydrogen=self.add_hydrogen, kekulize=self.kekulize)
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
        return self.__getitem_uncached(idx)
    
    # @lru_cache(maxsize=16)
    def __getitem_uncached(self, idx):
        data = self.reader[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data
    
class PocketLigandGeneratedPairDataset(Dataset):

    def __init__(self, raw_path, transform=None, version='4-decompdiff'):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.generated_path = os.path.join('./data/all_results', f'{version}_docked_pose_checked.pt')
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
    def __init__(self, raw_path, transform=None, version='final', radius=10):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        base_name = os.path.basename(self.raw_path)
        if 'pocket' in base_name:
            # replace pocket%d with pocket{radius}
            import re
            self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                               re.sub(r'pocket\d+', f'pocket{radius}', base_name) + f'_processed_{version}.lmdb')
        else:
            self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                            os.path.basename(self.raw_path) + f'_pocket{radius}_processed_{version}.lmdb')
        self.transform = transform
        self.reader = DBReader(self.processed_path)
        self.kekulize = version == 'kekulize'

        self.radius = radius
        # if lmdb is smaller than 1M, then remove it
        if os.path.exists(self.processed_path):
            if os.path.getsize(self.processed_path) < 1024 * 1024:
                os.remove(self.processed_path)
                print(f'{self.processed_path} is too small, removed')
                if os.path.exists(self.processed_path + '-lock'):
                    os.remove(self.processed_path + '-lock')
                    print(f'{self.processed_path}-lock is removed')

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
                    ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn), kekulize=self.kekulize)
                    protein = PDBProteinNew(os.path.join(data_prefix, pocket_fn))
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
                    print('Skipping (%d) %s due to %s' % (num_skipped, ligand_fn, str(e)))
                    with open('skipped.txt', 'a') as f:
                        f.write('Skip %s due to %s\n' % (ligand_fn, str(e)))
                    # stop if exception occurs
                    # raise e
        db.close()

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        data = self.reader[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data
    
def parse_sdf_to_dict(mol, i):
    ligand_dict = parse_sdf_file(mol)
    data = ProteinLigandData.from_protein_ligand_dicts(
        ligand_dict=torchify_dict(ligand_dict),
    )
    data.ligand_filename = f'{i}.sdf'
    return pickle.dumps(data)

def process_entry(i, pocket_fn, ligand_fn, processed_path, data_prefix, radius):
    try:
        ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn), kekulize=True)
        protein = PDBProteinNew(os.path.join(data_prefix, pocket_fn))
        selected = protein.query_residues_ligand(ligand_dict, radius)
        pdb_block_pocket = protein.residues_to_pdb_block(selected)
        pocket_dict = PDBProtein(pdb_block_pocket).to_dict_atom()

        data = ProteinLigandData.from_protein_ligand_dicts(
            protein_dict=torchify_dict(pocket_dict),
            ligand_dict=torchify_dict(ligand_dict),
        )
        data.protein_filename = pocket_fn
        data.ligand_filename = ligand_fn
        data = data.to_dict()

        # write data to a tmp file
        torch.save(data, f'./tmp/{i}.pt')

        return i
    except Exception as e:
        print(f'Skipping {ligand_fn} due to {str(e)}')
        with open('skipped.txt', 'a') as f:
            f.write(f'Skip {ligand_fn} due to {str(e)}\n')
        return None
    
def writer_process(queue, db_path, write_frequency):
    env = lmdb.open(db_path, map_size=10*(1024*1024*1024), create=True, subdir=False, readonly=False, max_readers=256)
    txn = env.begin(write=True, buffers=True)
    counter = 0

    while True:
        batch = []
        item = 0
        # Collect items in batch for more efficient writes
        while not queue.empty():
            item = queue.get()
            if item is None:
                break
            batch.append(item)

        # Commit batch to LMDB
        for i in batch:
            data = torch.load(f'./tmp/{i}.pt')
            txn.put(key=str(i).encode(), value=pickle.dumps(data))
            counter += 1
            if counter % write_frequency == 0:
                print(f'Committing {counter} items...')
                txn.commit()
                txn = env.begin(write=True, buffers=True)
        
        if queue.empty() and item is None:  # Stop signal received
            print(f'Stopped.')
            break

        if len(batch) > 0:
            print(f'Writing {len(batch)} items...') # TODO: removing this line seems to cause deadlock
            batch = []
    
    # Final commit and close
    txn.commit()
    env.close()

class PocketLigandPairDatasetFromComplexParallel(Dataset):
    def __init__(self, raw_path, transform=None, version='final', radius=10):
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

        # if lmdb is smaller than 1M, then remove it
        if os.path.exists(self.processed_path):
            if os.path.getsize(self.processed_path) < 1024 * 1024:
                os.remove(self.processed_path)
                print(f'{self.processed_path} is too small, removed')
                if os.path.exists(self.processed_path + '-lock'):
                    os.remove(self.processed_path + '-lock')
                    print(f'{self.processed_path}-lock is removed')

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()

    def _process(self):
        import multiprocessing as mp
        from functools import partial
        # Set the multiprocessing start method
        mp.set_start_method("fork", force=True)

        if not os.path.exists('./tmp'):
            os.makedirs('./tmp')

        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)
        
        print('Processing data...', 'index', self.index_path, index[0])

        # Initialize multiprocessing
        # lock = mp.Manager().Lock()
        queue = mp.Queue(maxsize=1000)  # TODO: check if removing maxsize constraint and commenting write ... line causes deadlock
        print(f'Using 20 out of {mp.cpu_count()} cores')
        pool = mp.Pool(20)
        pbar = tqdm(total=len(index), desc='Processing')

        # Combined callback function to both update progress and add to queue
        def put_and_update(result):
            if result is not None:
                queue.put(result)
            pbar.update(1)
            # Send stop signal to writer
            if len(results) == len(index):
                queue.put(None)

        writer = mp.Process(target=writer_process, args=(queue, self.processed_path, 500))
        writer.start()
        # Use partial to freeze arguments and pass lock to each worker
        process_func = partial(process_entry, processed_path=self.processed_path, data_prefix=self.raw_path, radius=self.radius)

        # Run in parallel and gather results
        results = []
        for i, (pocket_fn, ligand_fn, *_) in enumerate(tqdm(index)):
            if pocket_fn:
                results.append(pool.apply_async(process_func, args=(i, pocket_fn, ligand_fn), callback=put_and_update))
            
        # Close pool
        pool.close()
        pool.join()
        writer.join()

        # Cleanup
        # os.system('rm -r ./tmp')

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        data = self.reader[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data

    def _fix(self):
        # some entries are missing, and not committed to lmdb
        # fix the lmdb by reprocessing the missing entries
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            subdir=False,
            readonly=False,  # Writable
        )

        num_fixed, num_missing = 0, 0
        index = pickle.load(open(self.index_path, 'rb'))
        with db.begin(write=True, buffers=True) as txn:
            for i in tqdm(range(len(index))):
                if txn.get(str(i).encode()) is None:
                    num_missing += 1
                    if os.path.exists(f'./tmp/{i}.pt'):
                        data = torch.load(f'./tmp/{i}.pt')
                        txn.put(
                            key=str(i).encode(),
                            value=pickle.dumps(data)
                        )
                        num_fixed += 1
        
        db.close()
        print(f'Fixed {num_fixed} entries for {num_missing} missing entries')
    


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
                'ligand_fc_bond_index', 'ligand_fc_bond_type',
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
            
            # TODO: temp fix, filter out n_nodes >= threshold
            self.train_data = [d for d in self.train_data if len(d.protein_pos) + len(d.ligand_pos) + len(d.ligand_fc_bond_type) <= 4000]
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
            trans.FeaturizeLigandBond(),
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



class MoleculeDataset(Dataset):

    def __init__(self, raw_path, transform=None, version='final'):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path).replace('.sdf', '') + f'_processed_{version}.lmdb')
        self.transform = transform
        self.reader = DBReader(self.processed_path)

        self.add_hydrogen = False
        self.kekulize = version == 'kekulize'

        # if lmdb is smaller than 10M, then remove it
        if os.path.exists(self.processed_path):
            if os.path.getsize(self.processed_path) < 10 * 1024 * 1024:
                os.remove(self.processed_path)
                print(f'{self.processed_path} is too small, removed')
                if os.path.exists(self.processed_path + '-lock'):
                    os.remove(self.processed_path + '-lock')
                    print(f'{self.processed_path}-lock is removed')

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data (add_hydrogen={self.add_hydrogen}, kekulize={self.kekulize})')
            self._process()
        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=200*(1024*1024*1024),   # 200GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        suppl = Chem.SDMolSupplier(self.raw_path, removeHs=not self.add_hydrogen)

        molist = list(suppl)
        torch.multiprocessing.set_sharing_strategy('file_system')
        pool = Pool(processes=16)
        results = pool.map(parse_sdf_to_dict, molist, range(len(molist)))
        pool.close()
        pool.join()

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, res in enumerate(results):
                if res is None:
                    num_skipped += 1
                    continue
                try:
                    txn.put(
                        key=str(i).encode(),
                        value=res
                    )
                except:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, i, ))
                    continue

        db.close()
    
    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        data = self.reader[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data


if __name__ == '__main__':
    # original dataset
    dataset = PocketLigandPairDataset('./data/crossdocked_v1.1_rmsd1.0_pocket10', version='final')
    # dataset = PocketLigandPairDataset('./data/crossdocked_pocket10', version='final')
    print(len(dataset), dataset[0])
    