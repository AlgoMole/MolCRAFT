import os
import pickle
import lmdb
from tqdm.auto import tqdm
from core.datasets.pl_data import ProteinLigandData
import torch

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
            self.keys = sorted(list(txn.cursor().iternext(values=False)))

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
        if hasattr(data, 'protein_pos'):
            assert data.protein_pos.size(0) > 0, f'Empty protein_pos: {data.ligand_filename}, {data.protein_pos.size()}'
        return data

    def add(self, add_db):
        if self.db is None:
            self._connect_db()
        offset = len(self)
        with self.db.begin(write=True, buffers=True) as txn:
            for i in tqdm(range(len(add_db)), desc='Merging'):
                data = add_db[i]
                data.id = i + offset
                txn.put(str(data.id).encode(), pickle.dumps(data))
        self._close_db()
        

if __name__ == '__main__':
    dst_dir = './data/'
    dst_path = os.path.join(dst_dir, 'PDB+MOAD_processed_final.lmdb')
    pdb_path = os.path.join(dst_dir, 'PDBBind_processed_final.lmdb')
    moad_path = os.path.join(dst_dir, 'BindingMOAD_2020_pocket10_processed_final.lmdb')
    dataset_pdb = DBReader(pdb_path)
    dataset_moad = DBReader(moad_path)

    train_names = torch.load('./data/PDB+MOAD_train_names.pt')
    test_names = torch.load('./data/PDB+MOAD_test_names.pt') 

    pose_split = {'train': [], 'test': []}
    num_skipped = 0

    # db = lmdb.open(dst_path, map_size=10*(1024*1024*1024), create=True, subdir=False, readonly=False, lock=False, readahead=False, meminit=False, max_readers=256)
    # with db.begin(write=True, buffers=True) as txn:
    #     idx = -1
    #     for i in tqdm(range(len(dataset_pdb)), desc='Merging PDB'):
    #         idx += 1
    #         try:
    #             data = dataset_pdb[i]
    #             data.id = idx
    #             data = data.to_dict()
    #             txn.put(str(idx).encode(), pickle.dumps(data))
    #             protein_fn = data.protein_filename
    #             ligand_fn = data.ligand_filename
    #             if (protein_fn, ligand_fn) in train_names:
    #                 pose_split['train'].append(idx)
    #             elif (protein_fn, ligand_fn) in test_names:
    #                 pose_split['test'].append(idx)
    #         except Exception as e:
    #             num_skipped += 1
    #             # print(f'Error: {e}, {i}')
    #     for i in tqdm(range(len(dataset_moad)), desc='Merging MOAD'):
    #         idx += 1
    #         try:
    #             data = dataset_moad[i]
    #             data.id = idx
    #             data = data.to_dict()
    #             txn.put(str(idx).encode(), pickle.dumps(data))
    #             protein_fn = data.protein_filename
    #             ligand_fn = data.ligand_filename
    #             if (protein_fn, ligand_fn) in train_names:
    #                 pose_split['train'].append(idx)
    #             elif (protein_fn, ligand_fn) in test_names:
    #                 pose_split['test'].append(idx)
    #         except Exception as e:
    #             num_skipped += 1
    #             # print(f'Error: {e}, {i}')
        
    # db.close()

    ############################################################

    dataset = DBReader(dst_path)
    print(len(dataset), dataset[0])

    pose_split_prev = torch.load('./data/PDB+MOAD_pose_split_filtered_qed17.pt')
    
    for i in tqdm(pose_split_prev['train'], desc='Checking Train'):
        try:
            data = dataset[i]
            protein_fn = data.protein_filename
            ligand_fn = data.ligand_filename
            charge = data.ligand_charge.tolist()
            # if any charge is not 0 / -1 / 1, skip
            if any([c not in [-1, 0, 1] for c in charge]):
                num_skipped += 1
                continue
            
            assert (protein_fn, ligand_fn) in train_names, f'Not in train: {i}, {protein_fn}, {ligand_fn}'
            pose_split['train'].append(i)
        except Exception as e:
            num_skipped += 1
            # print(f'Error: {e}, {i}')

    for i in tqdm(pose_split_prev['test'], desc='Checking Test'):
        try:
            data = dataset[i]
            protein_fn = data.protein_filename
            ligand_fn = data.ligand_filename
            charge = data.ligand_charge.tolist()
            # if any charge is not 0 / -1 / 1, skip
            if any([c not in [-1, 0, 1] for c in charge]):
                num_skipped += 1
                continue
            
            assert (protein_fn, ligand_fn) in test_names, f'Not in test: {i}, {protein_fn}, {ligand_fn}'
            pose_split['test'].append(i)
        except Exception as e:
            num_skipped += 1
            # print(f'Error: {e}, {i}')

    print(len(train_names), len(test_names))
    torch.save(pose_split, './data/PDB+MOAD_pose_split_filtered_qed17_charge.pt')
    print(f'Train: {len(pose_split["train"])}, Test: {len(pose_split["test"])}, Skipped: {num_skipped}')