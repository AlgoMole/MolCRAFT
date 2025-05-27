import os
import pickle
import lmdb
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from multiprocessing import Pool

import torch
from rdkit import Chem
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

from core.datasets.utils import parse_sdf_file
from core.datasets.pl_data import ProteinLigandData, torchify_dict
from core.evaluation.utils.scoring_func import get_chem
from collections import Counter

import random

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


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
        if not isinstance(data, ProteinLigandData):
            data = ProteinLigandData(**data)
        # data.id = idx
        if hasattr(data, 'protein_pos'):
            assert data.protein_pos.size(0) > 0, f'Empty protein_pos: {data.ligand_filename}, {data.protein_pos.size()}'
        return data

    def _inject_charge(self, sid, data):
        # For PDBBind
        # data_prefix = '/sharefs/share/sbdd_data/combine_set'
        # For Crossdock
        data_prefix = '/sharefs/share/sbdd_data/crossdocked_pocket10'
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
       
def parse_sdf_to_dict(mol, i, kekulize):
    try:
        ligand_dict = parse_sdf_file(mol, kekulize=kekulize)
        data = ProteinLigandData.from_protein_ligand_dicts(
            ligand_dict=torchify_dict(ligand_dict),
        )
        data.ligand_filename = f'{i}.sdf'
        return pickle.dumps(data)
    except:
        return None


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
        self.version = version

        # if lmdb is smaller than 2M, then remove it
        if os.path.exists(self.processed_path):
            if os.path.getsize(self.processed_path) < 2 * 1024 * 1024:
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
            map_size=200*(1024*1024*1024),   # 200GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        import time
        import glob

        print(f'Processing {self.raw_path}')
        sdf_list = glob.glob(self.raw_path + '/*/*/*.sdf')
        # sdf_list = sorted(sdf_list)

        index = 0
        for sdf_supplier in tqdm(sdf_list):
            # start = time.time()
            try:
                molist = list(Chem.ForwardSDMolSupplier(sdf_supplier, removeHs=not self.add_hydrogen))
                # randomly sample 1k from molist
                if len(molist) > 1000:
                    molist = random.sample(molist, 1000)
            except Exception as e:
                print(f'Error reading {sdf_supplier}: {e}')
                continue

            # print(f'Reading {len(molist)} molecules. Time elapsed: {time.time() - start:.2f}s')

            # torch.multiprocessing.set_sharing_strategy('file_system')
            with mp.Pool(mp.cpu_count()) as pool:
                results = list(tqdm(pool.starmap(parse_sdf_to_dict, zip(molist, list(range(len(molist))), [self.kekulize] * len(molist))), total=len(molist)))
        
            # print('Writing to lmdb')

            num_skipped = 0
            with db.begin(write=True, buffers=True) as txn:
                for i, res in enumerate(results):
                    if res is None:
                        num_skipped += 1
                        continue
                    try:
                        txn.put(
                            key=str(index).encode(),
                            value=res
                        )
                        index += 1
                        if index % 10000 == 0:
                            print(f'Processed {index} molecules (skipped {num_skipped}). Current sdf: {sdf_supplier}')
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

def calculate_mol_props(smiles_list):
    """
    Calculate the number of unique scaffolds in a given dataset.
    Args:
        smiles_list: List of SMILES strings representing molecules.
    Returns:
        A set of unique scaffolds and their count.
    """
    scores = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)  # Convert SMILES to RDKit Mol object
            if mol:
                props = get_chem(mol)
                props['n_atoms'] = mol.GetNumAtoms()
                scores.append(props)
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
    
    return scores

def calculate_dataset_stats(dataset, pose_split):
    dataset_smiles = {}
    name = f'ZINC1M_{dataset.version}'
    for key in pose_split.keys():
        # get smiles for each molecule
        # plot the QED SA distribution
        dataset_smiles[key] = []
        for i in tqdm(pose_split[key], desc=f'Processing {key}'):
            data = dataset[i]
            dataset_smiles[key].append(data.ligand_smiles)

        smiles = dataset_smiles[key]
        freq = Counter(smiles)

        print(name, key, len(freq), 'smiles', 'in', len(pose_split[key]), 'molecules')

        # _, scaffold_count = calculate_unique_scaffolds(freq.keys())
        # print(name, key, scaffold_count, 'scaffolds', len(freq), 'smiles')
        mol_props = calculate_mol_props(freq.keys())
        qed_list = [m['qed'] for m in mol_props]
        sa_list = [m['sa'] for m in mol_props]
        print(key, 'QED', np.mean(qed_list), 'SA', np.mean(sa_list))
        sa_list = np.array(sa_list)
        print('SA (w/o < 0.4)', sa_list[sa_list >= 0.4].mean(), '(w/o < 0.3)', sa_list[sa_list >= 0.3].mean())

        plt.hist(qed_list, bins=50, log=True, color='blue', alpha=0.7)
        plt.title(f"Ligand Property Distribution ({name} {key})")
        plt.xlabel("QED")
        plt.ylabel("Count")
        plt.savefig(f"qed_freq_{name}_{key}.png", dpi=300, bbox_inches='tight')
        plt.clf()

        plt.hist(sa_list, bins=50, log=True, color='blue', alpha=0.7)
        plt.title(f"Ligand Property Distribution ({name} {key})")
        plt.xlabel("SA")
        plt.ylabel("Count")
        plt.savefig(f"sa_freq_{name}_{key}.png", dpi=300, bbox_inches='tight')
        plt.clf()

        n_atoms = [m['n_atoms'] for m in mol_props]
        plt.hist(n_atoms, bins=50, log=True, color='blue', alpha=0.7)
        plt.title(f"Ligand Property Distribution ({name} {key})")
        plt.xlabel("Number of Atoms")
        plt.ylabel("Count")
        plt.savefig(f"n_atoms_freq_{name}_{key}.png", dpi=300, bbox_inches='tight')
        plt.clf()
    
    pickle.dump(dataset_smiles, open(f'/mnt/data/{name}_smiles.pkl', 'wb'))


if __name__ == '__main__':
 
    ############################################################
    
    # test MoleculeDataset
    # print('Testing MoleculeDataset')
    # dataset = MoleculeDataset('/sharefs/share/sbdd_data/zinc', version='kekulize')
    # print('Loaded dataset')
    # print(len(dataset), dataset[0])

    # #  random sample 20M molecules (2M)
    # allowed_elements = [1, 6, 7, 8, 9, 15, 16, 17]
    # import random
    # i_list = random.sample(range(len(dataset)), 2000000)
    # id_filtered = []
    # for i in tqdm(i_list, total=len(i_list), desc='Filtering'):
    #     try:
    #         data = dataset[i]
    #         flag = False
    #         for element in data.ligand_element:
    #             if element not in allowed_elements:
    #                 flag = True
    #                 break
    #         if flag: 
    #             continue
    #         id_filtered.append(i)
    #     except:
    #         continue

    # print(f'Filtered {len(id_filtered)} molecules')

    # # select 10M molecules for training (1M), 1K for testing
    # random.shuffle(id_filtered)
    # split = {}
    # split['train'] = id_filtered[:1000000]
    # split['test'] = id_filtered[1000000:1001000]

    # # save split
    # print('Saving split', len(split['train']), len(split['test']))
    # torch.save(split, '/mnt/data/zinc_pose_split_1m_kekulize.pt')

    # calculate dataset stats
    # split = torch.load('/mnt/data/zinc_pose_split_1m_kekulize.pt')
    dataset = MoleculeDataset('/sharefs/share/sbdd_data/posebusters_benchmark_set_pocket10')
    print(len(dataset), dataset[0])
    split = torch.load('/sharefs/share/sbdd_data/posebusters_benchmark_184_pose_split.pt')
    calculate_dataset_stats(dataset, split)