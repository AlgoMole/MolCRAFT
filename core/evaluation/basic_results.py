import numpy as np


class BasicResults:
    def __init__(self, name, full_name, results: list[dict]=None):
        self.name = name
        self.full_name = full_name
        self.results = results
    
    def __len__(self):
        return len(self.results)
    
    def __getitem__(self, idx):
        return self.results[idx]

    @property
    def smiles_list(self):
        return np.array([x['smiles'] for x in self.results])
    
    @property
    def complete_list(self):
        return np.array([x['complete'] for x in self.results])

    @property
    def validity_list(self):
        return np.array([x['validity'] for x in self.results])

    @property
    def center_change_list(self):
        return np.array([x['center_change'] for x in self.results])
    
    @property
    def mol_pos_range_list(self):
        return np.array([x['mol_pos_range'] for x in self.results])

    @property
    def atom_num_list(self):
        return np.array([x['mol'].GetNumAtoms() for x in self.results])

    @property
    def qed_list(self):
        return np.array([(x['chem_results']['qed'] if 'chem_results' in x else np.nan) for x in self.results])
    
    @property
    def sa_list(self):
        return np.array([(x['chem_results']['sa'] if 'chem_results' in x else np.nan) for x in self.results])
    
    @property
    def logp_list(self):
        return np.array([(x['chem_results']['logp'] if 'chem_results' in x else np.nan) for x in self.results])
    
    @property
    def lipinski_list(self):
        return np.array([(x['chem_results']['lipinski'] if 'chem_results' in x else np.nan) for x in self.results])

    @property
    def atom_num_list(self):
        return np.array([(x['chem_results']['atom_num'] if 'chem_results' in x else np.nan) for x in self.results])

    @property
    def vina_score_list(self):
        return np.array([(x['vina']['score_only'][0]['affinity'] if 'vina' in x else np.nan) for x in self.results])
    
    @property
    def vina_min_list(self):
        return np.array([(x['vina']['minimize'][0]['affinity'] if 'vina' in x else np.nan) for x in self.results])
    
    @property
    def vina_dock_list(self):
        return np.array([(x['vina']['dock'][0]['affinity'] if 'vina' in x else np.nan) for x in self.results])

    @property
    def strain_list(self):
        return np.array([(x['pose_check']['strain'] if 'pose_check' in x else np.nan) for x in self.results])

    @property
    def clash_list(self):
        return np.array([(x['pose_check']['clash'] if 'pose_check' in x else np.nan) for x in self.results])

