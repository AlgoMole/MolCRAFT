from openbabel import pybel
from meeko import MoleculePreparation
from meeko import obutils
import subprocess
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import tempfile
import AutoDockTools
import os
import contextlib
# from posecheck import PoseCheck

from core.evaluation.docking_qvina import get_random_id, BaseDockingTask
from core.evaluation.docking_qvina import PrepLig, PrepProt, suppress_stdout

import os
from typing import Union

import datamol as dm
import numpy as np
from rdkit import Chem


def pdbqt_to_rdkit_mols(pdbqt_path):
    """Convert all molecules in a PDBQT file to a list of RDKit molecules."""

    # Set up Open Babel objects
    mol = ob.OBMol()
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("pdbqt", "pdb")

    # Ensure the file can be opened
    if not obConversion.OpenInAndOutFiles(pdbqt_path, ""):
        raise ValueError(f"Failed to read the PDBQT file: {pdbqt_path}")

    rdkit_mols = []
    # Loop over all molecules in the file
    not_end_of_file = True
    while not_end_of_file:
        not_end_of_file = obConversion.Read(mol)
        # Convert to PDB format
        pdb_block = obConversion.WriteString(mol)

        # Convert PDB block to RDKit molecule
        rdkit_mol = Chem.MolFromPDBBlock(pdb_block)

        if rdkit_mol is not None:
            rdkit_mols.append(rdkit_mol)

        # Clear the molecule object for the next iteration
        mol.Clear()

    if not rdkit_mols:
        raise ValueError("No molecules could be successfully converted.")

    # Remove the last molecule, which is always empty
    return rdkit_mols[:-1]


class SMINA(object):
    def __init__(
        self,
        executable='smina',
        cpu=None,
    ):
        """Wrapper class for smina docking software.

        Args:
            executable (str): Path to smina executable.
            cpu (int, optional): Number of CPU cores to use. If None, all available cores will be used (recommended).
        """
        self.executable = executable
        self.minimize = False
        self.score_only = False
        self.cpu = cpu

    def set_receptor(
        self,
        receptor_path: Union[str, os.PathLike],
        centre: Union[str, Chem.Mol, np.ndarray] = "pocket",
        size: int = BOX_SIZE,
    ) -> None:
        """
        Set the receptor for docking.

        Args:
            receptor_path (Union[str, os.PathLike]): Path to the receptor file.
            centre (Union[str, Chem.Mol, np.ndarray]): The centre of the docking box.
                If 'pocket', the centre is calculated from the PDBQT file.
                If Chem.Mol, the centre is calculated from the RDKit molecule.
                If np.ndarray, the centre is already specified as a numpy array.
                If str, the centre is read from an SDF file.
            size (int): The size of the docking box. Default is 25.

        Raises:
            ValueError: If the centre type is not one of the specified types.
        """

        self.receptor = receptor_path

        if centre == "pocket":
            # Get the receptor centre from the PDBQT file with rdkit
            receptor_mol = read_pdbqt_receptor(receptor_path)
            centre = get_centroid_ob(receptor_mol)
        elif type(centre) == Chem.Mol:
            # Get the receptor centre from the rdkit mol
            centre = get_centroid_rdmol(centre)
        elif type(centre) == np.ndarray:
            # Centre is already a numpy array
            pass
        elif type(centre) == str:  # TODO add support for PDBQT
            # get from SDF
            try:
                mol = dm.read_sdf(centre)[0]
                centre = get_centroid_rdmol(mol)
            except:
                raise ValueError(
                    f"Could not read centre from file {centre}. Make sure it is a valid SDF file."
                )
        else:
            raise ValueError(
                f'Invalid centre type {type(centre)}. Must be one of: "pocket", Chem.Mol, np.ndarray, or path to sdf.'
            )

    def set_grid(self, center: np.ndarray, box_size: int = BOX_SIZE) -> None:
        # Set the centre and size of the docking box
        self.centre_x = center[0]
        self.centre_y = center[1]
        self.centre_z = center[2]

        self.size_x = box_size
        self.size_y = box_size
        self.size_z = box_size

    def set_ligand_from_file(self, ligand_path):
        self.ligand = ligand_path

    # def set_ligand_from_mol(self, mol: Chem.Mol):
    #     self.task_id = np.random.randint(0, 1000000)
    #     self.ligand = f"tmp_{self.task_id}.pdbqt"

    #     rdkit_mol_to_pdbqt(mol, self.ligand)

    def clear_ligand(self):
        """Remove the ligand file. Important to do this after each docking run."""
        if os.path.exists(self.ligand):
            os.remove(self.ligand)

    def score(self):
        self.score_only = True
        mol, out = self.run()
        self.score_only = False

        out = parse_smina_output_score(out)

        if len(mol) == 1:
            out["mol"] = mol[0]
        else:
            out["mol"] = None

        return out

    def optimize(self):
        self.minimize = True
        mol, out = self.run()
        self.minimize = False

        out = parse_smina_output_minimize(out)

        if len(mol) == 1:
            out["mol"] = mol[0]
        else:
            out["mol"] = None

        return out

    def dock(self, exhaustiveness, n_poses):
        self.exhaustiveness = exhaustiveness
        self.n_poses = n_poses
        mols, out = self.run()
        out = parse_smina_output_docking(out)

        out["mols"] = mols

        return out

    # def calculate_all(self, mol=None):
    #     if mol is not None:
    #         self.set_ligand_from_mol(mol)

    #     try:
    #         score_only = self.score_pose()
    #         minimized = self.minimize_pose()
    #         redocked = self.redock()

    #         self.clear_ligand()

    #         return {
    #             "score_only": score_only,
    #             "minimized": minimized,
    #             "redocked": redocked,
    #         }

    #     except:
    #         self.clear_ligand()
    #         return {"score_only": None, "minimized": None, "redocked": None}

    def run(self):
        """Main function to run smina.
        Should be called from one of the other functions.
        """

        tmp_out = f"tmp/out_{self.task_id}.pdbqt"

        # Command over multiple lines for readability
        command = (
            f"{self.executable} -r {self.receptor} -l {self.ligand} "
            f"--center_x {self.centre_x} --center_y {self.centre_y} --center_z {self.centre_z} "
            f"--size_x {self.size_x} --size_y {self.size_y} --size_z {self.size_z} "
            f"-o {tmp_out} "
        )

        if not self.minimize and not self.score_only:
            command += f"--exhaustiveness 8 "
        else:
            command += f"--exhaustiveness {self.exhaustiveness} --num_modes {self.n_poses} "

        if self.minimize:
            command += "--minimize "
        if self.score_only:
            command += "--score_only "

        if self.cpu:
            command += f"--cpu {self.cpu} "  # only do if you want use less than the max number of cores on your machine

        # Perform docking/minimization
        # print(command)
        # os.system(command)
        out = os.popen(command).read()

        out_mols = pdbqt_to_rdkit_mols(tmp_out)
        os.remove(tmp_out)

        return out_mols, out


class SminaDock(object): 
    def __init__(self, lig_pdbqt, prot_pdbqt): 
        self.lig_pdbqt = lig_pdbqt
        self.prot_pdbqt = prot_pdbqt
    
    def _max_min_pdb(self, pdb, buffer):
        with open(pdb, 'r') as f: 
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HEATATM')]
            xs = [float(l[31:39]) for l in lines]
            ys = [float(l[39:47]) for l in lines]
            zs = [float(l[47:55]) for l in lines]
            print(max(xs), min(xs))
            print(max(ys), min(ys))
            print(max(zs), min(zs))
            pocket_center = [(max(xs) + min(xs))/2, (max(ys) + min(ys))/2, (max(zs) + min(zs))/2]
            box_size = [(max(xs) - min(xs)) + buffer, (max(ys) - min(ys)) + buffer, (max(zs) - min(zs)) + buffer]
            return pocket_center, box_size
    
    def get_box(self, ref=None, buffer=0):
        '''
        ref: reference pdb to define pocket. 
        buffer: buffer size to add 

        if ref is not None: 
            get the max and min on x, y, z axis in ref pdb and add buffer to each dimension 
        else: 
            use the entire protein to define pocket 
        '''
        if ref is None: 
            ref = self.prot_pdbqt
        self.pocket_center, self.box_size = self._max_min_pdb(ref, buffer)
        print(self.pocket_center, self.box_size)

    def dock(self, score_func='smina', seed=0, mode='dock', exhaustiveness=8, save_pose=False, **kwargs):  # seed=0 mean random seed
        v = SMINA(sf_name=score_func, seed=seed, verbosity=0, **kwargs)
        v.set_receptor(self.prot_pdbqt)
        v.set_ligand_from_file(self.lig_pdbqt)
        v.set_grid(center=self.pocket_center, box_size=self.box_size)
        if mode == 'score_only': 
            score = v.score()[0]
        elif mode == 'minimize':
            score = v.optimize()[0]
        elif mode == 'dock':
            v.dock(exhaustiveness=exhaustiveness, n_poses=1)
            score = v.energies(n_poses=1)[0][0]
        else:
            raise ValueError
        
        if not save_pose: 
            return score
        else: 
            if mode == 'score_only': 
                pose = None 
            elif mode == 'minimize': 
                tmp = tempfile.NamedTemporaryFile()
                with open(tmp.name, 'w') as f: 
                    v.write_pose(tmp.name, overwrite=True)             
                with open(tmp.name, 'r') as f: 
                    pose = f.read()
   
            elif mode == 'dock': 
                pose = v.poses(n_poses=1)
            else:
                raise ValueError
            return score, pose


class VinaDockingTask(BaseDockingTask):

    @classmethod
    def from_original_data(cls, data, ligand_root='./data/crossdocked_pocket10', protein_root='./data/crossdocked',
                           **kwargs):
        protein_fn = os.path.join(
            os.path.dirname(data.ligand_filename),
            os.path.basename(data.ligand_filename)[:10] + '.pdb'
        )
        protein_path = os.path.join(protein_root, protein_fn)

        ligand_path = os.path.join(ligand_root, data.ligand_filename)
        ligand_rdmol = next(iter(Chem.SDMolSupplier(ligand_path)))
        return cls(protein_path, ligand_rdmol, **kwargs)

    @classmethod
    def from_generated_mol(cls, ligand_rdmol, ligand_filename, protein_root='./data/crossdocked', **kwargs):
        # load original pdb
        protein_fn = os.path.join(
            os.path.dirname(ligand_filename),
            os.path.basename(ligand_filename)[:10] + '.pdb'  # PDBId_Chain_rec.pdb
        )
        protein_path = os.path.join(protein_root, protein_fn)
        return cls(protein_path, ligand_rdmol, **kwargs)

    def __init__(self, protein_path, ligand_rdmol, tmp_dir='./tmp', center=None,
                 size_factor=1., buffer=5.0, pos=None):
        super().__init__(protein_path, ligand_rdmol)
        # self.conda_env = conda_env
        self.tmp_dir = os.path.realpath(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        self.task_id = get_random_id()
        self.receptor_id = self.task_id + '_receptor'
        self.ligand_id = self.task_id + '_ligand'

        self.receptor_path = protein_path
        self.ligand_path = os.path.join(self.tmp_dir, self.ligand_id + '.sdf')

        self.recon_ligand_mol = ligand_rdmol
        ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)

        sdf_writer = Chem.SDWriter(self.ligand_path)
        sdf_writer.write(ligand_rdmol)
        sdf_writer.close()
        self.ligand_rdmol = ligand_rdmol

        if pos is None:
            raise ValueError('pos is None')
            pos = ligand_rdmol.GetConformer(0).GetPositions()
        if center is None:
            self.center = (pos.max(0) + pos.min(0)) / 2
        else:
            self.center = center

        if size_factor is None:
            self.size_x, self.size_y, self.size_z = 20, 20, 20
        else:
            self.size_x, self.size_y, self.size_z = (pos.max(0) - pos.min(0)) * size_factor + buffer

        self.proc = None
        self.results = None
        self.output = None
        self.error_output = None
        self.docked_sdf_path = None
        # self.pc = PoseCheck()

    def run(self, mode='dock', exhaustiveness=8, **kwargs):
        ligand_pdbqt = self.ligand_path[:-4] + '.pdbqt'
        protein_pqr = self.receptor_path[:-4] + '.pqr'
        protein_pdbqt = self.receptor_path[:-4] + '.pdbqt'

        lig = PrepLig(self.ligand_path, 'sdf')
        lig.get_pdbqt(ligand_pdbqt)

        prot = PrepProt(self.receptor_path)
        if not os.path.exists(protein_pqr):
            prot.addH(protein_pqr)
        if not os.path.exists(protein_pdbqt):
            prot.get_pdbqt(protein_pdbqt)

        dock = SminaDock(ligand_pdbqt, protein_pdbqt)
        dock.pocket_center, dock.box_size = self.center, [self.size_x, self.size_y, self.size_z]
        score, pose = dock.dock(score_func='smina', mode=mode, exhaustiveness=exhaustiveness, save_pose=True, **kwargs)
        return [{'affinity': score, 'pose': pose}]

    # def run_pose_check(self):
    #     self.pc.load_protein_from_pdb(self.receptor_path)
    #     # self.pc.load_ligands_from_sdf(self.ligand_path)
    #     self.pc.load_ligands_from_mols([self.ligand_rdmol])
    #     clashes = self.pc.calculate_clashes()
    #     strain = self.pc.calculate_strain_energy()
    #     return {'clashes': clashes[0], 'strain': strain[0]}


# if __name__ == '__main__':
#     lig_pdbqt = 'data/lig.pdbqt'
#     mol_file = 'data/1a4k_ligand.sdf'
#     a = PrepLig(mol_file, 'sdf')
#     # mol_file = 'CC(=C)C(=O)OCCN(C)C'
#     # a = PrepLig(mol_file, 'smi')
#     a.addH()
#     a.gen_conf()
#     a.get_pdbqt(lig_pdbqt)
#
#     prot_file = 'data/1a4k_protein_chainAB.pdb'
#     prot_dry = 'data/protein_dry.pdb'
#     prot_pqr = 'data/protein.pqr'
#     prot_pdbqt = 'data/protein.pdbqt'
#     b = PrepProt(prot_file)
#     b.del_water(prot_dry)
#     b.addH(prot_pqr)
#     b.get_pdbqt(prot_pdbqt)
#
#     dock = VinaDock(lig_pdbqt, prot_pdbqt)
#     dock.get_box()
#     dock.dock()
    

