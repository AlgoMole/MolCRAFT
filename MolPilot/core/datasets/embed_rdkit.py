from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom
import multiprocessing as mp
from tqdm import tqdm
import os

def embed_and_optimize_single(smiles, max_iters=200):
    """
    Generate a single 3D conformation and optimize it using force field methods.
    
    Parameters:
    - smiles: SMILES string of the molecule.
    - max_iters: Maximum number of iterations for the force field optimization.
    
    Returns:
    - optimized_mol: The optimized molecule with the lowest energy conformation.
    - energy: The energy of the optimized conformation.
    """
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  # Add hydrogens
    
    # Generate a single conformer
    params = rdDistGeom.ETKDGv3()
    conf_id = AllChem.EmbedMolecule(mol, params=params)
    if conf_id == -1:
        raise ValueError(f"Conformer generation failed for {smiles}")
    
    # Optimize the conformer using UFF
    AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
    return mol

def process_molecule(smiles, idx):
    try:
        if os.path.exists(f'./data/ligands/unimol_{idx}.sdf'):
            mol = Chem.SDMolSupplier(f'./data/ligands/unimol_{idx}.sdf')[0]
            return mol
        mol = embed_and_optimize_single(smiles)
        writer = Chem.SDWriter(f'./data/ligands/unimol_{idx}.sdf')
        writer.write(mol)
        return mol
    except Exception as e:
        print(f"Failed to process {smiles}: {e}")
        return None

def main(smiles_list, max_iters=200):
    """
    Process a list of SMILES strings in parallel to generate and optimize a single conformer.
    
    Parameters:
    - smiles_list: List of SMILES strings.
    - max_iters: Maximum number of iterations for the force field optimization.
    
    Returns:
    - results: Dictionary with SMILES as keys and tuple of optimized molecule and energy as values.
    """
    # Use multiprocessing to process molecules in parallel
    # with a progressbar to track progress
    id_list = list(range(len(smiles_list)))
    with mp.Pool(mp.cpu_count()) as pool:
        results = list(tqdm(pool.starmap(process_molecule, zip(smiles_list, id_list)), total=len(smiles_list)))

    # Convert the list of results to a dictionary
    results_dict = {smiles: result for smiles, result in zip(smiles_list, results) if result is not None}
    
    return results_dict

if __name__ == "__main__":
    # Example usage
    import pandas as pd

    # csv_path = './data/250k_rndm_zinc_drugs_clean_3.csv'
    # df = pd.read_csv(csv_path)
    # smiles_list = df['smiles']  # Replace with your own list of SMILES

    csv_path = './data/ligands/clean_smi.csv'
    df = pd.read_csv(csv_path, header=None)
    smiles_list = df[0].tolist()
    print(f'Number of molecules: {len(smiles_list)}')

    results = main(smiles_list)

    # Write molecules into a single sdf
    # Note: This will write the molecules with the energy as a property
    # writer = Chem.SDWriter('./data/zinc250k.sdf')
    writer = Chem.SDWriter('./data/ligands/unimol.sdf')
    for smiles, mol in results.items():
        writer.write(mol)
