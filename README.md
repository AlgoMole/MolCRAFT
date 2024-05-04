# MolCRAFT

Official implementation of ["MolCRAFT: Structure-Based Drug Design in Continuous Parameter Space"](https://arxiv.org/abs/2404.12141).


## Environment

### Prerequisite
You will need a host machine with gpu, and a docker with `nvidia-container-runtime` enabled.

> [!TIP]
> - This repo provide an easy-to-use script to install docker and nvidia-container-runtime, in `./docker` run `sudo ./setup_docker_for_host.sh` to setup your host machine.
> - You can also refer to [install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) if you don't have them installed.


### Install via Docker
```bash
cd ./docker
make
```

> [!NOTE]
> - The `make` will automatically build the docker image and run the container, inside which your host home directory will be mounted to `${HOME}/home`. The docker image supports running any command inside a tmux session.
> - If `make` exits with error messages with pip, try `make` again in case of any network connection timeout. 
> - Once `make` succeeds, note that exiting from container wound't stop it, nor the tmux sessions running within. To enter an existing container, simply run `make` again. If you need to stop this container, run `make kill`.
> - For customized environment setup, please refer to files `docker/Dockerfile`, `docker/asset/requirements.txt` and `docker/asset/apt_packages.txt`. 


### Install via Conda

You can also build your own environment through `conda env create -f environment.yml`. Here we list the main packages we used, in case you want a different version:

```bash
conda create -n bfnsbdd
conda activate bfnsbdd
conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
conda install lightning -c conda-forge
conda install pyg -c pyg
conda install rdkit openbabel pyyaml easydict python-lmdb -c conda-forge
```

For evaluation, you will need to install `vina` (affinity), `posecheck` (clash, strain energy, and key interactions), and `spyrmsd` (rmsd).

```bash
# for vina docking
pip install meeko==0.1.dev3 scipy pdb2pqr vina==1.2.2 
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3

# for posecheck
git clone https://github.com/cch1999/posecheck.git
cd posecheck
pip install -e .
pip install -r requirements.txt
conda install -c mx reduce

# for spyrmsd
conda install spyrmsd -c conda-forge
```

> [!NOTE]
> - If you encounter vina fail, please check `/opt/conda/lib/python3.9/site-packages/vina/vina.py`, line 260, change to `astype(np.int64)`
> - We find the latest version of [PoseCheck](https://github.com/cch1999/posecheck) contains some bugs. We encourage you to clone PoseCheck code and install by pip (as suggested by their official README). Besides, install commit `57a1938` will reproduce our results.
> - To resolve posecheck fail in loading protein,
> - For RMSD fail,

-----

## Folder Structure

- /checkpoints: The official checkpoint `last.ckpt`(43M) will be automatically cloned here.
- /configs: We use yaml file to manage configs for model, directory, data, train, and evaluation. Some of the parameters are provided as input arguments (e.g., `--test_only --no_wandb`), and will be automatically updated and converted to a `config` object.
- /core: The main code directory.
  - /callbacks: pytorch-lightning callbacks for validation, docking, etc.
  - /evaluation: Basic functions for evaluating conformation, affinity, etc.
  - /models:
    - bfn_base.py: BFN base class, implemented with Bayesian update and various loss.
    - bfn4sbdd.py: The score model for SBDD, implemented with SBDD output, loss, sampling.
    - sbdd_train_loop.py: A `LightningModule`, implemented with `training_step`, `validation_step`, `test_step` and `sampling_step`.
    - uni_transformer.py: backbone network, same as TargetDiff.
  - /utils: util functions for reconstructing molecule from point cloud, featurizing protein-ligand data, etc.
- /logs: The default output folder, containing checkpoints, generated molecules, evaluation results, etc.
- /test: Evaluation code.
- sample_for_pocket.py, scripts.mk, train_bfn.py: entry scripts.

-----
## Data
Data used for training / evaluating the model should be put in the `data` folder by default, and accessible in the [data](https://drive.google.com/drive/folders/16KiwfMGUIk4a6mNU20GnUd0ah-mjNlhC?usp=share_link) Google Drive folder.

To train the model from scratch, download the lmdb file and split file into data folder:
* `crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb`
* `crossdocked_pocket10_pose_split.pt`

To evaluate the model on the test set, download _and_ unzip the `test_set.zip` into data folder. It includes the original PDB files that will be used in Vina Docking.

By default, We transform the lmdb further into the featurized dataset as `crossdocked_v1.1_rmsd1.0_pocket10_add_aromatic_transformed_simple.pt` as described in `transform.py`, which might take several minutes. To enable accelerated training, the yaml file will be set as follows:

```yaml
data:
  name: pl_tr # [pl, pl_tr] where tr means transformed
```

---
## Training
Run `make -f scripts.mk` (without the need for data preparation), or alternatively (with data folder correctly configured),
```bash
python train_bfn.py --exp_name {EXP} --revision {REVISION}
```

where the default values should be set the same as:
```bash
python train_bfn.py --sigma1_coord 0.03 --beta1 1.5 --lr 5e-4 --time_emb_dim 1 --epochs 15 --max_grad_norm Q --destination_prediction True --use_discrete_t True --num_samples 10 --sampling_strategy end_back_pmf
```

### Debugging
For quick debugging training process, run `make debug -f scripts.mk`:
```bash
python train_bfn.py --no_wandb --debug --epochs 1
```

## Sampling
We provide the pretrained checkpoint as [last.ckpt](https://drive.google.com/file/d/1a1laBFYRNqaMpcS3Id0L0R6XoLEk4gDG/view?usp=share_link). 
### Sampling for pockets in the testset
Run `make evaluate -f scripts.mk`, or alternatively,
```bash
python train_bfn.py --config_file configs/test.yaml --exp_name {EXP} --revision {REVISION} --test_only --num_samples {NUM_MOLS_PER_POCKET} --sample_steps 100
```

The output molecules will be saved in `${HOME}/home/logs/{exp_name}/{revision}/test
_outputs/mols`. If the sampling is run multiple times, there will be several `mols-v{version_number}` folders.

### Sampling from pdb file
To sample from a whole protein pdb file, we need the corresponding reference ligand to clip the protein pocket (a 10A region around the reference ligand).

```bash
python sample_for_pocket.py --config_file configs/test.yaml --protein_path {PDB_ID}_protein.pdb --ligand_path {PDB_ID}_molecule.sdf
```

## Evaluation
### Evaluating molecules
For binding affinity (Vina Score / Min / Dock) and molecular properties (QED, SA), it is calculated upon sampling.

For PoseCheck (strain energy, clashes) and other conformational results (bond length, bond angle, torsion angle, RMSD), please refer to `test` folder.

### Evaluating meta files
We provide samples for all SBDD baselines in the [sample](https://drive.google.com/drive/folders/1A3Mthm9ksbfUnMCe5T2noGsiEV1RfChH?usp=sharing) Google Drive folder.

You may download the `all_samples.tar.gz` and then `tar xzvf all_samples.tar.gz`, which extracts all the pt files into `samples` folder for evaluation.
