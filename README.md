# MolCRAFT

Official implementation of "MolCRAFT: Structure-Based Drug Design in Continuous Parameter Space".


## Environment

### Prerequisite
You will need to have a host machine with gpu, and have a docker with `nvidia-container-runtime` enabled.

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


-----
## Data
Data used for training / evaluating the model should be put in the `data` folder by default, and accessible in the [data](https://drive.google.com/drive/folders/16KiwfMGUIk4a6mNU20GnUd0ah-mjNlhC?usp=share_link) Google Drive folder.

To train the model from scratch, download the lmdb file and split file into data folder:
* `crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb`
* `crossdocked_pocket10_pose_split.pt`

To evaluate the model on the test set, download _and_ unzip the `test_set.zip` into data folder. It includes the original PDB files that will be used in Vina Docking.

We also provide the script for preprocessing the featurization for the lmdb data in `transform.py`. To enable accelerated training, set the yaml file as follows:

```yaml
data:
  name: pl_tr # [pl, pl_tr] where tr means transformed
```

---
## Training
```bash
python train_bfn.py --exp_name {EXP} --revision {REVISION}
```

where the default values should be set the same as:
```bash
python train_bfn.py --sigma1_coord 0.03 --beta1 1.5 --lr 5e-4 --time_emb_dim 1 --epochs 15 --max_grad_norm Q --destination_prediction True --use_discrete_t True --num_samples 10 --sampling_strategy end_back_pmf
```

### Debugging
For quick debugging training process:
```bash
python train_bfn.py --no_wandb --debug --epochs 1
```

## Sampling
We provide the pretrained checkpoint as [pretrained.ckpt](https://drive.google.com/file/d/1a1laBFYRNqaMpcS3Id0L0R6XoLEk4gDG/view?usp=share_link). 
### Sampling for pockets in the testset
```bash
python train_bfn.py --config_file configs/test.yaml --exp_name {EXP} --revision {REVISION} --test_only --num_samples {NUM_MOLS_PER_POCKET} --sample_steps 100
```

### Sampling from pdb file
To sample from a whole protein pdb file, we need the corresponding reference ligand to clip the protein pocket (a 10A region around the reference ligand):
```bash
python sample_for_pocket.py --config_file configs/test.yaml --protein_path {PDB_ID}_protein.pdb --ligand_path {PDB_ID}_molecule.sdf
```

## Evaluation
### Evaluating molecules
For binding affinity (Vina Score / Min / Dock) and molecular properties (QED, SA), it is calculated upon sampling.

For PoseCheck (strain energy, clashes) and other conformational results (bond length, bond angle, torsion angle, RMSD), please refer to `test` folder.

### Evaluating meta files
We provide samples for all SBDD baselines in the [sample](https://drive.google.com/drive/folders/16KiwfMGUIk4a6mNU20GnUd0ah-mjNlhC?usp=share_link) Google Drive folder. 