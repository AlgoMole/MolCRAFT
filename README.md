# MolCRAFT
Official implementation of ICML 2024 ["MolCRAFT: Structure-Based Drug Design in Continuous Parameter Space"](https://arxiv.org/abs/2404.12141).

🎉 Our demo is now available at [120.240.170.153:10990](120.240.170.153:10990). The formal version will be at [http://gensi-thuair.com:10990/](http://gensi-thuair.com:10990/) soon. Welcome to have a try!

## Environment
It is highly recommended to install via docker if a Linux server with NVIDIA GPU is available.

Otherwise, you might check [README for env](docker/README.md) for further details of docker or conda setup.

### Prerequisite
A docker with `nvidia-container-runtime` enabled on your Linux system is required.

> [!TIP]
> - This repo provides an easy-to-use script to install docker and nvidia-container-runtime, in `./docker` run `sudo ./setup_docker_for_host.sh` to set up your host machine.
> - For details, please refer to the [install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).


### Install via Docker
We highly recommend you to set up the environment via docker, since all you need to do is a simple `make` command.
```bash
cd ./docker
make
```


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
  name: pl_tr # [pl, pl_tr] where tr means offline-transformed
```

---
## Training
Run `make -f scripts.mk` (without the need for data preparation), or alternatively (with data folder correctly configured),
```bash
python train_bfn.py --exp_name ${EXP_NAME} --revision ${REVISION}
```

where the default values should be set the same as:
```bash
python train_bfn.py --sigma1_coord 0.03 --beta1 1.5 --lr 5e-4 --time_emb_dim 1 --epochs 15 --max_grad_norm Q --destination_prediction True --use_discrete_t True --num_samples 10 --sampling_strategy end_back_pmf
```

### Testing
For quick evaluation of the official checkpoint, refer to `make evaluate` in `scripts.mk`:
```bash
python train_bfn.py --test_only --no_wandb --ckpt_path ./checkpoints/last.ckpt
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
python train_bfn.py --config_file configs/default.yaml --exp_name ${EXP_NAME} --revision ${REVISION} --test_only --num_samples ${NUM_MOLS_PER_POCKET} --sample_steps 100
```

The output molecules `vina_docked.pt` for all 100 test pockets will be saved in `./logs/${USER}_bfn_sbdd/${EXP_NAME}/${REVISION}/test_outputs/${TIMESTAMP}` folders.

### Sampling from pdb file
To sample from a whole protein pdb file, we need the corresponding reference ligand to clip the protein pocket (a 10A region around the reference position).

Below is an example that stores the generated 10 molecules under `output` folder. The configurations are managed in the ``call()`` function of ``sample_for_pocket.py``.

```bash
python sample_for_pocket.py ${PDB_PATH} ${SDF_PATH}
```

## Evaluation
### Evaluating molecules
For binding affinity (Vina Score / Min / Dock) and molecular properties (QED, SA), it is calculated upon sampling.

For PoseCheck (strain energy, clashes) and other conformational results (bond length, bond angle, torsion angle, RMSD), please refer to `test` folder.

### Evaluating meta files
We provide samples for all SBDD baselines in the [sample](https://drive.google.com/drive/folders/1A3Mthm9ksbfUnMCe5T2noGsiEV1RfChH?usp=sharing) Google Drive folder.

You may download the `all_samples.tar.gz` and then `tar xzvf all_samples.tar.gz`, which extracts all the pt files into `samples` folder for evaluation.

<!-- ## Demo
### Host our web app demo locally

With ``gradio`` and ``gradio_molecule3d`` installed, you can simply run ``python app.py`` to open the demo locally. Port mapping has been set in Makefile if you are using docker. You should also forward this port if you run the docker in an ssh server. We will share a permanent demo link later.

Great thanks to @duerrsimon for his kind support in resolving rendering issues! -->

## Citation

```
@article{qu2024molcraft,
  title={MolCRAFT: Structure-Based Drug Design in Continuous Parameter Space},
  author={Qu, Yanru and Qiu, Keyue and Song, Yuxuan and Gong, Jingjing and Han, Jiawei and Zheng, Mingyue and Zhou, Hao and Ma, Wei-Ying},
  journal={ICML 2024},
  year={2024}
}

@article{song2024unified,
  title={Unified Generative Modeling of 3D Molecules via Bayesian Flow Networks},
  author={Song, Yuxuan and Gong, Jingjing and Qu, Yanru and Zhou, Hao and Zheng, Mingyue and Liu, Jingjing and Ma, Wei-Ying},
  journal={ICLR 2024},
  year={2024}
}
```
