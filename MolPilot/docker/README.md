## Environment setup

### Install via Docker
With a Linux Debian server, it is highly recommended to set up the environment via docker, since all you need to do is a simple `make` command.
```bash
cd ./docker
make
```

> [!NOTE]
> - The `make` will automatically build the docker image and run the container, inside which your host home directory will be mounted to `${HOME}/home`. The docker image supports running any command inside a tmux session.
> - If `make` exits with error messages with pip, try `make` again in case of any network connection timeout. 
> - Once `make` succeeds, note that exiting from container wound't stop it, nor the tmux sessions running within. To enter an existing container, simply run `make` again. If you need to stop this container, run `make kill`.
> - For customized environment setup, please refer to files `docker/Dockerfile`, `docker/asset/requirements.txt` and `docker/asset/apt_packages.txt`. 
> - If your bash does not highlight properly, try `sh -c "$(wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"` to reinstall zsh template.
> - PoseCheck should be manually installed. See instructions below.

### Install via Conda

You can also build your own environment through `conda env create -f environment.yml`. Here the main packages are listed:

| Package           | Version   |
|-------------------|-----------|
| CUDA              | 11.6      |
| NumPy             | 1.23.1    |
| Python            | 3.9       |
| PyTorch           | 1.12.0    |
| PyTorch Geometric | 2.1.0     |
| RDKit             | 2023.9.5  |

For evaluation, you will need to install `vina` (affinity), `posecheck` (clash, strain energy, and key interactions), and `spyrmsd` (rmsd).

```bash
# for vina docking
pip install meeko==0.1.dev3 scipy pdb2pqr vina==1.2.2 
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3

# for posecheck evaluation
git clone https://github.com/cch1999/posecheck.git
cd posecheck
git checkout 57a1938  # the calculation of strain energy used in our paper
pip install -e .
pip install -r requirements.txt
conda install -c mx reduce

# for spyrmsd
conda install spyrmsd -c conda-forge
```