SHELL := /bin/bash

run:
	python train_bfn.py --config_file configs/default.yaml --epochs 15 --no_wandb
