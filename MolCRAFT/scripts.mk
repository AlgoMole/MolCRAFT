SHELL := /bin/bash

run: data
	python train_bfn.py --config_file configs/default.yaml --epochs 15 --no_wandb

data:
	[[ -e data/crossdocked_pocket10_pose_split.pt ]] || (set -e; ([[ -e crossdocked_pocket10_pose_split.pt ]] || ( gdown "https://drive.google.com/file/d/1WMo-i38KG1ZZSt6A5tTV7G7GzzbSGmXc/view?usp=share_link" --fuzzy)); mkdir -p data; mv crossdocked_pocket10_pose_split.pt data/;)
	[[ -e data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb ]] || (set -e; ([[ -e crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb ]] || ( gdown "https://drive.google.com/file/d/1jjJzc8Ur5igaYwRhWOJbdMxdbJCsA-YB/view?usp=share_link" --fuzzy)); mv crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb data/;)
	[[ -d data/test_set ]] || (set -e; ([[ -e test_set.zip ]] || ( gdown "https://drive.google.com/file/d/17jLiXNWY6EpfZhQgtGu_6reXJ1nuK439/view?usp=share_link" --fuzzy)); unzip test_set.zip -d data; rm test_set.zip)

checkpoint:
	[[ -e checkpoints/last.ckpt ]] ||  (set -e; ([[ -e last.ckpt ]] || ( gdown "https://drive.google.com/file/d/1a1laBFYRNqaMpcS3Id0L0R6XoLEk4gDG/view?usp=share_link" --fuzzy)); mkdir -p checkpoints; mv last.ckpt checkpoints/;)

debug: data
	python train_bfn.py --config_file configs/default.yaml --debug --epochs 1 --no_wandb

evaluate: data checkpoint
	python train_bfn.py --config_file configs/default.yaml --test_only --num_samples 10 --sample_steps 100 --no_wandb --ckpt_path ./checkpoints/last.ckpt
