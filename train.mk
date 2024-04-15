SHELL := /bin/bash

run:
	[[ -d data ]] || (set -e; ([[ -e crossdocked_pocket10_pose_split.pt ]] || 
	( gdown "https://drive.google.com/file/d/1WMo-i38KG1ZZSt6A5tTV7G7GzzbSGmXc/view?usp=share_link" --fuzzy)); mkdir -p data; mv crossdocked_pocket10_pose_split.pt data/;)
	[[ -d data ]] || (set -e; ([[ -e crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb ]] || 
	( gdown "https://drive.google.com/file/d/1jjJzc8Ur5igaYwRhWOJbdMxdbJCsA-YB/view?usp=share_link" --fuzzy)); mv crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb data/;)
	[[ -d data ]] || (set -e; ([[ -e test_set.zip ]] || 
	( gdown "https://drive.google.com/file/d/17jLiXNWY6EpfZhQgtGu_6reXJ1nuK439/view?usp=share_link" --fuzzy)); unzip test_set.zip -d data;)
	python train_bfn.py --config_file configs/default.yaml --epochs 15 --no_wandb
