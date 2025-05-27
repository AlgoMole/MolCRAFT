
for quick debug

python train_bfn.py --revision debug --no_wandb --debug --epochs 1

for sample & evaluation

python train_bfn.py --revision ep10 --test_only --num_samples 5 --sample_steps 100
