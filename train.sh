#!/bin/bash
#SBATCH --job-name=e4e
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --partition=a40
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=50GB

name=$1

python3 inversion/scripts/train_restyle_e4e.py \
--dataset_type ffhq_encode \
--encoder_type ProgressiveBackboneEncoder \
--exp_dir experiments/$name \
--batch_size 2 \
--test_batch_size 2 \
--workers 8 \
--test_workers 8 \
--start_from_latent_avg True \
--lpips_lambda 0.8 \
--l2_lambda 1 \
--id_lambda 0.1 \
--w_discriminator_lambda 0.1 \
--use_w_pool True \
--input_nc 6 \
--n_iters_per_batch 3 \
--truncation_psi 0.7 \
--output_size 256 \
--stylegan_weights ../stylegan-xl/pretrained_models/ffhq256.pkl \
--save_interval 2000 \
--ckpt_dir /checkpoint/kaselby/$name \
--n_styles 18