#!/bin/bash
#SBATCH --job-name=e4e
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --partition=rtx6000,t4v2
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=50GB

name=$1
path=$2
dataset=$3
resolution=$4
model=${5:-'sgxl'}
face_dataset=${6:-0}

encoder_type=ResNetProgressiveBackboneEncoder
if [ $face_dataset -eq 1 ]
then
    encoder_type=ProgressiveBackboneEncoder
fi

python3 inversion/scripts/train_restyle_e4e.py \
--dataset_type ${dataset}_encode \
--encoder_type $encoder_type \
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
--output_size $resolution \
--resolution $resolution \
--stylegan_weights $path \
--save_interval 2000 \
--ckpt_dir /checkpoint/kaselby/$name \
--n_styles 18 \
--decoder_type $model \
--resume_training_from_ckpt /checkpoint/kaselby/$name/checkpoint.pt