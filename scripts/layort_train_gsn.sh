#!/bin/bash
#SBATCH -J layort_gsn_train
#SBATCH -e layort_train.ero
#SBATCH -o layort_train.out
#SBATCH --gpus=1


module load anaconda/2021.05
module load cuda/11.1
source activate diffuseq


python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=12233 \
--use_env run_gsn_train_layort.py \
--model_dir diffusion_models/diffuseq_ubuntu_h128_lr0.0001_t2000_sqrt_lossaware_seed102_ubuntu20230216-10_32_51 \
--split train \
--step 100 \
--seed 1229 \
--bsz 2000 \
--microbatch 100 \
--lr 0.01 \
--alpha 0.2 \
--use_norm True \
--dropout 0.2 \
--epoches 50 \
--sub_seq_len 90 \
