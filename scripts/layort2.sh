#!/bin/bash
#SBATCH -J layort_2
#SBATCH -e layort_2.ero
#SBATCH -o layort_2.out
#SBATCH --gpus=1


module load anaconda/2021.05
module load cuda/11.3
module load gcc/11.2
source activate diffuseq

#bash run_decode.sh

#python -m torch.distributed.launch --nproc_per_node=4 --master_port=14233 --use_env run_train.py --diff_steps 2000 --lr 0.0001 --learning_steps 50000 --save_interval 10000 --seed 102 --noise_schedule sqrt --hidden_dim 128 --bsz 2048 --dataset ubuntu --data_dir datasets/ubuntu --vocab bert --seq_len 128 --schedule_sampler lossaware --notes ubuntu

#diffuseq_ubuntu_h128_lr0.0001_t2000_sqrt_lossaware_seed102_ubuntu20230216-10:32:51
python -u run_decode_layort_2.py --model_dir diffusion_models/diffuseq_ubuntu_h128_lr0.0001_t2000_sqrt_lossaware_seed102_ubuntu20230216-10_32_51 --split test --step 1000 --seed 12227

#flatbuffers
