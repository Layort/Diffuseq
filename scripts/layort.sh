#!/bin/bash
#SBATCH -J layort_sample
#SBATCH -e layort_sample.ero
#SBATCH -o layort-sample.out
#SBATCH --gpus=1


module load anaconda/2021.05
module load cuda/11.1
source activate diffuseq

#bash run_decode.sh

#python -m torch.distributed.launch --nproc_per_node=4 --master_port=14233 --use_env run_train.py --diff_steps 2000 --lr 0.0001 --learning_steps 50000 --save_interval 10000 --seed 102 --noise_schedule sqrt --hidden_dim 128 --bsz 2048 --dataset ubuntu --data_dir datasets/ubuntu --vocab bert --seq_len 128 --schedule_sampler lossaware --notes ubuntu

#diffuseq_ubuntu_h128_lr0.0001_t2000_sqrt_lossaware_seed102_ubuntu20230216-10:32:51
python -u run_decode_gcn_layort.py --model_dir diffusion_models/diffuseq_ubuntu_h128_lr0.0001_t2000_sqrt_lossaware_seed102_ubuntu20230216-10_32_51 --split test --step 3 --seed 1229 --bsz 20

#flatbuffers

