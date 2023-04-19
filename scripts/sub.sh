#!/bin/bash

module load anaconda/2021.05
module load cuda/11.1
source activate diffuseq

export PYTHONUNBUFFERED=1


#python eval_seq2seq.py --folder "../generation_outputs/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_test-qqp20221122-18:34:11/ema_0.9999_050000.pt.samples" --mbr
bash train.sh
#python eval_seq2seq.py --folder ../generation_outputs/diffuseq_ubuntu_h128_lr0.0001_t2000_sqrt_lossaware_seed102_ubuntu20230131-11:24:11/ema_0.9999_050000.pt.samples --mbr
#bash run_decode.sh
#python -m torch.distributed.launch --nproc_per_node=4 --master_port=14233 --use_env run_train.py --diff_steps 2000 --lr 0.0001 --learning_steps 50000 --save_interval 10000 --seed 102 --noise_schedule sqrt --hidden_dim 128 --bsz 2048 --dataset ubuntu --data_dir datasets/ubuntu --vocab not_bert --seq_len 128 --schedule_sampler lossaware --notes ubuntu

#python -m torch.distributed.launch --nproc_per_node=4 --master_port=14233 --use_env run_train.py --diff_steps 2000 --lr 0.0001 --learning_steps 50000 --save_interval 10000 --seed 102 --noise_schedule sqrt --hidden_dim 128 --bsz 2048 --dataset ubuntu --data_dir datasets/ubuntu --vocab bert --seq_len 128 --schedule_sampler lossaware --notes ubuntu
#python -u run_decode_gcn_layort.py --model_dir diffusion_models/diffuseq_ubuntu_h128_lr0.0001_t2000_sqrt_lossaware_seed102_ubuntu20230216-10_32_51 --split test --step 1000 --seed 1229 --bsz 20

#python -m torch.distributed.launch --nproc_per_node=1 \
#--master_port=12233 \
#--use_env run_gsn_train_layort.py \
#--model_dir diffusion_models/diffuseq_ubuntu_h128_lr0.0001_t2000_sqrt_lossaware_seed102_ubuntu20230216-10_32_51 \
#--split train \
#--step 100 \
#--seed 1229 \
#--bsz 2000 \
#--dataset ubuntu \
#--data_dir datasets/ubuntu \
#--microbatch 200 \
#--lr 0.01 \
#--alpha 0.2 \
#--use_norm True \
#--dropout 0.2 \
#--epoches 50 \
#--sub_seq_len 90 \
