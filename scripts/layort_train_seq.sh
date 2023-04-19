#!/bin/bash
#SBATCH -J layort_train_seq
#SBATCH -e layort_train_seq.ero
#SBATCH -o layort_train_seq.out
#SBATCH --gpus=4


module load anaconda/2021.05
module load cuda/11.1
source activate diffuseq


bash train.sh