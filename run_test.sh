#!/bin/bash

module load anaconda/2021.05
module load cuda/11.1
source activate diffuseq

export PYTHONUNBUFFERED=1


python test.py