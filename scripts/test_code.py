import os, sys, glob

model_dir = '../diffusion_models/diffuseq_ubuntu_h128_lr0.0001_t2000_sqrt_lossaware_seed102_ubuntu20230131-11:24:11'

for lst in glob.glob(model_dir):
    print(lst)
    checkpoints = sorted(glob.glob(f"{lst}/ema*.pt"))[::-1]

print(checkpoints[0])