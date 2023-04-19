import pdb
import time
import argparse
import os, json
import gc
import numpy as np
import torch as th
import torch.distributed as dist
from tqdm import tqdm
from tracemalloc import start
from transformers import set_seed
from diffuseq_layort import MAX_UTTERANCE_NUM
from diffuseq_layort.rounding import denoised_fn_round, get_weights
from diffuseq_layort.text_datasets_layort import load_data_text
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from diffuseq_layort.step_sample import create_named_schedule_sampler
from sample_utils_layort_copy import GSN_matrix,get_tgt_embeddings

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from train_gsn_util import GSN_train_loop, GetGPUInfo
from diffuseq_layort.utils import dist_util, logger
from functools import partial
from basic_utils_layort import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_model_emb,
    load_tokenizer
)

import wandb

def create_argparser():
    defaults = dict(model_path = '',sub_seq_len = 0,use_norm= True,
                top_p = 0, alpha = 0., epoches =0 ,step = 0
    )
    decode_defaults = dict(split='train', clamp_step=0, seed2=105, clip_denoised=False)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    # 设置args
    args = create_argparser().parse_args()
    dist_util.setup_dist()

    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")

    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    # print(training_args)
    #防止覆盖
    training_args['batch_size'] = args.batch_size
    training_args['microbatch'] = args.microbatch
    training_args['dropout']  = args.dropout
    training_args['lr']  = args.lr

    args.__dict__.update(training_args)
    #用于保存GSN模型 args.checkpoint_path

    print('\n')
    # print("args.gsn_checkpoint",args.gsn_checkpoint)
    logger.configure()
    print(training_args)

    #加载model 主要是获取embedding
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    # model.eval()

    # 拿到分词结果的embedding内容
    tokenizer = load_tokenizer(args)
    model_emb, tokenizer = load_model_emb(args, tokenizer)
    model_emb.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    # model_emb_copy = get_weights(model_emb, args)
    set_seed(args.seed2)


    with th.no_grad():#这里不需要梯度下降，使用该语法节省空间
        #加载训练集
        data_train = load_data_text(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            sub_seq_len = args.sub_seq_len,
            deterministic=True,
            data_args=args,
            split=args.split,
            loaded_vocab=tokenizer,
            model_emb=model_emb.cpu(),
            loop=True
        )


        #加载验证集
        data_valid = load_data_text(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            sub_seq_len = args.sub_seq_len,
            deterministic=True,
            data_args=args,
            split=args.split,
            loaded_vocab=tokenizer,
            model_emb=model_emb.cpu(), # using the same embedding wight with tranining data
            loop=True
        )

    #设置gpu 查看次数
    gpu_check_times = 1
    get_gpu = GetGPUInfo((0,1,2,3))
    # print("第%d次获取"%(gpu_check_times))
    # print(get_gpu())

    cuda = th.device(f"cuda:{os.environ['LOCAL_RANK']}")
    gsn_model = GSN_matrix(args, args.microbatch, MAX_UTTERANCE_NUM, args.sub_seq_len, args.hidden_dim).to(dist_util.dev())

    print("#"*30+"\nmodel loaded and begin to train\n")
    # print("该进程的cuda编号为：",os.environ['LOCAL_RANK'])
    # print(th.cuda.memory_summary(cuda))

    # gpu_check_times +=1
    # #第二次获取
    # print("第%d次获取"%(gpu_check_times))
    # print(get_gpu())


    #只取其中的word_embedding其他的不管
    word_embedding = model.word_embedding
    lm_head = model.lm_head
    logits_mode = model.logits_mode
    del model
    gc.collect()
    th.cuda.empty_cache()

    #第三次获取
    gpu_check_times += 1
    print("第%d次获取"%(gpu_check_times))
    print(get_gpu())

    pytorch_total_params = sum(p.numel() for p in gsn_model.parameters())
    print(f"### The parameter count is {pytorch_total_params}")
    # print("microbatch",args.microbatch)

    os.environ["WANDB_MODE"] = "offline"
    if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "DiffusionGCN"),
            name=args.checkpoint_path,
        )
        wandb.config.update(args.__dict__, allow_val_change=True)


    GSN_train_loop(
        model=gsn_model,
        diffusion=diffusion,
        data=data_train,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        weight_decay=args.weight_decay,
        learning_steps=args.learning_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_valid,
        eval_interval=args.eval_interval,
        word_embedding = word_embedding.to(cuda),
        lm_head = lm_head.to(cuda),
        logits_mode = logits_mode
    ).run_loop()

    

if __name__ == "__main__":
    main()
