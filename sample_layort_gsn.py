"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json
from tracemalloc import start

import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from diffuseq_layort.rounding import denoised_fn_round, get_weights
from diffuseq_layort.text_datasets_layort import load_data_text
from sample_utils_layort import GSN_matrix,langevin_fn_gsn,get_tgt_embeddings
from diffuseq_layort import MAX_UTTERANCE_NUM,MAX_UTTERANCE_TOKEN
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import time
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




def create_argparser():
    defaults = dict(model_path='', step=0, out_dir='', top_p=0, alpha = 0.02, use_norm = True, dropout = 0.2)
    decode_defaults = dict(split='valid', clamp_step=0, seed2=105, clip_denoised=False)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)


    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.to(dist_util.dev())
    model.eval()


    tokenizer = load_tokenizer(args)
    model_emb, tokenizer = load_model_emb(args, tokenizer)

    model_emb.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model_emb_copy = get_weights(model_emb, args)

    set_seed(args.seed2)
    #打包放进去
    need = [model ,tokenizer]

    print("### Sampling...on", args.split)

    ## load data
    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        sub_seq_len = args.sub_seq_len,
        deterministic=True,
        data_args=args,
        split=args.split,
        loaded_vocab=tokenizer,
        model_emb=model_emb.cpu(), # using the same embedding wight with tranining data
        loop=False
    )

    start_t = time.time()
    
    batch, cond = next(data_valid)

    
    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    out_dir = os.path.join(args.out_dir, f"{model_base_name.split('.ema')[0]}")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_path = os.path.join(out_dir, f"ema{model_base_name.split('.ema')[1]}.samples")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    out_path = os.path.join(out_path, f"seed{args.seed2}_step{args.clamp_step}.json")
    # fout = open(out_path, 'a')


    from tqdm import tqdm
    cuda = th.device(f"cuda:{os.environ['LOCAL_RANK']}")
    

    #定义GCN 更新函数
    gsn_model = GSN_matrix(args, args.batch_size, MAX_UTTERANCE_NUM, args.sub_seq_len, args.hidden_dim).to(cuda)

    langevin_fn_selected = partial(langevin_fn_gsn, gsn_model,need)

    for arr,cond in tqdm(data_valid):

        input_ids_x = cond.pop('input_ids').to(dist_util.dev())
        x_start = model.get_embeds(input_ids_x)
        input_ids_mask = cond.pop('input_mask')
        input_ids_mask_ori = input_ids_mask
        #print("cond['relation_at'][:2]",cond["relation_at"][:2])
        input_split_x_y_encode = model.get_embeds(cond['input_split_id_x'].to(dist_util.dev()))
        cond['input_split_x_y_encode'] = input_split_x_y_encode

        
        cond['pad_embed'] = model.get_embeds( th.tensor([tokenizer.pad_token_id]).to(cuda) )
        start = model.get_embeds( th.tensor([0]).to(cuda) )
        # print("cond['pad_embed']",cond['pad_embed'].shape)
        print("start",start)
        cond = get_tgt_embeddings(args.batch_size,MAX_UTTERANCE_NUM,cond)
        
        #构图（每个对话对应的图结构不同）
        noise = th.randn_like(x_start)
        input_ids_mask = th.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to(dist_util.dev())
        x_noised = th.where(input_ids_mask==0, x_start, noise)

        model_kwargs = {}

        if args.step == args.diffusion_steps:
            args.use_ddim = False
            step_gap = 1
        else:
            args.use_ddim = True
            step_gap = args.diffusion_steps//args.step
        # 在这里调用的p_sapme_loop
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        sample_shape = (batch.shape[0], args.seq_len, args.hidden_dim)

        samples = sample_fn(
            model,
            sample_shape,
            noise=x_noised,
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(denoised_fn_round, args, model_emb_copy.to(cuda)),
            model_kwargs=model_kwargs,
            top_p=args.top_p,
            clamp_step=args.clamp_step,
            clamp_first=True,
            mask=input_ids_mask,
            x_start=x_start,
            langevin_fn=langevin_fn_selected,
            gap=step_gap,
            group_lst = cond
        )

        model_emb_copy.cpu()
        # print(samples[0].shape) # samples for each step

        sample = samples[-1]
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_sentence = [sample.cpu().numpy() for sample in gathered_samples]

        # print('sampling takes {:.2f}s .....'.format(time.time() - start_t))

        word_lst_recover = []
        word_lst_ref = []
        word_lst_source = []


        arr = np.concatenate(all_sentence, axis=0)
        x_t = th.tensor(arr)
        # print('decoding for seq2seq', )
        # print(arr.shape)

        reshaped_x_t = x_t.to(cuda)
        # print("reshaped_x_t.shape",reshaped_x_t.shape)
        # exit(0)
        logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
        print("logits.shape",logits.shape)
        exit(0)
        cands = th.topk(logits, k=1, dim=-1)
        sample = cands.indices
        # tokenizer = load_tokenizer(args)

        for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
            len_x = args.seq_len - sum(input_mask).tolist()
            tokens = tokenizer.decode_token(seq[len_x:])
            word_lst_recover.append(tokens)

        for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
            # tokens = tokenizer.decode_token(seq)
            len_x = args.seq_len - sum(input_mask).tolist()
            word_lst_source.append(tokenizer.decode_token(seq[:len_x]))
            word_lst_ref.append(tokenizer.decode_token(seq[len_x:]))

        fout = open(out_path, 'a')
        for (recov, ref, src) in zip(word_lst_recover, word_lst_ref, word_lst_source):
            print(json.dumps({"recover": recov, "reference": ref, "source": src}), file=fout)
        fout.close()
        
    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {out_path}')




if __name__ == "__main__":
    main()
