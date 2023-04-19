import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import io

from diffuseq_layort.utils import dist_util, logger
from diffuseq_layort.utils.fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from diffuseq_layort.utils.nn import update_ema
from diffuseq_layort.step_sample import LossAwareSampler, UniformSampler


from train_utils import TrainLoop


class GSN_trian_loop(TrainLoop):
    def __init__(self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        learning_steps=0,
        checkpoint_path='',
        gradient_clipping=-1.,
        eval_data=None,
        eval_interval=-1,
        emb_model,
    ):
        # self.model = model
        # self.diffusion = diffusion
        # self.data = data
        # self.eval_data = eval_data
        # self.batch_size = batch_size
        # self.microbatch = microbatch if microbatch > 0 else batch_size
        # self.lr = lr
        # self.ema_rate = (
        #     [ema_rate]
        #     if isinstance(ema_rate, float)
        #     else [float(x) for x in ema_rate.split(",")]
        # )
        # self.log_interval = log_interval
        # self.eval_interval = eval_interval
        # self.save_interval = save_interval
        # self.resume_checkpoint = resume_checkpoint
        # self.use_fp16 = use_fp16
        # self.fp16_scale_growth = fp16_scale_growth
        # self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        # self.weight_decay = weight_decay
        # self.learning_steps = learning_steps
        # self.gradient_clipping = gradient_clipping

        # self.step = 0
        # self.resume_step = 0
        # self.global_batch = self.batch_size * dist.get_world_size()

        # self.model_params = list(self.model.parameters())
        # self.master_params = self.model_params
        # self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        # self.sync_cuda = th.cuda.is_available()

        # self.checkpoint_path = checkpoint_path # DEBUG **

        # self._load_and_sync_parameters()

        # if self.use_fp16:
        #     self._setup_fp16()

        # self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)

        # if self.resume_step:
        #     self._load_optimizer_state()
        #     frac_done = (self.step + self.resume_step) / self.learning_steps
        #     lr = self.lr * (1 - frac_done)
        #     self.opt = AdamW(self.master_params, lr=lr, weight_decay=self.weight_decay)
        #     # Model was resumed, either due to a restart or a checkpoint
        #     # being specified at the command line.
        #     self.ema_params = [
        #         self._load_ema_parameters(rate) for rate in self.ema_rate
        #     ]
        # else:
        #     self.ema_params = [
        #         copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
        #     ]

        # if th.cuda.is_available(): # DEBUG **
        #     self.use_ddp = True
        #     print(dist_util.dev())
        #     self.ddp_model = DDP(
        #         self.model,
        #         device_ids=[dist_util.dev()],
        #         output_device=dist_util.dev(),
        #         broadcast_buffers=False,
        #         bucket_cap_mb=128,
        #         find_unused_parameters=False,
        #     )
        # else:
        #     if dist.get_world_size() > 1:
        #         logger.warn(
        #             "Distributed training requires CUDA. "
        #             "Gradients will not be synchronized properly!"
        #         )
        #     self.use_ddp = False
        #     self.ddp_model = self.model
        super(GSN_trian_loop,self).__init__(*,
        model =model,
        diffusion = diffusion,
        data = data,
        batch_size=batch_size,
        microbatch=microbatch,
        lr=lr,
        ema_rate=ema_rate,
        log_interval=log_interval,
        save_interval=save_interval,
        resume_checkpoint=resume_checkpoint,
        use_fp16=use_fp16,
        fp16_scale_growth=fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=weight_decay,
        learning_steps=learning_steps,
        checkpoint_path=checkpoint_path,
        gradient_clipping=gradient_clipping,
        eval_data=eval_data,
        eval_interval=eval_interval)
        #新加入embed_model，这里没有使用ddp，感觉应该不需要，需要每一个子进程都需要一个完整的转换？
        self.emb_model = emb_model

    # def _load_and_sync_parameters(self):
    #     resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

    #     if resume_checkpoint[-3:] == '.pt':
    #         self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
    #         if dist.get_rank() == 0:
    #             logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
    #             self.model.load_state_dict(
    #                 dist_util.load_state_dict(
    #                     actual_model_path(resume_checkpoint), map_location=dist_util.dev()
    #                 )
    #             )

    #     dist_util.sync_params(self.model.parameters())

    # def _load_ema_parameters(self, rate):
    #     ema_params = copy.deepcopy(self.master_params)

    #     main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
    #     ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
    #     if ema_checkpoint:
    #         if dist.get_rank() == 0:
    #             logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
    #             state_dict = dist_util.load_state_dict(
    #                 actual_model_path(ema_checkpoint), map_location=dist_util.dev()
    #             )
    #             ema_params = self._state_dict_to_master_params(state_dict)

    #     dist_util.sync_params(ema_params)
    #     return ema_params

    # def _load_optimizer_state(self):
    #     main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
    #     if bf.exists(main_checkpoint):
    #         logger.log(f"loading optimizer state from checkpoint: {main_checkpoint}")
    #         state_dict = dist_util.load_state_dict(
    #             actual_model_path(main_checkpoint), map_location=dist_util.dev()
    #         )
    #         self.opt.load_state_dict(state_dict)

    # def _setup_fp16(self):
    #     self.master_params = make_master_params(self.model_params)
    #     self.model.convert_to_fp16()

    def run_loop(self):
        while (
            not self.learning_steps
            or self.step + self.resume_step < self.learning_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.eval_data is not None and self.step % self.eval_interval == 0:
                batch_eval, cond_eval = next(self.eval_data)
                self.forward_only(batch_eval, cond_eval)
                print('eval on validation set')
                logger.dumpkvs()
            if self.step > 0 and self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    # def run_step(self, batch, cond):
    #     self.forward_backward(batch, cond)
    #     if self.use_fp16:
    #         self.optimize_fp16()
    #     else:
    #         self.optimize_normal()
    #     self.log_step()

    def forward_only(self, batch, cond):
        with th.no_grad():
            zero_grad(self.model_params)
            for i in range(0, batch.shape[0], self.microbatch):
                micro = batch[i: i + self.microbatch].to(dist_util.dev())
                micro_cond = {
                    k: v[i: i + self.microbatch].to(dist_util.dev())
                    for k, v in cond.items()
                }
                last_batch = (i + self.microbatch) >= batch.shape[0]
                t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
                # print(micro_cond.keys())
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    micro,
                    t,
                    model_kwargs=micro_cond,
                )

                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

                log_loss_dict(
                    self.diffusion, t, {f"eval_{k}": v * weights for k, v in losses.items()}
                )


    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            # print(micro_cond.keys())
            
            #包装一个函数，目的是得到loss-》dict 里面有个key为“loss”
            K = 50 #?这个怎么设置比较好？
            compute_losses = functools.partial(
                gsn_compute_loss,
                self.ddp_model,
                micro_cond,
                K,
                self.emb_model
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    # def optimize_fp16(self):
    #     if any(not th.isfinite(p.grad).all() for p in self.model_params):
    #         self.lg_loss_scale -= 1
    #         logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
    #         return

    #     model_grads_to_master_grads(self.model_params, self.master_params)
    #     self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
    #     self._log_grad_norm()
    #     self._anneal_lr()
    #     self.opt.step()
    #     for rate, params in zip(self.ema_rate, self.ema_params):
    #         update_ema(params, self.master_params, rate=rate)
    #     master_params_to_model_params(self.model_params, self.master_params)
    #     self.lg_loss_scale += self.fp16_scale_growth

    # def grad_clip(self):
    #     # print('doing gradient clipping')
    #     max_grad_norm=self.gradient_clipping #3.0
    #     if hasattr(self.opt, "clip_grad_norm"):
    #         # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
    #         self.opt.clip_grad_norm(max_grad_norm)
    #     # else:
    #     #     assert False
    #     # elif hasattr(self.model, "clip_grad_norm_"):
    #     #     # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
    #     #     self.model.clip_grad_norm_(args.max_grad_norm)
    #     else:
    #         # Revert to normal clipping otherwise, handling Apex or full precision
    #         th.nn.utils.clip_grad_norm_(
    #             self.model.parameters(), #amp.master_params(self.opt) if self.use_apex else
    #             max_grad_norm,
    #         )

    # def optimize_normal(self):
    #     if self.gradient_clipping > 0:
    #         self.grad_clip()
    #     self._log_grad_norm()
    #     self._anneal_lr()
    #     self.opt.step()
    #     for rate, params in zip(self.ema_rate, self.ema_params):
    #         update_ema(params, self.master_params, rate=rate)

    # def _log_grad_norm(self):
    #     sqsum = 0.0
    #     # cnt = 0
    #     for p in self.master_params:
    #         # print(cnt, p) ## DEBUG
    #         # print(cnt, p.grad)
    #         # cnt += 1
    #         if p.grad != None:
    #             sqsum += (p.grad ** 2).sum().item()
    #     logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    # def _anneal_lr(self):
    #     if not self.learning_steps:
    #         return
    #     frac_done = (self.step + self.resume_step) / self.learning_steps
    #     lr = self.lr * (1 - frac_done)
    #     for param_group in self.opt.param_groups:
    #         param_group["lr"] = lr

    # def log_step(self):
    #     logger.logkv("step", self.step + self.resume_step)
    #     logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
    #     if self.use_fp16:
    #         logger.logkv("lg_loss_scale", self.lg_loss_scale)

    # def save(self):
    #     def save_checkpoint(rate, params):
    #         state_dict = self._master_params_to_state_dict(params)
    #         if dist.get_rank() == 0:
    #             logger.log(f"saving model {rate}...")
    #             if not rate:
    #                 filename = f"model{(self.step+self.resume_step):06d}.pt"
    #             else:
    #                 filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
    #             print('writing to', bf.join(get_blob_logdir(), filename))
    #             print('writing to', bf.join(self.checkpoint_path, filename))
    #             # with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
    #             #     th.save(state_dict, f)
    #             with bf.BlobFile(bf.join(self.checkpoint_path, filename), "wb") as f: # DEBUG **
    #                 th.save(state_dict, f) # save locally
    #                 # pass # save empty

    #     # save_checkpoint(0, self.master_params)
    #     for rate, params in zip(self.ema_rate, self.ema_params):
    #         save_checkpoint(rate, params)

    #     dist.barrier()

    # def _master_params_to_state_dict(self, master_params):
    #     if self.use_fp16:
    #         master_params = unflatten_master_params(
    #             list(self.model.parameters()), master_params # DEBUG **
    #         )
    #     state_dict = self.model.state_dict()
    #     for i, (name, _value) in enumerate(self.model.named_parameters()):
    #         assert name in state_dict
    #         state_dict[name] = master_params[i]
    #     return state_dict

    # def _state_dict_to_master_params(self, state_dict):
    #     params = [state_dict[name] for name, _ in self.model.named_parameters()]
    #     if self.use_fp16:
    #         return make_master_params(params)
    #     else:
    #         return params


def gsn_compute_loss(model_gsn,group_lst,K,model):

    loss = th.nn.CrossEntropyLoss()

    sample_y =  model.get_embeds(input_id_y)

    group_lst['sample'] =  sample_y

    #进行更新
    sample_y_update = model_gsn(group_lst,K)
    
    # #这里是把所有分布式训练的结果合并在一起
    # gathered_samples = [th.zeros_like(sample_y_update) for _ in range(dist.get_world_size())]
    # dist.all_gather(gathered_samples, sample_y_update)
    # all_sentence = [sample.cpu().numpy() for sample in gathered_samples]
    # arr = np.concatenate(all_sentence, axis=0)
    # x_t = th.tensor(arr).to(cuda)
    # reshaped_x_t = x_t
    
    logits = model.get_logits(sample_y_update)  # bsz, seqlen, dim
    cands = th.topk(logits, k=1, dim=-1)
    result = cands.indices

    print("input_id_y.shape",input_id_y.shape)
    print("result.shape",result.shape)
    #计算loss并进行梯度更新

    ouput = loss(result, input_id_y)
    ouput.backward()
    optim.step()

    end = time.time()
    update_times += 1
    print('图第%d次训练用时:%.2f min'%(update_times,(end-start)/60))

    return ouput