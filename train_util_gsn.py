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


from train_util import TrainLoop

INITIAL_LOG_LOSS_SCALE = 20.0
class GSN_train_loop(TrainLoop):
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
        word_embedding= None,
        lm_head = None,
        logits_mode = None,
    ):
        self.model = model
        # self.diffusion = diffusion
        self.data = data
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        # self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.learning_steps = learning_steps
        self.gradient_clipping = gradient_clipping

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        self.checkpoint_path = checkpoint_path # DEBUG **

        self._load_and_sync_parameters()

        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)

        if self.resume_step:
            self._load_optimizer_state()
            frac_done = (self.step + self.resume_step) / self.learning_steps
            lr = self.lr * (1 - frac_done)
            self.opt = AdamW(self.master_params, lr=lr, weight_decay=self.weight_decay)
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available(): # DEBUG **
            self.use_ddp = True
            print(dist_util.dev())
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        
        #新加入embed_model，这里没有使用ddp，感觉应该不需要，需要每一个子进程都需要一个完整的转换？
        self.word_embedding = word_embedding
        self.lm_head = lm_head
        self.logits_mode = logits_mode

    def run_loop(self):
        # local_rank = int(os.environ["LOCAL_RANK"])
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
            print("step:%d"%self.step)
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

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
                # t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

               #?这个怎么设置比较好？
                compute_losses = functools.partial(
                    self.gsn_compute_loss,
                    self.ddp_model,
                    micro_cond,
                )

                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

                log_loss_dict(
                    {f"eval_{k}": v  for k, v in losses.items()}
                )


    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)
        print("batch.shape[0],self.microbatch",batch.shape[0],self.microbatch)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            # t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            # print(micro_cond.keys())
            #包装一个函数，目的是得到loss-》dict 里面有个key为“loss”

            #?这个怎么设置比较好？
            compute_losses = functools.partial(
                self.gsn_compute_loss,
                self.ddp_model,
                micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()
            
            loss = (losses["loss"]).mean()
            print("loss.item",loss.item())
            log_loss_dict(
                {k: v  for k, v in losses.items()}
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()
            

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2: # standard cosine similarity
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight,
                                                                        text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                                hidden_repr.size(1)) # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError

    def gsn_compute_loss(self,model_gsn,group_lst):

        with th.no_grad():#这之下才是模型，在这之前的所有计算都不需要梯度下降
            loss = th.nn.CrossEntropyLoss()
            input_id_y = group_lst['input_id_y']

            group_lst['input_split_x_encode'] = self.word_embedding(group_lst['input_split_id_x'])
            
            sample_y =  self.word_embedding(input_id_y)

            group_lst['sample'] =  sample_y

        #进行更新
        sample_y_update = model_gsn(group_lst)
        
        logits = self.get_logits(sample_y_update)  # bsz, seqlen, dim
        #cands = th.topk(logits, k=1, dim=-1)

        logits = logits.permute(0,2,1)#N ,class/len_vocab,sub_seq_len
        
        # print("logits.shape\t==\t",logits.shape)
        # print("input_id_y.shape==\t",input_id_y.shape)
        # print("logits.shape\t==\t",logits.shape)

        #使用交叉熵计算loss
        ouput = loss(logits, input_id_y)

        return {"loss":ouput}


def log_loss_dict(losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        # for  sub_loss in values.detach().cpu().numpy():
        #     logger.logkv_mean(f"{key}", sub_loss)

import pynvml
class GetGPUInfo:
    """
     # ======================GPU========================#
     # 查看torch使用gpu情况，需要安装一个包pynvml
     # 直接使用pip可以安装
     # ======================GPU========================#
    """

    def __init__(self, use_index=(0,)):
        self.use_index = use_index

    @staticmethod
    def get_gpu_info(use_index=(0,)) -> str:
        """
        深度学习训练使用，可以回去显卡信息，
        使用到的包:pynvml
        :param use_index: 使用的GPU的物理编号
        :return: 显存使用的信息str
        """

        # 计算显存是GB还是MB的函数，方便后续查看数据
        def func(number):
            # number单位是MB
            if number // 1024 > 0:  # 如果number对1024取整是大于0的说明单位是GB
                return f"{number / 1024.0:.3f}GB"  # 返回值的单位是GB
            else:
                return f"{number:.3f}MB"

        # 初始化管理工具
        pynvml.nvmlInit()
        # device = torch.cuda.current_device()  # int
        gpu_count = pynvml.nvmlDeviceGetCount()  # int
        information = []
        for index in range(gpu_count):
            # 不是使用的gpu，就剔除
            if index not in use_index:
                continue
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total = meminfo.total / 1024 ** 2  # 总的显存大小,单位是MB
            used = meminfo.used / 1024 ** 2  # 已用显存大小
            free = meminfo.free / 1024 ** 2  # 剩余显存大小
            information.append(f"\nMemory Total:{func(total)}; Memory Used:{func(used)}; Memory Free:{func(free)}")
        # 关闭管理工具
        pynvml.nvmlShutdown()
        return "".join(information)

    def __call__(self):
        return self.get_gpu_info(use_index=self.use_index)
        

