import os
import time
import torch as th
import torch.nn as nn
from diffuseq.utils import dist_util
from diffuseq import MAX_UTTERANCE_NUM, MAX_UTTERANCE_TOKEN

def langevin_fn_gsn_with_loss(model_gsn, sample, mean, sigma, alpha ,t , pre_sample, group_lst):
    
    #这里没搞懂源代码为什么要这么写
    if t[0].item() < 10:
        K = 0
    else:
        K = 3
    step_size = 0.01
    input_embs_param = th.nn.Parameter(sample)
    with th.enable_grad():
        for i in range(K):
            optimizer = th.optim.Adagrad([input_embs_param], lr=step_size)
            optimizer.zero_grad()
            model_out = model_gsn(group_lst)

            coef = 0.01
            if sigma.mean() == 0:
                logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
            else:
                logp_term = coef * ((mean - input_embs_param)**2 / sigma).mean(dim=0).sum()
            
            loss = model_out.loss + logp_term

            loss.backward()
            optimizer.step()

            epsilon = th.randn_like(input_embs_param.data)
            #为什么要这样更新
            input_embs_param = th.nn.Parameter((input_embs_param.data + 0.0*sigma.mean().item() * epsilon).detach())

    return input_embs_param.data

def langevin_fn_gsn(model_gsn, sample, mean, sigma, alpha ,t , pre_sample, group_lst):
    start = time.time()
    #这里没搞懂源代码为什么要这么写
    if t[0].item() < 10:
        K = 0
    else:
        K = 3
    step_size = 0.01
    print('sample.shape',sample.shape)

    group_lst['sample'] =  sample
    sample_update = model_gsn(group_lst,K)
    end = time.time()

    print('图更新用时:%.2f min'%((end-start)/60))

    return sample_update


class GSN_matrix(nn.Module):
    def __init__(self, alpha, n_batch ,n_nodes,n_token,n_token_dim,dropout):
        super(GSN_matrix,self).__init__()
        self.alpha = alpha
        self.n_batch = n_batch
        self.n_nodes = n_nodes
        self.n_token = n_token
        self.n_token_dim = n_token_dim
        self.gru_fwd  = nn.GRUCell( input_size= n_token*n_token_dim , hidden_size = n_token*n_token_dim  ,bias=True, device = dist_util.dev())
        self.drop_fwd = nn.Dropout(dropout)
        self.gru_bwd  = nn.GRUCell( input_size= n_token*n_token_dim , hidden_size = n_token*n_token_dim  ,bias=True, device = dist_util.dev())
        self.drop_bwd = nn.Dropout(dropout)

    def _get_tgt_embeddings(self,group_lst):
        cpu = th.device('cpu')
        cuda = th.device(f"cuda:{os.environ['LOCAL_RANK']}")
        tgt_index = th.zeros((self.n_batch),dtype=th.long).to(cuda)
        tgt_index = group_lst['tgt_idx'].to(cuda) + th.arange(self.n_batch).to(cuda)*self.n_nodes + 1
        group_lst['tgt_idx'] = tgt_index.to(cuda)
        return 

    def _compute_update_info(self,group_lst):
        cpu = th.device('cpu')
        cuda = th.device(f"cuda:{os.environ['LOCAL_RANK']}")
        print('设备为',cuda)
        print(self.gru_fwd.device)
        zero_emb = th.zeros((1 ,self.n_token*self.n_token_dim), dtype=th.float32, device = cuda)
        # S.shape = [n_batch, n_max_nodes, n_max_token,n_token_dim]
        hidden_state_list = group_lst['hidden_state_list']
        state_matrix = group_lst['state_matrix']
        struct_conv = group_lst['struct_conv']

        struct_child  = th.matmul(state_matrix, struct_conv).long()
        struct_parent = th.matmul(struct_conv, state_matrix).long()

        S_p = th.zeros((self.n_batch, self.n_nodes, self.n_nodes, self.n_token*self.n_token_dim),device = cuda)
        S_c = th.zeros((self.n_batch, self.n_nodes, self.n_nodes, self.n_token*self.n_token_dim),device = cuda)

        for i in range(self.n_batch):
            for j in range(self.n_nodes):
                S_p[i][j][:] = th.index_select(hidden_state_list, 0, struct_parent[i][j])
                S_c[i][j][:] = th.index_select(hidden_state_list, 0, struct_child[i][j])

        S_p = S_p.reshape(self.n_batch*(self.n_nodes)**2,self.n_token*self.n_token_dim)
        S_c = S_c.reshape(self.n_batch*(self.n_nodes)**2,self.n_token*self.n_token_dim)
        #父节点的改变
        p_change = self.gru_bwd(S_c,S_p)
        p_change = self.drop_bwd(p_change)
        p_change = p_change.reshape(self.n_batch,self.n_nodes,self.n_nodes,self.n_token,self.n_token_dim)



        #计算其中的改变量

        p_change = th.sum(p_change, 1)
        p_change = th.reshape(p_change, 
                                    [self.n_batch * self.n_nodes, 
                                    self.n_token*self.n_token_dim])
        p_change = th.cat([zero_emb, p_change], 0)
        hidden_state_list += p_change




        #子节点的改变
        # S_p 和 S_c 都需要更新

        S_p = th.zeros((self.n_batch, self.n_nodes, self.n_nodes, self.n_token*self.n_token_dim),device = cuda)
        S_c = th.zeros((self.n_batch, self.n_nodes, self.n_nodes, self.n_token*self.n_token_dim),device = cuda)

        for i in range(self.n_batch):
            for j in range(self.n_nodes):
                S_p[i][j][:] = th.index_select(hidden_state_list, 0, struct_parent[i][j])
                S_c[i][j][:] = th.index_select(hidden_state_list, 0, struct_child[i][j])

        S_p = S_p.reshape(self.n_batch*(self.n_nodes)**2,self.n_token*self.n_token_dim).to(cpu)
        S_c = S_c.reshape(self.n_batch*(self.n_nodes)**2,self.n_token*self.n_token_dim).to(cpu)

        c_change = self.gru_fwd(S_p,S_c).to(cuda)
        c_change = self.drop_fwd(c_change)
        c_change = c_change.reshape(self.n_batch,self.n_nodes,self.n_nodes,self.n_token,self.n_token_dim)

        c_change = th.sum(c_change, 1)
        c_change = th.reshape(c_change, 
                                    [self.n_batch * self.n_nodes, 
                                    self.n_token*self.n_token_dim])
        c_change = th.cat([zero_emb, c_change], 0)
        hidden_state_list += c_change

        group_lst['hidden_state_list'] = hidden_state_list

        return hidden_state_list

    def _build_matirx(self,group_lst):
        #这里必须要改
        pad_token_id = 3

        cuda = th.device(f"cuda:{os.environ['LOCAL_RANK']}")

        # sentence.shape = n_batch, n_nodes-1, n_token, n_token_dim
        state_matrix = th.zeros((self.n_batch,self.n_nodes,self.n_nodes),dtype = th.float32)
        struct_conv  = th.zeros((self.n_batch,self.n_nodes,self.n_nodes),dtype = th.float32)
        
        relation_at = group_lst['relation_at']

        for i in range(self.n_batch):
            for j in range(len(relation_at[i])):
                if(relation_at[i][j][0] != -1):
                    struct_conv[i][relation_at[i,j][1],relation_at[i][0]] = 1
            for j in range(self.n_nodes):
                if(group_lst['input_split_ids'][i,j,0] != pad_token_id):
                    state_matrix[i][j][j] = self.n_nodes*i + j + 1
                else:
                    state_matrix[i][j][j] = self.n_nodes*i + j + 1
                    break

        group_lst['state_matrix'] = state_matrix.to(cuda)
        group_lst['struct_conv'] = struct_conv.to(cuda)
        group_lst['hidden_state_list'] = th.cat([th.zeros((1,self.n_token*self.n_token_dim),device = cuda),group_lst['input_split_encode'].reshape(self.n_batch*self.n_nodes,self.n_token*self.n_token_dim) ])
        group_lst['hidden_state_list'][ group_lst['tgt_idx']] = group_lst['sample'].reshape(self.n_batch,self.n_token*self.n_token_dim)

        return group_lst

    def forward(self,group_lst,K):
        self._get_tgt_embeddings(group_lst)
        self._build_matirx(group_lst)
        self._compute_update_info(group_lst,K)
        tgt_hidden_state = th.index_select(group_lst['hidden_state_list'],0,group_lst['tgt_idx'])
        print('shape = ',tgt_hidden_state.shape)
        assert(tgt_hidden_state.shape == (self.n_batch,self.n_token*self.n_token_dim))

        tgt_hidden_state = tgt_hidden_state.reshape(self.n_batch,self.n_token,self.n_token_dim)
        print('forward works well')
        return tgt_hidden_state
