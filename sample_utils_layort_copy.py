import os
import gc
import pdb
import time
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from diffuseq_layort.utils import dist_util
from diffuseq_layort import MAX_UTTERANCE_NUM
from train_gsn_util import GetGPUInfo
update_times = 0




def langevin_fn_gsn(model_gsn, need,sample, mean, sigma, alpha ,t , pre_sample, group_lst,mask):
    global update_times
    start = time.time()
    cuda = th.device(f"cuda:{os.environ['LOCAL_RANK']}")
    cpu = th.device('cpu')
    #这里没搞懂源代码为什么要这么写
    if t[0].item() < 10:
        K = 0
    else:
        K = 3
    step_size = 0.01

    #sample.shape = 20*128*128
    sample_y =  th.broadcast_to(group_lst['pad_embed'], (model_gsn.n_batch,model_gsn.n_token,model_gsn.n_token_dim)).clone()
    
    # print("sample_y[:3,:,10;20]",sample_y[:3,:,10:20])
    # print("group_lst['pad_embed'][:,10:20]",group_lst['pad_embed'][:,10:20])
    
    mask_de =  th.any(mask,dim = 2)
    #print((mask_de).shape)
    for i in range(model_gsn.n_batch):
        len_x = len(sample[i][mask_de[i]==0])
        len_y = len(sample[i][mask_de[i]==1])

        if(len_y >= model_gsn.n_token):
            # print("mask[i]==1",mask_de[i]==1)
            # print("mask[i]==1",(mask_de[i]==1).shape)
            # print("sample[i][mask[i]==1]",sample[i][mask_de[i]==1].shape)
            # print("len_y",len_y)
            sample_y[i][:][:] = sample[i][len_x:len_x+model_gsn.n_token][:]
        else:
            sample_y[i][:len_y][:] = sample[i][len_x:len_x+len_y][:]

    # print("sample_y[:,10:20]",sample_y[:,10:20])
    
    #exit(0)
    group_lst['sample'] =  sample_y
    decode_y = decode_func(sample_y,need,model_gsn.n_token)
    #进行更新
    sample_y_update = model_gsn(group_lst,K)

    for i in range(model_gsn.n_batch):
        len_x = len(sample[i][mask_de[i]==0])
        len_y = len(sample[i][mask_de[i]==1])
        if(len_y > model_gsn.n_token):
            print("error!!!!!!!!!")
            exit(0)
        else:
            sample[i][len_x:len_x+len_y] = sample_y_update[i][:len_y]

    end = time.time()
    update_times += 1
    print('图第%d更新用时:%.2f min'%(update_times,(end-start)/60))
    
    #解下码看看
    decode = decode_func(sample,need,model_gsn.n_token,mask_de)
    decode_y_update = decode_func(sample_y_update,need,model_gsn.n_token)
    # print(decode[0])
    # print(decode[1])
    # print("decode_y[0]",decode_y[0])
    # print("decode_y[1]",decode_y[1])
    # print("decode_y_update[0]",decode_y_update[0])
    # print("decode_y_update[1]",decode_y_update[1])
    if( "[START] [START] [START] [START]" in decode_y_update[0]):
        print("sample_y_update[:,:10]",sample_y_update[:,:10,:10])
        print("group_lst['hidden_state_list']",group_lst['hidden_state_list'][:,2:4])
        print("roup_lst['dec_idx']",group_lst['dec_idx'])
        print("group_lst['sample']",group_lst['sample'][:,:2,:2])
        print("index_select",  th.index_select(group_lst['hidden_state_list'],0,group_lst['dec_idx']))
        print("error!!!!")
        exit(0)
    return sample



class GSN_matrix(nn.Module):
    def __init__(self,args, n_batch ,n_nodes,n_token,n_token_dim):
        cuda = th.device(f"cuda:{os.environ['LOCAL_RANK']}")
        cpu = th.device('cpu')
        super(GSN_matrix,self).__init__()
        self.alpha = args.alpha
        self.use_norm = args.use_norm
        self.n_batch = n_batch
        self.n_nodes = n_nodes
        self.n_token = n_token

        self.n_lstm = 300
        self.positional_enc_dim = 64

        self.n_token_dim = n_token_dim

        self.gru_fwd  = GRUCell( input_size= self.n_lstm*2  , hidden_size = self.n_lstm*2  ,bias=True, device = cuda)
        self.drop_fwd = nn.Dropout(args.dropout)

        self.gru_bwd  = GRUCell(input_size= self.n_lstm*2  , hidden_size = self.n_lstm*2   ,bias=True, device = cuda)
        self.drop_bwd = nn.Dropout(args.dropout)

        # 这里hidden_size以后可以调大一点，调成两倍
        self.bi_lstm = nn.LSTM(input_size = self.n_token_dim,  hidden_size = self.n_lstm, num_layers = 1, batch_first=True, bidirectional=True,device = cuda)

        # 这是对bw_st_h的，bw_st_c等的全连接层+激活层，即一层MLP
        self.full_h = nn.Sequential(nn.Linear(self.n_lstm * 2, self.n_lstm), nn.ReLU())
        self.full_c = nn.Sequential(nn.Linear(self.n_lstm * 2, self.n_lstm), nn.ReLU())
        # 将600转成300维
        self.full_hidden = nn.Sequential(nn.Linear(self.n_lstm * 2, self.n_lstm), nn.ReLU())
        self.GetGPUInfo = GetGPUInfo((0, 1, 2, 3))


        #这里是decode部分的参数
        self.attn_size = self.n_lstm*2 + self.positional_enc_dim

        self.W_h = nn.Parameter(th.empty((self.attn_size,self.attn_size,1, 1))).to(dist_util.dev())
        th.nn.init.xavier_uniform_(self.W_h)  # 使用Xavier初始化方法初始化

        self.v = nn.Parameter(th.empty((self.attn_size))).to(dist_util.dev())
        nn.init.uniform_(self.v)  # 使用uniform初始化方法初始化

        self.matrix1 = th.nn.Parameter(th.empty( self.n_token_dim+ self.attn_size,self.attn_size)).to(dist_util.dev())
        self.bias1 = th.nn.Parameter(th.empty(self.attn_size)).to(dist_util.dev())

        self.matrix2 = th.nn.Parameter(th.empty(self.n_lstm + self.attn_size,self.n_lstm)).to(dist_util.dev())
        self.bias2 = th.nn.Parameter(th.empty(self.n_lstm)).to(dist_util.dev())

        self.matrix3 = th.nn.Parameter(th.empty(self.n_lstm,self.attn_size)).to(dist_util.dev())
        self.bias3 = th.nn.Parameter(th.empty(self.attn_size)).to(dist_util.dev())

        #这里面放解码阶段的cell
        self.cell = GRUCell(input_size=self.attn_size, hidden_size=self.n_lstm, bias=True, device=cuda)
        # self.drop_cell = nn.Dropout(args.dropout)

        #这里面面放output阶段的loss线性计算
        #self.out_linear = nn.Linear(self.vocab_size, self.vocab_size, bias=True)

    def _get_position_encoding(self, length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
        ''' add the position encoding
        '''
        import math
        cuda = th.device(f"cuda:{os.environ['LOCAL_RANK']}")
        position = th.arange(length).float().to(cuda)
        num_timescales = hidden_size // 2
        log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) /
                                (th.tensor(num_timescales) - 1))
        inv_timescales = min_timescale * th.exp(
            th.arange(num_timescales).float() * -log_timescale_increment)
        scaled_time = th.unsqueeze(position, 1) * th.unsqueeze(inv_timescales.to(cuda), 0)
        signal = th.concat([th.sin(scaled_time), th.cos(scaled_time)], axis=1)

        return signal

    @th.no_grad()
    def _build_matirx(self,group_lst):
        #这里必须要改
        pad_token_id = 3

        cuda = th.device(f"cuda:{os.environ['LOCAL_RANK']}")
        cpu = th.device('cpu')

        #得到所有x的embeddig 不包括y，这里是我的推测，应为只有9句话
        #得到所有y的embedding， 
        #设置双向lstm，这个和在self里面初始化好了，主要是要注意输入和输出的维度
        #将我们的embedding结果放入lstm里面，得到(encoder_outputs, (fw_st, bw_st))，然=其中fw_st 和bw_st是 元组， 都包含h和c
        #将前线个号后向过程的h和c的带enc_states_h，和enc_states_c，然后进行一个全连接层以及激活层，变换后再次cat组合，加上0层就变成了hidden_state
        encode_x_lst = group_lst['input_split_x_encode']
        print("encode_x_lst.shape",encode_x_lst.shape)
        encode_y = group_lst['sample']

        print("encode_y.shape",encode_y.shape)
        encode_x_lst = encode_x_lst.reshape(self.n_batch*self.n_nodes,self.n_token,self.n_token_dim)

        #encoder_ouput.shape = n_batch * n_node , 2*hidden_size
        encoder_ouput, (fw_st,bw_st) = self.bi_lstm(encode_x_lst)
        
        #ouput.shape n_batch * n_node, n_token 2 * n_lstm
        print("encoder_ouput.shape ==",encoder_ouput.shape)
        print("fw_st.shape",fw_st.shape)

        #这里记得加上位置编码
        positonal_encoding = self._get_position_encoding(self.n_token,self.positional_enc_dim) #把第二维融合
        positonal_encoding = positonal_encoding.unsqueeze(0).repeat((self.n_batch*self.n_nodes,1,1))
        
        sen_enc_states = th.cat([positonal_encoding,encoder_ouput],dim = 2)
        print("sen_enc_states.shape",sen_enc_states.shape)
        enc_state = th.index_select(sen_enc_states ,0, group_lst['tgt_idx']) 
        

        #对fw_st和bw_st进行线性层变化
        fw_st_h, fw_st_c = fw_st
        bw_st_h, bw_st_c = bw_st
        print("bw_st_h.shape",bw_st_h.shape)

        encode_state_h = th.cat([fw_st_h,bw_st_h],dim = 1)
        encode_state_c = th.cat([fw_st_c,bw_st_c],dim = 1)
        encode_state_h = self.full_h(encode_state_h)
        encode_state_c = self.full_c(encode_state_c)
        assert encode_state_h.shape == encode_state_c.shape == (self.n_batch*self.n_nodes,self.n_lstm)
        print("encode_state_h.shape",encode_state_h.shape)
        #合并
        encode_state = th.cat([encode_state_c,encode_state_h], dim = 1)
        #进行合并 这里就是最终的sentence embedding
        hidden_state_list = th.cat([th.zeros((1,self.n_lstm*2),device = cuda),encode_state],0)

        # sentence.shape = n_batch, n_nodes-1, n_token, n_token_dim
        state_matrix = th.zeros((self.n_batch,self.n_nodes,self.n_nodes),dtype = th.float32).to(cuda)
        struct_conv  = th.zeros((self.n_batch,self.n_nodes,self.n_nodes),dtype = th.float32).to(cuda)
        
        relation_at = group_lst['relation_at']
        try:
            assert relation_at.shape[0] == self.n_batch
        except:
            print("relation_at.shape:",relation_at.shape)
            exit(0)
        for i in range(self.n_batch):
            for j in range(len(relation_at[i])):
                if(relation_at[i][j][0] != -1):
                    struct_conv[i][relation_at[i,j,1],relation_at[i,j,0]] = 1
                else:
                    break
            for j in range(self.n_nodes):
                if(group_lst['input_split_id_x'][i,j,0] != pad_token_id):
                    state_matrix[i][j][j] = self.n_nodes*i + j + 1
                else:
                    state_matrix[i][j][j] = self.n_nodes*i + j + 1
                    # print("i,j==",i,j)
                    # print("group_lst['input_split_id_x'][i,j-1]",group_lst['input_split_id_x'][i,j-1])
                    # exit(0)
                    break
        
        group_lst['state_matrix'] = state_matrix.to(cuda)
        group_lst['struct_conv'] =  struct_conv.to(cuda)
        # print("group_lst['input_split_x_y_encode'].shape",group_lst['input_split_x_y_encode'].shape)
        # print("self.n_batch * self.n_nodes,self.n_token*self.n_token_dim",self.n_batch*self.n_nodes,self.n_token*self.n_token_dim)
        #group_lst['hidden_state_list'] = th.cat([th.zeros((1,self.n_token*self.n_token_dim),device = cuda),group_lst['input_split_x_y_encode'].reshape(self.n_batch*self.n_nodes,self.n_token*self.n_token_dim) ])
        # print(group_lst['dec_idx'][group_lst['dec_idx']>400])
        #print("group_lst['sample'].shape",group_lst['sample'].shape)
        #这里得更换，首先是sample的处理，换成对应的y，这里的sample是一整个句子，还有就是dec_idx的更改
        #group_lst['hidden_state_list'][ group_lst['dec_idx'] ] =  group_lst['sample'].reshape(self.n_batch,self.n_token*self.n_token_dim)

        group_lst['hidden_state_list'] = hidden_state_list
        #  encode_y是sample的结果
        # enc_state 被回复那个话的word embedding
        return encode_y,enc_state

    def _compute_update_info(self,group_lst):
        cpu = th.device('cpu')
        cuda = th.device(f"cuda:{os.environ['LOCAL_RANK']}")
        #print('设备为',cuda)
        zero_emb = th.zeros((1 ,self.n_lstm*2), dtype=th.float32, device = cuda)
        # S.shape = [n_batch, n_max_nodes, n_token,n_token_dim]

        hidden_state_list = group_lst['hidden_state_list']
        state_matrix = group_lst['state_matrix']
        struct_conv =  group_lst['struct_conv']

        # print("state_matrix.shape",state_matrix.shape)
        # print("struct_conv.shape",struct_conv.shape)
        #print("当前设备为:",cuda)
        #print("使用内存:",self.GetGPUInfo.get_gpu_info( [int(os.environ['LOCAL_RANK'])] ) )
        struct_child  = th.matmul(state_matrix, struct_conv).long()
        struct_parent = th.matmul(struct_conv, state_matrix).long()

        S_p = th.zeros((self.n_batch, self.n_nodes, self.n_nodes, self.n_lstm*2),device = cuda)
        S_c = th.zeros((self.n_batch, self.n_nodes, self.n_nodes, self.n_lstm*2),device = cuda)

        for i in range(self.n_batch):
            for j in range(self.n_nodes):
                S_p[i][j][:] = th.index_select(hidden_state_list, 0, struct_parent[i][j])
                S_c[i][j][:] = th.index_select(hidden_state_list, 0, struct_child[i][j])
        
        # print("group_lst['input_split_id_x'][0,:,2]",group_lst['input_split_id_x'][0,:,2])
        # print("group_lst['input_split_x_y_encode'][0][:][2][40:50]",group_lst['input_split_x_y_encode'][0,:,2,40:50])
        # print("hidden_state_list[:9,128*2+40:128*2+50]",hidden_state_list[:9,128*2+40:128*2+50])
        # print("struct_parent[0][0]\n",struct_parent[0][0])
        # print("struct_child[0][0]\n",struct_child[0][0])
        # print("S_p[0][0][:][40:50]\n",S_p[0,0,:,128*2+40:128*2+50])
        # print("S_c[0][0][:][40:50]\n",S_c[0,0,:,128*2+40:128*2+50])
        # exit(0)
        S_p = S_p.reshape(self.n_batch*(self.n_nodes)**2,self.n_lstm*2)
        S_c = S_c.reshape(self.n_batch*(self.n_nodes)**2,self.n_lstm*2)
        # self.gru_fwd = GRUCellCell(input_size=self.n_lstm * 2, hidden_size=self.n_lstm * 2, bias=True, device=cuda)
        # self.drop_fwd = nn.Dropout(args.dropout)
        #
        # self.gru_bwd = GRUCellCell(input_size=self.n_lstm * 2, hidden_size=self.n_lstm * 2, bias=True, device=cuda)
        # self.drop_bwd = nn.Dropout(args.dropout)

        p_change,_ = self.gru_bwd(S_c,S_p)
        p_change = self.drop_bwd(p_change)
        p_change = p_change.reshape(self.n_batch,self.n_nodes,self.n_nodes,self.n_lstm*2)

        #计算其中的改变量
        p_change = th.sum(p_change, 1)
        p_change = th.reshape(p_change, 
                                    [self.n_batch * self.n_nodes, 
                                    self.n_lstm*2])
        p_change = th.cat([zero_emb, p_change], 0)

        #正则化操作
        if(self.use_norm):
            vlid_norm_sent = th.norm(p_change,dim =1)
            vild_norm_sent = vlid_norm_sent**2
            #print("vlid_norm_sent[:10]",vlid_norm_sent[:10])
            #alpha用来防止小的时候过小
            vlid_norm_sent = (vild_norm_sent + self.alpha)/(vild_norm_sent+1)
            vlid_norm_sent = vlid_norm_sent.reshape((self.n_batch*self.n_nodes+1,1))
            p_change = p_change*vlid_norm_sent
            #print("vlid_norm_sent[:2]",vlid_norm_sent[:2])
        hidden_state_list += p_change
        # use the norm to control the information fusion！！！！！！！！！！

        #子节点的改变
        # S_p 和 S_c 都需要更新

        S_p = th.zeros((self.n_batch, self.n_nodes, self.n_nodes, self.n_lstm*2),device = cuda)
        S_c = th.zeros((self.n_batch, self.n_nodes, self.n_nodes, self.n_lstm*2),device = cuda)

        for i in range(self.n_batch):
            for j in range(self.n_nodes):
                S_p[i][j][:] = th.index_select(hidden_state_list, 0, struct_parent[i][j])
                S_c[i][j][:] = th.index_select(hidden_state_list, 0, struct_child[i][j])

        S_p = S_p.reshape(self.n_batch*(self.n_nodes)**2,self.n_lstm*2)
        S_c = S_c.reshape(self.n_batch*(self.n_nodes)**2,self.n_lstm*2)

        c_change,_ = self.gru_fwd(S_p,S_c)
        c_change = self.drop_fwd(c_change)
        c_change = c_change.reshape(self.n_batch,self.n_nodes,self.n_nodes,self.n_lstm*2)

        c_change = th.sum(c_change, 1)
        c_change = th.reshape(c_change, 
                                    [self.n_batch * self.n_nodes,
                                    self.n_lstm*2])
        c_change = th.cat([zero_emb, c_change], 0)
        #正则化操作
        if(self.use_norm):
            vlid_norm_sent = th.norm(c_change,dim =1)
            vild_norm_sent = vlid_norm_sent**2
            #print("vlid_norm_sent[:10]",vlid_norm_sent[:10])
            #alpha用来防止小的时候过小
            vlid_norm_sent = (vild_norm_sent + self.alpha)/(vild_norm_sent+1)
            vlid_norm_sent = vlid_norm_sent.reshape((self.n_batch*self.n_nodes+1,1))
            c_change = c_change*vlid_norm_sent
            #print("vlid_norm_sent[:2]",vlid_norm_sent[:2])
        hidden_state_list += c_change
        #进入全连接层，Relu激活  hidden_state_list是所有句子编码更新后的结果,，现在hidden_state_list最后一维是300
        hidden_state_list = self.full_hidden(hidden_state_list.reshape(self.n_batch*self.n_nodes+1,self.n_lstm*2))

        group_lst['hidden_state_list'] = hidden_state_list.reshape(self.n_batch*self.n_nodes+1,self.n_lstm)

       # 要生成那句话的sentence embedding
        dec_hidden_state = th.index_select(group_lst['hidden_state_list'],0,group_lst['dec_idx'])

        return dec_hidden_state

    def forward(self,group_lst):
        #gc.collect()
        #th.cuda.empty_cache()
        # emb_dec_inputs 是sample的结果th.Size([200, 90, 128])
        # enc_state 被回复那个话的word embedding th.Size([200, 90, 664])
        with th.no_grad():
            emb_dec_inputs,enc_state = self._build_matirx(group_lst) #这里返回两个需要的值，其中group-lst本身也改变了
        
       # dec_hidden_state_init 为要生成那句话的sentence embedding th.Size([200, 300])
        dec_hidden_state_init = self._compute_update_info(group_lst)#这里是把更新后的group_lst中抽取我们想要的部分，即y的部分

        print("emb_dec_inputs.shape", emb_dec_inputs.shape) # 要生成那句话的 word embedding th.Size([200, 90, 128])
        print("dec_hidden_state_init.shape", dec_hidden_state_init.shape) # th.Size([200, 300])
        print("enc_state.shape", enc_state.shape) # th.Size([200, 90, 664])
        
        #对attn进行操作
        attn_mask = group_lst['attn_mask'].reshape(-1,self.n_token)
        attn_mask = th.index_select(attn_mask,0,group_lst['tgt_idx'])
        print("attn_mask.shape",attn_mask.shape)

        dec_out, self.dec_out_state, self.attn_dists = self.attention_decoder(emb_dec_inputs,
                                                                               dec_hidden_state_init,
                                                                               enc_state,
                                                                               self.cell,
                                                                               attn_mask,
                                                                               group_lst["attn_mask"])

        # print("group_lst['dec_idx'][:8]",group_lst['dec_idx'][:8])
        # print("group_lst['hidden_state_list'][:10,40:50]",group_lst['hidden_state_list'][:10,40:50])
        #print('shape = ',tgt_hidden_state.shape)
        assert(dec_hidden_state_init.shape == (self.n_batch,self.n_lstm))
        
        return dec_hidden_state_init

    def attention_decoder(self,decoder_inputs,
                        initial_state,
                        encoder_states,
                        cell,
                        attn_mask,
                        initial_state_attention=False,
                        pointer=False):
        batch_size = encoder_states.size(0)
        attn_size = encoder_states.size(2)
        encoder_states = encoder_states.unsqueeze(2)
        encoder_states = encoder_states.permute((0,3,1,2))
        print("encoder_states.shape = \t",encoder_states.shape)
        print("self.W_h.shape = \t\t",self.W_h.shape)
        encoder_features = F.conv2d(encoder_states, self.W_h, stride=[1, 1], padding='same')
        
        encoder_features = encoder_features.permute(0,2,3,1)
        encoder_states = encoder_features.permute(0,2,3,1)
        print('encoder_states.shape:', encoder_states.shape)
        print("encoder_features.shape=\t",encoder_features.shape) #[200, 90, 1, 664]
        # self.v = tf.compat.v1.get_variable("self.v", [attn_size])
        print("attn_size==",attn_size)# 664

        def attention(decoder_state):
            """Calculate the context vector and attention distribution from the decoder state."""

            # 使用卷积层对decoder_state进行卷积操作，得到decoder_features
            print("decoder_state.shape[1],attn_size",decoder_state.shape[1],attn_size)
            decoder_features = self.concat_conv([decoder_state], self.matrix3,self.bias3)

            # 在decoder_features上插入两个维度，以便与encoder_features形状匹配
            decoder_features = decoder_features.unsqueeze(1).unsqueeze(1)
            # 计算加权和，得到e向量，然后加上attn_mask
            e = th.sum(self.v * th.tanh(encoder_features + decoder_features), [2, 3])
            print("attn_mask.shape",attn_mask.shape)
            print("e.shape",e.shape)
            e += attn_mask

            # 对e向量进行softmax操作，得到注意力分布
            attn_dist = F.softmax(e, dim=1)
            
            # 计算上下文向量，通过对encoder_states进行加权求和
            print("th.reshape(attn_dist, [batch_size, -1, 1, 1]).shape",th.reshape(attn_dist, [batch_size, -1, 1, 1]).shape,"\nencoder_states.shape",encoder_states.shape)
            context_vector = th.sum(th.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states, [2, 3])
            print("context_vector.shape",context_vector.shape)
            # 调整上下文向量的形状
            context_vector = context_vector.reshape(-1, attn_size)

            return context_vector, attn_dist


        outputs, attn_dists = [], []
        state = initial_state
        context_vector = th.zeros([batch_size, attn_size]).to(dist_util.dev())#200, 664
        print('context_vector.shape:', context_vector.shape)
        context_vector = context_vector.reshape(-1, attn_size)#200, 664
        print('context_vector.shape:', context_vector.shape)

        decoder_inputs = th.split(decoder_inputs, 1, dim=1) #返回是元组, pyth没办法进行向量运算
        #90*(200,1,128)

        print('INFO: Adding attention_decoder of {} timesteps...'.format(len(decoder_inputs)))

        # with th.no_grad():
        for i, inp in enumerate(decoder_inputs):

            inp = inp.squeeze(1)
            print(inp.shape)
            print(context_vector.shape)
            print(self.matrix1.shape)
            print(self.bias1.shape)
            x = self.concat_conv([inp] + [context_vector], self.matrix1, self.bias1)
            print("x.shape , state.shape == ",x.shape , state.shape )
            print("state.shape",state.shape) #
            cell_output, state = cell(x, state) #cell_ouput size = 70, 300
            print("state.shape",state.shape) #70 ,300

            context_vector, attn_dist = attention(state)
            attn_dists.append(attn_dist)

                
            output = self.concat_conv([cell_output] + [context_vector],self.matrix2,self.bias2)
            outputs.append(output)

        return outputs, state, attn_dists



    def concat_conv(self,args,  matrix , bias  ):#计算 合并后的卷积
        if len(args) == 1:
            res = th.matmul(args[0], matrix)
        else:
            res = th.matmul(th.concat( args, dim=1), matrix)
        if any(bias):
            bias_variable = bias
            res += bias_variable 
        return res



@th.no_grad()
def get_tgt_embeddings(n_batch,n_nodes,group_lst):
    cpu = th.device('cpu')
    cuda = th.device(f"cuda:{os.environ['LOCAL_RANK']}")
    #print("group_lst['dec_idx']",group_lst['dec_idx'])
    tgt_index = group_lst['dec_idx'].to(cuda) + th.arange(n_batch).to(cuda)*n_nodes + 1
    group_lst['dec_idx'] = tgt_index.to(cuda)
    return group_lst

def decode_func(tensor,need,seq_len,mask=None):
    assert tensor.shape[-1] == 128
    model,tokenizer = need
    logits = model.get_logits(tensor)
    cands = th.topk(logits, k=1, dim=-1)
    sample = cands.indices
    word_lst_recover = []
    if(mask==None):
        for seq in sample:
            tokens = tokenizer.decode_token(seq)
            word_lst_recover.append(tokens)
    else:
        for seq, input_mask in zip(sample, mask):
            len_x = seq_len - sum(input_mask).tolist()
            tokens = tokenizer.decode_token(seq[len_x:])
            word_lst_recover.append(tokens)
    return word_lst_recover


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, device = dist_util.dev()):
        super(GRUCell, self).__init__()
        self.hidden_size = hidden_size#调用已有的grucell
        self.grucell = nn.GRUCell(input_size, hidden_size ,bias=True, device = device )
        self.out_linear = nn.Linear(hidden_size, hidden_size,device = device)

    def forward(self, x, hid):
        if hid is None:
            hid = th.randn(x.shape[0], self.hidden_size)
        next_hid = self.grucell(x, hid)  # 需要传入隐藏层状态
        y = self.out_linear(next_hid)
        return y, next_hid.detach()  # detach()和detach_()都可以使


