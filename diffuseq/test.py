import dgl
from functools import partial
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from diffuseq import MAX_UTTERANCE_NUM, MAX_UTTERANCE_TOKEN

gsn_msg_forward  = fn.copy_u('h', 'm_forward') #这里直接把边复制过来
gsn_msg_backward = fn.copy_u('h','m_backward')  #反向过程把边复制过来

# dgl-> 
def gsn_reduce_forward(W,Wx,Wr,n_batch,n_nodes,n_token,n_dim,nodes):
    msg_fwd = nodes.mailbox['m_forward']
    h_self = nodes.data['h']
    #print(msg_fwd.shape)
    msg_full = th.zeros((n_batch,n_nodes,n_token,n_dim))
    msg_full[:msg_fwd.shape[0],:msg_fwd.shape[1],:msg_fwd.shape[2]] = msg_fwd
    h_full =  th.zeros((n_batch,n_nodes,n_token,n_dim))
    h_full[:] = th.tensor(n_batch*[h_self])
    cat_matrix = th.cat([msg_full,h_full],dim=1)
    
    #print(cat_matrix.shape,msg_full.shape)

    r  = Wr(cat_matrix.reshape(n_batch,n_token,n_dim,2*n_nodes)).reshape(n_batch,n_nodes,n_token,n_dim)
    r = th.sigmoid(r)
    x =  Wx(cat_matrix.reshape(n_batch,n_token,n_dim,2*n_nodes)).reshape(n_batch,n_nodes,n_token,n_dim)
    x= th.sigmoid(x)

    cat_matrix2 = th.cat([r*msg_full,h_full],dim=2)
    h = W(cat_matrix2.reshape(n_batch,n_token,n_dim,2*n_nodes)).reshape(n_batch,n_nodes,n_token,n_dim)
    h = th.tanh(h)

    delta = (1 - x)*msg_full + x*h
    delta = th.sum(delta,dim=1)
    h_self = h_self + delta[0]

    return {'h':h_self}

def gsn_reduce_backward(W,Wx,Wr,n_batch,n_nodes,n_token,n_dim,nodes):
    msg_bwd = nodes.mailbox['m_backward']
    h_self = nodes.data['h']
    #print(msg_bwd.shape)
    msg_full = th.zeros((n_batch,n_nodes,n_token,n_dim))
    msg_full[:msg_bwd.shape[0],:msg_bwd.shape[1],:msg_bwd.shape[2]] = msg_bwd
    h_full =  th.zeros((n_batch,n_nodes,n_token,n_dim))
    h_full[:] = th.tensor(n_batch*[h_self])
    cat_matrix = th.cat([msg_full,h_full],dim=1)
    
    #print(cat_matrix.shape,msg_full.shape)

    r  = Wr(cat_matrix.reshape(n_batch,n_token,n_dim,2*n_nodes)).reshape(n_batch,n_nodes,n_token,n_dim)
    r = th.sigmoid(r)
    x =  Wx(cat_matrix.reshape(n_batch,n_token,n_dim,2*n_nodes)).reshape(n_batch,n_nodes,n_token,n_dim)
    x= th.sigmoid(x)

    cat_matrix2 = th.cat([r*msg_full,h_full],dim=2)
    h = W(cat_matrix2.reshape(n_batch,n_token,n_dim,2*n_nodes)).reshape(n_batch,n_nodes,n_token,n_dim)
    h = th.tanh(h)

    delta = (1 - x)*msg_full + x*h
    delta = th.sum(delta,dim=1)
    h_self = h_self + delta[0]

    return {'h':h_self}

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(NodeApplyModule, self).__init__()
        # self.W = nn.Linear(,)
        # self.Wx = nn.Linear(,)
        # self.Wr = nn.Linear(,)

    def forward(self, node):
        return {'h'}

# dgl -> 
class GSN(nn.Module):
    def __init__(self, n_nodes, n_token,n_token_dim):
        super(GSN, self).__init__()
        self.n_nodes = n_nodes
        self.n_token = n_token
        self.n_token_dim = n_token_dim
        self.W = nn.Linear(2*n_nodes,n_nodes)
        self.Wx = nn.Linear(2*n_nodes,n_nodes)
        self.Wr = nn.Linear(2*n_nodes,n_nodes)
        #self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)
 
    def forward(self, g, feature):
        g.ndata['h'] = feature
        
        g.multi_update_all({'forward':(gsn_msg_forward,partial(gsn_reduce_forward,self.W,self.Wx,self.Wx,1,self.n_nodes,self.n_token,self.n_token_dim)),
                            'backward':(gsn_msg_backward,partial(gsn_reduce_backward,self.W,self.Wx,self.Wx,1,self.n_nodes,self.n_token,self.n_token_dim))}, 
                           'sum')
        #print(g.ndata)
        #print(g.num_nodes)
        #g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

class GSN_matrix(nn.Module):
    def __init__(self, alpha, n_batch ,n_nodes,n_token,n_token_dim,dropout):
        super(GSN_matrix,self).__init__()
        self.alpha = alpha
        self.n_batch = n_batch
        self.n_nodes = n_nodes
        self.n_token = n_token
        self.n_token_dim = n_token_dim
        self.gru_fwd = nn.GRUCell( input_size=n_nodes*n_token_dim , hidden_size = n_nodes*n_token_dim  ,bias=True)
        self.drop_fwd = nn.Dropout(dropout)
        self.gru_bwd = nn.GRUCell( input_size=n_nodes*n_token_dim , hidden_size = n_nodes*n_token_dim  ,bias=True)
        self.drop_bwd = nn.Dropout(dropout)
        self.zero_emb = th.zeros([1 ,self.n_token*self.n_token_dim], dtype=th.float32)

    def _get_tgt_embeddings(self,group_lst):
        tgt_index = th.zeros((self.n_batch),dtype=th.long)
        for i in range(self.n_batch):
            tgt_index[i] = group_lst['tgt_idx'][i] + i*self.n_nodes
        group_lst['tgt_idx'] = tgt_index
        return 

    def _compute_update_info(self,group_lst):
        # S.shape = [n_batch, n_max_nodes, n_max_token,n_token_dim]
        hidden_state_list = group_lst['hidden_state_list']
        state_matrix = group_lst['state_matrix']
        struct_conv = group_lst['struct_conv']

        struct_child  = th.matmul(state_matrix, struct_conv)
        struct_parent = th.matmul(struct_conv, state_matrix)
        print("struct_conv",struct_conv)
        print("state_matrix",state_matrix)
        print("struct_child",struct_parent)
        print("struct_child",struct_child)

        S_p = th.zeros((self.n_batch, self.n_nodes, self.n_nodes, self.n_token*self.n_token_dim))

        S_c = th.zeros((self.n_batch, self.n_nodes, self.n_nodes, self.n_token*self.n_token_dim))
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

        #子节点的改变
        c_change = self.gru_fwd(S_p,S_c)
        c_change = self.drop_fwd(c_change)
        c_change = c_change.reshape(self.n_batch,self.n_nodes,self.n_nodes,self.n_token,self.n_token_dim)

        #计算其中的改变量

        p_change = th.sum(p_change, 1)
        p_change = th.reshape(p_change, 
                                    [self.n_batch * self.n_nodes, 
                                    self.n_token*self.n_token_dim])
        p_change = th.concat([self.zero_emb, p_change], 0)
        hidden_state_list += p_change



        c_change = th.sum(c_change, 1)
        c_change = th.reshape(c_change, 
                                    [self.n_batch * self.n_nodes, 
                                    self.n_token*self.n_token_dim])
        c_change = th.concat([self.zero_emb, c_change], 0)
        hidden_state_list += c_change


        return hidden_state_list

    def build_matirx(group_lst):
        n_batch = len(group_lst['input_id_x'])
        n_nodes = MAX_UTTERANCE_NUM
        n_token = MAX_UTTERANCE_TOKEN
        n_token_dim = len(group_lst['input_id_x'][0,0]) 
        #print(n_batch,n_nodes)
        # sentence.shape = n_batch, n_nodes-1, n_token, n_token_dim
        state_matrix = th.zeros((n_batch,n_nodes,n_nodes),dtype = th.int32)
        struct_conv  = th.zeros((n_batch,n_nodes,n_nodes),dtype = th.int32)
        
        relation_at = group_lst['relation_at']

        for i in range(n_batch):
            for j in range(len(relation_at[i])):
                if(relation_at[i][j][0] != -1):
                    struct_conv[i][relation_at[i,j][1],relation_at[i][0]] = 1
            for j in range(n_nodes):
                if(group_lst['input_split_ids'][i,j,0,0] != -1):
                    state_matrix[i][j][j] = n_nodes*i + j + 1
                else:
                    print(i,j)
                    state_matrix[i][j][j] = n_nodes*i + j + 1
                    break
        
        group_lst['state_matrix'] = state_matrix
        group_lst['struct_cov'] = struct_conv
        group_lst['hidden_state_list'] = th.cat([th.zeros(1,n_token*n_token_dim),group_lst['sentence'].reshape(n_batch*n_nodes,n_token*n_token_dim) ])
        return group_lst

# 构建一个简单的图
g = dgl.heterograph({
    ('n','forward','n'):([0,1,0],[1,2,2]),
    ('n','backward','n'):([2],[0])
})

# 创建节点特征和标签
features = th.tensor([[[3.0]], [[1.0]], [[2.0]]])  # 8 * 30 * 16 

# 添加节点特征和标签到图中
g.ndata['feat'] = features

# 初始化模型
model = GSN(3,1,1)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = th.optim.Adam(model.parameters(), lr=0.01)

group_lst = {}
group_lst['input_id_x'] = th.tensor([[[1,2,3],[4,5,6],[7,8,9]],
                                     [[11,12,13],[14,15,16],[-1,-1,-1]]])
group_lst['sentence'] = th.tensor([ [ [[1,2,3],[4,5,6],[-1,-1,-1]] , [[7,8,9],[-1,-1,-1],[-1,-1,-1]] ,[[10,11,12],[13,14,15],[-1,-1,-1]]],
                                    [ [[11,12,13],[14,15,16],[-1,-1,-1]] , [[17,18,19],[-1,-1,-1],[-1,-1,-1]],[[21,22,23],[24,25,26],[-1,-1,-1]] ]]
)
group_lst['relation_at'] = th.Tensor([ [[1,0],[2,0]],
                                       [[0,1],[-1,-1]]]).long()

# 训练模型
for epoch in range(50):
    logits = model(g, features)
    print(logits)
    break;
    loss = criterion(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
