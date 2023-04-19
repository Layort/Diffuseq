# import numpy as np
import torch as th
# import pdb
import dgl
from dgl.nn import GraphConv

# u, v = th.tensor([0, 1, 2]), th.tensor([2, 3, 4])
# u, v = u.to('cuda:0'), v.to('cuda:0')
# g = dgl.graph((u, v))
# print(g.device)
# print(g)
g = dgl.graph(([0,1,2,3,2], [1,2,3,4,5]))
feat = th.ones(6, 30)
print(feat)
conv = GraphConv(30, 30, norm='both', weight=True, bias=True, allow_zero_in_degree=True)
print(conv)
res = conv(g, feat)
torch = th.relu(res)
print(res)

# b=np.array([[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]])
# for i , example in enumerate(b):
#     # print('i:\n',i)
#     # print('example\n',example)
#     for j , ex in enumerate(example):
#         print('j:\n', j)
#         print('ex\n', ex)
# a = [[0, 91, 53, 221, 115, 78, 890, 2, 78, 1658, 20, 78, 66, 951, 22, 518, 66, 2, 102, 21, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0], [181, 2, 179, 2, 374, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [557, 100, 308, 22, 130, 288, 2, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [70, 8, 300, 1627, 11, 12, 32, 22, 12, 11, 21, 134, 385, 22, 2, 358, 78, 14, 8, 188, 115, 13, 0, 0, 0, 0, 0, 0, 0, 0], [46, 2, 27, 90, 62, 92, 65, 20, 49, 601, 738, 82, 98, 343, 78, 66, 418, 565, 27, 513, 2, 53, 62, 92, 30, 565, 1186, 0, 0, 0], [22, 23, 149, 32, 34, 20, 686, 1175, 2, 300, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
# b = np.array(a)
# # print(len(a))
# relation_at = [[1, 0], [2, 1], [3, 2], [4, -1], [-1, -1]]
# #
# input_split_id_x =[[   0,   91,   53,  221,  115,   78,  890,    2,   78, 1658,   20,   78,
#           66,  951,   22,  518,   66,    2,  102,   21,    1,    3,    3,    3,
#            3,    3,    3,    3,    3,    3] ,[   0,   13,   70,    8,  300, 1627,   11,   12,   32,   22,   12,   11,
#           21,  134,  385,   22,    2,  358,   78,   14,    8,  188,  115,    1,
#            3,    3,    3,    3,    3,    3],[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
#         3, 3, 3, 3, 3, 3]]
# input_split_update = []
# for sublist in input_split_id_x:
#
#     if  sublist[0] != 3:
#         print('sublist',sublist)
#         input_split_update.append(sublist)
#
# print(len(input_split_update))


# enc_state = 存储编码的信息，batch attn
# emb_dec_input = tgt_hidden_state
# dec_hidden_state_init =  emb-dec-input * w+ b
# cell =  self.gru_fwd
# attn_mask = group_lst['input_mask']还缺这个参数
#
# dec_out, self.dec_out_state, self.attn_dists = attention_decoder(emb_dec_inputs,
#                                                                  dec_hidden_state_init,
#                                                                  enc_state,
#                                                                  cell,
#                                                                  attn_mask,
#                                                                  hps.mode=="decode")


#         return tgt_hidden_state
#
#
# def get_tgt_embeddings(n_batch,n_nodes,group_lst):
#     cpu = th.device('cpu')
#     cuda = th.device(f"cuda:{os.environ['LOCAL_RANK']}")
#     #print("group_lst['tgt_idx']",group_lst['tgt_idx'])
#     tgt_index = group_lst['tgt_idx'].to(cuda) + th.arange(n_batch).to(cuda)*n_nodes + 1
#     group_lst['tgt_idx'] = tgt_index.to(cuda)
#     return group_lst


# def attention_decoder(decoder_inputs,
#                       initial_state,
#                       encoder_states,
#                       cell,
#                       attn_mask,
#                       initial_state_attention=False,
#                       pointer=False):
#     with tf.variable_scope("attention_decoder") as scope:
#         batch_size = encoder_states.get_shape()[0].value
#         attn_size = encoder_states.get_shape()[2].value
#
#         encoder_states = tf.expand_dims(encoder_states, axis=2)
#
#         W_h = tf.compat.v1.get_variable("W_h", [1, 1, attn_size, attn_size])
#         encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1], "SAME")
#         v = tf.compat.v1.get_variable("v", [attn_size])
#
#         def attention(decoder_state):
#             """Calculate the context vector and attention distribution from the decoder state.
#             """
#             with tf.variable_scope("Attention"):
#                 decoder_features = concat_conv([decoder_state], attn_size, True)
#                 decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1)
#                 e = tf.reduce_sum(v * tf.tanh(encoder_features + decoder_features), [2, 3])
#                 e += attn_mask
#                 attn_dist = nn_ops.softmax(e)
#                 context_vector = tf.reduce_sum(tf.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states, [1, 2])
#                 context_vector = tf.reshape(context_vector, [-1, attn_size])
#
#             return context_vector, attn_dist
#
#         outputs, attn_dists = [], []
#         state = initial_state
#         context_vector = tf.zeros([batch_size, attn_size])
#         context_vector.set_shape([None, attn_size])
#         if initial_state_attention:
#             context_vector, _ = attention(initial_state)
#
#         decoder_inputs = tf.unstack(decoder_inputs, axis=1)
#
#         print('INFO: Adding attention_decoder of {} timesteps...'.format(len(decoder_inputs)))
#         for i, inp in enumerate(decoder_inputs):
#             if i > 0:
#                 tf.compat.v1.get_variable_scope().reuse_variables()
#
#             input_size = inp.get_shape().with_rank(2)[1]
#
#             if input_size.value is None:
#                 raise ValueError("Could not infer input size from input: %s" % inp.name)
#
#             x = concat_conv([inp] + [context_vector], input_size, True)
#
#             cell_output, state = cell(x, state)
#
#             if i == 0 and initial_state_attention:
#                 with tf.variable_scope(tf.compat.v1.get_variable_scope(), reuse=True):
#                     context_vector, attn_dist = attention(state)
#             else:
#                 context_vector, attn_dist = attention(state)
#             attn_dists.append(attn_dist)
#
#             with tf.variable_scope("AttnOutputProjection"):
#                 output = concat_conv([cell_output] + [context_vector], cell.output_size, True)
#             outputs.append(output)
#
#         return outputs, state, attn_dists





# arr = [[1, 2, 3],  [4, 5, 6],[3, 3, 3], [3, 3, 3]]
# indice = 0
# print(any(arr)!=1)
# for i in range(len(arr)):
#     if all(arr[i])==3:
#         print('1111')
#         indice = i
#         break
# arr_up = arr[:indice]
# print(arr_up)