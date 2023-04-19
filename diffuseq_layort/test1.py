import json, os
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f



# 创建 GRU 网络单元的实例
input_size = 300
hidden_size = 300
cell = nn.GRUCell(input_size, hidden_size)

# 初始化隐藏状态
batch_size = 70
h = torch.zeros(batch_size, hidden_size)


# 前向传递
# seq_len = 3
# for t in range(seq_len):
# 生成当前时间步的输入张量
x = torch.randn(batch_size, input_size)
# 使用 GRU 网络单元进行前向传递
h = cell(x, h)
print(h.shape)

# 反向传播
# loss = h.sum()
# loss.backward()
# x = torch.rand(70, 1)
# print('x.shape ', x.shape)
# state = torch.rand(70,300)
# print('state.shape ', state.shape)  # (70, 300)
# cell = nn.GRUCell(x,state)
# cell_pout = nn.Dropout(0.2)
# cell =  cell_pout(cell)

# cell_output, state = cell(x, state)
# print(state.shape)
# print(cell_output.shape)






# cell = nn.GRUCell(input_size=np.zeros([70,300]), hidden_size=np.zeros([70,300]), bias=True)
# drop_cell = nn.Dropout(0.2)
# cell = drop_cell(cell)
# x = np.zeros((70,300))
# y = np.zeros((70,300))
# cell_output, state = cell(x,y)
#
#

# with open('./config.json', 'r') as f:
#     rows = json.load(f)
#
# print(type(rows))

# y = torch.empty((20, ))
# print(y)
# print(y.shape)
# s = nn.init.uniform(y)
# print(s)

#
# t = torch.ones((70, 50, 300))
# list_of_tensors_2 = torch.split(t, 1, dim=1)

# print(type(list_of_tensors_2))
#
# for idx, t in enumerate(list_of_tensors_2):
#     print(type(t.squeeze(1).tolist()))
#     pdb.set_trace()
#     y = torch.squeeze(t, dim=1).tolist()
#
#     print("第{}个张量: shape is {} 类型{}".format(idx + 1, t.shape, type(y)))


# t = torch.squeeze(t, dim=1)
# t = torch.squeeze()
# print(t.shape)

# t1 = t.unsqueeze(1)
# t2 = t.unsqueeze(1).unsqueeze(1)
# print(t1.shape)
# print(t2.shape)
# e = torch.ones((2,3))
# dist = f.softmax(e)
# print(dist)


# attn_dist = f.softmax(e, dim=1)
# print(attn_dist)
# print(type(t))
# print(t.shape[0])
# print(t.shape[2])
# # list_of_tensors_1 = torch.split(t, 1, dim=0)
# list_of_tensors_2 = torch.split(t, 1, dim=1)
# # list_of_tensors_3 = torch.split(t, [2, 1, 2], dim=1)
# print(type(list_of_tensors_2))
#
# for idx, t in enumerate(list_of_tensors_1):
#     print("第{}个张量：shape is {}".format(idx + 1,  t.shape))
# print(len(list_of_tensors_2))
# for idx, t in enumerate(list_of_tensors_2):
#     y = torch.squeeze(t, dim=1).tolist()
#     print("第{}个张量: shape is {} 类型{}".format(idx + 1, t.shape, type(y)))
#
# # for idx, t in
# # enumerate(list_of_tensors_3):
# #     print("第{}个张量：shape is {}".format(idx + 1, t.shape))



# #squeeze 从第n维剪切矩阵（那一维必须只有一个元素）
# t = torch.ones((70, 1, 300))
# t1 = torch.squeeze(t, dim=1)
# # unsqueeze在第几维添加一个维度
# t2 = t.unsqueeze(1).unsqueeze(1)
# #cat 从第n维拼接矩阵
# t3 = torch.cat(t1,dim=1)