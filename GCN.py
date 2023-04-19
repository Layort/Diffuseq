import dgl
import torch
import torch.nn as nn
import torch.optim as optim

# 定义超参数
lr = 0.001
num_epochs = 10
hidden_size = 128
batch_size = 32

# 定义图结构和特征
g = dgl.DGLGraph()
g.add_nodes([0,1,2,3,4,5])
g.add_edges((3,1),(4,2),(2,1),(1,0),(5,3))
g.ndata['feat'] = [[1,2,3],[4,5,3],[12,3,4],[12,4,5],[5,6,6],[3,4,5]]
outsize = 3
# 定义模型
class GraphConvModel(nn.Module):
    def __init__(self, g, input_size, hidden_size, output_size):
        super(GraphConvModel, self).__init__()
        self.g = g
        self.conv1 = dgl.nn.GraphConv(input_size, hidden_size)
        self.conv2 = dgl.nn.GraphConv(hidden_size, output_size)

    def forward(self, features):
        h = self.conv1(self.g, features)
        h = torch.relu(h)
        h = self.conv2(self.g, h)
        return h

# 初始化模型和优化器
model = GraphConvModel(g, feat.shape[1], hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=lr)

# 训练模型
model.train()
for epoch in range(num_epochs):
    for batch_idx, (batch_g, batch_feats, batch_labels) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(batch_g, batch_feats)
        loss = nn.CrossEntropyLoss()(pred, batch_labels)
        loss.backward()
        optimizer.step()

# 使用训练好的模型进行推断
model.eval()
with torch.no_grad():
    pred = model(g, feat)