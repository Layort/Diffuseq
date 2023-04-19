import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, hidden_size)
        self.conv2 = dgl.nn.GraphConv(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


# 构建一个简单的图
g = dgl.graph(([0, 0, 1], [1, 2, 2]))

# 创建节点特征和标签
features = torch.tensor([[0.0], [1.0], [2.0]])
labels = torch.tensor([0, 1, 2])

# 添加节点特征和标签到图中
g.ndata['feat'] = features
g.ndata['label'] = labels


# 初始化模型
model = GCN(in_feats=1, hidden_size=2, num_classes=3)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(50):
    logits = model(g, features)
    loss = criterion(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
