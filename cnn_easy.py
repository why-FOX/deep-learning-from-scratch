import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data

torch.manual_seed(1)  # 设置随机种子, 用于复现

# 超参数
EPOCH = 1  # 前向后向传播迭代次数
LR = 0.001  # 学习率 learning rate
BATCH_SIZE = 50  # 批量训练时候一次送入数据的size
DOWNLOAD_MNIST = True

# 下载mnist手写数据集
# 训练集
train_data = torchvision.datasets.MNIST(
    root='./MNIST/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

# 测试集
test_data = torchvision.datasets.MNIST(root='./MNIST/', train=False)  # train设置为False表示获取测试集

# 一个批训练 50个样本, 1 channel通道, 图片尺寸 28x28 size:(50, 1, 28, 28)
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)
#  测试数据预处理；只测试前2000个
test_x = torch.unsqueeze(test_data.data, dim=1).float()[:2000] / 255.0
# shape from (2000, 28, 28) to (2000, 1, 28, 28)
test_y = test_data.targets[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(  # 输入的图片 （1，28，28）
                in_channels=1,
                out_channels=16,  # 经过一个卷积层之后 （16,28,28）
                kernel_size=5,
                stride=1,  # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 经过池化层处理，维度为（16,14,14）
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(  # 输入（16,14,14）
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),  # 输出（32,14,14）
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 输出（32,7,7）
        )

        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)  # （batch_size,16,14,14）
        x = self.conv2(x)  # 输出（batch_size,32,7,7）
        x = x.view(x.size(0), -1)  # (batch_size,32*7*7)
        out = self.out(x)  # (batch_size,10)
        return out


cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # 定义优化器
loss_func = nn.CrossEntropyLoss()  # 定义损失函数

for epoch in range(EPOCH):

    for step, (batch_x, batch_y) in enumerate(train_loader):
        pred_y = cnn(batch_x)
        loss = loss_func(pred_y, batch_y)
        optimizer.zero_grad()  # 清空上一层梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新优化器的学习率，一般按照epoch为单位进行更新

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].numpy()  # torch.max(test_out,1)返回的是test_out中每一行最大的数)
            # 返回的形式为torch.return_types.max(
            #           values=tensor([0.7000, 0.9000]),
            #           indices=tensor([2, 2]))
            # 后面的[1]代表获取indices
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

# 打印前十个测试结果和真实结果进行对比
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')