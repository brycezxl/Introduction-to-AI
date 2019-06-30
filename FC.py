import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm


# 定义网络结构
class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.name = "FC"

        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# def train(transform_train, epoches=20):
#     # 图像预处理 带有翻转等数据扩增
#     # transform
#
#     transform0 = transforms.Compose([
#         transforms.ToTensor()
#     ])
#
#     # 直接加载pytorch已有的mnist数据集
#     trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
#     testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform0)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
#
#     # device : GPU or CPU
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     # net 加载刚刚写好的网络
#     net = FC().to(device)
#     # 损失函数:这里用交叉熵
#     criterion = nn.CrossEntropyLoss()
#     # 优化器 这里用SGD 随机梯度下降
#     optimizer = optim.Adam(net.parameters())
#
#     # 开始训练
#     best_acc = 0
#     for epoch in range(epoches):
#
#         running_loss = 0
#         running_correct = 0
#         running_num = 0
#         testing_correct = 0
#         test_num = 0
#
#         for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
#             # 分批次取出数据
#             inputs, labels = data
#             inputs, labels = inputs.to(device), labels.to(device)
#
#             # 数据送到网络中
#             outputs = net(inputs)
#             _, pred = torch.max(outputs.data, 1)
#             # 结果放到损失函数里
#             loss = criterion(outputs, labels)
#             # 先清空之前的梯度
#             optimizer.zero_grad()
#             # 反向梯度的计算
#             loss.backward()
#             # 梯度更新
#             optimizer.step()
#
#             running_loss += loss.item()
#             running_correct += (pred == labels).sum().item()
#             running_num += len(labels)
#
#         with torch.no_grad():
#             # 在接下来的代码中，所有Tensor的requires_grad都会被设置为False 不计算梯度能让程序运行更快
#
#             for data in testloader:
#                 images, labels = data
#                 images, labels = images.to(device), labels.to(device)
#
#                 out = net(images)
#                 # 获得10个标签中概率最大的那个
#                 _, predicted = torch.max(out.data, 1)
#                 # 计算正确数
#                 testing_correct += (predicted == labels).sum().item()
#                 test_num += len(labels)
#
#         print('Epoch: %2d / %2d | Loss: %.4f | Train Acc: %2.2f%% '
#               '| Test Acc: %2.2f%% | Best: %s' % (epoch + 1, epoches, running_loss / len(trainloader),
#                                                   (running_correct / running_num * 100),
#                                                   (testing_correct / test_num * 100),
#                                                   ("Yes" if (testing_correct / test_num > best_acc) else "No")))
#         if (testing_correct / test_num) > best_acc:
#             best_acc = testing_correct / test_num
#             torch.save(net, 'model/FC.pkl')


def test():
    net = torch.load('model/CNN.pkl')

    transform1 = transforms.Compose([
        transforms.ToTensor()
    ])

    # device : GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

    # 开始识别
    with torch.no_grad():
        # 在接下来的代码中，所有Tensor的requires_grad都会被设置为False
        correct = 0
        total = 0

        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            out = net(images)
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy: {}%'.format(100 * correct / total))  # 输出识别准确率


if __name__ == '__main__':
    # transform = transforms.Compose([
    #     transforms.RandomGrayscale(),
    #     transforms.ToTensor(),
    # ])
    # train(transform)
    test()
