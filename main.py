from AlexNet import AlexNet
from Mix import Mix
from FC import FC
from CNN import CNN
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


torch.manual_seed(1337)
torch.cuda.manual_seed(1337)


def train(transform_train, model, epoches=25, lr=1e-3):
    print(">>>>>>>>>>>>>> %s start >>>>>>>>>>>>>>" % model().name)
    # 图像预处理 带有翻转等数据扩增
    # transform

    transform0 = transforms.Compose([
        transforms.ToTensor()
    ])

    # 直接加载pytorch已有的mnist数据集
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    # device : GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net 加载刚刚写好的网络
    net = model().to(device)
    # 损失函数:这里用交叉熵
    criterion = nn.CrossEntropyLoss()
    # 优化器 这里用SGD 随机梯度下降
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # 开始训练
    best_acc = 0
    loss_list = []
    running_correct_list = []
    testing_correct_list = []

    for epoch in range(epoches):

        running_loss = 0
        running_correct = 0
        running_num = 0
        testing_correct = 0
        test_num = 0

        for i, data in enumerate(trainloader):
            # 分批次取出数据
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 数据送到网络中
            outputs = net(inputs)
            _, pred = torch.max(outputs.data, 1)
            # 结果放到损失函数里
            loss = criterion(outputs, labels)
            # 先清空之前的梯度
            optimizer.zero_grad()
            # 反向梯度的计算
            loss.backward()
            # 梯度更新
            optimizer.step()

            running_loss += loss.item()
            running_correct += (pred == labels).sum().item()
            running_num += len(labels)

        with torch.no_grad():
            # 在接下来的代码中，所有Tensor的requires_grad都会被设置为False 不计算梯度能让程序运行更快

            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                out = net(images)
                # 获得10个标签中概率最大的那个
                _, predicted = torch.max(out.data, 1)
                # 计算正确数
                testing_correct += (predicted == labels).sum().item()
                test_num += len(labels)

        loss_list.append(running_loss / len(trainloader))
        running_correct_list.append((running_correct / running_num * 100))
        testing_correct_list.append(testing_correct / test_num * 100)
        print('Epoch: %2d / %2d | Loss: %.4f | Train Acc: %2.2f%% '
              '| Test Acc: %2.2f%% | Best: %s' % (epoch + 1, epoches, running_loss / len(trainloader),
                                                  (running_correct / running_num * 100),
                                                  (testing_correct / test_num * 100),
                                                  ("Yes" if (testing_correct / test_num > best_acc) else "No")))
        if (testing_correct / test_num) > best_acc:
            best_acc = testing_correct / test_num
            torch.save(net, 'model/%s.pkl' % net.name)

    print(">>>>>>>>>>>>>> %s done  >>>>>>>>>>>>>>" % model().name)
    return loss_list, running_correct_list, testing_correct_list


if __name__ == '__main__':

    transform1 = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform2 = transforms.Compose([
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
    ])
    transform3 = transforms.Compose([
        transforms.RandomResizedCrop(size=(28, 28)),
        transforms.ToTensor(),
    ])
    transform4 = transforms.Compose([
        transforms.RandomAffine(degrees=15),
        transforms.ToTensor(),
    ])
    transform5 = transforms.Compose([
        transforms.RandomPerspective(),
        transforms.ToTensor(),
    ])

    transform_list = [transform1, transform2, transform3, transform4, transform5]

    for i, transform in enumerate(transform_list):
        print("\r -----%d----- \r" % (i + 1))
        FC_loss_list, FC_running_correct_list, FC_testing_correct_list = train(transform, FC)
        CNN_loss_list, CNN_running_correct_list, CNN_testing_correct_list = train(transform, CNN)
        AlexNet_loss_list, AlexNet_running_correct_list, AlexNet_testing_correct_list = train(transform, AlexNet)
        Mix_loss_list, Mix_running_correct_list, Mix_testing_correct_list = train(transform, Mix)

        plt.figure(num=1)
        plt.plot(range(1, (len(FC_loss_list) + 1)), FC_loss_list)
        plt.plot(range(1, (len(CNN_loss_list) + 1)), CNN_loss_list)
        plt.plot(range(1, (len(AlexNet_loss_list) + 1)), AlexNet_loss_list)
        plt.plot(range(1, (len(Mix_loss_list) + 1)), Mix_loss_list)
        plt.savefig(r"./loss%d.png" % i)
        plt.show()

        plt.figure(num=2)
        plt.plot(range(1, (len(FC_running_correct_list) + 1)), FC_running_correct_list)
        plt.plot(range(1, (len(CNN_running_correct_list) + 1)), CNN_running_correct_list)
        plt.plot(range(1, (len(AlexNet_running_correct_list) + 1)), AlexNet_running_correct_list)
        plt.plot(range(1, (len(Mix_running_correct_list) + 1)), Mix_running_correct_list)
        plt.savefig(r"./running_correct%d.png" % i)
        plt.show()

        plt.figure(num=3)
        plt.plot(range(1, (len(FC_testing_correct_list) + 1)), FC_testing_correct_list)
        plt.plot(range(1, (len(CNN_testing_correct_list) + 1)), CNN_testing_correct_list)
        plt.plot(range(1, (len(AlexNet_testing_correct_list) + 1)), AlexNet_testing_correct_list)
        plt.plot(range(1, (len(Mix_testing_correct_list) + 1)), Mix_testing_correct_list)
        plt.savefig(r"./testing_correct%d.png" % i)
        plt.show()
