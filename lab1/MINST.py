# -*- coding: utf-8 -*-
"""
手写数字识别（MNIST）完整代码
功能：使用CNN训练MNIST数据集，测试准确率可达99%以上
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ==================== 1. 数据加载与预处理 ====================
def load_data(batch_size=64):
    """
    加载MNIST数据集，并进行标准化和分批
    参数:
        batch_size: 每批次数据量
    返回:
        train_loader: 训练集数据加载器
        test_loader: 测试集数据加载器
    """
    # 数据预处理：转换为Tensor -> 标准化（MNIST的均值和标准差）
    transform = transforms.Compose([
        transforms.RandomRotation(10),              # 随机旋转（-10°到+10°）
        transforms.ToTensor(),                      # PIL图像或numpy数组 -> Tensor，并归一化到[0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化到[-1,1]区间
    ])

    # 加载训练集和测试集
    train_data = datasets.MNIST(
        root='./data',          # 数据集保存路径
        train=True,             # 加载训练集
        download=True,          # 自动下载（如果本地不存在）
        transform=transform     # 应用预处理
    )
    test_data = datasets.MNIST(
        root='./data',
        train=False,           # 加载测试集
        transform=transform
    )

    # 创建数据加载器（DataLoader）
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,          # 打乱训练数据顺序
        num_workers=2          # 多线程加载（可选）
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False          # 测试集无需打乱
    )

    return train_loader, test_loader

# ==================== 2. 网络结构设计 ====================
class CNN(nn.Module):
    """
    卷积神经网络模型结构：
    输入 -> Conv1 -> ReLU -> MaxPool -> Conv2 -> ReLU -> MaxPool -> Flatten -> FC -> 输出
    """
    def __init__(self):
        super(CNN, self).__init__()
        # 第一层卷积：1通道输入 -> 16通道输出，5x5卷积核，填充2保持尺寸
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=1, padding=2),  # 输出尺寸: (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)                 # 输出尺寸: (16, 14, 14)
        )
        # 第二层卷积：16通道输入 -> 32通道输出
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, stride=1, padding=2),  # 输出尺寸: (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)                 # 输出尺寸: (32, 7, 7)
        )
        # 全连接层：32*7*7输入 -> 10类输出
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)          # 第一层卷积+池化
        x = self.conv2(x)          # 第二层卷积+池化
        x = x.view(x.size(0), -1)  # 展平多维特征图 [batch_size, 32*7*7]
        output = self.fc(x)        # 全连接层分类
        return output

# ==================== 3. 训练与评估函数 ====================
def train(model, device, train_loader, optimizer, epoch):
    """
    训练模型
    参数:
        model: CNN模型实例
        device: 训练设备（CPU/GPU）
        train_loader: 训练集数据加载器
        optimizer: 优化器
        epoch: 当前训练轮次
    """
    model.train()  # 设置为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()       # 清除历史梯度
        output = model(data)        # 前向传播
        loss = nn.CrossEntropyLoss()(output, target)  # 计算损失
        loss.backward()             # 反向传播
        optimizer.step()            # 更新参数

        # 每100个batch打印一次训练状态
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    """
    测试模型准确率
    参数:
        model: CNN模型实例
        device: 测试设备（CPU/GPU）
        test_loader: 测试集数据加载器
    返回:
        accuracy: 测试集准确率
    """
    model.eval()  # 设置为评估模式
    correct = 0
    with torch.no_grad():  # 禁用梯度计算
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)  # 获取预测类别
            correct += pred.eq(target).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

# ==================== 4. 主程序 ====================
def main():
    # 超参数设置
    batch_size = 64
    epochs = 5
    lr = 0.001

    # 选择设备（优先GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    train_loader, test_loader = load_data(batch_size)

    # 初始化模型、优化器
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练与测试
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        current_acc = test(model, device, test_loader)
        
        # 保存最佳模型
        if current_acc > best_acc:
            best_acc = current_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Best model saved with accuracy: {best_acc:.2f}%")

    
    # 获取一个批次的数据
    images, labels = next(iter(train_loader))

    # 显示前4张图像
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for i in range(4):
        ax = axes[i]
        ax.imshow(images[i].squeeze(), cmap='gray')  # 去掉通道维度并显示灰度图
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis('off')
    plt.show()

if __name__ == '__main__':
    main()