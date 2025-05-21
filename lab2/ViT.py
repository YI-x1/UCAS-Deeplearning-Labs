import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import os
import time
import matplotlib.pyplot as plt

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. 数据预处理和加载
def get_dataloaders():
    """
    创建训练和测试数据加载器
    返回:
        trainloader: 训练数据加载器
        testloader: 测试数据加载器
        classes: 类别名称列表
    """
    # 不再放大图像为224，直接处理32x32
    trans_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    trans_valid = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=trans_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=trans_valid)
    
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    
    classes = trainset.classes
    return trainloader, testloader, classes

# 2. 构建ViT模型组件

def pair(t):
    """将输入转换为元组"""
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    """前馈网络模块"""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),  # GELU激活函数
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """多头注意力机制模块"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5  # 缩放因子
        
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def forward(self, x):
        x = self.norm(x)
        
        # 生成查询、键、值
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        # 计算注意力分数
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        # 应用注意力权重
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    """Transformer编码器模块"""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # 残差连接
            x = ff(x) + x    # 残差连接
        return self.norm(x)

class ViT(nn.Module):
    """完整的ViT模型"""
    def __init__(self, *, image_size=224, patch_size=16, num_classes=10, dim=768, 
                 depth=6, heads=12, mlp_dim=3072, pool='cls', channels=3, 
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        
        # 检查图像尺寸是否能被分块大小整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        
        # 图像分块和嵌入层
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        
        # 位置编码和类别token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # Transformer编码器
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        # 分类头
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)
    
    def forward(self, img):
        # 图像分块和嵌入
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        
        # 添加类别token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 添加位置编码
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        # 通过Transformer编码器
        x = self.transformer(x)
        
        # 池化 (使用类别token或平均池化)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        
        # 分类
        x = self.to_latent(x)
        return self.mlp_head(x)

# 3. 训练和测试函数

def train(model, trainloader, criterion, optimizer, epoch):
    """训练模型"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计信息
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 打印训练进度
        if batch_idx % 50 == 49:  # 每50个batch打印一次
            print(f'Epoch: {epoch}, Batch: {batch_idx+1}, '
                  f'Loss: {running_loss/(batch_idx+1):.3f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

def test(model, testloader, criterion):
    """测试模型"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss / len(testloader)
    test_acc = 100. * correct / total
    print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')
    return test_loss, test_acc

def save_checkpoint(state, filename='vit_checkpoint.pth'):
    """保存模型检查点"""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def plot_curves(train_acc, test_acc, train_loss, test_loss):
    epochs = range(1, len(train_acc)+1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, 'bo-', label='Train Acc')
    plt.plot(epochs, test_acc, 'ro-', label='Test Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, 'bo-', label='Train Loss')
    plt.plot(epochs, test_loss, 'ro-', label='Test Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig('training_curve_64.png')  # 保存图像！
    plt.show()
# 4. 主训练流程

def main():
    # 加载数据
    trainloader, testloader, classes = get_dataloaders()
    
    model = ViT(
        image_size=32,       # CIFAR10 原始图像大小
        patch_size=4,        # 每个patch为4x4，共8x8=64个patch
        num_classes=10,
        dim=256,             # 嵌入维度缩小
        depth=4,             # 层数降低
        heads=4,             # 多头注意力数减少
        mlp_dim=512,         # FFN隐藏层缩小
        dropout=0.1,
        emb_dropout=0.1
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # 训练参数
    start_epoch = 0
    epochs = 50
    best_acc = 0.0
  
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []
    # 训练循环
    for epoch in range(start_epoch, epochs):
        # 训练和测试
        train_loss, train_acc = train(model, trainloader, criterion, optimizer, epoch)
        test_loss, test_acc = test(model, testloader, criterion)
        
        # 更新学习率
        scheduler.step()
        
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, filename=f'vit_best.pth')
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%, '
              f'Best Acc: {best_acc:.2f}%')
        
    plot_curves(train_acc_list, test_acc_list, train_loss_list, test_loss_list)


if __name__ == '__main__':
    main()