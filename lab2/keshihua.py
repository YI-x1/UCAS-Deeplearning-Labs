import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, datasets
import torch
from PIL import Image

# 设置matplotlib中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载原始CIFAR10数据（仅转换为Tensor，不做其他预处理）
original_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
original_loader = torch.utils.data.DataLoader(original_set, batch_size=5, shuffle=True)

# 2. 获取一批样本（原始图像已经是Tensor格式）
original_images, labels = next(iter(original_loader))

# 3. 定义预处理流程（修改为接受Tensor输入）
trans_train = transforms.Compose([
    transforms.ToPILImage(),  # 先将Tensor转为PIL Image
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trans_valid = transforms.Compose([
    transforms.ToPILImage(),  # 先将Tensor转为PIL Image
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. 处理图像（添加ToPILImage转换）
processed_train = torch.stack([trans_train(img) for img in original_images])
processed_valid = torch.stack([trans_valid(img) for img in original_images])

# 5. 反归一化函数
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# 6. 可视化函数
def plot_images(original, processed_train, processed_valid, labels, class_names):
    plt.figure(figsize=(15, 8))
    
    for i in range(5):
        # 原始图像 (32x32)
        plt.subplot(3, 5, i+1)
        img = original[i].numpy().transpose((1, 2, 0))  # C,H,W -> H,W,C
        plt.imshow(img)
        plt.title(f"原始\n{class_names[labels[i]]}")
        plt.axis('off')
        
        # 训练集预处理后 (224x224)
        plt.subplot(3, 5, i+6)
        img = denormalize(processed_train[i]).numpy().transpose((1, 2, 0))
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title("训练预处理\n(RandomCrop+Flip)")
        plt.axis('off')
        
        # 验证集预处理后 (224x224)
        plt.subplot(3, 5, i+11)
        img = denormalize(processed_valid[i]).numpy().transpose((1, 2, 0))
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title("验证预处理\n(Resize+CenterCrop)")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 7. CIFAR10类别名称
class_names = ('飞机', '汽车', '鸟', '猫', '鹿', 
               '狗', '青蛙', '马', '船', '卡车')

# 8. 执行可视化
plot_images(original_images, processed_train, processed_valid, labels, class_names)