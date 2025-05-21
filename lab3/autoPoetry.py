#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Description: 自动写诗程序，包含训练和生成功能（普通诗歌和藏头诗）
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from tensorboardX import SummaryWriter  # 用于可视化训练过程

# ===================== 1. 基础模块 =====================
class BasicModule(nn.Module):
    """
    封装nn.Module，提供模型加载和保存接口
    """
    def __init__(self):
        super(BasicModule, self).__init__()
        self.modelName = str(type(self))  # 模型名称

    def load(self, path):
        """加载指定路径的模型"""
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        """保存模型到指定路径"""
        if name is None:
            # 默认路径：models/模型名称_月日_时分.pth
            prepath = 'models/' + self.modelName + '_'
            name = time.strftime(prepath + '%m%d_%H_%M.pth')
        torch.save(self.state_dict(), name)
        print("模型保存路径：", name)
        return name

# ===================== 2. 诗歌模型 =====================
class PoetryModel(BasicModule):
    """
    诗歌生成模型：包含Embedding、LSTM和全连接层
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.modelName = 'PoetryModel'
        self.hidden_dim = hidden_dim
        
        # 网络结构
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)  # 词嵌入层
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=3)  # 3层LSTM
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, vocab_size))  # 输出层，预测下一个词的概率分布

    def forward(self, input, hidden=None):
        """
        前向传播
        - input: 输入序列 (seq_len, batch_size)
        - hidden: LSTM的隐藏状态 (h_0, c_0)
        """
        seq_len, batch_size = input.size()
        
        # 初始化隐藏状态
        if hidden is None:
            h_0 = input.data.new(3, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(3, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        
        # 通过Embedding和LSTM
        embeds = self.embeddings(input)
        output, hidden = self.lstm(embeds, (h_0, c_0))
        
        # 通过全连接层
        output = self.fc(output.view(seq_len * batch_size, -1))
        return output, hidden

# ===================== 3. 数据加载 =====================
def poetryData(filename, batch_size):
    """
    加载诗歌数据
    - filename: 数据文件路径（tang.npz）
    - batch_size: 批大小
    """
    datas = np.load(filename, allow_pickle=True)
    data = datas['data']  # 诗歌数据
    ix2word = datas['ix2word'].item()  # 索引到词的映射
    word2ix = datas['word2ix'].item()  # 词到索引的映射
    
    # 转换为Tensor并创建DataLoader
    data = torch.from_numpy(data)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)
    return dataloader, ix2word, word2ix

# ===================== 4. 训练函数 =====================
def train(model, filename, batch_size, lr, epochs, device, trainwriter, pre_model_path=None):

    dataloader, ix2word, word2ix = poetryData(filename, batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

    model.to(device)  # 模型转移到设备
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, data in enumerate(dataloader):
            # 数据转移到设备
            data = data.long().transpose(1, 0).contiguous().to(device)
            input, target = data[:-1, :], data[1:, :]
            
            # 前向传播
            output, _ = model(input)
            loss = criterion(output, target.view(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            trainwriter.add_scalar('Train Loss', loss.item(), epoch * len(dataloader) + i)
            
            if i % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {i}, Loss: {loss.item():.4f}')
        
        # 学习率调整
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
    
    trainwriter.close()
    model.save()

# ===================== 5. 生成函数 =====================
def generate(model, filename, device, start_words, max_gen_len, prefix_words=None):
    """
    生成诗歌（给定开头）
    - model: 诗歌模型
    - filename: 数据文件路径
    - device: 设备（CPU/GPU）
    - start_words: 诗歌开头
    - max_gen_len: 最大生成长度
    - prefix_words: 意境词（可选）
    """
    # 加载数据字典
    _, ix2word, word2ix = poetryData(filename, 1)
    model.to(device)
    results = list(start_words)
    
    # 初始输入为<START>
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long().to(device)
    hidden = None
    
    # 生成诗歌
    for i in range(max_gen_len):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]
        
        # 前几个词使用给定的开头
        if i < len(start_words):
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        else:
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        
        # 遇到结束符停止
        if w == '<EOP>':
            del results[-1]
            break
    
    return results

def generate_acrostic(model, filename, device, start_words_acrostic, max_gen_len_acrostic, prefix_words_acrostic=None):
    """
    生成藏头诗
    - model: 诗歌模型
    - filename: 数据文件路径
    - device: 设备（CPU/GPU）
    - start_words_acrostic: 藏头词
    - max_gen_len_acrostic: 最大生成长度
    - prefix_words_acrostic: 意境词（可选）
    """
    # 加载数据字典
    _, ix2word, word2ix = poetryData(filename, 1)
    model.to(device)
    results = []
    index = 0
    pre_word = '<START>'
    
    # 初始输入为<START>
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long().to(device)
    hidden = None
    
    # 生成藏头诗
    for i in range(max_gen_len_acrostic):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]
        
        # 遇到句尾或开头时，填入藏头词
        if pre_word in {'。', '！', '<START>'}:
            if index == len(start_words_acrostic):
                break
            else:
                w = start_words_acrostic[index]
                index += 1
                input = input.data.new([word2ix[w]]).view(1, 1)
        else:
            input = input.data.new([word2ix[w]]).view(1, 1)
        
        results.append(w)
        pre_word = w
    
    return results

# ===================== 6. 主程序 =====================
if __name__ == "__main__":
    # 超参数设置
    filename = 'tang.npz'
    batch_size = 16
    lr = 0.001
    epochs = 20
    vocab_size = 8293
    embedding_dim = 128
    hidden_dim = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    

    # 选择模式
    mode = 'generate'  # 先训练模型，生成后再改为 'generate' 或 'generate_acrostic'

    if mode == 'train':
        # 训练模型
        model = PoetryModel(vocab_size, embedding_dim, hidden_dim)
        visdir = time.strftime('assets/visualize/' + model.modelName + '_%m%d_%H_%M')
        trainwriter = SummaryWriter(f'{visdir}/Train')
        train(model, filename, batch_size, lr, epochs, device, trainwriter)

    elif mode == 'generate':
        # 加载模型
        model = PoetryModel(vocab_size, embedding_dim, hidden_dim)
        model.load('models/PoetryModel_0517_01_29.pth')  # 替换为实际文件名
        model.to(device)
        
        print("诗歌生成器已启动（输入'quit'退出）")
        while True:
            # 获取用户输入
            start_words = input("\n请输入首句诗：").strip()
            if start_words.lower() == 'quit':
                print("退出诗歌生成器")
                break
            
            # 生成诗歌
            max_gen_len = 128
            try:
                result = generate(model, filename, device, start_words, max_gen_len)
                poetry = ''.join(result)
                
                # 格式化输出
                formatted_poetry = ""
                for word in poetry:
                    formatted_poetry += word
                    if word in {'。', '！', '？', '，'}:
                        formatted_poetry += '\n'
                print("\n生成的诗歌：")
                print(formatted_poetry)
            except Exception as e:
                print(f"生成失败：{str(e)}")

    elif mode == 'generate_acrostic':
        # 加载模型
        model = PoetryModel(vocab_size, embedding_dim, hidden_dim)
        model.load('models/PoetryModel_0517_01_29.pth')  # 替换为实际文件名
        model.to(device)
        
        print("藏头诗生成器已启动（输入'quit'退出）")
        while True:
            # 获取用户输入
            start_words_acrostic = input("\n请输入藏头词：").strip()
            if start_words_acrostic.lower() == 'quit':
                print("退出藏头诗生成器")
                break
            
            # 生成藏头诗
            max_gen_len_acrostic = 128
            try:
                result = generate_acrostic(model, filename, device, start_words_acrostic, max_gen_len_acrostic)
                poetry = ''.join(result)
                
                # 高亮显示藏头字
                highlighted_poetry = ""
                for i, word in enumerate(poetry):
                    if i < len(start_words_acrostic) and word == start_words_acrostic[i]:
                        highlighted_poetry += f"\033[1;31m{word}\033[0m"  # 红色高亮
                    else:
                        highlighted_poetry += word
                    if word in {'。', '！', '？', '，'}:
                        highlighted_poetry += '\n'
                
                print("\n生成的藏头诗：")
                print(highlighted_poetry)
            except Exception as e:
                print(f"生成失败：{str(e)}")