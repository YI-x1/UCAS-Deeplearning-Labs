import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import math
from collections import Counter
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# ==================== 配置区域 ====================
mode = "test"  # 可选：train/test/translate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 训练参数
config = {
    'batch_size': 64,
    'd_model': 256,
    'nhead': 4,
    'num_layers': 4,
    'dim_feedforward': 1024,
    'dropout': 0.3,
    'lr': 3e-4,
    'epochs': 30,
    'clip': 1.0,
    'min_freq': 5  # 词汇表最小词频
}

# ==================== 数据模块 ====================
class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, src_vocab=None, tgt_vocab=None):
        # 读取文件时过滤空行
        with open(src_file, 'r', encoding='utf-8') as f:
            src_texts = [line.strip() for line in f if line.strip()]
        with open(tgt_file, 'r', encoding='utf-8') as f:
            tgt_texts = [line.strip() for line in f if line.strip()]
        
        # 检查文件行数
        if src_vocab is None:  # 只在训练模式检查
            assert len(src_texts) == len(tgt_texts), f"数据文件行数不匹配: {src_file}有{len(src_texts)}行, {tgt_file}有{len(tgt_texts)}行"
        
        self.src_vocab = src_vocab if src_vocab else self.build_vocab(src_texts)
        self.tgt_vocab = tgt_vocab if tgt_vocab else self.build_vocab(tgt_texts)
        
        self.data = []
        for src, tgt in zip(src_texts[:len(tgt_texts)], tgt_texts[:len(src_texts)]):  # 确保长度一致
            src_ids = [self.src_vocab['<sos>']] + [
                self.src_vocab.get(word, self.src_vocab['<unk>']) 
                for word in src.split()
            ][:100] + [self.src_vocab['<eos>']]
            
            tgt_ids = [self.tgt_vocab['<sos>']] + [
                self.tgt_vocab.get(word, self.tgt_vocab['<unk>']) 
                for word in tgt.split()
            ][:100] + [self.tgt_vocab['<eos>']]
            
            self.data.append((torch.LongTensor(src_ids), torch.LongTensor(tgt_ids)))

    def build_vocab(self, texts):
        vocab = {'<pad>':0, '<unk>':1, '<sos>':2, '<eos>':3}
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        
        for word, freq in counter.items():
            if freq >= config['min_freq'] and word not in vocab:
                vocab[word] = len(vocab)
        return vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = nn.utils.rnn.pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=0, batch_first=True)
    return src_padded, tgt_padded

# ==================== 模型模块 ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.d_model = config['d_model']
        
        # 嵌入层
        self.src_embed = nn.Embedding(src_vocab_size, config['d_model'])
        self.tgt_embed = nn.Embedding(tgt_vocab_size, config['d_model'])
        self.pos_encoder = PositionalEncoding(config['d_model'])
        
        # Transformer核心
        self.transformer = nn.Transformer(
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_encoder_layers=config['num_layers'],
            num_decoder_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            batch_first=True  # 添加batch_first解决警告
        )
        
        # 输出层
        self.fc_out = nn.Linear(config['d_model'], tgt_vocab_size)
        
    def forward(self, src, tgt):
        # 嵌入和位置编码
        src_emb = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        
        # 生成掩码
        src_mask = (src == 0)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)
        
        # Transformer前向传播
        output = self.transformer(
            src_emb,
            tgt_emb,
            src_key_padding_mask=src_mask,
            tgt_mask=tgt_mask
        )
        
        return self.fc_out(output)

# ==================== 训练模块 ====================
def train_model():
    # 数据加载
    train_set = TranslationDataset('train.zh', 'train.en')
    valid_set = TranslationDataset('valid.zh', 'valid.en', train_set.src_vocab, train_set.tgt_vocab)
    
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=config['batch_size'], collate_fn=collate_fn)
    
    # 模型初始化
    model = Transformer(len(train_set.src_vocab), len(train_set.tgt_vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # 训练记录
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(config['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            
            loss = criterion(
                output.reshape(-1, output.shape[-1]),
                tgt[:, 1:].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        val_loss = evaluate(model, valid_loader, criterion)
        scheduler.step(val_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss/len(train_loader))
        history['val_loss'].append(val_loss)
        
        # 打印信息
        print(f'Epoch {epoch+1:02d} | Train Loss: {history["train_loss"][-1]:.3f} | Val Loss: {val_loss:.3f}')
    
    # 保存模型和词汇表
    torch.save({
        'model_state': model.state_dict(),
        'src_vocab': train_set.src_vocab,
        'tgt_vocab': train_set.tgt_vocab,
        'config': config
    }, 'transformer_model.pth')
    
    # 绘制训练曲线
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_curve.png')
    plt.show()

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:, :-1])
            loss = criterion(
                output.reshape(-1, output.shape[-1]),
                tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
    return total_loss / len(loader)

# ==================== 测试模块 ====================
def test_model():
    import sacrebleu

    # 加载模型
    checkpoint = torch.load('transformer_model.pth', map_location=device)
    model = Transformer(len(checkpoint['src_vocab']), len(checkpoint['tgt_vocab'])).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # 加载词表
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']
    idx2word = {idx: word for word, idx in tgt_vocab.items()}

    # 加载测试集源语言（test.zh）
    with open('test.zh', 'r', encoding='utf-8') as f:
        src_sentences = [line.strip().split() for line in f if line.strip()]

    # 编码函数
    def encode(sentence, vocab):
        return [vocab.get(tok, vocab['<unk>']) for tok in sentence]

    # 解码函数
    def decode(indices):
        return [idx2word.get(idx, '<unk>') for idx in indices if idx not in [0, 2, 3]]

    predictions = []

    with torch.no_grad():
        for sent in src_sentences:
            src_indices = encode(sent, src_vocab)
            src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)
            
            generated = [tgt_vocab['<sos>']]
            for _ in range(100):
                tgt_tensor = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)
                output = model(src_tensor, tgt_tensor)
                next_token = output[0, -1].argmax(-1).item()
                if next_token == tgt_vocab['<eos>']:
                    break
                generated.append(next_token)

            pred_tokens = decode(generated[1:])
            predictions.append(' '.join(pred_tokens))

    # 加载参考译文（test.en）
    with open('test.en', 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f if line.strip()]

    assert len(predictions) == len(references), f"预测数量（{len(predictions)}）与参考译文数量（{len(references)}）不一致"

    # 计算 BLEU
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    print(f'Test BLEU Score: {bleu.score:.2f}')

    # 可选：保存预测输出
    with open('model_output.txt', 'w', encoding='utf-8') as f:
        for line in predictions:
            f.write(line + '\n')



# ==================== 翻译模块 ====================
def translate_interactive():
    checkpoint = torch.load('transformer_model.pth', map_location=device)
    model = Transformer(len(checkpoint['src_vocab']), len(checkpoint['tgt_vocab'])).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = {v:k for k,v in checkpoint['tgt_vocab'].items()}
    
    print("输入中文句子进行翻译（输入quit退出）：")
    while True:
        sentence = input("> ").strip()
        if sentence.lower() == 'quit':
            break
            
        # 编码
        tokens = ['<sos>'] + sentence.split() + ['<eos>']
        src = torch.LongTensor([src_vocab.get(word, src_vocab['<unk>']) for word in tokens]).unsqueeze(0).to(device)
        
        # 解码
        tgt_indices = [2]  # <sos>
        for _ in range(100):
            tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
            output = model(src, tgt_tensor)
            next_word = output.argmax(2)[:, -1].item()
            tgt_indices.append(next_word)
            if next_word == 3:  # <eos>
                break
                
        # 输出结果
        translation = ' '.join([tgt_vocab[idx] for idx in tgt_indices[1:-1] if idx in tgt_vocab])
        print(f"翻译结果: {translation}\n")

# ==================== 主程序 ====================
if __name__ == '__main__':
    if mode == "train":
        train_model()
    elif mode == "test":
        test_model()
    elif mode == "translate":
        translate_interactive()
    else:
        raise ValueError("Invalid mode. Choose from 'train', 'test', or 'translate'")