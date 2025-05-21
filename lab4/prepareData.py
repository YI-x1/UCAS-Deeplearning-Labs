from collections import Counter
import json

def build_vocab(file_path, min_freq=1):
    counter = Counter()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            counter.update(tokens)

    # 添加特殊标记
    vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
    idx = len(vocab)

    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab


def save_vocab(vocab, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def invert_vocab(vocab):
    return {idx: token for token, idx in vocab.items()}


# 1. 准备训练数据
with open('TM-training-set/chinese.txt', 'r', encoding='utf-8') as fc, \
     open('TM-training-set/english.txt', 'r', encoding='utf-8') as fe, \
     open('train.zh', 'w', encoding='utf-8') as fzh, \
     open('train.en', 'w', encoding='utf-8') as fen:

    for zh_line, en_line in zip(fc, fe):
        zh_clean = ' '.join(zh_line.strip().split())
        en_clean = ' '.join(en_line.strip().split())

        if zh_clean and en_clean:
            fzh.write(zh_clean + '\n')
            fen.write(en_clean + '\n')


# 2. 构建词表（基于训练集）
zh_vocab = build_vocab('train.zh')
en_vocab = build_vocab('train.en')

# 保存词表
save_vocab(zh_vocab, 'zh_vocab.json')
save_vocab(en_vocab, 'en_vocab.json')

# 反转词表供推理使用
zh_idx2word = invert_vocab(zh_vocab)
en_idx2word = invert_vocab(en_vocab)

# 3. 验证集准备
with open('Dev-set/Niu.dev.txt', 'r', encoding='utf-8') as fdev, \
     open('valid.zh', 'w', encoding='utf-8') as fzh, \
     open('valid.en', 'w', encoding='utf-8') as fen:

    lines = [line.strip() for line in fdev if line.strip()]
    for i in range(0, len(lines), 2):
        if i+1 < len(lines):
            zh = ' '.join(lines[i].split())
            en = ' '.join(lines[i+1].lower().split())
            fzh.write(zh + '\n')
            fen.write(en + '\n')


# 4. 测试集准备
# with open('Test-set/Niu.test.txt', 'r', encoding='utf-8') as ftest, \
#      open('test.zh', 'w', encoding='utf-8') as fzh:

#     for zh_line in ftest:
#         zh_clean = ' '.join(zh_line.strip().split())
#         if zh_clean:
#             fzh.write(zh_clean + '\n')

# with open('test.zh', 'r', encoding='utf-8') as f:
#     zh_lines = [line.strip() for line in f if line.strip()]
    
# with open('test.en', 'r', encoding='utf-8') as f:
#     en_lines = [line.strip() for line in f if line.strip()]


# 提取 test.zh（原始中文）
with open('Test-set/Niu.test.txt', 'r', encoding='utf-8') as f_in, \
        open('test.zh', 'w', encoding='utf-8') as f_zh:
    for line in f_in:
        clean_line = ' '.join(line.strip().split())
        if clean_line:
            f_zh.write(clean_line + '\n')

# 提取 test.en（英文参考，每3行取第3行）
with open('Reference-for-evaluation/Niu.test.reference', 'r', encoding='utf-8') as f_in, \
        open('test.en', 'w', encoding='utf-8') as f_en:
    lines = [line.strip() for line in f_in]
    if len(lines) % 3 != 0:
        print("警告：参考文件行数不是3的倍数，请检查数据格式。")

    for i in range(2, len(lines), 3):  # 每3行一组，取第3行（英文）
        en_line = ' '.join(lines[i].strip().lower().split())
        if en_line:
            f_en.write(en_line + '\n')

print("✅ 测试集文件提取完成：test.zh 和 test.en 已生成。")