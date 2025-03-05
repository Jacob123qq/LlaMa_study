import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd
import urllib.request
import os
from collections import OrderedDict
import json
# 设置环境变量以允许重复加载 OpenMP 库（仅作为临时解决方案）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MASTER_CONFIG = {
    'context_window': 16,
    'vocab_size': 4325,
    'd_model': 128,
    'epochs': 1000,
    'log_interval': 10,  # 每10个batch打印一次log
    'batch_size': 32,
    'n_heads': 8,
}

url = "https://raw.githubusercontent.com/mc112611/PI-ka-pi/main/xiyouji.txt"
filename = "xiyouji.txt"
# 暂时不需要
# urllib.request.urlretrieve(url,filename)

lines = open("xiyouji.txt", 'r', encoding='utf-8').read()

# 创建简易版词表（字符级）
vocab = sorted(list(set(lines)))  # set转为集合，集合的元素不会重复，sort进行排序，一般按ascii码排序

# 查看前n个字符
head_num = 50
# print("字符前{}个：".format(head_num),vocab[:head_num])
# print('词表大小：',len(vocab))
# 字符前50个： ['\n', ' ', '!', '"', '#', '*', ',', '.', '—', '‘', '’', '“', '”', '□', '、', '。', '《', '》', '一', '丁', '七', '万', '丈', '三', '上', '下', '不', '与', '丑', '专', '且', '丕', '世', '丘', '丙', '业', '丛', '东', '丝', '丞', '丢', '两', '严', '丧', '个', '丫', '中', '丰', '串', '临']
# 词表大小： 4325

# 创建两个字典，分别为序号映射到字符，和字符找到对应的索引
itos = {i: ch for i, ch in enumerate(
    vocab)}  # i:ch for ..是一个字典的推导式，enumerate(vocab)：这是一个内置函数，它接收一个可迭代对象（在这个例子中是 vocab 列表）并返回一个枚举对象。这个枚举对象是一个迭代器，它生成包含元素索引和元素值的元组。例如，如果 vocab 是 ['a', 'b', 'c']，那么 enumerate(vocab) 将生成 (0, 'a')，(1, 'b')，(2, 'c') 这样的元组序列。
stoi = {ch: i for i, ch in enumerate(vocab)}


# 编码器
def encode(s):
    return [stoi[ch] for ch in s]


# 解码器
def decode(l):
    return ''.join([itos[i] for i in l])


# 测试
# print(encode("悟空"))
# [1318, 2691]
# print(decode([4319,1694]))

# 对全文进行编码
dataset = torch.tensor(encode(lines), dtype=torch.int16)


# print(dataset.shape)
# print(dataset)
# torch.Size([658298])
# tensor([   0, 4319, 1694,  ...,   12,    0,    0], dtype=torch.int16)

def get_batches(data, split, batchsize, context_window, config=MASTER_CONFIG):
    # 对数据集进行切分，训练集，验证集，测试集，8:1:1
    train = data[:int(0.8 * len(data))]
    val = data[int(0.8 * len(data)): int(0.9 * len(data))]
    test = data[int(0.9 * len(data)):]

    # 将全部训练数据作为batch，验证集，测试集也换个变量存储（方便看）
    batch_data = train
    if split == 'val':
        batch_data = val
    if split == 'test':
        batch_data = test

    # randint生成随机数，从0到batch_data.size(0) - context_window - 1，这之间随机生成batchsize个随机数,只是从batch_data中随机选一个起始位置，用于之后的滑动窗口提取数据
    # 在这个例子中，最后的逗号 , 表示 size 参数是一个元组 (batch_size,)。这意味着生成的张量是一个一维张量，其长度为 batch_size。
    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batchsize,))

    # x 是输入序列：从 batch_data 中从随机起始位置 ix 开始，截取 context_window 长度的子序列作为输入。
    # torch.stack：torch.stack 将列表推导式生成的子序列张量堆叠成一个新的张量。最终 x 是一个形状为 (batch_size, context_window) 的张量，包含了每个样本的 context_window 长度的输入序列。
    # .long()：batch_data[i:i+context_window] 生成的子序列是整数类型的张量，因此 .long() 用于将结果转换为长整型张量。
    x = torch.stack([batch_data[i:i + context_window] for i in ix]).long()
    y = torch.stack([batch_data[i + 1:i + context_window + 1] for i in ix]).long()

    return x, y


# xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_size'])
#
# decoded_samples = [(decode(xs[i].tolist()), decode(ys[i].tolist())) for i in range(len(xs))]
#
# print(decoded_samples)
# # [('正话间，又见僮仆来安桌子，请吃粥', '话间，又见僮仆来安桌子，请吃粥。'), ('火气蒸\n人，二来心焦口渴，对火焰', '气蒸\n人，二来心焦口渴，对火焰山'), ('高寿，但劲节翁又千岁余矣。\n高年', '寿，但劲节翁又千岁余矣。\n高年得'), ('——\n嵯峨矗矗，峦削巍巍。嵯峨矗', '—\n嵯峨矗矗，峦削巍巍。嵯峨矗矗'), ('惊道：\n“收了神通罢，晓得是这般', '道：\n“收了神通罢，晓得是这般变'), ('仙道：“登坛祈雨。”行者道：\n“', '道：“登坛祈雨。”行者道：\n“你'), ('道：“陛下，人不可貌相，海水不可', '：“陛下，人不可貌相，海水不可斗'), ('\n我。你晓得师父没有坐性，他独步', '我。你晓得师父没有坐性，他独步林')]

# 这是一个装饰器，用来通知 PyTorch 在执行函数时不要计算梯度。这是因为在评估模式下，计算损失时我们不需要反向传播，也就不需要计算和存储梯度，这可以显著提高评估速度和节省内存。
# 它确保在评估过程中不会修改模型的参数。
@torch.no_grad()
def evaluate_loss(model, config=MASTER_CONFIG):
    # 评估结果存储变量
    out = {}

    # 模型设置为评估模式,PyTorch 中的神经网络模型有两种模式：训练模式和评估模式。在评估模式下，一些特定于训练的操作（如 dropout 和 batch normalization）将被禁用，确保模型在推理时的一致性和稳定性。
    model.eval()

    # 分别在训练集和验证集里使用get_batchs函数获取评估数据
    for split in ["train", "val"]:

        losses = []

        # 评估10个batch
        for _ in range(10):
            # 拿到特征值（输入值），以及目标值（输出数据）
            xb, yb = get_batches(dataset, split, config['batch_size'], config['context_window'])

            # 把拿到函数丢进模型，获得loss,模型返回两个值：输出（通常是预测值）和损失。我们只关心损失，所以使用 _ 忽略模型输出。
            _, loss = model(xb, yb)

            # 将当前批次的损失值（loss.item() 将 PyTorch 的张量转换为 Python 的标量）添加到 losses 列表中。
            losses.append(loss.item())

        # 这里就是大家经常在控制台看到的 "train_loss"  "valid_loss"由来
        out[split] = np.mean(losses)

    # 评估完，将模型重置到训练状态，下一个epoch继续训练,在评估结束后，将模型重新置为训练模式。这是因为评估模式和训练模式的行为略有不同（比如 dropout 层的行为）。我们在进行训练时需要启用训练模式，确保模型能够继续更新参数。
    model.train()

    return out


class SimpleBrokenModel(nn.Module):
    def __init__(self, config=MASTER_CONFIG):
        super().__init__()
        self.config = config

        # 用于将离散的整数索引映射到稠密向量表示的层。
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])

        # Sequential是PyTorch中用于将多个层按顺序组合在一起的容器。每一层都会按顺序执行，输出的结果作为下一层的输入。
        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),
            nn.Linear(config['d_model'], config['vocab_size']),
        )

    # 前向传播函数，
    def forward(self, idx, targets=None):
        # 词嵌入
        # 对应上面生成的embedding进行查找，为每个词的索引找嵌入向量，形成一个 batch_size × sequence_length × d_model 的嵌入矩阵 x
        x = self.embedding(idx)
        # 输入训练，这一步的操作可以被看作是将每个词的嵌入表示转换为词汇表中每个词的得分向量。每个词会有一个对应的 vocab_size 维度的得分，表示模型预测该词在该位置的概率。
        a = self.linear(x)
        # 计算loss时，为了使loss计算更精确，我们需要将softmax去除。 以保证交叉熵损失的计算效果更好
        # logits = F.softmax(a, dim=-1)

        logits = a
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss
        else:
            return logits


# model = SimpleBrokenModel(MASTER_CONFIG)
#
# # print("咱们的模型这么多参数量:", sum([m.numel() for m in model.parameters()]))
# # # 定义了一个 Adam 优化器，用来优化 model 中的所有参数。Adam 是一种常用的自适应优化算法，适用于大多数深度学习任务。
# optimizer = torch.optim.Adam(
#     model.parameters(),
# )

def train(model, optimizer, scheduler=None, config=MASTER_CONFIG, print_long=False):
    # 存储loss
    losses = []

    start_time = time.time()

    for epoch in range(config['epochs']):
        # 优化器初始化
        optimizer.zero_grad()
        # 训练数据获取
        xs, ys = get_batches(dataset, 'train', config['batch_size'], config['context_window'])
        # 前向传播计算概率矩阵与loss
        logits, loss = model(xs, targets=ys)
        # 反向传播更新权重参数，更新学习率优化器
        loss.backward()
        optimizer.step()
        # 如果提供学习率调度器，那么学习率会通过调度器进行修改，比如学习率周期性变化，或者梯度减小，增加，具体策略需要综合考虑进行设置，详情自行查询，关键字：lr_scheduler
        if scheduler:
            scheduler.step()
        # 每隔固定的epoch进行显示
        if epoch % config['log_interval'] == 0:

            batch_time = time.time() - start_time
            # 进入评估函数
            x = evaluate_loss(model)
            print("Evaluate loss result:", x)
            losses.append(x)

            if print_long:
                print(
                    f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f} | ETA in seconds {batch_time * (config['epochs'] - epoch) / config['log_interval'] :.3f}")

            start_time = time.time()

            if scheduler:
                print("lr:", scheduler.get_lr())

    print("Validation loss:", losses[-1]['val'])
    return pd.DataFrame(losses).plot()


# train(model, optimizer)

def generate(model, config=MASTER_CONFIG, max_new_token=20):
    # 生成5个0，作为输入数据,5行一列，代表输入5个字符。 这个地方可以自行替换其他随机数测试。
    idx = torch.zeros(5, 1).long()
    print(idx[:, -config['context_window']:])
    for _ in range(max_new_token):
        # 每次输入都是根据前几个的滑动窗口进行输入
        # logits 是模型的输出，形状为 (batch_size, sequence_length, vocab_size)，表示模型为每个 token 生成的预测分布。
        logits = model(idx[:, -config['context_window']:])
        print(logits.size())
        # 是取 logits 张量的最后一个时间步（即当前序列的最后一个 token）的预测分布。
        last_time_step_logits = logits[:, -1, :]
        print('last_time_step_logits')
        print(last_time_step_logits.shape)
        # 通过 Softmax 函数将 logits 转换为概率分布。dim=-1 表示在词汇表维度（vocab_size）上进行 Softmax 操作，即计算每个 token 的概率。
        p = F.softmax(last_time_step_logits, dim=-1)
        print('p_shape')
        print(p.shape)
        # 进行采样，根据 p 中的概率分布随机选择下一个 token。p 是一个概率分布，它的每一行代表当前时间步每个 token 的生成概率。
        idx_next = torch.multinomial(p, num_samples=1)
        print('idx_next_shape')
        print(idx_next.shape)
        idx = torch.cat([idx, idx_next], dim=-1)
    print(idx.shape)
    return [decode(x) for x in idx.tolist()]


# a=generate(model)
# print(a)
class RMSNorm(nn.Module):
    def __init__(self, layer_shape, eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()
        # register_parameter 是 torch.nn.Module 类的方法，用于将一个张量（通常是一个 可训练参数）注册到模型中,这个参数会成为模型的一部分，并且可以在训练过程中被优化器更新。
        # nn.Parameter 是 torch.Tensor 的一个子类，它标记一个张量为模型的 可训练参数。这意味着，scale 这个张量会自动被加入到模型的 parameters() 方法中，并在训练时被优化。.ones所有元素初始化为1
        self.register_parameter("scale", nn.Parameter(torch.ones(layer_shape)))

    def forward(self, x):
        # torch.linalg.norm 是计算张量的范数的函数。范数是一种度量张量（或向量）大小的方式，可以看作是“长度”或“幅度”。 在这里，x 是一个三维张量，假设其形状为 (batch_size,
        # seq_len, d_model)，其中batch_size 是批次中的样本数，seq_len 是序列长度（如文本中的词数），d_model 是特征的维度（每个 token 或样本的特征数） 计算每个样本的
        # RMS（均方根）值，作为后续归一化的基础。
        ff_rms = torch.linalg.norm(x, dim=(1, 2)) * x[0].numel() ** -.5
        # 这行代码是对输入张量 x 进行归一化操作，将 RMSNorm 中计算得到的 RMS 值应用于张量 x，进行缩放。
        raw = x / ff_rms.unsqueeze(-1).unsqueeze(-1)
        # 这行代码的作用是对经过 RMS 归一化处理后的张量 raw 进行缩放
        return self.scale[:x.shape[1], :].unsqueeze(0) * raw


class SimpleNotStupidModel_RMS(nn.Module):
    def __init__(self, config=MASTER_CONFIG):
        super().__init__()
        self.config = config

        # 用于将离散的整数索引映射到稠密向量表示的层。
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])

        self.rms = RMSNorm((config['context_window'], config['d_model']))
        # Sequential是PyTorch中用于将多个层按顺序组合在一起的容器。每一层都会按顺序执行，输出的结果作为下一层的输入。
        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),
            nn.Linear(config['d_model'], config['vocab_size']),
        )

    # 前向传播函数，
    def forward(self, idx, targets=None):
        # 词嵌入
        x = self.embedding(idx)
        # 输入训练，这一步的操作可以被看作是将每个词的嵌入表示转换为词汇表中每个词的得分向量。每个词会有一个对应的 vocab_size 维度的得分，表示模型预测该词在该位置的概率。
        x = self.rms(x)

        logits = self.linear(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss
        else:
            return logits


# model = SimpleNotStupidModel_RMS(MASTER_CONFIG)
#
# xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])
#
# logits, loss = model(xs, ys)
#
# optimizer = torch.optim.Adam(model.parameters())
#
# train(model, optimizer)


# 计算旋转位置编码
def get_rotary_matrix(context_window, embedding_dim):
    # 初始化一个0填充，形状为（context_window, embedding_dim, embedding_dim）的张量矩阵，其中context_window为token数量，后面两个embedding_dim组成正方形矩阵，与后面的attention计算对齐格式
    R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
    # 遍历所有 token 在序列中的位置，每个 position 都会计算一个 位置相关的旋转矩阵。
    for position in range(context_window):
        # 每两个维度 2*i 和 2*i+1 组成一组，用于二维旋转。
        for i in range(embedding_dim // 2):
            theta = 10000. ** (-2. * (i - 1) / embedding_dim)

            m_theta = position * theta
            R[position, 2 * i, 2 * i] = np.cos(theta)
            R[position, 2 * i, 2 * i] = -np.sin(m_theta)
            R[position, 2 * i + 1, 2 * i] = np.sin(m_theta)
            R[position, 2 * i + 1, 2 * i + 1] = np.cos(m_theta)
    return R


# 计算单头注意力机制
class RoPEMaskedAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 计算Q权重
        self.w_q = nn.Linear(config['d_model'], config['d_model'], bias=False)
        # 计算K权重
        self.w_k = nn.Linear(config['d_model'], config['d_model'], bias=False)
        # 计算V权重
        self.w_v = nn.Linear(config['d_model'], config['d_model'], bias=False)
        # 获得旋转位置编码矩阵，覆盖Q和K权重矩阵
        self.R = get_rotary_matrix(config['context_window'], config['d_model'])

    # 这里将上一个代码块中实现的创建旋转位置编码的功能函数原封不动的拿过来
    def get_rotary_matrix(context_window, embedding_dim):
        # 初始化一个0填充，形状为（context_window, embedding_dim, embedding_dim）的张量矩阵，其中context_window为token数量，后面两个embedding_dim组成正方形矩阵，与后面的attention计算对齐格式
        R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)

        # 遍历每一个位置的token
        for position in range(context_window):
            # 还记得我的上一篇文章中说的，对于特征，两两组合吗，因此需要循环的次数为embedding_dim除以2
            for i in range(embedding_dim // 2):
                # 设置θ值，采样频率，或者说旋转频率，旋转角都可以，除以embedding_dim防止梯度问题。
                theta = 10000. ** (-2. * (i - 1) / embedding_dim)
                # 根据欧拉公式，计算旋转的角度，分别有sin 和cos，将计算拉到复数空间，并将旋转角度应用在上面的0填充的矩阵
                m_theta = position * theta
                R[position, 2 * i, 2 * i] = np.cos(m_theta)
                R[position, 2 * i, 2 * i + 1] = -np.sin(m_theta)
                R[position, 2 * i + 1, 2 * i] = np.sin(m_theta)
                R[position, 2 * i + 1, 2 * i + 1] = np.cos(m_theta)
                # 得到的结果是旋转位置编码矩阵，到这里还没覆盖到attention
        return R

    # return_attn_weights: 可选布尔参数，默认为 False。如果设置为 True，则返回注意力权重矩阵。
    def forward(self, x, return_attn_weights=False):
        b, m, d = x.shape
        # 使用之前定义的线性层 self.w_q, self.w_k, 和 self.w_v 对输入 x 进行变换，得到查询（Query）、键（Key）和值（Value）矩阵。这三个矩阵的形状均为 [batch_size, sequence_length, d_model]
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        # 这两行代码的主要作用是将旋转位置编码（RoPE）应用到查询（q）和键（k）矩阵中，使得模型能够利用序列中的位置信息来增强注意力机制的效果。具体步骤包括：
        # 转置：将批次维度和时间步维度交换，以适应批量矩阵乘法的要求。
        # 批量矩阵乘法：将 RoPE 矩阵应用到每个时间步的查询和键向量中。
        # 再次转置：恢复原始维度顺序，使输出张量的形状与输入一致，便于后续处理。
        q_rotated = (torch.bmm(q.transpose(0, 1), self.R[:m])).transpose(0, 1)
        k_rotated = (torch.bmm(k.transpose(0, 1), self.R[:m])).transpose(0, 1)
        # 在这行代码中，F.scaled_dot_product_attention 被用来高效地实现缩放点积注意力机制，并且通过设置 is_causal=True 来确保因果关系，适用于自回归任务。该函数内部自动处理了点积、缩放、Softmax、因果掩码以及 Dropout 等操作，返回的是经过加权求和后的注意力输出 activations，其形状仍然是 [batch_size, sequence_length, d_model]。
        activations = F.scaled_dot_product_attention(
            q_rotated, k_rotated, v, dropout_p=0.1, is_causal=True
        )
        # 这段代码的主要目的是在返回注意力权重时，确保模型能够依据前 n 个 token 来预测下一个 token，而不是依赖于未来的 token。为此，使用了因果掩码来屏蔽未来的信息。具体步骤包括：
        # 创建因果掩码矩阵：使用 torch.tril 提取下三角部分。
        # 计算注意力权重矩阵：通过批量矩阵乘法计算查询和键之间的点积，并进行缩放。
        # 应用因果掩码：将未来位置的值设置为负无穷大，以确保 Softmax 后这些位置的权重接近于零。
        # 应用 Softmax 函数：将注意力分数转换为概率分布。
        if return_attn_weights:
            attn_mask = torch.tril(torch.ones((m, m)), diagonal=0)
            attn_weights = torch.bmm(q_rotated, k_rotated.transpose(1, 2)) / np.sqrt(d) + attn_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            return activations, attn_weights

        return activations


# 多头注意力机制
class RoPEMaskedMultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList([
            RoPEMaskedAttentionHead(config) for _ in range(config['n_heads'])
        ])

        self.linear = nn.Linear(config['n_heads'] * config['d_model'], config['d_model'])
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        heads = [h(x) for h in self.heads]
        # torch.cat(..., dim=-1): 在最后一个维度 拼接所有注意力头的输出，即：
        # 输入：n_heads 个 (batch, sequence length, d_model)
        # 输出：形状变为 (batch, sequence length, n_heads * d_model)
        x = torch.cat(heads, dim=-1)
        # 由于前面 拼接了多个头，现在需要用 self.linear 变换回原来的 d_model 维度。
        # self.linear 会学习 多头之间的关系，并生成最终的 多头注意力输出。
        x = self.linear(x)

        x = self.dropout(x)
        return x


# SwiGLU算子，它结合了 Swish 激活函数和 GLU（Gated Linear Unit） 机制，用于提升神经网络的表达能力和训练稳定性。
class SwiGLU(nn.Module):
    def __init__(self, size):
        super().__init__()
        # 定义一个门控的线性层，输入输出都是门控结构的尺寸
        self.linear_gate = nn.Linear(size, size)
        # 门控结构主干线性层
        self.linear = nn.Linear(size, size)
        # 初始化一个随机数作为beta系数，beta 是一个 可训练参数，用于调整门控结构的影响程度。
        self.beta = torch.randn(1, requires_grad=True)

        # nn.Parameter用于指定某一层参数为可学习的，即本来不能通过训练更改参数，现在变成了可以经过训练来更新的参数。
        self.beta = nn.Parameter(torch.ones(1))
        # 将随机数beta指定为一个名为beta的神经网络层
        self.register_parameter("beta", self.beta)

    def forward(self, x):
        # Swish门控计算
        # self.linear_gate(x)：对x做一次线性变换得到a
        # 先将 a 乘以 beta（可训练参数），用于动态调整数据的分布。然后通过 sigmoid 激活，使得其范围压缩到 (0,1)，起到门控作用。
        swish_gate = self.linear_gate(x) * torch.sigmoid(self.beta * self.linear_gate(x))
        # 这里 self.linear(x) 代表 主干网络的线性变换，相当于另一个全连接层
        # SwishGate(x) 是 门控单元，决定哪些信息可以通过。
        # Linear(x) 是 主干变换，即模型对输入的主要计算路径。
        out = swish_gate * self.linear(x)
        return out


# 将RMS、RopeAttention，swiglu都加入
class RopeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        # RMSNorm层
        self.rms = RMSNorm((config['context_window'], config['d_model']))
        # 旋转位置编码器+注意力机制
        self.rope_attention = RoPEMaskedMultiheadAttention(config)
        # 线性层+激活函数变为非线性输出！
        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            # 添加SwiGLU层
            SwiGLU(config['d_model'])
            # nn.ReLU(),
        )

        # 最终的输出，因为需要解码，因为输出的维度与词表大小统一！！！
        self.last_linear = nn.Linear(config['d_model'], config['vocab_size'])

    # 前向传播
    def forward(self, idx, targets=None):
        # embedding，不解释
        x = self.embedding(idx)
        # 归一化数值，不解释
        x = self.rms(x)
        # 旋转位置编码（RoPE）和注意力机制：
        # self.rope_attention(x) 代表 RoPE 位置编码结合 Masked Multihead Attention，它的作用是让模型更好地捕捉 序列信息 和 相对位置关系，特别适用于 自回归任务（如文本生成）。
        # x + self.rope_attention(x) 这一操作是 残差连接（Residual Connection）
        # 通过残差连接：
        # 让原始输入信息 不丢失，同时加入新的注意力信息。
        # 缓解深度神经网络的 梯度消失 问题，使训练更加稳定。
        x = x + self.rope_attention(x)
        # 再归一化！
        x = self.rms(x)
        # 因为直接计算归一化的数值可能出现梯度问题，因此把归一化的值作为修正系数，再覆盖！
        # x + self.linear(x) 依然是 残差连接：
        # x（归一化后的特征）和 self.linear(x)（非线性变换后的特征）相加，使得信息流动更加平稳。
        # 避免数值爆炸或梯度消失，提高模型的 训练稳定性。
        x = x + self.linear(x)
        # 到这里，才是最终输出vocab数量的神经元输出！！！！！！
        logits = self.last_linear(x)

        # 训练阶段有目标值
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss
        # 验证或者推理阶段，目标值y没有！只有结果，没有loss！
        else:
            return logits


# model = RopeModel(MASTER_CONFIG)
# xs, ys = get_batches(dataset,'train',MASTER_CONFIG['batch_size'],MASTER_CONFIG['context_window'])
# logits,loss= model(xs,ys)
# optimizer=torch.optim.Adam(model.parameters())
# train(model,optimizer)

MASTER_CONFIG.update({
    'n_layers': 4,
})


# 所有算子都有了，RMS，ROPE，SWIGLU
class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rms = RMSNorm((config['context_window'], config['d_model']))
        self.attention = RoPEMaskedMultiheadAttention(config)
        # 作用：构建前馈神经网络 (Feedforward Network, FFN)，用于 特征变换。
        self.feedforward = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
        )

    def forward(self, x):
        x = self.rms(x)
        # 使得模型能够关注不同 token 之间的关系。
        # 残差连接 (Residual Connection) 保留输入信息，同时允许梯度更稳定地传播，防止梯度消失。
        x = x + self.attention(x)
        # 再次 归一化，确保信息流动稳定。
        x = self.rms(x)
        x = x + self.feedforward(x)
        return x


# block = LlamaBlock(MASTER_CONFIG)
# # 生成一条随机数据，丢到这个llama功能块里，看一下是不是有bug
# random_input = torch.randn(MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'], MASTER_CONFIG['d_model'])
# # 执行以下看看输出
# output = block(random_input)
# output.shape


# 组装LlamMa
class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embeddings = nn.Embedding(config['vocab_size'], config['d_model'])
        self.llama_blocks = nn.Sequential(
            OrderedDict([(f"llama_{i}", LlamaBlock(config)) for i in range(config['n_layers'])])
        )
        self.ffn = nn.Sequential(
            nn.Linear(config['d_model'], config["d_model"]),
            SwiGLU(config['d_model']),
            nn.Linear(config['d_model'], config['vocab_size']),
        )
        # 看看咱们的大模型多少参数！
        print("model params:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        x = self.embeddings(idx)
        x = self.llama_blocks(x)
        logits = self.ffn(x)
        # 推理阶段没有目标值，只输出结果
        if targets is None:
            return logits
        # 训练阶段，有目标值，需要输出结果，以及loss，用于反向传播更新权重！
        else:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss


# 开始训练咱们的Llama
# llama = Llama(MASTER_CONFIG)
# xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])
# logits, loss = llama(xs, ys)
# optimizer = torch.optim.Adam(llama.parameters())
# train(llama, optimizer)
# # 再看一下推理效果（实际上也没什么效果-。-）
# # 别忘了generate里面的输入数据是咱们弄的5个0，如果替换为encode之后的数也是可以的！组成列表，转换tensor，这个应该没问题的吧~
# generated_text = generate(llama, MASTER_CONFIG, 500)[0]
# print(generated_text)
#
# # 测试集
# xs, ys = get_batches(dataset, 'test', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])
# # 丢进Llama获取loss
# logits, loss = llama(xs, ys)
# print(loss)

MASTER_CONFIG.update({
    "epochs": 1000
})
llama_with_cosine = Llama(MASTER_CONFIG)
llama_optimizer = torch.optim.Adam(
    llama_with_cosine.parameters(),
    betas=(.9, .95),
    weight_decay=.1,
    eps=1e-9,
    lr=1e-3
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(llama_optimizer, 300, eta_min=1e-5)
train(llama_with_cosine, llama_optimizer, scheduler=scheduler)

# 保存模型
model_save_path="./model_save/pytorch_model.bin"
torch.save(llama_with_cosine.state_dict(),model_save_path)
# 生成一个config文件
config_save_path = "./model_save/config.json"
with open(config_save_path, 'w') as f:
    json.dump(MASTER_CONFIG, f)
# 保存optimizer和学习率调度器的状态，方便继续微调
optimizer_save_path = "./model_save/optimizer.pt"
torch.save(llama_optimizer.state_dict(), optimizer_save_path)
scheduler_save_path = "./model_save/scheduler.pt"
torch.save(scheduler.state_dict(), scheduler_save_path)

# 接下来是加载模型
llama_with_cosine = Llama(MASTER_CONFIG)

# 加载模型权重
model_save_path = "./model_save/pytorch_model.bin"
llama_with_cosine.load_state_dict(torch.load(model_save_path))

# 设置为评估模式
llama_with_cosine.eval()
# 加载优化器和学习率调度器，如果需要继续训练什么的。
llama_optimizer.load_state_dict(torch.load(optimizer_save_path))
scheduler.load_state_dict(torch.load(scheduler_save_path))
output = generate(llama_with_cosine, MASTER_CONFIG)
print(output)