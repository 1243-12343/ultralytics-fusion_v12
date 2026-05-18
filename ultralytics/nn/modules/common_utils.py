import math

import torch

from torch import nn
from torch.nn import functional as F


class IN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# 原版
class Multiin(nn.Module):  # stereo attention block
    def __init__(self, out=1):
        super().__init__()
        self.out = out
        # 添加1x1卷积以保持通道数一致
        self.conv = nn.Conv2d(3, 32, kernel_size=1)
        self.bn = nn.BatchNorm2d(32)
        self.act = nn.SiLU()

    def forward(self, x):
        x1, x2 = x[:, :3, :, :], x[:, 3:, :, :]
        if self.out == 1:
            x = x1
        else:
            x = x2
        return x


###############
# class Multiin(nn.Module):  # stereo attention block
#     def __init__(self, out=1):
#         super().__init__()
#         self.out = out
#         # Add a small conv operation to make FLOPs countable
#         self.conv = nn.Conv2d(3, 3, kernel_size=1, bias=False)
#         # Initialize weights to identity
#         with torch.no_grad():
#             self.conv.weight.data = torch.eye(3).view(3, 3, 1, 1)
#         # Freeze the weights
#         self.conv.weight.requires_grad = False
#
#     def forward(self, x):
#         x1, x2 = x[:, :3, :, :], x[:, 3:, :, :]
#         if self.out == 1:
#             # Apply the identity conv to make it countable
#             x = self.conv(x1)
#         else:
#             # Apply the identity conv to make it countable
#             x = self.conv(x2)
#         return x
"fft版本"


# import torch.fft
# class Multiin(nn.Module):
#     def __init__(self, out=1, use_fft=True, channels=3):
#         super(Multiin, self).__init__()
#         self.out = out
#         self.use_fft = use_fft
#         self.channels = int(round(channels))
#
#         if use_fft:
#             # 用于频域增强的轻量卷积模块（作用在幅度谱上）
#             self.freq_enhance = nn.Sequential(
#                 nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
#                 nn.BatchNorm2d(channels),
#                 nn.ReLU(inplace=True)
#             )
#
#     def forward(self, x):
#         # x: [B, 6, H, W]
#         x1, x2 = x[:, :self.channels, :, :], x[:, self.channels:, :, :]
#
#         # 模态选择
#         x_out = x1 if self.out == 1 else x2
#
#         if self.use_fft:
#             # 保证输入是 float32，避免 torch.fft 不支持 half 精度
#             x_out = x_out.to(torch.float32)
#             # 傅里叶变换 + 幅度谱
#             x_fft = torch.fft.fft2(x_out, norm='ortho')
#             x_mag = torch.abs(x_fft)
#             x_mag = x_mag / (x_mag.amax(dim=(-2, -1), keepdim=True) + 1e-8)
#
#             # 归一化幅度谱
#             x_enhanced = self.freq_enhance(x_mag)
#
#             # 可以选择将时域与频域特征融合：
#             x_out = x_out + x_enhanced  # 简单相加融合（也可以concat）
#
#         return x_out
##########################################################    fft版本        ##############################################################
# class Multiin(nn.Module):
#     def __init__(self, out=1, use_fft=True, channels=3):
#         super(Multiin, self).__init__()
#         self.out = out
#         self.use_fft = use_fft
#         self.channels = int(round(channels))
#
#         if use_fft:
#             # 用于频域增强的轻量卷积模块（作用在幅度谱上）
#             self.freq_enhance = nn.Sequential(
#                 nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
#                 nn.BatchNorm2d(channels),
#                 nn.ReLU(inplace=True)
#             )
#             # 融合通道维度（拼接后通道数变为 2xchannels）
#             self.fuse_conv = nn.Sequential(
#                 nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(channels),
#                 nn.ReLU(inplace=True)
#             )
#
#     def forward(self, x):
#         # x: [B, 6, H, W] -> 双模态 [B, 3, H, W] + [B, 3, H, W]
#         x1, x2 = x[:, :self.channels, :, :], x[:, self.channels:, :, :]
#
#         # 模态选择（例如 RGB/IR 或 双时相图像）
#         x_out = x1 if self.out == 1 else x2
#
#         if self.use_fft:
#             x_out = x_out.to(torch.float32)  # torch.fft 不支持 half 精度
#             x_fft = torch.fft.fft2(x_out, norm='ortho')  # 傅里叶变换
#             x_mag = torch.abs(x_fft)  # 幅度谱
#             x_mag = x_mag / (x_mag.amax(dim=(-2, -1), keepdim=True) + 1e-8)  # 最大归一化
#
#             x_enhanced = self.freq_enhance(x_mag)  # 频域增强
#
#             # 融合方式改为 concat + 1x1卷积
#             fused = torch.cat([x_out, x_enhanced], dim=1)  # 通道维拼接
#             x_out = self.fuse_conv(fused)
#
#         return x_out
###有gflop版本
# class Multiinl(nn.Module):
#     def __init__(self, channels=3, use_fft=True):
#         super(Multiin, self).__init__()
#         self.channels = channels
#         self.use_fft = use_fft
#
#         if use_fft:
#             # 红外分支频域增强
#             self.ir_fft = nn.Sequential(
#                 nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
#                 nn.BatchNorm2d(channels),
#                 nn.ReLU(inplace=True)
#             )
#
#             # RGB 分支频域增强
#             self.rgb_fft = nn.Sequential(
#                 nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
#                 nn.BatchNorm2d(channels),
#                 nn.ReLU(inplace=True)
#             )
#         else:
#             # 空洞卷积作为结构增强（当不开启FFT）
#             self.ir_branch = nn.Sequential(
#                 nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, bias=False),
#                 nn.BatchNorm2d(channels),
#                 nn.ReLU(inplace=True)
#             )
#             self.rgb_branch = nn.Sequential(
#                 nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
#                 nn.BatchNorm2d(channels),
#                 nn.ReLU(inplace=True)
#             )
#
#         # 注意力融合
#         self.attn = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, channels // 4, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // 4, channels, 1, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         # x: [B, 6, H, W] → 分成 [B, 3, H, W] + [B, 3, H, W]
#         ir, rgb = x[:, :self.channels, :, :], x[:, self.channels:, :, :]
#
#         if self.use_fft:
#             # 红外傅里叶增强
#             ir = ir.to(torch.float32)
#             ir_fft = torch.fft.fft2(ir, norm='ortho')
#             ir_mag = torch.abs(ir_fft)
#             ir_mag = ir_mag / (ir_mag.amax(dim=(-2, -1), keepdim=True) + 1e-8)
#             ir_feat = self.ir_fft(ir_mag)
#
#             # RGB傅里叶增强
#             rgb = rgb.to(torch.float32)
#             rgb_fft = torch.fft.fft2(rgb, norm='ortho')
#             rgb_mag = torch.abs(rgb_fft)
#             rgb_mag = rgb_mag / (rgb_mag.amax(dim=(-2, -1), keepdim=True) + 1e-8)
#             rgb_feat = self.rgb_fft(rgb_mag)
#         else:
#             # 若不使用FFT，则走普通分支
#             ir_feat = self.ir_branch(ir)
#             rgb_feat = self.rgb_branch(rgb)
#
#         # 注意力融合
#         attn = self.attn(ir_feat + rgb_feat)
#         fused = ir_feat * (1 - attn) + rgb_feat * attn
#
#         return fused
# #

# class Multiin(nn.Module):
#     def __init__(self, c1=None, c2=None, out=1, use_fft=True, *args, **kwargs):
#         super().__init__()
#         if isinstance(out, bool):  # 错位参数修正
#             out, use_fft = 1, out
#         if out not in [1, 2]:
#             out = 1
#
#         self.out = out
#         self.use_fft = use_fft
#
#         # Laplacian高频边缘提取模块（非可学习）
#         laplace_kernel = torch.tensor([[0, 1, 0],
#                                        [1, -4, 1],
#                                        [0, 1, 0]], dtype=torch.float32).reshape(1, 1, 3, 3).repeat(3, 1, 1, 1)
#         self.laplace = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, groups=3)
#         self.laplace.weight.data = laplace_kernel
#         self.laplace.weight.requires_grad = False
#
#         # 动态系数生成模块（用于频域融合）
#         self.coeff_gen = nn.Sequential(
#             nn.Conv2d(3, 3, 1, bias=True),
#             nn.Sigmoid()
#         )
#
#         # 更深更强的融合卷积模块（代替原fuse_conv）
#         self.fuse_conv = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 3, kernel_size=1, bias=False)
#         )
#
#     def forward(self, x):
#         x_rgb, x_ir = x[:, :3], x[:, 3:]
#         x = x_rgb if self.out == 1 else x_ir
#
#         # 提取高频边缘信息
#         edge = self.laplace(x)
#
#         if self.use_fft:
#             x_fft = x.float()
#             fft = torch.fft.fft2(x_fft, norm='ortho')
#             fft_mag = torch.abs(fft)
#             fft_mag = torch.log(fft_mag + 1e-8)
#             fft_mag = (fft_mag - fft_mag.mean([2, 3], keepdim=True)) / (fft_mag.std([2, 3], keepdim=True) + 1e-5)
#
#             # 动态融合
#             coeff = self.coeff_gen(x)
#             x = coeff * x + (1 - coeff) * fft_mag
#
#         # 融合边缘细节（增强）
#         x = x + 0.1 * edge  # 控制边缘增强比例
#
#         # 融合卷积处理
#         x = x.to(self.fuse_conv[0].weight.dtype)
#         x = self.fuse_conv(x)
#         return x
######################动态系数fft********************


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class EFBlock(nn.Module):
    def __init__(self, c1, c2, reduction=16):
        super(EFBlock, self).__init__()
        self.mask_map_r = nn.Conv2d(c1 // 2, 1, 1, 1, 0, bias=True)
        self.mask_map_i = nn.Conv2d(c1 // 2, 1, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)
        self.bottleneck1 = nn.Conv2d(c1 // 2, c2 // 2, 3, 1, 1, bias=False)
        self.bottleneck2 = nn.Conv2d(c1 // 2, c2 // 2, 3, 1, 1, bias=False)
        self.se = SE_Block(c2, reduction)

    def forward(self, x):
        # print("EFBlock input:", x.shape)
        x_left_ori, x_right_ori = x[:, :3, :, :], x[:, 3:, :, :]
        x_left = x_left_ori * 0.5
        x_right = x_right_ori * 0.5

        x_mask_left = torch.mul(self.mask_map_r(x_left), x_left)
        x_mask_right = torch.mul(self.mask_map_i(x_right), x_right)

        out_IR = self.bottleneck1(x_mask_right + x_right_ori)
        out_RGB = self.bottleneck2(x_mask_left + x_left_ori)  # RGB
        out = self.se(torch.cat([out_RGB, out_IR], 1))

        return out


##################################CPCA Attention#########################################

class FeatureAdd(nn.Module):
    #  x + CPCA
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.add(x[0], x[1])


class CPCA_ChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(CPCA_ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class CPCA(nn.Module):
    def __init__(self, channels, out_channels, channelAttention_reduce=4):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(1, 1), padding=0)
        self.ca = CPCA_ChannelAttention(input_channels=channels, internal_neurons=channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.dconv1_7 = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.dconv7_1 = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.dconv1_11 = nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels)
        self.dconv11_1 = nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels)
        self.dconv1_21 = nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10), groups=channels)
        self.dconv21_1 = nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0), groups=channels)
        self.conv2 = nn.Conv2d(channels, out_channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, x):
        inputs = torch.cat((x[0], x[1]), dim=1)
        #   Global Perceptron
        inputs = self.conv1(inputs)
        inputs = self.act(inputs)

        inputs = self.ca(inputs)

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv1(x)
        out = spatial_att * inputs
        out = self.conv2(out)
        return out


##################################Transformer############################################
# 多头交叉注意力机制
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        assert (self.head_dim * num_heads == model_dim), "model_dim must be divisible by num_heads"

        self.query_vis = nn.Linear(model_dim, model_dim)
        self.key_vis = nn.Linear(model_dim, model_dim)
        self.value_vis = nn.Linear(model_dim, model_dim)

        self.query_inf = nn.Linear(model_dim, model_dim)
        self.key_inf = nn.Linear(model_dim, model_dim)
        self.value_inf = nn.Linear(model_dim, model_dim)

        self.fc_out_vis = nn.Linear(model_dim, model_dim)
        self.fc_out_inf = nn.Linear(model_dim, model_dim)

    def forward(self, vis, inf):
        batch_size, seq_length, model_dim = vis.shape

        # vis -> Q, K, V
        Q_vis = self.query_vis(vis)
        K_vis = self.key_vis(vis)
        V_vis = self.value_vis(vis)

        # inf -> Q, K, V
        Q_inf = self.query_inf(inf)
        K_inf = self.key_inf(inf)
        V_inf = self.value_inf(inf)

        # Reshape for multi-head attention
        Q_vis = Q_vis.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,
                                                                                            2)  # B, N, C --> B, n_h, N, d_h
        K_vis = K_vis.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V_vis = V_vis.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        Q_inf = Q_inf.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K_inf = K_inf.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V_inf = V_inf.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Cross attention via fused scaled_dot_product_attention (更省显存)
        out_inf = F.scaled_dot_product_attention(Q_vis, K_inf, V_inf, attn_mask=None, dropout_p=0.0, is_causal=False)
        out_vis = F.scaled_dot_product_attention(Q_inf, K_vis, V_vis, attn_mask=None, dropout_p=0.0, is_causal=False)

        # Concatenate and project back to the original dimension
        out_vis = out_vis.transpose(1, 2).contiguous().view(batch_size, seq_length, model_dim)
        out_inf = out_inf.transpose(1, 2).contiguous().view(batch_size, seq_length, model_dim)

        # out 的形状为 (batch_size, seq_length, model_dim)
        out_vis = self.fc_out_vis(out_vis)
        out_inf = self.fc_out_inf(out_inf)

        return out_vis, out_inf


# 前向全连接网络
class FeedForward(nn.Module):
    def __init__(self, model_dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(model_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout, max_len=6400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个从0到max_len-1的列向量，形状为 (max_len, 1)
        position = torch.arange(0, max_len).unsqueeze(1)
        # 计算用于位置编码的除数项
        div_term = torch.exp(torch.arange(0, model_dim, 2) * -(torch.log(torch.tensor(10000.0)) / model_dim))

        pe = torch.zeros(max_len, model_dim)  # 初始化一个位置编码矩阵，形状为 (max_len, model_dim)，所有元素初始化为0
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列使用sin函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列使用cos函数

        pe = pe.unsqueeze(0)  # 在位置编码矩阵的第一个维度前添加一个新的维度，变为 (1, max_len, model_dim)
        self.register_buffer('pe', pe)  # 将位置编码矩阵 pe 注册为模型的一个缓冲区。缓冲区类似于模型参数，但在训练过程中不会更新

    def forward(self, x):
        # 动态生成与输入长度/设备/精度匹配的位置编码，避免 max_len 限制
        # x: (batch, seq_len, model_dim)
        seq_len = x.size(1)
        model_dim = x.size(-1)
        device = x.device
        dtype = x.dtype

        position = torch.arange(0, seq_len, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2, device=device, dtype=dtype)
            * -(torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / model_dim)
        )
        pe = torch.zeros(1, seq_len, model_dim, device=device, dtype=dtype)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        x = x + pe
        return self.dropout(x)


# 编码器层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.cross_attention = MultiHeadCrossAttention(model_dim, num_heads)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ff = FeedForward(model_dim, hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, vis, inf):
        attn_out_vis, attn_out_inf = self.cross_attention(vis, inf)
        vis = self.norm1(vis + attn_out_vis)
        inf = self.norm1(inf + attn_out_inf)

        ff_out_vis = self.ff(vis)
        ff_out_inf = self.ff(inf)

        vis = self.norm2(vis + ff_out_vis)
        inf = self.norm2(inf + ff_out_inf)

        return vis, inf


# Transformer编码器
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, hidden_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(model_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, vis, inf):
        vis = self.embedding(vis) * torch.sqrt(torch.tensor(self.embedding.out_features, dtype=torch.float32))
        inf = self.embedding(inf) * torch.sqrt(torch.tensor(self.embedding.out_features, dtype=torch.float32))

        vis = self.positional_encoding(vis)
        inf = self.positional_encoding(inf)

        for layer in self.layers:
            vis, inf = layer(vis, inf)

        return vis, inf


# 定义用于交叉注意力的网络
class CrossTransformerFusion(nn.Module):
    def __init__(self, input_dim, num_heads=2, num_layers=1, dropout=0.1, max_tokens=2048, use_checkpoint=True):
        super(CrossTransformerFusion, self).__init__()
        self.hidden_dim = input_dim * 2
        self.model_dim = input_dim
        self.encoder = TransformerEncoder(input_dim, self.model_dim, num_heads, num_layers, self.hidden_dim, dropout)
        # 限制注意力的序列长度，超出则先进行平均池化降采样
        self.max_tokens = max_tokens
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        vis, inf = x[0], x[1]
        # 输入形状为 B, C, H, W
        B, C, H, W = vis.shape

        # 自适应下采样，限制序列长度 N=H*W <= max_tokens
        pool_stride = 1
        N = H * W
        if N > self.max_tokens:
            # 计算等比下采样步长，使得下采样后 H'*W' <= max_tokens
            # s = ceil(sqrt(N / max_tokens))
            s = int(math.ceil(math.sqrt(N / float(self.max_tokens))))
            pool_stride = max(1, s)
            if pool_stride > 1:
                vis = F.avg_pool2d(vis, kernel_size=pool_stride, stride=pool_stride, ceil_mode=True)
                inf = F.avg_pool2d(inf, kernel_size=pool_stride, stride=pool_stride, ceil_mode=True)

        H2, W2 = vis.shape[-2], vis.shape[-1]

        # 将输入变形为 B, H*W, C
        vis = vis.permute(0, 2, 3, 1).reshape(B, -1, C)
        inf = inf.permute(0, 2, 3, 1).reshape(B, -1, C)

        # 输入Transformer编码器（训练时启用梯度检查点以节省显存）
        if self.training and self.use_checkpoint:
            import torch.utils.checkpoint as cp
            vis_out, inf_out = cp.checkpoint(lambda a, b: self.encoder(a, b), vis, inf)
        else:
            vis_out, inf_out = self.encoder(vis, inf)

        # 将输出变形为 B, C, H2, W2（注意可能是下采样分辨率）
        vis_out = vis_out.view(B, H2, W2, -1).permute(0, 3, 1, 2)
        inf_out = inf_out.view(B, H2, W2, -1).permute(0, 3, 1, 2)

        # 若进行了下采样，则上采样回原分辨率
        if (H2 != H) or (W2 != W):
            vis_out = F.interpolate(vis_out, size=(H, W), mode='bilinear', align_corners=False)
            inf_out = F.interpolate(inf_out, size=(H, W), mode='bilinear', align_corners=False)

        # 在通道维度上进行级联
        out = torch.cat((vis_out, inf_out), dim=1)

        return out


import torch
import torch.nn as nn
from einops import rearrange


# 定义函数，将四维张量重排为三维
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')  # 将输入的四维张量重排为三维，合并高度和宽度维度


# 定义函数，将三维张量重排为四维
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)  # 将三维张量恢复为四维，并重排维度


# PixelAttention类，用于计算像素级别的注意力
class PixelAttention(nn.Module):
    def __init__(self, dim):  # 初始化PixelAttention模块，dim为输入特征的通道数
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)  # 定义卷积层
        self.sigmoid = nn.Sigmoid()  # 定义Sigmoid激活函数，用于生成注意力图

    # 前向传播函数
    def forward(self, x, pattn1):
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W  # 在通道维度插入一个维度
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W  # 同样在pattn1插入一个维度
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W  # 沿着通道维度将x和pattn1拼接
        x2 = rearrange(x2, 'b c t h w -> b (c t) h w')  # 重排张量维度为 [b, (c * t), h, w]
        pattn2 = self.pa2(x2)  # 经过卷积层计算注意力图
        pattn2 = self.sigmoid(pattn2)  # 通过Sigmoid归一化注意力图
        return pattn2  # 返回计算得到的注意力图


# 定义QKV_block类，用于计算多种池化操作后的特征
# class QKV_block(nn.Module):
#     def __init__(self, in_channels):
#         super(QKV_block, self).__init__()
#         self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)  # 定义2x2池化层
#         self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)  # 定义3x3池化层
#         self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)  # 定义5x5池化层
#         self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)  # 定义6x6池化层
#
#     # 前向传播函数
#     def forward(self, x):
#         b, c, h, w = x.size()  # 获取输入x的形状，b为batch_size，c为通道数，h和w为高度和宽度
#         pool_1 = self.pool1(x).view(b, c, -1)  # 对输入x进行2x2池化，并将输出展平
#         pool_2 = self.pool2(x).view(b, c, -1)  # 对输入x进行3x3池化，并将输出展平
#         pool_3 = self.pool3(x).view(b, c, -1)  # 对输入x进行5x5池化，并将输出展平
#         pool_4 = self.pool4(x).view(b, c, -1)  # 对输入x进行6x6池化，并将输出展平
#         pool_cat = torch.cat([pool_1, pool_2, pool_3, pool_4], -1)  # 将所有池化结果拼接
#         out = pool_cat.permute(0, 2, 1)  # 转置张量的维度为 [B,C,L]
#         return out  # 返回计算结果
#
# # 定义通道洗牌函数，用于打乱通道顺序
# def channel_shuffle(x, groups):
#     batchsize, num_channels, height, width = x.data.size()  # 获取输入张量的形状
#     channels_per_group = num_channels // groups  # 计算每组的通道数
#     x = x.view(batchsize, groups, channels_per_group, height, width)  # 重新调整张量的形状
#     x = torch.transpose(x, 1, 2).contiguous()  # 转置通道组
#     x = x.view(batchsize, -1, height, width)  # 展平通道并恢复张量的形状
#     return x  # 返回洗牌后的张量

# 定义GLFA类，用于全局与局部特征的聚合
# class GLFA(nn.Module):
#     def __init__(self, in_channels):  # 简化参数，只需要输入通道数
#         super(GLFA, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = in_channels
#         self.conv_1 = nn.Sequential(  # 定义第一层卷积层
#             nn.Conv2d(in_channels, in_channels, padding=1, kernel_size=3, dilation=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.conv_2 = nn.Sequential(  # 第二个卷积层
#             nn.Conv2d(in_channels, in_channels, padding=2, kernel_size=3, dilation=2),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.conv_3 = nn.Sequential(  # 第三个卷积层
#             nn.Conv2d(in_channels, in_channels, padding=3, kernel_size=3, dilation=3),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.conv_4 = nn.Sequential(  # 第四个卷积层
#             nn.Conv2d(in_channels, in_channels, padding=4, kernel_size=3, dilation=4),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.fuse = nn.Sequential(  # 定义1x1卷积用于特征融合,调整通道数
#             nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, padding=0),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.W = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0)  # 1x1卷积调整输出通道
#
#         self.SP_Pool_v = QKV_block(self.in_channels)  # 定义QKV块，用于计算值（V）
#         self.SP_Pool_k = QKV_block(self.in_channels)  # 定义QKV块，用于计算键（K）
#         nn.init.constant_(self.W.weight, 0)  # 初始化W卷积层的权重为0
#         nn.init.constant_(self.W.bias, 0)  # 初始化W卷积层的偏置为0
#     # 前向传播函数
#     def forward(self, x):
#         # 捕捉局部特征
#         c1 = self.conv_1(x)  # 经过第一个卷积层
#         c2 = self.conv_2(x)  # 经过第二个卷积层
#         c3 = self.conv_3(x)  # 经过第三个卷积层
#         c4 = self.conv_4(x)  # 经过第四个卷积层
#         cat = torch.cat([c1, c2, c3, c4], dim=1)  # 将四个特征拼接
#         fuse_out = self.fuse(cat)  # 融合特征
#         out = self.W(fuse_out)  # 最后的1x1卷积层输出
#         return out  # 返回最终输出

# MSAGF模块的定义：结合GLFA（全局与局部特征融合）与PixelAttention（像素级注意力机制）
class MSAGF(nn.Module):
    def __init__(self, in_dim, out_dim, *args):  # 初始化MSAGF模块，dim为输入特征的通道数
        super(MSAGF, self).__init__()
        self.GLFA = GLFA(in_dim)  # 初始化GLFA模块
        self.PixelAttention = PixelAttention(in_dim)  # 初始化PixelAttention模块
        self.conv = nn.Conv2d(in_dim, out_dim, 1, bias=True)  # 1x1卷积层，用于调整输出通道数
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

    # 前向传播函数
    def forward(self, x, y):
        initial = x + y  # 将输入x与y相加，作为初始特征
        pattn1 = self.GLFA(initial)  # 通过GLFA模块提取全局与局部特征
        pattn2 = self.sigmoid(self.PixelAttention(initial, pattn1))  # 通过PixelAttention计算像素级注意力
        result = initial + pattn2 * x + (1 - pattn2) * y  # 根据注意力图对x和y进行加权融合
        result = self.conv(result)  # 通过1x1卷积调整通道数
        return result  # 返回融合后的结果


class AKConv(nn.Module):
    def __init__(self, inc, outc, num_param=5, stride=1, bias=None):
        super(AKConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
                                  nn.BatchNorm2d(outc),
                                  nn.SiLU())  # the conv adds the BN and SiLU to compare original Conv in YOLOv5.
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # print(f"[AKConv] x dtype: {x.dtype}, conv weight dtype: {self.conv[0].weight.dtype}")  # 调试用
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # generating the inital sampled shapes for the AKConv with different sizes.
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0, base_int))
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1),
                torch.arange(0, mod_number))

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        # using Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        # using 1 × 1 Conv
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)
        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)

        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset


# 测试代码
if __name__ == '__main__':
    block = MSAGF(32, 32)  # 创建MSAGF模块实例，输入通道数in_dim=32，输出通道数out_dim=32
    input1 = torch.rand(1, 32, 64, 64)  # 生成一个随机输入张量input1，大小为(1, 32, 64, 64)
    input2 = torch.rand(1, 32, 64, 64)  # 生成一个随机输入张量input2，大小为(1, 32, 64, 64)
    output = block(input1, input2)  # 将input1和input2传入MSAGF模块进行前向传播
    print('input1_size:', input1.size())  # 打印input1的尺寸
    print('output_size:', output.size())  # 打印输出的尺寸

import torch
import torch.nn as nn


# 通道注意力模块 (Channel Attention Block)
class CAB(nn.Module):
    def __init__(self, in_planes, ratio=16):  # 初始化 CAB 类，输入通道数和比例（默认为 16）
        super(CAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 定义全局平均池化，输出大小为 1x1
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 定义全局最大池化，输出大小为 1x1

        # 定义通道注意力机制中的两层卷积
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # 第一层卷积，减少通道数
        self.relu1 = nn.ReLU()  # ReLU 激活函数
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # 第二层卷积，恢复通道数
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数，用于输出权重

    def forward(self, x):  # 定义前向传播方法
        # 分别使用平均池化和最大池化进行处理
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 使用平均池化，然后两层卷积
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 使用最大池化，然后两层卷积

        out = avg_out + max_out  # 将两者相加，融合通道信息
        return self.sigmoid(out)  # 对结果进行 Sigmoid 激活，生成通道注意力


# 空间注意力模块 (Spatial Attention Block)
class SAB(nn.Module):
    def __init__(self, kernel_size=7):  # 初始化 SAB 类，默认为 7x7 卷积核
        super(SAB, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)  # 卷积操作，输入通道为 2，输出通道为 1
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数

    def forward(self, x):  # 定义前向传播方法
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 沿着通道维度求平均值，保持维度
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 沿着通道维度求最大值，保持维度

        x = torch.cat([avg_out, max_out], dim=1)  # 将平均池化和最大池化的结果拼接起来
        x = self.conv1(x)  # 通过卷积提取空间特征
        return self.sigmoid(x)  # 使用 Sigmoid 激活，生成空间注意力


# CMEA (卷积多尺度增强注意力模块)
class CMEA(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16):  # 初始化 CMEA 类，输入通道数和比例（默认为 16）
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels

        # 定义深度可分离卷积（DWConv）模块，用于多尺度卷积
        self.dwconvs = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 3 // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU6(inplace=True))

        # 定义 1x1 卷积，用于通道融合
        self.conv1X1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 定义不同尺度的深度可分离卷积
        self.dwconv3X3 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 3 // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU6(inplace=True))
        self.dwconv5X5 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 5, 1, 5 // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU6(inplace=True))
        self.dwconv7X7 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 7, 1, 7 // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU6(inplace=True))
        self.dwconv9X9 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 9, 1, 9 // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU6(inplace=True))

        # 定义通道注意力机制
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 确保 fc1 的输入和输出通道数正确
        reduced_channels = max(in_channels // ratio, 1)
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, 1, bias=False)

        # 定义空间注意力机制
        self.conv1 = nn.Conv2d(2, 1, 7, padding=7 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 如果输出通道数与输入不同，添加一个投影层
        if self.out_channels != self.in_channels:
            self.projection = nn.Conv2d(in_channels, self.out_channels, 1, bias=False)
        else:
            self.projection = nn.Identity()

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x  # 保存输入用于残差连接

        # 多尺度卷积
        x = self.dwconv3X3(x)
        x = x + self.dwconv5X5(x) + self.dwconv7X7(x) + self.dwconv9X9(x)
        x = self.conv1X1(x)

        # 计算通道注意力
        avg_out1 = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out1 = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out1 = avg_out1 + max_out1
        c_attention = self.sigmoid(out1)

        # 计算空间注意力
        avg_out2 = torch.mean(x, dim=1, keepdim=True)
        max_out2, _ = torch.max(x, dim=1, keepdim=True)
        out2 = torch.cat([avg_out2, max_out2], dim=1)
        out2 = self.conv1(out2)
        s_attention = self.sigmoid(out2)

        # 应用注意力机制
        x = x * c_attention * s_attention

        # 残差连接
        x = x + identity

        # 投影到目标通道数
        x = self.projection(x)

        return x


# 输入 B C H W,  输出 B C H W
if __name__ == '__main__':
    model = CMEA(out_channels=32)  # 实例化 CMEA 模型，输入通道数为 32
    input = torch.randn(1, 32, 64, 64)  # 生成随机输入张量，形状为 [1, 32, 64, 64]
    output = model(input)  # 执行前向传播
    print('input_size:', input.size())  # 打印输入张量的形状
    print('output_size:', output.size())  # 打印输出张量的形状

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return (x, x)


class StarReLU(nn.Module):
    """StarReLU: s * relu(x) ** 2 + b"""

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, bias=False):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class DynamicFilter(nn.Module):
    def __init__(self, dim, expansion_ratio=2, num_filters=4, size=14, weight_resize=True):
        super().__init__()
        self.dim = dim
        self.num_filters = num_filters
        self.weight_resize = weight_resize

        # Channel attention with dimension reduction and recovery
        reduced_dim = max(dim // 4, 8)  # Ensure minimum channel size
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, reduced_dim, 1, bias=False),  # 修改输入通道为 dim * 2
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, dim * 2, 1, bias=False),  # 修改输出通道为 dim * 2
            nn.Sigmoid()
        )

        # Spatial attention
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        # FFT related layers with channel preservation
        self.fft_bn = nn.BatchNorm2d(dim * 2)  # 修改为 dim * 2
        self.fft_conv = nn.Conv2d(dim * 2, dim * 2, 1, bias=False)  # 修改为 dim * 2

        # Intermediate processing
        self.process_conv = nn.Conv2d(dim * 2, dim * 2, 1, bias=False)  # 修改为 dim * 2
        self.process_bn = nn.BatchNorm2d(dim * 2)  # 修改为 dim * 2

        # Output projection to match input channels
        self.out_conv = nn.Conv2d(dim * 2, dim * 2, 1, bias=False)  # 修改为 dim * 2

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x

        # Channel attention with dimension reduction
        ca_weights = self.ca(x)
        x = x * ca_weights

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.sa(torch.cat([avg_out, max_out], dim=1))
        x = x * sa

        # FFT processing with channel preservation
        x = self.process_conv(x)
        x = self.process_bn(x)

        x_freq = torch.fft.rfft2(x.float(), dim=(-2, -1), norm='ortho')
        x_freq_abs = torch.abs(x_freq)
        x_freq_abs = x_freq_abs / (x_freq_abs.amax(dim=(-2, -1), keepdim=True) + 1e-8)
        x_freq_filtered = x_freq * x_freq_abs

        x = torch.fft.irfft2(x_freq_filtered, s=(H, W), dim=(-2, -1), norm='ortho')
        x = self.fft_bn(x)
        x = self.fft_conv(x)

        # Final processing with residual connection
        x = self.out_conv(x)

        return x + identity


class DynamicFilterBlock(nn.Module):
    def __init__(self, in_channels, debug=False):
        super().__init__()
        self.in_channels = in_channels
        # 添加通道调整层
        self.channel_adj = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)
        self.dynamic_filter = DynamicFilter(in_channels)
        self.act = nn.SiLU()
        self.debug = debug
        print(f"[DynamicFilterBlock] Initialized with in_channels={in_channels}")

    def _print_tensor_info(self, tensor, name):
        """Helper function to print tensor statistics"""
        if not self.debug:
            return

        print(f"\n{'=' * 50}")
        print(f"[DynamicFilterBlock] {name} statistics:")
        print(f"Shape: {tensor.shape}")
        print(f"Dtype: {tensor.dtype}")
        print(f"Device: {tensor.device}")
        print(f"Min: {tensor.min().item():.6f}")
        print(f"Max: {tensor.max().item():.6f}")
        print(f"Mean: {tensor.mean().item():.6f}")
        print(f"Std: {tensor.std().item():.6f}")
        print(f"Has NaN: {torch.isnan(tensor).any().item()}")
        print(f"Has Inf: {torch.isinf(tensor).any().item()}")
        print(f"{'=' * 50}\n")

    def forward(self, x):
        try:
            if self.debug:
                print("\n[DynamicFilterBlock] Forward pass started")

            # Check input
            self._print_tensor_info(x, "Input")

            # Save identity branch
            identity = x
            if self.debug:
                print("[DynamicFilterBlock] Identity branch saved")

            # 调整通道数
            if x.size(1) != self.in_channels:
                if self.debug:
                    print(f"[DynamicFilterBlock] Adjusting channels from {x.size(1)} to {self.in_channels}")
                x = self.channel_adj(x)
                self._print_tensor_info(x, "After channel adjustment")

            # Dynamic filter processing
            try:
                x = self.dynamic_filter(x)
                if self.debug:
                    print("[DynamicFilterBlock] Dynamic filter processing completed")
                self._print_tensor_info(x, "After DynamicFilter")
            except Exception as e:
                print(f"[DynamicFilterBlock] Error in dynamic_filter: {str(e)}")
                raise

            # Residual connection
            try:
                if x.shape != identity.shape:
                    print(f"[DynamicFilterBlock] WARNING: Shape mismatch in residual connection")
                    print(f"x shape: {x.shape}, identity shape: {identity.shape}")

                output = x + identity
                if self.debug:
                    print("[DynamicFilterBlock] Residual connection added")
                self._print_tensor_info(output, "After residual connection")
            except Exception as e:
                print(f"[DynamicFilterBlock] Error in residual connection: {str(e)}")
                raise

            # Activation
            try:
                output = self.act(output)
                if self.debug:
                    print("[DynamicFilterBlock] Activation applied")
                self._print_tensor_info(output, "Final output")
            except Exception as e:
                print(f"[DynamicFilterBlock] Error in activation: {str(e)}")
                raise

            if self.debug:
                print("[DynamicFilterBlock] Forward pass completed successfully")

            return output

        except Exception as e:
            print(f"[DynamicFilterBlock] Error in forward pass: {str(e)}")
            raise

    def extra_repr(self) -> str:
        """Extra representation of module for print() and str()"""
        return f'in_channels={self.in_channels}, debug={self.debug}'


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class EFBlock(nn.Module):
    def __init__(self, c1, c2, reduction=16):
        super(EFBlock, self).__init__()
        self.mask_map_r = nn.Conv2d(c1 // 2, 1, 1, 1, 0, bias=True)
        self.mask_map_i = nn.Conv2d(c1 // 2, 1, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)
        self.bottleneck1 = nn.Conv2d(c1 // 2, c2 // 2, 3, 1, 1, bias=False)
        self.bottleneck2 = nn.Conv2d(c1 // 2, c2 // 2, 3, 1, 1, bias=False)
        self.se = SE_Block(c2, reduction)

    def forward(self, x):
        # print("EFBlock input:", x.shape)
        x_left_ori, x_right_ori = x[:, :3, :, :], x[:, 3:, :, :]
        x_left = x_left_ori * 0.5
        x_right = x_right_ori * 0.5

        x_mask_left = torch.mul(self.mask_map_r(x_left), x_left)
        x_mask_right = torch.mul(self.mask_map_i(x_right), x_right)

        out_IR = self.bottleneck1(x_mask_right + x_right_ori)
        out_RGB = self.bottleneck2(x_mask_left + x_left_ori)  # RGB
        out = self.se(torch.cat([out_RGB, out_IR], 1))

        return out


##################################CPCA Attention#########################################

class FeatureAdd(nn.Module):
    #  x + CPCA
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.add(x[0], x[1])


class CPCA_ChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(CPCA_ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class CPCA(nn.Module):
    def __init__(self, channels, out_channels, channelAttention_reduce=4):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(1, 1), padding=0)
        self.ca = CPCA_ChannelAttention(input_channels=channels, internal_neurons=channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.dconv1_7 = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.dconv7_1 = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.dconv1_11 = nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels)
        self.dconv11_1 = nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels)
        self.dconv1_21 = nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10), groups=channels)
        self.dconv21_1 = nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0), groups=channels)
        self.conv2 = nn.Conv2d(channels, out_channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, x):
        inputs = torch.cat((x[0], x[1]), dim=1)
        #   Global Perceptron
        inputs = self.conv1(inputs)
        inputs = self.act(inputs)

        inputs = self.ca(inputs)

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv1(x)
        out = spatial_att * inputs
        out = self.conv2(out)
        return out


##################################Transformer############################################
# 多头交叉注意力机制
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        assert (self.head_dim * num_heads == model_dim), "model_dim must be divisible by num_heads"

        self.query_vis = nn.Linear(model_dim, model_dim)
        self.key_vis = nn.Linear(model_dim, model_dim)
        self.value_vis = nn.Linear(model_dim, model_dim)

        self.query_inf = nn.Linear(model_dim, model_dim)
        self.key_inf = nn.Linear(model_dim, model_dim)
        self.value_inf = nn.Linear(model_dim, model_dim)

        self.fc_out_vis = nn.Linear(model_dim, model_dim)
        self.fc_out_inf = nn.Linear(model_dim, model_dim)

    def forward(self, vis, inf):
        batch_size, seq_length, model_dim = vis.shape

        # vis -> Q, K, V
        Q_vis = self.query_vis(vis)
        K_vis = self.key_vis(vis)
        V_vis = self.value_vis(vis)

        # inf -> Q, K, V
        Q_inf = self.query_inf(inf)
        K_inf = self.key_inf(inf)
        V_inf = self.value_inf(inf)

        # Reshape for multi-head attention
        Q_vis = Q_vis.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,
                                                                                            2)  # B, N, C --> B, n_h, N, d_h
        K_vis = K_vis.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V_vis = V_vis.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        Q_inf = Q_inf.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K_inf = K_inf.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V_inf = V_inf.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Cross attention: vis Q with inf K and inf Q with vis K
        # Q_vis 的形状为 (batch_size, num_heads, seq_length, head_dim)
        # K_inf 的形状为 (batch_size, num_heads, head_dim, seq_length)
        # 矩阵乘法后，scores_vis_inf 的形状为 (batch_size, num_heads, seq_length, seq_length)
        scores_vis_inf = torch.matmul(Q_vis, K_inf.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))
        scores_inf_vis = torch.matmul(Q_inf, K_vis.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))

        attention_inf = torch.softmax(scores_vis_inf, dim=-1)
        attention_vis = torch.softmax(scores_inf_vis, dim=-1)

        # attention_vis_inf 的形状为 (batch_size, num_heads, seq_length, seq_length)
        # V_inf 的形状为 (batch_size, num_heads, seq_length, head_dim)
        # out_vis_inf 的形状为 (batch_size, num_heads, seq_length, head_dim)
        out_inf = torch.matmul(attention_inf, V_inf)
        out_vis = torch.matmul(attention_vis, V_vis)

        # Concatenate and project back to the original dimension
        out_vis = out_vis.transpose(1, 2).contiguous().view(batch_size, seq_length, model_dim)
        out_inf = out_inf.transpose(1, 2).contiguous().view(batch_size, seq_length, model_dim)

        # out 的形状为 (batch_size, seq_length, model_dim)
        out_vis = self.fc_out_vis(out_vis)
        out_inf = self.fc_out_inf(out_inf)

        return out_vis, out_inf


# 前向全连接网络
class FeedForward(nn.Module):
    def __init__(self, model_dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(model_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout, max_len=6400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个从0到max_len-1的列向量，形状为 (max_len, 1)
        position = torch.arange(0, max_len).unsqueeze(1)
        # 计算用于位置编码的除数项
        div_term = torch.exp(torch.arange(0, model_dim, 2) * -(torch.log(torch.tensor(10000.0)) / model_dim))

        pe = torch.zeros(max_len, model_dim)  # 初始化一个位置编码矩阵，形状为 (max_len, model_dim)，所有元素初始化为0
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列使用sin函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列使用cos函数

        pe = pe.unsqueeze(0)  # 在位置编码矩阵的第一个维度前添加一个新的维度，变为 (1, max_len, model_dim)
        self.register_buffer('pe', pe)  # 将位置编码矩阵 pe 注册为模型的一个缓冲区。缓冲区类似于模型参数，但在训练过程中不会更新

    def forward(self, x):
        # 动态生成与输入长度/设备/精度匹配的位置编码，避免 max_len 限制
        # x: (batch, seq_len, model_dim)
        seq_len = x.size(1)
        model_dim = x.size(-1)
        device = x.device
        dtype = x.dtype

        position = torch.arange(0, seq_len, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2, device=device, dtype=dtype)
            * -(torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / model_dim)
        )
        pe = torch.zeros(1, seq_len, model_dim, device=device, dtype=dtype)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        x = x + pe
        return self.dropout(x)


# 编码器层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.cross_attention = MultiHeadCrossAttention(model_dim, num_heads)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ff = FeedForward(model_dim, hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, vis, inf):
        attn_out_vis, attn_out_inf = self.cross_attention(vis, inf)
        vis = self.norm1(vis + attn_out_vis)
        inf = self.norm1(inf + attn_out_inf)

        ff_out_vis = self.ff(vis)
        ff_out_inf = self.ff(inf)

        vis = self.norm2(vis + ff_out_vis)
        inf = self.norm2(inf + ff_out_inf)

        return vis, inf


# Transformer编码器
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, hidden_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(model_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, vis, inf):
        vis = self.embedding(vis) * torch.sqrt(torch.tensor(self.embedding.out_features, dtype=torch.float32))
        inf = self.embedding(inf) * torch.sqrt(torch.tensor(self.embedding.out_features, dtype=torch.float32))

        vis = self.positional_encoding(vis)
        inf = self.positional_encoding(inf)

        for layer in self.layers:
            vis, inf = layer(vis, inf)

        return vis, inf


# 定义用于交叉注意力的网络
class CrossTransformerFusion(nn.Module):
    def __init__(self, input_dim, num_heads=2, num_layers=1, dropout=0.1):
        super(CrossTransformerFusion, self).__init__()
        self.hidden_dim = input_dim * 2
        self.model_dim = input_dim
        self.encoder = TransformerEncoder(input_dim, self.model_dim, num_heads, num_layers, self.hidden_dim, dropout)

    def forward(self, x):
        vis, inf = x[0], x[1]
        # 输入形状为 B, C, H, W
        B, C, H, W = vis.shape

        # 将输入变形为 B, H*W, C
        vis = vis.permute(0, 2, 3, 1).reshape(B, -1, C)
        inf = inf.permute(0, 2, 3, 1).reshape(B, -1, C)

        # 输入Transformer编码器
        vis_out, inf_out = self.encoder(vis, inf)

        # 将输出变形为 B, C, H, W
        vis_out = vis_out.view(B, H, W, -1).permute(0, 3, 1, 2)
        inf_out = inf_out.view(B, H, W, -1).permute(0, 3, 1, 2)

        # 在通道维度上进行级联
        out = torch.cat((vis_out, inf_out), dim=1)

        return out


import torch
import torch.nn as nn
from einops import rearrange


# 定义函数，将四维张量重排为三维
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')  # 将输入的四维张量重排为三维，合并高度和宽度维度


# 定义函数，将三维张量重排为四维
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)  # 将三维张量恢复为四维，并重排维度


# PixelAttention类，用于计算像素级别的注意力
class PixelAttention(nn.Module):
    def __init__(self, dim):  # 初始化PixelAttention模块，dim为输入特征的通道数
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)  # 定义卷积层
        self.sigmoid = nn.Sigmoid()  # 定义Sigmoid激活函数，用于生成注意力图

    # 前向传播函数
    def forward(self, x, pattn1):
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W  # 在通道维度插入一个维度
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W  # 同样在pattn1插入一个维度
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W  # 沿着通道维度将x和pattn1拼接
        x2 = rearrange(x2, 'b c t h w -> b (c t) h w')  # 重排张量维度为 [b, (c * t), h, w]
        pattn2 = self.pa2(x2)  # 经过卷积层计算注意力图
        pattn2 = self.sigmoid(pattn2)  # 通过Sigmoid归一化注意力图
        return pattn2  # 返回计算得到的注意力图


# 定义QKV_block类，用于计算多种池化操作后的特征
# class QKV_block(nn.Module):
#     def __init__(self, in_channels):
#         super(QKV_block, self).__init__()
#         self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)  # 定义2x2池化层
#         self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)  # 定义3x3池化层
#         self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)  # 定义5x5池化层
#         self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)  # 定义6x6池化层
#
#     # 前向传播函数
#     def forward(self, x):
#         b, c, h, w = x.size()  # 获取输入x的形状，b为batch_size，c为通道数，h和w为高度和宽度
#         pool_1 = self.pool1(x).view(b, c, -1)  # 对输入x进行2x2池化，并将输出展平
#         pool_2 = self.pool2(x).view(b, c, -1)  # 对输入x进行3x3池化，并将输出展平
#         pool_3 = self.pool3(x).view(b, c, -1)  # 对输入x进行5x5池化，并将输出展平
#         pool_4 = self.pool4(x).view(b, c, -1)  # 对输入x进行6x6池化，并将输出展平
#         pool_cat = torch.cat([pool_1, pool_2, pool_3, pool_4], -1)  # 将所有池化结果拼接
#         out = pool_cat.permute(0, 2, 1)  # 转置张量的维度为 [B,C,L]
#         return out  # 返回计算结果
#
# # 定义通道洗牌函数，用于打乱通道顺序
# def channel_shuffle(x, groups):
#     batchsize, num_channels, height, width = x.data.size()  # 获取输入张量的形状
#     channels_per_group = num_channels // groups  # 计算每组的通道数
#     x = x.view(batchsize, groups, channels_per_group, height, width)  # 将通道维度重塑为(groups, channels_per_group)
#     x = torch.transpose(x, 1, 2).contiguous()  # 交换第1维和第2维，通道组与每组内的通道进行交换
#     x = x.view(batchsize, -1, height, width)  # 将混排后的张量展平为(b, num_channels, height, width)
#     return x  # 返回混排后的结果

# 定义GLFA类，用于全局与局部特征的聚合
# class GLFA(nn.Module):
#     def __init__(self, in_channels):  # 简化参数，只需要输入通道数
#         super(GLFA, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = in_channels
#         self.conv_1 = nn.Sequential(  # 定义第一层卷积层
#             nn.Conv2d(in_channels, in_channels, padding=1, kernel_size=3, dilation=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.conv_2 = nn.Sequential(  # 第二个卷积层
#             nn.Conv2d(in_channels, in_channels, padding=2, kernel_size=3, dilation=2),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.conv_3 = nn.Sequential(  # 第三个卷积层
#             nn.Conv2d(in_channels, in_channels, padding=3, kernel_size=3, dilation=3),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.conv_4 = nn.Sequential(  # 第四个卷积层
#             nn.Conv2d(in_channels, in_channels, padding=4, kernel_size=3, dilation=4),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.fuse = nn.Sequential(  # 定义1x1卷积用于特征融合,调整通道数
#             nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, padding=0),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.W = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0)  # 1x1卷积调整输出通道
#
#         self.SP_Pool_v = QKV_block(self.in_channels)  # 定义QKV块，用于计算值（V）
#         self.SP_Pool_k = QKV_block(self.in_channels)  # 定义QKV块，用于计算键（K）
#         nn.init.constant_(self.W.weight, 0)  # 初始化W卷积层的权重为0
#         nn.init.constant_(self.W.bias, 0)  # 初始化W卷积层的偏置为0
#     # 前向传播函数
#     def forward(self, x):
#         # 捕捉局部特征
#         c1 = self.conv_1(x)  # 经过第一个卷积层
#         c2 = self.conv_2(x)  # 经过第二个卷积层
#         c3 = self.conv_3(x)  # 经过第三个卷积层
#         c4 = self.conv_4(x)  # 经过第四个卷积层
#         cat = torch.cat([c1, c2, c3, c4], dim=1)  # 将四个特征拼接
#         fuse_out = self.fuse(cat)  # 融合特征
#         out = self.W(fuse_out)  # 最后的1x1卷积层输出
#         return out  # 返回最终输出

# MSAGF模块的定义：结合GLFA（全局与局部特征融合）与PixelAttention（像素级注意力机制）
class MSAGF(nn.Module):
    def __init__(self, in_dim, out_dim, *args):  # 初始化MSAGF模块，dim为输入特征的通道数
        super(MSAGF, self).__init__()
        self.GLFA = GLFA(in_dim)  # 初始化GLFA模块
        self.PixelAttention = PixelAttention(in_dim)  # 初始化PixelAttention模块
        self.conv = nn.Conv2d(in_dim, out_dim, 1, bias=True)  # 1x1卷积层，用于调整输出通道数
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

    # 前向传播函数
    def forward(self, x, y):
        initial = x + y  # 将输入x与y相加，作为初始特征
        pattn1 = self.GLFA(initial)  # 通过GLFA模块提取全局与局部特征
        pattn2 = self.sigmoid(self.PixelAttention(initial, pattn1))  # 通过PixelAttention计算像素级注意力
        result = initial + pattn2 * x + (1 - pattn2) * y  # 根据注意力图对x和y进行加权融合
        result = self.conv(result)  # 通过1x1卷积调整通道数
        return result  # 返回融合后的结果


class AKConv(nn.Module):
    def __init__(self, inc, outc, num_param=5, stride=1, bias=None):
        super(AKConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
                                  nn.BatchNorm2d(outc),
                                  nn.SiLU())  # the conv adds the BN and SiLU to compare original Conv in YOLOv5.
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # print(f"[AKConv] x dtype: {x.dtype}, conv weight dtype: {self.conv[0].weight.dtype}")  # 调试用
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # generating the inital sampled shapes for the AKConv with different sizes.
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0, base_int))
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1),
                torch.arange(0, mod_number))

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        # using Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        # using 1 × 1 Conv
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)
        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)

        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset


# 测试代码
if __name__ == '__main__':
    block = MSAGF(32, 32)  # 创建MSAGF模块实例，输入通道数in_dim=32，输出通道数out_dim=32
    input1 = torch.rand(1, 32, 64, 64)  # 生成一个随机输入张量input1，大小为(1, 32, 64, 64)
    input2 = torch.rand(1, 32, 64, 64)  # 生成一个随机输入张量input2，大小为(1, 32, 64, 64)
    output = block(input1, input2)  # 将input1和input2传入MSAGF模块进行前向传播
    print('input1_size:', input1.size())  # 打印input1的尺寸
    print('output_size:', output.size())  # 打印输出的尺寸

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return (x, x)


class StarReLU(nn.Module):
    """StarReLU: s * relu(x) ** 2 + b"""

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, bias=False):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class DynamicFilter(nn.Module):
    def __init__(self, dim, expansion_ratio=2, num_filters=4, size=14, weight_resize=True):
        super().__init__()
        self.dim = dim
        self.num_filters = num_filters
        self.weight_resize = weight_resize

        # Channel attention with dimension reduction and recovery
        reduced_dim = max(dim // 4, 8)  # Ensure minimum channel size
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, reduced_dim, 1, bias=False),  # 修改输入通道为 dim * 2
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, dim * 2, 1, bias=False),  # 修改输出通道为 dim * 2
            nn.Sigmoid()
        )

        # Spatial attention
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        # FFT related layers with channel preservation
        self.fft_bn = nn.BatchNorm2d(dim * 2)  # 修改为 dim * 2
        self.fft_conv = nn.Conv2d(dim * 2, dim * 2, 1, bias=False)  # 修改为 dim * 2

        # Intermediate processing
        self.process_conv = nn.Conv2d(dim * 2, dim * 2, 1, bias=False)  # 修改为 dim * 2
        self.process_bn = nn.BatchNorm2d(dim * 2)  # 修改为 dim * 2

        # Output projection to match input channels
        self.out_conv = nn.Conv2d(dim * 2, dim * 2, 1, bias=False)  # 修改为 dim * 2

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x

        # Channel attention with dimension reduction
        ca_weights = self.ca(x)
        x = x * ca_weights

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.sa(torch.cat([avg_out, max_out], dim=1))
        x = x * sa

        # FFT processing with channel preservation
        x = self.process_conv(x)
        x = self.process_bn(x)

        x_freq = torch.fft.rfft2(x.float(), dim=(-2, -1), norm='ortho')
        x_freq_abs = torch.abs(x_freq)
        x_freq_abs = x_freq_abs / (x_freq_abs.amax(dim=(-2, -1), keepdim=True) + 1e-8)
        x_freq_filtered = x_freq * x_freq_abs

        x = torch.fft.irfft2(x_freq_filtered, s=(H, W), dim=(-2, -1), norm='ortho')
        x = self.fft_bn(x)
        x = self.fft_conv(x)

        # Final processing with residual connection
        x = self.out_conv(x)

        return x + identity


class DynamicFilterBlock(nn.Module):
    def __init__(self, in_channels, debug=False):
        super().__init__()
        self.in_channels = in_channels
        # 添加通道调整层
        self.channel_adj = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)
        self.dynamic_filter = DynamicFilter(in_channels)
        self.act = nn.SiLU()
        self.debug = debug
        print(f"[DynamicFilterBlock] Initialized with in_channels={in_channels}")

    def _print_tensor_info(self, tensor, name):
        """Helper function to print tensor statistics"""
        if not self.debug:
            return

        print(f"\n{'=' * 50}")
        print(f"[DynamicFilterBlock] {name} statistics:")
        print(f"Shape: {tensor.shape}")
        print(f"Dtype: {tensor.dtype}")
        print(f"Device: {tensor.device}")
        print(f"Min: {tensor.min().item():.6f}")
        print(f"Max: {tensor.max().item():.6f}")
        print(f"Mean: {tensor.mean().item():.6f}")
        print(f"Std: {tensor.std().item():.6f}")
        print(f"Has NaN: {torch.isnan(tensor).any().item()}")
        print(f"Has Inf: {torch.isinf(tensor).any().item()}")
        print(f"{'=' * 50}\n")

    def forward(self, x):
        try:
            if self.debug:
                print("\n[DynamicFilterBlock] Forward pass started")

            # Check input
            self._print_tensor_info(x, "Input")

            # Save identity branch
            identity = x
            if self.debug:
                print("[DynamicFilterBlock] Identity branch saved")

            # 调整通道数
            if x.size(1) != self.in_channels:
                if self.debug:
                    print(f"[DynamicFilterBlock] Adjusting channels from {x.size(1)} to {self.in_channels}")
                x = self.channel_adj(x)
                self._print_tensor_info(x, "After channel adjustment")

            # Dynamic filter processing
            try:
                x = self.dynamic_filter(x)
                if self.debug:
                    print("[DynamicFilterBlock] Dynamic filter processing completed")
                self._print_tensor_info(x, "After DynamicFilter")
            except Exception as e:
                print(f"[DynamicFilterBlock] Error in dynamic_filter: {str(e)}")
                raise

            # Residual connection
            try:
                if x.shape != identity.shape:
                    print(f"[DynamicFilterBlock] WARNING: Shape mismatch in residual connection")
                    print(f"x shape: {x.shape}, identity shape: {identity.shape}")

                output = x + identity
                if self.debug:
                    print("[DynamicFilterBlock] Residual connection added")
                self._print_tensor_info(output, "After residual connection")
            except Exception as e:
                print(f"[DynamicFilterBlock] Error in residual connection: {str(e)}")
                raise

            # Activation
            try:
                output = self.act(output)
                if self.debug:
                    print("[DynamicFilterBlock] Activation applied")
                self._print_tensor_info(output, "Final output")
            except Exception as e:
                print(f"[DynamicFilterBlock] Error in activation: {str(e)}")
                raise

            if self.debug:
                print("[DynamicFilterBlock] Forward pass completed successfully")

            return output

        except Exception as e:
            print(f"[DynamicFilterBlock] Error in forward pass: {str(e)}")
            raise

    def extra_repr(self) -> str:
        """Extra representation of module for print() and str()"""
        return f'in_channels={self.in_channels}, debug={self.debug}'


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class EFBlock(nn.Module):
    def __init__(self, c1, c2, reduction=16):
        super(EFBlock, self).__init__()
        self.mask_map_r = nn.Conv2d(c1 // 2, 1, 1, 1, 0, bias=True)
        self.mask_map_i = nn.Conv2d(c1 // 2, 1, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)
        self.bottleneck1 = nn.Conv2d(c1 // 2, c2 // 2, 3, 1, 1, bias=False)
        self.bottleneck2 = nn.Conv2d(c1 // 2, c2 // 2, 3, 1, 1, bias=False)
        self.se = SE_Block(c2, reduction)

    def forward(self, x):
        # print("EFBlock input:", x.shape)
        x_left_ori, x_right_ori = x[:, :3, :, :], x[:, 3:, :, :]
        x_left = x_left_ori * 0.5
        x_right = x_right_ori * 0.5

        x_mask_left = torch.mul(self.mask_map_r(x_left), x_left)
        x_mask_right = torch.mul(self.mask_map_i(x_right), x_right)

        out_IR = self.bottleneck1(x_mask_right + x_right_ori)
        out_RGB = self.bottleneck2(x_mask_left + x_left_ori)  # RGB
        out = self.se(torch.cat([out_RGB, out_IR], 1))

        return out


##################################CPCA Attention#########################################

class FeatureAdd(nn.Module):
    #  x + CPCA
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.add(x[0], x[1])


class CPCA_ChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(CPCA_ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class CPCA(nn.Module):
    def __init__(self, channels, out_channels, channelAttention_reduce=4):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(1, 1), padding=0)
        self.ca = CPCA_ChannelAttention(input_channels=channels, internal_neurons=channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.dconv1_7 = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.dconv7_1 = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.dconv1_11 = nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels)
        self.dconv11_1 = nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels)
        self.dconv1_21 = nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10), groups=channels)
        self.dconv21_1 = nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0), groups=channels)
        self.conv2 = nn.Conv2d(channels, out_channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, x):
        inputs = torch.cat((x[0], x[1]), dim=1)
        #   Global Perceptron
        inputs = self.conv1(inputs)
        inputs = self.act(inputs)

        inputs = self.ca(inputs)

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv1(x)
        out = spatial_att * inputs
        out = self.conv2(out)
        return out


##################################Transformer############################################
# 多头交叉注意力机制
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        assert (self.head_dim * num_heads == model_dim), "model_dim must be divisible by num_heads"

        self.query_vis = nn.Linear(model_dim, model_dim)
        self.key_vis = nn.Linear(model_dim, model_dim)
        self.value_vis = nn.Linear(model_dim, model_dim)

        self.query_inf = nn.Linear(model_dim, model_dim)
        self.key_inf = nn.Linear(model_dim, model_dim)
        self.value_inf = nn.Linear(model_dim, model_dim)

        self.fc_out_vis = nn.Linear(model_dim, model_dim)
        self.fc_out_inf = nn.Linear(model_dim, model_dim)

    def forward(self, vis, inf):
        batch_size, seq_length, model_dim = vis.shape

        # vis -> Q, K, V
        Q_vis = self.query_vis(vis)
        K_vis = self.key_vis(vis)
        V_vis = self.value_vis(vis)

        # inf -> Q, K, V
        Q_inf = self.query_inf(inf)
        K_inf = self.key_inf(inf)
        V_inf = self.value_inf(inf)

        # Reshape for multi-head attention
        Q_vis = Q_vis.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,
                                                                                            2)  # B, N, C --> B, n_h, N, d_h
        K_vis = K_vis.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V_vis = V_vis.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        Q_inf = Q_inf.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K_inf = K_inf.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V_inf = V_inf.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Cross attention: vis Q with inf K and inf Q with vis K
        # Q_vis 的形状为 (batch_size, num_heads, seq_length, head_dim)
        # K_inf 的形状为 (batch_size, num_heads, head_dim, seq_length)
        # 矩阵乘法后，scores_vis_inf 的形状为 (batch_size, num_heads, seq_length, seq_length)
        scores_vis_inf = torch.matmul(Q_vis, K_inf.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))
        scores_inf_vis = torch.matmul(Q_inf, K_vis.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))

        attention_inf = torch.softmax(scores_vis_inf, dim=-1)
        attention_vis = torch.softmax(scores_inf_vis, dim=-1)

        # attention_vis_inf 的形状为 (batch_size, num_heads, seq_length, seq_length)
        # V_inf 的形状为 (batch_size, num_heads, seq_length, head_dim)
        # out_vis_inf 的形状为 (batch_size, num_heads, seq_length, head_dim)
        out_inf = torch.matmul(attention_inf, V_inf)
        out_vis = torch.matmul(attention_vis, V_vis)

        # Concatenate and project back to the original dimension
        out_vis = out_vis.transpose(1, 2).contiguous().view(batch_size, seq_length, model_dim)
        out_inf = out_inf.transpose(1, 2).contiguous().view(batch_size, seq_length, model_dim)

        # out 的形状为 (batch_size, seq_length, model_dim)
        out_vis = self.fc_out_vis(out_vis)
        out_inf = self.fc_out_inf(out_inf)

        return out_vis, out_inf


# 前向全连接网络
class FeedForward(nn.Module):
    def __init__(self, model_dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(model_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout, max_len=6400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个从0到max_len-1的列向量，形状为 (max_len, 1)
        position = torch.arange(0, max_len).unsqueeze(1)
        # 计算用于位置编码的除数项
        div_term = torch.exp(torch.arange(0, model_dim, 2) * -(torch.log(torch.tensor(10000.0)) / model_dim))

        pe = torch.zeros(max_len, model_dim)  # 初始化一个位置编码矩阵，形状为 (max_len, model_dim)，所有元素初始化为0
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列使用sin函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列使用cos函数

        pe = pe.unsqueeze(0)  # 在位置编码矩阵的第一个维度前添加一个新的维度，变为 (1, max_len, model_dim)
        self.register_buffer('pe', pe)  # 将位置编码矩阵 pe 注册为模型的一个缓冲区。缓冲区类似于模型参数，但在训练过程中不会更新

    def forward(self, x):
        # 动态生成与输入长度/设备/精度匹配的位置编码，避免 max_len 限制
        # x: (batch, seq_len, model_dim)
        seq_len = x.size(1)
        model_dim = x.size(-1)
        device = x.device
        dtype = x.dtype

        position = torch.arange(0, seq_len, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2, device=device, dtype=dtype)
            * -(torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / model_dim)
        )
        pe = torch.zeros(1, seq_len, model_dim, device=device, dtype=dtype)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        x = x + pe
        return self.dropout(x)


# 编码器层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.cross_attention = MultiHeadCrossAttention(model_dim, num_heads)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ff = FeedForward(model_dim, hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, vis, inf):
        attn_out_vis, attn_out_inf = self.cross_attention(vis, inf)
        vis = self.norm1(vis + attn_out_vis)
        inf = self.norm1(inf + attn_out_inf)

        ff_out_vis = self.ff(vis)
        ff_out_inf = self.ff(inf)

        vis = self.norm2(vis + ff_out_vis)
        inf = self.norm2(inf + ff_out_inf)

        return vis, inf


# Transformer编码器
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, hidden_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(model_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, vis, inf):
        vis = self.embedding(vis) * torch.sqrt(torch.tensor(self.embedding.out_features, dtype=torch.float32))
        inf = self.embedding(inf) * torch.sqrt(torch.tensor(self.embedding.out_features, dtype=torch.float32))

        vis = self.positional_encoding(vis)
        inf = self.positional_encoding(inf)

        for layer in self.layers:
            vis, inf = layer(vis, inf)

        return vis, inf


# 定义用于交叉注意力的网络
class CrossTransformerFusion(nn.Module):
    def __init__(self, input_dim, num_heads=2, num_layers=1, dropout=0.1):
        super(CrossTransformerFusion, self).__init__()
        self.hidden_dim = input_dim * 2
        self.model_dim = input_dim
        self.encoder = TransformerEncoder(input_dim, self.model_dim, num_heads, num_layers, self.hidden_dim, dropout)

    def forward(self, x):
        vis, inf = x[0], x[1]
        # 输入形状为 B, C, H, W
        B, C, H, W = vis.shape

        # 将输入变形为 B, H*W, C
        vis = vis.permute(0, 2, 3, 1).reshape(B, -1, C)
        inf = inf.permute(0, 2, 3, 1).reshape(B, -1, C)

        # 输入Transformer编码器
        vis_out, inf_out = self.encoder(vis, inf)

        # 将输出变形为 B, C, H, W
        vis_out = vis_out.view(B, H, W, -1).permute(0, 3, 1, 2)
        inf_out = inf_out.view(B, H, W, -1).permute(0, 3, 1, 2)

        # 在通道维度上进行级联
        out = torch.cat((vis_out, inf_out), dim=1)

        return out


import torch
import torch.nn as nn
from einops import rearrange


# 定义函数，将四维张量重排为三维
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')  # 将输入的四维张量重排为三维，合并高度和宽度维度


# 定义函数，将三维张量重排为四维
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)  # 将三维张量恢复为四维，并重排维度


# PixelAttention类，用于计算像素级别的注意力
class PixelAttention(nn.Module):
    def __init__(self, dim):  # 初始化PixelAttention模块，dim为输入特征的通道数
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)  # 定义卷积层
        self.sigmoid = nn.Sigmoid()  # 定义Sigmoid激活函数，用于生成注意力图

    # 前向传播函数
    def forward(self, x, pattn1):
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W  # 在通道维度插入一个维度
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W  # 同样在pattn1插入一个维度
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W  # 沿着通道维度将x和pattn1拼接
        x2 = rearrange(x2, 'b c t h w -> b (c t) h w')  # 重排张量维度为 [b, (c * t), h, w]
        pattn2 = self.pa2(x2)  # 经过卷积层计算注意力图
        pattn2 = self.sigmoid(pattn2)  # 通过Sigmoid归一化注意力图
        return pattn2  # 返回计算得到的注意力图


# 定义QKV_block类，用于计算多种池化操作后的特征
# class QKV_block(nn.Module):
#     def __init__(self, in_channels):
#         super(QKV_block, self).__init__()
#         self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)  # 定义2x2池化层
#         self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)  # 定义3x3池化层
#         self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)  # 定义5x5池化层
#         self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)  # 定义6x6池化层
#
#     # 前向传播函数
#     def forward(self, x):
#         b, c, h, w = x.size()  # 获取输入x的形状，b为batch_size，c为通道数，h和w为高度和宽度
#         pool_1 = self.pool1(x).view(b, c, -1)  # 对输入x进行2x2池化，并将输出展平
#         pool_2 = self.pool2(x).view(b, c, -1)  # 对输入x进行3x3池化，并将输出展平
#         pool_3 = self.pool3(x).view(b, c, -1)  # 对输入x进行5x5池化，并将输出展平
#         pool_4 = self.pool4(x).view(b, c, -1)  # 对输入x进行6x6池化，并将输出展平
#         pool_cat = torch.cat([pool_1, pool_2, pool_3, pool_4], -1)  # 将所有池化结果拼接
#         out = pool_cat.permute(0, 2, 1)  # 转置张量的维度为 [B,C,L]
#         return out  # 返回计算结果
#
# # 定义通道洗牌函数，用于打乱通道顺序
# def channel_shuffle(x, groups):
#     batchsize, num_channels, height, width = x.data.size()  # 获取输入张量的形状
#     channels_per_group = num_channels // groups  # 计算每组的通道数
#     x = x.view(batchsize, groups, channels_per_group, height, width)  # 将通道维度重塑为(groups, channels_per_group)
#     x = torch.transpose(x, 1, 2).contiguous()  # 交换第1维和第2维，通道组与每组内的通道进行交换
#     x = x.view(batchsize, -1, height, width)  # 将混排后的张量展平为(b, num_channels, height, width)
#     return x  # 返回混排后的结果

# 定义GLFA类，用于全局与局部特征的聚合
# class GLFA(nn.Module):
#     def __init__(self, in_channels):  # 简化参数，只需要输入通道数
#         super(GLFA, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = in_channels
#
#         # 将普通卷积替换为深度可分离卷积
#         # 深度卷积+点卷积的组合
#         self.conv_1 = nn.Sequential(
#             # 深度卷积
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
#             # 点卷积
#             nn.Conv2d(in_channels, in_channels, kernel_size=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )
#
#         self.conv_2 = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2, groups=in_channels),
#             nn.Conv2d(in_channels, in_channels, kernel_size=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )
#
#         self.conv_3 = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3, groups=in_channels),
#             nn.Conv2d(in_channels, in_channels, kernel_size=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )
#
#         self.conv_4 = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=4, dilation=4, groups=in_channels),
#             nn.Conv2d(in_channels, in_channels, kernel_size=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )
#
#         # 融合层也使用深度可分离卷积
#         self.fuse = nn.Sequential(
#             # 深度卷积
#             nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size=3, padding=1, groups=in_channels * 4),
#             # 点卷积用于降维
#             nn.Conv2d(in_channels * 4, in_channels, kernel_size=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )
#
#         # 保持1x1卷积，因为它已经足够轻量
#         self.W = nn.Conv2d(in_channels=self.in_channels,
#                            out_channels=self.out_channels,
#                            kernel_size=1,
#                            stride=1,
#                            padding=0)
#
#         self.SP_Pool_v = QKV_block(self.in_channels)
#         self.SP_Pool_k = QKV_block(self.in_channels)
#
#         # 初始化权重
#         nn.init.constant_(self.W.weight, 0)
#         nn.init.constant_(self.W.bias, 0)
#
#     def forward(self, x):
#         # 捕捉局部特征
#         c1 = self.conv_1(x)  # 经过第一个卷积层
#         c2 = self.conv_2(x)  # 经过第二个卷积层
#         c3 = self.conv_3(x)  # 经过第三个卷积层
#         c4 = self.conv_4(x)  # 经过第四个卷积层
#         cat = torch.cat([c1, c2, c3, c4], dim=1)  # 将四个特征拼接
#         fuse_out = self.fuse(cat)  # 融合特征
#         out = self.W(fuse_out)  # 最后的1x1卷积层输出
#         return out  # 返回最终输出

# MSAGF模块的定义：结合GLFA（全局与局部特征融合）与PixelAttention（像素级注意力机制）
class MSAGF(nn.Module):
    def __init__(self, in_dim, out_dim, *args):  # 初始化MSAGF模块，dim为输入特征的通道数
        super(MSAGF, self).__init__()
        self.GLFA = GLFA(in_dim)  # 初始化GLFA模块
        self.PixelAttention = PixelAttention(in_dim)  # 初始化PixelAttention模块
        self.conv = nn.Conv2d(in_dim, out_dim, 1, bias=True)  # 1x1卷积层，用于调整输出通道数
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

    # 前向传播函数
    def forward(self, x, y):
        initial = x + y  # 将输入x与y相加，作为初始特征
        pattn1 = self.GLFA(initial)  # 通过GLFA模块提取全局与局部特征
        pattn2 = self.sigmoid(self.PixelAttention(initial, pattn1))  # 通过PixelAttention计算像素级注意力
        result = initial + pattn2 * x + (1 - pattn2) * y  # 根据注意力图对x和y进行加权融合
        result = self.conv(result)  # 通过1x1卷积调整通道数
        return result  # 返回融合后的结果


class AKConv(nn.Module):
    def __init__(self, inc, outc, num_param=5, stride=1, bias=None):
        super(AKConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
                                  nn.BatchNorm2d(outc),
                                  nn.SiLU())  # the conv adds the BN and SiLU to compare original Conv in YOLOv5.
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # print(f"[AKConv] x dtype: {x.dtype}, conv weight dtype: {self.conv[0].weight.dtype}")  # 调试用
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # generating the inital sampled shapes for the AKConv with different sizes.
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0, base_int))
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1),
                torch.arange(0, mod_number))

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        # using Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        # using 1 × 1 Conv
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)
        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)

        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset


# 测试代码
if __name__ == '__main__':
    block = MSAGF(32, 32)  # 创建MSAGF模块实例，输入通道数in_dim=32，输出通道数out_dim=32
    input1 = torch.rand(1, 32, 64, 64)  # 生成一个随机输入张量input1，大小为(1, 32, 64, 64)
    input2 = torch.rand(1, 32, 64, 64)  # 生成一个随机输入张量input2，大小为(1, 32, 64, 64)
    output = block(input1, input2)  # 将input1和input2传入MSAGF模块进行前向传播
    print('input1_size:', input1.size())  # 打印input1的尺寸
    print('output_size:', output.size())  # 打印输出的尺寸

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from .conv import Conv


# class WaveletFusion(nn.Module):
#     def __init__(self, in_channels, wave=None):
#         super().__init__()
#         # 深度可分离卷积
#         self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
#         self.pwconv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
#         self.bn = nn.BatchNorm2d(in_channels)
#         self.act = nn.ReLU(inplace=True)
#         # 通道注意力
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc1 = nn.Conv2d(in_channels, in_channels // 16, 1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv2d(in_channels // 16, in_channels, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#         # 空间注意力
#         self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
#         self.spatial_sigmoid = nn.Sigmoid()
#
#     def channel_attention(self, x):
#         avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         scale = self.sigmoid(out)
#         return x * scale
#
#     def spatial_attention(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x_cat = torch.cat([avg_out, max_out], dim=1)
#         attn = self.spatial_conv(x_cat)
#         scale = self.spatial_sigmoid(attn)
#         return x * scale
#
#     def forward(self, x1, x2):
#         # 对齐空间尺寸
#         if x1.shape[2:] != x2.shape[2:]:
#             x2 = torch.nn.functional.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
#         x_fused = (x1 + x2) / 2
#         out = self.dwconv(x_fused)
#         out = self.pwconv(out)
#         out = self.bn(out)
#         out = self.act(out)
#         out = self.channel_attention(out)
#         out = self.spatial_attention(out)
#         out = out + x_fused
#         return out
#
#
# class CascadeWaveletFusion(nn.Module):
#     def __init__(self, in_channels, wave=None):
#         super().__init__()
#         self.wavelet_fusion = WaveletFusion(in_channels, wave)
#         # 不在这里定义fusion_conv，forward时动态定义
#
#     def forward(self, x1, x2, prev_fusion=None):
#         # 对齐x2
#         if x1.shape[2:] != x2.shape[2:]:
#             x2 = torch.nn.functional.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
#         fused = self.wavelet_fusion(x1, x2)
#         if prev_fusion is not None:
#             if prev_fusion.shape[2:] != fused.shape[2:]:
#                 prev_fusion = torch.nn.functional.interpolate(prev_fusion, size=fused.shape[2:], mode='bilinear',
#                                                               align_corners=False)
#             out = torch.cat([fused, prev_fusion], dim=1)
#             # 动态创建fusion_conv
#             if not hasattr(self, 'fusion_conv') or self.fusion_conv.in_channels != out.shape[
#                 1] or self.fusion_conv.out_channels != fused.shape[1]:
#                 self.fusion_conv = nn.Conv2d(out.shape[1], fused.shape[1], 1, bias=False).to(out.device)
#             out = self.fusion_conv(out)
#             return out
#         else:
#             return fused


class GLFA(nn.Module):
    """Lightweight GLFA module with AK convolution integration and scale-adaptive dilation"""

    def __init__(self,
                 in_channels,
                 scale='p3',
                 channel_factor=2,  # 通道压缩因子
                 attention_ratio=8,  # 注意力模块压缩比
                 use_coordconv=True,  # 是否使用CoordConv
                 act=nn.ReLU):  # 激活函数类型
        super(GLFA, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.scale = scale.lower()
        self.debug = False  # 是否开启特征图保存
        self.saved_features = {}  # 用于存放每一步的特征图（CPU 上的 tensor）

        # Set dilation rates based on scale
        if self.scale in ['p2', 'p3']:
            dilation1, dilation2 = 1, 2  # For higher resolution features
        else:  # p4, p5
            dilation1, dilation2 = 3, 4  # For lower resolution features

        # Add coordinate channels if using CoordConv
        self.add_coords = None
        self.coord_channels = 2 if use_coordconv else 0
        if use_coordconv:
            self.add_coords = AddCoords()

        conv_in_channels = in_channels + self.coord_channels
        mid_channels = conv_in_channels // channel_factor

        # AK convolution branch 1
        self.branch1 = nn.Sequential(
            nn.Conv2d(conv_in_channels, mid_channels, kernel_size=(1, 3),
                      padding=(0, dilation1), dilation=dilation1, bias=False),  # Remove bias
            nn.BatchNorm2d(mid_channels),
            act(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(3, 1),
                      padding=(dilation1, 0), dilation=dilation1, bias=False),  # Remove bias
            nn.BatchNorm2d(mid_channels),
            act(inplace=True)
        )

        # AK convolution branch 2
        self.branch2 = nn.Sequential(
            nn.Conv2d(conv_in_channels, mid_channels, kernel_size=(1, 3),
                      padding=(0, dilation2), dilation=dilation2, bias=False),  # Remove bias
            nn.BatchNorm2d(mid_channels),
            act(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(3, 1),
                      padding=(dilation2, 0), dilation=dilation2, bias=False),  # Remove bias
            nn.BatchNorm2d(mid_channels),
            act(inplace=True)
        )

        # Channel attention - now using the correct channel count
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        attention_channels = mid_channels * 2  # Combined channels from both branches
        self.fc = nn.Sequential(
            nn.Linear(attention_channels, attention_channels // attention_ratio, bias=False),  # Remove bias
            act(inplace=True),
            nn.Linear(attention_channels // attention_ratio, attention_channels, bias=False),  # Remove bias
            nn.Sigmoid()
        )

        # Final fusion with residual connection
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels * 2, in_channels, kernel_size=1, bias=False),  # Remove bias
            nn.BatchNorm2d(in_channels),
            act(inplace=True)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize the weights using kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def _save_feature(self, feat, name: str):
        """
        保存中间特征图到 self.saved_features 中，仅在 self.debug=True 时生效
        """
        if not getattr(self, "debug", False):
            return

        if not hasattr(self, "saved_features"):
            self.saved_features = {}

        # 只保存到 CPU，避免 GPU 显存占太多
        with torch.no_grad():
            self.saved_features[name] = feat.detach().cpu()

    @staticmethod
    def create_for_scale(in_channels, scale, **kwargs):
        """Factory method to create GLFA instance for specific scale"""
        return GLFA(in_channels, scale, **kwargs)

    # def forward(self, x):
    #     identity = x
    #     # print(f"[GLFA] x dtype: {x.dtype}, branch1 weight dtype: {self.branch1[0].weight.dtype}")  # 调试用
    #     x = x.to(self.branch1[0].weight.dtype)  # 保证输入和权重类型一致
    #     # Add coordinate channels if using CoordConv
    #     if self.add_coords is not None:
    #         x = self.add_coords(x)
    #
    #     # Branch processing
    #     feat1 = self.branch1(x)
    #     feat2 = self.branch2(x)
    #
    #     # Concatenate features from both branches
    #     feat = torch.cat([feat1, feat2], dim=1)
    #
    #     # Channel attention
    #     b, c, _, _ = feat.size()
    #     y = self.avg_pool(feat).view(b, c)
    #     y = self.fc(y).view(b, c, 1, 1)
    #     feat = feat * y.expand_as(feat)
    #
    #     # Final fusion with residual connection
    #     out = self.fuse(feat)
    #     out = out + identity
    #
    #     return out
    # def forward(self, x):
    #     identity = x
    #     x = x.to(self.branch1[0].weight.dtype)
    #
    #
    #     # Add coordinate channels if using CoordConv
    #     if self.add_coords is not None:
    #         x = self.add_coords(x)
    #
    #
    #     # Branch processing
    #     feat1 = self.branch1(x)
    #
    #     feat2 = self.branch2(x)
    #
    #     # Concatenate features from both branches
    #     feat = torch.cat([feat1, feat2], dim=1)
    #
    #     # Channel attention
    #     b, c, _, _ = feat.size()
    #     y = self.avg_pool(feat).view(b, c)
    #     y = self.fc(y).view(b, c, 1, 1)
    #     feat = feat * y.expand_as(feat)
    #
    #     # Final fusion with residual connection
    #     out = self.fuse(feat)
    #
    #     out = out + identity
    #
    #     return out
    def forward(self, x):
        identity = x
        x = x.to(self.branch1[0].weight.dtype)

        # 原始输入（或 CoordConv 之前）
        self._save_feature(x, "input_before_coord")

        # Add coordinate channels if using CoordConv
        if self.add_coords is not None:
            x = self.add_coords(x)
            self._save_feature(x, "after_coordconv")

        # Branch processing
        feat1 = self.branch1(x)
        self._save_feature(feat1, "branch1_out")

        feat2 = self.branch2(x)
        self._save_feature(feat2, "branch2_out")

        # Concatenate features from both branches
        feat = torch.cat([feat1, feat2], dim=1)
        self._save_feature(feat, "concat_feat")

        # Channel attention
        b, c, _, _ = feat.size()
        y = self.avg_pool(feat).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        feat = feat * y.expand_as(feat)
        self._save_feature(feat, "after_channel_attention")

        # Final fusion with residual connection
        out = self.fuse(feat)
        self._save_feature(out, "after_fuse")

        out = out + identity
        self._save_feature(out, "final_out_with_residual")

        return out

class AddCoords(nn.Module):
    """Add CoordConv channels"""

    def __init__(self):
        super(AddCoords, self).__init__()

    def forward(self, x):
        batch_size, _, height, width = x.size()

        xx_channel = torch.arange(width).float()
        xx_channel = xx_channel.repeat(batch_size, 1, height, 1)
        xx_channel = xx_channel.to(x.device) / (width - 1)

        yy_channel = torch.arange(height).float()
        yy_channel = yy_channel.repeat(batch_size, 1, width, 1).transpose(2, 3)
        yy_channel = yy_channel.to(x.device) / (height - 1)

        return torch.cat([x, xx_channel, yy_channel], dim=1)


class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, Conv.default_act)
        self.cv2 = Conv(c_, c_, 5, 1, p, c_, d, Conv.default_act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        # y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        # y = y.permute(0, 2, 1, 3, 4)
        # return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])

        b, n, h, w = x2.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)


# class WaveletFusion(nn.Module):
#     def __init__(self, in_channels, wave=None):
#         super().__init__()
#         self.in_channels = in_channels
#         # 轻量化两路：GSConv 路径 + 深度可分離卷積路径
#         # 路径A: GSConv（自带轻量卷积混洗）
#         self.gsconv = GSConv(in_channels, in_channels)
#
#         # 路径B: 深度可分離卷積
#         self.dwconv = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )
#
#         # 1×1 融合 + BN + ReLU（通道不变）
#         self.fuse = nn.Sequential(
#             nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )
#
#         # 动态可学习权重门控：生成两分支的按通道权重（softmax 归一化）
#         mid = max(in_channels // 4, 1)
#         self.branch_gate = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, mid, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid, in_channels * 2, 1, bias=False)
#         )
#
#     def forward(self, x1, x2):
#         # 对齐空间尺寸
#         if x1.shape[2:] != x2.shape[2:]:
#             x2 = torch.nn.functional.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
#
#         # 初始融合
#         x_fused = (x1 + x2) / 2
#
#         # 路径A/B 前向
#         a = self.gsconv(x_fused)
#         b = self.dwconv(x_fused)
#
#         # 动态门控（按通道 softmax 分配到两分支）
#         g = self.branch_gate(x_fused)  # [B, 2C, 1, 1]
#         B = g.size(0)
#         g = g.view(B, 2, self.in_channels, 1, 1)
#         g = torch.softmax(g, dim=1)
#         ga, gb = g[:, 0], g[:, 1]  # [B, C, 1, 1]
#         a = a * ga
#         b = b * gb
#
#         # 融合
#         out = self.fuse(torch.cat([a, b], dim=1))
#
#         # 殘差連接
#         out = out + x_fused
#
#         return out

class WaveletFusion(nn.Module):
    def __init__(self, in_channels, wave=None):
        super().__init__()
        self.force_equal_gate = True
        self.latest_gb = None
        self.latest_ga = None
        self.latest_branch_a = None
        self.latest_branch_b = None
        self.in_channels = in_channels
        # 轻量化两路：GSConv 路径 + 深度可分離卷積路径
        # 路径A: GSConv（自带轻量卷积混洗）
        self.gsconv = GSConv(in_channels, in_channels)

        # 路径B: 深度可分離卷積
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 1×1 融合 + BN + ReLU（通道不变）
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 动态可学习权重门控：生成两分支的按通道权重（softmax 归一化）
        mid = max(in_channels // 4, 1)
        self.branch_gate = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels * 2, 1, bias=False)
        )

    def forward(self, x1, x2):
        # 对齐空间尺寸
        if x1.shape[2:] != x2.shape[2:]:
            x2 = torch.nn.functional.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)

        # 初始融合
        x_fused = (x1 + x2) / 2

        # 路径A/B 前向
        a = self.gsconv(x_fused)
        b = self.dwconv(x_fused)

        # 动态门控（按通道 softmax 分配到两分支）
        g = self.branch_gate(x_fused)  # [B, 2C, H, W]
        B, _, H, W = g.shape
        g = g.view(B, 2, self.in_channels, H, W)
        g = torch.softmax(g, dim=1)
        ga, gb = g[:, 0], g[:, 1]  # [B, C, H, W]
        if self.force_equal_gate:
            equal_gate = 0.5
            ga = torch.full_like(ga, equal_gate)
            gb = torch.full_like(gb, equal_gate)
        self.latest_ga = ga.detach()
        self.latest_gb = gb.detach()
        a = a * ga
        b = b * gb
        self.latest_branch_a = a.detach()
        self.latest_branch_b = b.detach()

        # 融合
        out = self.fuse(torch.cat([a, b], dim=1))

        # 殘差連接
        out = out + x_fused

        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        scale = self.sigmoid(out)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(x_cat)
        scale = self.sigmoid(attn)
        return x * scale


class CascadeWaveletFusion(nn.Module):
    def __init__(self, in_channels, wave=None):
        super().__init__()
        self.wavelet_fusion = WaveletFusion(in_channels, wave)
        # 不在这里定义fusion_conv，forward时动态定义

    def forward(self, x1, x2, prev_fusion=None):
        # 对齐x2
        if x1.shape[2:] != x2.shape[2:]:
            x2 = torch.nn.functional.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        fused = self.wavelet_fusion(x1, x2)
        if prev_fusion is not None:
            if prev_fusion.shape[2:] != fused.shape[2:]:
                prev_fusion = torch.nn.functional.interpolate(prev_fusion, size=fused.shape[2:], mode='bilinear',
                                                              align_corners=False)
            out = torch.cat([fused, prev_fusion], dim=1)
            # 动态创建fusion_conv
            if not hasattr(self, 'fusion_conv') or self.fusion_conv.in_channels != out.shape[
                1] or self.fusion_conv.out_channels != fused.shape[1]:
                self.fusion_conv = nn.Conv2d(out.shape[1], fused.shape[1], 1, bias=False).to(out.device)
            out = self.fusion_conv(out)
            return out
        else:
            return fused




