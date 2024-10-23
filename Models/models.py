import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import pdb
from abc import abstractmethod
from einops import rearrange
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many, repeat_many

# class TemporalBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
#         """
#         A single temporal block in the TCN.

#         Args:
#             in_channels (int): Number of input channels.
#             out_channels (int): Number of output channels.
#             kernel_size (int): Size of the convolution kernel.
#             stride (int): Stride of the convolution.
#             dilation (int): Dilation factor.
#             padding (int): Padding applied to the input.
#             dropout (float): Dropout rate.
#         """
#         super(TemporalBlock, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
#         # self.dropout = nn.Dropout(dropout)
#         self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
#         self.relu = nn.ReLU()
#         self.norm1 = nn.BatchNorm1d(out_channels)
#         self.norm2 = nn.BatchNorm1d(out_channels)

#     def forward(self, x):
#         """
#         Forward pass through the temporal block.

#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, in_channels, sequence_length).

#         Returns:
#             torch.Tensor: Output tensor of shape (batch_size, out_channels, sequence_length).
#         """
#         out = self.relu(self.norm1(self.conv1(x)))
#         # out = self.dropout(out)
#         out = self.relu(self.norm2(self.conv2(out)))
#         # out = self.dropout(out)
#         print('check dimension', out.shape, x.shape)
#         # Residual connection
#         res = x if self.downsample is None else self.downsample(x)
#         return self.relu(out + res)

# class TCN(nn.Module):
#     def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
#         """
#         A Temporal Convolutional Network (TCN).

#         Args:
#             num_inputs (int): Number of input channels.
#             num_channels (list): List of output channels for each temporal block.
#             kernel_size (int): Kernel size for the convolutions.
#             dropout (float): Dropout rate.
#         """
#         super(TCN, self).__init__()
#         layers = []
#         num_levels = len(num_channels)
#         for i in range(num_levels):
#             dilation_size = 2 ** i
#             in_channels = num_inputs if i == 0 else num_channels[i-1]
#             out_channels = num_channels[i]
#             layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
#                                         padding=(kernel_size - 1) * dilation_size, dropout=dropout))
#         self.network = nn.Sequential(*layers)
#         self.pool = nn.AdaptiveAvgPool1d(1)  # Reduce the temporal dimension to 1
#         self.flatten = nn.Flatten()

#     def forward(self, x):
#         """
#         Forward pass through the TCN.

#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, num_inputs).

#         Returns:
#             torch.Tensor: Output tensor of shape (batch_size, num_channels[-1]), a feature vector.
#         """
#         x = x.permute(0, 2, 1)  # Change shape to (batch_size, num_inputs, sequence_length)
#         out = self.network(x)
#         out = self.pool(out)  # Apply pooling to reduce temporal dimension
#         out = self.flatten(out)  # Flatten to create a feature vector
#         return out

    
"""
used in non-autoregression
"""

class FCN(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.conv1 = nn.Conv1d(dim_in, 2 * dim_in, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(2 * dim_in)
        self.conv2 = nn.Conv1d(2 * dim_in, 4 * dim_in, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(4 * dim_in)
        self.conv3 = nn.Conv1d(4 * dim_in, 2 * dim_in, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(2 * dim_in)
        self.pool = nn.MaxPool1d(2, 2)
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        out = self.flatten(self.pool(x))
        return out

class ARModel(nn.Module):
    def __init__(self, d, H, L) -> None:
        """
        d: dimension of each vector
        H: number of copies
        L: number of steps
        """
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(1, d, H)) for _ in range(L)])
        self.bias = nn.Parameter(torch.randn(1, d, H))
        
    def forward(self, X):
        # z_ar = self.bias.clone()
        z_ar = self.bias.expand(X.size(0), -1, -1)
        # print('z_ar', z_ar.shape, X.shape)
        
        for i, W_i in enumerate(self.weights):
            X_i = X[:, :, i].unsqueeze(-1).expand(-1, -1, W_i.size(-1))
            z_ar = z_ar + W_i * X_i
        
        return z_ar

class ConditionEncoder(nn.Module):
    def __init__(self, dim_in, seq_len) -> None:
        super().__init__()
        self.d = dim_in
        self.H = seq_len
        self.L = seq_len
        # self.f_encoder = FCN(dim_in)
        self.f_encoder = nn.Conv1d(dim_in, dim_in, kernel_size=3, padding=1)
        self.AR_model = ARModel(dim_in, seq_len, seq_len)
        
    def forward(self, context, future=None):
        context = context.permute(0, 2, 1)
        # test
        if future is None:
            z_mix = self.f_encoder(context)
        # train
        else:
            future = future.permute(0, 2, 1)
            m_k = torch.rand(self.d, self.H).to(context.device)
            z_mix = m_k * self.f_encoder(context) + (1 - m_k) * future
        # print('check z_mix', z_mix.shape)
        
        z_ar = self.AR_model(context)
        # print('check z_ar', z_ar.shape)
        condition = torch.concatenate([z_mix, z_ar], dim=1) # (b,H,2*d)
        return condition

class TimeDiffBackbone(nn.Module):
    def __init__(self, point_dim, feature_dim, seq_len, iftest=False) -> None:
        super().__init__()
        self.encoder = ConditionEncoder(point_dim, seq_len)
        self.concat1 = ConcatSquashLinear(point_dim, feature_dim, 3072+3)
        self.net = VanillaUnet(feature_dim)
        self.final_linear = nn.Linear(feature_dim, point_dim)
        self.iftest = iftest
        
    """
    
    x: future data to be sampled
    beta: temporal mark, part of feature
    context: prior information, 
    """
    def forward(self, beta, context, future):
        batch_size = future.size(0)
        beta = beta.view(batch_size, 1, 1) # (B,1,1)
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)],dim=-1).to(future.device)
        if not self.iftest:
            context_encoded = self.encoder(context, future).view(batch_size, 1, -1)
        else:
            context_encoded = self.encoder(context).view(batch_size, 1, -1)
        # print('=------check context',time_emb.shape, context_encoded.shape)
        # pdb.set_trace()
        
        ctx_emb = torch.cat([time_emb, context_encoded], dim=-1)
        # print('check ctx_emb shape', future.shape, ctx_emb.device, ctx_emb.shape)
        
        """
        use context to filter future vector
        """
        future = self.concat1(ctx_emb, future)
        # print('check future shape', future.shape)
        
        """
        get backbone's output
        """
        predict = self.net(future)
        # print('check predict', predict.shape)
        
        """
        return to the input dimension
        """
        output = self.final_linear(predict)
        return output


"""
From gpt o1
"""

# Time-Series Backbone Model
class TimeSeriesBackbone(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, time_emb_dim=256):
        super(TimeSeriesBackbone, self).__init__()
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.time_proj = nn.Linear(time_emb_dim, input_channels*2)
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_channels * 2, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_channels, output_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, context, x_t, t):
        # context: (batch_size, seq_length, input_channels)
        # x_t: (batch_size, seq_length, input_channels)
        # t: (batch_size,)
        
        # Compute time embeddings
        t = t.float().unsqueeze(-1)  # Shape: (batch_size, 1)
        t_emb = self.time_mlp(t)     # Shape: (batch_size, time_emb_dim)
        t_emb = self.time_proj(t_emb)
        t_emb = t_emb.unsqueeze(-1)  # Shape: (batch_size, time_emb_dim, 1)
        
        # Permute context and x_t to (batch_size, input_channels, seq_length)
        context = context.permute(0, 2, 1)
        x_t = x_t.permute(0, 2, 1)
        
        # Concatenate context and x_t along the channel dimension
        x = torch.cat([context, x_t], dim=1)  # Shape: (batch_size, input_channels * 2, seq_length)
        
        # Add time embedding to x
        # print('----- check', x.shape, t_emb.shape) # x -- (b, 2*input_channel, seq_length)
        x = x + t_emb  # Broadcasting over seq_length
        
        # Pass through convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        
        # Permute back to (batch_size, seq_length, output_channels)
        x = x.permute(0, 2, 1)
        return x

class PureCNN(nn.Module):
    def __init__(self, in_dim) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, in_dim*2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(in_dim*2)
        self.conv2 = nn.Conv1d(in_dim*2, in_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(in_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        out = x2 + x
        return out.permute(0, 2, 1)

"""
baseline
"""

class TemporalCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, dilation, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, dilation, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        out = self.pool(self.relu(self.bn1(self.conv1(x))))
        out = self.pool(self.relu(self.bn2(self.conv2(out))))
        return out
        
class SimpleCNN(nn.Module):
    def __init__(self, in_channel, num_channels, kernel_size, dropout=0.0) -> None:
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            # dilation_size = 2 ** i
            dilation_size = 1
            in_channels = in_channel if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalCNN(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                        padding=(kernel_size - 1) * dilation_size, dropout=dropout))
        self.network = nn.Sequential(*layers)
        # self.pool = nn.AdaptiveAvgPool1d(1)  # Reduce the temporal dimension to 1
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.network(x)
        out = self.flatten(out)
        # print('check out shape', out.shape)
        return out
    
class VanillaUnet(nn.Module):
    def __init__(self, d_channel) -> None:
        super().__init__()
        self.encoder1 = self._conv_block(d_channel, 16)
        self.encoder2 = self._conv_block(16, 32)
        self.encoder3 = self._conv_block(32, 64)
        self.encoder4 = self._conv_block(64, 128)
        
        self.upsample1 = nn.ConvTranspose1d(128, 128, kernel_size=2, stride=2)
        self.decoder1 = self._conv_block(128+64, 64)
        self.upsample2 = nn.ConvTranspose1d(64, 64, kernel_size=2, stride=2)
        self.decoder2 = self._conv_block(64+32, 32)
        self.upsample3 = nn.ConvTranspose1d(32, 32, kernel_size=2, stride=2)
        self.decoder3 = self._conv_block(32+16, 16)
        self.decoder4 = self._conv_block(16+d_channel, d_channel)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5)
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _up_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            # nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5)
        )

    def forward(self, input):
        input = input.permute(0,2,1)
        # Encoder path
        enc1 = self.encoder1(input)
        enc2 = self.encoder2(F.max_pool1d(enc1, 2))
        enc3 = self.encoder3(F.max_pool1d(enc2, 2))
        enc4 = self.encoder4(F.max_pool1d(enc3, 2))
        
        # Decoder path
        enc4 = self.upsample1(enc4)
        dec1 = self.decoder1(torch.cat([enc4, enc3], dim=1))
        dec1 = self.upsample2(dec1)
        dec2 = self.decoder2(torch.cat([dec1, enc2], dim=1))
        dec2 = self.upsample3(dec2)
        # dec2 = self.upsample3(enc2)
        dec3 = self.decoder3(torch.cat([dec2, enc1], dim=1))
        dec4 = self.decoder4(torch.cat([dec3, input], dim=1))
        return dec4.permute(0,2,1)

class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        ret = self._layer(x) * gate + bias
        return ret

class UnetConcatLinear(nn.Module):
    def __init__(self, context_dim, point_dim, context_dim_out, feature_dim, prior_len) -> None:
        super().__init__()
        # simple one, not change it first, later encoder should be detaily considered
        # self.encoder = nn.Linear(context_dim, context_dim_out)
        self.encoder = SimpleCNN(6, [16, 32], kernel_size=3)
        # self.concat1 = ConcatSquashLinear(point_dim, feature_dim, prior_len*context_dim_out+3)
        self.concat1 = ConcatSquashLinear(point_dim, feature_dim, 448+3)
        self.net = VanillaUnet(feature_dim)
        self.final_linear = nn.Sequential(
            nn.Linear(feature_dim, point_dim),
            # nn.Dropout(p=0.1)
        )
        time_emb_dim = 64
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, point_dim*2)
        )
        
    """
    x: future data to be sampled
    beta: temporal mark, part of feature
    context: prior information, 
    """
    def forward(self, beta, context, future):
    # def forward(self, context, future, t)
        batch_size = future.size(0)
        beta = beta.view(batch_size, 1, 1) # (B,1,1)
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)],dim=-1).to(future.device)

        # t = t.float().unsqueeze(-1)
        # t_emb = self.time_mlp(t)
        # t_emb = t_emb.unsqueeze(-1)

        context_encoded = self.encoder(context).view(batch_size, 1, -1)
        # print('check context', context_encoded.shape)
        # pdb.set_trace()
        
        # print('check tensor device', time_emb.device, context_encoded.device)
        ctx_emb = torch.cat([time_emb, context_encoded], dim=-1)
        
        
        """
        use context to filter future vector
        """
        future = self.concat1(ctx_emb, future)
        
        """
        get backbone's output
        """
        predict = self.net(future)
        
        """
        return to the input dimension
        """
        output = self.final_linear(predict)
        return output



"""
gaussian method
"""

class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim=None, num_heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * num_heads
        self.scale = dim_head ** -0.5  # 缩放因子，防止注意力分数过大
        
        # 如果未定义context_dim，则将其设置为与输入维度相同
        context_dim = context_dim if context_dim is not None else dim

        # Query、Key、Value的线性变换
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, self.inner_dim * 2, bias=False)  # 输出为2倍inner_dim, 用于分割Key和Value

        # 最后的输出线性层
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        # x: [batch_size, dim, seq_len]
        # context: [batch_size, context_dim, context_len]
        x, context = x.permute(0, 2, 1), context.permute(0, 2, 1)
        
        b, n, _ = x.shape
        context_len = context.shape[1]
        
        # Query, Key, Value 计算
        q = self.to_q(x)  # [batch_size, seq_len, num_heads * dim_head]
        k, v = self.to_kv(context).chunk(2, dim=-1)  # 分割Key和Value
        
        # 调整形状为多头形式
        q = q.view(b, n, self.num_heads, self.dim_head).transpose(1, 2)  # [batch_size, num_heads, seq_len, dim_head]
        k = k.view(b, context_len, self.num_heads, self.dim_head).transpose(1, 2)  # [batch_size, num_heads, context_len, dim_head]
        v = v.view(b, context_len, self.num_heads, self.dim_head).transpose(1, 2)  # [batch_size, num_heads, context_len, dim_head]
        
        # 计算注意力分数
        attn = torch.einsum('bhqd, bhkd -> bhqk', q * self.scale, k)  # [batch_size, num_heads, seq_len, context_len]
        attn = attn.softmax(dim=-1)  # 对最后一个维度（context_len）进行 softmax
        
        # 加权求和，得到输出
        out = torch.einsum('bhqk, bhvd -> bhqd', attn, v)  # [batch_size, num_heads, seq_len, dim_head]
        out = out.transpose(1, 2).reshape(b, n, self.inner_dim)  # 调整回 [batch_size, seq_len, num_heads * dim_head]
        
        # 线性变换输出
        out = self.to_out(out)
        return out.permute(0, 2, 1)

def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps (Tensor): a 1-D Tensor of N indices, one per batch element. These may be fractional.
        dim (int): the dimension of the output.
        max_period (int, optional): controls the minimum frequency of the embeddings. Defaults to 10000.

    Returns:
        Tensor: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def norm_layer(channels):
    return nn.GroupNorm(num_groups=32, num_channels=channels)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        """
        Attention block with shortcut

        Args:
            channels (int): channels
            num_heads (int, optional): attention heads. Defaults to 1.
        """
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0

        self.norm = norm_layer(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, L= x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B * self.num_heads, -1, L).chunk(3, dim=1)
        scale = 1.0 / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, L)
        h = self.proj(h)
        return h + x

class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
        else:
            self.op = nn.AvgPool1d(stride=2)

    def forward(self, x):
        return self.op(x)

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, ctx, x, t):
        """
        Apply the module to `x` given `t` timestep embeddings.
        """
        pass

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that support it as an extra input.
    """

    def forward(self, ctx, x, t):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(ctx, x, t)
            else:
                x = layer(x)
        return x

class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1 = nn.Sequential(norm_layer(in_channels), nn.SiLU(), nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))

        # self.conv1_ctx = nn.Sequential(norm_layer(in_channels), nn.SiLU(), nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
        
        # pojection for time step embedding
        self.time_emb = nn.Sequential(nn.SiLU(), nn.Linear(time_channels, out_channels))

        self.cross_attention = CrossAttention(dim=out_channels, context_dim=out_channels)
        
        # self.conv2 = nn.Sequential(norm_layer(out_channels*2), nn.SiLU(), nn.Dropout(p=dropout), nn.Conv1d(out_channels*2, out_channels, kernel_size=3, padding=1))
        self.conv2 = nn.Sequential(norm_layer(out_channels), nn.SiLU(), nn.Dropout(p=dropout), nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1))

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()


    def forward(self, ctx, x, t):
        """
        `ctx_x` has shape `[batch_size, in_dim, length]`
        `x` has shape `[batch_size, in_dim, length]`
        `t` has shape `[batch_size, time_dim]`
        """
        h = self.conv1(x)
        # h_ctx = self.conv1_ctx(ctx)
        # Add time step embeddings
        h += self.time_emb(t)[:, :, None]
        # att = self.cross_attention(h, ctx)
        # h += att
        # print('check 646', h.shape, ctx.shape)
        """
        if context, then use concat h and ctx
        """
        # h = torch.concat((h, ctx), dim=1)
        # pdb.set_trace()
        h = self.conv2(h)
        return h + self.shortcut(x)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding
    """

    def __init__(
        self,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0,
        channel_mult=(1, 1, 1, 1),
        conv_resample=True,
        num_heads=4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads

        # time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.ctx_conv1 = nn.Conv1d(in_channels, model_channels, kernel_size=3, padding=1)
        self.ctx_conv2 = nn.Conv1d(in_channels, model_channels*2, kernel_size=3, padding=1)
        
        # down blocks
        self.down_blocks = nn.ModuleList([TimestepEmbedSequential(nn.Conv1d(in_channels, model_channels, kernel_size=3, padding=1))])
        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1:  # don't use downsample for the last stage
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2

        # middle block
        self.middle_block = TimestepEmbedSequential(ResidualBlock(ch, ch, time_embed_dim, dropout), AttentionBlock(ch, num_heads=num_heads), ResidualBlock(ch, ch, time_embed_dim, dropout))
        # up blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [ResidualBlock(ch + down_block_chans.pop(), model_channels * mult, time_embed_dim, dropout)]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv1d(model_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, context: torch.FloatTensor, x: torch.FloatTensor, timesteps: torch.LongTensor):
        """Apply the model to an input batch.

        Args:
            context (Tensor): [N x C x L]
            x (Tensor): [N x C x L]
            timesteps (Tensor): [N,] a 1-D batch of timesteps.

        Returns:
            Tensor: [N x C x ...]
        """
        hs = []
        # down stage
        h: torch.FloatTensor = x
        h_ctx: torch.FloatTensor = context
        t: torch.FloatTensor = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        h_ctx_1 = self.ctx_conv1(h_ctx)
        # h_ctx_2 = self.ctx_conv2(h_ctx)
        item = 0
        for module in self.down_blocks:
            # print('Down sample', item)
            item += 1
            h = module(h_ctx_1, h, t)
            hs.append(h)
        # middle stage
        h = self.middle_block(h_ctx_1, h, t)
        # up stage
        item = 0
        for module in self.up_blocks:
            # print('Up sample', item, h.shape, hs[-1].shape)
            item += 1
            cat_in = torch.cat([h, hs.pop()], dim=1)
            # print('Cat_in', cat_in.shape)
            h = module(h_ctx_1, cat_in, t)
        return self.out(h)

    # def forward(self, x: torch.FloatTensor, t: torch.LongTensor, y: torch.LongTensor):
    def forward_trend(self, context: torch.FloatTensor, x: torch.FloatTensor, timesteps: torch.LongTensor, trend:None):
        """Apply the model to an input batch.

        Args:
            context (Tensor): [N x C x L]
            x (Tensor): [N x C x L]
            timesteps (Tensor): [N,] a 1-D batch of timesteps.

        Returns:
            Tensor: [N x C x ...]
        """
        hs = []
        # down stage
        h: torch.FloatTensor = x
        h_ctx: torch.FloatTensor = context
        t: torch.FloatTensor = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        alpha = torch.rand(1, device=context.device)
        if trend is not None:
            h_ctx = alpha * h_ctx + (1-alpha) * trend
        else:
            h_ctx = alpha * h_ctx
        
        h_ctx_1 = self.ctx_conv1(h_ctx)
        # h_ctx_2 = self.ctx_conv2(h_ctx)
        item = 0
        for module in self.down_blocks:
            # print('Down sample', item)
            item += 1
            h = module(h_ctx_1, h, t)
            hs.append(h)
        # middle stage
        h = self.middle_block(h_ctx_1, h, t)
        # up stage
        item = 0
        for module in self.up_blocks:
            # print('Up sample', item, h.shape, hs[-1].shape)
            item += 1
            cat_in = torch.cat([h, hs.pop()], dim=1)
            # print('Cat_in', cat_in.shape)
            h = module(h_ctx_1, cat_in, t)
        return self.out(h)
        


"""
gaussian method for generate
"""

class UNetGenerate(nn.Module):
    """
    The full UNet model with attention and timestep embedding
    """

    def __init__(
        self,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0,
        channel_mult=(1, 1, 1, 1),
        conv_resample=True,
        num_heads=4,
        label_num = 12,
        seq_length = 24
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads

        # time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # # context embedding
        self.ctx_conv1 = nn.Conv1d(in_channels, model_channels, kernel_size=3, padding=1)
        # self.ctx_conv2 = nn.Conv1d(in_channels, model_channels*2, kernel_size=3, padding=1)
        
        
        # condition embedding
        # cond_dim = time_embed_dim
        self.label_embedding = nn.Embedding(label_num, seq_length)
        self.label_conv = nn.Sequential(nn.Conv1d(1, in_channels, kernel_size=3, stride=1, padding=1))
        # self.to_label_tokens = nn.Linear(cond_dim, seq_length * in_channels)
        
        # down blocks
        self.down_blocks = nn.ModuleList([TimestepEmbedSequential(nn.Conv1d(in_channels, model_channels, kernel_size=3, padding=1))])
        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1:  # don't use downsample for the last stage
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2

        # middle block
        self.middle_block = TimestepEmbedSequential(ResidualBlock(ch, ch, time_embed_dim, dropout), AttentionBlock(ch, num_heads=num_heads), ResidualBlock(ch, ch, time_embed_dim, dropout))
        # up blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [ResidualBlock(ch + down_block_chans.pop(), model_channels * mult, time_embed_dim, dropout)]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv1d(model_channels, out_channels, kernel_size=3, padding=1),
        )

    # def forward(self, x: torch.FloatTensor, t: torch.LongTensor, y: torch.LongTensor):
    # def forward(self, context: torch.FloatTensor, x: torch.FloatTensor, timesteps: torch.LongTensor):
    def forward(self, label: torch.LongTensor, x: torch.FloatTensor, timesteps: torch.LongTensor):
        """Apply the model to an input batch.

        Args:
            label (Tensor): [N,] acitvity labels
            x (Tensor): [N x C x L]
            timesteps (Tensor): [N,] a 1-D batch of timesteps.

        Returns:
            Tensor: [N x C x ...]
        """
        # label embedding
        label = self.label_embedding(label).unsqueeze(1)
        label_ctx = self.label_conv(label)
        
        
        # print('check label shape', label_ctx.shape)
        hs = []
        # down stage
        h: torch.FloatTensor = x
        # h_ctx: torch.FloatTensor = context
        t: torch.FloatTensor = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        h_ctx_1 = self.ctx_conv1(label_ctx)
        # h_ctx_2 = self.ctx_conv2(h_ctx)
        item = 0
        for module in self.down_blocks:
            # print('Down sample', item)
            item += 1
            h = module(h_ctx_1, h, t)
            hs.append(h)
        # middle stage
        h = self.middle_block(h_ctx_1, h, t)
        # up stage
        item = 0
        for module in self.up_blocks:
            # print('Up sample', item, h.shape, hs[-1].shape)
            item += 1
            cat_in = torch.cat([h, hs.pop()], dim=1)
            # print('Cat_in', cat_in.shape)
            h = module(h_ctx_1, cat_in, t)
        return self.out(h)


# Example usage
if __name__ == "__main__":
    # Example input: batch_size=32, sequence_length=128, num_channels=6
    x = torch.randn(32, 6, 256)
    ctx = torch.randn(32, 6, 128)
    label = torch.randint(0, 11, (32,), device='cpu').long()
    t = torch.randint(0, 20, (32,), device='cpu').long()
    # model = SimpleCNN(in_channel=6, num_channels=[16, 32], kernel_size=3, dropout=0)
    model = UNetGenerate(in_channels=6, out_channels=6)
    # model = UNetModel(in_channels=6, out_channels=6)
    output = model(label, x, t)
    # output = model(ctx, x, t)
    
    print("Output shape:", output.shape)  # Output should be: (32, 64)
        