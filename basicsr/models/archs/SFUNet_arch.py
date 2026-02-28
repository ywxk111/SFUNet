import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import numpy as np

from einops import rearrange

from timm.models.layers import to_2tuple, trunc_normal_

import torch.utils.checkpoint as checkpoint


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        #  relative position encoding
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SpatialTransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=8,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer('attn_mask', attn_mask)

    def calculate_mask(self, x_size):
        # calculate mask for shift
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x):
        b, c, h, w = x.shape
        x = to_3d(x)
        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)
        # padding
        size_par = self.window_size
        pad_l = pad_t = 0
        pad_r = (size_par - w % size_par) % size_par
        pad_b = (size_par - h % size_par) % size_par
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hd, Wd, _ = x.shape
        x_size = (Hd, Wd)

        if min(x_size) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(x_size)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)

        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, Hd, Wd)  # b h' w' c

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        # remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()
        x = x.view(b, h * w, c)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = to_4d(x, h, w)

        return x



class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class ChannelTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(ChannelTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x



class DWT(nn.Module):
    def __init__(self, wave='db4'):
        super(DWT, self).__init__()
        low_filter = np.array([0.1629, 0.5055, 0.4464, -0.0198, -0.1323, 0.0218, 0.0233, -0.0075])
        high_filter = np.array([-0.0075, -0.0233, 0.0218, 0.1323, -0.0198, -0.4464, 0.5055, -0.1629])
        
        ll_filter = self._create_2d_filter(low_filter, low_filter)
        lh_filter = self._create_2d_filter(low_filter, high_filter)
        hl_filter = self._create_2d_filter(high_filter, low_filter)
        hh_filter = self._create_2d_filter(high_filter, high_filter)

        self.register_buffer('ll_filter', torch.from_numpy(ll_filter).float().unsqueeze(0).unsqueeze(0))
        self.register_buffer('lh_filter', torch.from_numpy(lh_filter).float().unsqueeze(0).unsqueeze(0))
        self.register_buffer('hl_filter', torch.from_numpy(hl_filter).float().unsqueeze(0).unsqueeze(0))
        self.register_buffer('hh_filter', torch.from_numpy(hh_filter).float().unsqueeze(0).unsqueeze(0))
    
    def _create_2d_filter(self, filter1, filter2):
        return np.outer(filter1, filter2)

    def forward(self, x):
        b, c, h, w = x.shape
        
        # Apply filters to each channel
        ll = F.conv2d(x, self.ll_filter.repeat(c, 1, 1, 1), stride=2, groups=c)
        lh = F.conv2d(x, self.lh_filter.repeat(c, 1, 1, 1), stride=2, groups=c)
        hl = F.conv2d(x, self.hl_filter.repeat(c, 1, 1, 1), stride=2, groups=c)
        hh = F.conv2d(x, self.hh_filter.repeat(c, 1, 1, 1), stride=2, groups=c)
        
        # Concatenate frequency components
        return torch.cat([ll, lh, hl, hh], dim=1)


class IDWT(nn.Module):
    def __init__(self, wave='db4'):
        super(IDWT, self).__init__()
        low_filter = np.array([0.1629, 0.5055, 0.4464, -0.0198, -0.1323, 0.0218, 0.0233, -0.0075])
        high_filter = np.array([-0.0075, -0.0233, 0.0218, 0.1323, -0.0198, -0.4464, 0.5055, -0.1629])
        
        ll_filter = self._create_2d_filter(low_filter, low_filter)
        lh_filter = self._create_2d_filter(low_filter, high_filter)
        hl_filter = self._create_2d_filter(high_filter, low_filter)
        hh_filter = self._create_2d_filter(high_filter, high_filter)

        self.register_buffer('ll_filter', torch.from_numpy(ll_filter).float().unsqueeze(0).unsqueeze(0))
        self.register_buffer('lh_filter', torch.from_numpy(lh_filter).float().unsqueeze(0).unsqueeze(0))
        self.register_buffer('hl_filter', torch.from_numpy(hl_filter).float().unsqueeze(0).unsqueeze(0))
        self.register_buffer('hh_filter', torch.from_numpy(hh_filter).float().unsqueeze(0).unsqueeze(0))
    
    def _create_2d_filter(self, filter1, filter2):
        return np.outer(filter1, filter2)

    def forward(self, x):
        b, c, h, w = x.shape
        assert c % 4 == 0, "Input channels must be divisible by 4"
        
        c_out = c // 4
        ll = x[:, :c_out]
        lh = x[:, c_out:2*c_out]
        hl = x[:, 2*c_out:3*c_out]
        hh = x[:, 3*c_out:]
        
        # Transpose convolution to upsample
        ll_up = F.conv_transpose2d(ll, self.ll_filter.repeat(c_out, 1, 1, 1), stride=2, groups=c_out)
        lh_up = F.conv_transpose2d(lh, self.lh_filter.repeat(c_out, 1, 1, 1), stride=2, groups=c_out)
        hl_up = F.conv_transpose2d(hl, self.hl_filter.repeat(c_out, 1, 1, 1), stride=2, groups=c_out)
        hh_up = F.conv_transpose2d(hh, self.hh_filter.repeat(c_out, 1, 1, 1), stride=2, groups=c_out)
        
        return ll_up + lh_up + hl_up + hh_up



class EfficientWaveletConvBlock(nn.Module):
    def __init__(self, dim, num_layers=2, wavelet_type='db4', bias=False, reduction_ratio=4):
        super(EfficientWaveletConvBlock, self).__init__()
        
        self.dwt = DWT(wavelet_type)
        self.idwt = IDWT(wavelet_type)
        
        self.reduction_ratio = reduction_ratio
        compressed_dim = max(dim // reduction_ratio, 8)  
        
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.conv_layers.append(nn.Sequential(
                    nn.Conv2d(dim * 4, compressed_dim * 4, 1, bias=bias),
                    nn.GELU(),
                    nn.Conv2d(compressed_dim * 4, compressed_dim * 4, 3, padding=1, groups=compressed_dim, bias=bias),
                    nn.GELU(),
                    nn.Conv2d(compressed_dim * 4, dim * 4, 1, bias=bias),
                ))
            else:
                self.conv_layers.append(nn.Sequential(
                    nn.Conv2d(dim * 4, compressed_dim * 4, 1, bias=bias),
                    nn.GELU(),
                    nn.Conv2d(compressed_dim * 4, compressed_dim * 4, 3, padding=1, groups=compressed_dim, bias=bias),
                    nn.GELU(),
                    nn.Conv2d(compressed_dim * 4, dim * 4, 1, bias=bias),
                ))
        
        self.feature_adapt = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, 1, bias=bias)
        )
        
    def forward(self, x):
        residual = x
        
        x_wavelet = self.dwt(x)  # [B, C*4, H/2, W/2]
        
        for conv_layer in self.conv_layers:
            x_wavelet = x_wavelet + conv_layer(x_wavelet)
        
        x_reconstructed = self.idwt(x_wavelet)  # [B, C, H, W]
        
        x_out = self.feature_adapt(x_reconstructed) + residual
        
        return x_out



class WaveletConvBlock(nn.Module):
    def __init__(self, dim, num_layers=2, wavelet_type='db4', bias=False):
        super(WaveletConvBlock, self).__init__()
        
        self.dwt = DWT(wavelet_type)
        self.idwt = IDWT(wavelet_type)
        
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.conv_layers.append(nn.Sequential(
                    nn.Conv2d(dim * 4, dim * 2, 3, padding=1, bias=bias),
                    nn.GELU(),
                    nn.Conv2d(dim * 2, dim * 4, 3, padding=1, bias=bias),
                ))
            else:
                self.conv_layers.append(nn.Sequential(
                    nn.Conv2d(dim * 4, dim * 2, 3, padding=1, bias=bias),
                    nn.GELU(),
                    nn.Conv2d(dim * 2, dim * 4, 3, padding=1, bias=bias),
                ))
        
        self.feature_adapt = nn.Conv2d(dim, dim, 1, bias=bias)
        
    def forward(self, x):
        residual = x
        
        x_wavelet = self.dwt(x)  # [B, C*4, H/2, W/2]
        
        for conv_layer in self.conv_layers:
            x_wavelet = x_wavelet + conv_layer(x_wavelet)
        
        x_reconstructed = self.idwt(x_wavelet)  # [B, C, H, W]
        
        x_out = self.feature_adapt(x_reconstructed) + residual
        
        return x_out


class AdaptiveWaveletConvBlock(nn.Module):
    def __init__(self, dim, num_layers=2, wavelet_types=['db4'], bias=False):
        super(AdaptiveWaveletConvBlock, self).__init__()
        
        self.wavelet_blocks = nn.ModuleList([
            WaveletConvBlock(dim, num_layers, wavelet_type=wt, bias=bias) 
            for wt in wavelet_types
        ])
        
        self.wavelet_selector = nn.Parameter(torch.ones(len(wavelet_types)))
        
        nn.init.constant_(self.wavelet_selector, 1.0)
    
    def forward(self, x):
        weights = F.softmax(self.wavelet_selector, dim=0)
        
        outputs = [block(x) for block in self.wavelet_blocks]
        
        output = sum(w * out for w, out in zip(weights, outputs))
        
        return output



class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)



class LightweightStableGatedFusionModule(nn.Module):
    def __init__(self, dim, alpha=1.0, beta=1.0):
        super(LightweightStableGatedFusionModule, self).__init__()
        self.alpha = alpha
        self.beta = beta
        
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  
            nn.Conv2d(dim * 2, dim // 4, 1, bias=False),  
            nn.GELU(),  
            nn.Conv2d(dim // 4, 1, 1, bias=False),  
            nn.Sigmoid()
        )
        
        self.spatial_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.wavelet_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.constant_(self.gate_net[-2].weight, 0.0)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, spatial_feat, wavelet_feat):
        spatial_feat = torch.clamp(spatial_feat, -10, 10)
        wavelet_feat = torch.clamp(wavelet_feat, -10, 10)
        
        concat_feat = torch.cat([spatial_feat, wavelet_feat], dim=1)
        
        gate_alpha = self.gate_net(concat_feat)
        
        gate_alpha = torch.clamp(gate_alpha, 0.1, 0.9)
        
        spatial_processed = self.spatial_conv(spatial_feat)
        wavelet_processed = self.wavelet_conv(wavelet_feat)
        
        spatial_out = spatial_feat + self.alpha * gate_alpha * wavelet_processed
          
        wavelet_out = wavelet_feat + self.beta * (1 - gate_alpha) * spatial_processed
        
        return spatial_out, wavelet_out


class MemoryEfficientGatedFusionModule(nn.Module):
    def __init__(self, dim, alpha=1.0, beta=1.0):
        super(MemoryEfficientGatedFusionModule, self).__init__()
        self.alpha = alpha
        self.beta = beta
        
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim // 4, 1, bias=False),
            nn.GELU(),  
            nn.Conv2d(dim // 4, 1, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.spatial_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.wavelet_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.constant_(self.gate_net[-2].weight, 0.0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, spatial_feat, wavelet_feat):
        return self._forward_impl(spatial_feat, wavelet_feat)
    
    def _forward_impl(self, spatial_feat, wavelet_feat):
        spatial_feat = torch.clamp(spatial_feat, -10, 10)
        wavelet_feat = torch.clamp(wavelet_feat, -10, 10)
        
        concat_feat = torch.cat([spatial_feat, wavelet_feat], dim=1)
        
        gate_alpha = self.gate_net(concat_feat)
        
        gate_alpha = torch.clamp(gate_alpha, 0.1, 0.9)
        
        spatial_processed = self.spatial_conv(spatial_feat)
        wavelet_processed = self.wavelet_conv(wavelet_feat)
        
        spatial_out = spatial_feat + self.alpha * gate_alpha * wavelet_processed
        wavelet_out = wavelet_feat + self.beta * (1 - gate_alpha) * spatial_processed
        
        return spatial_out, wavelet_out


class SimpleFusionModule(nn.Module):
    def __init__(self, dim):
        super(SimpleFusionModule, self).__init__()
        self.dim = dim
    
    def forward(self, spatial_feat, wavelet_feat):
        spatial_out = spatial_feat + wavelet_feat
        wavelet_out = spatial_feat + wavelet_feat
        
        return spatial_out, wavelet_out


class PassthroughFusionModule(nn.Module):
    def __init__(self):
        super(PassthroughFusionModule, self).__init__()

    def forward(self, spatial_feat, wavelet_feat):
        return spatial_feat, wavelet_feat

class ProgressiveTrainingScheduler:
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.current_epoch = 0
    
    def get_gate_lr_scale(self):
        if self.current_epoch < 0.3 * self.total_epochs:
            return 0.1
        elif self.current_epoch < 0.7 * self.total_epochs:
            return 0.1 + 0.9 * (self.current_epoch - 0.3 * self.total_epochs) / (0.4 * self.total_epochs)
        else:
            return 1.0
    
    def step(self):
        self.current_epoch += 1



class SFUNet(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3,
        img_size = 128,
        dim = 48,
        num_blocks = [2,4,4], 
        spatial_num_blocks = [2,4,4,6],
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        window_size=[16,16,16,16],
        drop_path_rate=0.1,
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',
        dual_pixel_task = False,
        wavelet_type = 'db4',
        use_lightweight_gate = True,      
        use_memory_efficient = True,      
        use_adaptive_wavelet = False,     
        wavelet_types = ['db4'],          
        use_lightweight_wavelet = False,  
        reduction_ratio = 4,              
        fusion_enabled = True,            
        fusion_stages = None              
    ):

        super(XformerWaveletV2, self).__init__()
        self.alpha = 1
        self.beta = 1


        self.gated_fusion_modules = nn.ModuleList()

        if fusion_stages is None:
            enabled_stage_set = {0, 1, 2, 3} if fusion_enabled else set()
        else:
            enabled_stage_set = set(fusion_stages) if fusion_enabled else set()

        def make_fusion_module(stage_idx: int, ch: int) -> nn.Module:
            if stage_idx not in enabled_stage_set:
                return PassthroughFusionModule()
            if not use_lightweight_gate:
                return SimpleFusionModule(ch)
            if use_memory_efficient:
                return MemoryEfficientGatedFusionModule(ch)
            return LightweightStableGatedFusionModule(ch)

        self.gated_fusion_modules.append(make_fusion_module(0, dim * 2))       # Level 2 (encoder)
        self.gated_fusion_modules.append(make_fusion_module(1, dim * 2 ** 2))  # Level 3 (encoder)
        self.gated_fusion_modules.append(make_fusion_module(2, dim * 2))       # Level 2 (decoder)
        self.gated_fusion_modules.append(make_fusion_module(3, dim))           # Level 1 (decoder)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(spatial_num_blocks))]

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)


        self.spatial_encoder1 = nn.Sequential(*[
            SpatialTransformerBlock(dim=dim, input_resolution=(img_size, img_size),
                             num_heads=heads[0], window_size=window_size[0], shift_size=0 if (i % 2 == 0) else window_size[0] // 2,
                             mlp_ratio=ffn_expansion_factor,
                             drop_path=dpr[sum(spatial_num_blocks[:0]):sum(spatial_num_blocks[:1])][i]
                             ) for i in range(spatial_num_blocks[0])])

        self.spatial_down1_2 = Downsample(dim)
        self.spatial_encoder2 = nn.Sequential(*[
            SpatialTransformerBlock(dim=int(dim * 2 ** 1), input_resolution=(img_size//2, img_size//2),
                             num_heads=heads[1], window_size=window_size[1], shift_size=0 if (i % 2 == 0) else window_size[1] // 2,
                             mlp_ratio=ffn_expansion_factor,
                             drop_path=dpr[sum(spatial_num_blocks[:1]):sum(spatial_num_blocks[:2])][i]) for i in range(spatial_num_blocks[1])])

        self.spatial_down2_3 = Downsample(int(dim * 2 ** 1))
        self.spatial_encoder3 = nn.Sequential(*[
            SpatialTransformerBlock(dim=int(dim * 2 ** 2),input_resolution=(img_size//4, img_size//4), num_heads=heads[2], window_size=window_size[2], shift_size=0 if (i % 2 == 0) else window_size[2] // 2,
                             mlp_ratio=ffn_expansion_factor,
                             drop_path=dpr[sum(spatial_num_blocks[:2]):sum(spatial_num_blocks[:3])][i]) for i in range(spatial_num_blocks[2])])

        self.spatial_down3_4 = Downsample(int(dim * 2 ** 2))
        self.spatial_latent = nn.Sequential(*[
            SpatialTransformerBlock(dim=int(dim * 2 ** 3), input_resolution=(img_size//8, img_size//8),num_heads=heads[3], window_size=window_size[3], shift_size=0 if (i % 2 == 0) else window_size[3] // 2,
                             mlp_ratio=ffn_expansion_factor,
                             drop_path=dpr[sum(spatial_num_blocks[:3]):sum(spatial_num_blocks[:4])][i]) for i in range(spatial_num_blocks[3])])


        def create_wavelet_block(dim, num_layers, wavelet_type, bias):
            if use_adaptive_wavelet:
                return AdaptiveWaveletConvBlock(dim, num_layers, wavelet_types=wavelet_types, bias=bias)
            elif use_lightweight_wavelet:
                return EfficientWaveletConvBlock(dim, num_layers, wavelet_type, bias, reduction_ratio)
            else:
                return WaveletConvBlock(dim, num_layers, wavelet_type, bias)
        
        # Level 1
        if num_blocks[0] > 0:
            self.wavelet_encoder1 = nn.Sequential(*[create_wavelet_block(dim, 2, wavelet_type, bias) for i in range(num_blocks[0])])
        else:
            self.wavelet_encoder1 = nn.Identity()
        
        # Level 2
        if num_blocks[1] > 0:
            self.wavelet_down1_2 = Downsample(dim)
            self.wavelet_encoder2 = nn.Sequential(*[create_wavelet_block(int(dim*2**1), 2, wavelet_type, bias) for i in range(num_blocks[1])])
        else:
            self.wavelet_down1_2 = nn.Identity()
            self.wavelet_encoder2 = nn.Identity()
        
        # Level 3
        if num_blocks[2] > 0:
            self.wavelet_down2_3 = Downsample(int(dim*2**1))
            self.wavelet_encoder3 = nn.Sequential(*[create_wavelet_block(int(dim*2**2), 2, wavelet_type, bias) for i in range(num_blocks[2])])
        else:
            self.wavelet_down2_3 = nn.Identity()
            self.wavelet_encoder3 = nn.Identity()
        
        # Level 4 (Latent)
        if any(num_blocks) > 0:  
            self.wavelet_down3_4 = Downsample(int(dim*2**2))
            self.wavelet_latent = nn.Sequential(*[create_wavelet_block(int(dim*2**3), 2, wavelet_type, bias) for i in range(2)])
        else:
            self.wavelet_down3_4 = nn.Identity()
            self.wavelet_latent = nn.Identity()



        # Level 3
        self.spatial_up4_3 = Upsample(int(dim*2**3))
        self.spatial_reduce3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.spatial_decoder3 = nn.Sequential(*[
            SpatialTransformerBlock(dim=int(dim*2**2), input_resolution=(img_size//4, img_size//4),
                             num_heads=heads[2], window_size=window_size[2], shift_size=0 if (i % 2 == 0) else window_size[2] // 2,
                             mlp_ratio=ffn_expansion_factor,
                             drop_path=dpr[sum(spatial_num_blocks[:2]):sum(spatial_num_blocks[:3])][i]) for i in range(spatial_num_blocks[2])])
        
        # Level 2
        self.spatial_up3_2 = Upsample(int(dim*2**2))
        self.spatial_reduce2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.spatial_decoder2 = nn.Sequential(*[
            SpatialTransformerBlock(dim=int(dim*2**1), input_resolution=(img_size//2, img_size//2),
                             num_heads=heads[1], window_size=window_size[1], shift_size=0 if (i % 2 == 0) else window_size[1] // 2,
                             mlp_ratio=ffn_expansion_factor,
                             drop_path=dpr[sum(spatial_num_blocks[:1]):sum(spatial_num_blocks[:2])][i]) for i in range(spatial_num_blocks[1])])
        
        # Level 1
        self.spatial_up2_1 = Upsample(int(dim*2**1))
        self.spatial_reduce1 = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.spatial_decoder1 = nn.Sequential(*[
            SpatialTransformerBlock(dim=dim, input_resolution=(img_size, img_size),
                             num_heads=heads[0], window_size=window_size[0], shift_size=0 if (i % 2 == 0) else window_size[0] // 2,
                             mlp_ratio=ffn_expansion_factor,
                             drop_path=dpr[sum(spatial_num_blocks[:0]):sum(spatial_num_blocks[:1])][i]) for i in range(spatial_num_blocks[0])])
        

        if any(num_blocks) > 0:  
            # Level 3
            self.wavelet_up4_3 = Upsample(int(dim*2**3))
            self.wavelet_reduce3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
            if num_blocks[2] > 0:
                self.wavelet_decoder3 = nn.Sequential(*[create_wavelet_block(int(dim*2**2), 2, wavelet_type, bias) for i in range(num_blocks[2])])
            else:
                self.wavelet_decoder3 = nn.Identity()
            
            # Level 2
            self.wavelet_up3_2 = Upsample(int(dim*2**2))
            self.wavelet_reduce2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
            if num_blocks[1] > 0:
                self.wavelet_decoder2 = nn.Sequential(*[create_wavelet_block(int(dim*2**1), 2, wavelet_type, bias) for i in range(num_blocks[1])])
            else:
                self.wavelet_decoder2 = nn.Identity()
            
            # Level 1
            self.wavelet_up2_1 = Upsample(int(dim*2**1))
            self.wavelet_reduce1 = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
            if num_blocks[0] > 0:
                self.wavelet_decoder1 = nn.Sequential(*[create_wavelet_block(dim, 2, wavelet_type, bias) for i in range(num_blocks[0])])
            else:
                self.wavelet_decoder1 = nn.Identity()
        else:
            self.wavelet_up4_3 = nn.Identity()
            self.wavelet_reduce3 = nn.Identity()
            self.wavelet_decoder3 = nn.Identity()
            self.wavelet_up3_2 = nn.Identity()
            self.wavelet_reduce2 = nn.Identity()
            self.wavelet_decoder2 = nn.Identity()
            self.wavelet_up2_1 = nn.Identity()
            self.wavelet_reduce1 = nn.Identity()
            self.wavelet_decoder1 = nn.Identity()


        if any(num_blocks) > 0 and any(spatial_num_blocks) > 0:

            refinement_dim = int(dim * 2 ** 1)  
        elif any(num_blocks) > 0:
            refinement_dim = int(dim * 2 ** 1)  
        elif any(spatial_num_blocks) > 0:
            refinement_dim = dim  
        else:
            refinement_dim = dim  
        

        self.refinement = nn.Sequential(*[ChannelTransformerBlock(dim=refinement_dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_refinement_blocks)])
        
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, refinement_dim, kernel_size=1, bias=bias)
            
        self.output = nn.Conv2d(refinement_dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self, inp_img):
        inp = self.patch_embed(inp_img)


        # Level 1
        spatial_enc1 = self.spatial_encoder1(inp)
        wavelet_enc1 = self.wavelet_encoder1(inp)
        
        # Level 2
        spatial_enc2_inp = self.spatial_down1_2(spatial_enc1)
        wavelet_enc2_inp = self.wavelet_down1_2(wavelet_enc1)
        
        if (hasattr(self, 'gated_fusion_modules') and len(self.gated_fusion_modules) > 0 and 
            not isinstance(self.wavelet_encoder1, nn.Identity) and 
            not isinstance(self.spatial_encoder1, nn.Identity)):
            spatial_enc2_inp, wavelet_enc2_inp = self.gated_fusion_modules[0](spatial_enc2_inp, wavelet_enc2_inp)
        
        spatial_enc2 = self.spatial_encoder2(spatial_enc2_inp)
        wavelet_enc2 = self.wavelet_encoder2(wavelet_enc2_inp)

        # Level 3
        spatial_enc3_inp = self.spatial_down2_3(spatial_enc2)
        wavelet_enc3_inp = self.wavelet_down2_3(wavelet_enc2)
        
        if (hasattr(self, 'gated_fusion_modules') and len(self.gated_fusion_modules) > 1 and 
            not isinstance(self.wavelet_encoder1, nn.Identity) and 
            not isinstance(self.spatial_encoder1, nn.Identity)):
            spatial_enc3_inp, wavelet_enc3_inp = self.gated_fusion_modules[1](spatial_enc3_inp, wavelet_enc3_inp)
        
        spatial_enc3 = self.spatial_encoder3(spatial_enc3_inp)
        wavelet_enc3 = self.wavelet_encoder3(wavelet_enc3_inp)
        
        # Level 4 (Latent)
        spatial_latent_inp = self.spatial_down3_4(spatial_enc3)
        wavelet_latent_inp = self.wavelet_down3_4(wavelet_enc3)
        
        spatial_latent = self.spatial_latent(spatial_latent_inp)
        wavelet_latent = self.wavelet_latent(wavelet_latent_inp)
 
        # Level 3
        spatial_dec3 = self.spatial_up4_3(spatial_latent)
        spatial_dec3 = torch.cat([spatial_dec3, spatial_enc3], 1)
        spatial_dec3 = self.spatial_reduce3(spatial_dec3)
        spatial_dec3 = self.spatial_decoder3(spatial_dec3)
        
        wavelet_dec3 = self.wavelet_up4_3(wavelet_latent)
        wavelet_dec3 = torch.cat([wavelet_dec3, wavelet_enc3], 1)
        wavelet_dec3 = self.wavelet_reduce3(wavelet_dec3)
        wavelet_dec3 = self.wavelet_decoder3(wavelet_dec3)

        # Level 2
        spatial_dec2 = self.spatial_up3_2(spatial_dec3)
        spatial_dec2 = torch.cat([spatial_dec2, spatial_enc2], 1)
        spatial_dec2 = self.spatial_reduce2(spatial_dec2)
        
        wavelet_dec2 = self.wavelet_up3_2(wavelet_dec3)
        wavelet_dec2 = torch.cat([wavelet_dec2, wavelet_enc2], 1)
        wavelet_dec2 = self.wavelet_reduce2(wavelet_dec2)
        
        if (hasattr(self, 'gated_fusion_modules') and len(self.gated_fusion_modules) > 2 and 
            not isinstance(self.wavelet_encoder1, nn.Identity) and 
            not isinstance(self.spatial_encoder1, nn.Identity)):
            spatial_dec2, wavelet_dec2 = self.gated_fusion_modules[2](spatial_dec2, wavelet_dec2)
        
        spatial_dec2 = self.spatial_decoder2(spatial_dec2)
        wavelet_dec2 = self.wavelet_decoder2(wavelet_dec2)
        
        # Level 1
        spatial_dec1 = self.spatial_up2_1(spatial_dec2)
        spatial_dec1 = torch.cat([spatial_dec1, spatial_enc1], 1)
        spatial_dec1 = self.spatial_reduce1(spatial_dec1)
        
        wavelet_dec1 = self.wavelet_up2_1(wavelet_dec2)
        wavelet_dec1 = torch.cat([wavelet_dec1, wavelet_enc1], 1)
        wavelet_dec1 = self.wavelet_reduce1(wavelet_dec1)
        
        if (hasattr(self, 'gated_fusion_modules') and len(self.gated_fusion_modules) > 3 and 
            not isinstance(self.wavelet_encoder1, nn.Identity) and 
            not isinstance(self.spatial_encoder1, nn.Identity)):
            spatial_dec1, wavelet_dec1 = self.gated_fusion_modules[3](spatial_dec1, wavelet_dec1)
        
        spatial_dec1 = self.spatial_decoder1(spatial_dec1)
        wavelet_dec1 = self.wavelet_decoder1(wavelet_dec1)
        

        spatial_branch_exists = hasattr(self, 'spatial_decoder1') and not isinstance(self.spatial_decoder1, nn.Identity)
        wavelet_branch_exists = hasattr(self, 'wavelet_decoder1') and not isinstance(self.wavelet_decoder1, nn.Identity)
        
        if spatial_branch_exists and wavelet_branch_exists:
            x = torch.cat([spatial_dec1, wavelet_dec1], 1)
        elif spatial_branch_exists:
            x = spatial_dec1
        elif wavelet_branch_exists:
            x = wavelet_dec1
        else:
            x = inp

        res = self.refinement(x)

        if self.dual_pixel_task:
            res = res + self.skip_conv(inp)
            res = self.output(res)
        else:
            res = self.output(res) + inp_img

        return res