import math
import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=2, window_size=1, qkv_bias=False, qk_scale=None, dropout=0., causal=True, device=None):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."
        self.dim = dim
        self.num_heads = heads
        self.causal = causal
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        if causal and window_size > 0:
            mask = torch.tril(torch.ones(window_size, window_size))
            if device is not None:
                mask = mask.to(device)
            self.register_buffer('causal_mask', mask)

    def compute_window_attention(self, windows_x):
        B, T, C = windows_x.shape
        qkv = self.qkv(windows_x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.causal:
            if hasattr(self, 'causal_mask') and self.causal_mask is not None and T == self.causal_mask.size(0):
                attn = attn.masked_fill_(self.causal_mask == 0, float("-inf"))
            else:
                temp_mask = torch.tril(torch.ones(T, T, device=windows_x.device))
                attn = attn.masked_fill_(temp_mask == 0, float("-inf"))
        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def split_into_windows(self, x):
        B_prev, T_prev, C_prev = x.shape
        windows = []
        for i in range(0, T_prev, self.window_size):
            end_idx = min(i + self.window_size, T_prev)
            window = x[:, i:end_idx, :]
            windows.append(window)
        return windows

    def forward(self, x):
        B_prev, T_prev, C_prev = x.shape
        if self.window_size > 0 and self.window_size < T_prev:
            windows = self.split_into_windows(x)
            output_windows = [self.compute_window_attention(w) for w in windows]
            x = torch.cat(output_windows, dim=1)
        else:
            x = self.compute_window_attention(x)
        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MultiLevelCausalAttention(nn.Module):
    def __init__(self, dim, depth, heads, window_size, mlp_dim, num_time, dropout=0., device=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TemporalAttention(dim=dim, heads=heads, window_size=window_size, dropout=dropout, device=device),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        b, c, n, t = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b * n, t, c)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = x.reshape(b, n, t, c).permute(0, 3, 1, 2)
        return x



class AttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads
        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
    def forward(self, query, key, value):
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]
        query, key, value = self.FC_Q(query), self.FC_K(key), self.FC_V(value)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)
        key = key.transpose(-1, -2)
        attn_score = (query @ key) / self.head_dim ** 0.5
        if self.mask:
            mask = torch.ones(tgt_length, src_length, dtype=torch.bool, device=query.device).tril()
            attn_score.masked_fill_(~mask, -torch.inf)
        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        out = self.out_proj(out)
        return out

def sinusoidal_encode(n_positions, d_model):
    position = torch.arange(n_positions, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(n_positions, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def apply_node_position_aware_encoding(x, channel):
    B, T, N, C = x.shape
    sinusoidal_pe = sinusoidal_encode(N, channel).to(x.device)
    sinusoidal_pe = sinusoidal_pe.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
    x_encoded = torch.cat((x, sinusoidal_pe), dim=-1)
    return x_encoded

class NodePositionAwareSpatialAttention(nn.Module):
    def __init__(self, model_dim, feed_forward_dim=256, num_heads=8, dropout=0, mask=False):
        super().__init__()
        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim), nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim)
        )
        self.ln1, self.ln2 = nn.LayerNorm(model_dim), nn.LayerNorm(model_dim)
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)
    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        residual = x
        out = self.attn(x, x, x)
        out = self.ln1(residual + self.dropout1(out))
        residual = out
        out = self.feed_forward(out)
        out = self.ln2(residual + self.dropout2(out))
        out = out.transpose(dim, -2)
        return out

class MultiLevelTemporalAttention(nn.Module):
    def __init__(self, model_dim, feed_forward_dim=256, num_heads=8, dropout=0, mask=True, blocks=3, seq_len=12, device=None):
        super().__init__()
        self.temporal_modules = nn.ModuleList()
        mlp_dim_temporal = model_dim * 2
        for b in range(blocks):
            window_size = seq_len // 2 ** (blocks - b - 1)
            self.temporal_modules.append(MultiLevelCausalAttention(
                dim=model_dim, depth=1, heads=num_heads, window_size=window_size,
                mlp_dim=mlp_dim_temporal, num_time=seq_len, device=device
            ))
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim), nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim)
        )
        self.ln1, self.ln2 = nn.LayerNorm(model_dim), nn.LayerNorm(model_dim)
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)
    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        residual = x
        x_temp = x.permute(0, 3, 1, 2)
        out = x_temp
        for i in range(len(self.temporal_modules)):
             out = self.temporal_modules[i](out)
        out = out.permute(0, 2, 3, 1)
        out = self.ln1(residual + self.dropout1(out))
        residual = out
        out = self.feed_forward(out)
        out = self.ln2(residual + self.dropout2(out))
        out = out.transpose(dim, -2)
        return out