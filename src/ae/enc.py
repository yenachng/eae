from __future__ import annotations
import torch
import torch.nn as nn

import mne

from dataclasses import dataclass

'''
(B, 1, T) : batch_size x channels x time steps
'''

@dataclass
class AEConfig:
    mont = mne.channels.make_standard_montage("GSN-HydroCel-128")
    ref:str = "E129"
    sfreq:float = 256.0
    kernel_size:int = 3
    in_channels:int = 1
    c1:int = 32
    c2:int = 64
    c3:int = 128
    patch_size:int = 4
    token_count:int = 128
    embed_dim:int = 128
    window_size:int = 512
    overlap:float = 0.0
    normalize:bool = True
    n_heads: int = 4
    mlp_ratio:int = 4
    p_drop = 0.1
    depth=4


class ConvStem1D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.kernel_size % 2 == 1, "kernel_size must be odd for same-length padding"
        padding = cfg.kernel_size//2
        # (b,1,t) -> (b, c1, t)
        self.conv1 = nn.Conv1d(
            in_channels=cfg.in_channels,
            out_channels=cfg.c1,
            kernel_size=cfg.kernel_size,
            padding=padding,
            bias=False
        )
        # (b, t, c1) -> (b, t, c1)
        self.ln1 = nn.LayerNorm(
            normalized_shape=cfg.c1
        )
        # (b, c1, t) -> (b, c1, t)
        self.act1 = nn.ReLU(inplace=True)
        # (b, c1, t) -> (b, c2, t)
        self.conv2 = nn.Conv1d(
            in_channels=cfg.c1,
            out_channels=cfg.c2,
            kernel_size=cfg.kernel_size,
            padding=padding,
            bias=False
        )
        # (b, t, c2) -> (b, t, c2)
        self.ln2 = nn.LayerNorm(
            normalized_shape=cfg.c2
        )
        # (b, c2, t) -> (b, c2, t)
        self.act2 = nn.ReLU(inplace=True)
        # (b, c2, t) -> (b, c3, t)
        self.conv3 = nn.Conv1d(
            in_channels=cfg.c2,
            out_channels=cfg.c3,
            kernel_size=cfg.kernel_size,
            padding=padding,
            bias=False
        )
        # (b, t, c2) -> (b, t, c2)
        self.ln3 = nn.LayerNorm(
            normalized_shape=cfg.c3
        )
        # (b, c2, t) -> (b, c2, t)
        self.act3 = nn.ReLU(inplace=True)


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # layer 1
        x1 = self.conv1(x)
        x1_t = x1.transpose(1,2)
        x1_norm_t = self.ln1(x1_t)
        x1_norm = x1_norm_t.transpose(1,2)
        x1_act = self.act1(x1_norm)

        # layer 2
        x2 = self.conv2(x1_act)
        x2_t = x2.transpose(1,2)
        x2_norm_t = self.ln2(x2_t)
        x2_norm = x2_norm_t.transpose(1,2)
        x2_act = self.act2(x2_norm)

        # layer 3
        x3 = self.conv3(x2_act)
        x3_t = x3.transpose(1,2)
        x3_norm_t = self.ln3(x3_t)
        x3_norm = x3_norm_t.transpose(1,2)
        x3_act = self.act3(x3_norm)

        # -> (b, c2, t)
        return x3_act


class Patchify(nn.Module):
    '''
    patchify (b,c2,t) into patch_size tokens over time
    '''
    def __init__(self, cfg):
        super().__init__()
        self.patch_size = int(cfg.patch_size)
        
    def forward(self, x:torch.Tensor):
        b,h,t = x.shape
        if t < self.patch_size:
            raise ValueError("window timeframe is shorter than patch size")
        patch_count = t//self.patch_size
        x = x[:, :, :patch_count*self.patch_size]
        x = x.view(b, h, patch_count, self.patch_size)
        patches = x.permute(0, 2, 1, 3) # -> batch x patch x (features x patch_size)
        return patches

class PatchProject(nn.Module):
    '''
    project [b, n_patch, n_feat, patch_size] -> [b, n_patch, token_size]
    '''
    def __init__(self, cfg):
        super().__init__()
        self.n_feat = cfg.c3
        self.patch_size = cfg.patch_size
        self.proj = nn.Linear(in_channels=self.n_feat*self.patch_size,
                              out_channels=self.embed_dim)

    def forward(self, patches:torch.Tensor):
        # patches: (b, p, n_feat, patch_size t')
        b, n_patch, n_feat, ps = patches.shape
        assert n_feat == self.n_feat and ps == self.patch_size, "patch shape mismatch"
        x = patches.contiguous().reshape(b * n_patch, n_feat*ps)
        x = self.proj(x)
        return x.view(b, n_patch, -1)


class MHA(nn.Module):
    '''
    wrapper for multi-head self-attention
    '''
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.embed_dim = cfg.embed_dim
        self.mha = nn.MultiheadAttention(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.n_heads,
            bias=True,
            batch_first=True
        )

    def forward(self, tokens:torch.Tensor, need_weights:bool=False):
        _, _, dim = tokens.shape
        assert dim == self.embed_dim, "input dim mismatch"
        out, w = self.mha(tokens, tokens, tokens, need_weights=need_weights)
        return out, w


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.linexp = nn.Linear(cfg.embed_dim, cfg.embed_dim*cfg.mlp_ratio, bias=False)
        self.act1 = nn.GELU()
        self.lincomp = nn.Linear(cfg.embed_dim*cfg.mlp_ratio, cfg.embed_dim, bias=True)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # layer 1: expand
        x = self.linexp(x)
        x = self.act1(x)
        # layer 2: compress
        x = self.lincomp(x)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        max_len = cfg.window_size//cfg.patch_size
        self.max_len = max_len
        self.embed = nn.Embedding(max_len, cfg.embed_dim)
    def forward(self,x:torch.Tensor)-> torch.Tensor:
        pos = self.embed(torch.arange(self.max_len, device=x.device))
        pos = pos[:x.shape[1]]
        return x + pos.unsqueeze(0)
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.embed_dim)
        self.mha = MHA(cfg)
        self.drop1 = nn.Dropout(cfg.p_drop)
        self.ln2 = nn.LayerNorm(cfg.embed_dim)
        self.mlp = MLP(cfg)
        self.drop2 = nn.Dropout(cfg.p_drop)

    def forward(self, x:torch.Tensor):
        x_norm = self.ln1(x)
        attn, weights = self.mha(x_norm)
        # residual update
        x = x + self.drop1(attn)

        x_norm = self.ln2(x)
        y = self.mlp(x_norm)
        return x + self.drop2(y)
    
class TransformerEncoder1D(nn.Module):
    def __init__(self, depth:int, cfg):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerEncoderLayer(cfg) for _ in range(depth)])

    def forward(self, x:torch.Tensor):
        for block in self.blocks:
            x = block(x)
        return x

class Encoder1D(nn.Module):
    def __init__(self, cfg, depth:int):
        super().__init__()
        self.stem = ConvStem1D(cfg)
        self.patchify = Patchify(cfg)
        self.tokenize = PatchProject(cfg)
        self.pos = PositionalEmbedding(cfg)
        self.transf = TransformerEncoder1D(depth, cfg)
        self.ln = nn.LayerNorm(cfg.embed_dim)

    def forward(self, x:torch.Tensor):
        x = self.stem(x)
        x_patched = self.patchify(x)
        x_tokens = self.tokenize(x_patched)
        tokens = self.pos(x_tokens)
        tokenst = self.transf(tokens)
        tokens_norm = self.ln(tokenst)
        return tokens_norm


class TokenDecoder(nn.Module):
    '''
    (b, p, d) -> (b, h, p*ps)
    '''
    def __init__(self, cfg):
        super().__init__()
        self.h = cfg.c3
        self.ps = cfg.patch_size
        self.d = cfg.embed_dim
        self.proj = nn.Linear(self.d, self.h*self.ps)
        self.head = nn.Conv1d(self.h, 1, kernel_size=1, bias=True)
    
    def forward(self, tokens):
        b,p,d = tokens.shape
        x = self.proj(tokens)
        x = x.view(b, p, self.h, self.ps)
        x = x.permute(0,2,1,3).reshape(b, self.h, p*self.ps) # unpatchified
        x = self.head(x)
        return x