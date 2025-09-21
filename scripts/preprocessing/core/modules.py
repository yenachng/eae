# modules.py
# goal: encoder components to map (b,c,t) -> (b, L, d)

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv1d(nn.Module):
    '''
    applies a depthwise conv (per channel temporal filtering) then a pointwise conv (channel mixing) then normalization and gelu
    input: (b, in_ch, t)
    output: (b, out_ch, t_out) with same length if matching padding
    params: kernel k, stride s, dilation d
    '''
    def __init__(self, in_ch: int, out_ch: int, k: int=7, s: int=1, d:int=1):
        super().__init__()
        self.depthwise = nn.Conv1d(in_ch, in_ch, k, s, (k-1)//2, d, groups=in_ch)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, s, 0, groups=1)
        self.norm = nn.GroupNorm(1, out_ch)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.gelu(x)
        return x


class ConvStem(nn.Module):
    '''
    input: (b,c,t)
    output: (b, hidden, t//2)

    first block: a regular conv -> local patterns
    second block: a depthwise separable conv with stride=2 -> halves time length
    norm + activation after each
    '''
    def __init__(self, in_ch: int, hidden: int, k: int=7):
        super().__init__()
        self.conv_in = nn.Conv1d(
            in_channels=in_ch,
            out_channels=hidden,
            kernel_size=k,
            stride=1,
            padding=(k-1)//2,
            bias=False
        )
        self.dwsep = DepthwiseSeparableConv1d(
            in_ch = hidden,
            out_ch=hidden,
            k=5,
            s=2,
            d=1
        )
        self.norm = nn.GroupNorm(1, hidden)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv_in(x)
        x = self.norm(x)
        x = self.gelu(x)
        x = self.dwsep(x)
        return x
    

class Patchify1D(nn.Module):
    '''
    splits the time axis into non-overlapping patches of length ps. flattens each patch into a token.

    input: h (b, hidden, t') from ConvStem
    parameter: patch_size=ps
    output: (tokens, L) with tokens in R^{b x L x (hidden*ps)} where L=floor(t'/ps)
    '''
    def __init__(self, patch_size:int):
        super().__init__()
        self.ps = int(patch_size)

    def forward(self, x):
        b, h, t = x.shape
        L = t//self.ps
        if L==0:
            raise ValueError(f"time length {t} is shorter than patch size {self.ps}")
        x = x[:, :, :L*self.ps]
        x = x.view(b, h, L, self.ps)
        x = x.permute(0,2,1,3)
        tokens = x.reshape(b, L, h*self.ps)
        return tokens
    

class PositionalEmbedding(nn.Module):
    '''
    informs the model where each token lies in time

    input: tokens (b, L, h*ps)
    output:
    '''
    def __init__(self, max_len: int, dim: int):
        super().__init__()
        self.max_len = max_len
        self.dim = dim
        # learnable parameter
        self.pe = nn.Parameter(torch.zeros(1, max_len, dim))
        # init with small random values (instead of all zeros)
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        b,L,d = x.shape
        if L > self.max_len:
            raise ValueError(f"sequence length {L} > max len {self.max_len}")
        # slice positional embedding for L
        pos = self.pe[:, :L, :]
        return x + pos


class MLP(nn.Module):
    def __init__(self, token: )
