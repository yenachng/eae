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

    input: tokens (b, L, dim)
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
        _,L,_ = x.shape
        if L > self.max_len:
            raise ValueError(f"sequence length {L} > max len {self.max_len}")
        # slice positional embedding for L
        pos = self.pe[:, :L, :]
        return x + pos


class MLP(nn.Module):
    def __init__(self, d:int, mlp_ratio:int=4, pdrop:float=0.1):
        super().__init__()
        self.linear_expand = nn.Linear(
            in_features=d,
            out_features=d*mlp_ratio,
            bias=True
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=pdrop)
        self.linear_compress = nn.Linear(
            in_features=d*mlp_ratio,
            out_features=d,
            bias=True
        )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.linear_expand(x)
        x = self.act(x)
        x = self.linear_compress(x)
        return x
    

class EncoderBlock(nn.Module):
    def __init__(self, dim:int, heads:int, mlp_ratio:float, pdrop:float=0.1, p_attn:float=0.1):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        # rescale features per token to have mean 0, variance 1
        self.ln1 = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True,
            dropout=p_attn
        )
        self.drop1 = nn.Dropout(pdrop)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)
        self.drop2 = nn.Dropout(pdrop)
    
    def forward(self, x:torch.Tensor):
        # attention
        x_norm = self.ln1(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        x = x + self.drop1(attn_out)

        # mlp
        x_norm = self.ln2(x)
        mlp_out = self.mlp(x_norm)
        x = x + self.drop2(mlp_out) # residual add

        return x


class TransformerEncoder1D(nn.Module):
    def __init__(self, dim:int, depth:int, heads:int, mlp_ratio:float, pdrop:float=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([EncoderBlock(
            dim=dim,
            heads=heads,
            mlp_ratio=mlp_ratio,
            pdrop=pdrop
        ) for _ in range(depth)])

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)
        return x
    








