# modules.py
# goal: encoder components to map (b,c,t) -> (b, L, d)

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv1d(nn.Module):
    '''
    per-channel temporal filtering + 1x1 mixing
    '''
    def __init__(self, in_ch: int, out_ch: int, k: int=7, s: int=1, d:int=1):
        super().__init__()
        self.depthwise = nn.Conv1d(in_ch, in_ch, k, s, (k-1)//2, d, groups=in_ch)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.norm = nn.GroupNorm(1, out_ch)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ConvStem(nn.Module):
    '''
    input: (b,c,t)
    output: (b, hidden, t//2)
    '''
    def __init__(self, in_ch: int, hidden: int=256, k: int=7):
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
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv_in(x)
        x = self.norm(x)
        x = self.act(x)
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
    
class PatchProject(nn.Module):
    '''
    projection after patchify to match transformer dim
    (b, L, hidden*ps) -> (b, L, dim)
    '''
    def __init__(self, in_dim:int, dim:int):
        super().__init__()
        self.proj = nn.Linear(
            in_features=in_dim,
            out_features=dim,
            bias=False
        )

    def forward(self, tokens):
        return self.proj(tokens)

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
        self.postconv = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        

    def forward(self, x:torch.Tensor)->torch.Tensor:
        _,L,_ = x.shape
        if L > self.max_len:
            raise ValueError(f"sequence length {L} > max len {self.max_len}")
        # slice positional embedding for L
        pos = self.pe[:, :L, :]
        pos = self.postconv(pos.transpose(1,2)).transpose(1,2)
        return x + pos


class MLP(nn.Module):
    '''
    token mlp
    '''
    def __init__(self, d:int=256, mlp_ratio:int=4, pdrop:float=0.1):
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
        x = self.drop(x)
        x = self.linear_compress(x)
        return x
    

class EncoderBlock(nn.Module):
    '''
    pre-ln mha + mlp
    '''
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
    

class Encoder1D(nn.Module):
    '''
    (b,c,t) -> (b,L,d)
    '''
    def __init__(self, c:int, hidden:int, patch:int, dim:int, depth:int, heads:int, mlp_ratio:int, max_len:int):
        super().__init__()
        self.stem = ConvStem(c, hidden)
        self.patch = Patchify1D(patch)
        self.proj = PatchProject(hidden*patch, dim)
        self.pos = PositionalEmbedding(max_len, dim)
        self.tr = TransformerEncoder1D(dim, depth, heads, mlp_ratio)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.stem(x)
        tokens = self.patch(x)
        tokens = self.proj(tokens)
        tokensp = self.pos(tokens)
        tokenst = self.tr(tokensp)
        tokens_norm = self.ln(tokenst)
        return tokens_norm




