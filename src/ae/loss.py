# clean modular losses for 1d eeg ae
# expects: encoder(x)->(b,p,d), decoder(tokens)->(b,1,t)

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# config

@dataclass
class LossCfg:
    # reconstruction
    lambda_stft: float = 0.1
    n_fft: int = 256
    hop: int = 128
    # stability
    lambda_tr: float = 1e-3
    tau: float = 1.0
    lambda_orth: float = 1e-3
    # profiling
    beta_sup: float = 0.0
    mu_ridge: float = 1e-2
    nu_ridge: float = 1e-2

# linalg helpers

def _center_rows(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=0, keepdim=True)

def cov_zz(Z_rows: torch.Tensor) -> torch.Tensor:
    zc = _center_rows(Z_rows)
    n = max(zc.shape[0], 1)
    return (zc.t() @ zc) / n

def cross_cov(Z_rows: torch.Tensor, Y_rows: torch.Tensor) -> torch.Tensor:
    zc = _center_rows(Z_rows)
    yc = _center_rows(Y_rows)
    n = max(zc.shape[0], 1)
    return (zc.t() @ yc) / n

def qr_retraction(W: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(W, mode='reduced')
    return q

class OrthoProjector(nn.Module):
    # maintains d x k basis
    def __init__(self, d: int, k: int):
        super().__init__()
        self.W_raw = nn.Parameter(torch.randn(d, k) / (d**0.5))
    def forward(self) -> torch.Tensor:
        return qr_retraction(self.W_raw)

# recon

def stft_logmag(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    X = torch.stft(
        x.squeeze(1),
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=torch.hann_window(n_fft, device=x.device),
        return_complex=True,
        center=True,
        pad_mode='reflect',
    )
    return torch.log(torch.abs(X).clamp_min(1e-8))

def loss_reconstruction(x: torch.Tensor, x_hat: torch.Tensor, cfg: LossCfg) -> torch.Tensor:
    l = F.mse_loss(x_hat, x)
    if cfg.lambda_stft > 0:
        l = l + cfg.lambda_stft * F.mse_loss(stft_logmag(x, cfg.n_fft, cfg.hop),
                                             stft_logmag(x_hat, cfg.n_fft, cfg.hop))
    return l

# supervised
# inputs: Z_rows (n,d), Y_rows (n,q), W (d,k)

def phi_sup(Z_rows: torch.Tensor, Y_rows: torch.Tensor, W: torch.Tensor, mu: float) -> torch.Tensor:
    SZ = cov_zz(Z_rows)
    SZY = cross_cov(Z_rows, Y_rows)      # (d,q)
    SYZ = SZY.t()                        # (q,d)
    G = W.t() @ SZ @ W                   # (k,k)
    H = torch.linalg.inv(G + mu * torch.eye(G.shape[0], device=G.device))
    return torch.trace(SYZ @ W @ H @ W.t() @ SZY)

# nuisance
# inputs: Z_rows (n,d), S_rows (n,r), W (d,k)

def phi_nui(Z_rows: torch.Tensor, S_rows: torch.Tensor, W: torch.Tensor, nu: float) -> torch.Tensor:
    SZ = cov_zz(Z_rows)
    SZS = cross_cov(Z_rows, S_rows)      # (d,r)
    SSZ = SZS.t()                        # (r,d)
    G = W.t() @ SZ @ W
    H = torch.linalg.inv(G + nu * torch.eye(G.shape[0], device=G.device))
    return torch.trace(SSZ @ W @ H @ W.t() @ SZS)

# stability

def loss_stability(Z_rows: torch.Tensor, W: torch.Tensor, cfg: LossCfg) -> torch.Tensor:
    SZ = cov_zz(Z_rows)
    l_tr = cfg.lambda_tr * (torch.trace(SZ) - cfg.tau) ** 2
    I = torch.eye(W.shape[1], device=W.device)
    l_orth = cfg.lambda_orth * torch.norm(W.t() @ W - I, p='fro') ** 2
    return l_tr + l_orth

# unsupervised
def loss_anchor_explanation(Z_rows: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    # maximize variance captured in subspace (minimize negative)
    SZ = cov_zz(Z_rows)                       # (d,d)
    return -torch.trace(W.t() @ SZ @ W)       # scalar


# total loss wrapper

def compute_total_loss(
    *,
    x: torch.Tensor, # (b,1,t)
    x_hat: torch.Tensor, # (b,1,t)
    tokens: torch.Tensor, # (b,p,d)
    W: torch.Tensor, # (d,k)
    cfg: LossCfg,
    Y_rows: Optional[torch.Tensor] = None,  # (n,q)
    nuisances: Optional[Dict[str, Tuple[torch.Tensor, float]]] = None,  # name -> (S_rows, gamma)
) -> Dict[str, torch.Tensor]:
    Z_rows = tokens.reshape(-1, tokens.shape[-1])

    l_rec = loss_reconstruction(x, x_hat, cfg)
    l_sta = loss_stability(Z_rows, W, cfg)
    l_exp = loss_anchor_explanation(Z_rows, W)

    sup = torch.tensor(0.0, device=Z_rows.device)
    if (Y_rows is not None) and (cfg.beta_sup > 0):
        sup = phi_sup(Z_rows, Y_rows, W, cfg.mu_ridge)

    nui = torch.tensor(0.0, device=Z_rows.device)
    if nuisances:
        for (S_rows, gamma) in nuisances.values():
            if gamma > 0:
                nui = nui + gamma * phi_nui(Z_rows, S_rows, W, cfg.nu_ridge)

    total = l_rec + l_sta + l_exp + nui - cfg.beta_sup * sup
    return {'total': total, 'l_rec': l_rec.detach(),
            'l_sta': l_sta.detach(), 'l_exp': l_exp.detach(),
            'phi_sup': sup.detach(), 'phi_nui': nui.detach()}
