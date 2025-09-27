from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn

from src.ae.loss import compute_total_loss

# trainer
def ae_train_epoch(
    model: nn.Module,
    projector: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    loss_cfg,
    device: torch.device
) -> Dict[str, float]:
    # metrics
    agg = {'total': 0.0, 'l_rec': 0.0, 'l_sta': 0.0, 'phi_sup': 0.0, 'phi_nui': 0.0}
    n = 0

    model.train()
    projector.train()

    for batch in dataloader:
        # required
        x = batch['x'].to(device) # (b,1,t)

        # optional supervised
        Y_rows = batch.get('Y_rows')
        if Y_rows is not None:
            Y_rows = Y_rows.to(device) # (n,q), aligned with tokens

        # optional nuisances: {'name': (S_rows, gamma)}
        nuis = batch.get('nuisances')
        if nuis:
            nuis = {k: (S.to(device), float(gamma)) for k, (S, gamma) in nuis.items()}
        else:
            nuis = None

        optimizer.zero_grad(set_to_none=True)

        # forward
        x_hat, tokens = model(x)  # x_hat: (b,1,t), tokens: (b,p,d)
        W = projector() # (d,k)

        # loss
        outs = compute_total_loss(
            x=x, x_hat=x_hat, tokens=tokens, W=W,
            cfg=loss_cfg, Y_rows=Y_rows, nuisances=nuis
        )

        # backward
        outs['total'].backward()
        optimizer.step()

        # agg
        agg['total'] += float(outs['total'])
        agg['l_rec'] += float(outs['l_rec'])
        agg['l_sta'] += float(outs['l_sta'])
        agg['phi_sup'] += float(outs['phi_sup'])
        agg['phi_nui'] += float(outs['phi_nui'])
        n += 1

    for k in agg:
        agg[k] = agg[k] / max(n, 1)
    return agg


@torch.no_grad()
def ae_eval_epoch(
    model: nn.Module,
    projector: nn.Module,
    dataloader,
    loss_cfg,
    device: torch.device
) -> Dict[str, float]:
    agg = {'total': 0.0, 'l_rec': 0.0, 'l_sta': 0.0, 'phi_sup': 0.0, 'phi_nui': 0.0}
    n = 0

    model.eval()
    projector.eval()

    for batch in dataloader:
        x = batch['x'].to(device)

        Y_rows = batch.get('Y_rows')
        if Y_rows is not None:
            Y_rows = Y_rows.to(device)

        nuis = batch.get('nuisances')
        if nuis:
            nuis = {k: (S.to(device), float(gamma)) for k, (S, gamma) in nuis.items()}
        else:
            nuis = None

        x_hat, tokens = model(x)
        W = projector()

        outs = compute_total_loss(
            x=x, x_hat=x_hat, tokens=tokens, W=W,
            cfg=loss_cfg, Y_rows=Y_rows, nuisances=nuis
        )

        agg['total'] += float(outs['total'])
        agg['l_rec'] += float(outs['l_rec'])
        agg['l_sta'] += float(outs['l_sta'])
        agg['phi_sup'] += float(outs['phi_sup'])
        agg['phi_nui'] += float(outs['phi_nui'])
        n += 1

    for k in agg:
        agg[k] = agg[k] / max(n, 1)
    return agg
