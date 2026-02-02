# compressai/runtime/utils/metrics.py
from __future__ import annotations
import torch

# @torch.no_grad()
# def basic_metrics(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-12):
#     x_f = x.float()
#     y_f = x_hat.float()

#     diff = y_f - x_f
#     mse = torch.mean(diff * diff)
#     rmse = torch.sqrt(mse)

#     maxe = torch.max(torch.abs(diff))

#     range_ = torch.max(x_f) - torch.min(x_f)
#     nrmse = rmse / range_

#     psnr = 20.0 * torch.log10(range_ / (2.0 * rmse + eps))
#     psnr = 20 * torch.log10(range_) - 10 * torch.log10(mse)
#     return {
#         "rmse": float(rmse.item()),
#         "nrmse": float(nrmse.item()),
#         "maxe": float(maxe.item()),
#         "psnr": float(psnr.item()),
#     }
@torch.no_grad()
def basic_metrics(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    eps: float = 1e-12,
):
    if x.shape != x_hat.shape:
        raise ValueError(f"Shape mismatch: x {tuple(x.shape)} vs x_hat {tuple(x_hat.shape)}")

    if x.ndim == 3:
        # (N,H,W) -> (N, H*W)
        N = x.shape[0]
        x_f = x.float().view(N, -1)
        y_f = x_hat.float().view(N, -1)
    elif x.ndim == 4:
        # (N,C,H,W) -> (N, C*H*W)
        N = x.shape[0]
        x_f = x.float().view(N, -1)
        y_f = x_hat.float().view(N, -1)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {x.ndim}D")
    n = x.size(0)
    diff = y_f - x_f
    mse_n = torch.mean(diff * diff, dim=1)                  # (N,)
    rmse_n = torch.sqrt(mse_n)                        # (N,)
    range_n = (torch.max(x_f, dim=1).values - torch.min(x_f, dim=1).values) # (N,)

    nrmse_n = rmse_n / range_n  
    if n == 1:
        # (N,)
        psnr = -20.0 * torch.log10(nrmse_n[0])  # (N,)
    else:
        psnr = -20.0 * torch.log10(torch.sqrt((nrmse_n * nrmse_n).mean()))                     # scalar
    # Global max error (over all elements)
    maxe = torch.max(torch.abs(diff))

    # 也给你 macro mean / median 都返回，方便看异常 slice
    return {
        "rmse": float(torch.mean(rmse_n).item()),
        "nrmse": float(torch.mean(nrmse_n).item()),
        "maxe": float(maxe.item()),
        "psnr": float(psnr.item()),
    }
