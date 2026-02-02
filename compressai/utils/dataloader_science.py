import math
from typing import Tuple, Optional, Literal

import numpy as np
import torch
import torch.nn.functional as F

NormType = Literal["none", "minmax", "meanstd"]
FusionType = Literal["weighted", "overwrite"]

def normalize_tensor(
    x_f: torch.Tensor,
    norm_type: NormType = "minmax",
    *,
    vmin: float = 0.0,
    vmax: float = 1.0,
    vmean: float = 0.0,
    vstd: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    if norm_type == "none":
        print("Suggestion: consider applying normalization for better model performance.")
        print("You can choose 'minmax' or 'meanstd' normalization.")
        return x_f
    if norm_type == "minmax":
        denom = (vmax - vmin).clamp_min(eps)[:, None]   # (N,1)
        x_norm = (x_f - vmin[:, None]) / denom          # (N,HW)
        return x_norm
    if norm_type == "meanstd":
        denom = vstd.clamp_min(eps)[:, None]
        return (x_f - vmean[:, None]) / denom
    raise ValueError(f"Unknown norm_type: {norm_type}")

def denorm_perslice_minmax(
    x_norm: torch.Tensor, 
    norm_type: NormType,
    vmin: torch.Tensor, 
    vmax: torch.Tensor,
    vmean: torch.Tensor,
    vstd: torch.Tensor,
    ):
    # x_norm: (N,H,W)
    n = x_norm.shape[0]
    x_f = x_norm.view(n, -1)
    if norm_type == "none":
        return x_norm
    if norm_type == "minmax":
        return (x_f * (vmax - vmin)[:,None] + vmin[:,None]).view_as(x_norm)
    if norm_type == "meanstd":
        return (x_f * vstd[:,None] + vmean[:,None]).view_as(x_norm)
    raise ValueError(f"Unknown norm_type: {norm_type}")


def make_weight_window(bh: int, bw: int, device, dtype):
    # Hann window 
    wh = torch.hann_window(bh, periodic=False, device=device, dtype=dtype)
    ww = torch.hann_window(bw, periodic=False, device=device, dtype=dtype)
    w2d = wh[:, None] * ww[None, :]          # (bh,bw)
    w2d = w2d.clamp_min(1e-6)                # 避免全0
    return w2d

def load_bin_nhw_perslice_norm(
    bin_path: str,
    input_shape: Tuple[int,int,int],
    *,
    dtype: np.dtype = np.float32,
    device=None,
    eps: float = 1e-12,
    norm_type: NormType = "none",
):
    n,h,w = input_shape
    arr = np.fromfile(bin_path, dtype=dtype)[: n*h*w].reshape(n,h,w)
    x_ori = torch.from_numpy(arr)
    

    x_f = x_ori.float().view(n, -1)
    vmin = x_f.min(dim=1).values  # (N,)
    vmax = x_f.max(dim=1).values  # (N,)
    vmean = x_f.mean(dim=1)                # (N,)
    vstd  = x_f.std(dim=1, unbiased=False) # (N,)

    # (N,HW) normalize then reshape back
    x_norm = normalize_tensor(
        x_f,
        norm_type,
        vmin=vmin,
        vmax=vmax,
        vmean=vmean,
        vstd=vstd
    )
    x_norm = x_norm.view(n,h,w)

    if device is not None:
        x_norm = x_norm.to(device)
        x_ori = x_ori.to(device)
        vmin = vmin.to(device)
        vmax = vmax.to(device)
        vmean = vmean.to(device)
        vstd = vstd.to(device)

    return x_norm, x_ori, {"vmin": vmin, "vmax": vmax, "vmean": vmean, "vstd": vstd}


def extract_overlapped_blocks(
    x_g3hw: torch.Tensor,                 # (G,3,H,W)
    block_hw: Tuple[int, int],            # (bh,bw) model input size
    stride_hw: Tuple[int, int],           # (sh,sw) < (bh,bw) for overlap
    *,
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    Extract overlapped REAL blocks of size (3,bh,bw) using sliding window.

    We do ONE global pad on H/W so every window is valid, then extract.
    Return:
      blocks: (bn,3,bh,bw)
      meta: (H0, W0, Hp, Wp) where:
        - H0,W0 = original H,W
        - Hp,Wp = padded H,W used for window extraction
    """
    if x_g3hw.ndim != 4 or x_g3hw.shape[1] != 3:
        raise ValueError(f"Expected (G,3,H,W), got {tuple(x_g3hw.shape)}")

    bh, bw = block_hw
    sh, sw = stride_hw
    if sh <= 0 or sw <= 0:
        raise ValueError("stride must be positive")
    if sh >= bh or sw >= bw:
        raise ValueError("For overlap, require stride < block size (bh,bw)")

    G, _, H, W = x_g3hw.shape

    # We want start positions: 0, sh, 2sh, ... <= H-1 and ensure window covers end
    # Pad so that last window starting at last_h starts within range and ends at <= Hp
    # Compute how many steps
    n_h = math.ceil((H - bh) / sh) + 1 if H > bh else 1
    n_w = math.ceil((W - bw) / sw) + 1 if W > bw else 1
    Hp = (n_h - 1) * sh + bh
    Wp = (n_w - 1) * sw + bw

    pad_bottom = max(0, Hp - H)
    pad_right = max(0, Wp - W)

    if pad_bottom > 0 or pad_right > 0:
        # pad (left,right,top,bottom) on last two dims
        x_pad = F.pad(x_g3hw, (0, pad_right, 0, pad_bottom), mode="constant", value=pad_value)
    else:
        x_pad = x_g3hw

    # Extract windows
    blocks = []
    for g in range(G):
        for ih in range(n_h):
            h0 = ih * sh
            h1 = h0 + bh
            for iw in range(n_w):
                w0 = iw * sw
                w1 = w0 + bw
                blocks.append(x_pad[g, :, h0:h1, w0:w1])  # (3,bh,bw)

    return torch.stack(blocks, dim=0), (H, W, Hp, Wp)

# def blocks_to_nhw_overlap_weighted(
#     blocks_bn3bhbw: torch.Tensor,      # (bn,3,bh,bw) 模型输出块
#     input_shape: Tuple[int, int, int], # (N,H,W)
#     block_size: Tuple[int, int, int],  # (3,bh,bw)
#     stride_hw: Tuple[int, int],        # (sh,sw)
#     meta_hw: Tuple[int, int, int, int],# (H0,W0,Hp,Wp)
#     *,
#     pad_value: float = 0.0,
#     norm_type: NormType = "minmax",
#     vmin: torch.Tensor,
#     vmax: torch.Tensor,
#     vmean: torch.Tensor,
#     vstd: torch.Tensor,
# ) -> torch.Tensor:
#     N, H0, W0 = input_shape
#     c, bh, bw = block_size
#     sh, sw = stride_hw
#     H, W, Hp, Wp = meta_hw
#     assert (H, W) == (H0, W0)

#     # 计算 extraction 时的网格数量（必须和 extract_overlapped_blocks 一致）
#     n_h = math.ceil((H - bh) / sh) + 1 if H > bh else 1
#     n_w = math.ceil((W - bw) / sw) + 1 if W > bw else 1

#     G = math.ceil(N / 3)
#     bn_expected = G * n_h * n_w
#     if blocks_bn3bhbw.shape[0] != bn_expected:
#         raise ValueError(f"bn mismatch: expected {bn_expected}, got {blocks_bn3bhbw.shape[0]}")

#     device = blocks_bn3bhbw.device
#     dtype = blocks_bn3bhbw.dtype

#     # 加权窗（bh,bw）
#     w2d = make_weight_window(bh, bw, device=device, dtype=dtype)  # (bh,bw)
#     w2d = w2d[None, None, :, :]  # (1,1,bh,bw) 便于broadcast

#     # sum/weight 画布：用 Hp,Wp（和你 extraction pad 后一致）
#     sum_canvas = torch.zeros((G, 3, Hp, Wp), device=device, dtype=dtype)
#     w_canvas   = torch.zeros((G, 1, Hp, Wp), device=device, dtype=dtype)

#     idx = 0
#     for g in range(G):
#         for ih in range(n_h):
#             h0 = ih * sh
#             h1 = h0 + bh
#             for iw in range(n_w):
#                 w0 = iw * sw
#                 w1 = w0 + bw

#                 block = blocks_bn3bhbw[idx]  # (3,bh,bw)
#                 block = block[None, :, :, :] # (1,3,bh,bw)

#                 sum_canvas[g:g+1, :, h0:h1, w0:w1] += block * w2d
#                 w_canvas[g:g+1, :, h0:h1, w0:w1]   += w2d
#                 idx += 1

#     # 归一化除权重（避免除0）
#     out = sum_canvas / w_canvas.clamp_min(1e-6)

#     # 裁回原图大小并 reshape 回 (N,H,W)
#     nhw = out[:, :, :H, :W].contiguous().view(G * 3, H, W)[:N]
    
#     # nhw = nhw.clamp(0, 1)

#     # denorm
#     nhw = denorm_perslice_minmax(
#         nhw, norm_type,
#         vmin=vmin, vmax=vmax,
#         vmean=vmean, vstd=vstd,
#     )
#     return nhw

def blocks_to_nhw_overlap_weighted(
    blocks_bn3bhbw: torch.Tensor,
    input_shape: Tuple[int, int, int],
    block_size: Tuple[int, int, int],
    stride_hw: Tuple[int, int],
    meta_hw: Tuple[int, int, int, int],
    *,
    pad_value: float = 0.0,
    norm_type: NormType = "minmax",
    vmin: torch.Tensor,
    vmax: torch.Tensor,
    vmean: torch.Tensor,
    vstd: torch.Tensor,
    fusion: FusionType = "weighted",   # <<<<<< 开关
) -> torch.Tensor:
    N, H0, W0 = input_shape
    c, bh, bw = block_size
    sh, sw = stride_hw
    H, W, Hp, Wp = meta_hw
    assert (H, W) == (H0, W0)

    n_h = math.ceil((H - bh) / sh) + 1 if H > bh else 1
    n_w = math.ceil((W - bw) / sw) + 1 if W > bw else 1

    G = math.ceil(N / 3)
    bn_expected = G * n_h * n_w
    if blocks_bn3bhbw.shape[0] != bn_expected:
        raise ValueError(f"bn mismatch: expected {bn_expected}, got {blocks_bn3bhbw.shape[0]}")

    device = blocks_bn3bhbw.device
    dtype = blocks_bn3bhbw.dtype

    if fusion == "weighted":
        w2d = make_weight_window(bh, bw, device=device, dtype=dtype)  # (bh,bw)
        w2d = w2d[None, None, :, :]  # (1,1,bh,bw)

        sum_canvas = torch.zeros((G, 3, Hp, Wp), device=device, dtype=dtype)
        w_canvas   = torch.zeros((G, 1, Hp, Wp), device=device, dtype=dtype)

        idx = 0
        for g in range(G):
            for ih in range(n_h):
                h0 = ih * sh
                h1 = h0 + bh
                for iw in range(n_w):
                    w0 = iw * sw
                    w1 = w0 + bw

                    block = blocks_bn3bhbw[idx][None, :, :, :]  # (1,3,bh,bw)
                    sum_canvas[g:g+1, :, h0:h1, w0:w1] += block * w2d
                    w_canvas[g:g+1, :, h0:h1, w0:w1]   += w2d
                    idx += 1

        out = sum_canvas / w_canvas.clamp_min(1e-6)

    elif fusion == "overwrite":
        # 直接覆盖：后写覆盖前写
        out = torch.full((G, 3, Hp, Wp), pad_value, device=device, dtype=dtype)

        idx = 0
        for g in range(G):
            for ih in range(n_h):
                h0 = ih * sh
                h1 = h0 + bh
                for iw in range(n_w):
                    w0 = iw * sw
                    w1 = w0 + bw

                    out[g, :, h0:h1, w0:w1] = blocks_bn3bhbw[idx]  # (3,bh,bw)
                    idx += 1
    else:
        raise ValueError(f"Unknown fusion: {fusion}")

    # 裁回原图大小并 reshape 回 (N,H,W)
    nhw = out[:, :, :H, :W].contiguous().view(G * 3, H, W)[:N]

    # denorm
    nhw = denorm_perslice_minmax(
        nhw, norm_type,
        vmin=vmin, vmax=vmax,
        vmean=vmean, vstd=vstd,
    )
    return nhw

def nhw_to_g3hw(x_nhw: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
    """
    Convert (N,H,W) -> (G,3,H,W) by grouping N in consecutive 3.
    If N not divisible by 3, pad along N (post-pad) so every group has 3 slices.
    """
    if x_nhw.ndim != 3:
        raise ValueError(f"Expected (N,H,W), got {tuple(x_nhw.shape)}")
    n, h, w = x_nhw.shape
    g = math.ceil(n / 3)
    n2 = g * 3

    if n2 != n:
        out = torch.full((n2, h, w), pad_value, dtype=x_nhw.dtype, device=x_nhw.device)
        out[:n] = x_nhw
        x_nhw = out

    return x_nhw.view(g, 3, h, w)  # (G,3,H,W)


def center_pad_2d(
    x: torch.Tensor,  # (..., H, W)
    target_hw: Tuple[int, int],
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Center (balanced) pad last two dims to target (bh,bw).
    """
    bh, bw = target_hw
    H, W = x.shape[-2], x.shape[-1]
    if H > bh or W > bw:
        raise ValueError(f"Cannot pad: current {(H,W)} larger than target {(bh,bw)}")

    pad_h = bh - H
    pad_w = bw - W
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return F.pad(x, (left, right, top, bottom), mode="constant", value=pad_value)


def cut_then_pad_blocks(
    x_g3hw: torch.Tensor,                 # (G,3,H,W)
    block_hw: Tuple[int, int],            # (bh,bw) final
    p: float,
    *,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Step 2 (as you requested): FIRST cut blocks, THEN pad each block individually.

    - real block size: (3, rh, rw) where rh=ceil(bh*p), rw=ceil(bw*p)
    - cut along H/W into tiles of size rh/rw.
      IMPORTANT: we do NOT pad the whole image to rh/rw multiples first.
      Instead, each edge tile may be smaller (<rh or <rw).
    - each cut tile is then center-padded to (3,bh,bw).

    Output:
      blocks: (bn,3,bh,bw) where bn = G * tiles_h * tiles_w
    """
    if not (0 < p <= 1.0):
        raise ValueError("p must be in (0, 1]")
    if x_g3hw.ndim != 4 or x_g3hw.shape[1] != 3:
        raise ValueError(f"Expected (G,3,H,W), got {tuple(x_g3hw.shape)}")

    bh, bw = block_hw
    G, C, H, W = x_g3hw.shape

    rh = max(1, int(math.ceil(bh * p)))
    rw = max(1, int(math.ceil(bw * p)))
    if rh > bh or rw > bw:
        raise ValueError(f"Invalid p: real (rh,rw)=({rh},{rw}) exceeds (bh,bw)=({bh},{bw})")

    tiles_h = math.ceil(H / rh)
    tiles_w = math.ceil(W / rw)

    blocks = []
    for g in range(G):
        for th in range(tiles_h):
            h0 = th * rh
            h1 = min(h0 + rh, H)  # may be smaller on bottom edge
            for tw in range(tiles_w):
                w0 = tw * rw
                w1 = min(w0 + rw, W)  # may be smaller on right edge

                tile = x_g3hw[g, :, h0:h1, w0:w1]         # (3, h', w')  h'<=rh, w'<=rw
                tile = center_pad_2d(tile, (bh, bw), pad_value=pad_value)  # (3,bh,bw)
                blocks.append(tile)

    return torch.stack(blocks, dim=0)  # (bn,3,bh,bw)

def blockify_bin_overlap(
    bin_path: str,
    input_shape: Tuple[int, int, int],
    block_size: Tuple[int, int, int],      # (3,bh,bw)
    p: float,
    stride_hw: Tuple[int, int],            # (sh,sw)
    *,
    dtype: np.dtype = np.uint8,
    pad_value: float = 0.0,
    norm_type: NormType = "minmax",
    device: Optional[torch.device] = None,
):
    c, bh, bw = block_size
    if c != 3:
        raise ValueError("block_size must be (3,bh,bw)")

    x_nhw_norm, x_nhw_ori, status = load_bin_nhw_perslice_norm(
        bin_path,
        input_shape,
        dtype=dtype,
        device=device,
        norm_type=norm_type,
    )
    x_g3hw = nhw_to_g3hw(x_nhw_norm, pad_value=pad_value)

    blocks, meta_hw = extract_overlapped_blocks(
        x_g3hw,
        block_hw=(bh, bw),
        stride_hw=stride_hw,
        pad_value=pad_value,
    )
    return blocks, x_nhw_ori, status, meta_hw

def center_crop_2d(x: torch.Tensor, crop_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Take centered crop from last two dims.
    x: (..., H, W)
    returns: (..., ch, cw)
    """
    ch, cw = crop_hw
    H, W = x.shape[-2], x.shape[-1]
    if ch > H or cw > W:
        raise ValueError(f"Cannot crop: crop {(ch, cw)} larger than input {(H, W)}")

    top = (H - ch) // 2
    left = (W - cw) // 2
    return x[..., top:top + ch, left:left + cw]

