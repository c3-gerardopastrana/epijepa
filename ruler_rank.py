"""Effective rank of each ruler's feature matrix H (not z).

Shows that the low erank(z) of the CNN rulers is a *ruler-rank cap*, not a log-det
failure: erank(z) <= rank(W) <= rank(H). Prints the entropy-based effective rank of H
for each reservoir type on a batch of real (or random) images.

Run on the node:  python ruler_rank.py --data data/imagenette2-160
(falls back to random images if --data is omitted / unavailable).
"""
import argparse, math, sys
from pathlib import Path

import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from epiplexity import (Reservoir, RFFReservoir, MultiRFFReservoir, MultiScaleReservoir,
                        RandomLinearReservoir, RandomConvReservoir)

RES_CH = (16, 32, 64, 64, 128, 128)


def erank(feats):
    ev = _cov_eigs(feats)
    s = ev.sum()
    if s <= 0:
        return 1.0
    p = (ev / s); p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def _cov_eigs(feats):
    x = feats - feats.mean(0, keepdim=True)
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)
    return torch.linalg.eigvalsh(cov.double()).clamp_min(0)


def flatness(feats):
    """How flat is the covariance spectrum? Returns (erank/dim, top/median eig ratio).
    erank/dim -> 1.0 means perfectly isotropic (flat); large top/median means concentrated."""
    ev = _cov_eigs(feats)
    ev = ev[ev > 1e-12]
    if ev.numel() == 0:
        return 0.0, float("inf")
    s = ev / ev.sum(); s = s[s > 0]
    er = float(torch.exp(-(s * s.log()).sum()))
    med = ev.median()
    return er / feats.shape[1], float(ev.max() / med) if med > 0 else float("inf")


class OrthoLinear(torch.nn.Module):
    """Dense random-linear ruler with ORTHONORMALISED rows (Gaussian -> QR). Flat spectrum
    by construction; tests the 'orthonormal maximises effective rank' claim."""
    def __init__(self, m, seed, in_dim=3 * 128 * 128):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        R = torch.randn(in_dim, m, generator=g)        # [D, m]
        Q, _ = torch.linalg.qr(R, mode="reduced")      # [D, m] orthonormal columns
        self.register_buffer("Q", Q.T.contiguous())    # [m, D]
        self.requires_grad_(False)

    def forward(self, x):
        return x.flatten(1) @ self.Q.T


class DenseLinear(torch.nn.Module):
    """Dense i.i.d. Gaussian random-linear ruler (baseline for the ortho comparison)."""
    def __init__(self, m, seed, in_dim=3 * 128 * 128):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("R", torch.randn(m, in_dim, generator=g) / math.sqrt(in_dim))
        self.requires_grad_(False)

    def forward(self, x):
        return x.flatten(1) @ self.R.T


def get_batch(data, n=256, dev="cpu"):
    if data:
        try:
            from torchvision.datasets import ImageFolder
            from torchvision.transforms import v2
            tf = v2.Compose([v2.Resize(128), v2.CenterCrop(128), v2.ToImage(),
                             v2.ToDtype(torch.float32, scale=True),
                             v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            ds = ImageFolder(Path(data) / "val")
            idx = torch.randperm(len(ds))[:n]
            return torch.stack([tf(ds[i][0].convert("RGB")) for i in idx]).to(dev)
        except Exception as e:
            print(f"(data load failed: {e}; using random images)")
    return torch.randn(n, 3, 128, 128, device=dev)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--data", default=""); p.add_argument("--width", type=int, default=512)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = get_batch(a.data, dev=dev)
    W = a.width
    rulers = {
        "random_cnn":          Reservoir(W, 4000, RES_CH),
        "rff":                 RFFReservoir(W, 4000, RES_CH),
        "multiscale(1,2,4,8)": MultiScaleReservoir(W, 4000, grids=(1, 2, 4, 8), channels=RES_CH),
        "random_linear(SRHT)": RandomLinearReservoir(W, 4000, structured=True),
        "random_conv(CS)":     RandomConvReservoir(W, 4000),
        "linear_dense_gauss":  DenseLinear(W, 4000),
        "linear_ortho(QR)":    OrthoLinear(W, 4000),
    }
    print(f"batch={x.shape[0]}  width={W}  device={dev}")
    print(f"{'ruler':22}{'rank(H)':>9}{'erank/dim':>11}{'top/med':>10}   (erank/dim->1 = flat spectrum)")
    for name, r in rulers.items():
        r = r.to(dev).eval()
        with torch.no_grad():
            h = r(x)
        if isinstance(h, (list, tuple)):
            h = torch.cat(h, dim=1)
        fd, tm = flatness(h)
        print(f"{name:22}{erank(h):9.1f}{fd:11.3f}{tm:10.1f}")


if __name__ == "__main__":
    main()
