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
                        RandomLinearReservoir)

RES_CH = (16, 32, 64, 64, 128, 128)


def erank(feats):
    x = feats - feats.mean(0, keepdim=True)
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)
    ev = torch.linalg.eigvalsh(cov.double()).clamp_min(0)
    s = ev.sum()
    if s <= 0:
        return 1.0
    p = (ev / s); p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


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
        "random_cnn":        Reservoir(W, 4000, RES_CH),
        "rff":               RFFReservoir(W, 4000, RES_CH),
        "multiscale(1,2,4,8)": MultiScaleReservoir(W, 4000, grids=(1, 2, 4, 8), channels=RES_CH),
        "random_linear":     RandomLinearReservoir(W, 4000, structured=True),
    }
    print(f"batch={x.shape[0]}  width={W}  device={dev}")
    print(f"{'ruler':22}{'rank(H)':>10}   (max possible = width or batch)")
    for name, r in rulers.items():
        r = r.to(dev).eval()
        with torch.no_grad():
            h = r(x)
        if isinstance(h, (list, tuple)):  # multiscale returns a list; report concat + per-head
            cat = torch.cat(h, dim=1)
            print(f"{name:22}{erank(cat):10.1f}   (concat of {len(h)} heads; per-head "
                  f"{[round(erank(hi),1) for hi in h]})")
        else:
            print(f"{name:22}{erank(h):10.1f}")


if __name__ == "__main__":
    main()
