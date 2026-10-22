"""Is H actually whitened? Measure the ruler feature spectrum BEFORE vs AFTER _whiten,
in the real training setting (real Imagenette batch, same standardisation the score uses).

For each ruler it reports, on H:
  rank         = entropy effective rank
  rank/dim     = 1.0 means isotropic (fully whitened)
  cond(top/med)= top eigenvalue / median (1.0 = flat; large = still concentrated/collapsed)
for H_standardized (what the score sees before whiten) and H_whitened (_whiten output).
If whitened rank/dim ~ 1 and top/med ~ 1 -> whitening works. If whitened is still
concentrated (or collapses) -> whitening is NOT doing what we want.

Usage:  python check_whiten.py --data data/imagenette2-160 --width 512 --bs 256
Add --bf16 to replicate training's autocast dtype for H.
"""
import argparse, math, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from epiplexity import Reservoir, RFFReservoir, RandomLinearReservoir, _whiten

RES_CH = (16, 32, 64, 64, 128, 128)


def spectrum(h):
    h = h.double()
    x = h - h.mean(0)
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)
    ev = torch.linalg.eigvalsh(cov).clamp_min(0)
    ev = ev[ev > 1e-12]
    if ev.numel() == 0:
        return 0.0, 0.0, float("inf")
    p = ev / ev.sum(); p = p[p > 0]
    er = float(torch.exp(-(p * p.log()).sum()))
    med = ev.median()
    return er, er / h.shape[1], float(ev.max() / med) if med > 0 else float("inf")


def standardize(h):
    return (h - h.mean(0)) / h.std(0, unbiased=False).clamp_min(1e-6) / math.sqrt(h.shape[1])


def get_batch(data, n, dev, bf16):
    from torchvision.datasets import ImageFolder
    from torchvision.transforms import v2
    tf = v2.Compose([v2.Resize(128), v2.CenterCrop(128), v2.ToImage(),
                     v2.ToDtype(torch.float32, scale=True),
                     v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    ds = ImageFolder(Path(data) / "val")
    idx = torch.randperm(len(ds))[:n]
    x = torch.stack([tf(ds[i][0].convert("RGB")) for i in idx]).to(dev)
    return x.to(torch.bfloat16) if bf16 else x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--width", type=int, default=512)
    p.add_argument("--bs", type=int, default=256); p.add_argument("--bf16", action="store_true")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = get_batch(a.data, a.bs, dev, a.bf16)
    W = a.width
    rulers = {
        "random_cnn":    Reservoir(W, 4000, RES_CH),
        "rff":           RFFReservoir(W, 4000, RES_CH),
        "random_linear": RandomLinearReservoir(W, 4000, structured=True),
    }
    print(f"batch={a.bs} width={W} bf16={a.bf16} dev={dev}")
    hdr = f"{'ruler':14}{'stage':12}{'rank':>8}{'rank/dim':>10}{'top/med':>10}"
    print(hdr)
    for name, r in rulers.items():
        r = r.to(dev).eval()
        with torch.no_grad():
            h = r(x).float()
        hs = standardize(h)
        hw = _whiten(hs)
        for stage, hh in [("standardized", hs), ("whitened", hw)]:
            er, rd, tm = spectrum(hh)
            print(f"{name:14}{stage:12}{er:8.1f}{rd:10.3f}{tm:10.1f}")
    print("\nwhitening works iff 'whitened' rows show rank/dim ~1 and top/med ~1")


if __name__ == "__main__":
    main()
