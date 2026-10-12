"""Imagenette: the LeJEPA minimal example (MINIMAL.md, galilai-group/lejepa @ c293d29) with one switch.

SIGReg, ViTEncoder, augmentations, optimiser, schedule and online probe are the authors' code
verbatim. The only change is which regulariser sits in their loss:

    loss = regulariser * lamb  +  invariance * (1 - lamb)

    sigreg  their SIGReg statistic                  (replication)
    align   0                                       (collapse control)
    epi     -epiplexity / S0                        (ours)

S0 is calibrated once on four initial batches. Backbone runs in bfloat16 as in their recipe;
the epiplexity score is computed in float64 from the projector output.
"""
import argparse, json, sys, time
from pathlib import Path

import timm
import torch, torch.nn as nn, torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.ops import MLP
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from epiplexity import (Reservoir, PretrainedReservoir, ReconReservoir, RFFReservoir,  # noqa: E402
                        MultiReservoir, MultiRFFReservoir, epiplexity)


# ---------------------------------------------------------------- theirs, verbatim
class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class ViTEncoder(nn.Module):
    def __init__(self, proj_dim=128, embed_bn=True):
        super().__init__()
        self.backbone = timm.create_model("vit_small_patch8_224", pretrained=False, num_classes=512,
                                          drop_path_rate=0.1, img_size=128)
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)
        # Non-affine BatchNorm on the projector output: the standard "BN before the loss"
        # head. It pins the embedding scale (no learnable gain/shift) so the epiplexity
        # log-det score cannot be inflated by magnitude, matching the repo's CIFAR/EpiWM
        # recipe. The probe reads `emb` (pre-projector), so probe space stays separate.
        self.embed_bn = nn.BatchNorm1d(proj_dim, affine=False) if embed_bn else nn.Identity()

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        proj = self.embed_bn(self.proj(emb))
        return emb, proj.reshape(N, V, -1).transpose(0, 1)


NORM = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
AUG = v2.Compose([
    v2.RandomResizedCrop(128, scale=(0.08, 1.0)),
    v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
    v2.RandomGrayscale(p=0.2),
    v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
    v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
    v2.RandomHorizontalFlip(),
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True), NORM,
])
TEST = v2.Compose([v2.Resize(128), v2.CenterCrop(128), v2.ToImage(), v2.ToDtype(torch.float32, scale=True), NORM])
# ---------------------------------------------------------------- end theirs


class Views(torch.utils.data.Dataset):
    """Their HFDataset contract on a local Imagenette folder (train/ and val/ class subfolders)."""

    def __init__(self, root, split, V=1):
        self.V, self.ds, self.tf = V, ImageFolder(Path(root) / split), AUG if V > 1 else TEST

    def __getitem__(self, i):
        img, y = self.ds[i]
        return torch.stack([self.tf(img.convert("RGB")) for _ in range(self.V)]), y

    def __len__(self):
        return len(self.ds)


RESERVOIR_CHANNELS = (16, 32, 64, 64, 128, 128)  # 128px input -> 4x4 map before pooling


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True); p.add_argument("--data", required=True)
    p.add_argument("--method", choices=["sigreg", "align", "epi"], default="sigreg")
    p.add_argument("--lamb", type=float, default=0.02, help="theirs: 0.02; ours needs ~0.1")
    p.add_argument("--V", type=int, default=4); p.add_argument("--proj_dim", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--bs", type=int, default=256)
    p.add_argument("--epochs", type=int, default=800); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=64, help="reservoir output width")
    p.add_argument("--reservoir",
                   choices=["random", "pretrained", "recon_online", "random_resample",
                            "rff", "multi_random", "multi_rff", "random_drift"],
                   default="random",
                   help="random/pretrained/recon_online/random_resample as before; "
                        "rff: random Fourier features on a random-CNN descriptor; "
                        "multi_random: K random CNNs (see --K, --multi_agg); "
                        "random_drift: OU slow-drift random CNN (see --drift_alpha)")
    p.add_argument("--resample_every", type=int, default=2,
                   help="for random_resample: epochs between fresh-random reservoir redraws")
    p.add_argument("--recon_lr", type=float, default=1e-3, help="for recon_online: ruler reconstruction LR")
    p.add_argument("--direction", choices=["feat2z", "z2feat"], default="z2feat",
                   help="z2feat (default): z->features (linear-sufficiency reframe); feat2z: original")
    p.add_argument("--multi_agg", choices=["stack", "mean", "max"], default="stack",
                   help="for multi_random: stack (concat->one log-det), mean (avg K scores), "
                        "max (soft worst-case min-max over rulers, see --softmin_tau)")
    p.add_argument("--softmin_tau", type=float, default=0.5, help="temperature for multi_agg=max (soft-min)")
    p.add_argument("--kernel_family", choices=["rbf", "mixed"], default="rbf",
                   help="multi_rff diversity: rbf (K bandwidths of one RBF kernel) or "
                        "mixed (K different kernel families: RBF/Laplacian/Cauchy/Matern)")
    p.add_argument("--K", type=int, default=4, help="number of random CNNs for multi_random")
    p.add_argument("--drift_alpha", type=float, default=0.1, help="OU drift rate for random_drift (per epoch)")
    p.add_argument("--workers", type=int, default=8); p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--embed_bn", action="store_true", default=True,
                   help="non-affine BatchNorm on the projector output before the loss (default on)")
    p.add_argument("--no_embed_bn", dest="embed_bn", action="store_false",
                   help="disable the projector-output BatchNorm (original LeJEPA chassis behaviour)")
    a = p.parse_args()
    # SIGReg's Gaussian target already specifies a scale, so it must see raw projector
    # outputs; the BN head is only meaningful for the epiplexity/align arms.
    if a.method == "sigreg" and a.embed_bn:
        a.embed_bn = False
        print("note: --embed_bn disabled for method=sigreg (it needs raw projector scale)", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ck, res = out / "checkpoint.pt", out / "result.json"
    if res.exists():
        print("done already"); return
    torch.manual_seed(a.seed)
    train_ds, test_ds = Views(a.data, "train", V=a.V), Views(a.data, "val", V=1)
    train = DataLoader(train_ds, batch_size=a.bs, shuffle=True, drop_last=True, num_workers=a.workers, pin_memory=True, persistent_workers=True)
    test = DataLoader(test_ds, batch_size=256, num_workers=4, pin_memory=True, persistent_workers=True)
    net = ViTEncoder(proj_dim=a.proj_dim, embed_bn=a.embed_bn).cuda()
    if a.grad_ckpt: net.backbone.set_grad_checkpointing(True)
    probe = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 10)).cuda()
    sigreg = SIGReg().cuda()
    recon_opt = None
    if a.method == "epi":
        if a.reservoir == "pretrained":
            reservoir = PretrainedReservoir(a.width, 4000 + a.seed).cuda().eval()
        elif a.reservoir == "recon_online":
            reservoir = ReconReservoir(a.width, 4000 + a.seed, RESERVOIR_CHANNELS).cuda()
            recon_opt = torch.optim.AdamW(reservoir.parameters(), lr=a.recon_lr, weight_decay=1e-4)
        elif a.reservoir == "rff":
            reservoir = RFFReservoir(a.width, 4000 + a.seed, RESERVOIR_CHANNELS).cuda().eval()
        elif a.reservoir == "multi_random":
            reservoir = MultiReservoir(a.width, 4000 + a.seed, K=a.K, channels=RESERVOIR_CHANNELS).cuda().eval()
        elif a.reservoir == "multi_rff":
            smults = (1.0,) * a.K if a.kernel_family == "mixed" else (0.5, 1.0, 2.0, 4.0)[:a.K]
            reservoir = MultiRFFReservoir(a.width, 4000 + a.seed, K=a.K, channels=RESERVOIR_CHANNELS,
                                          sigma_mults=smults, kernel_family=a.kernel_family).cuda().eval()
        else:  # random, random_resample, random_drift
            reservoir = Reservoir(a.width, 4000 + a.seed, RESERVOIR_CHANNELS).cuda().eval()
    else:
        reservoir = None
    opt = torch.optim.AdamW([{"params": net.parameters(), "lr": a.lr, "weight_decay": 5e-2},
                             {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}])
    warmup, total = len(train), len(train) * a.epochs
    sched = SequentialLR(opt, [LinearLR(opt, start_factor=0.01, total_iters=warmup),
                               CosineAnnealingLR(opt, T_max=total - warmup, eta_min=1e-3)], milestones=[warmup])
    scaler = GradScaler()

    def views_score(proj, vs):
        return torch.stack([epiplexity(proj[v].float(), vs[:, v].float(), reservoir,
                                       direction=a.direction, multi_agg=a.multi_agg,
                                       softmin_tau=a.softmin_tau)
                            for v in range(a.V)]).mean()

    def calibrate_s0():
        with torch.no_grad():
            vals = []
            for i, (vs, _) in enumerate(train):
                vs = vs.cuda()
                with autocast("cuda", dtype=torch.bfloat16): _, pr = net(vs)
                vals.append(float(views_score(pr, vs)))
                if i >= 3: break
            return max(sum(vals) / len(vals), 1e-8)

    s0, start, hist = 1., 0, []
    if a.method == "epi":  # one-off scale calibration, same role as their loss normalisation
        s0 = calibrate_s0()
        print(f"S0={s0:.2f}", flush=True)
    if ck.exists():
        st = torch.load(ck, map_location="cuda", weights_only=False)
        net.load_state_dict(st["net"]); probe.load_state_dict(st["probe"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        start, hist, s0 = st["epoch"], st["hist"], st["s0"]
        if "reservoir" in st and reservoir is not None:
            reservoir.load_state_dict(st["reservoir"])
        if "recon_opt" in st and recon_opt is not None:
            recon_opt.load_state_dict(st["recon_opt"])
        print(f"resumed at epoch {start}", flush=True)

    for epoch in range(start, a.epochs):
        # random_resample: redraw the frozen ruler every --resample_every epochs, then
        # recalibrate S0 so the normalised score keeps the same meaning.
        if a.reservoir == "random_resample" and epoch > 0 and epoch % a.resample_every == 0:
            reservoir.reseed(4000 + a.seed + epoch)
            s0 = calibrate_s0()
            print(f"resampled reservoir @ epoch {epoch}, S0={s0:.2f}", flush=True)
        if a.reservoir == "random_drift" and epoch > 0:
            reservoir.drift(4000 + a.seed + epoch, a.drift_alpha)
            s0 = calibrate_s0()
        net.train(); probe.train(); t0 = time.perf_counter(); agg = {}
        for vs, y in train:
            vs, y = vs.cuda(non_blocking=True), y.cuda(non_blocking=True)
            # recon_online: train the ruler to reconstruct frames (its own optimiser);
            # stop-grad from the score keeps it encoder-independent.
            if recon_opt is not None:
                reservoir.train()
                rl = reservoir.recon_loss(vs[:, 0].float())
                recon_opt.zero_grad(); rl.backward(); recon_opt.step()
                agg["recon"] = agg.get("recon", 0.) + float(rl.detach())
            with autocast("cuda", dtype=torch.bfloat16):
                emb, proj = net(vs)
                inv_loss = (proj.mean(0) - proj).square().mean()
                if a.method == "sigreg": reg = sigreg(proj)
                elif a.method == "epi": reg = -views_score(proj, vs) / s0
                else: reg = inv_loss.new_zeros(())
                core = reg * a.lamb + inv_loss * (1 - a.lamb)
                probe_loss = F.cross_entropy(probe(emb.detach()), y.repeat_interleave(a.V))
                loss = core + probe_loss
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            for k, v in [("core", core), ("inv", inv_loss), ("reg", reg), ("probe", probe_loss)]:
                agg[k] = agg.get(k, 0.) + float(v.detach())
        log = {k: v / len(train) for k, v in agg.items()}
        net.eval(); probe.eval(); correct = 0
        with torch.inference_mode():
            for vs, y in test:
                with autocast("cuda", dtype=torch.bfloat16):
                    correct += (probe(net(vs.cuda(non_blocking=True))[0]).argmax(1) == y.cuda()).sum().item()
        log.update(epoch=epoch + 1, test_acc=correct / len(test_ds), seconds=time.perf_counter() - t0)
        hist.append(log)
        ckpt = dict(net=net.state_dict(), probe=probe.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                    scaler=scaler.state_dict(), epoch=epoch + 1, hist=hist, s0=s0)
        if a.reservoir in ("recon_online", "random_resample", "random_drift"):
            ckpt["reservoir"] = reservoir.state_dict()
        if recon_opt is not None:
            ckpt["recon_opt"] = recon_opt.state_dict()
        torch.save(ckpt, ck)
        (out / "history.json").write_text(json.dumps(hist))
        print(f"{a.method} lamb={a.lamb:g} seed={a.seed}: {epoch + 1}/{a.epochs} acc={log['test_acc'] * 100:.2f}% "
              f"inv={log['inv']:.4f} reg={log['reg']:.4f} {log['seconds']:.0f}s", flush=True)
    best = max(h["test_acc"] for h in hist)
    res.write_text(json.dumps(dict(method=a.method, lamb=a.lamb, seed=a.seed, reservoir=a.reservoir,
                                   embed_bn=a.embed_bn, direction=a.direction, multi_agg=a.multi_agg,
                                   softmin_tau=a.softmin_tau, kernel_family=a.kernel_family,
                                   K=a.K, final_acc=hist[-1]["test_acc"], best_acc=best,
                                   epochs=a.epochs, history=hist), indent=1))
    print(f"FINAL {a.method} lamb={a.lamb:g} seed={a.seed}: final {hist[-1]['test_acc'] * 100:.2f}%  best {best * 100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
