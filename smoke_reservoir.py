"""CPU/MPS smoke test for the reservoir swap (PoC #1).

Does NOT train anything. It just exercises the exact loss path used in training:

    images --(frozen reservoir)--> H --(ridge_map)--> a --(score vs embedding z)--> S

for both the random and the pretrained reservoir, and checks that:
  * the reservoir output contract is [batch, width];
  * the score is finite and differentiable w.r.t. the embedding;
  * a collapsed embedding (constant) scores ~0, an informative one scores > 0.

Run:  python smoke_reservoir.py
"""
import torch

from epiplexity import (Reservoir, PretrainedReservoir, ReconReservoir, RFFReservoir,
                        MultiReservoir, epiplexity)

DEVICE = "cpu"  # ridge/score run in float64; MPS lacks float64, real training uses CUDA
WIDTH, PROJ_DIM, BATCH = 64, 16, 32


def check(name, reservoir):
    reservoir = reservoir.to(DEVICE).eval()
    x = torch.randn(BATCH, 3, 128, 128, device=DEVICE)

    with torch.no_grad():
        h = reservoir(x)
    assert h.shape == (BATCH, WIDTH), f"{name}: reservoir output {tuple(h.shape)} != {(BATCH, WIDTH)}"

    # informative embedding: requires grad, should give a positive, finite score
    z = torch.randn(BATCH, PROJ_DIM, device=DEVICE, requires_grad=True)
    s = epiplexity(z, x, reservoir)
    s.backward()
    assert torch.isfinite(s), f"{name}: non-finite score {s}"
    assert z.grad is not None and torch.isfinite(z.grad).all(), f"{name}: bad grad"
    informative = float(s)

    # collapsed embedding: constant across batch -> Zc = 0 -> score ~ 0
    zc = torch.ones(BATCH, PROJ_DIM, device=DEVICE)
    collapsed = float(epiplexity(zc, x, reservoir))

    print(f"{name:>22}: out={tuple(h.shape)}  informative_score={informative:8.4f}  "
          f"collapsed_score={collapsed:8.4f}  grad_ok=True")
    assert abs(collapsed) < 1e-3, f"{name}: collapsed score should be ~0, got {collapsed}"
    assert informative > collapsed, f"{name}: informative should beat collapsed"


if __name__ == "__main__":
    print(f"device={DEVICE}")
    torch.manual_seed(0)
    check("random Reservoir", Reservoir(WIDTH, 4000, (16, 32, 64, 64, 128, 128)))
    check("PretrainedReservoir", PretrainedReservoir(WIDTH, 4000))

    # recon_online ruler: forward contract + a recon step runs and is finite
    recon = ReconReservoir(WIDTH, 4000, (16, 32, 64, 64, 128, 128)).to(DEVICE)
    check("ReconReservoir(fwd)", recon)
    xr = torch.randn(BATCH, 3, 128, 128, device=DEVICE)
    rl = recon.recon_loss(xr); rl.backward()
    assert torch.isfinite(rl), f"recon loss non-finite {rl}"
    print(f"      ReconReservoir: recon_loss={float(rl):.4f} grad_ok=True")

    # reseed: random reservoir weights change after reseed, contract preserved
    r = Reservoir(WIDTH, 4000, (16, 32, 64, 64, 128, 128)).to(DEVICE)
    x = torch.randn(BATCH, 3, 128, 128, device=DEVICE)
    with torch.no_grad():
        before = r(x).clone()
        r.reseed(9999)
        after = r(x)
    assert after.shape == before.shape, "reseed changed output shape"
    assert (before - after).abs().max() > 1e-4, "reseed did not change the reservoir"
    print("      Reservoir.reseed: output changed after reseed, shape preserved")

    # RFF ruler: contract + collapse->0, informative>0
    check("RFFReservoir", RFFReservoir(WIDTH, 4000, (16, 32, 64, 64, 128, 128)))

    # multi_random: stack and mean aggregation both finite & differentiable
    multi = MultiReservoir(WIDTH, 4000, K=4, channels=(16, 32, 64, 64, 128, 128)).to(DEVICE)
    xm = torch.randn(BATCH, 3, 128, 128, device=DEVICE)
    for agg in ("stack", "mean", "max"):
        z = torch.randn(BATCH, PROJ_DIM, device=DEVICE, requires_grad=True)
        s = epiplexity(z, xm, multi, direction="z2feat", multi_agg=agg); s.backward()
        assert torch.isfinite(s) and z.grad is not None
        print(f"      multi_random({agg}): score={float(s):8.4f} grad_ok=True")

    # direction flip z2feat on a single random ruler
    r2 = Reservoir(WIDTH, 4000, (16, 32, 64, 64, 128, 128)).to(DEVICE)
    z = torch.randn(BATCH, PROJ_DIM, device=DEVICE, requires_grad=True)
    s = epiplexity(z, xm, r2, direction="z2feat"); s.backward()
    zc = torch.ones(BATCH, PROJ_DIM, device=DEVICE)
    collapsed = float(epiplexity(zc, xm, r2, direction="z2feat"))
    assert torch.isfinite(s) and abs(collapsed) < 1e-3, f"z2feat: collapse should be 0, got {collapsed}"
    print(f"      z2feat direction: informative={float(s):8.4f} collapsed={collapsed:8.4f} grad_ok=True")

    print("OK: all reservoirs satisfy the score contract.")
