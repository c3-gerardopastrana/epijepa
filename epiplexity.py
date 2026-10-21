"""Epiplexity: how much of an embedding a ridge readout from a frozen random CNN can predict.

score(z, a) = 0.5 * log2 det(I + eta * W^T W),   W = a @ (z - mean(z))

where a is the ridge map from the reservoir's features on the same batch. A collapsed
embedding scores ~0; spreading predictable structure across more directions raises it.
Maximising this score, traded off against a view-invariance loss, is the whole method.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

RIDGE = 3.0  # readout regularisation
ETA = 30.0   # saturation inside the log-det


class ChannelNorm(nn.Module):
    def forward(self, x):
        return F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],)).permute(0, 3, 1, 2)


class Reservoir(nn.Module):
    """Frozen random CNN. Stride 2 after the first stage, so `channels` sets the depth."""

    def __init__(self, width, seed, channels=(16, 32, 64, 64)):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            layers, inc = [], 3
            for i, c in enumerate(channels):
                layers += [nn.Conv2d(inc, c, 3, 1 if i == 0 else 2, 1), ChannelNorm(), nn.ELU()]
                inc = c
            self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(2), nn.Flatten(),
                                     nn.Linear(channels[-1] * 4, width))
        self.requires_grad_(False)

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def reseed(self, seed):
        """Re-draw all weights from a fresh seed, in place (for --reservoir random_resample).

        Encoder-independent: new weights depend only on `seed`, never on the encoder, so
        this cannot create the collusion failure mode. Stays frozen (no_grad)."""
        with torch.random.fork_rng(devices=[0] if torch.cuda.is_available() else []):
            torch.manual_seed(seed)
            for m in self.net:
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    m.reset_parameters()
        self.requires_grad_(False)

    @torch.no_grad()
    def drift(self, seed, alpha):
        """OU drift toward fresh random weights: w <- sqrt(1-a) w + sqrt(a) w_new.

        Keeps the weight variance stationary (same init distribution), so the ruler moves
        smoothly without shrinking or drifting off-distribution. Encoder-independent."""
        fresh = Reservoir(self.net[-1].out_features, seed,
                          tuple(m.out_channels for m in self.net if isinstance(m, nn.Conv2d))).to(
                              next(self.parameters()).device)
        for p, q in zip(self.parameters(), fresh.parameters()):
            p.mul_(math.sqrt(1 - alpha)).add_(q, alpha=math.sqrt(alpha))
        self.requires_grad_(False)


class PretrainedReservoir(nn.Module):
    """Frozen *pretrained* CNN ruler (PoC #1).

    Drop-in replacement for `Reservoir` with the identical contract: takes the
    already-ImageNet-normalised image batch the training pipeline feeds and returns
    `[batch, width]` features. We keep an early slice of a pretrained ResNet (through
    `layer2`), global-average-pool it, then apply a fixed seeded random linear projection
    to `width`. Using an early block keeps the ruler a generic feature extractor rather
    than an ImageNet-class detector, which is the fair test of a stronger fixed ruler.

    Everything downstream (standardisation in `ridge_map`, the ridge solve, the log-det
    score, and the S0 calibration) is unchanged, so this isolates ruler *quality*.
    """

    def __init__(self, width, seed, arch="resnet18"):
        super().__init__()
        import torchvision
        weights = {"resnet18": torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
                   "resnet34": torchvision.models.ResNet34_Weights.IMAGENET1K_V1}[arch]
        backbone = getattr(torchvision.models, arch)(weights=weights)
        # conv1, bn1, relu, maxpool, layer1, layer2  -> 128 channels for resnet18/34
        self.body = nn.Sequential(*list(backbone.children())[:6])
        feat_ch = 128
        self.pool = nn.AdaptiveAvgPool2d(1)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.proj = nn.Linear(feat_ch, width)
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        h = self.pool(self.body(x)).flatten(1)
        return self.proj(h)


class ReconReservoir(nn.Module):
    """Reservoir whose encoder is the SAME random-CNN architecture as `Reservoir`, but
    trained ONLINE to reconstruct the input frames (its own decoder + MSE recon loss).

    The epiplexity score reads `forward(x)` under no_grad (see `epiplexity`), so the score
    never shapes this ruler -- only reconstruction does. That keeps the ruler
    encoder-independent (no collusion). Step `recon_loss(x)` with its own optimiser each
    step; use `forward(x)` for the (read-time frozen) features fed to the ridge/score.
    """

    def __init__(self, width, seed, channels=(16, 32, 64, 64)):
        super().__init__()
        self.encoder = Reservoir(width, seed, channels)
        self.encoder.requires_grad_(True)  # this ruler IS trained, by reconstruction
        self.dec_ch = channels[-1]
        self.dec_fc = nn.Linear(width, channels[-1] * 16)  # -> 4x4 map
        ups, inc = [], channels[-1]
        for c in (64, 32, 16, 8):
            ups += [nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(inc, c, 3, 1, 1), ChannelNorm(), nn.ELU()]
            inc = c
        self.decoder = nn.Sequential(*ups, nn.Conv2d(inc, 3, 3, 1, 1))

    def forward(self, x):
        return self.encoder(x)

    def recon_loss(self, x):
        z = self.encoder(x)
        h = self.dec_fc(z).reshape(z.shape[0], self.dec_ch, 4, 4)
        xhat = F.interpolate(self.decoder(h), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return F.mse_loss(xhat, x)


class RFFReservoir(nn.Module):
    """Random Fourier Features on top of a frozen random-CNN descriptor.

    phi(u) = sqrt(2/D) cos(Omega u + b), with Omega drawn at a MIXTURE of bandwidths so
    the ruler covers multiple length-scales (local+global). This is an unbiased estimator
    of an RBF-kernel feature map (Rahimi & Recht 2007) in the random-CNN descriptor space,
    which is why it's a principled random ruler rather than an arbitrary one.
    """

    def __init__(self, width, seed, channels=(16, 32, 64, 64), bandwidths=(0.5, 1.0, 2.0, 4.0)):
        super().__init__()
        self.backbone = Reservoir(width, seed, channels)   # frozen random-CNN descriptor u
        D_per = max(width // len(bandwidths), 1)
        with torch.random.fork_rng(devices=[0] if torch.cuda.is_available() else []):
            torch.manual_seed(seed + 1)
            omegas, biases, scales = [], [], []
            for i, sigma in enumerate(bandwidths):
                omegas.append(torch.randn(width, D_per) / sigma)      # N(0, 1/sigma^2)
                biases.append(torch.rand(D_per) * 2 * math.pi)
                scales.append(sigma)
            self.register_buffer("omega", torch.cat(omegas, dim=1))   # [width, D]
            self.register_buffer("bias", torch.cat(biases))           # [D]
        self.D = self.omega.shape[1]
        self.requires_grad_(False)

    def forward(self, x):
        u = self.backbone(x)                                          # [N, width]
        proj = u @ self.omega + self.bias                             # [N, D]
        return math.sqrt(2.0 / self.D) * torch.cos(proj)


class MultiReservoir(nn.Module):
    """K independent random-CNN reservoirs. Returns a LIST of K feature sets; the score
    aggregates them by 'stack' (concat -> one log-det, rewards diversity) or 'mean'
    (average K scores -> Monte-Carlo estimate of E_R[S]).

    NOTE: K iid same-arch random CNNs all approximate the SAME NNGP kernel, so their
    blind spots are redundant -- averaging/stacking just reduces sampling noise around one
    kernel. For genuinely diverse rulers (different kernels), use MultiRFFReservoir."""

    def __init__(self, width, seed, K=4, channels=(16, 32, 64, 64)):
        super().__init__()
        self.nets = nn.ModuleList([Reservoir(width, seed + 100 * k, channels) for k in range(K)])
        self.requires_grad_(False)

    def forward(self, x):
        return [net(x) for net in self.nets]


class MultiRFFReservoir(nn.Module):
    """A DIVERSE bank of K Random-Fourier-Feature rulers on a shared frozen random-CNN
    descriptor. Two diversity modes:

    - kernel_family='rbf'  (default): K RBF kernels at DIFFERENT bandwidths (Gaussian
      omega / sigma). Different length-scales -> local<->global blind spots.
    - kernel_family='mixed': K DIFFERENT kernel families, one per ruler, by sampling omega
      from different spectral densities (Bochner duality):
        RBF        <- Gaussian(omega)        (smooth)
        Laplacian  <- Cauchy/standard-Cauchy (heavy tail -> spiky, local)
        Cauchy     <- Laplace(omega)         (exponential spectrum)
        Matern-1/2 <- Student-t(omega)       (intermediate roughness)
      Different families = different *shapes* of similarity, a stronger notion of diverse
      blind spots than bandwidth alone.

    All bandwidths/scales use the median pairwise distance (median heuristic) so every
    kernel is in-regime and a min-max soft-min cannot degenerate onto a mis-scaled ruler.
    Returns a LIST of K feature sets.
    """

    def __init__(self, width, seed, K=4, channels=(16, 32, 64, 64),
                 sigma_mults=(0.5, 1.0, 2.0, 4.0), kernel_family="rbf"):
        super().__init__()
        assert len(sigma_mults) == K, "need one sigma multiplier per ruler"
        self.backbone = Reservoir(width, seed, channels)   # shared random-CNN descriptor
        self.sigma_mults = sigma_mults
        self.kernel_family = kernel_family
        self.D = width
        families = ["rbf", "laplacian", "cauchy", "matern"]
        with torch.random.fork_rng(devices=[0] if torch.cuda.is_available() else []):
            torch.manual_seed(seed + 1)
            omegas = []
            for k in range(K):
                if kernel_family == "mixed":
                    fam = families[k % len(families)]
                    omegas.append(self._sample_omega(fam, width))
                else:  # rbf: all Gaussian, bandwidth varied at forward
                    omegas.append(torch.randn(width, width))
            self.register_buffer("omega", torch.stack(omegas))            # [K, width, D]
            self.register_buffer("bias", torch.rand(K, width) * 2 * math.pi)
        self.requires_grad_(False)

    @staticmethod
    def _sample_omega(family, width):
        """Spectral density whose Fourier dual is the named kernel (Bochner)."""
        if family == "rbf":            # Gaussian spectrum
            return torch.randn(width, width)
        if family == "laplacian":      # standard Cauchy spectrum (heavy tails)
            return torch.distributions.Cauchy(0.0, 1.0).sample((width, width))
        if family == "cauchy":         # Laplace/exponential spectrum
            return torch.distributions.Laplace(0.0, 1.0).sample((width, width))
        if family == "matern":         # Student-t (df=3) spectrum, intermediate roughness
            return torch.distributions.StudentT(3.0).sample((width, width))
        raise ValueError(family)

    def forward(self, x):
        u = self.backbone(x)                                   # [N, width]
        with torch.no_grad():
            d = torch.cdist(u, u)
            med = d[d > 0].median().clamp_min(1e-6)
        outs = []
        for k, m in enumerate(self.sigma_mults):
            # for 'mixed', each ruler already has its family's omega; still scale by the
            # median heuristic (and the per-ruler sigma_mult) to keep all in-regime.
            sigma = m * med
            proj = (u @ self.omega[k]) / sigma + self.bias[k]  # [N, D]
            outs.append(math.sqrt(2.0 / self.D) * torch.cos(proj))
        return outs


class RandomLinearReservoir(nn.Module):
    """Random LINEAR measurement ruler: y = R vec(x), R a fixed random m x D map.

    Basis-agnostic observer of the image (compressed-sensing / RIP view): with enough rows
    a random linear map preserves image information in no particular basis, and -- unlike
    iid random CNNs, which all approximate ONE NNGP kernel -- independent draws of R probe
    genuinely different random subspaces, giving the diversity a min-max family needs.

    structured=True uses a fast sign-flip + rFFT projection (O(D log D), SRHT-like) instead
    of a dense O(mD) matmul. resample=True re-draws R every forward (dynamic min-max).
    """

    def __init__(self, width, seed, in_dim=3 * 128 * 128, structured=True, resample=False):
        super().__init__()
        self.m, self.in_dim, self.structured, self.resample = width, in_dim, structured, resample
        self.seed, self._step = seed, 0
        self._build(seed)

    def _build(self, seed):
        g = torch.Generator().manual_seed(seed)
        if self.structured:
            # SRHT-like: random sign flip -> rFFT -> random frequency subsample. Cheap.
            self.register_buffer("signs", (torch.randint(0, 2, (self.in_dim,), generator=g) * 2 - 1).float())
            freqs = self.in_dim // 2 + 1
            self.register_buffer("idx", torch.randperm(freqs, generator=g)[: self.m])
        else:
            self.register_buffer("R", torch.randn(self.m, self.in_dim, generator=g) / math.sqrt(self.in_dim))
        self.requires_grad_(False)

    def forward(self, x):
        if self.resample:
            self._build(self.seed + 1000 * self._step)
            if x.is_cuda:
                self.cuda()
            self._step += 1
        xv = x.flatten(1)                                        # [N, D]
        if self.structured:
            z = xv * self.signs
            F = torch.fft.rfft(z, dim=1)                         # [N, D//2+1] complex
            return F[:, self.idx].real                           # [N, m]
        return xv @ self.R.T                                     # [N, m]


class MinMaxReservoir(nn.Module):
    """Faithful max-min objective: on EVERY forward, draw a FRESH family of K random
    reservoirs (new seeds from an internal step counter) and return their K feature sets.

    Paired with multi_agg='max' (soft-min), this implements  max_z min_R S(z; R)  with the
    inner min ranging over the *distribution* of random observers -- not 4 frozen rulers.
    Because the family is resampled each step, the worst-case ruler changes as z improves,
    so the 'min' is a genuine moving worst-case (an adversary sampled from the family),
    fixing the static-adversary limitation of a fixed MultiReservoir.

    Encoder-independent: seeds depend only on the step counter, never on z -> no collusion.
    All reservoirs are frozen (no_grad); gradients flow only through z in the score.
    """

    def __init__(self, width, seed, K=4, channels=(16, 32, 64, 64)):
        super().__init__()
        self.width, self.base_seed, self.K, self.channels = width, seed, K, channels
        self._step = 0
        # build the K reservoirs ONCE; each forward reseeds them in place (cheap) rather
        # than reconstructing new modules + moving to GPU (the old, expensive path).
        self.nets = nn.ModuleList([Reservoir(width, seed + k, channels) for k in range(K)])
        self.requires_grad_(False)

    def forward(self, x):
        # fresh family each call via in-place reseed (no module construction / device moves)
        for k, n in enumerate(self.nets):
            n.reseed(self.base_seed + 1000 * self._step + k)
        self._step += 1
        with torch.no_grad():
            return [n(x) for n in self.nets]


class MultiScaleReservoir(nn.Module):
    """K observers that pool a shared random conv feature map to K DIFFERENT spatial grids
    (e.g. 1x1 global -> 8x8 local), each with its own random linear readout to `width`.

    Unlike iid random CNNs (which all approximate the SAME NNGP kernel -> redundant blind
    spots), these observers differ in the *scale of structure* they expose: the 1x1 pool
    sees global layout, the 8x8 pool sees local detail. This gives the min-max a family
    with genuinely complementary blind spots -- the diversity that was missing.

    resample=True re-draws the conv + readouts every forward (for the dynamic max-min);
    seeds depend only on a step counter, never on z (encoder-independent, no collusion).
    """

    def __init__(self, width, seed, grids=(1, 2, 4, 8), channels=(16, 32, 64, 64), resample=False):
        super().__init__()
        self.width, self.base_seed, self.grids, self.channels = width, seed, grids, channels
        self.resample = resample
        self._step = 0
        self._build(seed)

    def _build(self, seed):
        with torch.random.fork_rng(devices=[0] if torch.cuda.is_available() else []):
            torch.manual_seed(seed)
            layers, inc = [], 3
            for i, c in enumerate(self.channels):
                layers += [nn.Conv2d(inc, c, 3, 1 if i == 0 else 2, 1), ChannelNorm(), nn.ELU()]
                inc = c
            self.backbone = nn.Sequential(*layers)
            self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(g) for g in self.grids])
            self.readouts = nn.ModuleList(
                [nn.Linear(self.channels[-1] * g * g, self.width) for g in self.grids])
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        if self.resample:
            self._build(self.base_seed + 1000 * self._step)
            self._step += 1
            if x.is_cuda:
                self.cuda()
        fmap = self.backbone(x)
        return [ro(p(fmap).flatten(1)) for p, ro in zip(self.pools, self.readouts)]


def ridge_map(h, lam=RIDGE):
    """(H^T H + lam I)^-1 H^T on standardised features, via QR. Shape [features, batch]."""
    h = ((h - h.mean(0)) / h.std(0, unbiased=False).clamp_min(1e-6) / math.sqrt(h.shape[1])).double()
    aug = torch.cat([h, math.sqrt(lam) * torch.eye(h.shape[1], dtype=h.dtype, device=h.device)])
    q, r = torch.linalg.qr(aug, mode="reduced")
    return torch.linalg.solve_triangular(r, q[: h.shape[0]].T, upper=True)


def score(z, a, eta=ETA):
    z = z.double()
    w = a @ (z - z.mean(0))
    eye = torch.eye(w.shape[1], dtype=w.dtype, device=w.device)
    return 0.5 * torch.linalg.slogdet(eye + eta * w.T @ w)[1] / math.log(2)


def score_z2feat(z, h, eta=ETA, lam=RIDGE):
    """Flipped direction (Axis D): z is the REGRESSOR, random features h the TARGET.

    W = (Z^T Z + rho I)^-1 Z^T H_c  -- 'how well does z linearly predict the random
    features'. Score = 1/2 log2 det(I + eta W W^T) via the small (d x d) form (Sylvester),
    so stacking many target features stays cheap. This encodes 'z is linearly sufficient
    for the random targets', i.e. 'for any random y, regression from z is meaningful'.
    """
    z = (z - z.mean(0)).double()
    h = ((h - h.mean(0)) / h.std(0, unbiased=False).clamp_min(1e-6) / math.sqrt(h.shape[1])).double()
    ztz = z.T @ z + lam * torch.eye(z.shape[1], dtype=z.dtype, device=z.device)
    w = torch.linalg.solve(ztz, z.T @ h)                 # [d, feat]
    eye = torch.eye(w.shape[0], dtype=w.dtype, device=w.device)
    return 0.5 * torch.linalg.slogdet(eye + eta * w @ w.T)[1] / math.log(2)


def score_z2feat_batched(z, hs, eta=ETA, lam=RIDGE):
    """Batched z2feat: K per-ruler scores in ONE batched solve + slogdet (no Python loop).

    hs: list of K target feature matrices, each [N, m]. Returns a length-K tensor of scores.
    The K ridge solves (Z^T Z + rho I)^-1 Z^T H_k share the SAME d x d system, so we solve
    the shared system once and batch only the K right-hand sides; the K log-dets are a single
    batched slogdet over a [K, d, d] stack. This makes K add negligible wall-clock vs K=1."""
    z = (z - z.mean(0)).double()
    H = torch.stack([((h - h.mean(0)) / h.std(0, unbiased=False).clamp_min(1e-6)
                      / math.sqrt(h.shape[1])).double() for h in hs])      # [K, N, m]
    ztz = z.T @ z + lam * torch.eye(z.shape[1], dtype=z.dtype, device=z.device)  # [d, d] shared
    rhs = torch.einsum("dn,knm->kdm", z.T, H)                              # [K, d, m]
    w = torch.linalg.solve(ztz, rhs)                                       # [K, d, m] (shared LHS)
    eye = torch.eye(w.shape[1], dtype=w.dtype, device=w.device)
    gram = eye + eta * torch.einsum("kdm,kem->kde", w, w)                  # [K, d, d]
    return 0.5 * torch.linalg.slogdet(gram)[1] / math.log(2)               # [K]


def epiplexity(z, x, reservoir, direction="feat2z", multi_agg="stack", softmin_tau=0.5):
    """Score of embedding z (differentiable) against a frozen reservoir's view of x.

    direction: 'feat2z' (features->z, the original) or 'z2feat' (z->features, the reframe).
    multi_agg (when the reservoir returns a list of K feature sets):
      'stack' concatenates them into one score (joint/union, rewards diversity);
      'mean'  averages the K scores (average-case, E_R[S]);
      'max'   soft worst-case over rulers: -tau*logsumexp(-S_k/tau). This is the
              differentiable min-max objective -- it maximises the WORST ruler's score,
              so no single random direction can be ignored. softmin_tau->0 is a hard min
              (sparse gradient); larger tau softens toward the mean.
    """
    with torch.no_grad():
        feats = reservoir(x)
    feats_list = feats if isinstance(feats, (list, tuple)) else [feats]

    def one(h):
        if direction == "z2feat":
            return score_z2feat(z, h)
        return score(z, ridge_map(h))

    # z2feat mean/max with K>1: batch the K solves+slogdets into one op (K adds ~no cost)
    if direction == "z2feat" and multi_agg in ("mean", "max") and len(feats_list) > 1:
        scores = score_z2feat_batched(z, feats_list)      # [K]
        if multi_agg == "mean":
            return scores.mean()
        return -softmin_tau * torch.logsumexp(-scores / softmin_tau, dim=0)

    if multi_agg == "mean":
        return torch.stack([one(h) for h in feats_list]).mean()
    if multi_agg == "max":
        scores = torch.stack([one(h) for h in feats_list])
        if scores.numel() == 1:
            return scores.squeeze()
        # soft-min (worst-case) score: differentiable, temperature-controlled
        return -softmin_tau * torch.logsumexp(-scores / softmin_tau, dim=0)
    # stack: concat features along the feature dim, single score
    h = torch.cat(feats_list, dim=1) if len(feats_list) > 1 else feats_list[0]
    return one(h)
