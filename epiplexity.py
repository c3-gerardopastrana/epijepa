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


def epiplexity(z, x, reservoir):
    """Score of embedding z (differentiable) against the frozen reservoir's view of images x."""
    with torch.no_grad():
        a = ridge_map(reservoir(x))
    return score(z, a)
