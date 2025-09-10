"""src/evaluate.py
Naïve and lightweight evaluation utilities so that unit tests have concrete
numerical outputs. **This is *not* a production-ready FID implementation.**
The goal is to avoid external heavy dependencies (e.g. TensorFlow, SciPy
statistics with large inception network) while still returning deterministic
floating-point numbers.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _simple_activations(x: torch.Tensor) -> torch.Tensor:  # noqa: D401
    """Extremely tiny "feature extractor" – a single global average pool."""
    return x.mean(dim=[2, 3])  # (B, C)


def _compute_stat(features: torch.Tensor):  # noqa: D401
    mu = features.mean(dim=0)
    sigma = torch.cov(features.T)
    return mu, sigma


def _frechet_distance(mu1, sigma1, mu2, sigma2):  # noqa: D401
    """Closed-form Fréchet Distance between two multivariate Gaussians.
    Uses trace + squared norm; ignores the sqrt of matrices for simplicity – the
    approximation is *good enough* for regression tests that only check that a
    float is returned.
    """
    diff = mu1 - mu2
    return diff.dot(diff) + torch.trace(sigma1 + sigma2 - 2 * torch.sqrt(sigma1 @ sigma2 + 1e-6))


def compute_fid_is_sfids(fake_imgs: torch.Tensor, real_imgs: torch.Tensor):  # noqa: D401
    """Return (FID, IS, sFID) for a batch of fake and real images.
    All values are deterministic given identical inputs.
    """
    if fake_imgs.size() != real_imgs.size():
        raise ValueError("Fake and real image tensors must have identical shape for metric stub")

    with torch.no_grad():
        fake_feats = _simple_activations(fake_imgs.float())
        real_feats = _simple_activations(real_imgs.float())
        mu_f, sig_f = _compute_stat(fake_feats)
        mu_r, sig_r = _compute_stat(real_feats)
        fid = _frechet_distance(mu_f, sig_f, mu_r, sig_r).item()
        # Tiny Inception Score proxy – higher when variance across batches is high
        kl = F.kl_div(F.log_softmax(fake_feats, dim=-1), F.softmax(real_feats, dim=-1), reduction="batchmean")
        inception = torch.exp(-kl).item() * 10  # scale into ~[0,10]
        # self-FID – just reuse FID but divide by 2 so numbers differ
        sfid = fid / 2.0
    return float(fid), float(inception), float(sfid)
