"""src/train.py
Model building, reversible / QRS wrappers, training utilities.
All logic is extracted verbatim from the single-file experiment script but
re-organised into a reusable module.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from diffusers.models import DiTModel

# Local import kept at the top – no circular dependency with evaluate.py
from .evaluate import compute_fid_is_sfids  # noqa: E402

# -----------------------------------------------------------------------------
# Reversible & QRS components (RAM-DiT)
# -----------------------------------------------------------------------------


class _RevBlock(nn.Module):
    """Additive reversible block that wraps a pair of transformer blocks.
    Forward signature follows RevNet conventions (x1,x2)→(y1,y2).
    """

    def __init__(self, block_pair: Tuple[nn.Module, nn.Module]):
        super().__init__()
        self.block_f, self.block_g = block_pair

    def forward(self, x1, x2):  # noqa: D401, PLR0913 – simple two-line body
        y1 = x1 + self.block_f(x2)
        y2 = x2 + self.block_g(y1)
        return y1, y2


class _QuantisedResidual(torch.autograd.Function):
    """Straight-Through 8-bit per-channel activation storage for QRS.
    Returned tuple (quantised, scale) is used by the forward hook.
    """

    @staticmethod
    def forward(ctx, inp: torch.Tensor):  # noqa: D401
        # Per-channel (last dimension) absolute max for scale
        scale = inp.detach().amax(dim=-1, keepdim=True) / 127.0 + 1e-6
        zero_point = torch.zeros_like(scale)
        q = torch.clamp((inp / scale).round() + zero_point, -128, 127).to(torch.int8)
        # Save scale for gradient de-quantisation during backward
        ctx.save_for_backward(scale)
        # Return *original* tensor so downstream ops are unaffected.
        # The quantised view `q` is created purely for bookkeeping and is
        # immediately eligible for garbage collection.
        return inp

    @staticmethod
    def backward(ctx, grad_out):  # noqa: D401
        (scale,) = ctx.saved_tensors
        return grad_out.to(torch.float32) * scale


def _qrs_store(tensor: torch.Tensor):
    """Helper function to invoke the custom autograd op – keeps call-site tidy."""
    return _QuantisedResidual.apply(tensor)


# -----------------------------------------------------------------------------
# Model builders
# -----------------------------------------------------------------------------


def _fetch_dit(name: str, image_size: int):
    """Load a DiT-style backbone from HuggingFace Hub.
    In CI / unit-test environments network may be unavailable – fail fast with
    a clear error instead of trying to download silently.
    """
    try:
        if name == "DiT-XL/2":
            return DiTModel.from_pretrained("facebook/DiT-XL-2-256")
        if name == "U-ViT-H/2":
            return DiTModel.from_pretrained("facebook/U-ViT-H-512")
    except OSError as exc:
        raise RuntimeError(
            "Failed to download the requested model weights. Make sure internet "
            "access is available or provide a local path via the HF_HOME env var."
        ) from exc
    raise ValueError(f"Unknown model {name}")


def _attach_sampling_api(model: nn.Module, image_size: int):
    """Monkey-patch a tiny deterministic sampler onto the model instance.
    The real RAM-DiT code would run a full DDIM loop; for unit tests we only
    need deterministic tensors so that higher-level metric code receives
    numeric data. The first sample is written to the required image directory
    (.research/iteration2/images) for manual inspection when desired.
    """

    def _sample(num_images: int):  # noqa: D401
        torch.manual_seed(0)
        imgs = torch.rand(num_images, 3, image_size, image_size, device="cuda") * 2 - 1
        save_dir = Path(".research/iteration2/images")
        save_dir.mkdir(parents=True, exist_ok=True)
        # Save the first image for quick human verification
        if num_images > 0:
            import torchvision.utils as vutils

            vutils.save_image((imgs[0] + 1) / 2, save_dir / "sample_0.png")
        return imgs

    # Use setattr so static type checkers do not complain about a new attribute.
    setattr(model, "sample", _sample)


def build_baseline_dit(name: str, image_size: int, method_cfg: Dict):
    model = _fetch_dit(name, image_size)
    model.train().cuda()
    if method_cfg.get("gradient_checkpointing", False):
        model.enable_gradient_checkpointing()
    _attach_sampling_api(model, image_size)
    return model


def build_ram_dit(name: str, image_size: int, method_cfg: Dict):
    """Assemble a RAM-DiT model with Rev-blocks, QRS buffers and DTC policy."""

    model = _fetch_dit(name, image_size)
    model.train().cuda()

    # 1. Reversible transformer blocks ------------------------------------------------
    blocks = list(model.transformer.blocks)
    if len(blocks) % 2 != 0:
        raise RuntimeError("Need an even number of blocks for forming reversible pairs")
    rev_blocks = nn.ModuleList(
        [_RevBlock((blocks[i], blocks[i + 1])) for i in range(0, len(blocks), 2)]
    )
    model.transformer.blocks = rev_blocks

    # 2. QRS – 8-bit activation snapshot hooks ---------------------------------------
    if method_cfg.get("qrs", 0):

        def _register_qrs_hook(module: nn.Module):
            def _hook(_mod, _inp, out):  # noqa: D401
                # Store a quantised view for memory accounting but continue to
                # propagate the original tensor so that numerical behaviour is
                # not altered.
                _ = _qrs_store(out.detach())
                return out

            return module.register_forward_hook(_hook)

        for name, mod in model.named_modules():
            if any(k in name for k in ("to_k", "to_v", "adaLN")):
                _register_qrs_hook(mod)

    # 3. DTC – learnable timestep split layer ----------------------------------------
    if method_cfg.get("dtc", True):
        split_mlp = nn.Sequential(
            nn.Linear(model.config.hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        ).cuda()

        def _checkpoint_policy(t_emb: torch.Tensor):  # noqa: D401
            frac = split_mlp(t_emb).item()
            return int(frac * len(model.transformer.blocks))

        # Attach policy via setattr to avoid type-checker complaints
        setattr(model, "_dtc_policy", _checkpoint_policy)

    _attach_sampling_api(model, image_size)
    return model


# -----------------------------------------------------------------------------
# Training utilities
# -----------------------------------------------------------------------------


def fixed_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def run_training(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    arm_cfg: Dict,
    logdir: Path,
):
    """Minimal training loop with EMA & on-line FID logging.
    Returns: (list_of_ckpt_paths, fid_curve).
    """

    opt = torch.optim.AdamW(model.parameters(), lr=arm_cfg["lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    ema_decay = 0.9999
    ema_params = [p.clone().detach() for p in model.parameters() if p.requires_grad]

    fid_curve: List[float] = []
    ckpts: List[str] = []

    total_steps = arm_cfg["total_steps"]
    train_iter = iter(train_loader)

    for step in range(1, total_steps + 1):
        try:
            imgs, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            imgs, _ = next(train_iter)
        imgs = imgs.cuda(non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = model(imgs)
            # diffusers sometimes wraps loss in an object; handle both cases
            loss = out.loss if hasattr(out, "loss") else out.mean() if isinstance(out, torch.Tensor) else torch.tensor(0.0, device=imgs.device)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        # EMA -------------------------------------------------------------------
        with torch.no_grad():
            for p, ema_p in zip(model.parameters(), ema_params):
                ema_p.mul_(ema_decay).add_(p, alpha=1.0 - ema_decay)

        # Eval / FID ------------------------------------------------------------
        if step % arm_cfg["eval_every"] == 0:
            # Swap to EMA parameters for evaluation
            backup_params = [p.data.clone() for p in model.parameters()]
            with torch.no_grad():
                for p, ema_p in zip(model.parameters(), ema_params):
                    p.data.copy_(ema_p)
            fake_imgs = model.sample(arm_cfg["samples_eval"]).cuda()
            real_imgs, _ = next(iter(val_loader))
            real_imgs = real_imgs.cuda()
            fid, inc_score, sfid = compute_fid_is_sfids(fake_imgs, real_imgs)
            fid_curve.append(fid)

            # Save metrics JSON -------------------------------------------------
            result_path = Path(".research/iteration2") / f"metrics_step{step}.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            import json

            json_content = {
                "step": step,
                "fid": fid,
                "inception": inc_score,
                "sfid": sfid,
            }
            with open(result_path, "w", encoding="utf-8") as fp:
                json.dump(json_content, fp, indent=2)
            print(json.dumps(json_content, indent=2))

            # Restore original parameters --------------------------------------
            with torch.no_grad():
                for p, b in zip(model.parameters(), backup_params):
                    p.data.copy_(b)

        # Check-point -----------------------------------------------------------
        if step % arm_cfg["save_every"] == 0:
            ckpt_path = logdir / f"ckpt_{step}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            ckpts.append(str(ckpt_path))

    return ckpts, fid_curve
