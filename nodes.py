"""ComfyUI nodes for DyPE-M on MOSS-Image."""

from __future__ import annotations

import json
import math
import os
import time
import torch

import comfy.model_base
import comfy.patcher_extension as pe

from .dype_core import (
    AXES, METHODS, MSCALE_MODES, SCHEDULES,
    DyPEConfig, build_pe, compute_kappa, compute_kappa_pair, describe_schedule,
    effective_scales, spectral_bandwidths, spectral_detail, spectral_stats,
)

_TRACE_PATH = os.environ.get("MOSS_DYPE_TRACE")


def _trace(record: dict) -> None:
    path = os.environ.get("MOSS_DYPE_TRACE") or _TRACE_PATH
    if not path:
        return
    try:
        record.setdefault("t", round(time.time(), 3))
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _is_moss(model) -> bool:
    try:
        return isinstance(model, comfy.model_base.MOSSImage)
    except Exception:
        return False


def _model_geometry(diffusion_model) -> tuple[int, float]:
    """(head_dim, rope theta) straight off the loaded module."""
    head_dim, theta = 128, 10000.0
    embed = getattr(diffusion_model, "pe_embedder", None)
    if embed is not None:
        axes = getattr(embed, "axes_dim", None)
        if axes:
            head_dim = int(sum(axes))
        theta = float(getattr(embed, "theta", theta))
    return head_dim, theta


def _step_from_sigmas(sigmas, sigma: float):
    """Index of the current sigma inside the sampler's schedule."""
    if sigmas is None or not torch.is_tensor(sigmas) or sigmas.numel() < 2:
        return None, None
    s = sigmas.detach().float().flatten().cpu()
    total = max(int(s.numel()) - 1, 1)
    idx = int((s > sigma + 1e-7).sum().item())
    return min(idx, total), total


def _make_rope(original, cfg: DyPEConfig, kappa_h: float, kappa_w: float):
    """Wrap ``MOSSImage.rope`` so the h/w axes use DyPE-M frequencies."""

    def rope(h, w, text_len, device):
        if kappa_h <= 0.0 and kappa_w <= 0.0:
            return original(h, w, text_len, device)
        out = build_pe(h, w, text_len, device, cfg, kappa_h, kappa_w)
        if _TRACE_PATH:
            base = build_pe(h, w, text_len, device, cfg, 0.0, 0.0)
            s_h, s_w = effective_scales(h, w, cfg)
            _trace({"event": "rope", "lat_h": h, "lat_w": w, "text_len": text_len,
                    "kappa_h": round(kappa_h, 5), "kappa_w": round(kappa_w, 5),
                    "pe_delta": round(float((out - base).abs().mean()), 6),
                    "scale_h": round(s_h, 4), "scale_w": round(s_w, 4)})
        return out

    return rope


def _make_wrapper(cfg: DyPEConfig, stats: dict):
    """DIFFUSION_MODEL wrapper: derives kappa, installs the rope, restores it."""

    def wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
        dm = executor.class_obj
        original = getattr(dm, "rope", None)
        if original is None or not hasattr(dm, "pe_embedder"):
            return executor(x, timestep, context, transformer_options, **kwargs)

        try:
            sigma = float(timestep.detach().reshape(-1)[0]) if torch.is_tensor(timestep) \
                else float(timestep)
        except Exception:
            sigma = 1.0

        step, total = _step_from_sigmas(transformer_options.get("sample_sigmas"), sigma)
        if step is None:
            # fall back to counting forwards; a rising sigma means a new run
            if sigma > stats.get("last_sigma", 2.0) + 1e-6:
                stats["count"] = 0
            step = stats.get("count", 0)
            total = stats.get("total", 1)
            stats["count"] = step + 1
        stats["last_sigma"] = sigma

        if step == 0:
            stats["detail"] = (None, None)   # new sampling run
            stats["bw"] = None

        _, _, lat_h, lat_w = x.shape
        s_h = max(lat_h / cfg.base_patches, 1.0)
        s_w = max(lat_w / cfg.base_patches, 1.0)
        detail_h, detail_w = stats.get("detail", (None, None))

        k_h, k_w = compute_kappa_pair(cfg, sigma, step, total,
                                      (detail_h, detail_w), stats.get("bw"))
        stats["sigma"], stats["step"], stats["total"] = sigma, step, total
        stats["kappa"] = (k_h, k_w)
        _trace({"event": "step", "step": step, "total": total,
                "sigma": round(sigma, 5), "kappa_h": round(k_h, 5),
                "kappa_w": round(k_w, 5),
                "detail_h": None if detail_h is None else round(detail_h, 4),
                "detail_w": None if detail_w is None else round(detail_w, 4),
                "scale_h": round(s_h, 4), "scale_w": round(s_w, 4),
                "schedule": cfg.schedule})

        dm.rope = _make_rope(original, cfg, k_h, k_w)
        try:
            out = executor(x, timestep, context, transformer_options, **kwargs)
        finally:
            dm.rope = original

        # measure the emerging spectrum for the *next* step (flow: x0 = x - s*v)
        if cfg.enabled and cfg.schedule in ("spectral", "aniso") and torch.is_tensor(out):
            try:
                with torch.no_grad():
                    x0 = (x - sigma * out).float()
                    stats["detail"] = spectral_detail(x0, (s_h, s_w), cfg.spectral_hi)
                    stats["bw"] = spectral_bandwidths(x0, cfg.spectral_hi)
                    if _TRACE_PATH or os.environ.get("MOSS_DYPE_TRACE"):
                        sh = spectral_stats(x0, 2, 0.5 / max(s_h, 1.0))
                        sw = spectral_stats(x0, 3, 0.5 / max(s_w, 1.0))
                        _trace({"event": "spectrum", "step": step,
                                "h": {k: round(v, 4) for k, v in sh.items()},
                                "w": {k: round(v, 4) for k, v in sw.items()}})
            except Exception:
                stats["detail"] = (None, None)
        return out

    return wrapper


# --------------------------------------------------------------------------- #
# nodes
# --------------------------------------------------------------------------- #
class MossDyPEPatch:
    """Patch a MOSS-Image model with DyPE-M dynamic position extrapolation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "method": (list(METHODS), {"default": "yarn"}),
                "schedule": (list(SCHEDULES), {"default": "adaptive"}),
                "base_resolution": ("INT", {
                    "default": 1024, "min": 256, "max": 4096, "step": 16,
                    "tooltip": "Reference resolution the model was trained at. "
                               "Scale = patch extent / base_patches. "
                               "No-op at or below this resolution.",
                }),
                "lambda_s": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0,
                                       "step": 0.05, "tooltip": "Overall scaling strength."}),
                "lambda_t": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0,
                                       "step": 0.1, "tooltip": "Decay exponent of kappa(t)."}),
            },
            "optional": {
                "target_scale": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 4.0, "step": 0.05,
                    "tooltip": "Pin the extrapolation strength instead of deriving it from "
                               "base_resolution. 0 = off (use base_resolution). 1.25 means "
                               "'pretend the model is 1.25x smaller than this canvas' at ANY "
                               "resolution, so one setting transfers from 6MP to 8MP to 16MP. "
                               "Smaller = milder = less object inflation.",
                }),
                "isotropic_scale": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Derive ONE scale from the pixel area, applied to both axes. "
                               "Per-axis scaling (False) makes the axes disagree by exactly "
                               "the aspect ratio - a 25% horizontal stretch at 3200x2560 and "
                               "78% at 4096x2304. Only turn off to reproduce the old behaviour.",
                }),
                "mscale_mode": (list(MSCALE_MODES), {"default": "kappa"}),
                "spectral_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,
                                              "step": 0.05,
                                              "tooltip": "Blend of the measured spectrum against "
                                                         "the analytic time curve (spectral only)."}),
                "spectral_hi": ("FLOAT", {"default": 0.9, "min": 0.5, "max": 0.99, "step": 0.01,
                                          "tooltip": "Cumulative energy fraction that defines the "
                                                     "bandwidth."}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "axes": (list(AXES), {"default": "both"}),
                "fix_ramp_clamp": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "model/patch/moss_image"
    DESCRIPTION = ("DyPE-M: time-scheduled, spectrum-adaptive RoPE extrapolation for "
                   "ultra-high-resolution MOSS-Image. Lets the model render well beyond "
                   "its 4MP training range.")

    def apply(self, model, enabled, method, schedule, base_resolution,
              lambda_s, lambda_t, target_scale=0.0, isotropic_scale=True,
              mscale_mode="kappa",
              spectral_weight=0.5,
              spectral_hi=0.9, start_percent=0.0, end_percent=1.0,
              axes="both", fix_ramp_clamp=True):
        if not _is_moss(model.model):
            raise ValueError(
                "DyPE-M: this node patches MOSS-Image only "
                f"(got {type(model.model).__name__})."
            )

        dm = model.model.diffusion_model
        head_dim, theta = _model_geometry(dm)

        cfg = DyPEConfig(
            enabled=enabled, method=method, schedule=schedule,
            base_resolution=int(base_resolution), lambda_s=float(lambda_s),
            target_scale=float(target_scale),
            isotropic_scale=bool(isotropic_scale),
            lambda_t=float(lambda_t), mscale_mode=mscale_mode,
            start_percent=float(start_percent), end_percent=float(end_percent),
            axes=axes, fix_ramp_clamp=bool(fix_ramp_clamp),
            spectral_weight=float(spectral_weight), spectral_hi=float(spectral_hi),
            patch_size=16, theta=theta, head_dim=head_dim,
        )

        if cfg.target_scale > 0.0:
            print(f"[DyPE-M] target_scale={cfg.target_scale:g} is in control; "
                  f"base_resolution={cfg.base_resolution} is ignored. "
                  f"(Set target_scale=0 to drive it from base_resolution.)", flush=True)
        else:
            print(f"[DyPE-M] strength from base_resolution={cfg.base_resolution} "
                  f"(target_scale=0), isotropic={cfg.isotropic_scale}", flush=True)

        patched = model.clone()
        patched.add_wrapper_with_key(
            pe.WrappersMP.DIFFUSION_MODEL, "moss_dype", _make_wrapper(cfg, {})
        )
        patched.model_options.setdefault("moss_dype", cfg.to_dict())
        return (patched,)


class MossDyPESchedule:
    """Preview the kappa schedule DyPE-M will use (text report)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "method": (list(METHODS), {"default": "yarn"}),
                "schedule": (list(SCHEDULES), {"default": "hybrid"}),
                "lambda_s": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "lambda_t": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1}),
                "base_resolution": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "report"
    CATEGORY = "model/patch/moss_image"
    OUTPUT_NODE = True

    def report(self, steps, shift, method, schedule, lambda_s, lambda_t, base_resolution):
        cfg = DyPEConfig(
            method=method, schedule=schedule, base_resolution=int(base_resolution),
            lambda_s=float(lambda_s), lambda_t=float(lambda_t),
        )
        return (describe_schedule(cfg, int(steps), float(shift)),)


class MossDyPEResolution:
    """Convert a target megapixel budget into 16-aligned MOSS dimensions."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 5.0,
                                           "step": 0.01, "tooltip": "width / height"}),
                "megapixels": ("FLOAT", {"default": 6.0, "min": 0.25, "max": 64.0, "step": 0.25}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT")
    RETURN_NAMES = ("width", "height", "actual_megapixels")
    FUNCTION = "compute"
    CATEGORY = "model/patch/moss_image"

    def compute(self, aspect_ratio, megapixels):
        total = float(megapixels) * 1_000_000.0
        h = math.sqrt(total / float(aspect_ratio))
        w = h * float(aspect_ratio)
        w = max(16, int(round(w / 16.0)) * 16)
        h = max(16, int(round(h / 16.0)) * 16)
        return (w, h, (w * h) / 1_000_000.0)


NODE_CLASS_MAPPINGS = {
    "MossDyPEPatch": MossDyPEPatch,
    "MossDyPESchedule": MossDyPESchedule,
    "MossDyPEResolution": MossDyPEResolution,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MossDyPEPatch": "MOSS DyPE-M (Position Extrapolation)",
    "MossDyPESchedule": "MOSS DyPE-M Schedule (Info)",
    "MossDyPEResolution": "MOSS DyPE-M Resolution (MP)",
}
