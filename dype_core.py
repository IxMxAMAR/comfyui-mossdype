"""DyPE-M: dynamic position extrapolation for MOSS-Image (and other 4-axis RoPE DiTs).

Vanilla DyPE (Issachar et al., arXiv 2510.20766) rescales the rotary positional
encoding (RoPE) of a diffusion transformer as a function of the *normalised
diffusion time* t, so that early (low-frequency) steps see NTK-extrapolated
frequencies and late (high-frequency) steps see the original training
frequencies:

    kappa(t) = lambda_s * t ** lambda_t,          (paper: lambda_s = 1, lambda_t = 2)

MOSS-Image is a 42-layer single-stream DiT with a 4-axis RoPE
(r, f, h, w) of 32 dimensions per axis (16 frequency bands each).  Its
positions are *centred*, and the h/w axes carry the image extent, which is
exactly the quantity that must be extrapolated beyond the training resolution.

This module reimplements the DyPE frequency construction for that layout and
adds four changes over the reference implementation:

1. ``fix_ramp_clamp`` - the reference clamps only the upper end of the YaRN
   correction range.  Once kappa is small enough that the lower end exceeds
   ``dim/2``, ``low > high`` and the ramp inverts, so the frequency vector
   degenerates to *full linear interpolation* instead of the training
   frequencies.  We clamp both ends, which makes the kappa -> 0 limit the
   well-defined "no extrapolation" state the paper intends.

2. ``schedule`` - kappa as a function of the sampler's actual progress rather
   than of sigma alone.  With a shifted flow schedule and few steps (the DMD /
   4-8 step regime) sigma stays above ~0.17 until the final step, so
   ``sigma ** 2`` never decays and the high-frequency restoration that DyPE
   relies on never happens.  ``hybrid`` (default) uses
   ``(sigma * (1 - p)) ** (lambda_t / 2)``.

3. ``mscale_mode = "kappa"`` - YaRN's attention-temperature correction is
   scaled by the *effective* interpolation amount ``1 + (s - 1) * kappa``
   instead of the raw scale, so it vanishes together with the extrapolation.

4. ``start_percent`` / ``end_percent`` and per-axis control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch

TWO_PI = 2.0 * math.pi

# "aniso" is the legacy name for "adaptive". It stays in the list so workflows saved with
# it still pass ComfyUI's widget validation -- the backend checks the combo BEFORE the node
# runs, so a runtime-only alias is not enough.
SCHEDULES = ("adaptive", "aniso", "spectral", "hybrid", "sigma", "progress", "cosine", "sqrt_sigma")
ADAPTIVE_ALIASES = ("adaptive", "aniso")
METHODS = ("yarn", "uniform", "ntk", "off")
MSCALE_MODES = ("kappa", "vanilla", "off")
AXES = ("both", "h", "w")


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class DyPEConfig:
    """Everything the frequency builder needs.  Immutable for a sampling run."""

    enabled: bool = True
    method: str = "yarn"
    schedule: str = "hybrid"
    base_resolution: int = 1024
    target_scale: float = 0.0          # >0 pins the extrapolation strength, overriding base_resolution
    isotropic_scale: bool = True       # derive one scale from area instead of per-axis
    lambda_s: float = 1.0
    lambda_t: float = 2.0
    mscale_mode: str = "kappa"
    start_percent: float = 0.0
    end_percent: float = 1.0
    axes: str = "both"
    fix_ramp_clamp: bool = True
    # spectral-adaptive scheduling
    spectral_weight: float = 0.5
    spectral_hi: float = 0.9
    # geometry, filled in from the model at patch time
    patch_size: int = 16
    theta: float = 10000.0
    head_dim: int = 128

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def n_bands(self) -> int:
        """Frequency pairs per axis (MOSS: 128 // 8 == 16)."""
        return self.head_dim // 8

    @property
    def axis_dim(self) -> int:
        """Rotary dimension per axis (MOSS: 128 // 4 == 32)."""
        return self.head_dim // 4

    @property
    def base_patches(self) -> float:
        return max(1.0, float(self.base_resolution) / float(self.patch_size))


# --------------------------------------------------------------------------- #
# kappa schedules
# --------------------------------------------------------------------------- #
def compute_kappa(cfg: DyPEConfig, sigma: float, step: int, total: int,
                  detail: float | None = None) -> float:
    """Normalised scaling strength in [0, ~1] for the current denoising step.

    ``sigma`` is the flow-matching noise level (1.0 = pure noise, 0.0 = clean),
    which is exactly the quantity MOSS receives as its timestep.

    ``detail`` is the *measured* fraction of the base-resolution frequency band
    that the emerging image already fills (0 = still pure low-frequency
    structure, 1 = detail fully resolved).  When supplied and
    ``cfg.schedule == "spectral"`` it replaces the hand-tuned time curve, which
    is the central idea of DyPE-M: the paper asserts a fixed low->high spectral
    progression, we measure it.
    """
    if not cfg.enabled or cfg.method == "off":
        return 0.0

    total = max(int(total), 1)
    p = min(max(float(step) / float(total), 0.0), 1.0)
    sig = min(max(float(sigma), 0.0), 1.0)

    if cfg.schedule == "sigma":
        base = sig
    elif cfg.schedule == "progress":
        base = 1.0 - p
    elif cfg.schedule == "cosine":
        base = 0.5 * (1.0 + math.cos(math.pi * p))
    elif cfg.schedule == "sqrt_sigma":
        base = math.sqrt(sig)
    else:  # hybrid
        base = math.sqrt(sig * (1.0 - p))

    if cfg.schedule == "spectral" and detail is not None:
        measured = 1.0 - min(max(float(detail), 0.0), 1.0)
        w = min(max(float(cfg.spectral_weight), 0.0), 1.0)
        base = w * measured + (1.0 - w) * base

    if p < cfg.start_percent - 1e-9 or p > cfg.end_percent + 1e-9:
        return 0.0

    k = cfg.lambda_s * (max(base, 0.0) ** cfg.lambda_t)
    return min(max(k, 0.0), 8.0)


def compute_kappa_pair(cfg: DyPEConfig, sigma: float, step: int, total: int,
                       details: tuple[float | None, float | None] = (None, None),
                       bws: tuple[float, float] | None = None
                       ) -> tuple[float, float]:
    """Per-axis kappa.

    For most schedules the two axes get the same value.  ``schedule="adaptive"``
    keeps the analytic curve's *magnitude* -- which is where the temporal
    information is, since the latent spectrum plateaus after 2-3 steps -- and
    takes only the h/w *ratio* from the measurement, which is the part that is
    stable over the whole trajectory.  (``"aniso"`` is accepted as an alias.)

    The direction is deliberate: an axis whose spectrum is wider is carrying
    more high-frequency content, and high frequencies are exactly what position
    extrapolation protects against aliasing, so it receives *more* scaling.
    """
    dh, dw = details
    if cfg.schedule not in ADAPTIVE_ALIASES:
        return (compute_kappa(cfg, sigma, step, total, dh),
                compute_kappa(cfg, sigma, step, total, dw))

    base = compute_kappa(cfg, sigma, step, total, None)          # hybrid curve
    w = min(max(float(cfg.spectral_weight), 0.0), 1.0)
    if w <= 0.0:
        return (base, base)

    if bws is not None and max(bws) > 1e-6:
        ref = max(0.5 * (bws[0] + bws[1]), 1e-6)
        ratios = [bws[0] / ref, bws[1] / ref]
    elif dh is not None and dw is not None:
        ref = max(0.5 * (dh + dw), 1e-6)
        ratios = [dh / ref, dw / ref]
    else:
        return (base, base)

    out = []
    for r in ratios:
        r = min(max(r, 1.0 / 3.0), 3.0)
        out.append(min(max(base * (r ** w), 0.0), 8.0))
    return (out[0], out[1])


# --------------------------------------------------------------------------- #
# spectral measurement (the DyPE-M contribution)
# --------------------------------------------------------------------------- #
def _axis_psd(x0: torch.Tensor, axis: int, noise_floor: bool = True):
    """Mean power spectrum of ``x0`` along ``axis``, DC removed.

    Returns ``(psd, n_bins)`` where ``psd`` is ``[B, n]`` normalised so the
    frequencies are ``i / n`` in units of Nyquist (0 = DC, 1 = Nyquist).
    """
    x = x0.float()
    if axis == 2:
        spec = torch.fft.rfft(x, dim=2, norm="ortho")
        spec = spec.abs().square().mean(dim=(1, 3))          # [B, H/2+1]
    else:
        spec = torch.fft.rfft(x, dim=3, norm="ortho")
        spec = spec.abs().square().mean(dim=(1, 2))          # [B, W/2+1]

    ac = spec[..., 1:]                                        # drop DC
    if noise_floor and ac.shape[-1] >= 8:
        nf = max(ac.shape[-1] // 4, 1)
        floor = ac[..., -nf:].median(dim=-1, keepdim=True).values
        ac = (ac - floor).clamp_min(0.0)
    return ac


def axis_bandwidth(x0: torch.Tensor, axis: int, hi: float = 0.9,
                   noise_floor: bool = True) -> torch.Tensor:
    """Normalised bandwidth of ``x0`` along ``axis``, averaged over B/C.

    Returns the fraction of the Nyquist range below which ``hi`` of the AC
    energy lies, per batch element: 0 = all energy at DC, 1 = white noise.

    At high sigma the denoised estimate still carries residual white noise whose
    flat spectrum inflates the measurement until it saturates.  ``noise_floor``
    subtracts the median of the top quarter of the band (an estimate of that
    floor) before accumulating, which is plain spectral subtraction.
    """
    ac = _axis_psd(x0, axis, noise_floor)
    if ac.shape[-1] <= 1:
        return torch.zeros(ac.shape[0], device=ac.device)

    total = ac.sum(-1, keepdim=True)
    if float(total.min()) <= 0.0:
        return torch.zeros(ac.shape[0], device=ac.device)
    cum = ac.cumsum(-1) / total
    idx = (cum >= hi).float().argmax(-1).float()              # first crossing
    return idx / max(ac.shape[-1] - 1, 1)


def spectral_stats(x0: torch.Tensor, axis: int, f_base: float = 0.2,
                   noise_floor: bool = True) -> dict:
    """Diagnostic summary of the spectrum along one axis.

    ``centroid`` and the bandwidths are fractions of Nyquist; ``oob`` is the
    share of AC energy above ``f_base`` (the model's training band).
    """
    ac = _axis_psd(x0, axis, noise_floor)
    n = ac.shape[-1]
    if n <= 1:
        return {"bw50": 0.0, "bw70": 0.0, "bw90": 0.0, "centroid": 0.0,
                "oob": 0.0, "ac": 0.0}
    freqs = torch.arange(1, n + 1, device=ac.device, dtype=ac.dtype) / n
    total = ac.sum(-1).clamp_min(1e-12)

    out = {}
    cum = ac.cumsum(-1) / total.unsqueeze(-1)
    for q in (0.5, 0.7, 0.9):
        idx = (cum >= q).float().argmax(-1).float()
        out[f"bw{int(q*100)}"] = float((idx / max(n - 1, 1)).mean().item())
    out["centroid"] = float(((ac * freqs).sum(-1) / total).mean().item())
    out["oob"] = float((ac * (freqs > f_base)).sum(-1).div(total).mean().item())
    out["ac"] = float(total.mean().item())
    return out


def spectral_bandwidths(x0: torch.Tensor, hi: float = 0.9) -> tuple[float, float]:
    """Raw (unclamped) h/w bandwidths as fractions of Nyquist."""
    bh = float(axis_bandwidth(x0, 2, hi).mean().item())
    bw = float(axis_bandwidth(x0, 3, hi).mean().item())
    return bh, bw


def spectral_detail(x0: torch.Tensor, scales: tuple[float, float],
                    hi: float = 0.9) -> tuple[float, float]:
    """Measured band-fill along h and w, expressed against the training band.

    ``scales`` are the per-axis extrapolation factors s_h, s_w.  A value of 0
    means every frequency present is still inside the band the model was trained
    on; 1 means content has filled out to the current Nyquist limit.
    """
    bh = axis_bandwidth(x0, 2, hi)
    bw = axis_bandwidth(x0, 3, hi)
    dh = float((bh * max(scales[0], 1.0)).clamp(0.0, 1.0).mean().item())
    dw = float((bw * max(scales[1], 1.0)).clamp(0.0, 1.0).mean().item())
    return dh, dw


# --------------------------------------------------------------------------- #
# YaRN helpers (ported from the DyPE reference, plus the clamp fix)
# --------------------------------------------------------------------------- #
def _correction_factor(num_rotations: float, dim: int, base: float,
                       max_position_embeddings: float) -> float:
    if num_rotations <= 1e-12:
        return float("inf")
    arg = max_position_embeddings / (num_rotations * TWO_PI)
    if arg <= 0.0:
        return float("inf")
    return (dim * math.log(arg)) / (2.0 * math.log(base))


def correction_range(low_ratio: float, high_ratio: float, dim: int, base: float,
                     max_position_embeddings: float, n_bands: int,
                     clamp_both: bool = True) -> tuple[int, int]:
    """Band index window over which the YaRN ramp transitions."""
    low = _correction_factor(low_ratio, dim, base, max_position_embeddings)
    high = _correction_factor(high_ratio, dim, base, max_position_embeddings)
    low_i = 0 if math.isinf(low) else int(math.floor(low))
    high_i = n_bands if math.isinf(high) else int(math.ceil(high))

    low_i = max(low_i, 0)
    high_i = min(high_i, n_bands)
    if clamp_both:
        # reference only clamps `high`; clamping `low` keeps low <= high and
        # makes the kappa -> 0 limit the training frequencies.
        low_i = min(low_i, n_bands)
    if high_i < low_i:
        low_i, high_i = high_i, low_i
    return low_i, high_i


def linear_ramp(low: int, high: int, n: int, dtype=torch.float64) -> torch.Tensor:
    if low == high:
        high = high + 0.001
    t = (torch.arange(n, dtype=dtype) - float(low)) / float(high - low)
    return t.clamp(0.0, 1.0)


def _base_bands(cfg: DyPEConfig) -> torch.Tensor:
    i = torch.arange(cfg.n_bands, dtype=torch.float64)
    return cfg.theta ** (-i / cfg.n_bands)


def yarn_bands(cfg: DyPEConfig, scale: float, kappa: float) -> torch.Tensor:
    """Per-axis frequency vector, YaRN-interpolated and DyPE-scheduled."""
    bands_base = _base_bands(cfg)
    if scale <= 1.0 + 1e-9 or kappa <= 0.0:
        return bands_base

    s = float(scale)
    dim = cfg.axis_dim
    i = torch.arange(cfg.n_bands, dtype=torch.float64) / cfg.n_bands

    bands_linear = bands_base / s
    new_base = cfg.theta * (s ** (dim / (dim - 2)))
    bands_ntk = new_base ** (-i)

    # ramp 1: linear <-> NTK, window set by beta
    lo, hi = correction_range(1.25 * kappa, 0.75 * kappa, dim, cfg.theta,
                              cfg.base_patches, cfg.n_bands, cfg.fix_ramp_clamp)
    m = 1.0 - linear_ramp(lo, hi, cfg.n_bands)
    bands = bands_linear * (1.0 - m) + bands_ntk * m

    # ramp 2: blended <-> training bands, window set by gamma
    lo, hi = correction_range(16.0 * kappa, 2.0 * kappa, dim, cfg.theta,
                              cfg.base_patches, cfg.n_bands, cfg.fix_ramp_clamp)
    m = 1.0 - linear_ramp(lo, hi, cfg.n_bands)
    bands = bands * (1.0 - m) + bands_base * m
    return bands


def uniform_bands(cfg: DyPEConfig, scale: float, kappa: float) -> torch.Tensor:
    """Every band scaled by the *same* factor, scheduled by kappa.

    YaRN mixes frequency groups: at kappa=1 the top four bands keep their raw
    frequencies while everything below is compressed by ``1/s``.  The encoding is
    then internally inconsistent -- different band groups imply different canvas
    sizes -- which is what inflates large objects (PAPER.md 5.4).  Scaling all
    bands by one factor keeps the implied canvas size consistent at every kappa,
    so kappa -> 1 is plain position interpolation and kappa -> 0 is the training
    frequencies.
    """
    bands_base = _base_bands(cfg)
    if scale <= 1.0 + 1e-9 or kappa <= 0.0:
        return bands_base
    k = min(max(float(kappa), 0.0), 1.0)
    factor = 1.0 + k * (1.0 / float(scale) - 1.0)          # 1 -> 1/s as k -> 1
    return bands_base * factor


def ntk_bands(cfg: DyPEConfig, scale: float, kappa: float) -> torch.Tensor:
    """DY-NTK variant: blend the NTK base between 1 (off) and s^(d/(d-2))."""
    bands_base = _base_bands(cfg)
    if scale <= 1.0 + 1e-9 or kappa <= 0.0:
        return bands_base
    dim = cfg.axis_dim
    factor = max(1.0, float(scale) ** (dim / (dim - 2)))
    factor = factor ** kappa
    i = torch.arange(cfg.n_bands, dtype=torch.float64) / cfg.n_bands
    return (cfg.theta * factor) ** (-i)


def mscale_for(cfg: DyPEConfig, scale: float, kappa: float) -> float:
    """YaRN attention-temperature factor, optionally tied to the effective scale."""
    if cfg.mscale_mode == "off" or scale <= 1.0 + 1e-9:
        return 1.0
    if cfg.mscale_mode == "vanilla":
        s_eff = scale
    else:
        s_eff = 1.0 + (scale - 1.0) * kappa
    if s_eff <= 1.0:
        return 1.0
    return 0.1 * math.log(s_eff) + 1.0


# --------------------------------------------------------------------------- #
# MOSS-Image position encoding
# --------------------------------------------------------------------------- #
def _positions(h: int, w: int, text_len: int) -> torch.Tensor:
    """(r, f, h, w) coordinates in the layout MOSS-Image was trained on."""
    n = h * w
    ids = torch.zeros(n + text_len, 4, dtype=torch.float64)
    ids[:n, 1] = float(text_len)
    ids[:n, 2] = (torch.arange(h, dtype=torch.float64) - (h - 1) / 2.0).repeat_interleave(w)
    ids[:n, 3] = (torch.arange(w, dtype=torch.float64) - (w - 1) / 2.0).repeat(h)
    ids[n:, 1] = torch.arange(text_len, dtype=torch.float64)
    return ids


def effective_scales(h: int, w: int, cfg: DyPEConfig) -> tuple[float, float]:
    """The (h, w) frequency-scale factors this config actually uses.

    Single source of truth, so the trace and the encoder can never disagree --
    the trace used to recompute per-axis values and would have reported the wrong
    numbers while isotropic scaling was in force.
    """
    if getattr(cfg, "target_scale", 0.0) and cfg.target_scale > 0.0:
        s = max(float(cfg.target_scale), 1.0)
        return (s, s)
    if getattr(cfg, "isotropic_scale", True):
        # One scale from the AREA keeps the aspect ratio.  Per-axis scaling divides
        # by base_patches, so on a non-square canvas the axes disagree by exactly the
        # aspect ratio: 25% horizontal stretch at 3200x2560, 78% at 4096x2304.
        s = max(math.sqrt((float(h) * float(w)) / (cfg.base_patches ** 2)), 1.0)
        return (s, s)
    return (max(float(h) / cfg.base_patches, 1.0),
            max(float(w) / cfg.base_patches, 1.0))


def build_pe(h: int, w: int, text_len: int, device, cfg: DyPEConfig,
             kappa_h: float, kappa_w: float | None = None) -> torch.Tensor:
    """Return the RoPE tensor in the exact layout ``MOSSImage.rope`` produces.

    Shape: ``[1, h*w + text_len, 1, n_bands * 4, 2, 2]``, each 2x2 block being
    ``[[cos, -sin], [sin, cos]]`` for one (axis, band) pair.

    ``kappa_h`` / ``kappa_w`` are the per-axis scaling strengths, which lets the
    spectral scheduler treat the two image axes independently.
    """
    if kappa_w is None:
        kappa_w = kappa_h
    ids = _positions(h, w, text_len)
    nb = cfg.n_bands

    scale_h = max(float(h) / cfg.base_patches, 1.0)
    scale_w = max(float(w) / cfg.base_patches, 1.0)

    # Absolute reference: base_resolution is a fixed pixel count, so the effective
    # extrapolation strength grows with the output size -- a base that looks right at
    # 6.5 MP overshoots at 8 MP.  target_scale pins the strength instead: scale is
    # constant for every resolution, so one setting transfers across output sizes.
    # isotropic_scale derives one scale from the area, which also keeps the aspect
    # ratio (see effective_scales).
    scale_h, scale_w = effective_scales(h, w, cfg)

    bands_base = _base_bands(cfg)
    builder = {"ntk": ntk_bands, "uniform": uniform_bands}.get(cfg.method, yarn_bands)

    if cfg.axes in ("both", "h") and scale_h > 1.0 and kappa_h > 0.0:
        bands_h = builder(cfg, scale_h, kappa_h)
        ms_h = mscale_for(cfg, scale_h, kappa_h)
    else:
        bands_h, ms_h = bands_base, 1.0

    if cfg.axes in ("both", "w") and scale_w > 1.0 and kappa_w > 0.0:
        bands_w = builder(cfg, scale_w, kappa_w)
        ms_w = mscale_for(cfg, scale_w, kappa_w)
    else:
        bands_w, ms_w = bands_base, 1.0

    per_axis = ((bands_base, 1.0), (bands_base, 1.0), (bands_h, ms_h), (bands_w, ms_w))

    blocks = []
    for axis, (bands, ms) in enumerate(per_axis):
        phase = ids[:, axis].unsqueeze(-1) * bands.unsqueeze(0)  # [N, nb]
        cos = phase.cos() * ms
        sin = phase.sin() * ms
        mats = torch.stack([cos, -sin, sin, cos], dim=-1)        # [N, nb, 4]
        blocks.append(mats.reshape(-1, nb, 2, 2))

    pe = torch.cat(blocks, dim=1)                                # [N, 4*nb, 2, 2]
    pe = pe.unsqueeze(0).unsqueeze(1).transpose(1, 2)            # [1, N, 1, 4*nb, 2, 2]
    return pe.to(device=device, dtype=torch.float32).contiguous()


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def sigma_schedule(steps: int, shift: float = 5.0, sigma_max: float = 1.0) -> list[float]:
    """Flow-matching sigmas as ComfyUI's ModelSamplingFlux produces them."""
    out = []
    for k in range(steps):
        t = 1.0 - k / steps
        out.append(sigma_max * (shift * t) / (1.0 + (shift - 1.0) * t))
    return out


def describe_schedule(cfg: DyPEConfig, steps: int, shift: float = 5.0) -> str:
    """Human-readable table of kappa / band behaviour per step."""
    sigmas = sigma_schedule(steps, shift)
    lines = [
        f"DyPE-M schedule  method={cfg.method} schedule={cfg.schedule} "
        f"lambda_s={cfg.lambda_s:g} lambda_t={cfg.lambda_t:g}",
        f"base_resolution={cfg.base_resolution} base_patches={cfg.base_patches:g} "
        f"mscale={cfg.mscale_mode} clamp_fix={cfg.fix_ramp_clamp}",
        "",
        f"{'step':>4} {'sigma':>8} {'kappa':>8} {'b[0]':>9} {'b[8]':>9} {'b[15]':>9}",
    ]
    for k in range(steps):
        sig = sigmas[k]
        kap = compute_kappa(cfg, sig, k, steps)
        b = yarn_bands(cfg, 2.0, kap)
        lines.append(
            f"{k:>4} {sig:>8.4f} {kap:>8.4f} "
            f"{b[0].item():>9.6f} {b[8].item():>9.6f} {b[15].item():>9.6f}"
        )
    lines.append("")
    lines.append("b[*] = frequency of band * at scale 2.0 (base band 8 = "
                 f"{_base_bands(cfg)[8].item():.6f})")
    return "\n".join(lines)
