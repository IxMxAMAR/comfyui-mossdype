# MOSS-Image DyPE-M

Dynamic position extrapolation for [MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image), so
it can render well past the ~4 MP it was trained on — no retraining, no extra sampling time
(+0.5% wall clock, measured).

I ported DyPE ([arXiv 2510.20766](https://arxiv.org/abs/2510.20766)) to MOSS-Image's 4-axis
rotary encoding. The port itself was the easy part. Getting it to produce *better* images
rather than sharper-looking worse ones took much longer, and came down to two bugs in how the
scale factor is computed. Both are written up below, because anyone porting this to another
model will hit the same two.

## Install

```
comfy node install comfyui-mossdype
```

Or clone this repo into `ComfyUI/custom_nodes/` and restart. No extra dependencies — it only
uses what ComfyUI already ships.

## Nodes

All three live under `model/patch/moss_image`.

- **MOSS-Image DyPE-M Patch** — MODEL in, MODEL out. This is the one you want.
- **MOSS-Image DyPE-M Schedule** — prints the κ table for a given configuration without
  rendering anything. Useful for seeing what a setting actually does over the step schedule.
- **MOSS-Image DyPE-M Resolution** — megapixels + aspect ratio to width/height, rounded to
  multiples of 16.

The patch is a true no-op at or below the base resolution, and `enabled = false` is
byte-identical to not having the node at all, so it's safe to leave wired in.

## Settings

| setting | default | what it does |
|---|---|---|
| `enabled` | `true` | Bypass without rewiring. |
| `target_scale` | `0.0` | Pins the extrapolation strength. `0` falls back to `base_resolution`. **This is the main dial.** |
| `isotropic_scale` | `true` | Derives one scale from the pixel area and applies it to both axes. Leave it on. |
| `base_resolution` | `2048` | The resolution the model "believes" it is rendering at. Only used when `target_scale = 0`. |
| `method` | `yarn` | `yarn` is the paper's branch. `uniform` scales all frequency bands alike (milder). `ntk` is the smoothest. |
| `schedule` | `adaptive` | How κ decays over the run. `adaptive` measures the h/w ratio per step; `hybrid` is the pure analytic curve; `sigma` is the paper's own κ = σ². |
| `lambda_s` | `1.0` | Overall strength multiplier. |
| `lambda_t` | `2.0` | Decay exponent. |
| `start_percent` | `0.0` | `0.125` skips step 0, which is the step that commits the layout. |
| `end_percent` | `1.0` | Cut the patch off early. |
| `axes` | `both` | Apply to `h`, `w`, or both. |
| `fix_ramp_clamp` | `true` | See below. Leave on. |
| `mscale_mode` | `kappa` | `kappa` ties the attention temperature to the effective scale. `vanilla` is the reference behaviour. |
| `spectral_weight` | `0.5` | How much the measured spectrum is trusted vs the analytic curve (adaptive schedule only). |
| `spectral_hi` | `0.9` | Cumulative energy fraction that defines the measured bandwidth. |

Lower `target_scale` = milder extrapolation = less object inflation. **1.2–1.3 is the sweet
spot** in my testing; 2.5 and up inflates large objects noticeably and starts inventing detail.

## The two bugs

### 1. Scaling the axes separately distorts the image

The scale factor gets derived per axis: `s_h = h/base`, `s_w = w/base`. On a square canvas
those are equal. On any other canvas they are not, and the ratio is exactly the aspect ratio:

```
s_w / s_h = w / h
```

So the two axes disagree about how big the canvas is. At 3200×2560 that's a 25% horizontal
stretch; at 4096×2304 it's 78%. It shows up as elongated limbs, squashed faces, warped arches
— and it's subtle enough that you might blame the prompt or the sampler.

Measured with a sphere at 8.19 MP (3200×2560, aspect 1.25). A correct render is a circle:

| setting | sphere axis ratio |
|---|---|
| per-axis scaling, base 2048 | **1.200** |
| isotropic, base 1024 | 1.091 |
| isotropic, `target_scale` 1.25 | **1.015** |

The fix is `isotropic_scale`: derive one scale from the pixel *area*, `s = √(w·h)/base`, and
apply it to both axes. On square canvases this reduces to the old value, so square output is
unchanged.

### 2. Strength grows with output size

`base_resolution` is an absolute pixel count, so `s = extent/base` climbs with resolution:
1.25× at 2560², 1.5× at 9.4 MP, 3.2× at 8 MP if you use a 1024 base. A base that looks right
at one size overshoots at the next, which is why "it was fine yesterday" happens.

`target_scale` pins the strength directly, so one value transfers across output sizes.

### Also: three fixes to the reference formulation

- **Ramp inversion.** The reference clamps only the top of the YaRN correction window. Below
  κ ≈ 8e-4 the window inverts and the frequency vector degenerates into *full linear
  interpolation* — the opposite of what DyPE is supposed to do at late steps. `fix_ramp_clamp`
  clamps both ends, which makes the κ → 0 limit exactly the training frequencies.
- **Few-step under-decay.** With a shifted flow schedule and 4–8 steps, σ² never decays
  (κ = 0.17 at the final step), so the high-frequency restoration never happens. The
  progress-aware schedule fixes it.
- **Inconsistent attention temperature.** YaRN's `mscale` was applied even after the
  frequencies had returned to base. Tying it to the effective scale removes it as the
  extrapolation vanishes.

## What it does not fix

This shifts where the model puts its frequencies. It does not add information the model
doesn't have. Push `target_scale` too high and you get invented texture — halos, plastic skin,
crunchy grain — plus corrupted text. At 2.5× I had a marquee that should read `MOSS` render as
`MOUSS`. If an image looks over-sharpened rather than detailed, you're past the useful range;
come back down.

It's also not a substitute for a bigger model. On a single dominant subject at 8 MP+ you'll get
better proportions from `target_scale` around 1.1–1.2 than from any scheduling trick.

## Example workflow

`example_workflows/mossdype_ultrares.json` — 2560², 8 steps, DMD LoRA, with the patch wired in
and a note on the canvas explaining the settings. Load it from the workflow browser.

## Credit

DyPE is by Issachar, Yariv et al. — [arXiv 2510.20766](https://arxiv.org/abs/2510.20766). The
YaRN ramps, κ schedule and `mscale` are theirs. Everything about the 4-axis MOSS port, the two
geometry bugs and the `adaptive` schedule is from my own testing.

## License

MIT — see [LICENSE](LICENSE).
