# PPISP vs. the Appearance-Correction Lineage — Evaluation

**Status:** Notes / evaluation
**Last updated:** 2026-06-23
**Goal:** Evaluate **PPISP** (NVIDIA, 2026) against the prior approaches to photometric variation in
radiance-field reconstruction it sets itself against — per-image **appearance embeddings**
(NeRF-in-the-Wild), **bilateral grids** (BilaRF), and physically-based **calibration** (ADOP) — and
note its relevance to Brush.

> **Scope note.** PPISP is a *photometric/appearance* method, **not** geometry. It is unrelated to the
> PlanarGS port or splat registration — it can't reconstruct or align anything. Where "the original"
> appears below it means the prior **appearance-correction** approach PPISP improves on, not PlanarGS.
>
> **Provenance.** Claims/numbers are as reported in the PPISP paper (arXiv 2601.18336), its
> [project page](https://research.nvidia.com/labs/sil/projects/ppisp/), and the
> [repo README](https://github.com/nv-tlabs/ppisp), read 2026-06-23 (the paper text was read via the
> arXiv HTML; verify exact figures against the PDF). Baseline facts are from the cited papers (§10).

---

## TL;DR

- **The problem:** cameras' auto-exposure / auto-white-balance / ISP tone-curve / lens vignetting make
  the *same surface look different across views*, breaking the multi-view photometric-consistency that
  every radiance field assumes → floaters, blur, color casts.
- **The original approach** absorbs this **per training frame** (latent appearance embeddings, or a
  bilateral grid per image). It fits training views well but has **no parameters for a novel view**, so
  it generalizes poorly — and benchmarks often cheat by fitting the correction against the test image.
- **PPISP's two ideas:** (1) a **physically-based, interpretable** decomposition into 4 ISP effects,
  factored **per-camera** (vignetting, response curve) vs **per-frame** (exposure, white balance); and
  (2) a **controller** that *predicts* the per-frame parameters from the rendered image — a learned
  auto-exposure/AWB — so novel views get ISP parameters **without seeing their ground truth**.
- **Result:** state-of-the-art novel-view PSNR; the striking evidence is that the per-frame baselines
  *reduce* novel-view quality below doing nothing, because they overfit training frames.
- **For Brush:** a small, **Apache-2.0**, reconstruction-agnostic plug-in — far lighter to adopt than
  the PlanarGS port, and it targets a real Brush pain point (exposure/WB drift across input photos).

---

## 1. The shared problem

Cameras don't store raw radiance. Per shot, the ISP applies exposure, white balance, a non-linear tone
curve (the camera response function), and the lens adds vignetting. So one physical surface is recorded
with different brightness/color in different views. That violates the **photometric-consistency**
assumption underlying NeRF/3DGS, and the optimizer "explains" the inconsistency with floaters, blur, and
baked-in casts. Every method here exists to **absorb these nuisance transforms** so the recovered
radiance/geometry stays clean.

## 2. The "original" approach — per-frame / black-box correction

| Method | How it absorbs appearance | The flaw PPISP targets |
|---|---|---|
| **NeRF-in-the-Wild** (Martin-Brualla et al., CVPR 2021) — *the seminal original* | a per-image **latent appearance embedding** (GLO code) fed to the radiance MLP | entangled, not physical; a novel view has **no embedding** — you interpolate or fit one against the test image |
| **Bilateral grids / BilaRF** (Wang et al., SIGGRAPH 2024) — *recent SOTA baseline* | a **per-frame learned bilateral grid** = a local 3D affine ISP | the grid is *per training frame*; none exists for a novel view, so it **overfits** training views |
| **ADOP** (Rückert et al., SIGGRAPH 2022) | explicit **physically-based calibration** (exposure, WB, CRF, vignetting) | physically grounded, but static per-image calibration with **no novel-view predictor** |

The common thread: correction is **per training frame**, with no principled way to produce parameters
for an unseen viewpoint — and novel-view "evaluation" is often done by fitting the per-frame correction
against the ground-truth test image, i.e. with **privileged information**.

## 3. The "new" approach — PPISP

**Idea 1 — physically-based decomposition, factored by where the effect lives:**

| ISP effect | Model | Scope |
|---|---|---|
| **Exposure** | `I = L · 2^Δt` (Δt in stops) | **per-frame** (capture-dependent) |
| **Vignetting** | radial polynomial `v(r; α)`, optical center μ | **per-camera** (lens-intrinsic) |
| **Color / white balance** | homography on chromaticities + intensity norm | **per-frame** |
| **Camera response (CRF)** | S-shaped piecewise power curve + gamma | **per-camera** (sensor-intrinsic) |

The first three act linearly on scene radiance; the CRF is the final non-linearity. The per-camera vs
per-frame split mirrors reality — vignetting and the response curve are fixed by the sensor/lens, while
exposure and white balance drift shot to shot.

**Idea 2 — the controller.** A small network predicts the per-frame parameters from the *rendered*
radiance, `(Δt, color) = T(L)` (a coarse feature extractor → 5×5 grid → MLP with separate heads). This
is a learned **auto-exposure / auto-white-balance**: a novel view gets ISP parameters **without ever
seeing its ground truth**, which is what makes novel-view evaluation honest.

Implemented as a **differentiable CUDA kernel**, **reconstruction-agnostic** (drops onto 3DGS / 3DGUT /
NeRF); at inference, `frame_idx = -1` invokes the controller instead of stored per-frame parameters.

## 4. Compare & contrast

| Dimension | Original (embeddings / bilateral grids) | PPISP |
|---|---|---|
| **Philosophy** | black-box, data-driven absorption | physically-based, interpretable transforms |
| **Factorization** | per-frame only | **per-camera vs per-frame** (matches the optics) |
| **Novel-view params** | none → interpolate or fit to GT | **predicted by a controller** (auto-exp/AWB analogue) |
| **Evaluation honesty** | often aligns to the test GT (leakage) | **no GT needed** for novel-view params |
| **Control / metadata** | not interpretable; can't inject EXIF | read exposure in stops, set WB, **plug in EXIF** |
| **Training-view fit** | very strong (overfits) | sometimes **slightly worse** |
| **Novel-view fit** | weak / can degrade | **state of the art** |
| **Spatially-local effects** | bilateral grid captures some local tone-mapping | **global** per-frame only (a limitation) |
| **License** | varies | **Apache-2.0** |

## 5. Evidence

The headline result: on **Tanks & Temples novel-view PSNR**, the per-frame correctors *hurt*, while PPISP
helps (3DGUT backbone, as reported):

| Setup | Novel-view PSNR ↑ |
|---|---|
| 3DGUT, no correction | 22.86 |
| + BilaRF (bilateral grid) | 19.78 ⬇ |
| + ADOP (calibration) | 20.28 ⬇ |
| **+ PPISP (with controller)** | **24.62** ⬆ |

The original methods overfit training frames so hard they *lose* ~3 dB on novel views versus doing
nothing; PPISP gains ~1.8 dB. (Mip-NeRF 360: 27.74 → 28.15 PSNR; Tanks & Temples SSIM 0.790 → 0.809.)

**Datasets:** Mip-NeRF 360, Tanks & Temples, the BilaRF dataset, HDR-NeRF, nine static Waymo Open
sequences, plus a new 3-camera capture (iPhone 13 Pro / Nikon Z7 / OM System OM-1 Mark II).

## 6. Net assessment

PPISP is an **interpretable, generalizable replacement** for black-box appearance embeddings and
per-frame bilateral grids. Its real contributions are conceptual, not just a metric bump: (1) the
**per-camera / per-frame physical factorization**, and (2) the **controller that makes novel-view
evaluation honest** — removing the test-time GT-alignment crutch the subfield had leaned on.

**Where the original still wins / PPISP's limits (stated):**
- Raw **training-view** reconstruction — overfitting helps the baselines there.
- **Spatially-varying** effects — bilateral grids capture *local* tone-mapping / flares that PPISP's
  global per-frame model ignores. PPISP also ignores lens flares and similar spatially-adaptive effects.
- The **controller needs meaningful correlations** in the rendered radiance to infer parameters; when
  absent, it must fall back on metadata.

## 7. Relevance to Brush

Unlike the PlanarGS port, PPISP is a **small, self-contained, reconstruction-agnostic** module: an ISP
CUDA kernel + a tiny controller MLP that operate on rendered RGB **before the loss**. Implications:

- **Cheap to adopt.** It plugs into a training loop at exactly the point Brush already has (render →
  compare to GT). No rasterizer surgery, no foundation-model preprocessing.
- **Clean license.** **Apache-2.0** — no Inria non-commercial gate to clear (contrast the PlanarGS
  rasterizer math). Safe to draw from or port under Brush's Apache-2.0.
- **Real Brush pain point.** Brush ingests COLMAP / Nerfstudio photo sets that routinely carry
  exposure/white-balance drift; a physically-based per-frame correction + a novel-view controller would
  improve robustness on casual captures.
- **Caveat for a Rust port.** The reference is a PyTorch CUDA extension; reproducing it in Brush means
  re-deriving the four ISP transforms + controller in WGSL/`burn` (the math is small and interpretable,
  which helps), and the controller is a tiny CNN+MLP — modest compared to PlanarGS's geometry kernels.

This makes PPISP a plausible **lighter-weight "improve Brush" candidate** than the planar-reconstruction
port — different goal (appearance robustness, not surface meshes), much smaller surface area.

## 8. Glossary

- **ISP (Image Signal Processor)** — the on-camera pipeline that turns raw sensor data into a viewable
  image (exposure, white balance, tone curve, etc.). The source of the photometric variation here.
- **Photometric consistency** — the assumption that a surface point has the same observed color across
  views; broken by per-shot ISP, causing floaters/blur in radiance fields.
- **Radiance field** — a 3D scene representation optimized from posed images (NeRF, 3D Gaussian
  Splatting). PPISP is a plug-in for the training of these.
- **3DGS / 3DGUT** — 3D Gaussian Splatting (Kerbl et al. 2023) and NVIDIA's 3D Gaussian Unscented
  Transform variant; the backbones PPISP was tested on. *(Brush is a 3DGS engine.)*
- **Reconstruction-agnostic** — works regardless of the underlying radiance-field method (it only sees
  rendered RGB).
- **Appearance embedding (GLO code)** — a per-image latent vector that lets a NeRF absorb appearance
  variation; introduced by NeRF-in-the-Wild. The "original" black-box approach.
- **Bilateral grid** — a small 3D lookup that applies a locally-varying affine color/tone transform;
  BilaRF learns one per training frame.
- **CRF (Camera Response Function)** — the non-linear map from scene radiance to pixel value (the tone
  curve); per-camera in PPISP.
- **Vignetting** — radial brightness falloff toward image edges from lens optics; per-camera.
- **White balance / AWB** — correction of the illuminant's color cast; drifts per frame, modeled by
  PPISP's chromaticity homography and predicted by the controller.
- **Exposure (stops)** — brightness scaling; one "stop" = a factor of 2 (`2^Δt`), per frame.
- **Controller** — PPISP's network that predicts per-frame exposure/color from the rendered image,
  giving novel views ISP parameters without ground truth (a learned auto-exposure/AWB).
- **Training views vs novel views** — images used during optimization vs held-out viewpoints; the gap
  between them is exactly where the per-frame baselines fail and PPISP's controller helps.
- **Privileged information / test-time leakage** — using the test image's ground truth to tune the
  correction for that view; inflates novel-view metrics. PPISP's controller removes the need for it.

## 9. Sources & references

### PPISP (the "new" paper)
- **PPISP: Physically-Plausible Compensation and Control of Photometric Variations in Radiance Field
  Reconstruction** — Deutsch, Moënne-Loccoz, State, Gojcic (NVIDIA), CVPR 2026 (oral).
  [arXiv 2601.18336](https://arxiv.org/abs/2601.18336) ·
  [project page](https://research.nvidia.com/labs/sil/projects/ppisp/) ·
  [code (nv-tlabs/ppisp)](https://github.com/nv-tlabs/ppisp) (Apache-2.0).

### The "original" appearance-correction lineage (baselines)
- **NeRF in the Wild** (per-image appearance embeddings) — Martin-Brualla et al., CVPR 2021 —
  [arXiv 2008.02268](https://arxiv.org/abs/2008.02268) · [project page](https://nerf-w.github.io/).
- **Bilateral Guided Radiance Field Processing** (BilaRF) — Wang, Wang, Gong, Xue, ACM TOG 43(4) /
  SIGGRAPH 2024 — [arXiv 2406.00448](https://arxiv.org/abs/2406.00448) ·
  [project page](https://bilarfpro.github.io/).
- **ADOP: Approximate Differentiable One-Pixel Point Rendering** (physically-based camera calibration) —
  Rückert, Franke, Stamminger, SIGGRAPH 2022 — [arXiv 2110.06635](https://arxiv.org/abs/2110.06635).

### Backbones & datasets referenced
- **3D Gaussian Splatting** — Kerbl et al., SIGGRAPH 2023 —
  https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ · **3DGUT / 3dgrut** (NVIDIA) —
  https://github.com/nv-tlabs/3dgrut · **gsplat** — https://github.com/nerfstudio-project/gsplat.
- Datasets: **Mip-NeRF 360** (https://jonbarron.info/mipnerf360/), **Tanks and Temples**
  (https://www.tanksandtemples.org/), **HDR-NeRF** (https://xhuangcv.github.io/hdr-nerf/),
  **Waymo Open** (https://waymo.com/open/).

### Brush
- **Brush** — https://github.com/ArthurBrussee/brush (Apache-2.0; this fork's port docs live in
  `working_docs/` and `docs/`).
