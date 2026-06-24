# VGGT-Ω → Brush — Integration Plan & PR Chain

**Status:** Plan / scoping (no code written yet)
**Last updated:** 2026-06-23
**Goal:** Turn the "use VGGT-Omega as Brush's pose/point-cloud front-end" recommendation from the
[comparison doc](vggt-omega-vs-brush.md) into a **concrete, dependency-ordered PR chain** an engineer
could pick up — with teaching diagrams for the two ideas that make it tractable.

> Companion to [`vggt-omega-vs-brush.md`](vggt-omega-vs-brush.md) (the *why*; this is the *how*). Same
> citation discipline: Brush `file:line` from this checkout, read 2026-06-23 (lines drift); VGGT-Omega
> from the local `../../../vggt-omega` clone. Effort numbers are for **one strong engineer** comfortable
> with Rust, 3D geometry, and 3DGS; ranges are wide on purpose.

> 📖 Most vocabulary (SfM, 3DGS, pose encoding, extrinsics…) is defined in the
> [comparison doc's glossary](vggt-omega-vs-brush.md#10-glossary). This doc's [§6 Glossary](#6-glossary)
> adds only the *implementation* terms (format loader, golden test, ingestion seam…).

---

## 1. Why this is tractable — the two paradigms

The integration is clean for one structural reason: **Brush already leaves the pose slot open.** It
*requires* externally-computed cameras and never refines them, so dropping in a different producer of
cameras is additive, not invasive. The two systems are different *kinds* of thing — a feed-forward
front-end and a per-scene-optimization back-end — and that's exactly why they compose:

![Feed-forward (VGGT-Omega): images through one frozen ~1B network in a single pass → cameras + dense depth. Per-scene optimization (Brush): posed images through a render→loss→backprop→update loop repeated ~30K times → a photoreal splat. Chaining VGGT-Omega's cameras into Brush gets both speed of geometry and photorealism.](paradigm-feedforward-vs-optimization.svg)

*Two paradigms. VGGT-Omega pays its cost once in pretraining (per-scene = one GPU pass, seconds);
Brush pays per scene (~30K Adam steps, minutes). Neither replaces the other — chaining them does
COLMAP-free capture-to-splat. Source: [`paradigm-feedforward-vs-optimization.svg`](paradigm-feedforward-vs-optimization.svg).*

---

## 2. The data mapping — what converts to what

VGGT-Omega's outputs line up almost 1:1 with Brush's inputs; the only real subtlety is **coordinate
convention**, and routing through COLMAP format side-steps even that (Brush's COLMAP reader already
encodes the right inversion).

![VGGT-Omega predicts a 9-D pose encoding (decoded to extrinsics, camera-from-world OpenCV, + pinhole intrinsics), per-pixel depth+confidence, and a derived dense point cloud. Brush needs a Camera (position, rotation, fov) and a Point3D list (xyz, rgb→SH band-0). The conversion is a decode and an unprojection; COLMAP format is the lossless bridge both already speak.](data-mapping.svg)

*The conversion surface. The one place to be careful is the world-to-camera → camera-to-world
inversion (`crates/brush-dataset/src/formats/colmap.rs:196-201`); everything else is a decode + an
unprojection. Source: [`data-mapping.svg`](data-mapping.svg).*

---

## 3. The PR chain

**Strategy: validate cheaply before writing Rust, then integrate natively, then enhance.** PR1 proves
the whole chain works end-to-end with *zero* Brush code (just COLMAP export). Only once VGGT-Omega's
pose quality is confirmed on real scenes do we invest in a native loader (PR3) and better init (PR4).
PR5–PR6 are orthogonal Brush features that integration makes more valuable but that can ship anytime.

![A dependency graph: PR0 docs informs PR1 (VGGT→COLMAP export script, Option A). PR1 and the optional PR2 ingestion-seam refactor feed PR3 (native vggt.rs loader, Option B). PR3 feeds PR4 (dense-depth init). Stretch items PR5 (pose refinement, motivated by PR3) and PR6 (depth-supervised loss, depends on PR4) branch off. Critical path is PR1→PR3→PR4.](pr-chain.svg)

*The chain at a glance. Critical path PR1 → PR3 → PR4; PR2 optional; PR5–PR6 orthogonal stretch.
Source: [`pr-chain.svg`](pr-chain.svg).*

| PR | Title | Track | Depends on | Brush code | Est. |
|---|---|---|---|---|---|
| **PR1** | VGGT-Omega → COLMAP export script | validate (Option A) | — | **none** | 2–4 days |
| **PR2** | Ingestion seam refactor | core (optional) | — | refactor only | 2–3 days |
| **PR3** | Native `vggt.rs` format loader | core (Option B) | PR1, PR2 | small, additive | 1–1.5 weeks |
| **PR4** | Dense-depth point-cloud init | core | PR3 | small | 3–5 days |
| **PR5** | Optional camera pose refinement | stretch ‡ | (PR3 motivates) | medium | 1.5–2.5 weeks |
| **PR6** | Depth-supervised loss | stretch | PR4 | medium | ~1 week |

**Critical path (PR1 → PR3 → PR4): ~2.5–3.5 weeks.** PR2 adds ~half a week if taken; PR5/PR6 are
separate initiatives.

> **‡ Reprioritize for on-device capture.** PR5 (pose refinement) is *stretch* for offline/COLMAP poses,
> but the **top priority** for **on-device (ARKit/LiDAR) capture** like SplatKing, whose visual-inertial
> poses drift — see [comparison §4.3](vggt-omega-vs-brush.md#43-what-this-changes).

---

## 4. Per-PR detail

### PR1 — VGGT-Omega → COLMAP export (the validation harness) · *Option A*
- **Goal:** run the full chain `images → VGGT-Omega → Brush → splat` with no Brush changes, and judge
  whether VGGT-Omega poses are good enough.
- **Scope:** a standalone Python script that runs `VGGTOmega`, decodes `pose_enc` → extrinsics/intrinsics
  (`encoding_to_camera`, `vggt_omega/utils/pose_enc.py:29-52`), unprojects depth → colored points
  (`demo_gradio.py:75-104`), and writes a COLMAP `sparse/` model (`cameras`, `images`, `points3D`) via
  `pycolmap` (the `export` extra, `pyproject.toml`). Subsample the dense cloud to a COLMAP-sparse-like size.
- **Files:** new `tools/vggt_to_colmap.py` (outside the Rust workspace; or in the vggt-omega clone). No
  Brush Rust.
- **Verify:** point Brush at the exported folder (Brush auto-detects COLMAP,
  `crates/brush-dataset/src/formats/mod.rs:56-72`); it trains to completion. **Compare** final PSNR /
  visual quality against a real COLMAP run on the same images — this is the go/no-go signal for PR3.
- **Why first:** cheapest possible test of the core hypothesis (VGGT poses → good splats).

### PR2 — Ingestion seam refactor (optional) · *core*
- **Goal:** make adding a new pose source a small, isolated change instead of a copy of the COLMAP path.
- **Scope:** factor the "cameras + point cloud → `Scene` + initial splat message" logic so COLMAP and a
  future `vggt.rs` share it; keep `load_dataset`'s detection dispatch
  (`crates/brush-dataset/src/formats/mod.rs:56-72`) thin. **No behavior change.**
- **Files:** `crates/brush-dataset/src/formats/mod.rs`, possibly a small new shared module; touch
  `colmap.rs` only to extract, not to alter.
- **Verify:** existing dataset tests pass unchanged (`cargo test -p brush-dataset`).
- **Note:** skip if PR3's author finds the COLMAP loader already cleanly reusable — this is a *de-risking*
  step, not a requirement.

### PR3 — Native `vggt.rs` format loader · *Option B*
- **Goal:** drop a VGGT-Omega output folder straight into Brush, no COLMAP round-trip.
- **Scope:** define the on-disk schema (images + a `vggt.json`/`.npz` of per-frame extrinsics/intrinsics,
  optional depth-derived `points.ply`); parse into Brush's `Camera`
  (`crates/brush-render/src/camera.rs:11-19`) and `Point3D` (`crates/colmap-reader/src/lib.rs:102-114`),
  **reusing** the w2c→c2w inversion (`colmap.rs:196-201`), FoV→intrinsics, and RGB→SH
  (`colmap.rs:254-289`) logic. Register the format in the detection order (`formats/mod.rs:56-72`).
- **Files:** new `crates/brush-dataset/src/formats/vggt.rs`; edit `crates/brush-dataset/src/formats/mod.rs`.
- **Verify:** **golden test** — a checked-in tiny VGGT fixture loads to expected `Camera`
  extrinsics/intrinsics within tolerance; a few train iterations run. Cross-check: a vggt-loaded scene ≈
  the COLMAP-loaded scene for the same capture (guards the convention gotcha — see
  [§2](#2-the-data-mapping--what-converts-to-what)).
- **Risk:** silent coordinate/convention mismatch → plausible-but-wrong (mirrored/rotated) scene. The
  COLMAP cross-check is the guard.

### PR4 — Dense-depth point-cloud init · *core*
- **Goal:** use VGGT-Omega's dense unprojected depth cloud (subsampled) to seed splats, instead of random.
- **Scope:** add an init path alongside `create_random_splats`
  (`crates/brush-train/src/splat_init.rs:54-128`) that ingests the dense cloud; subsample so densification
  dynamics aren't swamped; gate behind a config knob.
- **Files:** `crates/brush-train/src/splat_init.rs`, `crates/brush-train/src/config.rs`.
- **Verify:** ablate convergence speed / final PSNR vs. random init on a sample scene; confirm
  subsampling keeps splat counts sane.

### PR5 — Optional camera pose refinement · *stretch — but top priority for on-device (ARKit) capture*
- **Goal:** let Brush refine cameras during training, so slightly-off feed-forward poses don't cap quality.
- **Scope:** make `batch.camera` parameters optimizable (today read-only,
  `crates/brush-train/src/train.rs:167`); add pose LR schedules; default **off**.
- **Files:** `crates/brush-train/src/train.rs`, `crates/brush-train/src/config.rs`.
- **Verify:** perturb known-good poses → refinement recovers them and PSNR improves; off by default = no
  regression. **Independent of VGGT** (helps COLMAP too) but its value rises with a feed-forward front-end.
- **On-device priority:** for ARKit/LiDAR captures (e.g. **SplatKing**), VIO pose **drift** makes this the
  *highest-leverage* feature — promote it ahead of PR3/PR4 if on-device is the primary capture path. See
  [comparison §4.3](vggt-omega-vs-brush.md#43-what-this-changes).

### PR6 — Depth-supervised loss · *stretch*
- **Goal:** use VGGT-Omega's `depth` + `depth_conf` to regularize Brush geometry (Brush has no depth loss
  today).
- **Scope:** add an optional confidence-weighted depth loss term; wire through the loss config.
- **Files:** `crates/brush-loss`, `crates/brush-train`.
- **Verify:** improved depth/geometry metrics on a scene with reference depth; default off = no change.
- **Depends on** PR3/PR4 (needs depth ingested).

---

## 5. Risks specific to the chain

The full risk list is in the [comparison doc §8](vggt-omega-vs-brush.md#8-gaps-mismatches--risks). The
chain-specific ones:

- **Convention drift (PR3)** — top risk; fails silently. Mitigated by routing PR1 through COLMAP and by
  PR3's COLMAP cross-check test.
- **Pose quality gate (PR1)** — the whole chain is only worth building if PR1 shows VGGT poses yield
  COLMAP-comparable splats. PR5 (pose refine) is the fallback if they're close-but-not-quite.
- **Dense-vs-sparse init (PR4)** — un-subsampled dense clouds change densification; needs tuning.
- **Cross-platform (PR3/PR4)** — keep the loader pure-Rust; **don't** pull a CUDA/Python inference dep
  into `brush-dataset` (that's the deferred Option C, which fights Brush's WASM/Android promise).

---

## 6. Glossary

Implementation terms specific to this plan. Shared 3D/ML vocabulary lives in the
[comparison doc glossary](vggt-omega-vs-brush.md#10-glossary).

- **Option A / B / C** — the three integration depths from the comparison doc: A = offline COLMAP export
  (no Brush code), B = native `vggt.rs` loader, C = in-engine inference (deferred). This plan executes
  A then B.
- **Format loader** — a module under `crates/brush-dataset/src/formats/` that turns one on-disk dataset
  layout (COLMAP, Nerfstudio, …) into Brush's internal `Scene` + initial splats. PR3 adds a `vggt.rs` one.
- **Format detection / dispatch** — `load_dataset` probing for known files and picking a loader
  (`formats/mod.rs:56-72`); PR3 adds VGGT to the order.
- **Ingestion seam** — the shared "cameras + points → Scene + init" code path PR2 factors out so loaders
  don't duplicate it.
- **COLMAP `sparse/` model** — the `cameras` + `images` + `points3D` files Brush reads and PR1's script
  writes; the lossless bridge format.
- **`pycolmap`** — the Python COLMAP bindings VGGT-Omega's `export` extra uses to emit a `sparse/` model
  (`pyproject.toml`).
- **Golden / fixture test** — a checked-in tiny input with an expected output; PR3's guard that loading
  produces the right cameras (and that conventions didn't silently flip).
- **w2c / c2w** — world-to-camera vs. camera-to-world extrinsics. VGGT-Omega emits w2c (OpenCV); Brush
  works in c2w; the inversion (`colmap.rs:196-201`) is the convention-sensitive step.
- **Pose refinement** — optimizing cameras *during* splat training (PR5). Brush freezes poses today, so
  this is net-new and is what makes slightly-imperfect feed-forward poses acceptable.

---

## 7. Sources & references

Code targets for each PR (this checkout, read 2026-06-23; lines drift):

- **Brush ingestion:** `crates/brush-dataset/src/formats/mod.rs:56-72` (detection order);
  `crates/brush-dataset/src/formats/colmap.rs` (w2c→c2w `:196-201`, RGB→SH `:254-289`, camera-model build
  `:304-380`).
- **Brush types:** `crates/brush-render/src/camera.rs:11-19` (`Camera`);
  `crates/colmap-reader/src/lib.rs:102-114` (`Point3D`).
- **Brush init/train:** `crates/brush-train/src/splat_init.rs:54-128` (random init → PR4 adds dense);
  `crates/brush-train/src/train.rs:167` (`batch.camera` read-only → PR5 makes optimizable);
  `crates/brush-train/src/config.rs:9` (`total_train_iters`).
- **VGGT-Omega export surface:** `vggt_omega/utils/pose_enc.py:29-52` (`encoding_to_camera`);
  `demo_gradio.py:75-104` (`unproject_depth_map_to_point_map`); `pyproject.toml` (`export` extra =
  `pycolmap`).
- **Background & full rationale:** [`vggt-omega-vs-brush.md`](vggt-omega-vs-brush.md) (comparison,
  integration surface, risks, sources).
