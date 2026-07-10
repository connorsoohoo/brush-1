# dino-extract — algorithmic speedups

Three of the four levers below are implemented; measured on the family_room
capture (10 images, 1920×1440, ViT-B/14 at max-size 1260, M-series GPU):

| configuration                        | s/image | vs. original |
|--------------------------------------|---------|--------------|
| original (naive attention, serial)   | 3.54    | 1.0×         |
| SDPA + pipeline overlap (f32, default) | 1.85  | 1.9×         |
| `--dtype f16`                        | 1.41    | 2.5×         |

Parity of the shipped paths against the original naive-f32 implementation
(via `compare_dino_features.py`, same harness as the original Python A/B):
f32/SDPA raw-feature mean cosine 1.000000, PCA principal angle ≤0.002°;
f16 mean cosine 0.999985, projected-map structure correlation 0.999994.

1. **fp16/bf16 inference** — implemented (`--dtype f16|bf16`, f32 default).
   Loads the `VarBuilder` at the requested dtype; compute runs at that dtype
   and features are cast back to f32 before PCA/output. Costs a little
   numerical parity (see table above) — re-verify with the A/B below when
   touching this path.

2. **Chunked / fused attention** — implemented. On Metal, attention goes
   through candle's fused SDPA kernel (`candle_nn::ops::sdpa`; all supported
   models have head dim 64). Elsewhere it runs in 512-row query chunks. Both
   avoid materializing the `[heads, T, T]` score tensor (~1.7 GB f32 at
   T≈6031) the naive path softmaxed over. The torch reference *can't* do this
   without xformers, so this is where Rust pulls ahead rather than matching.

3. **Pipeline overlap** — implemented. A loader thread decodes + bicubic-
   resizes image `i+1` while the GPU processes image `i`, hiding the per-image
   CPU cost behind inference.

4. **Micro-batching** (not implemented, small). Batching B images through the
   model amortizes kernel-launch overhead, but the kernels are large and
   memory-bound so the gain is minor — and peak attention memory scales
   with B.

## Fast impl-vs-impl correctness checks

`cargo test -p dino-extract` (~1 s in release, no dataset or network needed)
covers the parity questions that previously required the full A/B:

- `golden_features_match_reference` — **real-weight golden check**: forwards
  the two small frames in `fixtures/` (ViT-B f32, CPU, max-size 224) and
  compares per-pixel cosine against `fixtures/*.raw.npy`, which were generated
  by the original Python-parity-validated implementation (naive f32
  attention). Skips — never downloads — if the ViT-B weights aren't already in
  the local HF cache.
- `chunked_attention_matches_naive` — chunked vs. the original full-matrix
  attention on random tensors (CPU).
- `sdpa_attention_matches_naive_on_metal` — the fused Metal kernel vs. the
  original attention (skips when Metal is unavailable).
- `f16_forward_close_to_f32` — end-to-end forward on a tiny random-weight
  model, f16 vs. f32 cosine.

The naive attention path (the one originally parity-validated against
Python/torch) is kept in `dinov2.rs` under `#[cfg(test)]` as the baseline
these tests compare against.

To regenerate the goldens (only if the *reference semantics* intentionally
change): build the last-validated implementation, run it on a copy of the
fixture images with `--dump-raw --max-size 224 --device cpu`, and copy the
`*.raw.npy` outputs back into `fixtures/`.

The full A/B below remains the deep-validation path (full-resolution images,
PCA subspace + projected-map checks); the golden test is the everyday
regression gate.

## Parity A/B procedure

Exact commands from the original evaluation (10 images, 1920×1440). Two
dataset copies are used so each extractor writes its own `dino_features/`.

The Python/torch reference script has since been removed from the tree
(after a training-level A/B confirmed parity — see
[#34](https://github.com/connorsoohoo/brush/pull/34)); recover it from git
history when re-running this procedure:

```bash
git show 222a1220:scripts/extract_dino_features.py > /tmp/extract_dino_features.py
```

```bash
# 0. Twin datasets: same 10 images (every 17th frame) into two copies.
mkdir -p /tmp/eval_rs/images /tmp/eval_py/images
for f in $(ls /path/to/capture/images/*.jpg | awk 'NR % 17 == 1' | head -10); do
  cp "$f" /tmp/eval_rs/images/; cp "$f" /tmp/eval_py/images/
done

# 1. Rust extractor, keeping raw pre-PCA maps for comparison.
cargo run --release -p dino-extract -- --data /tmp/eval_rs --dump-raw

# 2. Python reference (recovered from git history, above). The stock script
#    doesn't save raw maps, so run a patched copy that also dumps
#    <stem>.raw.npy next to each projection:
sed 's|feats_per_image.append(desc.cpu())|feats_per_image.append(desc.cpu())\n        np.save(out_dir / f"{path.stem}.raw.npy", np.ascontiguousarray(desc.cpu().numpy().astype(np.float32)))|' \
  /tmp/extract_dino_features.py > /tmp/extract_dino_features_raw.py
uv run /tmp/extract_dino_features_raw.py --data /tmp/eval_py

# 3. Compare raw features, PCA subspaces, aligned projections, meta.json.
uv run scripts/compare_dino_features.py \
  /tmp/eval_py/dino_features /tmp/eval_rs/dino_features

# 4. Timing (warm caches — run each once beforehand so HF/torch caches and
#    uv envs are populated; model download happens on the first run).
/usr/bin/time cargo run --release -p dino-extract -- --data /tmp/eval_rs
/usr/bin/time uv run /tmp/extract_dino_features.py --data /tmp/eval_py
```

Reference results (ViT-B/14, fp32, M-series): raw-feature per-pixel cosine
mean 0.999975; PCA leading-subspace median principal angle 0.09°;
projected-map pairwise-similarity correlation 0.99996.

The one non-obvious reading: `compare` step 2 can report a large *max*
principal angle (tens of degrees — and it varies between reruns of the
*Python* script, since `torch.pca_lowrank` is unseeded) between the two
`pca.npy` bases. That tail rotation comes from the Python script's
`torch.pca_lowrank` (randomized SVD), not from this crate. Control experiment — exact PCA (numpy `eigh`) on the *Python*
run's own raw features vs. the basis torch produced from those same features:

```bash
uv run --with numpy,scipy python3 -c "
import numpy as np
from scipy.linalg import subspace_angles
from pathlib import Path
d = Path('/tmp/eval_py/dino_features')
feats = np.concatenate([np.load(p).reshape(-1, 768)
                        for p in sorted(d.glob('*.raw.npy'))]).astype(np.float64)
c = feats - feats.mean(0)
w, v = np.linalg.eigh(c.T @ c)
v_exact = v[:, np.argsort(w)[::-1][:96]]
a = np.degrees(subspace_angles(v_exact, np.load(d / 'pca.npy').astype(np.float64)))
print(f'torch pca_lowrank vs exact PCA on identical data: max={a.max():.1f} deg')
"
```

This reproduces the same ~54° tail on identical inputs, while this crate's
exact PCA agrees with numpy's exact PCA to <1°.
