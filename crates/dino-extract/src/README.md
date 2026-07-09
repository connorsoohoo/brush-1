# dino-extract — potential algorithmic speedups

Per-image inference currently matches the Python/torch reference (~3.2 s/image
for ViT-B/14 at max-size 1260 on an M-series GPU). That is expected: the
workload is GPU-bound, and both implementations run the same unfused math on
the same hardware. A ViT-B forward at ~6k tokens is ~1.2 TFLOP, dominated by
attention: each of the 12 blocks materializes a `[heads, T, T]` f32 score
tensor (~1.7 GB at T≈6031) and softmaxes over it — memory-bandwidth-bound, so
host-language overhead is irrelevant. The Rust win today is fixed overhead
only (0.64 s model load vs. several seconds of interpreter + torch import +
hub checks).

Levers to become genuinely faster per image, in rough order of value:

1. **fp16/bf16 inference** (~2×, cheapest). Load the `VarBuilder` at
   `DType::F16` and cast the input; bandwidth-bound attention scales almost
   linearly with element size. Costs a little numerical parity — re-run the
   raw-feature cosine comparison (`--dump-raw` + `compare` against the Python
   outputs) before trusting it; expect mean cosine to drop from ~0.99998 to
   ~0.999x. Best gated behind a `--dtype f16` flag, keeping f32 the default.

2. **Chunked / fused attention** (avoids the 1.7 GB score tensor). Either use
   candle's Metal SDPA path (`candle_nn::ops::sdpa`, available for supported
   head dims in recent candle) or compute attention in query chunks (e.g.
   512 rows at a time: scores chunk → softmax → apply, never holding the full
   `[T, T]` matrix). Cuts peak memory by ~1.7 GB and the softmax memory
   traffic substantially. Note the torch reference *can't* do this without
   xformers, so this is where Rust pulls ahead rather than matching.

3. **Pipeline overlap** (~0.3–0.5 s/image). Image decode + the antialiased
   bicubic resize run single-threaded on the CPU between GPU calls. Decode and
   resize image `i+1` on a worker thread (or rayon) while the GPU processes
   image `i`; the per-image CPU cost then hides entirely behind inference.

4. **Micro-batching** (small). Batching B images through the model amortizes
   kernel-launch overhead, but since the kernels are large and memory-bound
   the gain is minor — and peak attention memory scales with B. Only worth it
   after (2).

Items 1–3 stack: plausibly ~2.5–3× per image combined. All of them change
performance only — outputs should be re-verified against the Python reference
(raw-feature cosine, PCA subspace angles) whenever one lands, using the A/B
procedure below (the same one used for the original parity evaluation in the
PR that introduced this crate).

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
