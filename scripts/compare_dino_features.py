# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
"""A/B parity comparison of two dino_features/ output directories.

Used to validate crates/dino-extract (Rust/candle) against the retired
Python/torch reference (`git show 222a1220:scripts/extract_dino_features.py`),
and to re-verify dino-extract against its own f32 baseline after performance
changes (fp16, fused attention): run both extractors on copies of the same
dataset (the Rust tool with --dump-raw, the Python script patched to also
save raw maps — see crates/dino-extract/src/README.md for the exact
commands), then:

    uv run scripts/compare_dino_features.py <ref_dir> <test_dir>

Compares (1) raw pre-PCA features per image, (2) PCA bases as subspaces,
(3) final projected .npy outputs after orthogonal alignment of the two bases.
PCA bases are only defined up to sign/rotation within near-degenerate
eigenspaces, so raw elementwise diff of projected maps is not meaningful —
alignment first is the correct comparison.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles, svd

ref_dir, test_dir = Path(sys.argv[1]), Path(sys.argv[2])

stems = sorted(p.name[:-8] for p in ref_dir.glob("*.raw.npy"))
assert stems, f"no *.raw.npy dumps found in {ref_dir}"
feat_dim = np.load(ref_dir / f"{stems[0]}.raw.npy").shape[-1]

print(f"=== 1. Raw pre-PCA features ({feat_dim}-dim, per pixel) ===")
all_cos, all_rel = [], []
for stem in stems:
    a = np.load(ref_dir / f"{stem}.raw.npy").reshape(-1, feat_dim).astype(np.float64)
    b = np.load(test_dir / f"{stem}.raw.npy").reshape(-1, feat_dim).astype(np.float64)
    assert a.shape == b.shape, f"{stem}: shape {a.shape} vs {b.shape}"
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    rel = np.linalg.norm(a - b, axis=1) / np.linalg.norm(a, axis=1)
    all_cos.append(cos)
    all_rel.append(rel)
    print(f"  {stem}: cos mean={cos.mean():.6f} p1={np.percentile(cos, 1):.6f} "
          f"min={cos.min():.6f} | relL2 mean={rel.mean():.4f} max={rel.max():.4f}")
cos = np.concatenate(all_cos)
rel = np.concatenate(all_rel)
print(f"  ALL: cos mean={cos.mean():.6f} p1={np.percentile(cos, 1):.6f} min={cos.min():.6f} "
      f"| relL2 mean={rel.mean():.4f} p99={np.percentile(rel, 99):.4f}")

print("\n=== 2. PCA basis subspace comparison ===")
va = np.load(ref_dir / "pca.npy").astype(np.float64)
vb = np.load(test_dir / "pca.npy").astype(np.float64)
q = va.shape[1]
angles = np.degrees(subspace_angles(va, vb))
print(f"  principal angles (deg): max={angles.max():.3f} mean={angles.mean():.3f} "
      f"median={np.median(angles):.3f}")
print(f"  orthonormality check: |V^T V - I| ref={np.abs(va.T @ va - np.eye(q)).max():.2e} "
      f"test={np.abs(vb.T @ vb - np.eye(q)).max():.2e}")

print("\n=== 3. Projected outputs after orthogonal alignment ===")
# Align test basis to ref basis: R = argmin ||Vb R - Va||_F (Procrustes).
u, _, vt = svd(vb.T @ va)
r_align = u @ vt
resid = np.linalg.norm(vb @ r_align - va) / np.linalg.norm(va)
print(f"  basis Procrustes residual: {resid:.4f}")
all_rel = []
for stem in stems:
    a = np.load(ref_dir / f"{stem}.npy").reshape(-1, q).astype(np.float64)
    b = np.load(test_dir / f"{stem}.npy").reshape(-1, q).astype(np.float64) @ r_align
    all_rel.append(np.linalg.norm(a - b, axis=1) / np.linalg.norm(a, axis=1))
rel = np.concatenate(all_rel)
print(f"  aligned projected maps: relL2 mean={rel.mean():.4f} p99={np.percentile(rel, 99):.4f}")

# Basis-independent check: feature-similarity structure of the projected maps.
rng = np.random.default_rng(0)
a = np.load(ref_dir / f"{stems[0]}.npy").reshape(-1, q)
b = np.load(test_dir / f"{stems[0]}.npy").reshape(-1, q)
idx = rng.choice(len(a), 500, replace=False)
sim = np.corrcoef((a[idx] @ a[idx].T).ravel(), (b[idx] @ b[idx].T).ravel())[0, 1]
print(f"  pairwise-similarity structure correlation (500 px sample): {sim:.6f}")

print("\n=== 4. meta.json ===")
ma = json.loads((ref_dir / "meta.json").read_text())
mb = json.loads((test_dir / "meta.json").read_text())
for k in sorted(set(ma) | set(mb)):
    match = "OK" if ma.get(k) == mb.get(k) else "DIFF"
    print(f"  {k}: ref={ma.get(k)} test={mb.get(k)} [{match}]")
