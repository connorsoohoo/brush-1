//! Bicubic resampling matching the two `PyTorch` code paths the Python script
//! exercises:
//!
//! - `resize_image_bicubic_aa`: `torchvision.transforms.functional.resize`
//!   with `antialias=True` — PIL-compatible bicubic kernel (`a = -0.5`), kernel
//!   support scaled by the downsampling ratio, weights renormalized.
//! - `resize_grid_bicubic_torch`: `torch.nn.functional.interpolate` with
//!   `mode="bicubic", antialias=False` and `scale_factor` semantics — torch
//!   cubic kernel (`a = -0.75`), fixed 4-tap support, `src = (dst + 0.5) /
//!   scale - 0.5`. Used for `DINOv2`'s positional-embedding interpolation,
//!   which passes `scale_factor = (n + 0.1) / m` (the `interpolate_offset`
//!   quirk), so the scale is *not* simply `out / in`.

fn cubic(x: f32, a: f32) -> f32 {
    let x = x.abs();
    if x < 1.0 {
        (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0
    } else if x < 2.0 {
        a * (x * x * x - 5.0 * x * x + 8.0 * x - 4.0)
    } else {
        0.0
    }
}

/// Per-output-index taps: start index into the source axis plus normalized
/// (or raw, for the non-AA path) kernel weights.
struct Taps {
    start: usize,
    weights: Vec<f32>,
}

/// PIL/torchvision-antialias taps: kernel stretched by `scale` when
/// downsampling, weights renormalized to sum to 1.
fn taps_antialias(in_len: usize, out_len: usize, a: f32) -> Vec<Taps> {
    let scale = in_len as f32 / out_len as f32;
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    (0..out_len)
        .map(|dst| {
            let center = (dst as f32 + 0.5) * scale - 0.5;
            let lo = ((center - support).floor() as i64).max(0) as usize;
            let hi = ((center + support).ceil() as i64).min(in_len as i64 - 1) as usize;
            let mut weights: Vec<f32> = (lo..=hi)
                .map(|i| cubic((i as f32 - center) / filter_scale, a))
                .collect();
            let sum: f32 = weights.iter().sum();
            for w in &mut weights {
                *w /= sum;
            }
            Taps { start: lo, weights }
        })
        .collect()
}

/// torch non-antialiased bicubic taps under explicit `scale`: always 4 taps
/// around `src = (dst + 0.5) / scale - 0.5`, indices clamped to the border.
/// Weights sum to 1 by construction; border clamping folds out-of-range tap
/// weight onto the edge sample, exactly like torch's index clamp.
fn taps_torch_no_aa(in_len: usize, out_len: usize, scale: f32, a: f32) -> Vec<Taps> {
    (0..out_len)
        .map(|dst| {
            let center = (dst as f32 + 0.5) / scale - 0.5;
            let base = center.floor() as i64;
            let frac = center - base as f32;
            let lo = (base - 1).clamp(0, in_len as i64 - 1) as usize;
            let hi = (base + 2).clamp(0, in_len as i64 - 1) as usize;
            let mut weights = vec![0.0f32; hi - lo + 1];
            for k in -1i64..=2 {
                let w = cubic(k as f32 - frac, a);
                let idx = (base + k).clamp(0, in_len as i64 - 1) as usize;
                weights[idx - lo] += w;
            }
            Taps { start: lo, weights }
        })
        .collect()
}

/// Separable resample of a `[h, w, c]` row-major grid along both spatial axes.
fn resample_2d(
    src: &[f32],
    (_h, w, c): (usize, usize, usize),
    row_taps: &[Taps],
    col_taps: &[Taps],
) -> Vec<f32> {
    let out_h = row_taps.len();
    let out_w = col_taps.len();
    // Vertical pass: [h, w, c] -> [out_h, w, c].
    let mut tmp = vec![0.0f32; out_h * w * c];
    for (oy, taps) in row_taps.iter().enumerate() {
        for (dy, &wgt) in taps.weights.iter().enumerate() {
            let sy = taps.start + dy;
            let src_row = &src[sy * w * c..(sy + 1) * w * c];
            let dst_row = &mut tmp[oy * w * c..(oy + 1) * w * c];
            for (d, s) in dst_row.iter_mut().zip(src_row) {
                *d += wgt * s;
            }
        }
    }
    // Horizontal pass: [out_h, w, c] -> [out_h, out_w, c].
    let mut out = vec![0.0f32; out_h * out_w * c];
    for oy in 0..out_h {
        let src_row = &tmp[oy * w * c..(oy + 1) * w * c];
        let dst_row = &mut out[oy * out_w * c..(oy + 1) * out_w * c];
        for (ox, taps) in col_taps.iter().enumerate() {
            for (dx, &wgt) in taps.weights.iter().enumerate() {
                let sx = taps.start + dx;
                for ch in 0..c {
                    dst_row[ox * c + ch] += wgt * src_row[sx * c + ch];
                }
            }
        }
    }
    out
}

/// `torchvision.resize(..., BICUBIC, antialias=True)` on a `[h, w, c]` f32
/// grid in `[0, 1]`.
pub fn resize_image_bicubic_aa(
    src: &[f32],
    (h, w, c): (usize, usize, usize),
    (out_h, out_w): (usize, usize),
) -> Vec<f32> {
    let row_taps = taps_antialias(h, out_h, -0.5);
    let col_taps = taps_antialias(w, out_w, -0.5);
    resample_2d(src, (h, w, c), &row_taps, &col_taps)
}

/// `F.interpolate(..., mode="bicubic", antialias=False, scale_factor=(sy, sx))`
/// on a `[h, w, c]` f32 grid (used for the `DINOv2` pos-embed grid).
pub fn resize_grid_bicubic_torch(
    src: &[f32],
    (h, w, c): (usize, usize, usize),
    (out_h, out_w): (usize, usize),
    (scale_y, scale_x): (f32, f32),
) -> Vec<f32> {
    let row_taps = taps_torch_no_aa(h, out_h, scale_y, -0.75);
    let col_taps = taps_torch_no_aa(w, out_w, scale_x, -0.75);
    resample_2d(src, (h, w, c), &row_taps, &col_taps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_resize_is_identity() {
        let src: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let out = resize_image_bicubic_aa(&src, (3, 4, 1), (3, 4));
        for (a, b) in out.iter().zip(&src) {
            assert!((a - b).abs() < 1e-5, "identity resize changed values");
        }
    }

    #[test]
    fn constant_grid_stays_constant() {
        let src = vec![0.7f32; 37 * 37 * 2];
        let out = resize_grid_bicubic_torch(
            &src,
            (37, 37, 2),
            (67, 90),
            ((67.0 + 0.1) / 37.0, (90.0 + 0.1) / 37.0),
        );
        assert_eq!(out.len(), 67 * 90 * 2, "output shape");
        for v in out {
            assert!((v - 0.7).abs() < 1e-5, "constant not preserved: {v}");
        }
    }
}
