//! PCA fit mirroring the Python script's `torch.pca_lowrank` usage: fit on
//! (up to 4-million-row subsampled) mean-centered rows, then project *uncentered*
//! features through the resulting `[feat_dim, pca_dim]` basis.
//!
//! Unlike `torch.pca_lowrank` (randomized SVD, `niter=20`), this computes the
//! exact top-`q` eigenvectors of the feature covariance. Both yield an
//! orthonormal basis of the same principal subspace; individual columns can
//! differ in sign (and near-degenerate trailing components can rotate), which
//! downstream `DiG` training is invariant to. Columns are sign-canonicalized
//! (largest-magnitude entry positive) so runs are reproducible.

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use rand::{SeedableRng, rngs::StdRng};

pub const PCA_MAX_ROWS: usize = 4_000_000;
const COV_CHUNK_ROWS: usize = 16_384;

/// Fit on `rows` (`n x dim`, row-major); returns a row-major `[dim, q]` basis.
pub fn fit_pca(rows: &[f32], dim: usize, q: usize, device: &Device) -> Result<Vec<f32>> {
    let n = rows.len() / dim;
    assert_eq!(rows.len(), n * dim, "rows not a multiple of dim");

    // Subsample at most PCA_MAX_ROWS rows (seeded, like the script).
    let mut rng = StdRng::seed_from_u64(0);
    let sampled: Vec<f32>;
    let (fit_rows, n_fit) = if n > PCA_MAX_ROWS {
        let idx = rand::seq::index::sample(&mut rng, n, PCA_MAX_ROWS);
        sampled = idx
            .iter()
            .flat_map(|i| rows[i * dim..(i + 1) * dim].iter().copied())
            .collect();
        println!("Subsampled {PCA_MAX_ROWS} of {n} rows for PCA fit");
        (sampled.as_slice(), PCA_MAX_ROWS)
    } else {
        (rows, n)
    };

    // Column means in f64.
    let mut mean = vec![0.0f64; dim];
    for row in fit_rows.chunks_exact(dim) {
        for (m, v) in mean.iter_mut().zip(row) {
            *m += f64::from(*v);
        }
    }
    for m in &mut mean {
        *m /= n_fit as f64;
    }
    let mean_f32: Vec<f32> = mean.iter().map(|&m| m as f32).collect();

    // Covariance X^T X of centered rows: chunked f32 matmuls on the device,
    // accumulated in f64 on the host.
    let mean_t = Tensor::from_vec(mean_f32, (1, dim), device)?;
    let mut cov = vec![0.0f64; dim * dim];
    for chunk in fit_rows.chunks(COV_CHUNK_ROWS * dim) {
        let rows_in_chunk = chunk.len() / dim;
        let x = Tensor::from_vec(chunk.to_vec(), (rows_in_chunk, dim), device)?
            .broadcast_sub(&mean_t)?;
        let c = x.t()?.matmul(&x)?.flatten_all()?.to_vec1::<f32>()?;
        for (acc, v) in cov.iter_mut().zip(&c) {
            *acc += f64::from(*v);
        }
    }

    // Exact symmetric eigendecomposition; take the top-q eigenvectors.
    let cov = nalgebra::DMatrix::from_row_slice(dim, dim, &cov);
    let eig = nalgebra::SymmetricEigen::new(cov);
    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_by(|&a, &b| {
        eig.eigenvalues[b]
            .partial_cmp(&eig.eigenvalues[a])
            .context("NaN eigenvalue")
            .expect("eigenvalues comparable")
    });

    let mut basis = vec![0.0f32; dim * q];
    for (out_col, &src_col) in order.iter().take(q).enumerate() {
        let col = eig.eigenvectors.column(src_col);
        // Sign canonicalization: largest-|entry| positive.
        let flip = col
            .iter()
            .copied()
            .max_by(|a, b| a.abs().partial_cmp(&b.abs()).expect("finite eigenvector"))
            .map_or(1.0, |v| if v < 0.0 { -1.0 } else { 1.0 });
        for r in 0..dim {
            basis[r * q + out_col] = (col[r] * flip) as f32;
        }
    }
    Ok(basis)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_dominant_direction() {
        // Rows spread along (1, 1, 0)/sqrt(2) with small noise elsewhere.
        let n = 1000;
        let mut rows = Vec::with_capacity(n * 3);
        for i in 0..n {
            let t = (i as f32 / n as f32 - 0.5) * 10.0;
            let eps = (i as f32 * 0.7).sin() * 0.01;
            rows.extend_from_slice(&[t, t, eps]);
        }
        let basis = fit_pca(&rows, 3, 1, &Device::Cpu).expect("pca fit");
        let inv_sqrt2 = 1.0 / 2.0f32.sqrt();
        assert!(
            (basis[0].abs() - inv_sqrt2).abs() < 1e-3
                && (basis[1].abs() - inv_sqrt2).abs() < 1e-3
                && basis[2].abs() < 0.05,
            "principal axis wrong: {basis:?}"
        );
        assert!(basis[0] > 0.0, "sign canonicalization");
    }
}
