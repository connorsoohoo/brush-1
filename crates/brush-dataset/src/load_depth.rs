use brush_vfs::BrushVfs;
use burn::tensor::TensorData;
use std::{
    io::Cursor,
    path::{Path, PathBuf},
    sync::Arc,
};
use thiserror::Error;
use tokio::io::AsyncReadExt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DepthFormat {
    Tiff,
    Bin { min_confidence: u8 },
}

/// Lazily-loaded per-view depth map, supporting single-channel float32 depth stored as TIFF
/// or raw binary files.
#[derive(Clone, Debug)]
pub struct LoadDepth {
    vfs: Arc<BrushVfs>,
    path: PathBuf,
    format: DepthFormat,
}

impl PartialEq for LoadDepth {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path && self.format == other.format
    }
}

impl LoadDepth {
    pub fn new(vfs: Arc<BrushVfs>, path: PathBuf, format: DepthFormat) -> Self {
        Self { vfs, path, format }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub async fn load(
        &self,
        expected_h: usize,
        expected_w: usize,
    ) -> Result<TensorData, LoadDepthError> {
        let depth = self.load_vec(expected_h, expected_w).await?;
        Ok(TensorData::new(depth, [expected_h, expected_w]))
    }

    pub async fn load_vec(
        &self,
        expected_h: usize,
        expected_w: usize,
    ) -> Result<Vec<f32>, LoadDepthError> {
        let mut bytes = vec![];
        self.vfs
            .reader_at_path(&self.path)
            .await?
            .read_to_end(&mut bytes)
            .await?;

        match &self.format {
            DepthFormat::Tiff => {
                let (depth, w, h) = decode_f32_tiff(&bytes)?;
                if w != expected_w || h != expected_h {
                    Err(LoadDepthError::ReadTiffError(format!(
                        "invalid depth size {w} x {h}, expected {expected_w} x {expected_h}"
                    )))
                } else {
                    Ok(depth)
                }
            }
            DepthFormat::Bin { min_confidence } => {
                // Try loading corresponding confidence file
                let path_str = self.path.to_str().ok_or_else(|| {
                    LoadDepthError::ReadTiffError("Invalid non-UTF-8 path".to_owned())
                })?;
                // Confidence file matches depth path except replacing "_depth.bin" with "_confidence.bin"
                let conf_path_str = path_str.replace("_depth.bin", "_confidence.bin");
                let conf_path = PathBuf::from(conf_path_str);

                let mut conf_bytes = None;
                if self.vfs.reader_at_path(&conf_path).await.is_ok() {
                    let mut c_bytes = vec![];
                    if let Ok(mut reader) = self.vfs.reader_at_path(&conf_path).await
                        && reader.read_to_end(&mut c_bytes).await.is_ok()
                    {
                        conf_bytes = Some(c_bytes);
                    }
                }

                decode_bin_depth(
                    &bytes,
                    conf_bytes.as_deref(),
                    *min_confidence,
                    expected_w,
                    expected_h,
                )
            }
        }
    }
}

/// Decode a single-channel float32 TIFF into in row-major order.
fn decode_f32_tiff(bytes: &[u8]) -> Result<(Vec<f32>, usize, usize), LoadDepthError> {
    let mut decoder = tiff::decoder::Decoder::new(Cursor::new(bytes))?;

    let tiff::decoder::DecodingResult::F32(depth) = decoder.read_image()? else {
        return Err(LoadDepthError::ReadTiffError(
            "unsupported TIFF sample format (expected float32 depth)".to_owned(),
        ));
    };

    let (w, h) = decoder.dimensions()?;
    let (w, h) = (w as usize, h as usize);

    if w * h != depth.len() {
        Err(LoadDepthError::ReadTiffError(
            "expected only a single channel".to_owned(),
        ))
    } else {
        Ok((depth, w, h))
    }
}

/// Parse Splat King binary float32 depth and apply confidence mask, then upscale.
fn decode_bin_depth(
    depth_bytes: &[u8],
    confidence_bytes: Option<&[u8]>,
    min_confidence: u8,
    expected_w: usize,
    expected_h: usize,
) -> Result<Vec<f32>, LoadDepthError> {
    const SRC_W: usize = 256;
    const SRC_H: usize = 192;

    if depth_bytes.len() != SRC_W * SRC_H * 4 {
        return Err(LoadDepthError::ReadTiffError(format!(
            "invalid raw depth binary size {}, expected {}",
            depth_bytes.len(),
            SRC_W * SRC_H * 4
        )));
    }

    let mut depth = vec![0.0f32; SRC_W * SRC_H];
    for i in 0..(SRC_W * SRC_H) {
        let offset = i * 4;
        let bytes = [
            depth_bytes[offset],
            depth_bytes[offset + 1],
            depth_bytes[offset + 2],
            depth_bytes[offset + 3],
        ];
        depth[i] = f32::from_le_bytes(bytes);
    }

    // Replace NaNs with 0.0 before upscaling to prevent propagation
    for d in &mut depth {
        if d.is_nan() {
            *d = 0.0;
        }
    }

    if let Some(conf) = confidence_bytes
        && conf.len() == SRC_W * SRC_H
    {
        for i in 0..(SRC_W * SRC_H) {
            if conf[i] < min_confidence {
                depth[i] = 0.0;
            }
        }
    }

    let resized = resize_bilinear(&depth, SRC_W, SRC_H, expected_w, expected_h);
    Ok(resized)
}

/// Helper function to perform bilinear interpolation on a float grid
fn resize_bilinear(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<f32> {
    let mut dst = vec![0.0; dst_w * dst_h];
    let x_ratio = if dst_w > 1 {
        (src_w - 1) as f32 / (dst_w - 1) as f32
    } else {
        0.0
    };
    let y_ratio = if dst_h > 1 {
        (src_h - 1) as f32 / (dst_h - 1) as f32
    } else {
        0.0
    };

    for y in 0..dst_h {
        for x in 0..dst_w {
            let px = x as f32 * x_ratio;
            let py = y as f32 * y_ratio;

            let x_l = px.floor() as usize;
            let y_l = py.floor() as usize;
            let x_h = (x_l + 1).min(src_w - 1);
            let y_h = (y_l + 1).min(src_h - 1);

            let x_diff = px - x_l as f32;
            let y_diff = py - y_l as f32;

            let a = src[y_l * src_w + x_l];
            let b = src[y_l * src_w + x_h];
            let c = src[y_h * src_w + x_l];
            let d = src[y_h * src_w + x_h];

            let val = a * (1.0 - x_diff) * (1.0 - y_diff)
                + b * x_diff * (1.0 - y_diff)
                + c * (1.0 - x_diff) * y_diff
                + d * x_diff * y_diff;

            dst[y * dst_w + x] = val;
        }
    }
    dst
}

#[derive(Error, Debug)]
pub enum LoadDepthError {
    #[error("I/O error while loading depth map: {0}")]
    Io(#[from] std::io::Error),

    #[error("Error while loading TIFF file: {0}")]
    LoadTiffError(#[from] tiff::TiffError),

    #[error("Error while reading TIFF file: {0}")]
    ReadTiffError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resize_bilinear() {
        let src = vec![1.0, 2.0, 3.0, 4.0];
        let resized = resize_bilinear(&src, 2, 2, 4, 4);
        assert_eq!(resized.len(), 16);
        // Check corners
        assert_eq!(resized[0], 1.0);
        assert_eq!(resized[3], 2.0);
        assert_eq!(resized[12], 3.0);
        assert_eq!(resized[15], 4.0);

        // Check center interpolation at index 5 (which is 1,1 in 4x4)
        assert!((resized[5] - 2.0).abs() < 1e-6);
    }
}
