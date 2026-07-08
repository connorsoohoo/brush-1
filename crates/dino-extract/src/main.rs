//! Rust port of `scripts/extract_dino_features.py`: offline `DINOv2` feature +
//! PCA preprocessing for DiG-in-Brush, on candle (Metal on Apple silicon).
//!
//! Writes the same artifacts to `<data>/dino_features/`: per-view
//! `[h/14, w/14, pca_dim]` f32 `.npy` maps, `pca.npy`, and `meta.json`.

mod dinov2;
mod npy;
mod pca;
mod resize;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use clap::Parser;

use dinov2::{Config, DinoVisionTransformer};

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
/// Feature maps are divided by this, matching DIG's dataloader.
const SCALE_DIV: f64 = 10.0;

#[derive(Parser)]
#[command(about = "Extract DINOv2 feature maps for a dataset (DiG preprocessing)")]
struct Args {
    /// Dataset directory (images in `images/` or directly inside).
    #[arg(long)]
    data: PathBuf,

    /// PCA output dimension.
    #[arg(long, default_value_t = 96)]
    pca_dim: usize,

    /// Model name, in torch.hub naming (`dinov2_vits14` / `dinov2_vitb14` /
    /// `dinov2_vitl14`). Weights come from the matching HF `facebook/dinov2-*`.
    #[arg(long, default_value = "dinov2_vitb14")]
    model: String,

    /// Max image dimension before feature extraction (rounded down to the
    /// patch size).
    #[arg(long, default_value_t = 1260)]
    max_size: usize,

    /// Device override: cpu or metal (default: metal if available).
    #[arg(long)]
    device: Option<String>,

    /// Local safetensors path (skips the HF hub download).
    #[arg(long)]
    weights: Option<PathBuf>,

    /// Also write raw pre-PCA `[h/14, w/14, feat_dim]` maps as
    /// `<stem>.raw.npy` (for verification against the reference).
    #[arg(long, default_value_t = false)]
    dump_raw: bool,
}

/// Matches the Python script's `get_img_resolution` (DIG's dataloader math).
fn get_img_resolution(h: usize, w: usize, max_size: usize, p: usize) -> (usize, usize) {
    if h < w {
        ((h * max_size / w) / p * p, (max_size / p) * p)
    } else {
        ((max_size / p) * p, (w * max_size / h) / p * p)
    }
}

fn find_images(data_dir: &Path) -> Result<Vec<PathBuf>> {
    let img_dir = data_dir.join("images");
    let dir = if img_dir.is_dir() {
        img_dir
    } else {
        data_dir.to_path_buf()
    };
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| matches!(e.to_lowercase().as_str(), "jpg" | "jpeg" | "png"))
        })
        .collect();
    paths.sort();
    Ok(paths)
}

fn pick_device(override_: Option<&str>) -> Result<Device> {
    match override_ {
        Some("cpu") => Ok(Device::Cpu),
        Some("metal") => Ok(Device::new_metal(0)?),
        Some("cuda") => Ok(Device::new_cuda(0)?),
        Some(other) => bail!("unknown device '{other}'"),
        None => {
            if candle_core::utils::metal_is_available() {
                Ok(Device::new_metal(0)?)
            } else if candle_core::utils::cuda_is_available() {
                Ok(Device::new_cuda(0)?)
            } else {
                Ok(Device::Cpu)
            }
        }
    }
}

/// Load, resize (bicubic + antialias), and ImageNet-normalize an image into a
/// `[1, 3, h, w]` tensor. Returns the tensor plus the original `(H, W)`.
fn load_image(
    path: &Path,
    max_size: usize,
    patch: usize,
    device: &Device,
) -> Result<(Tensor, (usize, usize))> {
    let img = image::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .into_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let hwc: Vec<f32> = img
        .into_raw()
        .iter()
        .map(|&v| f32::from(v) / 255.0)
        .collect();

    let (nh, nw) = get_img_resolution(h, w, max_size, patch);
    let resized = resize::resize_image_bicubic_aa(&hwc, (h, w, 3), (nh, nw));

    // HWC -> CHW with per-channel normalization.
    let mut chw = vec![0.0f32; 3 * nh * nw];
    for c in 0..3 {
        let (mean, std) = (IMAGENET_MEAN[c], IMAGENET_STD[c]);
        let plane = &mut chw[c * nh * nw..(c + 1) * nh * nw];
        for (i, px) in resized.chunks_exact(3).enumerate() {
            plane[i] = (px[c] - mean) / std;
        }
    }
    Ok((Tensor::from_vec(chw, (1, 3, nh, nw), device)?, (h, w)))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cfg = Config::from_model_name(&args.model)?;
    let image_paths = find_images(&args.data)?;
    if image_paths.is_empty() {
        bail!(
            "No images found in {} or {}/images",
            args.data.display(),
            args.data.display()
        );
    }

    let device = pick_device(args.device.as_deref())?;
    println!(
        "Found {} images, using device {device:?}",
        image_paths.len()
    );

    let t_load = Instant::now();
    let weights_path = match &args.weights {
        Some(p) => p.clone(),
        None => hf_hub::api::sync::Api::new()?
            .model(cfg.hf_repo.to_owned())
            .get("model.safetensors")
            .with_context(|| format!("downloading {} weights", cfg.hf_repo))?,
    };
    // SAFETY: the safetensors file is memory-mapped read-only and not
    // modified while the process runs (standard candle weight loading).
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[&weights_path], DType::F32, &device)?
    };
    let model = DinoVisionTransformer::new(&vb, cfg, &device)?;
    println!("Model loaded in {:.2}s", t_load.elapsed().as_secs_f32());

    let out_dir = args.data.join("dino_features");
    std::fs::create_dir_all(&out_dir)?;

    // Extract per-image features, kept on the host like the script.
    let t_extract = Instant::now();
    let mut feats_per_image: Vec<(usize, usize, Vec<f32>)> = Vec::new();
    let mut image_shape: Option<(usize, usize)> = None;
    for (i, path) in image_paths.iter().enumerate() {
        let (tensor, orig_hw) = load_image(path, args.max_size, cfg.patch_size, &device)?;
        image_shape.get_or_insert(orig_hw);
        let desc = (model.forward_features(&tensor)? / SCALE_DIV)?;
        let (h, w, c) = desc.dims3()?;
        let host = desc.flatten_all()?.to_vec1::<f32>()?;
        feats_per_image.push((h, w, host));
        println!(
            "[{}/{}] {}: features ({h}, {w}, {c})",
            i + 1,
            image_paths.len(),
            path.file_name().unwrap_or_default().to_string_lossy()
        );
    }
    let extract_s = t_extract.elapsed().as_secs_f32();

    // Fit PCA over all images' features.
    let t_pca = Instant::now();
    let feat_dim = cfg.embed_dim;
    let flat: Vec<f32> = feats_per_image
        .iter()
        .flat_map(|(_, _, f)| f.iter().copied())
        .collect();
    println!(
        "Fitting PCA {feat_dim} -> {} on {} rows...",
        args.pca_dim,
        (flat.len() / feat_dim).min(pca::PCA_MAX_ROWS)
    );
    let basis = pca::fit_pca(&flat, feat_dim, args.pca_dim, &device)?;
    drop(flat);
    let pca_s = t_pca.elapsed().as_secs_f32();

    // Project and save per-view maps.
    let t_proj = Instant::now();
    let basis_t = Tensor::from_vec(basis.clone(), (feat_dim, args.pca_dim), &device)?;
    for (path, (h, w, feats)) in image_paths.iter().zip(&feats_per_image) {
        let stem = path.file_stem().unwrap_or_default().to_string_lossy();
        if args.dump_raw {
            npy::write_npy_f32(
                &out_dir.join(format!("{stem}.raw.npy")),
                feats,
                &[*h, *w, feat_dim],
            )?;
        }
        let x = Tensor::from_vec(feats.clone(), (h * w, feat_dim), &device)?;
        let projected = x.matmul(&basis_t)?.flatten_all()?.to_vec1::<f32>()?;
        npy::write_npy_f32(
            &out_dir.join(format!("{stem}.npy")),
            &projected,
            &[*h, *w, args.pca_dim],
        )?;
    }
    let proj_s = t_proj.elapsed().as_secs_f32();

    npy::write_npy_f32(&out_dir.join("pca.npy"), &basis, &[feat_dim, args.pca_dim])?;
    let (orig_h, orig_w) = image_shape.expect("at least one image");
    let meta = serde_json::json!({
        "model": args.model,
        "patch_size": cfg.patch_size,
        "pca_dim": args.pca_dim,
        "scale_div": SCALE_DIV as i64,
        "max_size": args.max_size,
        "image_shape": [orig_h, orig_w],
    });
    std::fs::write(
        out_dir.join("meta.json"),
        serde_json::to_string_pretty(&meta)?,
    )?;

    println!(
        "Wrote {} feature maps, pca.npy, meta.json to {}",
        image_paths.len(),
        out_dir.display()
    );
    println!(
        "Timings: extract {extract_s:.2}s ({:.2}s/image), pca {pca_s:.2}s, project+write {proj_s:.2}s",
        extract_s / image_paths.len() as f32
    );
    Ok(())
}
