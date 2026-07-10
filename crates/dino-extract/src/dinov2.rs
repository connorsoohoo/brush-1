//! `DINOv2` backbone on candle, loading Hugging Face `facebook/dinov2-*`
//! safetensors directly (HF `Dinov2Model` tensor naming — no weight
//! conversion step).
//!
//! candle-transformers ships a `dinov2`, but it interpolates positional
//! embeddings with `upsample_nearest2d` (the reference uses bicubic with a
//! `+0.1` scale offset), hardcodes an `ImageNet` head absent from backbone
//! checkpoints, and only constructs ViT-S — so the model is implemented here,
//! mirroring `facebookresearch/dinov2` `vision_transformer.py` semantics.

use std::cell::RefCell;

use anyhow::{Result, bail};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{
    Conv2d, Conv2dConfig, LayerNorm, LayerNormConfig, Linear, Module, VarBuilder, conv2d,
    layer_norm, linear, ops::softmax_last_dim,
};

use crate::resize::resize_grid_bicubic_torch;

const LN_EPS: f64 = 1e-6;
/// `interpolate_offset` in the reference implementation.
const POS_INTERP_OFFSET: f32 = 0.1;
/// Query rows per chunk in `attend_chunked`. At T≈6k tokens the full score
/// tensor is `[heads, T, T]` (~1.7 GB f32 for ViT-B); chunking caps it at
/// `[heads, CHUNK, T]`.
const ATTN_CHUNK: usize = 512;

#[derive(Clone, Copy)]
pub struct Config {
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub patch_size: usize,
    pub hf_repo: &'static str,
}

impl Config {
    /// From a `torch.hub` model name as taken by the Python script.
    pub fn from_model_name(name: &str) -> Result<Self> {
        let (embed_dim, depth, num_heads, hf_repo) = match name {
            "dinov2_vits14" => (384, 12, 6, "facebook/dinov2-small"),
            "dinov2_vitb14" => (768, 12, 12, "facebook/dinov2-base"),
            "dinov2_vitl14" => (1024, 24, 16, "facebook/dinov2-large"),
            _ => bail!(
                "unsupported model '{name}' (supported: dinov2_vits14, dinov2_vitb14, dinov2_vitl14)"
            ),
        };
        Ok(Self {
            embed_dim,
            depth,
            num_heads,
            patch_size: 14,
            hf_repo,
        })
    }
}

struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    proj: Linear,
    num_heads: usize,
    scale: f64,
}

impl Attention {
    /// `vb` is at the `encoder.layer.N.attention` prefix.
    fn new(vb: &VarBuilder, dim: usize, num_heads: usize) -> Result<Self> {
        Ok(Self {
            query: linear(dim, dim, vb.pp("attention.query"))?,
            key: linear(dim, dim, vb.pp("attention.key"))?,
            value: linear(dim, dim, vb.pp("attention.value"))?,
            proj: linear(dim, dim, vb.pp("output.dense"))?,
            num_heads,
            scale: 1.0 / ((dim / num_heads) as f64).sqrt(),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, d) = xs.dims3()?;
        let heads = self.num_heads;
        let split = |t_: Tensor| -> candle_core::Result<Tensor> {
            t_.reshape((b, t, heads, d / heads))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = split(self.query.forward(xs)?)?;
        let k = split(self.key.forward(xs)?)?;
        let v = split(self.value.forward(xs)?)?;
        let out = if q.device().is_metal() {
            candle_nn::ops::sdpa(&q, &k, &v, None, false, self.scale as f32, 1.0)?
        } else {
            attend_chunked(&q, &k, &v, self.scale)?
        };
        let out = out.transpose(1, 2)?.reshape((b, t, d))?;
        Ok(self.proj.forward(&out)?)
    }
}

/// Attention in query chunks: never materializes more than
/// `[b, heads, ATTN_CHUNK, kv]` of scores. Mathematically identical to full
/// attention — softmax rows are independent.
fn attend_chunked(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    let t = q.dim(2)?;
    let kt = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
    let mut outs = Vec::with_capacity(t.div_ceil(ATTN_CHUNK));
    let mut start = 0;
    while start < t {
        let len = ATTN_CHUNK.min(t - start);
        let attn = softmax_last_dim(&(q.narrow(2, start, len)?.matmul(&kt)? * scale)?)?;
        outs.push(attn.matmul(v)?);
        start += len;
    }
    Ok(Tensor::cat(&outs, 2)?)
}

/// Reference attention (the originally Python-parity-validated path):
/// materializes the full `[b, heads, T, T]` score tensor. Kept as the
/// baseline the fused/chunked paths are unit-tested against.
#[cfg(test)]
fn attend_naive(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    let attn = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
    Ok(softmax_last_dim(&attn)?.matmul(v)?)
}

struct Block {
    norm1: LayerNorm,
    attn: Attention,
    ls1: Tensor,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    ls2: Tensor,
}

impl Block {
    fn new(vb: &VarBuilder, dim: usize, num_heads: usize) -> Result<Self> {
        let ln_cfg = LayerNormConfig {
            eps: LN_EPS,
            ..Default::default()
        };
        Ok(Self {
            norm1: layer_norm(dim, ln_cfg, vb.pp("norm1"))?,
            attn: Attention::new(&vb.pp("attention"), dim, num_heads)?,
            ls1: vb.get(dim, "layer_scale1.lambda1")?,
            norm2: layer_norm(dim, ln_cfg, vb.pp("norm2"))?,
            fc1: linear(dim, dim * 4, vb.pp("mlp.fc1"))?,
            fc2: linear(dim * 4, dim, vb.pp("mlp.fc2"))?,
            ls2: vb.get(dim, "layer_scale2.lambda1")?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let attn_out = self.attn.forward(&self.norm1.forward(xs)?)?;
        let xs = (xs + attn_out.broadcast_mul(&self.ls1)?)?;
        let mlp_out = self
            .fc2
            .forward(&self.fc1.forward(&self.norm2.forward(&xs)?)?.gelu_erf()?)?;
        Ok((&xs + mlp_out.broadcast_mul(&self.ls2)?)?)
    }
}

pub struct DinoVisionTransformer {
    patch_proj: Conv2d,
    cls_token: Tensor,
    /// Patch part of the positional embedding as a CPU-side `[m, m, dim]`
    /// row-major grid, pre-extracted for bicubic interpolation.
    pos_patch_grid: Vec<f32>,
    pos_cls: Tensor,
    pos_grid_side: usize,
    blocks: Vec<Block>,
    norm: LayerNorm,
    cfg: Config,
    device: Device,
    /// Interpolated pos-embed cache keyed on the patch-grid size — datasets
    /// have uniform resolution, so this is computed once.
    pos_cache: RefCell<Option<((usize, usize), Tensor)>>,
}

impl DinoVisionTransformer {
    pub fn new(vb: &VarBuilder, cfg: Config, device: &Device) -> Result<Self> {
        let dim = cfg.embed_dim;
        let emb = vb.pp("embeddings");
        let patch_proj = conv2d(
            3,
            dim,
            cfg.patch_size,
            Conv2dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            },
            emb.pp("patch_embeddings.projection"),
        )?;
        let cls_token = emb.get((1, 1, dim), "cls_token")?;
        // All dinov2 checkpoints are pretrained at 518px -> a 37x37 patch grid.
        let side = 518 / cfg.patch_size;
        let pos_embed = emb.get((1, side * side + 1, dim), "position_embeddings")?;
        let pos_cls = pos_embed.i((.., ..1, ..))?.contiguous()?;
        let pos_patch_grid = pos_embed
            .i((0, 1.., ..))?
            .contiguous()?
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let layers = vb.pp("encoder.layer");
        let blocks = (0..cfg.depth)
            .map(|i| Block::new(&layers.pp(i), dim, cfg.num_heads))
            .collect::<Result<Vec<_>>>()?;
        let norm = layer_norm(
            dim,
            LayerNormConfig {
                eps: LN_EPS,
                ..Default::default()
            },
            vb.pp("layernorm"),
        )?;
        Ok(Self {
            patch_proj,
            cls_token,
            pos_patch_grid,
            pos_cls,
            pos_grid_side: side,
            blocks,
            norm,
            cfg,
            device: device.clone(),
            pos_cache: RefCell::new(None),
        })
    }

    /// Reference `interpolate_pos_encoding`: bicubic (`a=-0.75`, no
    /// antialias) with `scale_factor = (n + interpolate_offset) / m`.
    fn pos_embed_for(&self, h0: usize, w0: usize) -> Result<Tensor> {
        if let Some((key, cached)) = self.pos_cache.borrow().as_ref()
            && *key == (h0, w0)
        {
            return Ok(cached.clone());
        }
        let m = self.pos_grid_side;
        let dim = self.cfg.embed_dim;
        let patch = if (h0, w0) == (m, m) {
            self.pos_patch_grid.clone()
        } else {
            let sy = (h0 as f32 + POS_INTERP_OFFSET) / m as f32;
            let sx = (w0 as f32 + POS_INTERP_OFFSET) / m as f32;
            resize_grid_bicubic_torch(&self.pos_patch_grid, (m, m, dim), (h0, w0), (sy, sx))
        };
        let patch = Tensor::from_vec(patch, (1, h0 * w0, dim), &self.device)?
            .to_dtype(self.pos_cls.dtype())?;
        let full = Tensor::cat(&[&self.pos_cls, &patch], 1)?;
        *self.pos_cache.borrow_mut() = Some(((h0, w0), full.clone()));
        Ok(full)
    }

    /// `get_intermediate_layers(x, n=1, reshape=True, norm=True)` for a
    /// single `[1, 3, h, w]` f32 image: returns `[h/14, w/14, dim]` f32.
    /// Compute runs at the model's load dtype (f32 or f16).
    pub fn forward_features(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = xs.dims4()?;
        let (h0, w0) = (h / self.cfg.patch_size, w / self.cfg.patch_size);
        let xs = xs.to_dtype(self.cls_token.dtype())?;
        let patches = self
            .patch_proj
            .forward(&xs)?
            .flatten_from(2)?
            .transpose(1, 2)?; // [1, h0*w0, dim]
        let mut tokens = Tensor::cat(&[&self.cls_token, &patches], 1)?;
        tokens = tokens.broadcast_add(&self.pos_embed_for(h0, w0)?)?;
        for block in &self.blocks {
            tokens = block.forward(&tokens)?;
        }
        let tokens = self.norm.forward(&tokens)?;
        let features = tokens.i((0, 1.., ..))?.to_dtype(DType::F32)?; // drop cls -> [h0*w0, dim]
        Ok(features.reshape((h0, w0, self.cfg.embed_dim))?)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    /// Deterministic pseudo-random values in roughly [-scale, scale].
    fn pseudo_rand(n: usize, scale: f32, seed: u32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state as f32 / u32::MAX as f32 - 0.5) * 2.0 * scale
            })
            .collect()
    }

    fn rand_tensor(shape: &[usize], scale: f32, seed: u32, device: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor::from_vec(pseudo_rand(n, scale, seed), shape, device).expect("tensor")
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        a.iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// Spans multiple chunks plus a remainder (700 = 512 + 188).
    #[test]
    fn chunked_attention_matches_naive() {
        let dev = Device::Cpu;
        let (b, heads, t, hd) = (1, 2, 700, 16);
        let q = rand_tensor(&[b, heads, t, hd], 1.0, 1, &dev);
        let k = rand_tensor(&[b, heads, t, hd], 1.0, 2, &dev);
        let v = rand_tensor(&[b, heads, t, hd], 1.0, 3, &dev);
        let scale = 1.0 / (hd as f64).sqrt();
        let naive = attend_naive(&q, &k, &v, scale).expect("naive");
        let chunked = attend_chunked(&q, &k, &v, scale).expect("chunked");
        let diff = max_abs_diff(&naive, &chunked);
        assert!(diff < 1e-5, "chunked vs naive max abs diff {diff}");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn sdpa_attention_matches_naive_on_metal() {
        if !candle_core::utils::metal_is_available() {
            return;
        }
        let dev = Device::new_metal(0).expect("metal device");
        // Head dim 64 as in all supported DINOv2 variants (SDPA requirement).
        let (b, heads, t, hd) = (1, 2, 300, 64);
        let q = rand_tensor(&[b, heads, t, hd], 1.0, 4, &dev);
        let k = rand_tensor(&[b, heads, t, hd], 1.0, 5, &dev);
        let v = rand_tensor(&[b, heads, t, hd], 1.0, 6, &dev);
        let scale = 1.0 / (hd as f64).sqrt();
        let naive = attend_naive(&q, &k, &v, scale).expect("naive");
        let sdpa = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.0).expect("sdpa");
        let diff = max_abs_diff(&naive, &sdpa);
        assert!(diff < 1e-4, "sdpa vs naive max abs diff {diff}");
    }

    /// Random-weight model small enough to run the full forward in
    /// milliseconds: dim 32, 2 heads, 2 blocks. Norm weights sit near 1 so
    /// activations keep a healthy scale.
    fn tiny_model_weights(dim: usize, depth: usize, device: &Device) -> HashMap<String, Tensor> {
        let mut seed = 100;
        let mut next = |shape: &[usize], scale: f32, offset: f64| {
            seed += 1;
            rand_tensor(shape, scale, seed, device)
                .affine(1.0, offset)
                .expect("affine")
        };
        let mut map = HashMap::new();
        let mut entries: Vec<(String, Tensor)> = vec![
            (
                "embeddings.patch_embeddings.projection.weight".to_owned(),
                next(&[dim, 3, 14, 14], 0.05, 0.0),
            ),
            (
                "embeddings.patch_embeddings.projection.bias".to_owned(),
                next(&[dim], 0.05, 0.0),
            ),
            (
                "embeddings.cls_token".to_owned(),
                next(&[1, 1, dim], 0.05, 0.0),
            ),
            (
                "embeddings.position_embeddings".to_owned(),
                next(&[1, (518 / 14) * (518 / 14) + 1, dim], 0.05, 0.0),
            ),
            ("layernorm.weight".to_owned(), next(&[dim], 0.05, 1.0)),
            ("layernorm.bias".to_owned(), next(&[dim], 0.05, 0.0)),
        ];
        for i in 0..depth {
            let p = format!("encoder.layer.{i}");
            for ln in ["norm1", "norm2"] {
                entries.push((format!("{p}.{ln}.weight"), next(&[dim], 0.05, 1.0)));
                entries.push((format!("{p}.{ln}.bias"), next(&[dim], 0.05, 0.0)));
            }
            for qkv in ["query", "key", "value"] {
                entries.push((
                    format!("{p}.attention.attention.{qkv}.weight"),
                    next(&[dim, dim], 0.1, 0.0),
                ));
                entries.push((
                    format!("{p}.attention.attention.{qkv}.bias"),
                    next(&[dim], 0.05, 0.0),
                ));
            }
            entries.push((
                format!("{p}.attention.output.dense.weight"),
                next(&[dim, dim], 0.1, 0.0),
            ));
            entries.push((
                format!("{p}.attention.output.dense.bias"),
                next(&[dim], 0.05, 0.0),
            ));
            entries.push((format!("{p}.layer_scale1.lambda1"), next(&[dim], 0.5, 0.0)));
            entries.push((format!("{p}.layer_scale2.lambda1"), next(&[dim], 0.5, 0.0)));
            entries.push((
                format!("{p}.mlp.fc1.weight"),
                next(&[dim * 4, dim], 0.1, 0.0),
            ));
            entries.push((format!("{p}.mlp.fc1.bias"), next(&[dim * 4], 0.05, 0.0)));
            entries.push((
                format!("{p}.mlp.fc2.weight"),
                next(&[dim, dim * 4], 0.1, 0.0),
            ));
            entries.push((format!("{p}.mlp.fc2.bias"), next(&[dim], 0.05, 0.0)));
        }
        map.extend(entries);
        map
    }

    fn tiny_forward(dtype: DType, device: &Device) -> Tensor {
        let cfg = Config {
            embed_dim: 32,
            depth: 2,
            num_heads: 2,
            patch_size: 14,
            hf_repo: "",
        };
        let weights = tiny_model_weights(cfg.embed_dim, cfg.depth, device);
        let vb = VarBuilder::from_tensors(weights, dtype, device);
        let model = DinoVisionTransformer::new(&vb, cfg, device).expect("model");
        // 2x3 patch grid; deterministic input shared across dtypes.
        let img = rand_tensor(&[1, 3, 28, 42], 1.0, 7, device);
        model.forward_features(&img).expect("forward")
    }

    /// End-to-end f16 vs f32 parity on a tiny random-weight model — the fast
    /// stand-in for the full extractor A/B when validating dtype changes.
    #[test]
    fn f16_forward_close_to_f32() {
        let dev = Device::Cpu;
        let f32_out = tiny_forward(DType::F32, &dev)
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let f16_out = tiny_forward(DType::F16, &dev)
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let dot: f32 = f32_out.iter().zip(&f16_out).map(|(a, b)| a * b).sum();
        let na = f32_out.iter().map(|a| a * a).sum::<f32>().sqrt();
        let nb = f16_out.iter().map(|b| b * b).sum::<f32>().sqrt();
        let cosine = dot / (na * nb);
        assert!(cosine > 0.999, "f16 vs f32 cosine {cosine}");
    }

    /// The f32 forward itself must be deterministic and finite.
    #[test]
    fn forward_output_is_finite() {
        let out = tiny_forward(DType::F32, &Device::Cpu);
        assert_eq!(out.dims(), [2, 3, 32]);
        let vals = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()));
    }
}
