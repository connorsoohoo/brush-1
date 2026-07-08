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
use candle_core::{D, Device, IndexOp, Tensor};
use candle_nn::{
    Conv2d, Conv2dConfig, LayerNorm, LayerNormConfig, Linear, Module, VarBuilder, conv2d,
    layer_norm, linear, ops::softmax_last_dim,
};

use crate::resize::resize_grid_bicubic_torch;

const LN_EPS: f64 = 1e-6;
/// `interpolate_offset` in the reference implementation.
const POS_INTERP_OFFSET: f32 = 0.1;

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
        let attn = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * self.scale)?;
        let attn = softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, t, d))?;
        Ok(self.proj.forward(&out)?)
    }
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
        let patch = Tensor::from_vec(patch, (1, h0 * w0, dim), &self.device)?;
        let full = Tensor::cat(&[&self.pos_cls, &patch], 1)?;
        *self.pos_cache.borrow_mut() = Some(((h0, w0), full.clone()));
        Ok(full)
    }

    /// `get_intermediate_layers(x, n=1, reshape=True, norm=True)` for a
    /// single `[1, 3, h, w]` image: returns `[h/14, w/14, dim]`.
    pub fn forward_features(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = xs.dims4()?;
        let (h0, w0) = (h / self.cfg.patch_size, w / self.cfg.patch_size);
        let patches = self
            .patch_proj
            .forward(xs)?
            .flatten_from(2)?
            .transpose(1, 2)?; // [1, h0*w0, dim]
        let mut tokens = Tensor::cat(&[&self.cls_token, &patches], 1)?;
        tokens = tokens.broadcast_add(&self.pos_embed_for(h0, w0)?)?;
        for block in &self.blocks {
            tokens = block.forward(&tokens)?;
        }
        let tokens = self.norm.forward(&tokens)?;
        let features = tokens.i((0, 1.., ..))?; // drop cls -> [h0*w0, dim]
        Ok(features.reshape((h0, w0, self.cfg.embed_dim))?)
    }
}
