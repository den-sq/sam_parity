use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{LayerNorm, Linear, Module};
use candle_transformers::models::sam3::{Config, Sam3CheckpointSource};

fn main() -> Result<()> {
    let device = Device::Cpu;
    let checkpoint_path = resolve_checkpoint_path();
    let actual_path = std::env::var("SAM3_BLOCK7_ACTUAL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/parity_box_positive_debug_b7_report/actual.safetensors"));
    let output_path = std::env::var("SAM3_BLOCK7_PROBE_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/sam3_block7_attention_probe.safetensors"));

    let actual = candle::safetensors::load(&actual_path, &device)?;
    let input = actual
        .get("vision.block_debug.7.input")
        .ok_or_else(|| candle::Error::Msg(format!("missing vision.block_debug.7.input in {}", actual_path.display())))?
        .permute((0, 2, 3, 1))?
        .contiguous()?;

    let config = Config::default();
    let vision = &config.vision;
    let checkpoint = Sam3CheckpointSource::upstream_pth(checkpoint_path);
    let vb = checkpoint.load_var_builder(DType::F32, &device)?;
    let block_vb = vb.pp("backbone.vision_backbone.trunk.blocks.7");

    let norm1 = LayerNorm::new(
        block_vb.pp("norm1").get(vision.embed_dim, "weight")?,
        block_vb.pp("norm1").get(vision.embed_dim, "bias")?,
        1e-5,
    );
    let qkv = Linear::new(
        block_vb
            .pp("attn")
            .pp("qkv")
            .get((vision.embed_dim * 3, vision.embed_dim), "weight")?,
        Some(
            block_vb
                .pp("attn")
                .pp("qkv")
                .get(vision.embed_dim * 3, "bias")?,
        ),
    );
    let proj = Linear::new(
        block_vb
            .pp("attn")
            .pp("proj")
            .get((vision.embed_dim, vision.embed_dim), "weight")?,
        Some(
            block_vb
                .pp("attn")
                .pp("proj")
                .get(vision.embed_dim, "bias")?,
        ),
    );
    let num_heads = vision.num_heads;
    let head_dim = vision.embed_dim / vision.num_heads;
    let rotary = VisionRotaryEmbedding::new(
        vision,
        vision.image_size / vision.patch_size,
        vision.image_size / vision.patch_size,
        vision.rope_pt_size as f32 / (vision.image_size / vision.patch_size) as f32,
        &device,
    )?;

    let norm1_out = norm1.forward(&input)?;
    let (batch_size, height, width, channels) = norm1_out.dims4()?;
    let seq_len = height * width;
    let qkv_out = qkv
        .forward(&norm1_out.contiguous()?)?
        .reshape((batch_size, seq_len, 3, num_heads, head_dim))?
        .permute((2, 0, 3, 1, 4))?
        .contiguous()?;
    let q = qkv_out.i(0)?.contiguous()?;
    let k = qkv_out.i(1)?.contiguous()?;
    let v = qkv_out.i(2)?.contiguous()?;
    let (q_rope, k_rope) = rotary.apply(&q, &k)?;
    let scale = Tensor::new((head_dim as f32).powf(-0.5), &device)?;
    let q_scaled = q_rope.to_dtype(DType::F32)?.broadcast_mul(&scale)?.contiguous()?;
    let k_f32 = k_rope.to_dtype(DType::F32)?.contiguous()?;
    let v_f32 = v.to_dtype(DType::F32)?.contiguous()?;
    let attn_scores = q_scaled.matmul(&k_f32.transpose(2, 3)?)?;
    let attn_probs = candle_nn::ops::softmax_last_dim(&attn_scores)?;
    let context = attn_probs.matmul(&v_f32)?.contiguous()?;
    let attn_probs_generic = candle_nn::ops::softmax(&attn_scores, D::Minus1)?;
    let context_generic = attn_probs_generic.matmul(&v_f32)?.contiguous()?;
    let q_scaled_f64 = q_scaled.to_dtype(DType::F64)?;
    let k_f64 = k_f32.to_dtype(DType::F64)?;
    let v_f64 = v_f32.to_dtype(DType::F64)?;
    let attn_scores_f64 = q_scaled_f64.matmul(&k_f64.transpose(2, 3)?)?;
    let attn_probs_f64 = candle_nn::ops::softmax_last_dim(&attn_scores_f64)?;
    let context_f64 = attn_probs_f64.matmul(&v_f64)?.to_dtype(DType::F32)?.contiguous()?;
    let attn_nhwc = context
        .to_dtype(DType::F32)?
        .transpose(1, 2)?
        .reshape((batch_size, height, width, channels))?
        .contiguous()?;
    let attn_output = proj.forward(&attn_nhwc)?.contiguous()?;
    let attn_nhwc_generic = context_generic
        .to_dtype(DType::F32)?
        .transpose(1, 2)?
        .reshape((batch_size, height, width, channels))?
        .contiguous()?;
    let attn_output_generic = proj.forward(&attn_nhwc_generic)?.contiguous()?;
    let attn_nhwc_f64 = context_f64
        .transpose(1, 2)?
        .reshape((batch_size, height, width, channels))?
        .contiguous()?;
    let attn_output_f64 = proj.forward(&attn_nhwc_f64)?.contiguous()?;

    let hot_h = 19usize;
    let hot_w = 55usize;
    let hot_seq = hot_h * width + hot_w;
    let hot_scores = attn_scores.i((.., .., hot_seq..hot_seq + 1, ..))?.contiguous()?;
    let hot_probs = attn_probs.i((.., .., hot_seq..hot_seq + 1, ..))?.contiguous()?;
    let hot_context = context.i((.., .., hot_seq..hot_seq + 1, ..))?.contiguous()?;
    let focus_seq = 31usize * width + 57usize;
    let focus_scores = attn_scores
        .i((.., .., focus_seq..focus_seq + 1, ..))?
        .contiguous()?;
    let focus_probs = attn_probs
        .i((.., .., focus_seq..focus_seq + 1, ..))?
        .contiguous()?;
    let focus_context = context
        .i((.., .., focus_seq..focus_seq + 1, ..))?
        .contiguous()?;
    let focus_scores_max = focus_scores.max_keepdim(D::Minus1)?.contiguous()?;
    let focus_scores_diff = focus_scores.broadcast_sub(&focus_scores_max)?.contiguous()?;
    let focus_scores_exp = focus_scores_diff.exp()?.contiguous()?;
    let focus_scores_exp_sum = focus_scores_exp.sum_keepdim(D::Minus1)?.contiguous()?;
    let focus_probs_generic = attn_probs_generic
        .i((.., .., focus_seq..focus_seq + 1, ..))?
        .contiguous()?;
    let focus_context_generic = context_generic
        .i((.., .., focus_seq..focus_seq + 1, ..))?
        .contiguous()?;
    let focus_probs_f64 = attn_probs_f64
        .i((.., .., focus_seq..focus_seq + 1, ..))?
        .to_dtype(DType::F32)?
        .contiguous()?;
    let focus_context_f64 = context_f64
        .i((.., .., focus_seq..focus_seq + 1, ..))?
        .contiguous()?;

    let mut tensors = HashMap::new();
    tensors.insert("block7.input".to_owned(), input.permute((0, 3, 1, 2))?);
    tensors.insert("block7.norm1".to_owned(), norm1_out.permute((0, 3, 1, 2))?);
    tensors.insert("block7.attn.q".to_owned(), q);
    tensors.insert("block7.attn.k".to_owned(), k);
    tensors.insert("block7.attn.v".to_owned(), v);
    tensors.insert("block7.attn.q_rope".to_owned(), q_rope);
    tensors.insert("block7.attn.k_rope".to_owned(), k_rope);
    tensors.insert("block7.attn.q_scaled".to_owned(), q_scaled);
    tensors.insert("block7.attn.scores_hot".to_owned(), hot_scores);
    tensors.insert("block7.attn.probs_hot".to_owned(), hot_probs);
    tensors.insert("block7.attn.context_hot".to_owned(), hot_context);
    tensors.insert("block7.attn.scores_focus".to_owned(), focus_scores);
    tensors.insert("block7.attn.scores_focus_max".to_owned(), focus_scores_max);
    tensors.insert("block7.attn.scores_focus_diff".to_owned(), focus_scores_diff);
    tensors.insert("block7.attn.scores_focus_exp".to_owned(), focus_scores_exp);
    tensors.insert("block7.attn.scores_focus_exp_sum".to_owned(), focus_scores_exp_sum);
    tensors.insert("block7.attn.probs_focus".to_owned(), focus_probs);
    tensors.insert("block7.attn.context_focus".to_owned(), focus_context);
    tensors.insert("block7.attn.probs_focus_generic".to_owned(), focus_probs_generic);
    tensors.insert("block7.attn.context_focus_generic".to_owned(), focus_context_generic);
    tensors.insert("block7.attn.probs_focus_f64".to_owned(), focus_probs_f64);
    tensors.insert("block7.attn.context_focus_f64".to_owned(), focus_context_f64);
    tensors.insert("block7.attn.context".to_owned(), context);
    tensors.insert("block7.attn.context_generic".to_owned(), context_generic);
    tensors.insert("block7.attn.context_f64".to_owned(), context_f64);
    tensors.insert("block7.attn.output".to_owned(), attn_output.permute((0, 3, 1, 2))?);
    tensors.insert(
        "block7.attn.output_generic".to_owned(),
        attn_output_generic.permute((0, 3, 1, 2))?,
    );
    tensors.insert(
        "block7.attn.output_f64".to_owned(),
        attn_output_f64.permute((0, 3, 1, 2))?,
    );
    candle::safetensors::save(&tensors, &output_path)?;

    println!("saved block 7 attention probe to {}", output_path.display());
    Ok(())
}

fn resolve_checkpoint_path() -> PathBuf {
    let base = std::env::var("SAM3_TEST_CHECKPOINT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/home/dnorthover/extcode/hf_sam3"));
    if base.is_dir() {
        base.join("sam3.pt")
    } else {
        base
    }
}

struct VisionRotaryEmbedding {
    freqs_real: Tensor,
    freqs_imag: Tensor,
}

impl VisionRotaryEmbedding {
    fn new(
        config: &candle_transformers::models::sam3::VisionConfig,
        end_x: usize,
        end_y: usize,
        scale: f32,
        device: &Device,
    ) -> Result<Self> {
        let head_dim = config.embed_dim / config.num_heads;
        let rotary_dim = head_dim / 4;
        let seq_len = end_x * end_y;
        let inv_freqs: Vec<f32> = (0..rotary_dim)
            .map(|i| 1f32 / (config.rope_theta as f32).powf((4 * i) as f32 / head_dim as f32))
            .collect();
        let mut freqs_real = vec![0f32; seq_len * (head_dim / 2)];
        let mut freqs_imag = vec![0f32; seq_len * (head_dim / 2)];
        for flat_idx in 0..seq_len {
            let x_pos = (flat_idx % end_x) as f32 * scale;
            let y_pos = (flat_idx / end_x) as f32 * scale;
            let row_real =
                &mut freqs_real[flat_idx * (head_dim / 2)..(flat_idx + 1) * (head_dim / 2)];
            let row_imag =
                &mut freqs_imag[flat_idx * (head_dim / 2)..(flat_idx + 1) * (head_dim / 2)];
            for (i, inv_freq) in inv_freqs.iter().copied().enumerate() {
                let x_freq = x_pos * inv_freq;
                let y_freq = y_pos * inv_freq;
                row_real[i] = x_freq.cos();
                row_imag[i] = x_freq.sin();
                row_real[rotary_dim + i] = y_freq.cos();
                row_imag[rotary_dim + i] = y_freq.sin();
            }
        }
        Ok(Self {
            freqs_real: Tensor::from_vec(freqs_real, (seq_len, head_dim / 2), device)?,
            freqs_imag: Tensor::from_vec(freqs_imag, (seq_len, head_dim / 2), device)?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, head_dim) = q.dims4()?;
        let freqs_real = self
            .freqs_real
            .narrow(0, 0, seq_len)?
            .reshape((1, 1, seq_len, head_dim / 2))?;
        let freqs_imag = self
            .freqs_imag
            .narrow(0, 0, seq_len)?
            .reshape((1, 1, seq_len, head_dim / 2))?;
        Ok((
            apply_rotary_enc_real(q, &freqs_real, &freqs_imag)?,
            apply_rotary_enc_real(k, &freqs_real, &freqs_imag)?,
        ))
    }
}

fn apply_rotary_enc_real(xs: &Tensor, freqs_real: &Tensor, freqs_imag: &Tensor) -> Result<Tensor> {
    let (batch_size, num_heads, seq_len, head_dim) = xs.dims4()?;
    let xs_dtype = xs.dtype();
    let xs = xs
        .to_dtype(DType::F32)?
        .reshape((batch_size, num_heads, seq_len, head_dim / 2, 2))?;
    let xs_real = xs.i((.., .., .., .., 0))?;
    let xs_imag = xs.i((.., .., .., .., 1))?;
    let real = (xs_real.broadcast_mul(freqs_real)? - xs_imag.broadcast_mul(freqs_imag)?)?;
    let imag = (xs_real.broadcast_mul(freqs_imag)? + xs_imag.broadcast_mul(freqs_real)?)?;
    Tensor::stack(&[&real, &imag], 4)?
        .reshape((batch_size, num_heads, seq_len, head_dim))?
        .to_dtype(xs_dtype)
}

#[allow(dead_code)]
fn _display_path(path: &Path) -> String {
    path.display().to_string()
}
