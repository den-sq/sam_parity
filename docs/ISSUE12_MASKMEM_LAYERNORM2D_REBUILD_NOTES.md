# Issue #12 Rebuild Notes: Mask-Memory LayerNorm2d Fast Path

These notes capture the optimization pass that was prototyped for issue #12, and the safer rebuild path that avoids changing shared transformer/SAM behavior.

## Why The Prototype Should Be Discarded

The prototype changed the shared `segment_anything::LayerNorm2d` implementation in:

`/home/dnorthover/ChengCode/candle_sam3/candle-transformers/src/models/segment_anything/mod.rs`

That type is used outside the SAM3 mask-memory downsampler path. Even though the CUDA fused result matched numerically in the tested paths, changing the behavior of that shared module is too broad for issue #12. The rebuild should leave that existing function untouched and introduce a SAM3-local wrapper instead.

## What The Prototype Added

The prototype added a fused CUDA channels-first LayerNorm2d over the `C` dimension for contiguous `NCHW` tensors:

- `candle-kernels/src/reduce.cu`
  - Added `layernorm2d` device helper.
  - Added `LAYERNORM2D_OP`.
  - Exported `layernorm2d_bf16`, `layernorm2d_f16`, `layernorm2d_f32`, and `layernorm2d_f64`.

- `candle-nn/src/ops.rs`
  - Added a new `LayerNorm2d` `CustomOp3`.
  - Added CPU reference implementation.
  - Added CUDA forward path loading `kernel_name::<T>("layernorm2d")` from `kernels::REDUCE`.
  - Added public API:

```rust
pub fn layer_norm_2d(xs: &Tensor, alpha: &Tensor, beta: &Tensor, eps: f32) -> Result<Tensor>
```

- `candle-nn/tests/ops.rs`
  - Added CPU and CUDA tests comparing `layer_norm_2d` to the old tensor-op expression.

- `candle-nn/benches/benchmarks/norm.rs`
  - Added focused SAM3-shaped benchmarks comparing fused `layer_norm_2d` with the old tensor-op chain.
  - Added `CANDLE_NN_BENCH_SAM3_LAYER_NORM_2D_ONLY=1` to skip the older norm benches when collecting only this signal.

- `candle-transformers/src/models/sam3/tracker/maskmem_backbone.rs`
  - Removed one local, unnecessary `.contiguous()` after permuting fuser output back to `NCHW`.

- Non-starter part:
  - Modified shared `segment_anything::LayerNorm2d` to call the fused op on CUDA and cache reshaped affine tensors. Do not rebuild this part.

## Benchmark Signal From The Prototype

Command shape:

```bash
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
export CARGO_TARGET_DIR=/tmp/candle_sam3_issue12_cuda_target
export CANDLE_NN_BENCH_SAM3_LAYER_NORM_2D_ONLY=1
cargo bench --manifest-path /home/dnorthover/ChengCode/candle_sam3/Cargo.toml \
  -p candle-nn --features cuda --bench bench_main -- sam3_layer_norm_2d
```

Observed Criterion medians:

| Shape | Fused | Old tensor ops | Speedup |
| --- | ---: | ---: | ---: |
| `4x128x128` | `33.99 us` | `289.76 us` | `~8.5x` |
| `16x64x64` | `18.33 us` | `292.47 us` | `~16.0x` |
| `64x32x32` | `18.01 us` | `283.71 us` | `~15.8x` |
| `256x16x16` | `19.18 us` | `315.29 us` | `~16.4x` |

Benchmark outputs landed under:

`/tmp/candle_sam3_issue12_cuda_target/criterion/cuda_sam3_layer_norm_2d_*`

## Safer Rebuild Plan

1. Keep the low-level fused op as a new Candle API.

   Adding a new `candle_nn::ops::layer_norm_2d` function is fine because it does not alter any existing public function. It should remain opt-in.

2. Do not modify `segment_anything::LayerNorm2d`.

   Leave `/home/dnorthover/ChengCode/candle_sam3/candle-transformers/src/models/segment_anything/mod.rs` exactly as upstream/local baseline has it.

3. Add a SAM3-local wrapper in the tracker mask-memory module.

   Put the wrapper in `candle-transformers/src/models/sam3/tracker/maskmem_backbone.rs` or a new SAM3 tracker-local file. It should own the same checkpoint parameters (`weight`, `bias`) and use the same variable names, but it should only be referenced by SAM3 mask-memory code.

```rust
#[derive(Debug)]
struct TrackerMaskmemLayerNorm2d {
    weight: Tensor,
    bias: Tensor,
    weight_4d: Tensor,
    bias_4d: Tensor,
    eps: f64,
}

impl TrackerMaskmemLayerNorm2d {
    fn new(num_channels: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(num_channels, "weight")?;
        let bias = vb.get(num_channels, "bias")?;
        let weight_4d = weight.reshape((1, num_channels, 1, 1))?;
        let bias_4d = bias.reshape((1, num_channels, 1, 1))?;
        Ok(Self { weight, bias, weight_4d, bias_4d, eps })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if xs.is_contiguous() && xs.device().is_cuda() {
            return candle_nn::ops::layer_norm_2d(xs, &self.weight, &self.bias, self.eps as f32);
        }

        let mean = xs.mean_keepdim(1)?;
        let centered = xs.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(1)?;
        let normed = centered.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        normed.broadcast_mul(&self.weight_4d)?.broadcast_add(&self.bias_4d)
    }
}
```

4. Use the wrapper only in `TrackerSimpleMaskDownSampler`.

   Replace:

```rust
norms: Vec<LayerNorm2d>,
```

   with:

```rust
norms: Vec<TrackerMaskmemLayerNorm2d>,
```

   And construct it with the same `encoder_vb.pp(layer_idx * 3 + 1)` path, preserving checkpoint compatibility.

5. Keep the fuser `.contiguous()` removal if tests still pass.

   This is SAM3-local and does not affect shared functions:

```rust
xs = xs.permute((0, 3, 1, 2))?;
residual.broadcast_add(&xs)
```

6. Gate CUDA only at the SAM3 call site.

   The direct CPU implementation in `candle_nn::ops::layer_norm_2d` is useful for correctness tests, but the SAM3 wrapper should use the fused op only on CUDA contiguous tensors. The old tensor-op chain was better for normal CPU parity tests.

## Validation Commands To Re-run

```bash
cargo check -p sam3-parity-cli --features full-parity
cargo test -p sam3-parity-cli --features full-parity \
  tracker_track_frame_matches_single_click_point_fixture_values -- --nocapture
```

CUDA op tests:

```bash
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
CARGO_TARGET_DIR=/tmp/candle_sam3_issue12_cuda_target \
cargo test --manifest-path /home/dnorthover/ChengCode/candle_sam3/Cargo.toml \
  -p candle-nn --features cuda ln2d_gpu -- --nocapture
```

CUDA SAM3 smoke:

```bash
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
export SAM3_TEST_CHECKPOINT_DIR=/home/dnorthover/extcode/hf_sam3
CARGO_TARGET_DIR=/tmp/candle_sam3_issue12_cuda_target \
cargo test --release -p sam3-parity-cli --features full-parity,cuda \
  video_propagation_matches_text_prompt_suppressed_reference_bundle_cuda \
  -- --ignored --nocapture --test-threads=1
```

## Files To Rebuild

Safe to rebuild:

- `/home/dnorthover/ChengCode/candle_sam3/candle-kernels/src/reduce.cu`
- `/home/dnorthover/ChengCode/candle_sam3/candle-nn/src/ops.rs`
- `/home/dnorthover/ChengCode/candle_sam3/candle-nn/tests/ops.rs`
- `/home/dnorthover/ChengCode/candle_sam3/candle-nn/benches/benchmarks/norm.rs`
- `/home/dnorthover/ChengCode/candle_sam3/candle-transformers/src/models/sam3/tracker/maskmem_backbone.rs`

Do not rebuild:

- `/home/dnorthover/ChengCode/candle_sam3/candle-transformers/src/models/segment_anything/mod.rs`

