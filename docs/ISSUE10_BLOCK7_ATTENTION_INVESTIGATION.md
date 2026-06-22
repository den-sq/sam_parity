# Issue 10 Block 7 Attention Investigation

Updated: 2026-05-15

## Scope

GitHub issue `#10` is `Localize and fix the first vision trunk drift at block 8/9`.

This note records the current root-cause investigation state for the first failing
vision-trunk branch on CPU parity.

## Current Repro

The first issue-10 parity threshold breach is still the block-8 branch output:

- `vision.block_debug.8.output`: about `1.373291e-04`

Related baseline branch maxima:

- `vision.block_debug.7.output`: about `8.344650e-05`
- `vision.block_debug.8.input`: about `8.392334e-05`
- `vision.block_debug.8.attn_output`: about `2.908707e-05`
- `vision.block_debug.8.mlp_output`: about `8.630753e-05`

The later failing coordinate at the branch level is `(batch=0, channel=679, y=19, x=55)`.
That coordinate is not the dominant error location at block 0. It becomes the dominant
location only by block 7 and crosses the issue tolerance at block 8.

## What Has Been Ruled Out

These paths were investigated and are not the primary cause of the issue-10 split:

- image preprocessing
  - the parity runner consumes `inputs.image` directly from the reference bundle
- patch-embed convolution implementation choice
  - `Direct` and `FullIm2Col` variants changed the seed slightly but made downstream parity worse
- patch-embed as the direct source of the later failing coordinate
  - for the eventual branch-hot coordinate `(0, 679, 19, 55)`, the pre-block drift is tiny and
    is effectively cancelled back to `0.0` at `block0.output`
- block-7 / block-8 MLP as the first causal layer
  - given the same `post_attn`, Rust and upstream `norm2 -> fc1 -> gelu -> fc2 -> output`
    stay within a few `1e-06`
- block-8 as a large same-input implementation bug
  - replaying block 8 on the exact saved `block8.input` keeps `block8.output` to about `2.29e-05`,
    far below the branch-level failure at about `1.37e-04`
- RoPE generation / application as the first major split
  - same-input `q/k/v` and RoPE tensors stay within about `5e-06` to `8e-06`
- checkpoint loading or qkv/proj weight mismatch
  - on the focused same-input query, upstream reproduces Candle's attention-score row exactly
    from Candle's own saved `q` and `k`
- simple thread-count effects for the value reduction
  - `RAYON_NUM_THREADS=1` did not remove the residual `probs @ v` gap
- value-side `probs @ v` as the dominant attention bug
  - using Candle's own saved probability row and value tensor, Candle vs PyTorch differs by only
    about `2.86e-06` on the focused context row
- a full-attention `f64` runtime patch as a viable fix
  - the microprobe improved, but branch-level `block7.output` and `block8.output` got slightly worse

## Method

The current investigation uses two layers of focused replay:

1. Vision-block replay hooks in `candle_sam3`
   - `Sam3ViTDetTrunk::forward_block_with_debug_from_hidden_states`
   - ignored tests in `image.rs` that replay blocks 7 and 8 from captured hidden states
2. A local probe binary in `sam_parity`
   - `rust/sam3-parity-cli/src/bin/block7_attention_probe.rs`
   - exports Candle block-7 attention internals from the exact captured `block7.input`

That setup allows same-input Rust vs upstream comparisons without rerunning the whole
image stack for each hypothesis.

## Same-Input Block 7 Result

On the exact same captured `vision.block_debug.7.input`, Rust and upstream still differ.

Key same-input stage diffs:

- `norm1`: about `1.144409e-05`
- `attn_output`: about `4.482269e-05`
- `post_attn`: about `4.577637e-05`
- `output`: about `4.577637e-05`

Given Rust `post_attn`, the remaining MLP-side same-input gaps collapse:

- `norm2`: about `6.675720e-06`
- `mlp_fc1`: about `5.722046e-06`
- `mlp_output`: about `3.099442e-06`
- `output` with Rust `norm2`: about `3.814697e-06`

This localizes the meaningful same-input split to the attention sublayer.

## Same-Input Block 8 Result

Block 8 is a windowed-attention block, so it was replayed with the correct `24 x 24`
window partitioning. On the exact same captured `vision.block_debug.8.input`, the
same-input gaps are much smaller than the branch-level issue-10 failure:

- `norm1`: about `7.629395e-06`
- `attn_output`: about `7.629395e-06`
- `post_attn`: about `7.629395e-06`
- `norm2`: about `6.675720e-06`
- `mlp_output`: about `2.318621e-05`
- `output`: about `2.288818e-05`

This means block 8 is not introducing a large new same-input implementation mismatch.
It is mostly amplifying the carried block-7 state error until the issue tolerance is crossed.

## Attention Decomposition

The worst same-input attention-context split occurs at:

- head `4`
- query index `2289`
- spatial query `(y=31, x=57)`
- channel `39`

For the attention path:

- `q`: about `7.629395e-06`
- `k`: about `7.629395e-06`
- `v`: about `5.245209e-06`
- `q_rope`: about `7.629395e-06`
- `k_rope`: about `8.583069e-06`
- `q_scaled`: about `9.536743e-07`
- full `context`: about `5.769730e-05`
- projected `attn_output`: about `4.482269e-05`

This means the split is not being born in checkpoint loading, qkv projection, or RoPE.
The first large amplification happens after scores are formed.

## Earlier Small Differences

There are still smaller same-input deltas before the large softmax amplification:

- `norm1`: about `1.14e-05`
- `q/k`: about `7.63e-06`
- `v`: about `5.25e-06`
- `q_rope/k_rope`: about `7.63e-06` to `8.58e-06`

The current best interpretation is:

- `norm1` is likely the same family of issue as softmax: CPU `f32` reduction-order drift
  - Candle layer norm computes row-wise `sum` and `sum2` explicitly in `f32`
- `q/k/v` are likely inherited from those earlier numeric differences plus ordinary GEMM
  accumulation order
- RoPE does not currently look like an independent source
  - the RoPE-sized deltas stay at roughly the same scale as the incoming `q/k` deltas

So these earlier differences may be related numerically, but they are not yet the first place where
the same-input error becomes large enough to explain the issue-10 threshold breach by themselves.

## Softmax Breakdown

Using Candle's own saved focus-query score row, the softmax stages compare to PyTorch as follows:

- `max(scores)`: exact
- `scores - max(scores)`: exact
- `exp(scores - max(scores))`: about `5.960464e-08`
- `sum(exp(...))`: about `2.441406e-04`
- normalized probability row: about `1.531839e-05`

This is the first exact runtime layer where the issue becomes materially larger.

Using strict scalar `f32` loops on the saved focus-query exponentials:

- Candle `sum_exp` matches scalar `f32` accumulation exactly across all heads
- PyTorch / NumPy `sum_exp` differ from that scalar result by as much as `2.441406e-04`

So the primary parity split is now narrower than "Candle softmax is wrong". The current
default Candle CPU build is reproducing scalar left-to-right `f32` accumulation, while
upstream CPU softmax is effectively using a different reduction order.

## Value Reduction Breakdown

Using Candle's own saved probability row and value tensor:

- `context_focus` from Candle vs PyTorch `probs @ v`: only about `2.861023e-06`
- both Candle and PyTorch differ from a strict scalar `f32` loop by about `1.907349e-05`

This means the value-side matmul is not a large independent Candle-vs-upstream parity bug
for the focused query. The dominant attention-context drift is still being driven by the
probability-row difference from the softmax denominator stage.

## Runtime Layers Implicated

Primary causal layer:

- CPU softmax denominator accumulation
  - `candle_nn::ops::softmax_last_dim` in
    `candle_sam3/candle-nn/src/ops.rs`
  - row-sum reduction through `vec_reduce_sum`
  - underlying SIMD `f32` row-sum in
    `candle_sam3/candle-core/src/cpu/mod.rs`

Secondary numeric layer:

- CPU layer-norm row reductions
  - `candle_nn::ops::layer_norm` in `candle_sam3/candle-nn/src/ops.rs`
  - row-wise `f32` accumulation for `sum` and `sum2`
  - likely part of the same general numeric family as the softmax split, but not yet shown to be
    the dominant branch-level amplifier
- CPU `probs @ v` accumulation order
  - `Tensor::matmul` CPU path in
    `candle_sam3/candle-core/src/cpu_backend/mod.rs`
  - `gemm`-based `MatMul`
  - this exists numerically, but it is much smaller as a Candle-vs-upstream parity issue
    than the softmax denominator split

## Experiment That Improved The Microprobe But Not The Issue

Running CPU attention reductions in `f64` reduced the same-input attention microprobe gap:

- focused probability row gap: `1.34e-05 -> 2.15e-06`
- focused context gap: `5.77e-05 -> 7.87e-06`
- projected `attn_output` gap: `4.48e-05 -> 1.24e-05`

But the branch-level issue-10 replay did not improve overall:

- block-7 `attn_output` got better
- block-7 `output` got slightly worse
- block-8 `output` got slightly worse

That experiment was reverted.

## Current Best Root-Cause Statement

The first major same-input block-7 divergence is not an MLP bug and not a RoPE bug.
It is a CPU attention-core numeric split whose dominant parity cause is:

1. a softmax denominator reduction-order mismatch in `f32`

There are also earlier small numeric differences in `norm1`, `q/k/v`, and RoPE-sized tensors.
Those now look more like the same general family of CPU `f32` reduction / accumulation-order
effects than like separate functional bugs. The value-side matmul also has ordinary `f32`
accumulation-order effects, but that now looks like a much smaller parity contributor than the
softmax denominator path.

In the current generic build, Candle appears to be taking a scalar-style CPU reduction path,
while upstream CPU softmax behaves like a different reduction order.

Block 8 does not appear to have a comparable same-input implementation bug. The issue-10
threshold crossing is best explained as:

1. block-7 same-input attention-core drift in the generic CPU build
2. residual carry into later blocks
3. ordinary downstream amplification, with block 8 being the first place that breaches tolerance

## Native Build Check

Rebuilding the probe with `RUSTFLAGS=\"-C target-cpu=native\"` moved Candle much closer
to upstream on the same saved tensors:

- focus-row probabilities: about `1.53e-05 -> 1.67e-06`
- focus context: about `5.77e-05 -> 7.15e-06`
- projected attention output: about `4.48e-05 -> 9.54e-06`

That strongly suggests the dominant issue-10 attention-core drift is tied to the generic
CPU build path and its reduction order, rather than to checkpoint loading, RoPE, or the
block structure itself.

Running the existing Rust block replay tests under `RUSTFLAGS=\"-C target-cpu=native\"`:

- block 7 improved slightly but did not fully resolve
  - `attn_output`: about `5.2452e-05 -> 2.4796e-05`
  - `output`: about `8.3447e-05 -> 8.5175e-05`
- block 8 threshold surface did not move materially
  - `output`: stayed about `1.373291e-04`

So the native build narrows the same-input attention mismatch but does not, by itself,
eliminate the issue-10 branch failure.

## Useful Files And Artifacts

- Probe binary:
  - `rust/sam3-parity-cli/src/bin/block7_attention_probe.rs`
- Block replay helpers:
  - `candle-transformers/src/models/sam3/vitdet.rs`
  - `candle-transformers/src/models/sam3/image.rs`
- Captured tensors:
  - `/tmp/parity_box_positive_debug_b7_report/actual.safetensors`
  - `/tmp/reference_box_positive_debug_b7_current/reference.safetensors`
  - `/tmp/sam3_block7_attention_probe.safetensors`

## Next Steps

If issue `#10` continues immediately, the best next technical step is:

- isolate the earliest row-reduction mismatch on the same saved `block7.input`, starting with
  `norm1`
  - export Candle `norm1` row statistics for the focused row
  - reproduce upstream row `mean`, `var`, and normalized output on the same input
  - test whether substituting upstream `norm1` into Candle qkv materially shrinks the later
    softmax denominator split

After that, the next highest-signal step is:

- test a narrow CPU reduction-path change rather than a full `f64` attention patch
  - specifically compare generic-build `vec_reduce_sum` behavior with a targeted alternate
    reduction for softmax denominator rows
  - re-run the block-7 and block-8 replay hooks to see whether the branch threshold actually moves

If issue `#10` is put on hold, the best handoff step is:

- preserve the current same-input decomposition and pick back up from `norm1 -> qkv -> softmax`
  hybrid substitution tests
  - this is the shortest path to deciding whether the remaining branch failure is primarily
    a layer-norm reduction-order mismatch feeding attention, or whether softmax denominator order
    alone is still sufficient to explain most of the surviving drift
