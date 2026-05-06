# SAM3 Parity Checklist

Actionable follow-up is tracked in GitHub issues. This document is a parity-run
record only.

Last updated: 2026-04-10

## Current Reference Bundle

- Bundle: `tests/reference-bundles/reference_box_positive_debug_b10_b11`
- Source: upstream Python export from the installed `sam3` package
- Prompt: exact positive box on `test_image.jpg`
- Extra debug tensors: `vision.block_debug.10.*`, `vision.block_debug.11.*`

## Fresh Parity Runs

### CPU

- Output: `output/parity_box_positive_cpu_debug_b10_b11`
- Report: `output/parity_box_positive_cpu_debug_b10_b11/parity_report.json`
- Failure count: `45`
- First failing trunk stage: `vision.block.8` (`max_abs_diff=0.000144958`)
- First failing debug stage in bundle order: `vision.block_debug.10.input` (`max_abs_diff=0.000186920`)

### CPU After Geometry ROIAlign Fix

- Output: `output/parity_box_positive_cpu_debug_b10_b11_post_geomfix`
- Report: `output/parity_box_positive_cpu_debug_b10_b11_post_geomfix/parity_report.json`
- Failure count: `44`
- First failing trunk stage: `vision.block.8` (`max_abs_diff=0.000144958`)
- First failing debug stage in bundle order: `vision.block_debug.10.input` (`max_abs_diff=0.000186920`)
- Most important deltas vs the previous CPU run:
  - `geometry.features` now passes (`max_abs_diff=0.000017`)
  - `fusion.memory` dropped from about `18.51423` to `0.002718`
  - the large remaining failures are now concentrated in the decoder heads:
    - `decoder.pred_logits`: `5.508646`
    - `decoder.pred_boxes_xyxy`: `1.233163`
    - `decoder.presence_logits`: `1.300597`
    - `segmentation.mask_logits`: `110.744904`

### CPU After Decoder Unit-Test Fixes

- Output: `output/parity_box_positive_cpu_debug_b10_b11_post_decoderfix`
- Report: `output/parity_box_positive_cpu_debug_b10_b11_post_decoderfix/parity_report.json`
- Failure count: `43`
- First failing trunk stage: `vision.block.8` (`max_abs_diff=0.000144958`)
- Most important deltas vs the post-geometry CPU run:
  - `decoder.pred_logits` dropped from `5.508646` to `0.000338`
  - `decoder.pred_boxes_xyxy` dropped from `1.233163` to `0.000121`
  - `decoder.presence_logits` now passes (`max_abs_diff=0.000054`)
  - `segmentation.mask_logits` is still very large (`110.912315`)

### CPU After Segmentation Unit-Test Fixes

- Output: `output/parity_box_positive_cpu_debug_b10_b11_post_segfix`
- Report: `output/parity_box_positive_cpu_debug_b10_b11_post_segfix/parity_report.json`
- Failure count: `43`
- First failing trunk stage: `vision.block.8` (`max_abs_diff=0.000144958`)
- Most important deltas vs the post-decoder CPU run:
  - `segmentation.mask_logits` dropped from `110.912315` to `0.031055`
  - `geometry.features` still passes (`max_abs_diff=0.000017`)
  - `decoder.pred_logits` is still small (`max_abs_diff=0.000338`)
  - `decoder.pred_boxes_xyxy` is still small (`max_abs_diff=0.000121`)
  - the dominant remaining issue is still the early vision trunk / FPN drift, not segmentation-head semantics

### CUDA

- Output: `output/parity_box_positive_cuda_debug_b10_b11`
- Report: `output/parity_box_positive_cuda_debug_b10_b11/parity_report.json`
- Failure count: `50`
- First failing trunk stage: `vision.block.7` (`max_abs_diff=0.000102997`)
- First failing debug stage in bundle order: `vision.block_debug.10.input` (`max_abs_diff=0.000473022`)

## What The Fresh Runs Say

- On current CPU parity after the geometry, decoder, and segmentation fixes, the broad failure shape is now:
  - small but real trunk drift starting at `vision.block.8`
  - somewhat larger downstream FPN drift
  - small fusion / decoder / segmentation residuals that are consistent with that earlier drift
- The remaining full-model numbers support that interpretation:
  - `geometry.features` passes (`0.000017`)
  - `fusion.memory` is small (`0.002718`)
  - `decoder.pred_logits` is small (`0.000338`)
  - `decoder.pred_boxes_xyxy` is small (`0.000121`)
  - `segmentation.mask_logits` is now small (`0.031055`)
- CUDA was worse numerically in the earlier pre-fix run, but not qualitatively different:
  - it failed earlier in the trunk
  - it showed the same general block-MLP drift pattern
  - it showed the same overall drift family as the CPU run
- The new block-level debug shows the block-10 attention path is close:
  - CPU: `vision.block_debug.10.attn_output` passes (`max_abs_diff=0.000039816`)
  - CUDA: `vision.block_debug.10.attn_output` also passes (`max_abs_diff=0.000085831`)
- The block-10 MLP path is where additional error is introduced:
  - CPU: `mlp_fc1`, `mlp_gelu`, `mlp_output`, `output` all fail
  - CUDA: same pattern, just larger
- `vision.block_debug.10.input` already fails, so the true first trunk split is earlier than block 10:
  - CPU: block 8 is the first failing trunk output
  - CUDA: block 7 is the first failing trunk output

After the ROIAlign fix, the geometry conclusion changed materially:

- `geometry.features` is no longer a live parity problem on CPU.
- The fusion mismatch is now small enough to treat as mostly downstream of the early trunk drift.
- After the segmentation fixture fixes, the dominant remaining mismatch in this run set is the early vision trunk drift rather than segmentation-head math.

## Earlier Trunk Localization

- Bundle: `tests/reference-bundles/reference_box_positive_debug_b8_b9`
- CPU report: `output/parity_box_positive_cpu_debug_b8_b9/parity_report.json`

Key result:

- `vision.block_debug.8.input` through `vision.block_debug.8.mlp_output` all pass.
- The first failing internal/output stage is `vision.block_debug.8.output` (`max_abs_diff=0.000144958`).
- `vision.block_debug.9.input` then fails by the same amount.
- Inside block 9:
  - `norm1` and `attn_output` still pass
  - `post_attn` fails only because the incoming residual already drifted
  - `norm2` still passes
  - `mlp_fc1`, `mlp_gelu`, `mlp_output`, and `output` all fail

Interpretation:

- The first threshold breach is at the residual-summed output of block 8.
- The first clearly failing internal branch is the block-9 MLP path.
- This makes the trunk investigation much narrower:
  - residual accumulation / summation behavior around block-8 output
  - MLP branch math in block 9 and later

## Targeted Geometry Tests

- Fixture generator: `python_debug/export_geometry_unit_fixture.py`
- Fixture dir: `tests/data/sam3_geometry_unit`
- Investigation tests live in `candle-transformers/src/models/sam3/geometry.rs`
- The new fixture tests are `#[ignore]` so they can be run explicitly while parity work is in progress.

Commands used:

- `cargo test -p candle-transformers geometry_fixture_point_helpers_match_upstream --lib -- --ignored --nocapture`
- `cargo test -p candle-transformers geometry_fixture_box_helpers_match_upstream --lib -- --ignored --nocapture`
- `cargo test -p candle-transformers geometry_fixture_box_feature_composition_matches_upstream --lib -- --ignored --nocapture`
- `cargo test -p candle-transformers geometry_fixture_mini_encoder_matches_upstream --lib -- --ignored --nocapture`

Results:

- Helper layer:
  - points helper parity passes
  - box position encoding passes
  - raw box pooling now passes

- Feature-composition layer:
  - `geometry/label_embed` matches
  - `geometry/direct_proj` matches
  - `geometry/pooled_boxes_raw` matches
  - `geometry/pool_proj` matches
  - `geometry/box_features` matches

- Mini encoder layer:
  - `geometry/features_initial_norm` matches
  - `geometry/features_after_layer_0` matches
  - `geometry/features_final` matches

Conclusion from targeted tests:

- The geometry investigation successfully localized and fixed the first box-conditioned mismatch.
- The ROIAlign emulation had two real bugs:
  - boxes were not scaled from normalized coordinates into feature-space pixels before pooling
  - pooled spatial axes were assembled as `[C, W, H]` instead of `[C, H, W]`
- After fixing those, all three targeted box-conditioned geometry layers match upstream.

## Targeted Decoder Tests

- Fixture generator: `python_debug/export_decoder_unit_fixture.py`
- Fixture dir: `tests/data/sam3_decoder_unit`
- Investigation tests live in `candle-transformers/src/models/sam3/decoder.rs`
- The new fixture tests are `#[ignore]` so they can be run explicitly while parity work is in progress.

Commands used:

- `cargo test -p candle-transformers decoder_fixture_helper_parity_matches_upstream --lib -- --ignored --nocapture`
- `cargo test -p candle-transformers decoder_fixture_layer_parity_matches_upstream --lib -- --ignored --nocapture`
- `cargo test -p candle-transformers decoder_fixture_final_parity_matches_upstream --lib -- --ignored --nocapture`

Results:

- Helper layer:
  - first failing function was `gen_sineembed_for_position`
  - bug: the Rust implementation generated only half of the expected 4D box-position channels
  - after fixing the frequency-table construction, helper parity passes

- Decoder core layer:
  - per-layer internals now match upstream on the standalone fixture:
    - query position
    - box relative position bias
    - self attention
    - text cross attention
    - image cross attention
    - FFN
    - box delta / refined boxes

- Final scoring/output layer:
  - first remaining mismatch was `decoder.dotprod.prompt_after_mlp`
  - bug: Rust used `LayerNorm(eps=1e-6)` for the dot-product scorer prompt MLP output norm, while upstream uses `nn.LayerNorm` default `eps=1e-5`
  - the full-model decoder presence mismatch was also traced to output behavior:
    - upstream currently does not actually clamp decoder presence logits because it drops the return value of `clamp(...)`
    - Rust was clamping them, which caused the `decoder.presence_logits` full-parity miss
  - after fixing those, all standalone decoder fixture layers match upstream

Conclusion from targeted decoder tests:

- The decoder investigation successfully localized and fixed the first real decoder mismatches.
- The decoder-specific fixes were:
  - `gen_sineembed_for_position` for 4D box coordinates
  - prompt-MLP output `LayerNorm` epsilon in `DotProductScoringHead`
  - decoder presence-logit output semantics (match upstream’s current unclamped behavior)
- After those fixes, the large full-model decoder head errors collapsed to small residual differences that are consistent with upstream trunk/fusion drift.

## Targeted Segmentation Tests

- Fixture generator: `python_debug/export_segmentation_unit_fixture.py`
- Fixture dir: `tests/data/sam3_segmentation_unit`
- Investigation tests live in `candle-transformers/src/models/sam3/segmentation.rs`
- The new fixture tests are `#[ignore]` so they can be run explicitly while parity work is in progress.

Commands used:

- `cargo test -p candle-transformers segmentation_fixture_pixel_path_matches_upstream --lib -- --ignored --nocapture`
- `cargo test -p candle-transformers segmentation_fixture_mask_predictor_matches_upstream --lib -- --ignored --nocapture`
- `cargo test -p candle-transformers segmentation_fixture_final_parity_matches_upstream --lib -- --ignored --nocapture`

Results:

- Pixel / prompt path:
  - first failing tensor was `segmentation.encoder_hidden_states_normed`
  - bug: Rust used `LayerNorm(eps=1e-6)` for `cross_attn_norm`, while upstream `nn.LayerNorm` uses the default `eps=1e-5`
  - after fixing the epsilon, the prompt-cross-attention and pixel-decoder path matches upstream

- Mask-predictor path:
  - first large failing tensor was `segmentation.mask_predictor.query_embed`
  - bug: Rust built `mask_predictor.mask_embed` with only two linear layers, while upstream `MLP(..., num_layers=3)` has three
  - after fixing the layer count, `query_embed` and `mask_logits` match on the standalone fixture

- Final segmentation outputs:
  - standalone fixture now matches upstream for:
    - `segmentation.pixel_embed`
    - `segmentation.instance_embeds`
    - `segmentation.mask_logits`
    - `segmentation.semantic_logits`

Conclusion from targeted segmentation tests:

- The segmentation investigation successfully localized and fixed the two real segmentation-head mismatches.
- The segmentation-specific fixes were:
  - `cross_attn_norm` `LayerNorm` epsilon
  - `mask_predictor.mask_embed` depth
- After those fixes, the full-model `segmentation.mask_logits` error collapsed from `110.912315` to `0.031055`, which is consistent with the remaining upstream trunk/fusion drift rather than a live segmentation semantic bug.
