# Issue 10 Full Parity Sweep, 2026-05-21

## Context

This sweep was run after the issue #10 encoder-drift fixes on the sibling
`candle_sam3` checkout:

- branch: `issue-10-encoder-drift`
- revision: `2ed3c9a0`
- checkpoint: `/home/dnorthover/extcode/hf_sam3/sam3.pt`
- parity tolerance: `1e-4`

The focused issue #10 diagnostic now keeps the early trunk drift under the
threshold:

- `patch_embed`: `1.431e-6`
- `ln_pre`: `9.537e-6`
- `vision.block.0`: `1.2398e-5`
- `vision.block.8`: `4.5776e-5`

The full image and video sweep below checks whether any larger end-to-end
parity gaps remain after that localized fix.

## Commands

Image stage parity:

```bash
CARGO_TARGET_DIR=/tmp/sam_parity_target \
cargo run --release -p sam3-parity-cli --bin sam3-parity-cli -- \
  --checkpoint /home/dnorthover/extcode/hf_sam3/sam3.pt \
  --parity-bundle tests/reference-bundles/reference_box_positive \
  --output-dir /tmp/sam3-full-parity-image-box-positive \
  --parity-atol 1e-4

CARGO_TARGET_DIR=/tmp/sam_parity_target \
cargo run --release -p sam3-parity-cli --bin sam3-parity-cli -- \
  --checkpoint /home/dnorthover/extcode/hf_sam3/sam3.pt \
  --parity-bundle tests/reference-bundles/reference_shoe \
  --output-dir /tmp/sam3-full-parity-image-shoe \
  --parity-atol 1e-4
```

Image output/reference comparison:

```bash
CARGO_TARGET_DIR=/tmp/sam_parity_target \
cargo run --release -p sam3-parity-cli --bin sam3-parity-cli -- \
  --checkpoint /home/dnorthover/extcode/hf_sam3/sam3.pt \
  --compare-reference-bundle tests/reference-bundles/reference_box_positive \
  --output-dir /tmp/sam3-full-compare-image-box-positive \
  --parity-atol 1e-4

CARGO_TARGET_DIR=/tmp/sam_parity_target \
cargo run --release -p sam3-parity-cli --bin sam3-parity-cli -- \
  --checkpoint /home/dnorthover/extcode/hf_sam3/sam3.pt \
  --compare-reference-bundle tests/reference-bundles/reference_shoe \
  --output-dir /tmp/sam3-full-compare-image-shoe \
  --parity-atol 1e-4
```

Video full-parity harness:

```bash
CARGO_TARGET_DIR=/tmp/sam_parity_target \
SAM3_TEST_CHECKPOINT_DIR=/home/dnorthover/extcode/hf_sam3 \
SAM3_TEST_CHECKPOINT=/home/dnorthover/extcode/hf_sam3/sam3.pt \
SAM3_TOKENIZER=/home/dnorthover/extcode/hf_sam3/tokenizer.json \
cargo test --release -p sam3-parity-cli --features full-parity \
  video_parity_harness::video_parity::tests:: \
  -- --nocapture --test-threads=1
```

Ignored single-click video row:

```bash
CARGO_TARGET_DIR=/tmp/sam_parity_target \
SAM3_TEST_CHECKPOINT_DIR=/home/dnorthover/extcode/hf_sam3 \
SAM3_TEST_CHECKPOINT=/home/dnorthover/extcode/hf_sam3/sam3.pt \
SAM3_TOKENIZER=/home/dnorthover/extcode/hf_sam3/tokenizer.json \
cargo test --release -p sam3-parity-cli --features full-parity \
  video_process_frame_matches_single_click_point_reference_bundle_frame1 \
  -- --ignored --nocapture --test-threads=1
```

## Image Results

Both image bundles still fail strict internal stage parity at `1e-4`.

`reference_box_positive`:

- failing stages: `32`
- first failing stage: `vision.block.7`, `max_abs_diff=0.00010108948`
- `vision.block.8`: `0.00019073486`
- `vision.block.31` / `vision.trunk.0`: `0.011360168`
- `vision.backbone_fpn.1`: `0.00078219175`
- `vision.backbone_fpn.2`: `0.0005891323`
- `fusion.memory`: `0.0013389587`
- `decoder.pred_logits`: `0.002557993`
- `decoder.pred_boxes_xyxy`: `0.00021111965`
- `segmentation.mask_logits`: `0.21522522`
- report: `/tmp/sam3-full-parity-image-box-positive/parity_report.json`

`reference_shoe`:

- failing stages: `32`
- first failing stage: `vision.block.7`, `max_abs_diff=0.00010108948`
- `vision.block.8`: `0.00019073486`
- `vision.block.31` / `vision.trunk.0`: `0.011360168`
- `fusion.memory`: `0.0007853508`
- `decoder.pred_logits`: `0.0005683899`
- `decoder.pred_boxes_xyxy`: `0.00036489964`
- `segmentation.mask_logits`: `0.4892273`
- report: `/tmp/sam3-full-parity-image-shoe/parity_report.json`

The residual full-bundle drift is later and larger than the focused issue #10
diagnostic. The issue #10 seed issue appears fixed, but the full 32-block image
bundle still crosses the threshold at block 7 and then compounds through the
trunk and segmentation head.

## Image Output Comparisons

`reference_box_positive` completed but does not match the same selected query:

- reference best query: `44`
- Candle best query: `51`
- reference score: `0.94150615`
- Candle score: `0.9427111`
- score absolute diff: `0.0012049675`
- box mean absolute diff: `0.22025153`
- box IoU: `0.0`
- mask mean absolute diff: `0.039529104`
- mask IoU at 0.5: `0.0`
- report: `/tmp/sam3-full-compare-image-box-positive/reference_comparison_report.json`

This is a visible output mismatch: the best query switches, so the final box and
mask land on a different object.

`reference_shoe` completed with the same selected query and close visible
output:

- best query: `135`
- reference score: `0.89198434`
- Candle score: `0.92611885`
- score absolute diff: `0.034134507`
- box mean absolute diff: `0.000093743205`
- box IoU: `0.99270105`
- mask mean absolute diff: `0.000015501633`
- mask IoU at 0.5: `0.99292785`
- report: `/tmp/sam3-full-compare-image-shoe/reference_comparison_report.json`

The `shoe` case still has score drift, but the selected query, box, and mask are
effectively aligned.

## Video Results

The full video harness ran for about `12100.07s` and finished with:

- passed: `18`
- failed: `5`
- ignored: `1`
- filtered out: `79`

The ignored single-click point row was run separately and passed in `179.60s`.

The repeated `unsupported storage type ComplexFloatStorage` messages did not
stop the suite and were not the cause of the five failures.

### Artifact Failures

Two failures are blocked by bad or missing upstream artifacts:

- `video_process_frame_matches_reverse_reference_bundle_frames_20_and_19`
  failed while parsing empty JSON. Both
  `tests/reference-bundles/reference_video_reverse_propagation_debug/reference.json`
  and `tests/reference-bundles/reference_video_reverse_propagation_debug/video_results.json`
  are `0` bytes.
- `video_process_frame_matches_visual_box_reference_bundle_frame0` failed
  because `tests/reference-bundles/reference_video_box_debug` is missing. The
  matrix expects this directory, but the checkout currently only has
  `reference_video_box` and `reference_video_box_debug_temporal_disambiguation`
  for the box rows.

These rows should be rerun only after regenerating the upstream reference
bundles.

### Behavioral Failures

Three failures look like remaining video parity gaps:

- `video_process_frame_matches_output_non_overlap_reference_bundle_frames_0_and_1`
  expected object ids `[2]`, but Candle returned `[1, 2]`. This points at
  output non-overlap or final output suppression keeping object `1` when the
  upstream reference suppresses it.
- `video_propagation_matches_temporal_disambiguation_reference_bundle`
  expected non-empty frames `[0]`, but Candle returned `[0, 1, 2, 3, 4]`. This
  points at temporal-disambiguation or masklet confirmation suppression not
  removing later frames.
- `video_propagation_matches_unconfirmed_producer_reference_bundle` expected
  frame `0` `unconfirmed_obj_ids` to contain `{0}`, but Candle produced `{}`.
  This points at the unconfirmed producer metadata path, or at a stale
  reference if regeneration changes the expected metadata.

## Interpretation

The issue #10 parity seed fix did what it was meant to do for the shorter
diagnostic, including the earliest trunk blocks. It did not close every
end-to-end image gap. The remaining image evidence is now:

- a reproducible full-bundle trunk drift that first crosses `1e-4` at block 7,
  then compounds through the late trunk and downstream heads
- one image prompt where the final selected query changes, producing a visible
  box/mask mismatch
- one image prompt where the final geometry remains close despite score drift

The video evidence is mixed until references are regenerated. Two rows are
currently artifact quality failures, not implementation evidence. The remaining
three video failures are plausible implementation gaps in output non-overlap,
temporal disambiguation, and unconfirmed-object metadata.

## Recommended Next Steps

1. Regenerate the upstream reference bundles before making a final call on the
   video failures. At minimum this must produce
   `reference_video_box_debug` and non-empty reverse-propagation JSON files.
2. Rerun the full video harness after regeneration. If the same three behavior
   failures remain, investigate them as video tracker/postprocess parity issues.
3. Rerun image stage parity after regenerating `reference_box_positive` and
   `reference_shoe`. If block 7 still first crosses the threshold, start the
   next localization run around blocks 6 through 8 under the full image bundle
   path rather than the shorter issue #10 diagnostic path.
4. Investigate the `reference_box_positive` best-query switch separately from
   raw stage drift, since small score/order differences can become visible when
   two candidate queries are close.
