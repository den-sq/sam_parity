# CANDLE-1 Certification Checklist

Tracking issue: https://github.com/den-sq/sam_parity/issues/28

This checklist records the current certification state for using the Candle
SAM3 runtime from `ouroboros_autoseg_plugin`.

## Revision Alignment

- [x] Plugin Dockerfile pins Candle SAM3 at
  `770d20ca8db4f834ba4c89c845bca196fbfc97ea`.
- [x] The local pinned checkout at `../candle_sam3_main` is detached at
  `770d20ca8db4f834ba4c89c845bca196fbfc97ea`.
- [x] `ouroboros_autoseg_plugin/backend/Cargo.toml` path dependencies resolve to
  `../../candle_sam3_main/{candle-core,candle-nn,candle-transformers}`.
- [ ] Reconfirm the pin before final manuscript certification if the plugin
  Dockerfile or path dependency layout changes.

Note: the active development checkout `../candle_sam3` was at
`6c85ff87f348b5ff5863bbdd1f04641dc25837e1` on `issue-10-encoder-drift` during
this run. That checkout is useful context, but it is not the plugin-certified
pin.

## Completed Non-Checkpoint Checks

- [x] Pinned Candle SAM3 tests:
  `cargo test -p candle-transformers sam3` from `../candle_sam3_main`
  passed with 16 tests.
- [x] Current development Candle SAM3 tests:
  `cargo test -p candle-transformers sam3` from `../candle_sam3`
  passed with 22 tests and 4 ignored manual parity investigations.
- [x] Rust parity workspace:
  `cargo test --workspace` passed in `sam_parity`.
- [x] Python parity contract tests:
  `PYTHONPATH=python python -m pytest python/sam3_parity/tests -q`
  passed with 3 tests.
- [x] Feature-gated full-parity harness compilation:
  `cargo test -p sam3-parity-cli --features full-parity --no-run`
  completed successfully.
- [x] Plugin SAM3 contract tests:
  `cargo test --manifest-path ouroboros_autoseg_plugin/backend/Cargo.toml sam3`
  passed with 19 focused tests and left the checkpoint-backed load test ignored
  because no checkpoint path was configured.

## Output Contract

- [x] The plugin converts raw mask logits to binary `FrameMask` pixels by
  selecting query 0, bilinear upsampling, applying sigmoid, and thresholding
  probability `> 0.5` to `255`; all other pixels are `0`.
- [x] Focused plugin tests cover positive, negative, mixed, and upsampled mask
  logits and confirm the `u8` `0/255` convention.
- [x] Video helper tests cover empty-object output and all-positive single-object
  frame output.
- [ ] Run a checkpoint-backed image smoke test to confirm the same output
  contract with a real Medical-SAM3 checkpoint.
- [ ] Run a checkpoint-backed video propagation smoke test on a small biological
  stack to confirm output geometry and frame count on the plugin path.

## Checkpoint-Backed Checks Still Required

No `SAM3_*` checkpoint environment variables were present during this run, and
the workspace did not contain `sam3.pt` or `medical_sam3.pt`. The only local
workspace `.pt` model artifact found was the legacy SAM2
`ouroboros_autoseg_plugin/sam2_hiera_base_plus.pt`, which is not sufficient for
CANDLE-1.

Run these before closing CANDLE-1:

```bash
export OUROBOROS_SAM3_CHECKPOINT=/path/to/medical_sam3.pt
cargo test --manifest-path ouroboros_autoseg_plugin/backend/Cargo.toml \
  --ignored sam3_medical_checkpoint_load -- --nocapture

cargo run --manifest-path ouroboros_autoseg_plugin/backend/Cargo.toml \
  --release --bin sam3_checkpoint_smoke -- \
  --checkpoint "$OUROBOROS_SAM3_CHECKPOINT" \
  --mask-out /tmp/sam3_smoke_mask.tif
```

For parity-side video certification, stage the official SAM3 checkpoint and run
at least one supported bundle-backed propagation check:

```bash
export SAM3_TEST_CHECKPOINT=/path/to/sam3.pt
cargo test -p sam3-parity-cli --features full-parity \
  video_propagation_matches_text_prompt_suppressed_reference_bundle -- --nocapture
```

If CUDA is the certification target:

```bash
export SAM3_TEST_CHECKPOINT=/path/to/sam3.pt
cargo test -p sam3-parity-cli --features full-parity,cuda \
  video_propagation_matches_text_prompt_suppressed_reference_bundle_cuda \
  -- --ignored --nocapture --test-threads=1
```

## Known Parity Gaps

- Issue 10 image parity still documents internal vision-trunk drift. The
  `reference_shoe` case has close final mask agreement, while the
  `box_positive` case can still select a different query and produce a visible
  output mismatch.
- The strict video tracker port matrix has broad bundle coverage, but exact
  upstream parity remains a separate certification question from the plugin's
  `u8` mask contract.
- These gaps are manuscript-risk items for exact-upstream claims. They are not
  currently blockers for the plugin's local output-format contract, but they
  should remain visible in any publication language.

## Certification State

- Non-checkpoint SAM3 runtime, parity contract, and plugin output-format checks
  are green.
- The plugin pin is aligned with the intended certified Candle checkout.
- CANDLE-1 should remain open until checkpoint-backed Medical-SAM3 image smoke
  and at least one video propagation smoke/parity path are run with staged
  weights.
