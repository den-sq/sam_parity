# CANDLE-2.13 F16 CUDA acceptance contract

Date declared: 2026-07-23

Scope: `den-sq/sam_parity#46` and the draft F16 implementation in
`den-sq/candle_sam3#10`.

This file declares the correctness thresholds before running the committed
checkpoint-backed CUDA fixture. A failing result must be preserved and
investigated; these thresholds must not be tuned to make a new result pass
without a recorded numerical justification.

## Conditioned-frame integration fixture

The ignored, serial
`conditioned_frame_f16_cuda_matches_f32_and_facebook_references` Rust test:

- loads the same upstream checkpoint in native F32 and native F16 on CUDA;
- uses the official single-click Facebook reference inputs;
- processes prompted frame 0 and non-prompt frame 1;
- requires frame 1 to retain frame 0 as its conditioning prompt;
- retains mask-memory features in BF16, matching the throughput candidate;
- caps non-conditioning tracker history at the tracker-required minimum of 16;
- requires the model and tracker compute dtypes to equal the requested dtype;
- rejects NaN or infinity in output masks, mask logits, object-score logits,
  object pointers, mask-memory features, and mask-memory position encodings;
- compares F16 to F32 and both Candle modes to the recorded Facebook tensors.

The private checkpoint and large Facebook tensor payload remain external. The
test code, fixture identity, tensor selection, and tolerances are committed.

## Predeclared thresholds

Tensor closeness is elementwise:

`abs(actual - reference) <= atol + rtol * abs(reference)`.

| Comparison | Tensor | atol | rtol |
| --- | --- | ---: | ---: |
| F16 vs F32 | low/high-resolution mask logits | 0.75 | 0.05 |
| F16 vs F32 | object-score logits | 0.25 | 0.02 |
| F16 vs F32 | object pointer | 0.25 | 0.03 |
| F16 vs F32 | retained mask-memory features/position encoding | 0.25 | 0.04 |
| Candle vs Facebook | low/high-resolution mask logits | 1.0 | 0.05 |
| Candle vs Facebook | object-score logits | 0.5 | 0.05 |
| Candle vs Facebook | object pointer | 0.5 | 0.05 |
| Candle vs Facebook | retained mask-memory features/position encoding | 0.5 | 0.08 |

Binary output masks use the existing fixed `0.5` threshold; no dtype-specific
threshold adjustment is permitted.

| Comparison | Minimum IoU | Maximum changed-pixel rate |
| --- | ---: | ---: |
| F16 vs F32 | 0.99 | 0.005 |
| Candle vs Facebook | 0.97 | 0.01 |

The Facebook envelopes preserve the existing strict-port fixture tolerances
(`1.0` mask-logit absolute tolerance and `0.5` object-score/object-pointer
absolute tolerance) while adding a relative term for large-magnitude logits.
The F16/F32 envelope is intentionally tighter.

## Required invocation

```bash
CUDA_HOME=/usr/local/cuda-12.9 CUDA_COMPUTE_CAP=75 \
SAM3_TEST_CHECKPOINT=/absolute/path/to/sam3.pt \
SAM3_PARITY_BUNDLE_ROOT=/absolute/path/to/reference-bundles \
cargo test --release -p sam3-parity-cli --features full-parity,cuda \
  conditioned_frame_f16_cuda_matches_f32_and_facebook_references \
  -- --ignored --nocapture --test-threads=1
```

Record the exact Candle and parity SHAs, checkpoint/reference hashes, GPU,
driver, command, exit status, and emitted `ISSUE46_TENSOR`/`ISSUE46_MASK`
metrics in the certification report.
