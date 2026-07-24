# CANDLE-2.13 F16 CUDA certification report

Last updated: 2026-07-24

Status: **RED — do not enable F16 in the consumer plugin.**

This report records the current `den-sq/sam_parity#46` acceptance run. It
preserves failing results rather than widening the predeclared tolerances in
`CANDLE2_F16_CUDA_ACCEPTANCE.md`.

## Revisions and environment

- sam_parity fixture/diagnostic revision:
  `3b9fa22595a52391177f79937f911df61dfca998`
- candle_sam3 revision:
  `95e0c186cdf7a7d51224659a103ce73294b7efad`
  (`den-sq/candle_sam3#10`)
- GPU: NVIDIA Quadro RTX 5000 with Max-Q Design, 16,384 MiB, compute
  capability 7.5
- driver: 581.60
- CUDA toolkit: 12.9 (`cuda_12.9.r12.9/compiler.36037853_0`)
- checkpoint SHA-256:
  `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`
- single-click `reference.json` SHA-256:
  `cec1bc102e415a7b372d9a12496752d260e0fa7c5848157ddda1cc09b44ab5b6`
- single-click internal tensors SHA-256:
  `fe0755aadb61c9186844a853cfca379c6d7f8ca38e6d11fa4d663787e8acf1fb`

## Defects exposed and fixed

The committed checkpoint-backed fixtures exposed three native-F16/Turing
defects. All revisions below are reachable in `den-sq/candle_sam3`:

1. Packed retained BF16 mask memory was indexed on CUDA before conversion to
   the compute dtype. Turing has no BF16 `index_select` kernel. Fixed by
   candle_sam3
   `3e69ce59378459299ed6063c26a76d1f6dd9af3e`.
2. The packed execution cache concatenated authoritative retained BF16 tensors
   on CUDA. Turing has no BF16 `copy2d` kernel. The authoritative retained state
   remains BF16, while the packed execution cache now uses F16/F32 compute
   dtype. Fixed by candle_sam3
   `71690361a0e4eb839cfc22a52fcdf5cfbf047f0a`.
3. The correction/mask-input path unconditionally converted the mask prompt to
   F32 immediately before an F16 `mask_downsample` convolution. Both mask-output
   entry points now use tracker/backbone compute dtype. Fixed by candle_sam3
   `95e0c186cdf7a7d51224659a103ce73294b7efad`.

Focused Turing CUDA regressions for the first two fixes pass. The third is
covered by the committed full-checkpoint correction workflow below.

## Conditioned-frame numerical fixture

Invocation:

```bash
CUDA_HOME=/usr/local/cuda-12.9 CUDA_COMPUTE_CAP=75 \
SAM3_TEST_CHECKPOINT=/home/dnorthover/extcode/hf_sam3/sam3.pt \
SAM3_PARITY_BUNDLE_ROOT=/home/dnorthover/ChengCode/sam_parity/tests/reference-bundles \
cargo test --release -p sam3-parity-cli --features full-parity,cuda \
  conditioned_frame_f16_cuda_matches_f32_and_facebook_references \
  -- --ignored --nocapture --test-threads=1
```

Run date: 2026-07-24. Exit status: 101. Eleven declared comparisons fail.
The complete timed log is
`/tmp/pr47_conditioned_95e0c186_timed.log` (SHA-256
`b70a8c477fa1ecd826285149d8d3324bf366309ed26d4bbaa178541b9ce7842d`).

The original high-resolution comparison incorrectly used public postprocessed
binary `mask_logits`. The committed fixture now reconstructs raw
high-resolution logits from retained raw tracker logits. This correction
reduced the F32/Facebook high-resolution mean absolute difference from
19.557600 to 1.739621 without changing any tolerance.

### F16 versus F32

| Tensor/output | Max abs | Mean abs | Violations | Result |
| --- | ---: | ---: | ---: | --- |
| frame-1 low-res logits | 2.380661 | 0.029134 | 3 / 82,944 | fail |
| frame-1 raw high-res logits | 2.196854 | 0.026028 | 7 / 1,016,064 | fail |
| object-score logits | 0.001161 | 0.001161 | 0 | pass |
| object pointer | 0.004957 | 0.001433 | 0 | pass |
| mask-memory features | 1.675781 | 0.005673 | 137 / 331,776 | fail |
| mask-memory position encoding | 0.000244 | 0.000051 | 0 | pass |
| frame-0 binary mask | IoU 0.999865 | delta 0.000004 | — | pass |
| frame-1 binary mask | IoU 0.999799 | delta 0.000006 | — | pass |

The F16/F32 failures are sparse, but the contract is elementwise and therefore
remains failed. The excellent binary-mask agreement does not override the
declared intermediate gate.

The fixture also records sign agreement and diagnostics for values whose
reference is near the binary-logit boundary
(`abs(reference) <= 0.5`). These are diagnostic only and do not relax the
predeclared elementwise gate:

| F16/F32 logits | Overall sign agreement | Near-boundary elements | Near-boundary max abs | Near-boundary sign agreement |
| --- | ---: | ---: | ---: | ---: |
| frame-1 low-res | 1.00000000 | 13 | 0.023353 | 1.00000000 |
| frame-1 raw high-res | 0.99999410 | 227 | 0.055650 | 0.97356826 |

Thus, the few large F16/F32 logit tolerance violations occur away from the
decision boundary. The six high-resolution sign changes are confined to
reference-near-boundary values with an absolute difference no greater than
0.055650. The complete log emits the same diagnostics for every compared
tensor.

### Candle versus Facebook

| Tensor/output | F32 max / mean abs | F16 max / mean abs | Result |
| --- | ---: | ---: | --- |
| frame-1 low-res logits | 16.510910 / 1.809129 | 16.562500 / 1.816134 | fail / fail |
| frame-1 raw high-res logits | 15.301519 / 1.739621 | 15.341211 / 1.747062 | fail / fail |
| object-score logits | 0.350402 / 0.350402 | 0.351562 / 0.351562 | pass / pass |
| object pointer | 0.703797 / 0.216309 | 0.703308 / 0.216025 | fail / fail |
| mask-memory features | 4.769531 / 0.203696 | 4.769531 / 0.203805 | fail / fail |
| mask-memory position encoding | 0.001949 / 0.000355 | 0.001953 / 0.000350 | pass / pass |
| frame-1 binary mask | IoU 0.991147, delta 0.000255 | IoU 0.991346, delta 0.000249 | pass / pass |

The broad intermediate discrepancy is already present in F32 and is therefore
not caused by enabling F16. Absolute localization and disposition of that F32
baseline discrepancy now belong to `den-sq/sam_parity#50` and do not block F16
certification. `den-sq/sam_parity#46` retains the existing absolute Facebook
binary-mask/output acceptance and requires a predeclared relative
F32/Facebook intermediate non-regression rule for F16.

For the Facebook logit reference, overall sign agreement remains high while
the very small near-boundary subsets expose the baseline discrepancy:

| Candle/Facebook logits | Overall sign agreement | Near-boundary elements | Near-boundary max abs | Near-boundary sign agreement |
| --- | ---: | ---: | ---: | ---: |
| F32 low-res | 0.99973476 | 19 | 3.268288 | 0.47368422 |
| F16 low-res | 0.99973476 | 19 | 3.275391 | 0.47368422 |
| F32 raw high-res | 0.99975497 | 203 | 3.263259 | 0.58128077 |
| F16 raw high-res | 0.99975890 | 203 | 3.252073 | 0.58620691 |

## Native-F16 workflow matrix

The ignored
`f16_cuda_video_workflow_acceptance_matrix` fixture loads the checkpoint once
in native F16 and verifies:

- streamed forward propagation with a hotstart delay and ordered final drain;
- retained BF16 mask-memory state and a 16-state non-conditioning bound;
- bounded feature cache;
- reset cleanup, close cleanup, and a distinct fresh-session restart;
- backward propagation from frame 20;
- correction at frame 8 and subsequent frame-9 propagation;
- finite public masks, logits, boxes, scores, and presence scores;
- Facebook mask IoU for every asserted output.

Run date: 2026-07-24. Exit status: 0. The complete timed log is
`/tmp/pr47_workflow_95e0c186_timed.log` (SHA-256
`785f7df57f9009adbffd39486ffab0ff53a752d470ad0832479ff4493e683120`).

| Scenario | Frames | Facebook mask IoU |
| --- | --- | --- |
| forward + hotstart stream | 0–4 | 0.994767, 0.991346, 0.995512, 0.990698, 0.989918 |
| backward | 20–18 | 0.996782, 0.995987, 0.995074 |
| correction + continuation | 8–9 | 0.993056, 0.993860 |

After reset, tracked objects, tracker states, output indices, cached output
frames, feature entries, hotstart buffer, and reported CPU/device session bytes
are all zero. After close, the session is absent. Restart produces a distinct
session ID.

## Host RSS

`/usr/bin/time -v` was attached directly to the test process, not to a
profiler wrapper:

| Fixture | Wall time | Peak host RSS | Swaps | Exit |
| --- | ---: | ---: | ---: | ---: |
| F16 workflow matrix | 67.70 s | 635,284 KiB (620.39 MiB) | 0 | 0 |
| sequential F32 + F16 conditioned frame | 26.93 s | 670,556 KiB (654.84 MiB) | 0 | 101 (expected) |

An earlier pre-rebase capture recorded 621.97 MiB for the F16 workflow and
649.69 MiB for sequential F32 + F16 conditioned execution. Those are separate
process-RSS captures, not alternate readings from the exact-head run above.
The 1.58 MiB lower workflow value and 5.15 MiB higher conditioned value in the
exact-head rerun are treated as ordinary host-RSS variability; neither pair
shows retained per-frame growth.

The previously reported multi-gigabyte host RSS high-water is not reproduced
when the model process is measured directly. These runs indicate that the
earlier value came from wrapper/container/page-cache accounting rather than
model-owned resident memory. Reproduction instructions must attach RSS
measurement to the inference PID before treating the older high-water as a
runtime leak.

## CUDA attribution

Nsight Systems 2025.6.3 report:

- run date: 2026-07-24
- candle_sam3 revision:
  `95e0c186cdf7a7d51224659a103ce73294b7efad`
- path: `/tmp/pr47_f16_workflow_95e0c186.nsys-rep`
- SHA-256:
  `33c887083ca3ae38adce41f905a95b643b662785eb868644c62f2c1c2e34392f`
- trace: `cuda-sw,cublas-verbose,cuDNN-verbose`
- workload: the passing F16 workflow matrix above

This host reports that CUDA hardware tracing is unsupported. Both hardware and
explicit software traces contain CUDA API activity but no GPU kernel or GPU
memory rows. The software trace attributes:

| CUDA API | Calls | Total API time |
| --- | ---: | ---: |
| `cuLaunchKernel` | 58,827 | 22.724 s |
| `cudaLaunchKernel` | 15,907 | 8.216 s |
| `cuMemcpyHtoDAsync_v2` | 28,572 | 14.395 s |
| `cuMemcpyDtoDAsync_v2` | 10,241 | 3.098 s |
| `cuMemcpyDtoHAsync_v2` | 136 | 4.292 s |
| `cuMemAllocAsync` / `cuMemFreeAsync` | 101,452 each | 0.883 / 0.536 s |

Because Nsight cannot emit kernel rows on this host, an equivalent temporary
trace at Candle's CUDA function loader was used to attribute 58,295 direct
Candle kernel launches. The instrumentation was removed after capture; the raw
trace is `/tmp/issue46_f16_kernel_loader_trace.log`, SHA-256
`4ff82e2267c689f89cc7b8ee8d421cf2d8b9395448d75d71382a488cddb73e9b`.
That supplemental loader trace predates the rebase, but its uninstrumented
Candle source tree was verified byte-for-byte identical to reachable revision
`95e0c186cdf7a7d51224659a103ce73294b7efad`; the fresh current-head Nsight
trace above remains the primary attribution artifact.

| Direct Candle kernel | Calls |
| --- | ---: |
| `im2col_f16` | 10,730 |
| `copy2d_f16` | 10,608 |
| `badd_f16` | 6,661 |
| `bmul_f32` | 6,224 |
| `cast_f16_f32` | 3,748 |
| `copy2d_f32` | 3,528 |
| `cast_f32_f16` | 3,510 |
| `ucopy_f16` | 2,740 |
| `layernorm_f16` | 1,598 |
| `bsub_f32` | 1,408 |
| `badd_f32` | 1,408 |
| `softmax_f32` | 844 |
| `affine_f32` | 778 |
| `ugelu_erf_f16` | 772 |
| `conv_transpose2d_f16` | 148 |
| `upsample_bilinear2d_f16` | 61 |

The difference between the 58,827 driver launches and 58,295 directly named
Candle kernels is 532 launches. The report independently records 532
`cuKernelGetFunction` calls; it does not expose enough information to prove
that the equal counts have the same cause. The unnamed launches are inferred
to be library/custom-module activity outside Candle's direct module loader.
Kernel names and API-side time are therefore attributed, but per-kernel GPU
duration remains unavailable on this host. The recorded operator/kernel/API
attribution satisfies `den-sq/sam_parity#46`'s original "Nsight or equivalent"
requirement. Paired F32/F16 per-kernel duration on a trace-capable host is
non-blocking follow-up work in `den-sq/sam_parity#51`.

## Final-head throughput, memory, and output evidence

The exact 32/128/512-frame matrix was rerun on 2026-07-24 at reachable
candle_sam3 revision
`95e0c186cdf7a7d51224659a103ce73294b7efad`, using benchmark-harness revision
`af28c046b4092743da9d412c1673d5729f8c7c56`. Each dtype/length pair was an
isolated container run on the same SM75 GPU and driver. The container CUDA
runtime was 12.4.1. The machine-readable summary is
`/tmp/candle_pr10_rebased_95e0c186_evidence/summary.json` (SHA-256
`525f4ab5ecc8750fe50fd11b9df3b4edbd2e0ef93eee869cf9210a3484b9190f`).

| Frames | Dtype | Startup s | Transfer/setup s | Propagation s | Propagation FPS | End-to-end s | Peak VRAM MiB | Mean propagation GPU util. | Peak container RSS MiB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | F32 | 21.864 | 4.076 | 149.634 | 0.213855 | 178 | 10,967 | 94.73% | 331.9 |
| 32 | F16 | 21.791 | 4.102 | 114.916 | 0.278464 | 142 | 7,773 | 94.11% | 461.5 |
| 128 | F32 | 21.238 | 4.759 | 600.861 | 0.213028 | 629 | 11,060 | 96.46% | 333.1 |
| 128 | F16 | 20.053 | 4.506 | 454.808 | 0.281438 | 484 | 7,774 | 96.65% | 462.9 |
| 512 | F32 | 20.625 | 7.050 | 2,392.833 | 0.213972 | 2,425 | 10,978 | 97.07% | 369.4 |
| 512 | F16 | 19.474 | 7.317 | 1,810.273 | 0.282830 | 1,837 | 7,781 | 96.42% | 488.9 |

| Frames | Propagation-time reduction | Propagation-throughput increase | End-to-end-time reduction | Peak-VRAM reduction |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 23.20% | 30.21% | 20.22% | 29.12% |
| 128 | 24.31% | 32.11% | 23.05% | 29.71% |
| 512 | 24.35% | 32.18% | 24.25% | 29.12% |

All six runs reported zero retained-mask-memory dtype mismatches. The retained
BF16 mask-memory payload was identical by dtype (5,971,968 bytes at 32 frames;
5,308,416 bytes at 128/512), while aggregate device tracker-state accounting
fell from 28,565,632 to 17,268,800 bytes at 32 frames (39.55%) and from
26,907,780 to 16,108,098 bytes at 128/512 (40.14%).

The output-mask comparison passes the 0.99 per-frame F16/F32 IoU criterion at
all three lengths:

| Frames | Mean IoU | Median IoU | Minimum IoU | Frames below 0.99 | Changed pixels |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 0.998385851 | 0.998478637 | 0.994896697 | 0 | 1,604 / 1,280,000 (0.125313%) |
| 128 | 0.998492800 | 0.998687465 | 0.994908810 | 0 | 5,947 / 5,120,000 (0.116152%) |
| 512 | 0.998639036 | 0.998745481 | 0.994908810 | 0 | 21,412 / 20,480,000 (0.104551%) |

Output SHA-256 values:

| Frames | F32 | F16 |
| ---: | --- | --- |
| 32 | `356f510633f869facab3cfcc560a0cb237739f886c706c21e2f97960f365c791` | `ce922a8fe18ae8f12225c78a3a610ae7ad300216fa7776a2b82ff61ab19db7d8` |
| 128 | `727e02e688e565d4d8384ec2b05d704256d20e41f0a997c014c4ad32c21fa2ca` | `6b24aa65ea4d2b4b852dfed5336e872ce63d6007561912605cae1745a68afefa` |
| 512 | `cdc4408b8629fed7dbc2d8124cb39f38ceb0685268ac5e13c2288f8197795653` | `1a7f611771858285995e1eb9019d498756fb75f7db96aa32090a844303c3764c` |

All six output hashes exactly reproduce the corresponding pre-rebase
recordings. Propagation time moved upward by 0.24–1.31% and sampled peak VRAM
by 0.82–1.58% between the two run sets. The within-run F16/F32 comparisons are
the acceptance measurements; this small cross-run drift affects no gate.

The 32-frame F32 hash also exactly matches the separately recorded
pre-accounting control, showing that the accounting fix did not alter output.

F16 container RSS is 119.5–129.8 MiB higher than F32 even though device memory
is about 29% lower. The increase is roughly flat rather than frame-count
dependent, and direct-process fixture RSS remains near 620–655 MiB, so this is
not evidence of retained per-frame growth. Its allocator/container source has
not been isolated and remains a documented caveat.

## CI scope

Both workflows now clone the reachable candle_sam3 branch
`codex/candle-2-13-f16-turing` and verify exact SHA
`95e0c186cdf7a7d51224659a103ce73294b7efad`.

The ordinary Rust CI jobs are CPU/default-feature jobs. They neither compile
nor run the `full-parity,cuda` fixture paths. The current nightly job also
lacks the private checkpoint/reference environment and invokes ignored tests
without those features. Therefore a green GitHub Actions result validates the
ordinary repository surface and the dependency pin, but it is not evidence
that either CUDA acceptance fixture ran. The explicit local invocations,
hardware metadata, artifact hashes, and results in this report are the CUDA
certification evidence.

## Remaining red gates

1. Amend the committed acceptance contract with tensor-specific F16/F32
   intermediate metrics and thresholds plus a predeclared relative
   F32/Facebook non-regression rule. The current recording may justify the
   metric shape but must not both select and pass the replacement thresholds.
2. Run an independent reachable-head confirmation after that contract is
   frozen.
3. Record the final green-certification or no-ship decision, including exact
   revisions and the supported configuration.

Absolute F32/Facebook intermediate localization is tracked separately in
`den-sq/sam_parity#50`. Paired per-kernel duration and host-RSS attribution are
non-blocking follow-up work in `den-sq/sam_parity#51`. Consumer activation is
owned by `ChengLabResearch/ouroboros_autoseg_plugin#52` only after a green
decision.

This PR may be reviewed and merged as the committed fixture and evidence
ledger while certification remains red. The consumer plugin must continue
rejecting `SAM3_COMPUTE_DTYPE=f16` until `den-sq/sam_parity#46` records a green
decision and the separate activation work is completed.
