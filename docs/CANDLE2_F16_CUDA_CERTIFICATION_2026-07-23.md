# CANDLE-2.13 F16 CUDA certification report

Date: 2026-07-23

Status: **RED — do not enable F16 in the consumer plugin.**

This report records the current `den-sq/sam_parity#46` acceptance run. It
preserves failing results rather than widening the predeclared tolerances in
`CANDLE2_F16_CUDA_ACCEPTANCE.md`.

## Revisions and environment

- sam_parity fixture revision: `2c3afc2013545bd840b90cd0bc3486f9b154a23e`
- candle_sam3 revision: `54b6cb5999aabd5d8dd2291d78ef9e65c7bc9fcc`
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
defects:

1. Packed retained BF16 mask memory was indexed on CUDA before conversion to
   the compute dtype. Turing has no BF16 `index_select` kernel. Fixed by
   candle_sam3 `19bbaaf2`.
2. The packed execution cache concatenated authoritative retained BF16 tensors
   on CUDA. Turing has no BF16 `copy2d` kernel. The authoritative retained state
   remains BF16, while the packed execution cache now uses F16/F32 compute
   dtype. Fixed by candle_sam3 `4d1e1add`.
3. The correction/mask-input path unconditionally converted the mask prompt to
   F32 immediately before an F16 `mask_downsample` convolution. Both mask-output
   entry points now use tracker/backbone compute dtype. Fixed by candle_sam3
   `54b6cb59`.

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

Exit status: 101. Eleven declared comparisons fail.

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
not caused by enabling F16. It still blocks Issue #46 until the F32 strict-port
baseline discrepancy is explained or the acceptance contract is deliberately
revised with numerical justification.

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

Exit status: 0.

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
| F16 workflow matrix | 67.66 s | 636,896 KiB (621.97 MiB) | 0 | 0 |
| sequential F32 + F16 conditioned frame | 28.22 s | 665,284 KiB (649.69 MiB) | 0 | expected numerical-gate failure |

The previously reported multi-gigabyte host RSS high-water is not reproduced
when the model process is measured directly. These runs indicate that the
earlier value came from wrapper/container/page-cache accounting rather than
model-owned resident memory. Reproduction instructions must attach RSS
measurement to the inference PID before treating the older high-water as a
runtime leak.

## CUDA attribution

Nsight Systems 2025.6.3 report:

- path: `/tmp/issue46_f16_workflow_cuda_sw.nsys-rep`
- SHA-256:
  `fe4a33b15c03d57f8b73c18d9f68c62d880a88796278788053ad2e8af997e587`
- trace: `cuda-sw,cublas-verbose,cuDNN-verbose`
- workload: the passing F16 workflow matrix above

This host reports that CUDA hardware tracing is unsupported. Both hardware and
explicit software traces contain CUDA API activity but no GPU kernel or GPU
memory rows. The software trace attributes:

| CUDA API | Calls | Total API time |
| --- | ---: | ---: |
| `cuLaunchKernel` | 58,827 | 23.100 s |
| `cudaLaunchKernel` | 15,907 | 8.196 s |
| `cuMemcpyHtoDAsync_v2` | 28,572 | 14.629 s |
| `cuMemcpyDtoDAsync_v2` | 10,241 | 4.352 s |
| `cuMemcpyDtoHAsync_v2` | 136 | 4.352 s |
| `cuMemAllocAsync` / `cuMemFreeAsync` | 101,452 each | 0.572 / 0.293 s |

Because Nsight cannot emit kernel rows on this host, an equivalent temporary
trace at Candle's CUDA function loader was used to attribute 58,295 direct
Candle kernel launches. The instrumentation was removed after capture; the raw
trace is `/tmp/issue46_f16_kernel_loader_trace.log`, SHA-256
`4ff82e2267c689f89cc7b8ee8d421cf2d8b9395448d75d71382a488cddb73e9b`.

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
duration remains unavailable on this host.

## Historical throughput evidence requiring final-head rerun

The existing exact 32/128/512-frame evidence predates the three Turing fixes
above (candle_sam3 `a90726b8`). Its 512-frame result was:

- F32: 2,387.145 s, 10,889 MiB peak device memory
- F16: 1,801.839 s, 7,686 MiB peak device memory
- time reduction: 24.52%
- throughput increase: approximately 32.48%
- F16/F32 mean/min IoU: 0.998639 / 0.994909

This demonstrates the expected margin over the Issue #46 performance target,
but it is not final-head certification. The exact 32/128/512 suite and device
memory capture must be rerun at `54b6cb59` or its final successor.

## Remaining red gates

1. Explain or deliberately disposition the sparse F16/F32 intermediate
   outliers without tuning thresholds to this result.
2. Explain the much broader F32/Facebook strict-port intermediate discrepancy;
   F16 cannot pass a Facebook envelope that F32 already fails.
3. Rerun exact 32/128/512 correctness, throughput, and peak device-memory
   measurements on the final Candle revision.
4. If per-kernel GPU duration is mandatory rather than equivalent
   kernel/operator/API attribution, rerun the committed workload on a host
   where Nsight GPU kernel tracing is supported.

Until those gates pass, the PR remains draft and the consumer plugin must
continue rejecting `SAM3_COMPUTE_DTYPE=f16`.
