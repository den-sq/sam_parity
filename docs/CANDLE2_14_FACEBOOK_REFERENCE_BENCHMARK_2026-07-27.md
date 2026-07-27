# CANDLE-2.14 Facebook SAM3 reference benchmark — 2026-07-27

Tracking issue: <https://github.com/den-sq/sam_parity/issues/48>

## Status

Complete. The default-dtype and forced-F32 32/128/512 ladders were measured on
the target host. The full 4,901-frame default arm was also attempted and has a
demonstrated host-memory-infeasible disposition.

## Reference revision, runtime, and private inputs

| Item | Recorded value |
|---|---|
| Reference implementation | `facebookresearch/sam3` |
| Reference commit | `84cc43bca4347b772f17d1078a1ddb4c054655c2` |
| Runtime | Python 3.12.12, PyTorch `2.7.0+cu126`, CUDA runtime 12.6, cuDNN 9.5.1 |
| GPU | NVIDIA Quadro RTX 5000 with Max-Q Design, 16,384 MiB, compute capability 7.5 |
| Driver | `581.60` |
| Host | Intel Xeon W-10855M, 70 GiB RAM, WSL2 Linux 6.6.87.2 |
| Medical-SAM3 checkpoint SHA-256 | `6e40bbaa739ac44e3e47dc6355ef6dedc560a30411377ad891f8af9e6df0dbd6` |
| Full fixture SHA-256 | `1a4fddfcdf2981d81602948f1bb06e1d30e55773b12e4e3234ef004533322155` |
| Full fixture | 4,901 × 200 × 200 grayscale `uint16`; 50 z-annotations; one trajectory |

The checkpoint and fixture payloads are private and are not committed. The
imported `sam3/` source directory is byte-clean at the recorded commit. The
checkout has an unrelated executed example-notebook modification outside the
imported runtime package.

The exact ladder fixtures are the same private inputs used by the Candle
32/128/512 matrix:

| Frames | Input SHA-256 |
|---:|---|
| 32 | `cd06100f8d8a2aaec5f2c91d0f3ae81c396b58c36f3ab53402a5e2c385b91df5` |
| 128 | `cb7a2b8170fc12cdd744f4bf2b86516f2fd409d675e6fabde91f0750660fadb6` |
| 512 | `2150b3db007f167e90f4081a8daa445b3d5508de5798d1513d632acedd59f858` |

These ladder fixtures repeat the same 16 biological frames and retain one
frame-0 positive annotation, exactly as in the recorded Candle matrix. The
full arm uses the original 4,901-frame stack and all 50 annotations.

## Configuration and measurement method

The reference default enters CUDA autocast with `torch.bfloat16`; model
parameters remain F32. This GPU has no native BF16 tensor-core execution.
PyTorch was built with FlashAttention support and has flash, memory-efficient,
math, and cuDNN SDPA toggles enabled. The harness queried
`aten::_fused_sdp_choice` on the exact tensors passed to every distinct SDPA
shape. Every observed default-arm call selected the **math** backend.
FlashAttention was therefore available in the PyTorch build but not selected
for this BF16/SM75 workload.

`torch.compile` was explicitly disabled. This matches the prior direct
Facebook tracker comparison and avoids including compilation/warmup policy in
the baseline.

The direct tracker was configured with:

- upstream async frame loading;
- `offload_video_to_cpu=true`, required to avoid materializing 4,901 resized
  input frames on the 16 GiB GPU;
- `offload_state_to_cpu=false`;
- GPU-resident, unbounded per-frame tracker outputs;
- one tracked object;
- streamed binary masks hashed in frame order and discarded by the harness.

The last point is important: the harness does **not** collect whole-run logits
and probability masks. It therefore avoids the separate 42.43 GiB
application-level retention mechanism identified in issue #30, while
preserving the reference tracker’s own unbounded per-frame state.

Every arm ran in an isolated process. CUDA was synchronized around model
startup, session setup, prompt setup, and propagation. A sampler thread inside
the inference process recorded direct `psutil.Process(os.getpid()).rss`,
PyTorch allocated/reserved bytes, and one-second `nvidia-smi` utilization,
board-memory, temperature, P-state, clock, and power samples. This attaches RSS
to the inference PID itself, not to a shell, profiler, or container wrapper.

Frame preprocessing matches the consumer path: `uint16 // 255`, saturation to
U8, grayscale-to-RGB expansion, and JPEG quality 90. Frame preparation is
recorded separately and excluded from model/inference timing.

## Default BF16-autocast results

| Frames | Startup s | Session s | Prompt s | Propagation s | Propagation fps | E2E s | E2E fps | Peak board VRAM MiB | Mean propagation GPU | Peak direct RSS MiB | Output SHA-256 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 32 | 23.499 | 0.024 | 2.343 | 72.754 | 0.439841 | 98.619 | 0.324480 | 11,658 | 99.47% | 13,774.3 | `0f86cd2a99927529eb806d5fa1a0980acdd2c207cba02a38894d032ebc1a88de` |
| 128 | 21.514 | 0.024 | 2.189 | 300.979 | 0.425279 | 324.707 | 0.394201 | 11,541 | 99.73% | 14,107.8 | `11603ea429ee01fc6cbdb17c9d0ccdaebaa69477353776d813040bc7bdde7e18` |
| 512 | 21.837 | 0.024 | 2.231 | 1,227.093 | **0.417246** | 1,251.186 | 0.409212 | 11,978 | 99.47% | 14,092.0 | `79a1f005be074bd59b47882dcdcdc05cb40c4e04ccc281f19e66e75cbb2f6df9` |

All three propagation phases remained in P0. The observed propagation clock
floors were 990, 855, and 825 MHz for 32, 128, and 512 frames respectively;
none reproduced the P3/300 MHz collapse from issue #35. The 512-frame peak
temperature was 77 °C.

The 512-frame arm is the primary sustained-rate anchor. At `0.417246 fps`, a
4,901-frame propagation projects to 11,746 seconds, or **3.26 hours**, before
prompt and startup overhead.

## Full 4,901-frame disposition

**The reference cannot complete this workload on the target host with its
available memory options.** CPU input offload avoids an immediate 16 GiB GPU
failure, but it relocates the resized-frame cache to host memory. Async loading
defers the allocation; it does not bound the eventual cache.

The run loaded at least 3,483 of 4,901 resized frames and propagated at least 31
frames before reaching the host-memory safety boundary:

| Measurement at disposition | Value |
|---|---:|
| Direct inference-PID RSS | **63,739.4 MiB** |
| System RAM | **69 / 70 GiB used** |
| System swap | **16 / 18 GiB used** |
| Peak board VRAM | 11,460 MiB |
| Peak PyTorch allocated / reserved | 6,894.8 / 9,004 MiB |
| End-to-end time before termination | 361.953 s |
| GPU under paging pressure | 8% utilization, P8, 300 MHz |

The process began in P0 at saturated utilization, then reached P3/P5/P8 and
300–375 MHz as host paging dominated. This is distinct from the issue #35
thermal collapse: it coincided with exhaustion of host RAM and 16 GiB of swap,
while temperature had fallen from a 78 °C peak to about 60–67 °C.

The run was stopped at that safety boundary rather than allowing further host
instability. Docker’s final state records `OOMKilled=true`. After termination,
host use recovered to 7.7 GiB RAM and 380 MiB swap. The reference process was
killed before it could finalize its in-memory partial output digest, so there
is no completed full-run output SHA-256; `result.json` records that
unavailability explicitly rather than presenting an empty or fabricated hash.

Leaving inputs on GPU is not an alternative on the 16 GiB target: 4,901
normalized 3 × 1008 × 1008 frames alone require roughly 55.7 GiB at F32 or
27.8 GiB at BF16, before model and tracker state. Offloading tracker state to
CPU would add further host pressure. The failure is therefore a demonstrated
reference memory limit on the matched host, not a missing retry.

## Forced-F32 disposition

Forced F32 is feasible and materially faster on SM75:

| Frames | Propagation s | Propagation fps | E2E s | Peak board VRAM MiB | Mean propagation GPU | Output SHA-256 |
|---:|---:|---:|---:|---:|---:|---|
| 32 | 45.421 | 0.704527 | 74.811 | 7,165 | 98.84% | `e82578d6ec996b35a3676b401024b6872490c8f1afe4ca7e69193c9d1bd47034` |
| 128 | 203.193 | 0.629942 | 224.900 | 7,283 | 99.76% | `13a26c38436d7cb4f48f25c8bddf64e2f24c309368b34a16bfebfaa2f4679a8d` |
| 512 | 830.264 | **0.616671** | 853.702 | 7,661 | 99.84% | `649473799daec339ea0d390b97d239ef6a4ee1000635fc4ce405eb903c10fb1e` |

All automatic-policy F32 arms selected **memory-efficient SDPA** rather than
the default BF16 arm’s math backend. F32 also used materially less workspace:
at 512 frames its peak board memory was 7,661 MiB versus 11,978 MiB for default
BF16. The 512-frame rate projects a 4,901-frame propagation to 7,948 seconds,
or **2.21 hours**, if the upstream frame cache could be bounded.

### Attention-backend ablation

A controlled 32-frame F32 rerun disabled flash, memory-efficient, and cuDNN
SDPA globally and left only math SDPA enabled. This holds the implementation,
weights, input, dtype, tracker state, and orchestration constant:

| Facebook F32 policy | Propagation s | Propagation fps | Peak board VRAM MiB | Output SHA-256 |
|---|---:|---:|---:|---|
| Automatic | 45.421 | 0.704527 | 7,165 | `e82578d6ec996b35a3676b401024b6872490c8f1afe4ca7e69193c9d1bd47034` |
| Math only | 45.517 | 0.703034 | 10,443 | `1f5c02b36290ed0856015ea0526be770add80dd8edb2772a7d283962312160e4` |

Forcing math changed throughput by only **-0.21%**, although it increased
board memory by 45.75% and changed the numerically sensitive output digest.
Attention backend selection therefore does not explain the Facebook/Candle F32
speed gap. The F32 gain over default BF16 is consistent with other
dtype-dependent kernels on SM75—this GPU has no native BF16 tensor-core
execution—rather than with SDPA selection alone. The automatic
memory-efficient backend is still valuable for workspace reduction.

The full F32 arm is not repeated because the demonstrated full-run failure is
the dtype-independent CPU resized-frame cache: it exhausts host RAM and swap
while GPU use remains 4.9 GiB below the card ceiling. F32 cannot repair that
input-retention mechanism.

## Comparison with the Candle candidate

The exact-final-head Candle matrix recorded in
`CANDLE2_F16_CUDA_CERTIFICATION_2026-07-23.md` used Candle
`95e0c186cdf7a7d51224659a103ce73294b7efad`, F32 compute, BF16 retained mask
memory, the bounded 32-state GPU-resident profile, and the same ladder inputs.
It is the longest synchronized candidate matrix currently available; it is
not mislabeled as the frozen F32/F32 retained-state configuration.

| Frames | Facebook default fps | Facebook forced-F32 fps | Candle F32 fps | Candle / default | Candle / forced-F32 |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.439841 | 0.704527 | 0.213855 | 48.62% | 30.35% |
| 128 | 0.425279 | 0.629942 | 0.213028 | 50.09% | 33.82% |
| 512 | **0.417246** | **0.616671** | **0.213972** | **51.28%** | **34.70%** |

At 512 frames, the default Facebook reference is 1.95× the Candle candidate’s
sustained propagation rate; the candidate gap is 48.72% of default-reference
throughput. Forced F32 widens that to 2.88× and a 65.30% gap. The 32-frame
math-only ablation rules out PyTorch's fused attention selection as the cause
of this wider F32 gap.

A matched synchronized stage diagnostic is recorded in
`CANDLE2_14_SPEED_DISCREPANCY_DIAGNOSTIC_2026-07-27.md`. It attributes 96.48%
of the F32 per-frame time gap to the image backbone and neck, with the
new-memory encoder a much smaller secondary source.

| 4,901-frame projection | Rate | Propagation time |
|---|---:|---:|
| Facebook default | 0.417246 fps | 3.26 h |
| Facebook forced F32 | 0.616671 fps | 2.21 h |
| Candle candidate | 0.213972 fps | 6.36 h |

The retired `0.68 fps` value is 1.63× the measured reference rate. It would
require Facebook SAM3 itself to complete the same propagation in about two
hours, while the measured reference projects to 3.26 hours. The constant is
therefore confirmed to be a timeout-derived requirement, not a reference-speed
criterion. Even against the faster forced-F32 control, `0.68 fps` remains
10.27% above measured Facebook throughput.

## Acceptance-criteria disposition

| Criterion | Disposition |
|---|---|
| Exact Facebook revision and private input identities | Recorded by commit and SHA-256 |
| Matched GPU/driver and 32/128/512/full ladder | Complete |
| Actual dtype, autocast, attention backend, and compile state | Recorded per arm and in every `result.json` |
| Startup, propagation, E2E, VRAM, GPU utilization, and direct inference-PID RSS | Recorded |
| Deterministic output digest | Recorded for every completed arm; correctly unavailable for the killed full arm |
| Full 4,901-frame completion or concrete memory reason | Host-memory infeasibility demonstrated and quantified |
| Forced-F32 arm if default is not F32 | Complete at 32/128/512 |
| Candle comparison | Reported against the longest synchronized candidate matrix, with its non-frozen retention profile labeled |

## Raw evidence and reproduction

Machine-readable results and one-second telemetry are committed under
`docs/candle2_14_reference_benchmark_2026-07-27/`. Each run directory contains:

- `environment.json`;
- `invocation.txt`;
- `result.json`;
- `telemetry.csv`.

The directory-level `summary.json` contains the compact cross-arm result table,
projections, and full-run disposition.

The reusable runner is
`python/sam3_parity/reference_benchmark.py`, exposed as
`sam3-reference-benchmark`. Every `invocation.txt` records the exact command,
source revision, checkpoint path, fixture path, frame-cache path, dtype arm,
and output directory. Private paths are operational provenance only; hashes,
not payloads, are the public identity.
