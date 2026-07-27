# CANDLE-2.14 SAM3 speed-discrepancy diagnostic — 2026-07-27

Tracking issue: <https://github.com/den-sq/sam_parity/issues/48>

## Result

The concise matched diagnostic attributes **96.48% of the measured
Candle–Facebook F32 frame-time gap to the image encoder**.

| Synchronized median component | Candle ms | Facebook ms | Candle / Facebook | Share of total gap |
|---|---:|---:|---:|---:|
| Image encoder (backbone + neck) | 3,596.781 | 1,157.088 | 3.11× | **96.48%** |
| Base tracker head | 27.990 | 8.943 | 3.13× | 0.75% |
| Previous-memory conditioning increment | 84.416 | 78.183 | 1.08× | 0.25% |
| New-memory encoder increment | 69.002 | 5.195 | 13.28× | 2.52% |
| **Estimated complete frame** | **3,778.188** | **1,249.408** | **3.02×** | **100%** |

The corresponding estimated rates are 0.2647 fps for current Candle main and
0.8004 fps for Facebook F32. This diagnostic is deliberately coarse: it
identifies the subsystem to profile next, not the individual encoder
operator. The next useful narrow test is therefore inside the image
backbone/neck rather than in video orchestration, history selection, or
attention-backend policy.

The unusually large new-memory encoder ratio is real but contributes only
63.8 ms of the 2,528.8 ms total gap. Optimizing it first cannot materially
close the observed discrepancy.

## What the diagnostic measures

Both runners use the same Medical-SAM3 checkpoint, preprocessed 200 × 200
biological JPEGs, normalized center-point seed, target frame, F32 compute,
one prior conditioning state, and CUDA synchronization around every sample.
Model construction, checkpoint loading, JPEG decoding, session creation, and
prompt setup are excluded from the component timings.

The four measured stages are:

1. `image_encoder`: image backbone and neck;
2. `tracker_base`: tracker head with no prior history and no new-memory
   encoding;
3. `tracker_with_history`: tracker head with one prior conditioning state and
   no new-memory encoding;
4. `tracker_full`: tracker head with that state and new-memory encoding.

The previous-memory increment is stage 3 minus stage 2. The new-memory
increment is stage 4 minus stage 3. Estimated complete-frame time is the image
encoder plus stage 4. Each result records all raw samples and uses their
median.

The seed is frame `000000.jpg`; the timed image is `000001.jpg`. Each stage
uses one untimed warmup and three timed samples in the recorded smoke. Five or
more samples should be used for a longer confirmation run.

## Reusable runners

- Facebook: `python/sam3_parity/speed_diagnostic.py`, exposed as
  `sam3-speed-diagnostic`;
- Candle: `rust/sam3-parity-cli/src/bin/sam3_speed_diagnostic.rs`;
- comparison: `python/sam3_parity/compare_speed_diagnostics.py`, exposed as
  `sam3-compare-speed-diagnostics`.

Machine-readable evidence is in
`docs/candle2_14_speed_diagnostic_2026-07-27/`:

- `facebook_f32.json`;
- `candle_f32.json`;
- `comparison_f32.json`.

## Candle build and run commands

The Rust workspace resolves its Candle dependencies from the adjacent local
checkout, `/home/dnorthover/ChengCode/candle_sam3`. The recorded build used
Candle main `32882747726e4803af494a6e67f47098f45c9893`:

```bash
cd /home/dnorthover/ChengCode/sam_parity_issue48
PATH=/usr/local/cuda-12.9/bin:/home/dnorthover/.cargo/bin:/usr/local/bin:/usr/bin:/bin \
CUDA_HOME=/usr/local/cuda-12.9 \
CUDA_COMPUTE_CAP=75 \
cargo build --release --features cuda --bin sam3_speed_diagnostic
```

The host-built binary requires glibc 2.39, so the recorded run used an
Ubuntu 24.04 shell and mounted the host CUDA 12.9 libraries:

```bash
docker run --rm --gpus all \
  -e LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  -v /home/dnorthover/ChengCode/sam_parity_issue48/target/release/sam3_speed_diagnostic:/bench:ro \
  -v /tmp/sam3_issue48_reference_frames_32:/frames:ro \
  -v /home/dnorthover/ChengCode/sam_parity_issue48/docs/candle2_14_speed_diagnostic_2026-07-27:/output \
  -v candle2-cert-old-cold:/volume:ro \
  -v /usr/local/cuda-12.9/targets/x86_64-linux/lib:/usr/local/cuda/lib64:ro \
  ubuntu:24.04 \
  /bench \
    --checkpoint /volume/sam3-segmentation/chkpts/medical_sam3.pt \
    --seed-frame /frames/000000.jpg \
    --frame /frames/000001.jpg \
    --output /output/candle_f32.json \
    --dtype f32 \
    --warmup 1 \
    --samples 3 \
    --num-frames 32 \
    --candle-revision 32882747726e4803af494a6e67f47098f45c9893
```

For a compatible host, `/bench ...` can be run directly; the container is not
part of the timed interval.

## Facebook and comparison commands

Inside the prepared Facebook/PyTorch environment:

```bash
cd /home/dnorthover/ChengCode/sam_parity_issue48/python
PYTHONPATH=. /home/dnorthover/miniconda3/envs/sam3/bin/python \
  -m sam3_parity.speed_diagnostic \
  --checkpoint /volume/sam3-segmentation/chkpts/medical_sam3.pt \
  --frames-dir /tmp/sam3_issue48_reference_frames_32 \
  --target-frame 1 \
  --warmup 1 \
  --samples 3 \
  --sam3-revision 84cc43bca4347b772f17d1078a1ddb4c054655c2 \
  --output ../docs/candle2_14_speed_diagnostic_2026-07-27/facebook_f32.json
```

Then produce the gap attribution:

```bash
cd /home/dnorthover/ChengCode/sam_parity_issue48/python
PYTHONPATH=. /home/dnorthover/miniconda3/envs/sam3/bin/python \
  -m sam3_parity.compare_speed_diagnostics \
  --candle ../docs/candle2_14_speed_diagnostic_2026-07-27/candle_f32.json \
  --facebook ../docs/candle2_14_speed_diagnostic_2026-07-27/facebook_f32.json \
  --output ../docs/candle2_14_speed_diagnostic_2026-07-27/comparison_f32.json
```

## Environment qualification

Both results were taken on the same Quadro RTX 5000 Max-Q (SM75) and driver
581.60. Facebook used PyTorch 2.7.0+cu126 with its CUDA 12.6 runtime; Candle
used locally built CUDA 12.9 kernels. Thus this is a matched model/input/dtype
subsystem diagnostic, not a claim that the framework runtime stacks are
binary-identical. The 3.11× image-encoder result is large enough that the
minor runtime-version difference does not alter the localization conclusion.

The checkpoint SHA-256 is
`6e40bbaa739ac44e3e47dc6355ef6dedc560a30411377ad891f8af9e6df0dbd6`.
Facebook is pinned to
`84cc43bca4347b772f17d1078a1ddb4c054655c2`; Candle is pinned to
`32882747726e4803af494a6e67f47098f45c9893`.

## Validated local Candle image-encoder profile

`scripts/profile_candle_image_encoder.sh` builds a dedicated CUDA binary, warms
up the Candle image encoder, and places CUDA profiler start/stop calls around
exactly one `encode_image_features` invocation. It then runs that binary under
the locally installed Nsight Systems and writes both the `.nsys-rep` and a text
kernel summary.

Run it from the parity checkout:

```bash
cd /path/to/sam_parity
SAM3_CHECKPOINT=/path/to/sam3-compatible-checkpoint.pt \
SAM3_PROFILE_FRAMES_DIR=/path/to/prepared-jpegs \
scripts/profile_candle_image_encoder.sh
```

Useful overrides include:

```bash
WARMUP=1 \
PROFILE_STEM=candle_image_encoder_trial \
CHECKPOINT_PATH=/path/to/sam3-compatible-checkpoint.pt \
FRAMES_DIR=/path/to/prepared-jpegs \
scripts/profile_candle_image_encoder.sh
```

`CHECKPOINT_PATH` falls back to `SAM3_CHECKPOINT`, and `FRAMES_DIR` falls back
to `SAM3_PROFILE_FRAMES_DIR`. Medical-SAM3 and upstream `sam3.pt` have the same
encoder architecture and tensor shapes, so their kernel graph is the same;
weights do not affect kernel selection or launch geometry.

The wrapper applies NVIDIA's required WSL workaround
`CuptiUseRawGpuTimestamps=false` to the user Nsight configuration. Without it,
Nsight records CUDA API calls but discards the GPU kernel timestamps. It also
uses `cuda-sw` explicitly because the target GPU is Turing rather than
Blackwell.

The validated one-warmup capture took 3,585.822 ms and produced:

| Kernel family | GPU time | Share |
|---|---:|---:|
| `conv_transpose2d_f32` (6 launches) | 1,656.996 ms | **47.4%** |
| `volta_sgemm_128x64_tn` | 442.420 ms | 12.7% |
| `volta_sgemm_128x128_tn` | 355.714 ms | 10.2% |
| `softmax_f32` | 251.802 ms | 7.2% |
| `volta_sgemm_64x64_nn` | 199.887 ms | 5.7% |
| Other kernels | 588.1 ms | 16.8% |

This makes the dual-neck transposed convolutions the first concrete target,
ahead of the explicit attention GEMM/softmax path. The raw `.nsys-rep`, SQLite,
and generated text reports remain local because they are tool-version-specific
artifacts; the portable measurements are recorded in this report.

## Matching local Facebook image-encoder profile

`scripts/profile_facebook_image_encoder.sh` performs the same capture against
official Facebook SAM3 commit
`84cc43bca4347b772f17d1078a1ddb4c054655c2`. It uses the same checkpoint,
frame 1, forced F32 compute, warmup policy, CUDA profiler boundaries, Nsight
version, GPU, driver, and WSL timestamp workaround as the Candle capture.

Run it with:

```bash
cd /path/to/sam_parity
SAM3_CHECKPOINT=/path/to/sam3-compatible-checkpoint.pt \
SAM3_PROFILE_FRAMES_DIR=/path/to/prepared-jpegs \
scripts/profile_facebook_image_encoder.sh
```

The wrapper uses an installed `sam3` package by default. Set
`UPSTREAM_ROOT=/path/to/facebookresearch/sam3` to profile a source checkout
instead, and set `PYTHON_BIN` when that checkout uses a dedicated environment.

The validated one-warmup captures provide the following direct comparison:

| Measurement | Candle | Facebook F32 | Candle / Facebook |
|---|---:|---:|---:|
| Captured wall time | 3,585.822 ms | 1,205.516 ms | 2.97× |
| Summed GPU kernel time | 3,495.453 ms | 1,205.029 ms | 2.90× |
| Kernel launches | 1,225 | 579 | 2.12× |
| Six transposed-convolution launches | 1,656.996 ms | 27.085 ms | **61.18×** |
| Attention core | 545.237 ms | 242.701 ms | 2.25× |
| Principal linear/MLP SGEMMs | 798.133 ms | 788.037 ms | 1.01× |

Facebook implements the six transposed convolutions through cuDNN backward-data
(`dgrad`) kernels. Candle's six custom `conv_transpose2d_f32` kernels account
for **71.16% of the total summed-kernel-time gap** by themselves.

Facebook's 32 fused memory-efficient attention kernels total 242.701 ms.
Candle's corresponding 32 `softmax_f32`, 32 `volta_sgemm_64x64_nn`, and 32
`volta_sgemm_64x64_tn` launches total 545.237 ms. This is a real secondary
target, but substantially smaller than transposed convolution.

The principal transformer linear/MLP SGEMMs are already essentially equal in
aggregate. Optimizing those GEMMs first would therefore not address the
observed encoder discrepancy.
