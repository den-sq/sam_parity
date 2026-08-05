# CANDLE-2.9 candidate certification — 2026-07-21

Tracking issue: <https://github.com/den-sq/sam_parity/issues/35>

Last updated: 2026-08-05

## Current production candidate freeze — 2026-08-05

The original CANDLE-2.9 candidate below remains the historical certification
baseline. The production release has since incorporated the accepted CUDA
performance work, so the following exact stack supersedes that baseline as the
candidate for the next performance measurement:

- Candle: `c0400c6513c21655828bb92633cc190a3501a6f6`
- Performance implementation anchor: `ff48ac5102c3d9da8638110d5b8da5141f5d52bb`
  from <https://github.com/den-sq/candle_sam3/pull/13>
- Plugin/release source: `e0d2d82e7687f940e5481ac2252bf626aae1be1f`
- Plugin release: `ChengLabResearch/ouroboros_autoseg_plugin@v0.4.0-beta.3`
- Production backend image:
  `ghcr.io/chenglabresearch/ouroboros-autoseg-backend@sha256:238346625628119a310dce102f813a4386011a3265d9e29775a7a3449fde49ce`
- Compiled Candle features: `cuda,cudnn`
- Compute / retained dtype: F32 / F32
- State profile: bounded GPU resident
- Feature cache entries: 1
- Non-conditioning tracker-state limit: 32
- Hotstart delay: 0
- Trim past non-conditioning memory: enabled
- Target: NVIDIA Quadro RTX 5000 Max-Q, compute capability 7.5,
  16,384 MiB, driver `581.60`

The current Candle SHA descends both the original accepted candidate
`71690361a0e4eb839cfc22a52fcdf5cfbf047f0a` and the performance implementation
anchor. The current plugin SHA descends the original plugin candidate
`7438668043d5021d016e5e0402012657dab69309`. The released image was built from
the current plugin SHA with the exact Candle SHA and `cuda,cudnn` features
above.

This is an exact candidate freeze, not a current end-to-end performance claim.
Do not change a revision, image digest, build feature, dtype, state profile, or
runtime control during the comparison. Any changed value defines a distinct
candidate and must be reported separately.

### Current performance evidence and missing test

The historical `0.213972 fps` 512-frame Candle result predates the cuDNN
transposed-convolution route, fused F32/SM75 attention, fused F32 RoPE, and
encoder layout reuse now in the production image. It remains evidence for its
recorded historical revision and F32/BF16-retained configuration; it is not a
measurement of the frozen production candidate above.

The matched encoder diagnostic at the production optimization anchor records:

| Encoder measurement | Pre-optimization Candle | Current Candle | Facebook F32 |
|---|---:|---:|---:|
| Captured wall time | 3,585.822 ms | 1,341.011 ms | 1,205.516 ms |
| Summed GPU kernel time | 3,495.453 ms | 1,323.803 ms | 1,205.029 ms |
| Kernel launches | 1,225 | 659 | 579 |

The former dominant transposed-convolution and attention gaps are closed in
that diagnostic. The remaining measured encoder residual is primarily ordinary
linear-layer bias/layout plumbing: 215 standalone `badd_f32` kernels totaling
180.383 ms and 82 `ucopy_f32` materializations totaling 66.712 ms. This stage
evidence narrows any later optimization investigation, but it cannot be
substituted for an end-to-end production measurement.

<https://github.com/den-sq/sam_parity/issues/58> owns the required exact-SHA
test. It must run the unchanged frozen production stack on the established
32/128/512-frame ladder, capture end-to-end and propagation timing plus the
specified GPU, memory, output, and state-retention evidence, and repeat the
five-stage synchronized diagnostic. Results must be compared with both measured
Facebook reference arms (`0.417246 fps` default BF16 autocast and `0.616671 fps`
forced F32 at 512 frames) under a qualified hardware state. The old `0.68 fps`
threshold remains retired. The resulting evidence, rather than the historical
throughput figure or an extrapolation from encoder timing, will decide whether
further optimization should be deferred.

F16 remains excluded from this candidate. Its correctness/certification work is
separate and does not alter the F32/F32 production freeze.

## Original candidate and ancestry (historical)

- Candle candidate: `71690361a0e4eb839cfc22a52fcdf5cfbf047f0a`
- Plugin candidate: `7438668043d5021d016e5e0402012657dab69309`
- Required Candle 0.11 integration merge: `c11c900354467d50985a78a1895945199c9f4ecb`
- Integration parents: fork `8fb0a15e148353129d76987bbfd5f751f8c336d9`, accepted integration branch `71fad7110bcd1d861102ef256a76eeeac1300bce`
- `git merge-base --is-ancestor c11c9003 71690361` succeeds.
- The mixed `issue-10-encoder-drift` branch is excluded.

The candidate includes the merged changes from:

- <https://github.com/den-sq/sam_parity/issues/39> through Candle merge `1ee4e11afc6e6a78a3d709c9c7b526b041015b8f` and plugin merge `015083247831ffe40c028198c50c8ef80a56b8f0`.
- <https://github.com/den-sq/sam_parity/issues/41> through Candle merge `f49f707508892743440c339c30feb8f92a8dfe44` and plugin merge `576c66a883344837534d8f4167ad6d704bf4249d`.

<https://github.com/den-sq/sam_parity/issues/42> is explicitly excluded and remains deferred until after <https://github.com/den-sq/sam_parity/issues/36>. Its draft branches are not ancestors of either candidate SHA.

The progress/reconnect dependency from
<https://github.com/den-sq/sam_parity/issues/38> merged through
<https://github.com/ChengLabResearch/ouroboros_autoseg_plugin/pull/49> at merge
`7239c9c70061c3e380e04eb647d21f7304870425`. The plugin candidate contains both
that merge and accepted AC6 coverage commit
`0e8196276289c2a4cb27258fb6fa09a2e5b1a575`.

## Selected configuration

- Compute dtype: F32
- Retained mask-memory dtype: F32
- State profile: bounded GPU resident
- Feature cache entries: 1
- Non-conditioning tracker-state limit: 32
- Hotstart delay: 0
- Trim past non-conditioning memory: enabled
- CUDA target: compute capability 7.5, Quadro RTX 5000 with Max-Q Design, 16,384 MiB
- CUDA runtime/driver: 12.4.1 / 581.60

F16 compute is rejected. It did not complete the eight-frame video fixture after bounded compatibility fixes: the last disposition run reached frame 1 (`Inference=12%`) and failed at another F32/F16 convolution boundary. Because it did not complete, no mask-parity or material-speedup claim is possible. The plugin rejects `SAM3_COMPUTE_DTYPE=f16` rather than exposing a known-failing mode. BF16 compute was not evaluated because compute capability 7.5 has no native BF16 tensor-core execution.

## Pre-selection retained-dtype certification

These pre-selection CPU-offload runs used Candle
`2cf6179b4f10b9ddfb973f1b154931f68e7a9f56`, plugin
`e41b1ca8e4c4626b522c2ac72519f9a9141e773a`, and:

- image `sha256:67956d26e644e3ca620243283647165832b73156cddeee7221429cb1fec4b4f1`
- eight `200 x 200` frames derived from the private representative Medical-SAM3 fixture
- input SHA-256 `6a861cb25d8931b3d111639b6694ed9ae5cbf6a5051ac1b72c0457be16e61823`
- checkpoint SHA-256 `6e40bbaa739ac44e3e47dc6355ef6dedc560a30411377ad891f8af9e6df0dbd6`
- one tracked object, streamed output, and the same device/configuration other than retained dtype

The checkpoint and input fixture are private test artifacts and are not redistributed by this repository.

| Compute | Retained | Wall telemetry | Effective rate | Peak VRAM | Tracker-state CPU bytes | Total session CPU bytes | Output SHA-256 |
|---|---|---:|---:|---:|---:|---:|---|
| F32 | BF16 | 173 s | 0.0462 fps | 7,537 MiB | 18,587,680 | 476,687,392 | `3b585209390c37ee9d706dba7deba404b78140e5ba011eba47cc79cc57081589` |
| F32 | F32 | 176 s | 0.0455 fps | 7,505 MiB | 23,896,096 | 481,995,808 | `2f6c2e306196ef3a46ce52e4baabe2c48b0e46c0c086e22077c290a1ef46ca03` |

The short-run wall telemetry includes model startup and is not an authoritative
long-stack throughput estimate. It does prove the 14 GiB VRAM gate with more
than 6 GiB headroom. The previously inherited approximately 0.212 fps
512-frame result is not accepted for the current exact candidate.

BF16 retained storage saved 5,308,416 bytes (22.2%) of tracker-state CPU memory in this eight-state run. It changed 23 of 320,000 binary output pixels relative to F32 retained state, with global IoU `0.9999084544`; per-frame IoU was:

```text
1.0, 1.0, 0.9999675061, 0.9999031477,
0.9999686619, 0.9999687373, 1.0, 0.9994762140
```

This is a favorable CPU-offload storage tradeoff, but it is not selected for the
provisional default because the F32 GPU-resident controls are byte-identical and
remain comfortably within the VRAM gate. This table does not certify
GPU-resident BF16 storage. The final Candle candidate fixes GPU-resident retained
storage semantics and casts packed retained features back to compute dtype on
read. The packed execution cache is kept in compute dtype so BF16 retained state
does not invoke Ampere-only BF16 indexing or copy kernels on the target Turing
GPU; the authoritative retained tensors remain in the requested storage dtype.
BF16 remains an opt-in control that requires fresh parity/performance evidence
before selection; F32 is the library, example, plugin, and parity-CLI default.

## Bounded GPU-resident selection

The immediate predecessor Candle/plugin candidate was run on a repeated 64-frame
medical fixture with F32 compute, F32 retained state, feature cache 1, trim
enabled, a 32-state non-conditioning bound, and hotstart delay 0. The final
candidate adds the BF16 packed-history cast-before-index boundary, keeps the
packed execution cache in compute dtype, makes the selected F32 retained dtype
the public default, restores blank-frame confirmation-reset semantics, and adds
coverage. On the selected F32 configuration the packed conversions are
equal-dtype no-ops, the confirmation gate is disabled, and plugin configuration
was already explicit F32. These results are therefore carried-forward selection
evidence, not an exact-final-SHA throughput run.

| Profile | Propagation | Peak reported VRAM | Output SHA-256 |
|---|---:|---:|---|
| GPU resident, cool control | 318.48 s | 10,691 MiB | `895aceb638295c0b46e7122fea5ae588ec8bc9d9cef657fab1169a8fa41ff8c6` |
| CPU offload | 384.43 s | 9,098 MiB | `895aceb638295c0b46e7122fea5ae588ec8bc9d9cef657fab1169a8fa41ff8c6` |
| GPU resident, heat-soaked control | 326.59 s | 10,647 MiB | `895aceb638295c0b46e7122fea5ae588ec8bc9d9cef657fab1169a8fa41ff8c6` |

The matched heat-soaked GPU-resident control used 15.0% less propagation time
than CPU offload. All runs retained 33 tracker states including 32
non-conditioning states, and the selected GPU-resident profile remained below
the 14 GiB gate.

For cross-issue measurement context only, the 512-frame F32 arm on
<https://github.com/den-sq/candle_sam3/pull/10> measured `0.214482 fps`
propagation and `0.211483 fps` end-to-end with peak VRAM 10,889 MiB. That run
used F32 compute with **BF16 retained mask memory**; it is not an exact
frozen-configuration result for this F32/F32 candidate. Exact selected-
configuration throughput evidence tops out at the 64-frame A-B-A above
(318.48 s / 326.59 s, approximately 0.20 fps). The 512-frame result is retained
only as a configuration-labelled anchor for
<https://github.com/den-sq/sam_parity/issues/49>.

A 512-frame follow-up held an approximately 10.5 GiB VRAM plateau through 25%
progress, but it was stopped after the host driver collapsed to P3/300 MHz at
100% utilization and requested a 50 W power limit instead of its 90 W default.
That run is memory evidence only, not valid sustained-throughput evidence.

## Hot-path and synchronization disposition

The Candle candidate:

- selects model compute dtype at load and preserves it across image, attention, prompt, tracker, and mask-memory boundaries;
- stores retained mask-memory in the selected storage dtype and casts both packed and unpacked memory back to compute dtype on read;
- skips non-overlap score extraction unless multiple visible objects require it;
- skips confirmation-score reads when the confirmation gate is disabled;
- tensorizes multi-object score ranking and binary threshold reconstruction, normalizing narrow score tensors to F32 before ranking;
- removes duplicate foreground checks on the ordinary single-object path.

Both accepted eight-frame runs recorded `postprocess_score_scalar_reads=0` and `postprocess_foreground_scalar_reads=8`. The remaining foreground read is one required output-presence decision per frame; disabled/single-object non-overlap adds no score reads.

When the optional confirmation gate is enabled, blank frames still update and
reset the confirmation streak before being filtered, preserving the prior state
machine. Enabling output non-overlap with one visible object produces the same
binary-mask reconstruction as the ordinary path; this corrects the former
single-object raw-mask special case. Mixed-precision boundaries also include
explicit attention-output, image-input, prompt-encoder, F32 Q/K/V-softmax, and
mask-memory-backbone weight alignment casts; equal-dtype F32 calls are no-op
clones.

Two-frame backbone batching was not selected. The candidate already fits the VRAM gate, and batching was not justified by synchronized evidence before freeze.

## Throughput attribution and target disposition

No material throughput speedup is claimed for this candidate. The separate
512-frame F32/BF16 measurement above is `+5.1%` against the `0.204 fps` warm
baseline from <https://github.com/den-sq/sam_parity/issues/30>, and it does not
use the exact selected retained dtype. The synchronized evidence instead
attributes the material result to memory behavior: retained history plateaus at
33 states, the selected configuration remains below the 14 GiB gate, and
GPU-resident state avoids the measured CPU-offload penalty.

The synthesized `0.68 fps` threshold is retired from this issue. It was derived
from the pre-existing 7,200-second harness timeout, not from a measured Facebook
SAM3 reference or a user requirement. Reference-speed measurement is delegated
to <https://github.com/den-sq/sam_parity/issues/48>; cross-candidate,
user-specific measurement and target derivation are delegated to
<https://github.com/den-sq/sam_parity/issues/49>.

## Dependency and verification record

The plugin uses `0.11.0` for `candle-core`, `candle-nn`, and `candle-transformers`; its lockfile is already on the reviewed 0.11 dependency set. Both Docker pin declarations, the smoke default, and scaffold documentation name the exact Candle candidate SHA.

Completed verification:

- `cargo test -p candle-transformers sam3::video`: 22 passed at final Candle `71690361`.
- `cargo test -p candle-transformers --lib`: 51 passed, 3 ignored at final Candle `71690361`.
- `CUDARC_CUDA_VERSION=12040 CUDA_COMPUTE_CAP=75 cargo check -p candle-transformers --features cuda`: passed for the candidate stack (11 PTX and 15 CUDA kernels).
- Plugin `cargo test` at predecessor plugin `dd03d084` against the then-current candidate Candle worktree: 77 passed; annotation contract passed; checkpoint-load test ignored unless a private checkpoint path is supplied.
- Focused plugin Candle-SAM3 module after changing the defaults: 13 passed,
  including default/fallback parsing and video lifecycle coverage.
- `sam_parity` workspace tests: 16 passed, 10 fixture investigations ignored; contract tests 3 passed.
- `cargo test -p sam3-parity-cli --features full-parity --no-run`: compile-only check passed; the private-fixture video parity test was not executed.
- Pre-selection predecessor-SHA F32/BF16 and F32/F32 GPU smokes: passed
  geometry, binary-output, CUDA-device, image-revision, and checkpoint-revision
  checks; their CPU-offload scope is recorded above.
- Final Candle head regressions: the packed BF16-to-F32 cast-on-read test,
  compute-dtype packed-cache test, and caller-level CPU retained-storage test
  passed; the SAM3 video suite passed 22/22; the full library passed 51 with 3
  ignored; and both Turing-specific packed-cache CUDA tests passed on compute
  capability 7.5 after compiling 11 PTX and 15 CUDA kernels under CUDA 12.4.
- Final plugin head against the final Candle worktree: backend library tests
  passed 78/78; frontend reconnect tests passed 4/4; lint passed. The accepted
  initial-hotstart reconnect tests from
  <https://github.com/ChengLabResearch/ouroboros_autoseg_plugin/pull/49> remain
  present.

## Freeze disposition

The original provisional freeze was Candle
`71690361a0e4eb839cfc22a52fcdf5cfbf047f0a` plus plugin
`7438668043d5021d016e5e0402012657dab69309`. It remains the historical CANDLE-2.9
certification baseline.

The current production performance candidate is the exact Candle, plugin,
release, backend-image digest, feature set, and bounded GPU-resident F32/F32
configuration recorded in **Current production candidate freeze — 2026-08-05**.
It is frozen provisionally pending the unchanged-candidate 32/128/512-frame
measurement and refreshed five-stage diagnostic in
<https://github.com/den-sq/sam_parity/issues/58>. No current-production
end-to-end throughput is claimed before that evidence is recorded.

CPU offload and BF16 retained storage remain explicit fallback/benchmark
controls; F16 remains unselected. After the frozen candidate is measured and
accepted, the full biological-stack acceptance remains
<https://github.com/den-sq/sam_parity/issues/36>, while user-specific target
disposition remains <https://github.com/den-sq/sam_parity/issues/49>.
