# CANDLE-2.9 candidate certification — 2026-07-21

Tracking issue: <https://github.com/den-sq/sam_parity/issues/35>

## Candidate and ancestry

- Candle candidate: `2cf6179b4f10b9ddfb973f1b154931f68e7a9f56`
- Plugin candidate: `dd03d0842f087a166e7000ee155c7512b9db00bf`
- Required Candle 0.11 integration merge: `c11c900354467d50985a78a1895945199c9f4ecb`
- Integration parents: fork `8fb0a15e148353129d76987bbfd5f751f8c336d9`, accepted integration branch `71fad7110bcd1d861102ef256a76eeeac1300bce`
- `git merge-base --is-ancestor c11c9003 2cf6179b` succeeds.
- The mixed `issue-10-encoder-drift` branch is excluded.

The candidate includes the merged changes from:

- <https://github.com/den-sq/sam_parity/issues/39> through Candle merge `1ee4e11afc6e6a78a3d709c9c7b526b041015b8f` and plugin merge `015083247831ffe40c028198c50c8ef80a56b8f0`.
- <https://github.com/den-sq/sam_parity/issues/41> through Candle merge `f49f707508892743440c339c30feb8f92a8dfe44` and plugin merge `576c66a883344837534d8f4167ad6d704bf4249d`.

<https://github.com/den-sq/sam_parity/issues/42> is explicitly excluded and remains deferred until after <https://github.com/den-sq/sam_parity/issues/36>. Its draft branches are not ancestors of either candidate SHA.

The plugin candidate is stacked on progress/reconnect commit `43203f2e7a5f27e4d39dfa1c9736dc1c050fc51a` from <https://github.com/den-sq/sam_parity/issues/38>; that dependency must land first or the candidate must be rebased without changing the certified runtime content.

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

These pre-selection CPU-offload runs used plugin
`e41b1ca8e4c4626b522c2ac72519f9a9141e773a` and:

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

This is a favorable storage tradeoff, but it is not selected for the provisional
default because the F32 GPU-resident controls are byte-identical and remain
comfortably within the VRAM gate.

## Bounded GPU-resident selection

The exact Candle candidate and pre-default plugin image were run on a repeated
64-frame medical fixture with F32 compute, F32 retained state, feature cache 1,
trim enabled, a 32-state non-conditioning bound, and hotstart delay 0.

| Profile | Propagation | Peak reported VRAM | Output SHA-256 |
|---|---:|---:|---|
| GPU resident, cool control | 318.48 s | 10,691 MiB | `895aceb638295c0b46e7122fea5ae588ec8bc9d9cef657fab1169a8fa41ff8c6` |
| CPU offload | 384.43 s | 9,098 MiB | `895aceb638295c0b46e7122fea5ae588ec8bc9d9cef657fab1169a8fa41ff8c6` |
| GPU resident, heat-soaked control | 326.59 s | 10,647 MiB | `895aceb638295c0b46e7122fea5ae588ec8bc9d9cef657fab1169a8fa41ff8c6` |

The matched heat-soaked GPU-resident control used 15.0% less propagation time
than CPU offload. All runs retained 33 tracker states including 32
non-conditioning states, and the selected GPU-resident profile remained below
the 14 GiB gate.

A 512-frame follow-up held an approximately 10.5 GiB VRAM plateau through 25%
progress, but it was stopped after the host driver collapsed to P3/300 MHz at
100% utilization and requested a 50 W power limit instead of its 90 W default.
That run is memory evidence only, not valid sustained-throughput evidence.

## Hot-path and synchronization disposition

The Candle candidate:

- selects model compute dtype at load and preserves it across image, attention, prompt, tracker, and mask-memory boundaries;
- stores retained mask-memory in the selected storage dtype and casts it on read;
- skips non-overlap score extraction unless multiple visible objects require it;
- skips confirmation-score reads when the confirmation gate is disabled;
- tensorizes multi-object score ranking and binary threshold reconstruction;
- removes duplicate foreground checks on the ordinary single-object path.

Both accepted eight-frame runs recorded `postprocess_score_scalar_reads=0` and `postprocess_foreground_scalar_reads=8`. The remaining foreground read is one required output-presence decision per frame; disabled/single-object non-overlap adds no score reads.

Two-frame backbone batching was not selected. The candidate already fits the VRAM gate, and batching was not justified by synchronized evidence before freeze.

## Dependency and verification record

The plugin uses `0.11.0` for `candle-core`, `candle-nn`, and `candle-transformers`; its lockfile is already on the reviewed 0.11 dependency set. Both Docker pin declarations, the smoke default, and scaffold documentation name the exact Candle candidate SHA.

Completed verification:

- `cargo test -p candle-transformers sam3::video`: 21 passed at the final Candle candidate.
- `cargo test -p candle-transformers --lib`: passed earlier in the candidate stack; the final additions are covered by the focused suite, including an F16 CPU-mask-cleanup regression.
- `CUDARC_CUDA_VERSION=12040 CUDA_COMPUTE_CAP=75 cargo check -p candle-transformers --features cuda`: passed for the candidate stack (11 PTX and 15 CUDA kernels).
- Plugin `cargo test` against the candidate Candle worktree: 77 passed; annotation contract passed; checkpoint-load test ignored unless a private checkpoint path is supplied.
- Focused plugin Candle-SAM3 module after changing the defaults: 13 passed,
  including default/fallback parsing and video lifecycle coverage.
- `sam_parity` workspace tests: 16 passed, 10 fixture investigations ignored; contract tests 3 passed.
- `cargo test -p sam3-parity-cli --features full-parity --no-run`: passed.
- Exact-SHA F32/BF16 and F32/F32 GPU smokes: passed geometry, binary-output, CUDA-device, image-revision, and checkpoint-revision checks.

## Freeze disposition

The proposed provisional freeze is Candle
`2cf6179b4f10b9ddfb973f1b154931f68e7a9f56` plus plugin
`dd03d0842f087a166e7000ee155c7512b9db00bf`, selecting bounded GPU-resident F32
compute and retained state. CPU offload and BF16 retained storage remain explicit
fallback/benchmark controls. <https://github.com/den-sq/sam_parity/issues/35>
should remain open until sustained throughput can be measured without the host
power-policy collapse and reviewers either approve a revised runtime target or
require additional bounded optimization before
<https://github.com/den-sq/sam_parity/issues/36> freezes the unchanged SHAs.
