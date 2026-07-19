# CANDLE-2 Candle 0.11 certification report

Date: 2026-07-19

## Verdict

The integration merge, native tests, parity contracts, official-reference CUDA
replay, target-GPU smoke, performance gate, and temporary consumer build all
pass. The same-fork GitHub CUDA workflow cannot acquire its private upstream
runner group, so target-hardware manual evidence is recorded instead.

The integration is **not yet certified for merge** because the representative
Medical-SAM3 point-prompt output moves materially farther from a newly exported
result from the official Facebook tracker. The pre-upgrade result has IoU
`0.961942` against the official output; Candle 0.11 falls to `0.911318` and adds
36,370 pixels relative to the control. This unexplained regression requires a
parity disposition or correction before issue #40 can be closed.

Correction: an earlier version of this report incorrectly staged the 16-bit
TIFF for the official exporter using a direct Pillow RGB conversion, clipping
the input frames to white. The results below supersede that invalid run. The
corrected export uses the plugin's exact integer scaling, Rust `image` crate,
RGB expansion, and JPEG encoder settings.

## Revisions and provenance

| Component | Revision |
|---|---|
| fork parent | `8fb0a15e148353129d76987bbfd5f751f8c336d9` |
| upstream Candle 0.11 parent | `31f35b147389700ed2a178ee66a91c3cc25cc80d` |
| merge base | `3df8203a2ab0f7d12866ef392d5ea7504b0255e4` |
| integration merge | `6a86b8ea07c87050ffb23d37b1ad407ac55f6acb` |
| certified PR head | `71fad7110bcd1d861102ef256a76eeeac1300bce` |
| pre-upgrade Candle control | `770d20ca8db4f834ba4c89c845bca196fbfc97ea` |
| plugin control | `c8d3c0218acde2c31f77f9a37d630368c83667cc` |
| plugin compatibility commit | `742a7910f57b83ff3f66ba06c7309231c9e4d822` |
| Facebook SAM3 source | `84cc43bca4347b772f17d1078a1ddb4c054655c2` |

The upstream parent is an ancestor of the integration head and there are zero
upstream-only commits relative to the fixed 0.11 SHA. The merge has the two
recorded parents and preserves both histories.

## Merge and dependency review

The sole textual conflict, `candle-transformers/Cargo.toml`, is the additive
union required by issue #40. The three overlapping files that Git auto-merged
were reviewed semantically:

- `candle-nn/src/ops.rs`: upstream's Metal buffer migration, RMSNorm CPU
  scheduling split, SDPA sequence thresholds, and byte-offset updates are
  preserved. The RMSNorm per-row arithmetic is unchanged; only serial versus
  Rayon scheduling is selected by row count. Fork SAM3 operations remain
  present and reachable.
- `candle-nn/tests/ops.rs`: upstream's large-magnitude RMSNorm finite/parity
  coverage is registered. The complete NN suite passes.
- `candle-transformers/src/models/mod.rs`: downstream `sam3` and upstream
  `lfm2`/`quantized_lfm2` exports are all retained.

`cargo tree -d` shows no duplicate Candle, `image`, or `tokenizers` versions.
The material duplicates are upstream transitive families (`safetensors` 0.4.5
through `ug` versus 0.8.0, `gemm` 0.18 versus 0.19, and `fancy-regex` 0.14
through `tokenizers` versus 0.18); none is a second fork-specific dependency.

`cargo fmt --all -- --check` still reports 15 pre-existing downstream SAM3
files. Their intersection with merge-touched files is empty. The explicit
disposition for issue #40 is to accept this as pre-existing formatting debt and
not mix a repository-wide formatting rewrite into the ancestry-preserving
integration merge.

## Native and parity gates

The following pass against the certified head:

```text
cargo check -p candle-transformers --features sam3-parity-support
cargo test -p candle-core
cargo test -p candle-nn
cargo test -p candle-transformers sam3
git diff --check
```

The `sam_parity` workspace was temporarily pointed at the Candle 0.11 worktree.
These gates pass:

```text
cargo test --workspace
  CLI: 16 passed, 10 ignored; contracts: 3 passed
PYTHONPATH=python python -m pytest python/sam3_parity/tests -q
  3 passed
cargo test -p sam3-parity-cli --features full-parity --no-run
```

The generated Candle 0.11 lockfile diff was reviewed and contained the expected
workspace transitions, including Candle 0.10.2 to 0.11.0, `cudarc` 0.19.6 to
0.19.8, `safetensors` 0.7 to 0.8, and `hf-hub` 0.4.3 to 0.5.0. Its diff SHA-256
was `12068540b652b3a42ebc93a1957c06c6bfa7ec20471c635876e903ac80f83f2f`.
It was not committed because the default sibling Candle checkout remains 0.10.2
until the integration PR merges.

Checkpoint/reference-backed results:

- CPU single-click tracker fixture: pass.
- CPU video propagation against the official CUDA-exported bundle: pass.
- CUDA video propagation against that bundle, release mode and serial: pass.
- bundle validator: pass, with only the expected absolute-path provenance
  warnings.

The CUDA bundle provenance and primary hashes are recorded in
`docs/CANDLE2_CUDA_REFERENCE_2026-07-18.md`. Its 5 GiB tensor payload remains a
private, ignored local artifact and is not redistributed by this repository.

## CUDA gate and runner disposition

The target was a Quadro RTX 5000 with Max-Q Design, compute capability 7.5,
16,384 MiB, driver 581.60, CUDA runtime 12.4.1, and F32 model execution.

The guarded same-fork workflow was dispatched at:

https://github.com/den-sq/candle_sam3/actions/runs/29699030845

It failed before any job step. Job `88224800404` has no runner, no labels, and
an empty step list. The fork has no registered runners and cannot acquire the
upstream-only `aws-g5-4xlarge-cache` runner group. This is an infrastructure
admission failure, not a test failure.

Manual replacement evidence on the target GPU passes:

- CUDA compilation with `CUDA_COMPUTE_CAP=75`;
- CPU and CUDA plugin image builds;
- official-reference CUDA tracker/video replay;
- the 16-frame Medical-SAM3 smoke, including geometry and pixel validation.

The CUDA plugin image is
`sha256:02a245b67383bf32ddd127af4cbda79a6cd02a678c9219799254dc073c037f81`.

## Synchronized 16-frame benchmark

The fixture is a 16-page, 200 x 200 grayscale TIFF with one frame-0 annotation.
Its SHA-256 is
`c59dab9ca0d90a3f5f5c0d56cff045eaa38feaaa6aa5626dc4b027b8ec16aa9a`.
The private Medical checkpoint SHA-256 is
`6e40bbaa739ac44e3e47dc6355ef6dedc560a30411377ad891f8af9e6df0dbd6`.

All four runs used the same harness, checkpoint, fixture, GPU, telemetry scope,
and output validator. Elapsed time starts with telemetry after container start
and includes lazy model loading plus inference.

| Revision | Run | Time | Throughput | Peak VRAM | Avg GPU | Peak RSS | Output SHA-256 |
|---|---|---:|---:|---:|---:|---:|---|
| old control | cold | 319 s | 0.0502 fps | 9,521 MiB | 90.4% | 335.0 MiB | `bf6d366d33adc09053ba6da7065c88916d6780a133bb2779204f458af751c67e` |
| old control | warm | 312 s | 0.0513 fps | 9,521 MiB | 92.1% | 331.1 MiB | `bf6d366d33adc09053ba6da7065c88916d6780a133bb2779204f458af751c67e` |
| Candle 0.11 | cold | 320 s | 0.0500 fps | 9,169 MiB | 91.1% | 325.8 MiB | `3a6bacb71268826c03b97dc25210155f0f2f6135c171ffebaa77203daeb73913` |
| Candle 0.11 | warm | 321 s | 0.0498 fps | 9,553 MiB | 90.5% | 331.1 MiB | `3a6bacb71268826c03b97dc25210155f0f2f6135c171ffebaa77203daeb73913` |

Candidate versus control elapsed-time deltas are +0.31% cold and +2.88% warm,
within the 5% gate. Peak memory and RSS show no material regression. Each pin is
deterministic across its cold/warm pair.

Raw evidence remains local under `/tmp/candle2_cert` pending private artifact
storage. File hashes are:

| Run | `revisions.env` | `telemetry.csv` | `backend.log` |
|---|---|---|---|
| old cold | `832bade6fb8b542e397ba1097f1a1bcf7208c5b75f0099b02c7dcfeafad1c5a5` | `f7794bbb89abc1f4ba5f58693012a1f2c53b56215dbfd748aa992e1d8821ae50` | `35ff3d1f644d30cc030fd8927ca97586bb6173626e46380d4bd79ea33c99a443` |
| old warm | `832bade6fb8b542e397ba1097f1a1bcf7208c5b75f0099b02c7dcfeafad1c5a5` | `fa35ba3dc22a8c477f8a0c2c349457e6798fa8a4bb87d98c57c434d7288108ee` | `396c73df9ac2bb081977acf56f12851ba8b2488ea115962708b56f730ec8040d` |
| candidate cold | `f16163bd2d0d3c0fc03e8d4fc25641f77eb40013d0dc476cf0f29834417d3e5b` | `8268f6beca8a5a3cf99b7d93abbf28c9841b07653e48d907cfd5565e790c9ff5` | `5d3e84a1234831a01ebb9a28f9c6cb6f3aa0655fa8d98493cb189a4f54032262` |
| candidate warm | `f16163bd2d0d3c0fc03e8d4fc25641f77eb40013d0dc476cf0f29834417d3e5b` | `ed591a25db9387d8ac400e15a1383b924f8cc8dcfec27d8131f143025665d779` | `02e9c63a9656c416328b746ec8487fdd085d0f34c5e58fca63d0a52c10e22513` |

The old and candidate masks differ by 36,370 pixels. The candidate is a strict
superset of the control, with IoU `0.927453409`.

## Medical-SAM3 upstream comparison: blocking

To distinguish an upgrade regression from inherited behavior, the official
Facebook tracker was run on CUDA with the same checkpoint, 16-frame geometry,
frame-0 positive point, and forward propagation. The 16-bit TIFF was staged
exactly like the plugin: integer division by 255 with saturation to U8,
grayscale-to-RGB expansion, then Rust `image` 0.25.10 JPEG encoding at quality
90. The direct tracker engine was used because that is the path corresponding
to Candle's tracker point prompt.

| Result | Foreground pixels | IoU versus upstream |
|---|---:|---:|
| official Facebook tracker | 458,829 | 1.0 |
| pre-upgrade Candle | 464,963 | 0.961942 |
| Candle 0.11 | 501,333 | 0.911318 |

The pre-upgrade result is close but not exact; Candle 0.11 materially increases
the divergence and adds mask area. The corrected official
reference metadata SHA-256 is
`00220007d3e1daec484842212e07a34004c4cba1bba0a54cf4ae190c3d45b6d3`,
the result metadata SHA-256 is
`3c5ca663fad07daca6a15cf857710a1c6d08ce49e5bb921c1715928a02f8ec35`,
and the aggregate ordered PNG-mask hash is
`0ed3b58361b6fe1ab9c169b111745dcc92a4346fb141b855b41ba86eab3fea2f`.

Disposition: the Candle 0.11 candidate introduces a new unexplained Medical
mask regression relative to both the old pin and official output. Issue #40
should remain open until the divergence is corrected or an explicit
project-owner exception is accepted.

## Temporary consumer compatibility

The auditable branch is:

https://github.com/ChengLabResearch/ouroboros_autoseg_plugin/tree/codex/candle-2-02-consumer-compat

Commit `742a7910f57b83ff3f66ba06c7309231c9e4d822` changes all three plugin Candle
constraints from 0.9.2 to 0.11.0 and regenerates `backend/Cargo.lock`. The
production pin is unchanged.

Results:

- `cargo test --manifest-path backend/Cargo.toml`: 57 unit tests plus one
  annotation contract pass; the manual checkpoint test remains ignored.
- Python network contracts: 19 pass.
- CPU image build: pass,
  `sha256:7581364e967fcefe42114c2be70a43002ca4a4626c59d1fbba6b6b6ebf1f6d69`.
- CUDA image build at compute capability 7.5: pass, image hash recorded above.
- checkpoint-backed CUDA Medical smoke: pass for execution, shape, dtype, and
  binary-output validation; upstream numerical parity remains blocked as
  described above.
