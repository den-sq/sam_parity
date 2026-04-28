# candle-sam3-parity

SAM3 parity tooling extracted from the Candle runtime repo.

This sibling workspace owns upstream export, bundle contracts, comparison
commands, parity documentation, and committed seed fixtures. The runtime SAM3
implementation stays in the sibling `../candle_sam3` checkout.

## Layout

- `rust/sam3-parity-lib`: shared bundle schemas and comparison helpers
- `rust/sam3-parity-cli`: migrated image, interactive, and video parity CLI
- `python/sam3_parity`: reference exporters and matrix generation
- `python/python_debug`: Python-side fixture exporters and debug utilities
- `tests/data`: committed seed fixtures
- `tests/reference-bundles`: generated upstream export bundles for Rust/Python checks
- `docs`: migration inventory, parity runbooks, and matrix docs

## Build

Rust workspace:

```bash
cargo check --manifest-path Cargo.toml
```

Python package:

```bash
python -m pip install -e python
```

## Example Runs

Lightweight checks that do not require generated video bundles:

```bash
cargo test --workspace
python -m pytest python/sam3_parity/tests -q
```

Compile the feature-gated Rust full-parity harness:

```bash
cargo test -p sam3-parity-cli --features full-parity --no-run
```

Run a couple of fast full-parity Rust tests that only exercise the Candle
parity-support surface and fixture contracts:

```bash
cargo test -p sam3-parity-cli --features full-parity \
  tracker_build_config_matches_upstream_contract -- --nocapture

cargo test -p sam3-parity-cli --features full-parity \
  tracker_transformer_contract_matches_upstream_builder -- --nocapture
```

Run a bundle-backed Rust parity test against generated artifacts in
`tests/reference-bundles`:

```bash
SAM3_TEST_CHECKPOINT_DIR=/path/to/hf_sam3 \
cargo test -p sam3-parity-cli --features full-parity \
  video_process_frame_matches_visual_box_reference_bundle_frame0 -- --nocapture
```

Run the Python full-parity suite against an installed upstream `sam3` package:

```bash
SAM3_CHECKPOINT=/path/to/hf_sam3/sam3.pt \
python -m pytest -m full_parity python/python_debug/sam3_debug/tests -q
```

Generate or validate bundle artifacts under `tests/reference-bundles`:

```bash
sam3-generate-video-matrix

sam3-validate-bundles reference_video_box_debug \
  --bundle-root tests/reference-bundles
```

Notes:

- Rust full-parity tests expect checkpoint access through
  `SAM3_TEST_CHECKPOINT` or `SAM3_TEST_CHECKPOINT_DIR`.
- Python full-parity tests expect an installed upstream `sam3` package plus
  `SAM3_CHECKPOINT`.
- Bundle-backed tests read from `tests/reference-bundles` by default and skip or
  fail meaningfully when the required generated artifacts are absent.

## Dependency Model

The Rust crates depend on the sibling Candle checkout through path
dependencies rooted at `../candle_sam3`. That keeps parity runs pinned to an
exact local Candle revision.

## Primary Contracts

Frozen migration contracts live in [docs/MIGRATION_INVENTORY.md](docs/MIGRATION_INVENTORY.md)
and currently cover:

- `reference.safetensors`
- `reference.json`
- interactive replay bundle layout
- video debug manifest layout
- `video_tracker_strict_port_matrix.json`

## Operator Notes

Use this repo for:

- exporting upstream references
- comparing Candle image outputs to reference bundles
- replaying interactive parity bundles
- comparing video bundles and debug manifests
- regenerating strict-port coverage matrices

Portable path defaults:

- `SAM3_PARITY_BUNDLE_ROOT`: generated bundle root, default `tests/reference-bundles`
- `SAM3_PARITY_DATA_ROOT`: reusable fixture root, default `tests/data`
- `SAM3_CHECKPOINT`: local `sam3.pt` or checkpoint directory
- `SAM3_TOKENIZER`: local `tokenizer.json`
- `SAM3_UPSTREAM_URL`: optional upstream or fork URL recorded in generated metadata
- `SAM3_UPSTREAM_REF`: optional upstream commit, tag, or branch recorded in generated metadata

Use `candle_sam3` for:

- normal image inference
- interactive refinement runs
- normal video prediction
- non-parity smoke and unit testing
