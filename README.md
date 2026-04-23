# candle-sam3-parity

SAM3 parity tooling extracted from the Candle runtime repo.

This sibling workspace owns upstream export, bundle contracts, comparison
commands, parity documentation, and committed seed fixtures. The runtime SAM3
implementation stays in `/home/dnorthover/ChengCode/candle_sam3`.

## Layout

- `rust/sam3-parity-lib`: shared bundle schemas and comparison helpers
- `rust/sam3-parity-cli`: migrated image, interactive, and video parity CLI
- `python/sam3_parity`: reference exporters and matrix generation
- `python/python_debug`: Python-side fixture exporters and debug utilities
- `tests/data`: committed seed fixtures
- `docs`: migration inventory, parity runbooks, and matrix docs

## Build

Rust workspace:

```bash
cargo check --manifest-path /home/dnorthover/ChengCode/sam_parity/Cargo.toml
```

Python package:

```bash
python -m pip install -e /home/dnorthover/ChengCode/sam_parity/python
```

## Dependency Model

The Rust crates depend on the sibling Candle checkout through path
dependencies rooted at `../candle_sam3`. That keeps parity runs pinned to an
exact local Candle revision.

## Primary Contracts

Frozen migration contracts live in [docs/MIGRATION_INVENTORY.md](/home/dnorthover/ChengCode/sam_parity/docs/MIGRATION_INVENTORY.md)
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

Use `candle_sam3` for:

- normal image inference
- interactive refinement runs
- normal video prediction
- non-parity smoke and unit testing
