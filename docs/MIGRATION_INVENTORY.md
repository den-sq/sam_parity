# SAM3 Parity Migration Inventory

## Command/Entry Point Contracts

- Rust image parity used `candle-examples --example sam3 -- --parity-bundle <bundle>`.
- Rust interactive comparison used `--compare-reference-bundle <interactive bundle>`.
- Rust video comparison used `--compare-reference-bundle <video bundle>`.
- Python exporters used:
  - `export_reference.py`
  - `generate_video_tracker_strict_port_matrix.py`
  - `python_debug/export_geometry_unit_fixture.py`
  - `python_debug/export_decoder_unit_fixture.py`
  - `python_debug/export_segmentation_unit_fixture.py`
  - `python_debug/sam3_debug/exports/*`

## File/Schema Contracts

- Image parity bundle:
  - `reference.safetensors`
  - `reference.json`
- Interactive replay bundle:
  - `reference.safetensors`
  - `reference.json`
  - per-step `step_XXX_<name>/...` render artifacts
- Video reference bundle:
  - `reference.json`
  - `video_results.json`
  - `frames/`
  - `masks/`
  - `masked_frames/`
- Video debug bundle:
  - `debug/debug_manifest.json`
  - `debug/debug_compare.json`
  - mask artifacts keyed by frame and object id
- Matrix manifest:
  - `video_tracker_strict_port_matrix.json`

## Environment/Path Contracts

- Path-based Candle dependency root is currently `../candle_sam3`.
- Current Python parity/export flow expects an installed upstream `sam3` package,
  plus `$SAM3_CHECKPOINT` for model weights.
- Optional provenance/pinning for parity runs is recorded via
  `$SAM3_UPSTREAM_URL` and `$SAM3_UPSTREAM_REF`.
- Existing Python-side fixture utilities now live under
  `python_debug.sam3_debug.common` to avoid shadowing the real upstream `sam3`
  package.

## Fixture Inventory

- Seed fixtures copied from runtime repo:
  - `tests/data/sam3_decoder_unit`
  - `tests/data/sam3_fusion_unit`
  - `tests/data/sam3_geometry_unit`
  - `tests/data/sam3_interactive_geometry_seed`
  - `tests/data/sam3_interactive_visual_seed`
  - `tests/data/sam3_segmentation_unit`
