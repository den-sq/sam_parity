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
  - `python_debug/sam3/exports/*`

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
- Existing upstream checkout assumptions in docs/scripts still refer to
  `/home/dnorthover/extcode/sam3_baseline` and `/home/dnorthover/extcode/hf_sam3`.
- Existing Python-side fixture utilities historically referenced the old repo via
  `python_debug.sam3.common`.

## Fixture Inventory

- Seed fixtures copied from runtime repo:
  - `tests/data/sam3_decoder_unit`
  - `tests/data/sam3_fusion_unit`
  - `tests/data/sam3_geometry_unit`
  - `tests/data/sam3_interactive_geometry_seed`
  - `tests/data/sam3_interactive_visual_seed`
  - `tests/data/sam3_segmentation_unit`
