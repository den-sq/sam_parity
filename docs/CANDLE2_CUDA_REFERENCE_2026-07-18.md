# CANDLE-2 Facebook SAM3 CUDA reference

Date: 2026-07-18

The local reference bundle is:

`tests/reference-bundles/reference_video_suppressed_obj_ids_text_bed_debug_cuda`

It was exported on CUDA device 0 from the official Facebook SAM3 Python
implementation at commit `84cc43bca4347b772f17d1078a1ddb4c054655c2`, using
PyTorch `2.7.0+cu126` on the cc7.5 Quadro RTX 5000. The bundle contains 30
source frames, a five-frame temporal-disambiguation propagation, visible masks,
and 5.0 GiB of internal BF16/F32 CUDA tracker tensors. Exact environment and
input/output hashes are recorded in the bundle's `cuda_provenance.json`.

## Validation

The bundle validator passes:

```bash
PYTHONPATH=python /home/dnorthover/miniconda3/envs/sam3/bin/python \
  -m sam3_parity.validate_bundles \
  reference_video_suppressed_obj_ids_text_bed_debug_cuda \
  --bundle-root tests/reference-bundles
```

The Candle 0.11 integration at
`71fad7110bcd1d861102ef256a76eeeac1300bce` passes the serial CUDA replay on
compute capability 7.5:

```bash
CUDA_HOME=/usr/local/cuda-12.9 CUDA_COMPUTE_CAP=75 \
SAM3_PARITY_BUNDLE_ROOT=/tmp/candle2-facebook-sam3-cuda-test-root \
cargo test --release -p sam3-parity-cli --features full-parity,cuda \
  video_propagation_matches_text_prompt_suppressed_reference_bundle_cuda \
  -- --ignored --nocapture --test-threads=1
```

The temporary bundle root maps the CUDA bundle to the canonical bundle name
expected by the existing replay test. The replay consumer now treats reference
frames with no visible object IDs as empty outputs rather than requiring
nonexistent internal mask tensors. This fixes the same latent problem in the
historical bundle.

## Primary hashes

- checkpoint: `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`
- `reference.json`: `c35a657dad7e533c77e42ff6f9e310774f5702e282df4b8f0da8f5aa281781f1`
- `video_results.json`: `9e4252ec466576bbc1b27edc8e3fd43b5fd129279e813307b66c8327214987d8`
- `debug/internal_manifest.json`: `efc22f10f55ad72e941671ac560c9907664879a5e6737d1ee55366b131d16c9c`
- `debug/internal_fixtures.safetensors`: `6e80d91475ab51eae268545a0aa0c0fa1b1f5badb1c8d2b573cc248f74d849ba`

The generated reference bundle is intentionally ignored by Git because of its
size. Publish it through the project's artifact storage before relying on it in
remote CI.
