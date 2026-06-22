## SAM3 Notebook Video Example Commands

Run these from the `sam_parity` repo root.

All three commands use the real SAM3 notebook video sample copied under
`tests/data/sam3_video_notebook_0001_first30/frames` rather than the older
bedroom-based debug fixtures that were previously used for ad-hoc issue
examples.

### Text Prompt Example

This mirrors the notebook's initial `person` text prompt on frame 0.

```bash
cargo run -p sam3-parity-cli --features cuda --bin sam3-parity-cli -- \
  --checkpoint /home/dnorthover/extcode/hf_sam3/sam3.pt \
  --tokenizer /home/dnorthover/extcode/hf_sam3/tokenizer.json \
  --video tests/data/sam3_video_notebook_0001_first30/frames \
  --video-prompt person \
  --video-frame-stride 1 \
  --output-dir /tmp/sam3-notebook-text-gpu
```

### Single-Click Example

This mirrors the notebook's single-click refinement on frame 0:

- point `(760 / 1280, 550 / 720)` -> `(0.593750, 0.763889)`
- label `1`

```bash
cargo run -p sam3-parity-cli --features cuda --bin sam3-parity-cli -- \
  --checkpoint /home/dnorthover/extcode/hf_sam3/sam3.pt \
  --tokenizer /home/dnorthover/extcode/hf_sam3/tokenizer.json \
  --video tests/data/sam3_video_notebook_0001_first30/frames \
  --point 0.593750,0.763889 \
  --point-label 1 \
  --video-frame-stride 1 \
  --output-dir /tmp/sam3-notebook-single-click-gpu
```

### Refined Multi-Click Example

This mirrors the notebook's refined-click prompt set on frame 0:

- `(740 / 1280, 450 / 720)` -> `(0.578125, 0.625000)`, label `1`
- `(760 / 1280, 630 / 720)` -> `(0.593750, 0.875000)`, label `0`
- `(840 / 1280, 640 / 720)` -> `(0.656250, 0.888889)`, label `0`
- `(760 / 1280, 550 / 720)` -> `(0.593750, 0.763889)`, label `1`

```bash
cargo run -p sam3-parity-cli --features cuda --bin sam3-parity-cli -- \
  --checkpoint /home/dnorthover/extcode/hf_sam3/sam3.pt \
  --tokenizer /home/dnorthover/extcode/hf_sam3/tokenizer.json \
  --video tests/data/sam3_video_notebook_0001_first30/frames \
  --point 0.578125,0.625000 --point-label 1 \
  --point 0.593750,0.875000 --point-label 0 \
  --point 0.656250,0.888889 --point-label 0 \
  --point 0.593750,0.763889 --point-label 1 \
  --video-frame-stride 1 \
  --output-dir /tmp/sam3-notebook-refined-clicks-gpu
```
