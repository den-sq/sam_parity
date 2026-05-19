This directory contains the first 30 frames from the upstream SAM3 notebook
video asset `assets/videos/0001`.

Source used for this local copy:

- `/home/dnorthover/extcode/sam3_baseline/assets/videos/0001`
- verified to match the notebook example cache under
  `/tmp/sam3-notebook-gpu/_notebook_assets/videos/0001`

Why this exists:

- issue `#20`, `#21`, and `#22` example runs should use the real SAM3 notebook
  clip rather than the older bedroom-based debug video bundles
- keeping a local first-30-frame subset makes it easy to run short reproducible
  examples without depending on a cached notebook download directory

Frame naming matches the upstream asset (`0.jpg` through `29.jpg`).
