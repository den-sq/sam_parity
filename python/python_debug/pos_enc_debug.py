#!/usr/bin/env python3
"""Inspect the Python position encoding for a single point."""

import math
import sys
from pathlib import Path

try:
    from sam3_parity.paths import sam3_repo_root
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sam3_parity.paths import sam3_repo_root


def main():
    """Run a single-point position encoding inspection."""
    sam3_path = sam3_repo_root()
    if sam3_path is None:
        raise RuntimeError("SAM3_REPO is required for this debug utility")
    package_parent = str(Path(sam3_path).expanduser().resolve())
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)

    import torch
    from sam3.model.position_encoding import PositionEmbeddingSine

    # Create position encoder for geometry (not spatial)
    pe = PositionEmbeddingSine(num_pos_feats=256, normalize=True, scale=None, temperature=10000)

    # Test encoding a single point (using the encode_points method)
    x = torch.tensor([[0.418]], dtype=torch.float32)
    y = torch.tensor([[0.653]], dtype=torch.float32)
    labels = torch.tensor([[1]], dtype=torch.int32)

    pos = pe.encode_points(x, y, labels)
    print(f"Python position encoding shape: {pos.shape}")
    print(f"First 10 values of pos[0,0,:10]: {pos[0, 0, :10]}")
    print(f"Values from index 256-256+10: {pos[0, 0, 256:266]}")
    print(f"Values from index 512-512+2 (labels): {pos[0, 0, 512:514]}")

    # Let me also manually compute what it should be
    # According to _encode_xy():
    # x_embed = x * self.scale where self.scale = 2π
    # y_embed = y * self.scale
    # dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    #
    # self.num_pos_feats = 256 // 2 = 128

    num_pos_feats = 256 // 2  # = 128
    temperature = 10000
    scale = 2 * math.pi

    x_coord = 0.418
    y_coord = 0.653

    x_embed = x_coord * scale  # 0.418 * 2π
    y_embed = y_coord * scale  # 0.653 * 2π

    print("\nManual computation:")
    print(f"scale = 2π = {scale}")
    print(f"x_embed = {x_embed}")
    print(f"y_embed = {y_embed}")

    # Compute dim_t
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    print(f"First 5 dim_t values: {dim_t[:5]}")
    print(f"Last 5 dim_t values: {dim_t[-5:]}")

    # Now compute position encodings manually
    pos_x = (x_embed / dim_t).unsqueeze(0)
    pos_y = (y_embed / dim_t).unsqueeze(0)

    print(f"\npos_y before sin/cos stacking shape: {pos_y.shape}")
    print(f"First 5 values: {pos_y[0, :5]}")

    # Apply sin/cos
    pos_y_encoded = torch.stack(
        (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2
    ).flatten(1)
    pos_x_encoded = torch.stack(
        (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2
    ).flatten(1)

    print(f"\npos_y_encoded shape: {pos_y_encoded.shape}")
    print(f"First 10 values: {pos_y_encoded[0, :10]}")
    print(f"pos_x_encoded first 10 values: {pos_x_encoded[0, :10]}")

    # Combined should be [pos_y, pos_x, labels]
    print("\nExpected combined: [pos_y (128), pos_x (128), labels (1)] = 257 dims")
    print(f"Python pos shape: {pos.shape}")


if __name__ == "__main__":
    main()
