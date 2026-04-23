#!/usr/bin/env python3
"""Debug script to understand `grid_sample` behavior in PyTorch."""

import numpy as np
import torch
import torch.nn.functional as F


def test_grid_sample_bilinear():
    """Test `grid_sample` with bilinear interpolation."""
    print("=" * 60)
    print("Grid Sample Bilinear Test")
    print("=" * 60)

    # Create a simple 4x4 feature map with known values
    features = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    print("\nFeature map (4x4):")
    print(features[0, 0])

    # Test point sampling
    print("\n" + "=" * 60)
    print("POINT SAMPLING TEST")
    print("=" * 60)

    # Test point at (0.5, 0.5) normalized = center of image
    grid = torch.tensor([[[0.5, 0.5]]], dtype=torch.float32)  # [1, 1, 1, 2]

    # Convert to [-1, 1] as done in Python baseline
    grid_normalized = (grid * 2) - 1
    print(f"\nNormalized point (0.5, 0.5) as grid: {grid_normalized[0, 0, 0].tolist()}")

    result = F.grid_sample(features, grid_normalized, mode='bilinear', align_corners=False)
    print(f"Grid sample result: {result[0, 0, 0, 0].item():.6f}")

    # Decode to pixel coordinates
    grid_val = grid_normalized[0, 0, 0]
    x_pixel_from_grid = (grid_val[0] + 1) / 2 * 4 - 0.5
    y_pixel_from_grid = (grid_val[1] + 1) / 2 * 4 - 0.5
    print(f"Pixel coordinates from grid: x={x_pixel_from_grid.item():.2f}, y={y_pixel_from_grid.item():.2f}")

    # Manual bilinear at (1.5, 1.5)
    x, y = 1.5, 1.5
    x0, x1 = int(np.floor(x)), int(np.ceil(x))
    y0, y1 = int(np.floor(y)), int(np.ceil(y))
    fx, fy = x - x0, y - y0
    fx_inv, fy_inv = 1 - fx, 1 - fy

    v00 = features[0, 0, y0, x0].item()
    v01 = features[0, 0, y0, x1].item()
    v10 = features[0, 0, y1, x0].item()
    v11 = features[0, 0, y1, x1].item()

    manual_result = v00 * fx_inv * fy_inv + v01 * fx * fy_inv + v10 * fx_inv * fy + v11 * fx * fy
    print("\nManual bilinear calculation at (1.5, 1.5):")
    print(f"  v00={v00}, v01={v01}, v10={v10}, v11={v11}")
    print(f"  fx={fx}, fy={fy}")
    print(f"  Result: {manual_result:.6f}")


if __name__ == "__main__":
    test_grid_sample_bilinear()
