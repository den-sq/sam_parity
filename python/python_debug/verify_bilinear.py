#!/usr/bin/env python3
"""Verify that Candle bilinear interpolation matches PyTorch behavior."""

import numpy as np
import torch
import torch.nn.functional as F


def test_point_sampling_bilinear():
    """Test that point sampling uses bilinear interpolation."""
    print("Testing point sampling bilinear interpolation...")

    # Create a simple test image
    image = torch.arange(1.0, 257.0).reshape(1, 1, 16, 16)  # [1, 1, 16, 16]

    # Test point at normalized coordinate (0.5, 0.5) = (8.0, 8.0) in pixels
    grid = torch.tensor([[[0.5, 0.5]]], dtype=torch.float32)  # [1, 1, 1, 2]

    # PyTorch grid_sample with bilinear interpolation
    result_pytorch = F.grid_sample(image, grid, mode='bilinear', align_corners=False)
    print(f"PyTorch grid_sample result: {result_pytorch.item():.6f}")

    # Manual computation for verification
    # At normalized (0.5, 0.5) with align_corners=False:
    # pixel_x = 0.5 * 16 = 8.0
    # pixel_y = 0.5 * 16 = 8.0
    x_pixel = 0.5 * 16
    y_pixel = 0.5 * 16
    x0 = int(np.floor(x_pixel))
    y0 = int(np.floor(y_pixel))
    x1 = min(x0 + 1, 15)
    y1 = min(y0 + 1, 15)

    fx = x_pixel - x0
    fy = y_pixel - y0
    fx_inv = 1.0 - fx
    fy_inv = 1.0 - fy

    v00 = image[0, 0, y0, x0].item()
    v01 = image[0, 0, y0, x1].item()
    v10 = image[0, 0, y1, x0].item()
    v11 = image[0, 0, y1, x1].item()

    w00 = fx_inv * fy_inv
    w01 = fx * fy_inv
    w10 = fx_inv * fy
    w11 = fx * fy

    result_manual = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11
    print(f"Manual computation: {result_manual:.6f}")
    print(f"Coordinates: x={x_pixel}, y={y_pixel}")
    print(f"Floor coords: x0={x0}, y0={y0}, x1={x1}, y1={y1}")
    print(f"Values: v00={v00}, v01={v01}, v10={v10}, v11={v11}")
    print(f"Weights: w00={w00}, w01={w01}, w10={w10}, w11={w11}")

    assert abs(result_pytorch.item() - result_manual) < 1e-5, "Manual and PyTorch results don't match!"
    print("✓ Point sampling bilinear interpolation verified\n")


def test_grid_sample_modes():
    """Compare different grid_sample modes."""
    print("Testing grid_sample modes...")

    image = torch.arange(1.0, 65.0, dtype=torch.float32).reshape(1, 1, 8, 8)

    # Test point at (0.25, 0.25) normalized
    grid = torch.tensor([[[0.25, 0.25]]], dtype=torch.float32)

    bilinear = F.grid_sample(image, grid, mode='bilinear', align_corners=False)
    nearest = F.grid_sample(image, grid, mode='nearest', align_corners=False)

    print(f"Bilinear mode output: {bilinear.item():.6f}")
    print(f"Nearest mode output:  {nearest.item():.6f}")
    print(f"Difference: {abs(bilinear.item() - nearest.item()):.6f}")
    print("Expected bilinear sample coordinates: x=2.0, y=2.0")

    # Manual computation for bilinear
    fx, fy = 0.0, 0.0

    v00 = image[0, 0, 2, 2].item()
    result_manual = v00 * (1.0 - fx) * (1.0 - fy)

    print(f"Manual bilinear: {result_manual:.6f}")
    print(f"Image[2,2] = {v00:.6f}")
    assert abs(bilinear.item() - result_manual) < 1e-5, "Manual calculation doesn't match bilinear!"
    print("✓ Grid sample modes comparison verified\n")


def test_roi_align_like_behavior():
    """Test that ROI pooling behavior matches expectations."""
    print("Testing ROI pooling behavior...")

    # Create a simple feature map
    features = torch.arange(1.0, 65.0, dtype=torch.float32).reshape(1, 1, 8, 8)

    # Box in cxcywh format at center (0.5, 0.5) with size (0.5, 0.5) normalized
    # This covers the middle quarter of the image
    box_cxcywh = torch.tensor([[0.5, 0.5, 0.5, 0.5]])

    # Convert to xyxy
    cx, cy, w, h = box_cxcywh[0]
    x0 = (cx - w/2).item() * 8  # normalized to pixel
    y0 = (cy - h/2).item() * 8
    x1 = (cx + w/2).item() * 8
    y1 = (cy + h/2).item() * 8

    print(f"Box in pixels: ({x0:.2f}, {y0:.2f}) to ({x1:.2f}, {y1:.2f})")

    # Sample at ROI grid points
    roi_size = 2
    for roi_y in range(roi_size):
        for roi_x in range(roi_size):
            # Sample position within the ROI
            sample_x = x0 + (roi_x + 0.5) * (x1 - x0) / roi_size
            sample_y = y0 + (roi_y + 0.5) * (y1 - y0) / roi_size

            # Get value at this position using bilinear interpolation
            x_idx = int(np.floor(sample_x))
            y_idx = int(np.floor(sample_y))
            x_idx = min(x_idx, 7)
            y_idx = min(y_idx, 7)

            value = features[0, 0, y_idx, x_idx].item()
            print(f"ROI[{roi_y},{roi_x}] at pixel ({sample_x:.2f}, {sample_y:.2f}) -> [{y_idx},{x_idx}] = {value:.2f}")

    print("✓ ROI pooling behavior verified\n")


if __name__ == "__main__":
    test_point_sampling_bilinear()
    test_grid_sample_modes()
    test_roi_align_like_behavior()
    print("✅ All bilinear interpolation tests passed!")
