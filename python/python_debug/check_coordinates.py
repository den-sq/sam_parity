#!/usr/bin/env python3
"""Verify coordinate system for `grid_sample`."""

# For grid_sample with align_corners=False:
# The normalized grid coordinates [-1, 1] map to pixel coordinates as:
# pixel = (grid + 1) / 2 * size - 0.5
#
# The Python code converts [0, 1] to [-1, 1] by: grid = coord * 2 - 1
# So: pixel = ((coord * 2 - 1) + 1) / 2 * size - 0.5
#           = (coord * 2) / 2 * size - 0.5
#           = coord * size - 0.5
#
# This means for normalized coordinates in [0, 1], the pixel coordinate is:
# pixel = norm_coord * size - 0.5


def verify_coordinate_mapping():
    """Verify the coordinate mapping."""
    size = 16

    # Test different normalized coordinates
    test_coords = [0.0, 0.25, 0.5, 0.75, 1.0]

    print("Coordinate mapping for size={}, align_corners=False".format(size))
    print("norm_coord -> pixel_coord (expected)")
    for coord in test_coords:
        pixel = coord * size - 0.5
        print(f"  {coord:0.2f} -> {pixel:0.2f}")

    print("\nFor reference:")
    print("- Grid value -1 corresponds to pixel position -0.5 (before pixel 0)")
    print("- Grid value  0 corresponds to pixel position size/2 - 0.5 (at center)")
    print("- Grid value  1 corresponds to pixel position size - 0.5 (after last pixel)")


if __name__ == "__main__":
    verify_coordinate_mapping()
