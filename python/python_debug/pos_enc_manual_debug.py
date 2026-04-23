#!/usr/bin/env python3
"""Manually inspect the 1D position encoding values for a box."""

import math


def encode_1d_position_python(coord, num_pos_feats=128, temperature=10000):
    """Python-style position encoding."""
    scale = 2 * math.pi
    coord_scaled = coord * scale

    result = []
    for idx in range(num_pos_feats):
        exponent = (2 * (idx // 2)) / num_pos_feats
        dim_t = temperature ** exponent
        angle = coord_scaled / dim_t
        if idx % 2 == 0:
            result.append(math.sin(angle))
        else:
            result.append(math.cos(angle))

    return result


def main():
    """Run a manual position encoding inspection."""
    # Test with box coordinates
    cx = 0.418
    cy = 0.653
    w = 0.086
    h = 0.5

    print("Python Position Encoding Results:")
    print("=" * 50)

    pos_x = encode_1d_position_python(cx)
    pos_y = encode_1d_position_python(cy)

    print(f"Box coordinates: cx={cx}, cy={cy}, w={w}, h={h}")
    print(f"\npos_y (first 10): {[f'{v:.5f}' for v in pos_y[:10]]}")
    print(f"pos_y (last 5): {[f'{v:.5f}' for v in pos_y[-5:]]}")
    print(f"\npos_x (first 10): {[f'{v:.5f}' for v in pos_x[:10]]}")
    print(f"pos_x (last 5): {[f'{v:.5f}' for v in pos_x[-5:]]}")

    # Full encoding: [pos_y, pos_x, h, w]
    full_pos = pos_y + pos_x + [h, w]
    print(f"\nFull position encoding shape: {len(full_pos)}")
    print(f"Full encoding (first 10): {[f'{v:.5f}' for v in full_pos[:10]]}")
    print(f"Full encoding (dim 26): {full_pos[26]:.5f}")
    print(f"Full encoding (dims 128-138): {[f'{v:.5f}' for v in full_pos[128:138]]}")


if __name__ == "__main__":
    main()
