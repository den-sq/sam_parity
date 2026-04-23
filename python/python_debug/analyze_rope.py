#!/usr/bin/env python3
"""Analyze RoPE pairwise rotation behavior."""

import torch


def upstream_rotate_pairwise(xs):
    """Upstream PyTorch RoPE using complex numbers."""
    xq_ = torch.view_as_complex(xs.float().reshape(*xs.shape[:-1], -1, 2))
    # This is for the rope rotation
    return torch.stack([-xq_.imag, xq_.real], dim=-1)


def candle_rotate_pairwise_equivalent(xs):
    """Candle RoPE using direct tensor operations."""
    # (B, H, L, D) -> (B, H, L, D//2, 2)
    batch_size, num_heads, seq_len, head_dim = xs.shape
    xs_reshaped = xs.reshape((batch_size, num_heads, seq_len, head_dim // 2, 2))

    # Split into even and odd pairs: [x0,x1,x2,x3,...] -> [[x0,x2,...], [x1,x3,...]]
    xs_split = torch.split(xs_reshaped, 1, dim=3)  # Split on axis 3
    even = xs_split[0]  # [x0, x2, x4, ...]
    odd = xs_split[1] if len(xs_split) > 1 else torch.zeros_like(even)  # [x1, x3, x5, ...]

    # Rotate: [-x1, x0, -x3, x2, ...]
    rotated = torch.cat([odd.neg(), even], dim=4)
    return rotated.reshape((batch_size, num_heads, seq_len, head_dim))


def main():
    """Run a small RoPE rotation trace."""
    # Test with small example
    batch_size, num_heads, seq_len, head_dim = 1, 2, 4, 8
    x = torch.randn(batch_size, num_heads, seq_len, head_dim)

    print("Input shape:", x.shape)
    print("\nCandle rotate_pairwise logic:")
    print("- Reshape to (B, H, L, D//2, 2)")
    print("- Split pairs on dimension with 2 elements")
    print("- Rotate: [-odd, even]")
    print("- Reshape back to (B, H, L, D)")

    # Let's trace through with actual data
    xs = x.reshape(batch_size, num_heads, seq_len, head_dim // 2, 2)
    print("\nAfter reshape to (B, H, L, D//2, 2):", xs.shape)

    pair = torch.chunk(xs, 2, dim=3)  # chunk by 2 on dimension 3
    print("After chunk by 2 on dim 3:", len(pair), "tensors of shape", pair[0].shape)

    even = pair[0]
    odd = pair[1]
    print("Even shape:", even.shape)
    print("Odd shape:", odd.shape)

    rotated = torch.cat([odd.neg(), even], dim=4)
    print("After cat on dim 4:", rotated.shape)

    result = rotated.reshape(batch_size, num_heads, seq_len, head_dim)
    print("Final shape:", result.shape)

    # Show first element of input and output
    print("\nFirst head, first position, first 8 elements:")
    print("Input x[0, 0, 0]:", x[0, 0, 0].numpy())
    print("After rotate_pairwise[0, 0, 0]:", result[0, 0, 0].numpy())

    # Expected RoPE rotation: [x0, x1] -> [-x1, x0]
    print("\nExpected pattern for RoPE rotation:")
    print("[x0, x1] -> [-x1, x0]")
    print("[x2, x3] -> [-x3, x2]")
    print("etc.")

    # Verify our result matches
    print("\nVerification:")
    for i in range(0, head_dim, 2):
        print(f"Input [{i}, {i + 1}]: [{x[0, 0, 0, i]:.3f}, {x[0, 0, 0, i + 1]:.3f}]", end="")
        print(
            f" -> Output [{i}, {i + 1}]: "
            f"[{result[0, 0, 0, i]:.3f}, {result[0, 0, 0, i + 1]:.3f}]",
            end="",
        )
        expected_i = -x[0, 0, 0, i + 1]
        expected_i1 = x[0, 0, 0, i]
        match = abs(result[0, 0, 0, i] - expected_i) < 0.001 and abs(
            result[0, 0, 0, i + 1] - expected_i1
        ) < 0.001
        print(f" Match: {match}")


if __name__ == "__main__":
    main()
