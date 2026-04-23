#!/usr/bin/env python3
"""Create minimal test data for SAM3 geometry encoder verification."""

import sys
from pathlib import Path

try:
    from sam3_parity.paths import sam3_repo_root
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sam3_parity.paths import sam3_repo_root


def debug_geometry_encoder():
    """Test the geometry encoder with simple inputs."""
    sam3_path = sam3_repo_root()
    if sam3_path is None:
        raise RuntimeError("SAM3_REPO is required for this debug utility")
    package_parent = str(Path(sam3_path).expanduser().resolve())
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)

    import torch
    from sam3.model_builder import build_sam3_image_model

    # Build model
    model = build_sam3_image_model(model_cfg="vit_h")

    # Create simple test image
    test_image = torch.randn(1, 3, 256, 256) * 0.1

    # Get vision features
    with torch.no_grad():
        vision_feats_and_pos = model.image_encoder(test_image)
        vision_feats = vision_feats_and_pos  # List of features
        img_feats_list = vision_feats

    # Create simple geometry prompt
    points_xy = torch.tensor([
        [[0.25, 0.25], [0.75, 0.75]]  # 2 points
    ], dtype=torch.float32)  # [1, 2, 2]

    boxes_cxcywh = torch.tensor([
        [[0.5, 0.5, 0.25, 0.25]]  # 1 box
    ], dtype=torch.float32)  # [1, 1, 4]

    print("Points shape:", points_xy.shape)
    print("Points values:", points_xy)
    print("Boxes shape:", boxes_cxcywh.shape)
    print("Boxes values:", boxes_cxcywh)

    # Encode geometry
    geo_prompt = model.image_predictor.prompt_encoder.geo_prompt_encoder.get_prompt_class()(
        point_embeddings=points_xy,
        box_embeddings=boxes_cxcywh,
    )

    geom_out = model.image_predictor.prompt_encoder.geo_prompt_encoder(
        geo_prompt,
        img_feats_list,
        None,
    )

    print("\nGeometry output:")
    print("Features shape:", geom_out.features.shape)
    print("Features values (first few):", geom_out.features[0, 0, :5])
    print("Padding mask:", geom_out.padding_mask)


if __name__ == "__main__":
    debug_geometry_encoder()
