import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_SAM3_DIR = REPO_ROOT / "candle-examples" / "examples" / "sam3"
DEFAULT_SAM3_REPO = Path("/home/dnorthover/extcode/sam3_baseline")


def ensure_repo_root_on_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def ensure_example_sam3_on_path() -> None:
    path = str(EXAMPLE_SAM3_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


def sam3_repo_root() -> Path:
    return Path(os.environ.get("SAM3_REPO", str(DEFAULT_SAM3_REPO))).expanduser().resolve()


def add_upstream_sam3_to_path() -> Path:
    package_dir = sam3_repo_root() / "sam3"
    parent = str(package_dir.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    return package_dir


def apply_cpu_safe_upstream_patches(
    sam3_model_builder,
    transformer_decoder_cls=None,
    sequence_geometry_encoder_cls=None,
) -> None:
    from sam3.model.position_encoding import PositionEmbeddingSine

    def create_cpu_position_encoding(precompute_resolution=None):
        return PositionEmbeddingSine(
            num_pos_feats=256,
            normalize=True,
            scale=None,
            temperature=10000,
            precompute_resolution=None,
        )

    sam3_model_builder._create_position_encoding = create_cpu_position_encoding

    if transformer_decoder_cls is not None:
        def get_coords_cpu_safe(H, W, device):
            if device == "cuda":
                device = "cpu"
            coords_h = torch.arange(0, H, device=device, dtype=torch.float32) / H
            coords_w = torch.arange(0, W, device=device, dtype=torch.float32) / W
            return coords_h, coords_w

        transformer_decoder_cls._get_coords = staticmethod(get_coords_cpu_safe)

    if sequence_geometry_encoder_cls is not None:
        def encode_boxes_cpu_safe(self, boxes, boxes_mask, boxes_labels, img_feats):
            boxes_embed = None
            n_boxes, bs = boxes.shape[:2]
            if self.boxes_direct_project is not None:
                proj = self.boxes_direct_project(boxes)
                boxes_embed = proj
            if self.boxes_pool_project is not None:
                import torchvision
                from sam3.model.box_ops import box_cxcywh_to_xyxy

                H, W = img_feats.shape[-2:]
                boxes_xyxy = box_cxcywh_to_xyxy(boxes)
                scale = torch.tensor(
                    [W, H, W, H], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device
                ).view(1, 1, 4)
                boxes_xyxy = boxes_xyxy * scale
                sampled = torchvision.ops.roi_align(
                    img_feats, boxes_xyxy.float().transpose(0, 1).unbind(0), self.roi_size
                )
                proj = self.boxes_pool_project(sampled)
                proj = proj.view(bs, n_boxes, self.d_model).transpose(0, 1)
                boxes_embed = proj if boxes_embed is None else boxes_embed + proj
            if self.boxes_pos_enc_project is not None:
                cx, cy, w, h = boxes.unbind(-1)
                enc = self.pos_enc.encode_boxes(
                    cx.flatten(), cy.flatten(), w.flatten(), h.flatten()
                )
                enc = enc.view(boxes.shape[0], boxes.shape[1], enc.shape[-1])
                proj = self.boxes_pos_enc_project(enc)
                boxes_embed = proj if boxes_embed is None else boxes_embed + proj
            type_embed = self.label_embed(boxes_labels.long())
            return type_embed + boxes_embed, boxes_mask

        sequence_geometry_encoder_cls._encode_boxes = encode_boxes_cpu_safe

