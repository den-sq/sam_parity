#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export tiny SAM3 geometry fixtures for cross-framework unit tests."
    )
    parser.add_argument(
        "--sam3-repo",
        required=True,
        help="Path to the local facebookresearch/sam3 repository root or inner sam3 package.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where fixture.safetensors, weights.safetensors, and metadata.json will be written.",
    )
    return parser.parse_args()


def resolve_sam3_package_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "model_builder.py").exists():
        return path
    if (path / "sam3" / "model_builder.py").exists():
        return path / "sam3"
    raise FileNotFoundError(
        f"could not find sam3/model_builder.py under {path};"
        "pass either the repo root or the inner sam3 package directory"
    )


def to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to("cpu").contiguous().clone()


def encode_points_position(pos_enc, points_xy: torch.Tensor) -> torch.Tensor:
    x, y = points_xy.unbind(-1)
    enc_x, enc_y = pos_enc._encode_xy(x.flatten(), y.flatten())
    enc_x = enc_x.view(points_xy.shape[0], points_xy.shape[1], enc_x.shape[-1])
    enc_y = enc_y.view(points_xy.shape[0], points_xy.shape[1], enc_y.shape[-1])
    return torch.cat([enc_y, enc_x], -1)


def bilinear_interpolate_single(input: torch.Tensor, batch_idx: int, y: float, x: float) -> torch.Tensor:
    _, _, height, width = input.shape

    y = max(y, 0.0)
    x = max(x, 0.0)
    y_low = int(y)
    x_low = int(x)

    if y_low >= height - 1:
        y_low = height - 1
        y_high = height - 1
    else:
        y_high = y_low + 1

    if x_low >= width - 1:
        x_low = width - 1
        x_high = width - 1
    else:
        x_high = x_low + 1

    ly = y - y_low
    lx = x - x_low
    hy = 1.0 - ly
    hx = 1.0 - lx

    v1 = input[batch_idx, :, y_low, x_low]
    v2 = input[batch_idx, :, y_low, x_high]
    v3 = input[batch_idx, :, y_high, x_low]
    v4 = input[batch_idx, :, y_high, x_high]
    return hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4


def roi_align_with_debug(
    input: torch.Tensor,
    boxes_per_image: list[torch.Tensor],
    output_size: int,
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = False,
):
    import torchvision

    _, channels, height, width = input.shape
    pooled_height = output_size
    pooled_width = output_size

    roi_rows = []
    for batch_idx, boxes in enumerate(boxes_per_image):
        for box in boxes:
            roi_rows.append(
                torch.tensor([batch_idx, box[0], box[1], box[2], box[3]], dtype=input.dtype, device=input.device)
            )
    rois = torch.stack(roi_rows, dim=0)

    roi_batch_ind = rois[:, 0].to(torch.int64)
    offset = 0.5 if aligned else 0.0
    roi_start_w = rois[:, 1] * spatial_scale - offset
    roi_start_h = rois[:, 2] * spatial_scale - offset
    roi_end_w = rois[:, 3] * spatial_scale - offset
    roi_end_h = rois[:, 4] * spatial_scale - offset

    roi_width = roi_end_w - roi_start_w
    roi_height = roi_end_h - roi_start_h
    if not aligned:
        roi_width = roi_width.clamp(min=1.0)
        roi_height = roi_height.clamp(min=1.0)

    bin_size_h = roi_height / pooled_height
    bin_size_w = roi_width / pooled_width
    if sampling_ratio > 0:
        roi_bin_grid_h = torch.full_like(roi_height, sampling_ratio, dtype=torch.int64)
        roi_bin_grid_w = torch.full_like(roi_width, sampling_ratio, dtype=torch.int64)
    else:
        roi_bin_grid_h = torch.ceil(roi_height / pooled_height).to(torch.int64)
        roi_bin_grid_w = torch.ceil(roi_width / pooled_width).to(torch.int64)

    max_grid_h = int(roi_bin_grid_h.max().item())
    max_grid_w = int(roi_bin_grid_w.max().item())
    num_rois = rois.shape[0]

    sample_y = torch.zeros((num_rois, pooled_height, max_grid_h), dtype=input.dtype, device=input.device)
    sample_x = torch.zeros((num_rois, pooled_width, max_grid_w), dtype=input.dtype, device=input.device)
    sample_values = torch.zeros(
        (num_rois, channels, pooled_height, pooled_width, max_grid_h, max_grid_w),
        dtype=input.dtype,
        device=input.device,
    )
    output = torch.zeros((num_rois, channels, pooled_height, pooled_width), dtype=input.dtype, device=input.device)

    for roi_idx in range(num_rois):
        grid_h = int(roi_bin_grid_h[roi_idx].item())
        grid_w = int(roi_bin_grid_w[roi_idx].item())
        count = max(grid_h * grid_w, 1)

        for ph in range(pooled_height):
            for iy in range(grid_h):
                y = (
                    roi_start_h[roi_idx]
                    + ph * bin_size_h[roi_idx]
                    + (iy + 0.5) * (bin_size_h[roi_idx] / grid_h)
                )
                sample_y[roi_idx, ph, iy] = y

            for pw in range(pooled_width):
                for ix in range(grid_w):
                    x = (
                        roi_start_w[roi_idx]
                        + pw * bin_size_w[roi_idx]
                        + (ix + 0.5) * (bin_size_w[roi_idx] / grid_w)
                    )
                    sample_x[roi_idx, pw, ix] = x

                accum = torch.zeros((channels,), dtype=input.dtype, device=input.device)
                for iy in range(grid_h):
                    y = sample_y[roi_idx, ph, iy].item()
                    for ix in range(grid_w):
                        x = sample_x[roi_idx, pw, ix].item()
                        val = bilinear_interpolate_single(input, int(roi_batch_ind[roi_idx].item()), y, x)
                        sample_values[roi_idx, :, ph, pw, iy, ix] = val
                        accum += val
                output[roi_idx, :, ph, pw] = accum / count

    oracle = torchvision.ops.roi_align(
        input,
        boxes_per_image,
        output_size,
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio,
        aligned=aligned,
    )
    if not torch.allclose(output, oracle, atol=1e-6, rtol=1e-6):
        max_abs = (output - oracle).abs().max().item()
        raise RuntimeError(f"debug roi_align wrapper diverged from torchvision.ops.roi_align (max_abs_diff={max_abs})")

    debug = {
        "helper/boxes_feature_xyxy": to_cpu(rois[:, 1:]),
        "helper/boxes_roi_params": to_cpu(
            torch.stack(
                [
                    roi_start_w,
                    roi_start_h,
                    roi_end_w,
                    roi_end_h,
                    roi_width,
                    roi_height,
                    bin_size_w,
                    bin_size_h,
                ],
                dim=-1,
            )
        ),
        "helper/boxes_grid_size": to_cpu(torch.stack([roi_bin_grid_h, roi_bin_grid_w], dim=-1).to(torch.float32)),
        "helper/boxes_sample_y": to_cpu(sample_y),
        "helper/boxes_sample_x": to_cpu(sample_x),
        "helper/boxes_sample_values": to_cpu(sample_values),
    }
    return oracle, debug


def encode_boxes_with_debug(encoder, boxes, boxes_mask, boxes_labels, img_feats):
    debug = {}
    boxes_embed = None
    n_boxes, bs = boxes.shape[:2]

    type_embed = encoder.label_embed(boxes_labels.long())
    debug["geometry/label_embed"] = to_cpu(type_embed)

    if encoder.boxes_direct_project is not None:
        proj = encoder.boxes_direct_project(boxes)
        debug["geometry/direct_proj"] = to_cpu(proj)
        boxes_embed = proj if boxes_embed is None else boxes_embed + proj

    if encoder.boxes_pool_project is not None:
        from sam3.model.box_ops import box_cxcywh_to_xyxy

        h, w = img_feats.shape[-2:]
        boxes_xyxy = box_cxcywh_to_xyxy(boxes)
        scale = torch.tensor([w, h, w, h], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device).view(
            1, 1, 4
        )
        sampled, roi_debug = roi_align_with_debug(
            img_feats,
            (boxes_xyxy * scale).float().transpose(0, 1).unbind(0),
            encoder.roi_size,
        )
        debug.update(roi_debug)
        debug["geometry/pooled_boxes_raw"] = to_cpu(sampled)
        proj = encoder.boxes_pool_project(sampled)
        proj = proj.view(bs, n_boxes, encoder.d_model).transpose(0, 1)
        debug["geometry/pool_proj"] = to_cpu(proj)
        boxes_embed = proj if boxes_embed is None else boxes_embed + proj

    if encoder.boxes_pos_enc_project is not None:
        cx, cy, w, h = boxes.unbind(-1)
        enc = encoder.pos_enc.encode_boxes(cx.flatten(), cy.flatten(), w.flatten(), h.flatten())
        enc = enc.view(boxes.shape[0], boxes.shape[1], enc.shape[-1])
        proj = encoder.boxes_pos_enc_project(enc)
        debug["geometry/pos_enc_proj"] = to_cpu(proj)
        boxes_embed = proj if boxes_embed is None else boxes_embed + proj

    final = type_embed + boxes_embed
    debug["geometry/box_features"] = to_cpu(final)
    return final, boxes_mask, debug


def encode_boxes_cpu_safe(encoder, boxes, boxes_mask, boxes_labels, img_feats):
    boxes_embed, boxes_mask, _debug = encode_boxes_with_debug(
        encoder,
        boxes=boxes,
        boxes_mask=boxes_mask,
        boxes_labels=boxes_labels,
        img_feats=img_feats,
    )
    return boxes_embed, boxes_mask


def run_geometry_with_debug(encoder, geo_prompt, img_feats, img_sizes, img_pos_embeds):
    from sam3.model.geometry_encoders import concat_padded_sequences

    debug = {}

    points = geo_prompt.point_embeddings
    points_mask = geo_prompt.point_mask
    points_labels = geo_prompt.point_labels
    boxes = geo_prompt.box_embeddings
    boxes_mask = geo_prompt.box_mask
    boxes_labels = geo_prompt.box_labels

    seq_first_img_feats = img_feats[-1]
    seq_first_img_pos_embeds = img_pos_embeds[-1]

    pooled_img_feats = None
    if encoder.points_pool_project is not None or encoder.boxes_pool_project is not None:
        pooled_img_feats = encoder.img_pre_norm(seq_first_img_feats)
        h, w = img_sizes[-1]
        n, c = pooled_img_feats.shape[-2:]
        pooled_img_feats = pooled_img_feats.permute(1, 2, 0).view(n, c, h, w)
    else:
        pooled_img_feats = img_feats

    final_embeds, final_mask = encoder._encode_points(
        points=points,
        points_mask=points_mask,
        points_labels=points_labels,
        img_feats=pooled_img_feats,
    )

    boxes_embeds, boxes_mask, box_debug = encode_boxes_with_debug(
        encoder,
        boxes=boxes,
        boxes_mask=boxes_mask,
        boxes_labels=boxes_labels,
        img_feats=pooled_img_feats,
    )
    debug.update(box_debug)
    final_embeds, final_mask = concat_padded_sequences(
        final_embeds, final_mask, boxes_embeds, boxes_mask
    )

    if encoder.cls_embed is not None:
        bs = final_embeds.shape[1]
        cls = encoder.cls_embed.weight.view(1, 1, encoder.d_model).repeat(1, bs, 1)
        cls_mask = torch.zeros(bs, 1, dtype=final_mask.dtype, device=final_mask.device)
        final_embeds, final_mask = concat_padded_sequences(final_embeds, final_mask, cls, cls_mask)

    if encoder.final_proj is not None:
        final_embeds = encoder.norm(encoder.final_proj(final_embeds))
        debug["geometry/features_initial_norm"] = to_cpu(final_embeds)

    if encoder.encode is not None:
        for idx, layer in enumerate(encoder.encode):
            final_embeds = layer(
                tgt=final_embeds,
                memory=seq_first_img_feats,
                tgt_key_padding_mask=final_mask,
                pos=seq_first_img_pos_embeds,
            )
            debug[f"geometry/features_after_layer_{idx}"] = to_cpu(final_embeds)
        final_embeds = encoder.encode_norm(final_embeds)

    debug["geometry/features_final"] = to_cpu(final_embeds)
    return final_embeds, final_mask, debug, pooled_img_feats


def main():
    args = parse_args()
    sam3_package_dir = resolve_sam3_package_dir(Path(args.sam3_repo))
    sys.path.insert(0, str(sam3_package_dir.parent))

    from sam3.model.box_ops import box_cxcywh_to_xyxy
    from sam3.model.encoder import TransformerEncoderLayer
    from sam3.model.geometry_encoders import Prompt, SequenceGeometryEncoder
    from sam3.model.model_misc import MultiheadAttentionWrapper as MultiheadAttention
    from sam3.model.position_encoding import PositionEmbeddingSine

    torch.manual_seed(1234)
    device = torch.device("cpu")

    d_model = 8
    num_heads = 2
    dim_feedforward = 16
    num_layers = 1
    roi_size = 2
    height = 4
    width = 4

    pos_enc = PositionEmbeddingSine(
        num_pos_feats=d_model,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=None,
    )
    layer = TransformerEncoderLayer(
        activation="relu",
        d_model=d_model,
        dim_feedforward=dim_feedforward,
        dropout=0.0,
        pos_enc_at_attn=False,
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=num_heads,
            dropout=0.0,
            embed_dim=d_model,
            batch_first=False,
        ),
        cross_attention=MultiheadAttention(
            num_heads=num_heads,
            dropout=0.0,
            embed_dim=d_model,
            batch_first=False,
        ),
    )
    encoder = SequenceGeometryEncoder(
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=d_model,
        pos_enc=pos_enc,
        num_layers=num_layers,
        layer=layer,
        roi_size=roi_size,
        add_cls=True,
        add_post_encode_proj=True,
        use_act_ckpt=False,
    ).to(device)
    encoder.eval()
    encoder._encode_boxes = lambda boxes, boxes_mask, boxes_labels, img_feats: encode_boxes_cpu_safe(
        encoder,
        boxes=boxes,
        boxes_mask=boxes_mask,
        boxes_labels=boxes_labels,
        img_feats=img_feats,
    )

    image_features = torch.randn(1, d_model, height, width, device=device)
    image_pos_embeds = pos_enc(torch.zeros((1, 1, height, width), device=device))
    seq_first_image_features = image_features.permute(2, 3, 0, 1).reshape(height * width, 1, d_model)
    seq_first_image_pos = image_pos_embeds.permute(2, 3, 0, 1).reshape(height * width, 1, d_model)

    points_xy = torch.tensor(
        [[[0.2, 0.3]], [[0.8, 0.7]]],
        dtype=torch.float32,
        device=device,
    )
    boxes_cxcywh = torch.tensor(
        [[[0.45, 0.55, 0.35, 0.5]]],
        dtype=torch.float32,
        device=device,
    )
    box_labels = torch.tensor([[1]], dtype=torch.uint8, device=device)

    geo_prompt = Prompt(
        box_embeddings=boxes_cxcywh,
        box_labels=box_labels.bool(),
    )

    final_embeds, final_mask, debug_tensors, pooled_img_feats = run_geometry_with_debug(
        encoder,
        geo_prompt,
        img_feats=[seq_first_image_features],
        img_sizes=[(height, width)],
        img_pos_embeds=[seq_first_image_pos],
    )
    direct_embeds, direct_mask = encoder(
        geo_prompt,
        [seq_first_image_features],
        [(height, width)],
        [seq_first_image_pos],
    )
    if not torch.allclose(final_embeds, direct_embeds):
        raise RuntimeError("manual geometry debug path diverged from upstream encoder output")
    if not torch.equal(final_mask, direct_mask):
        raise RuntimeError("manual geometry debug path diverged from upstream encoder mask")

    point_grid = (points_xy.transpose(0, 1).unsqueeze(2) * 2.0) - 1.0
    point_sampled = F.grid_sample(pooled_img_feats, point_grid, align_corners=False)
    point_sampled = point_sampled.squeeze(-1).permute(2, 0, 1)
    point_position = encode_points_position(pos_enc, points_xy)

    boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)
    box_scale = torch.tensor([width, height, width, height], dtype=boxes_xyxy.dtype, device=device).view(
        1, 1, 4
    )
    box_sampled, box_roi_debug = roi_align_with_debug(
        pooled_img_feats,
        (boxes_xyxy * box_scale).float().transpose(0, 1).unbind(0),
        roi_size,
    )
    box_position = pos_enc.encode_boxes(
        boxes_cxcywh[..., 0].flatten(),
        boxes_cxcywh[..., 1].flatten(),
        boxes_cxcywh[..., 2].flatten(),
        boxes_cxcywh[..., 3].flatten(),
    ).view(boxes_cxcywh.shape[0], boxes_cxcywh.shape[1], -1)

    fixture_tensors = {
        "inputs/image_features": to_cpu(image_features),
        "inputs/image_pos_embeds": to_cpu(image_pos_embeds),
        "inputs/pool_image_features": to_cpu(pooled_img_feats),
        "inputs/points_xy": to_cpu(points_xy),
        "inputs/boxes_cxcywh": to_cpu(boxes_cxcywh),
        "inputs/box_labels": to_cpu(box_labels),
        "helper/points_position": to_cpu(point_position),
        "helper/boxes_position": to_cpu(box_position),
        "helper/points_sampled": to_cpu(point_sampled),
        "helper/boxes_sampled_raw": to_cpu(box_sampled),
        "geometry/padding_mask": to_cpu(final_mask.to(torch.uint8)),
        "geometry/returned_features": to_cpu(final_embeds),
    }
    fixture_tensors.update(box_roi_debug)
    fixture_tensors.update(debug_tensors)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(fixture_tensors, str(output_dir / "fixture.safetensors"))
    save_file(
        {key: to_cpu(value) for key, value in encoder.state_dict().items()},
        str(output_dir / "weights.safetensors"),
    )
    metadata = {
        "d_model": d_model,
        "num_heads": num_heads,
        "dim_feedforward": dim_feedforward,
        "num_layers": num_layers,
        "roi_size": roi_size,
        "height": height,
        "width": width,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"saved fixture bundle to {output_dir}")


if __name__ == "__main__":
    main()
