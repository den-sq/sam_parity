#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python_debug.sam3.common import (
    apply_cpu_safe_upstream_patches,
    ensure_example_sam3_on_path,
)
from python_debug.sam3.exports.export_interactive_reference import load_interactive_script

ensure_example_sam3_on_path()
from export_reference import (
    build_preprocessed_image,
    resolve_repo_file,
    resolve_sam3_package_dir,
    to_cpu_contiguous,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export an upstream SAM3 geometry fixture for an interactive replay step."
    )
    parser.add_argument("--sam3-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--interactive-script", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--bpe-path", default=None)
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    sam3_package_dir = resolve_sam3_package_dir(Path(args.sam3_repo))
    sys.path.insert(0, str(sam3_package_dir.parent))

    from PIL import Image
    from safetensors.torch import save_file
    from torchvision.transforms import v2

    import sam3.model_builder as sam3_model_builder
    from sam3.model.decoder import TransformerDecoder
    from sam3.model.geometry_encoders import SequenceGeometryEncoder
    from sam3.model.position_encoding import PositionEmbeddingSine
    from sam3.model.sam3_image_processor import Sam3Processor

    checkpoint_path = resolve_repo_file(args.checkpoint, "sam3.pt").expanduser().resolve()
    bpe_path = (
        Path(args.bpe_path).expanduser().resolve()
        if args.bpe_path is not None
        else sam3_package_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    )
    script_path = Path(args.interactive_script).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    replay_steps = load_interactive_script(script_path)
    if args.step_index < 0 or args.step_index >= len(replay_steps):
        raise ValueError(
            f"step index {args.step_index} is out of range for {len(replay_steps)} replay step(s)"
        )

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if device.type == "cpu":
        apply_cpu_safe_upstream_patches(
            sam3_model_builder,
            transformer_decoder_cls=TransformerDecoder,
            sequence_geometry_encoder_cls=SequenceGeometryEncoder,
        )

    image = Image.open(args.image).convert("RGB")
    effective_prompt = "visual"

    print("[interactive-geometry-fixture] building upstream model", flush=True)
    model = sam3_model_builder.build_sam3_image_model(
        checkpoint_path=str(checkpoint_path),
        bpe_path=str(bpe_path),
        device=str(device),
        eval_mode=True,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )
    print("[interactive-geometry-fixture] upstream model ready", flush=True)

    processor = Sam3Processor(
        model=model,
        resolution=args.image_size,
        device=str(device),
        confidence_threshold=0.5,
    )

    with torch.inference_mode():
        image_tensor = v2.functional.to_image(image).to(device)
        preprocessed_image = build_preprocessed_image(v2, image_tensor, args.image_size)
        backbone_out = model.backbone.forward_image(preprocessed_image)
        text_outputs = model.backbone.forward_text([effective_prompt], device=device)
        backbone_out.update(text_outputs)
        find_input = processor.find_stage
        geometry_encoder = model.geometry_encoder
        geometric_prompt = model._get_dummy_prompt()

        for step in replay_steps[: args.step_index + 1]:
            for point, label in zip(
                step["step_points_xy_normalized"], step["step_point_labels"]
            ):
                points = torch.tensor(point, device=device, dtype=torch.float32).view(1, 1, 2)
                labels = torch.tensor([label], device=device, dtype=torch.long).view(1, 1)
                geometric_prompt.append_points(points, labels)

        _backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = model._get_img_feats(
            backbone_out, find_input.img_ids
        )
        seq_first_img_feats = img_feats[-1]
        seq_first_img_pos_embeds = img_pos_embeds[-1]
        pooled_img_feats = seq_first_img_feats
        if geometry_encoder.points_pool_project is not None or geometry_encoder.boxes_pool_project is not None:
            pooled_img_feats = geometry_encoder.img_pre_norm(seq_first_img_feats)
            H, W = vis_feat_sizes[-1]
            pooled_img_feats = pooled_img_feats.permute(1, 2, 0).view(
                seq_first_img_feats.shape[1],
                seq_first_img_feats.shape[2],
                H,
                W,
            )

        points = geometric_prompt.point_embeddings
        points_mask = geometric_prompt.point_mask
        points_labels = geometric_prompt.point_labels
        if points is None or points_mask is None or points_labels is None:
            raise ValueError("interactive geometry fixture expected accumulated point prompts")

        point_features, point_padding_mask = geometry_encoder._encode_points(
            points, points_mask, points_labels, pooled_img_feats
        )

        x, y = points.unbind(-1)
        enc_x, enc_y = geometry_encoder.pos_enc._encode_xy(x.flatten(), y.flatten())
        enc_x = enc_x.view(points.shape[0], points.shape[1], enc_x.shape[-1])
        enc_y = enc_y.view(points.shape[0], points.shape[1], enc_y.shape[-1])
        point_position = torch.cat([enc_x, enc_y], dim=-1)

        grid = points.transpose(0, 1).unsqueeze(2)
        grid = (grid * 2) - 1
        point_sampled = torch.nn.functional.grid_sample(
            pooled_img_feats, grid, align_corners=False
        )
        point_sampled = point_sampled.squeeze(-1).permute(2, 0, 1)

        point_label_embed = geometry_encoder.label_embed(points_labels.long())
        point_direct_proj = (
            geometry_encoder.points_direct_project(points)
            if geometry_encoder.points_direct_project is not None
            else None
        )
        point_pool_proj = (
            geometry_encoder.points_pool_project(point_sampled)
            if geometry_encoder.points_pool_project is not None
            else None
        )
        point_pos_enc_proj = (
            geometry_encoder.points_pos_enc_project(point_position)
            if geometry_encoder.points_pos_enc_project is not None
            else None
        )

        features = point_features
        padding_mask = point_padding_mask
        if geometry_encoder.cls_embed is not None:
            cls_embed = geometry_encoder.cls_embed.weight.view(1, 1, geometry_encoder.d_model)
            cls_embed = cls_embed.repeat(1, point_features.shape[1], 1)
            cls_mask = torch.zeros(
                point_features.shape[1],
                1,
                dtype=padding_mask.dtype,
                device=padding_mask.device,
            )
            features = torch.cat([features, cls_embed], dim=0)
            padding_mask = torch.cat([padding_mask, cls_mask], dim=1)

        if geometry_encoder.final_proj is not None:
            features = geometry_encoder.final_proj(features)
        features = geometry_encoder.norm(features)

        tensors = {
            "inputs/points_xy": to_cpu_contiguous(points),
            "inputs/point_labels": to_cpu_contiguous(points_labels),
            "inputs/image_features": to_cpu_contiguous(seq_first_img_feats.permute(1, 2, 0).view(
                seq_first_img_feats.shape[1],
                seq_first_img_feats.shape[2],
                vis_feat_sizes[-1][0],
                vis_feat_sizes[-1][1],
            )),
            "inputs/image_pos_embeds": to_cpu_contiguous(
                seq_first_img_pos_embeds.permute(1, 2, 0).view(
                    seq_first_img_pos_embeds.shape[1],
                    seq_first_img_pos_embeds.shape[2],
                    vis_feat_sizes[-1][0],
                    vis_feat_sizes[-1][1],
                )
            ),
            "inputs/pool_image_features": to_cpu_contiguous(pooled_img_feats),
            "helper/points_position": to_cpu_contiguous(point_position),
            "helper/points_sampled": to_cpu_contiguous(point_sampled),
            "geometry/point_label_embed": to_cpu_contiguous(point_label_embed),
            "geometry/point_sampled_raw": to_cpu_contiguous(point_sampled.clone()),
            "geometry/point_pos_enc": to_cpu_contiguous(point_position.clone()),
            "geometry/point_features": to_cpu_contiguous(point_features),
            "geometry/features_initial_norm": to_cpu_contiguous(features),
        }
        if point_direct_proj is not None:
            tensors["geometry/point_direct_proj"] = to_cpu_contiguous(point_direct_proj)
        if point_pool_proj is not None:
            tensors["geometry/point_pool_proj"] = to_cpu_contiguous(point_pool_proj)
        if point_pos_enc_proj is not None:
            tensors["geometry/point_pos_enc_proj"] = to_cpu_contiguous(point_pos_enc_proj)

        for layer_idx, layer in enumerate(geometry_encoder.encode):
            features = layer(
                tgt=features,
                memory=seq_first_img_feats,
                tgt_key_padding_mask=padding_mask,
                pos=seq_first_img_pos_embeds,
            )
            tensors[f"geometry/features_after_layer_{layer_idx}"] = to_cpu_contiguous(features)
        if geometry_encoder.encode_norm is not None:
            features = geometry_encoder.encode_norm(features)
        tensors["geometry/returned_features"] = to_cpu_contiguous(features)
        tensors["geometry/padding_mask"] = to_cpu_contiguous(padding_mask.to(torch.uint8))

        weights = {}
        for name, tensor in geometry_encoder.state_dict().items():
            if isinstance(tensor, torch.Tensor):
                weights[name] = tensor.detach().cpu().contiguous()

    save_file(tensors, str(output_dir / "fixture.safetensors"), metadata={"bundle_version": "1"})
    save_file(weights, str(output_dir / "weights.safetensors"), metadata={"bundle_version": "1"})

    metadata = {
        "d_model": int(geometry_encoder.d_model),
        "num_heads": int(geometry_encoder.encode[0].self_attn.num_heads),
        "dim_feedforward": int(geometry_encoder.encode[0].linear1.out_features),
        "num_layers": int(len(geometry_encoder.encode)),
        "roi_size": int(geometry_encoder.roi_size),
        "image_size": int(args.image_size),
        "step_index": int(args.step_index),
        "image_path": str(Path(args.image).expanduser().resolve()),
        "replay_script_path": str(script_path),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[interactive-geometry-fixture] wrote fixture to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
