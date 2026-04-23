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

ensure_example_sam3_on_path()
from export_reference import (
    build_preprocessed_image,
    resolve_repo_file,
    resolve_sam3_package_dir,
    to_cpu_contiguous,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export an upstream SAM3 fusion-encoder fixture for cross-framework tests."
    )
    parser.add_argument("--sam3-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
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
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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

    print("[fusion-fixture] building upstream model", flush=True)
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
    print("[fusion-fixture] upstream model ready", flush=True)

    processor = Sam3Processor(
        model=model,
        resolution=args.image_size,
        device=str(device),
        confidence_threshold=0.5,
    )

    point_xy = [0.48, 0.5]
    point_label = 1
    effective_prompt = "visual"

    with torch.inference_mode():
        image_tensor = v2.functional.to_image(image).to(device)
        preprocessed_image = build_preprocessed_image(v2, image_tensor, args.image_size)
        backbone_out = model.backbone.forward_image(preprocessed_image)
        backbone_out.update(model.backbone.forward_text([effective_prompt], device=device))
        geometric_prompt = model._get_dummy_prompt()
        points = torch.tensor(point_xy, device=device, dtype=torch.float32).view(1, 1, 2)
        labels = torch.tensor([point_label], device=device, dtype=torch.long).view(1, 1)
        geometric_prompt.append_points(points, labels)

        prompt, prompt_mask, backbone_out = model._encode_prompt(
            backbone_out=backbone_out,
            find_input=processor.find_stage,
            geometric_prompt=geometric_prompt,
            encode_text=False,
        )
        backbone_out, encoder_out, _ = model._run_encoder(
            backbone_out=backbone_out,
            find_input=processor.find_stage,
            prompt=prompt,
            prompt_mask=prompt_mask,
        )

        tensors = {
            "inputs.image_preprocessed": to_cpu_contiguous(preprocessed_image),
            "inputs.prompt": to_cpu_contiguous(prompt),
            "inputs.prompt_mask": to_cpu_contiguous(prompt_mask.to(torch.uint8)),
            "fusion.memory": to_cpu_contiguous(encoder_out["encoder_hidden_states"]),
            "fusion.pos_embed": to_cpu_contiguous(encoder_out["pos_embed"]),
            "fusion.spatial_shapes": to_cpu_contiguous(encoder_out["spatial_shapes"]),
            "fusion.level_start_index": to_cpu_contiguous(
                encoder_out["level_start_index"]
            ),
            "fusion.valid_ratios": to_cpu_contiguous(encoder_out["valid_ratios"]),
        }
        if encoder_out["padding_mask"] is not None:
            tensors["fusion.padding_mask"] = to_cpu_contiguous(
                encoder_out["padding_mask"].to(torch.uint8)
            )
        for level_idx, feature_map in enumerate(backbone_out["backbone_fpn"]):
            tensors[f"inputs.backbone_fpn.{level_idx}"] = to_cpu_contiguous(feature_map)
        for level_idx, pos in enumerate(backbone_out["vision_pos_enc"]):
            tensors[f"inputs.vision_pos_enc.{level_idx}"] = to_cpu_contiguous(pos)

        encoder_weights = {}
        for name, tensor in model.transformer.encoder.state_dict().items():
            if isinstance(tensor, torch.Tensor):
                encoder_weights[name] = tensor.detach().cpu().contiguous()

    save_file(tensors, str(output_dir / "fixture.safetensors"))
    save_file(encoder_weights, str(output_dir / "encoder_weights.safetensors"))

    first_layer = model.transformer.encoder.layers[0]
    metadata = {
        "bundle_version": 1,
        "image_path": str(Path(args.image).expanduser().resolve()),
        "image_size": args.image_size,
        "checkpoint_path": str(checkpoint_path),
        "bpe_path": str(bpe_path),
        "point_xy_normalized": point_xy,
        "point_label": point_label,
        "d_model": int(first_layer.linear2.out_features),
        "num_layers": int(len(model.transformer.encoder.layers)),
        "num_feature_levels": int(model.transformer.encoder.num_feature_levels),
        "num_heads": int(first_layer.self_attn.num_heads),
        "dim_feedforward": int(first_layer.linear1.out_features),
        "has_padding_mask": encoder_out["padding_mask"] is not None,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[fusion-fixture] wrote fixture to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
