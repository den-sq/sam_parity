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
    to_cpu_nchw,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export an upstream SAM3 visual fixture for interactive parity debugging."
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
    from sam3.model.vitdet import get_abs_pos, window_partition, window_unpartition

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
        )

    def run_trunk_with_block_outputs(trunk, image_tensor):
        x = trunk.patch_embed(image_tensor)
        height, width = x.shape[1], x.shape[2]

        if trunk.pos_embed is not None:
            x = x + get_abs_pos(
                trunk.pos_embed,
                trunk.pretrain_use_cls_token,
                (height, width),
                trunk.retain_cls_token,
                tiling=trunk.tile_abs_pos,
            )

        x = trunk.ln_pre(x)

        if trunk.retain_cls_token:
            raise NotImplementedError(
                "interactive visual fixture export does not support retained cls token"
            )

        block_outputs = []
        for block_idx, block in enumerate(trunk.blocks):
            shortcut = x
            x_norm1 = block.norm1(x)

            if block.window_size > 0:
                hw = (x_norm1.shape[1], x_norm1.shape[2])
                x_attn, pad_hw = window_partition(x_norm1, block.window_size)
            else:
                hw = None
                x_attn = x_norm1

            x_attn = block.ls1(block.attn(x_attn))
            if block.window_size > 0:
                x_attn = window_unpartition(x_attn, block.window_size, pad_hw, hw)

            x = shortcut + block.dropout(block.drop_path(x_attn))
            x_norm2 = block.norm2(x)
            mlp_fc1 = block.mlp.fc1(x_norm2)
            mlp_gelu = block.mlp.act(mlp_fc1)
            mlp_fc2 = block.mlp.fc2(mlp_gelu)
            mlp_output = block.dropout(block.drop_path(block.ls2(mlp_fc2)))
            x = x + mlp_output
            block_outputs.append((block_idx, to_cpu_nchw(x)))

        return to_cpu_contiguous(x.permute(0, 3, 1, 2)), block_outputs

    image = Image.open(args.image).convert("RGB")

    print("[interactive-visual-fixture] building upstream model", flush=True)
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
    print("[interactive-visual-fixture] upstream model ready", flush=True)

    with torch.inference_mode():
        image_tensor = v2.functional.to_image(image).to(device)
        resized_uint8 = v2.functional.resize(
            image_tensor,
            [args.image_size, args.image_size],
            interpolation=v2.InterpolationMode.BILINEAR,
            antialias=True,
        )
        resized_f32 = v2.functional.to_dtype(resized_uint8, torch.float32, scale=True)
        resized_float_path = (
            v2.functional.resize(
                image_tensor.to(torch.float32),
                [args.image_size, args.image_size],
                interpolation=v2.InterpolationMode.BILINEAR,
                antialias=True,
            )
            / 255.0
        )
        preprocessed_image = build_preprocessed_image(v2, image_tensor, args.image_size)

        trunk_last, block_outputs = run_trunk_with_block_outputs(
            model.backbone.vision_backbone.trunk,
            preprocessed_image,
        )

        backbone_out = model.backbone.forward_image(preprocessed_image)
        backbone_fpn_last = backbone_out["backbone_fpn"][-1]
        vision_pos_last = backbone_out["vision_pos_enc"][-1]

        tensors = {
            "inputs.image_decoded_u8": to_cpu_contiguous(image_tensor),
            "inputs.image_resized_u8": to_cpu_contiguous(resized_uint8),
            "inputs.image_resized_f32": to_cpu_contiguous(resized_f32.unsqueeze(0)),
            "inputs.image_resized_floatpath_f32": to_cpu_contiguous(
                resized_float_path.unsqueeze(0)
            ),
            "inputs.image_preprocessed": to_cpu_contiguous(preprocessed_image),
            "vision.trunk.last": to_cpu_contiguous(trunk_last),
            "vision.backbone_fpn.last": to_cpu_contiguous(backbone_fpn_last),
            "vision.vision_pos_enc.last": to_cpu_contiguous(vision_pos_last),
        }
        for block_idx, feature_map in block_outputs:
            tensors[f"vision.block.{block_idx}"] = feature_map

        weights = {}
        for name, tensor in model.backbone.vision_backbone.state_dict().items():
            if isinstance(tensor, torch.Tensor):
                if tensor.is_complex():
                    continue
                weights[name] = tensor.detach().cpu().contiguous()

    save_file(
        tensors,
        str(output_dir / "fixture.safetensors"),
        metadata={"bundle_version": "1"},
    )
    save_file(
        weights,
        str(output_dir / "vision_backbone_weights.safetensors"),
        metadata={"bundle_version": "1"},
    )

    metadata = {
        "bundle_version": 1,
        "image_path": str(Path(args.image).expanduser().resolve()),
        "image_size": args.image_size,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[interactive-visual-fixture] wrote fixture to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
