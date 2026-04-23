#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python_debug.sam3_debug.common import (
    apply_cpu_safe_upstream_patches,
    ensure_example_sam3_on_path,
)
from sam3_parity.upstream import (
    import_sam3_module,
    import_sam3_symbol,
    resolve_default_bpe_path,
)

ensure_example_sam3_on_path()
from export_reference import (
    build_preprocessed_image,
    resolve_repo_file,
    to_cpu_contiguous,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export an upstream SAM3 interactive replay comparison bundle."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to sam3.pt or a directory containing sam3.pt.",
    )
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument(
        "--interactive-script",
        required=True,
        help="JSON replay manifest with point clicks to accumulate step by step.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where reference.safetensors and reference.json will be written.",
    )
    parser.add_argument(
        "--bpe-path",
        default=None,
        help="Optional path to bpe_simple_vocab_16e6.txt.gz. Defaults to the installed sam3 package assets.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1008,
        help="Square image size used by the upstream processor.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Explicit torch device, e.g. cpu or cuda. Defaults to cuda when available.",
    )
    return parser.parse_args()


def default_positive_label():
    return 1


def load_interactive_script(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        steps = raw
    else:
        steps = raw.get("steps", [])
    if not steps:
        raise ValueError(f"interactive replay script {path} does not contain any steps")

    parsed_steps = []
    accumulated_points = []
    accumulated_labels = []
    for idx, step in enumerate(steps):
        points = step.get("points", [])
        if not points:
            raise ValueError(f"interactive replay step {idx} does not contain any points")
        step_points = []
        step_labels = []
        for point in points:
            step_points.append([float(point["x"]), float(point["y"])])
            step_labels.append(int(point.get("label", default_positive_label())))
        accumulated_points.extend(step_points)
        accumulated_labels.extend(step_labels)
        parsed_steps.append(
            {
                "name": step.get("name"),
                "step_points_xy_normalized": step_points,
                "step_point_labels": step_labels,
                "accumulated_points_xy_normalized": [list(point) for point in accumulated_points],
                "accumulated_point_labels": list(accumulated_labels),
            }
        )
    return parsed_steps


def main():
    args = parse_args()

    from PIL import Image
    from safetensors.torch import save_file
    from torchvision.transforms import v2

    sam3_model_builder = import_sam3_module("sam3.model_builder")
    TransformerDecoder = import_sam3_symbol("sam3.model.decoder", "TransformerDecoder")
    SequenceGeometryEncoder = import_sam3_symbol(
        "sam3.model.geometry_encoders", "SequenceGeometryEncoder"
    )
    Sam3Processor = import_sam3_symbol("sam3.model.sam3_image_processor", "Sam3Processor")

    checkpoint_path = resolve_repo_file(args.checkpoint, "sam3.pt").expanduser().resolve()
    bpe_path = resolve_default_bpe_path(args.bpe_path)
    script_path = Path(args.interactive_script).expanduser().resolve()
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
    replay_steps = load_interactive_script(script_path)
    print("[interactive-export] building upstream model", flush=True)
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
    print("[interactive-export] upstream model ready", flush=True)
    decoder = model.transformer.decoder
    if hasattr(decoder, "compilable_cord_cache"):
        decoder.compilable_cord_cache = None
        decoder.compilable_stored_size = None

    processor = Sam3Processor(
        model=model,
        resolution=args.image_size,
        device=str(device),
        confidence_threshold=0.5,
    )

    effective_prompt = "visual"

    with torch.inference_mode():
        print("[interactive-export] preprocessing image", flush=True)
        image_tensor = v2.functional.to_image(image).to(device)
        preprocessed_image = build_preprocessed_image(v2, image_tensor, args.image_size)
        print("[interactive-export] running backbone", flush=True)
        backbone_out = model.backbone.forward_image(preprocessed_image)
        text_outputs = model.backbone.forward_text([effective_prompt], device=device)
        backbone_out.update(text_outputs)
        base_backbone_out = backbone_out
        find_input = processor.find_stage
        geometric_prompt = model._get_dummy_prompt()

        tensors = {"inputs.image": to_cpu_contiguous(preprocessed_image)}
        for step_idx, step in enumerate(replay_steps):
            print(
                f"[interactive-export] step {step_idx + 1}/{len(replay_steps)} "
                f"({step.get('name') or f'step_{step_idx:02}'})",
                flush=True,
            )
            for point, label in zip(
                step["step_points_xy_normalized"], step["step_point_labels"]
            ):
                points = torch.tensor(point, device=device, dtype=torch.float32).view(1, 1, 2)
                labels = torch.tensor([label], device=device, dtype=torch.long).view(1, 1)
                geometric_prompt.append_points(points, labels)

            step_backbone_out = dict(base_backbone_out)
            prompt, prompt_mask, step_backbone_out = model._encode_prompt(
                backbone_out=step_backbone_out,
                find_input=find_input,
                geometric_prompt=geometric_prompt.clone(),
                encode_text=False,
            )
            step_backbone_out, encoder_out, _ = model._run_encoder(
                backbone_out=step_backbone_out,
                find_input=find_input,
                prompt=prompt,
                prompt_mask=prompt_mask,
            )
            out = {"encoder_hidden_states": encoder_out["encoder_hidden_states"]}
            out, hs = model._run_decoder(
                pos_embed=encoder_out["pos_embed"],
                memory=out["encoder_hidden_states"],
                src_mask=encoder_out["padding_mask"],
                out=out,
                prompt=prompt,
                prompt_mask=prompt_mask,
                encoder_out=encoder_out,
            )
            model._run_segmentation_heads(
                out=out,
                backbone_out=step_backbone_out,
                img_ids=find_input.img_ids,
                vis_feat_sizes=encoder_out["vis_feat_sizes"],
                encoder_hidden_states=out["encoder_hidden_states"],
                prompt=prompt,
                prompt_mask=prompt_mask,
                hs=hs,
            )

            tensors[f"step.{step_idx}.geometry.features"] = to_cpu_contiguous(prompt)
            tensors[f"step.{step_idx}.geometry.padding_mask"] = to_cpu_contiguous(
                prompt_mask.to(torch.uint8)
            )
            tensors[f"step.{step_idx}.fusion.memory"] = to_cpu_contiguous(
                encoder_out["encoder_hidden_states"]
            )
            tensors[f"step.{step_idx}.decoder.pred_logits"] = to_cpu_contiguous(
                out["pred_logits"]
            )
            tensors[f"step.{step_idx}.decoder.pred_boxes_xyxy"] = to_cpu_contiguous(
                out["pred_boxes_xyxy"]
            )
            tensors[f"step.{step_idx}.segmentation.mask_logits"] = to_cpu_contiguous(
                out["pred_masks"]
            )
            if "presence_logit_dec" in out:
                tensors[f"step.{step_idx}.decoder.presence_logits"] = to_cpu_contiguous(
                    out["presence_logit_dec"]
                )

    save_file(
        tensors,
        str(output_dir / "reference.safetensors"),
        metadata={"bundle_version": "1"},
    )
    metadata = {
        "bundle_version": 1,
        "image_path": str(Path(args.image).expanduser().resolve()),
        "image_size": args.image_size,
        "preprocess_mode": "exact",
        "replay_script_path": str(script_path),
        "effective_prompt": effective_prompt,
        "steps": replay_steps,
        "checkpoint_path": str(checkpoint_path),
        "bpe_path": str(bpe_path),
    }
    with open(output_dir / "reference.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[interactive-export] wrote bundle to {output_dir}", flush=True)
    print(f"saved interactive reference bundle to {output_dir}")
    print(f"  tensors: {output_dir / 'reference.safetensors'}")
    print(f"  metadata: {output_dir / 'reference.json'}")


if __name__ == "__main__":
    main()
