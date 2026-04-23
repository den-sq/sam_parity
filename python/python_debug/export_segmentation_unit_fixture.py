#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export tiny SAM3 segmentation fixtures for cross-framework unit tests."
    )
    parser.add_argument(
        "--sam3-repo",
        required=True,
        help="Path to the local facebookresearch/sam3 repository root or inner sam3 package.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where fixture and weight safetensors files will be written.",
    )
    return parser.parse_args()


def resolve_sam3_package_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "model_builder.py").exists():
        return path
    if (path / "sam3" / "model_builder.py").exists():
        return path / "sam3"
    raise FileNotFoundError(
        f"could not find sam3/model_builder.py under {path}; pass either the repo root or the inner sam3 package directory"
    )


def to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to("cpu").contiguous().clone()


def embed_pixels_with_debug(head, backbone_feats, encoder_hidden_states):
    debug = {}
    batch_size, channels, height, width = backbone_feats[-1].shape
    encoder_visual_embed = encoder_hidden_states.permute(1, 2, 0).reshape(
        batch_size, channels, height, width
    )
    debug["segmentation.encoder_visual_embed"] = to_cpu(encoder_visual_embed)

    backbone_visual_feats = [feat.clone() for feat in backbone_feats]
    backbone_visual_feats[-1] = encoder_visual_embed

    prev_fpn = backbone_visual_feats[-1]
    debug["segmentation.pixel_decoder.initial_prev_fpn"] = to_cpu(prev_fpn)
    fpn_feats = backbone_visual_feats[:-1]
    for layer_idx, bb_feat in enumerate(fpn_feats[::-1]):
        conv_idx = 0 if getattr(head.pixel_decoder, "shared_conv", False) else layer_idx
        debug[f"segmentation.pixel_decoder.stage.{layer_idx}.curr_fpn"] = to_cpu(bb_feat)
        upsampled = torch.nn.functional.interpolate(
            prev_fpn,
            size=bb_feat.shape[-2:],
            mode=head.pixel_decoder.interpolation_mode,
        )
        debug[f"segmentation.pixel_decoder.stage.{layer_idx}.upsampled_prev_fpn"] = to_cpu(
            upsampled
        )
        prev_fpn = bb_feat + upsampled
        debug[f"segmentation.pixel_decoder.stage.{layer_idx}.sum"] = to_cpu(prev_fpn)
        prev_fpn = head.pixel_decoder.conv_layers[conv_idx](prev_fpn)
        debug[f"segmentation.pixel_decoder.stage.{layer_idx}.conv"] = to_cpu(prev_fpn)
        prev_fpn = torch.relu(head.pixel_decoder.norms[conv_idx](prev_fpn))
        debug[f"segmentation.pixel_decoder.stage.{layer_idx}.output"] = to_cpu(prev_fpn)

    oracle = head._embed_pixels(
        backbone_feats=backbone_feats,
        image_ids=torch.tensor([0], dtype=torch.int64, device=encoder_hidden_states.device),
        encoder_hidden_states=encoder_hidden_states,
    )
    if not torch.allclose(prev_fpn, oracle, atol=1e-6, rtol=1e-6):
        max_abs = (prev_fpn - oracle).abs().max().item()
        raise RuntimeError(
            f"manual pixel embed diverged from upstream segmentation head (max_abs_diff={max_abs})"
        )

    return prev_fpn, debug


def run_segmentation_with_debug(
    head,
    backbone_feats,
    obj_queries,
    encoder_hidden_states,
    prompt,
    prompt_mask,
):
    encoder_hidden_states_input = encoder_hidden_states
    debug = {
        "segmentation.encoder_hidden_states_input": to_cpu(encoder_hidden_states_input),
    }

    if head.cross_attend_prompt is not None:
        normed_encoder = head.cross_attn_norm(encoder_hidden_states)
        debug["segmentation.encoder_hidden_states_normed"] = to_cpu(normed_encoder)
        prompt_attn = head.cross_attend_prompt(
            query=normed_encoder,
            key=prompt,
            value=prompt,
            key_padding_mask=prompt_mask,
        )[0]
        debug["segmentation.prompt_attn"] = to_cpu(prompt_attn)
        encoder_hidden_states = prompt_attn + encoder_hidden_states
        debug["segmentation.encoder_hidden_states_after_prompt"] = to_cpu(
            encoder_hidden_states
        )

    pixel_embed, pixel_debug = embed_pixels_with_debug(
        head, backbone_feats, encoder_hidden_states
    )
    debug.update(pixel_debug)
    debug["segmentation.pixel_embed"] = to_cpu(pixel_embed)

    instance_embeds = head.instance_seg_head(pixel_embed)
    debug["segmentation.instance_embeds"] = to_cpu(instance_embeds)
    semantic_logits = head.semantic_seg_head(pixel_embed)
    debug["segmentation.semantic_logits"] = to_cpu(semantic_logits)

    query_input = obj_queries[-1]
    debug["segmentation.mask_predictor.query_input"] = to_cpu(query_input)
    query_embed = head.mask_predictor.mask_embed(query_input)
    debug["segmentation.mask_predictor.query_embed"] = to_cpu(query_embed)
    pixel_flat = instance_embeds.reshape(
        instance_embeds.shape[0], instance_embeds.shape[1], -1
    )
    debug["segmentation.mask_predictor.pixel_flat"] = to_cpu(pixel_flat)
    mask_logits = torch.matmul(query_embed, pixel_flat).reshape(
        query_input.shape[0], query_input.shape[1], instance_embeds.shape[-2], instance_embeds.shape[-1]
    )
    debug["segmentation.mask_logits"] = to_cpu(mask_logits)

    actual = head(
        backbone_feats=backbone_feats,
        obj_queries=obj_queries,
        image_ids=torch.tensor([0], dtype=torch.int64, device=encoder_hidden_states_input.device),
        encoder_hidden_states=encoder_hidden_states_input,
        prompt=prompt,
        prompt_mask=prompt_mask,
    )
    if not torch.allclose(mask_logits, actual["pred_masks"], atol=1e-6, rtol=1e-6):
        max_abs = (mask_logits - actual["pred_masks"]).abs().max().item()
        raise RuntimeError(
            f"manual segmentation mask logits diverged from upstream output (max_abs_diff={max_abs})"
        )
    if not torch.allclose(semantic_logits, actual["semantic_seg"], atol=1e-6, rtol=1e-6):
        max_abs = (semantic_logits - actual["semantic_seg"]).abs().max().item()
        raise RuntimeError(
            f"manual segmentation semantic logits diverged from upstream output (max_abs_diff={max_abs})"
        )

    return {
        "mask_logits": mask_logits,
        "semantic_logits": semantic_logits,
    }, debug


def main():
    args = parse_args()
    sam3_package_dir = resolve_sam3_package_dir(Path(args.sam3_repo))
    sys.path.insert(0, str(sam3_package_dir.parent))

    from sam3.model.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
    from sam3.model.model_misc import MultiheadAttentionWrapper as MultiheadAttention

    torch.manual_seed(1234)
    device = torch.device("cpu")

    hidden_dim = 8
    upsampling_stages = 3
    num_queries = 3
    num_query_layers = 2

    pixel_decoder = PixelDecoder(
        hidden_dim=hidden_dim,
        num_upsampling_stages=upsampling_stages,
        interpolation_mode="nearest",
    )
    cross_attend_prompt = MultiheadAttention(
        num_heads=8,
        dropout=0.0,
        embed_dim=hidden_dim,
    )
    head = UniversalSegmentationHead(
        hidden_dim=hidden_dim,
        upsampling_stages=upsampling_stages,
        pixel_decoder=pixel_decoder,
        aux_masks=False,
        no_dec=False,
        act_ckpt=False,
        presence_head=False,
        dot_product_scorer=None,
        cross_attend_prompt=cross_attend_prompt,
    ).to(device)
    head.eval()

    backbone_feats = [
        torch.randn(1, hidden_dim, 8, 8, device=device),
        torch.randn(1, hidden_dim, 4, 4, device=device),
        torch.randn(1, hidden_dim, 2, 2, device=device),
    ]
    encoder_hidden_states = torch.randn(4, 1, hidden_dim, device=device)
    prompt = torch.randn(5, 1, hidden_dim, device=device)
    prompt_mask = torch.tensor([[False, False, False, True, True]], dtype=torch.bool, device=device)
    obj_queries = torch.randn(num_query_layers, 1, num_queries, hidden_dim, device=device)

    outputs, debug_tensors = run_segmentation_with_debug(
        head=head,
        backbone_feats=backbone_feats,
        obj_queries=obj_queries,
        encoder_hidden_states=encoder_hidden_states,
        prompt=prompt,
        prompt_mask=prompt_mask,
    )

    fixture_tensors = {
        "inputs/backbone_fpn.0": to_cpu(backbone_feats[0]),
        "inputs/backbone_fpn.1": to_cpu(backbone_feats[1]),
        "inputs/backbone_fpn.2": to_cpu(backbone_feats[2]),
        "inputs/decoder_queries": to_cpu(obj_queries[-1]),
        "inputs/encoder_hidden_states": to_cpu(encoder_hidden_states),
        "inputs/prompt": to_cpu(prompt),
        "inputs/prompt_mask": to_cpu(prompt_mask.to(torch.uint8)),
        "segmentation.output.mask_logits": to_cpu(outputs["mask_logits"]),
        "segmentation.output.semantic_logits": to_cpu(outputs["semantic_logits"]),
    }
    fixture_tensors.update(debug_tensors)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(fixture_tensors, str(output_dir / "fixture.safetensors"))
    save_file(
        {key: to_cpu(value) for key, value in head.state_dict().items()},
        str(output_dir / "segmentation_weights.safetensors"),
    )

    metadata = {
        "hidden_dim": hidden_dim,
        "upsampling_stages": upsampling_stages,
        "num_queries": num_queries,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"saved fixture bundle to {output_dir}")


if __name__ == "__main__":
    main()
