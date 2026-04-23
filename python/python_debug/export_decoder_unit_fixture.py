#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3_parity.upstream import import_sam3_symbol


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export tiny SAM3 decoder fixtures for cross-framework unit tests."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where fixture and weight safetensors files will be written.",
    )
    return parser.parse_args()


def to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to("cpu").contiguous().clone()


def run_decoder_layer_with_debug(
    layer,
    tgt,
    tgt_query_pos,
    memory_text,
    text_attention_mask,
    memory,
    memory_key_padding_mask,
    memory_pos,
    cross_attn_mask,
    presence_token,
):
    debug = {}

    if presence_token is not None:
        tgt = torch.cat([presence_token, tgt], dim=0)
        tgt_query_pos = torch.cat([torch.zeros_like(presence_token), tgt_query_pos], dim=0)

    debug["decoder.layer.0.input_with_presence"] = to_cpu(tgt)
    debug["decoder.layer.0.query_pos_with_presence"] = to_cpu(tgt_query_pos)

    q = k = tgt + tgt_query_pos
    self_attn_out = layer.self_attn(q, k, tgt, attn_mask=None)[0]
    debug["decoder.layer.0.self_attn_output"] = to_cpu(self_attn_out)
    tgt = layer.norm2(tgt + layer.dropout2(self_attn_out))
    debug["decoder.layer.0.post_self_attn"] = to_cpu(tgt)

    text_attn_out = layer.ca_text(
        tgt + tgt_query_pos,
        memory_text,
        memory_text,
        key_padding_mask=text_attention_mask,
    )[0]
    debug["decoder.layer.0.text_attn_output"] = to_cpu(text_attn_out)
    tgt = layer.catext_norm(tgt + layer.catext_dropout(text_attn_out))
    debug["decoder.layer.0.post_text_cross"] = to_cpu(tgt)

    if presence_token is not None:
        presence_token_mask = torch.zeros_like(cross_attn_mask[:, :1, :])
        cross_attn_mask = torch.cat([presence_token_mask, cross_attn_mask], dim=1)
    debug["decoder.layer.0.cross_attn_mask"] = to_cpu(cross_attn_mask)

    image_attn_out = layer.cross_attn(
        query=tgt + tgt_query_pos,
        key=memory + memory_pos,
        value=memory,
        attn_mask=cross_attn_mask,
        key_padding_mask=memory_key_padding_mask.transpose(0, 1),
    )[0]
    debug["decoder.layer.0.image_attn_output"] = to_cpu(image_attn_out)
    tgt = layer.norm1(tgt + layer.dropout1(image_attn_out))
    debug["decoder.layer.0.post_image_cross"] = to_cpu(tgt)

    ffn_hidden = layer.activation(layer.linear1(tgt))
    debug["decoder.layer.0.ffn_hidden"] = to_cpu(ffn_hidden)
    ffn_output = layer.linear2(layer.dropout3(ffn_hidden))
    debug["decoder.layer.0.ffn_output"] = to_cpu(ffn_output)
    tgt = layer.norm3(tgt + layer.dropout4(ffn_output))
    debug["decoder.layer.0.output"] = to_cpu(tgt)

    presence_out = tgt[:1]
    queries = tgt[1:]
    debug["decoder.layer.0.queries_output"] = to_cpu(queries)
    debug["decoder.layer.0.presence_output"] = to_cpu(presence_out)
    return queries, presence_out, debug


def run_decoder_with_debug(
    decoder,
    dot_prod_scoring,
    tgt,
    memory,
    memory_key_padding_mask,
    pos_embed,
    prompt,
    prompt_mask,
    spatial_shapes,
    valid_ratios,
):
    from sam3.model.box_ops import box_cxcywh_to_xyxy
    from sam3.model.model_misc import gen_sineembed_for_position, inverse_sigmoid

    debug = {}
    bs = memory.shape[1]

    output = tgt
    reference_boxes = decoder.reference_points.weight.unsqueeze(1).repeat(1, bs, 1).sigmoid()
    debug["decoder.initial_queries"] = to_cpu(output)
    debug["decoder.initial_reference_boxes"] = to_cpu(reference_boxes)

    valid_ratios_twice = torch.cat([valid_ratios, valid_ratios], dim=-1)
    debug["decoder.valid_ratios_twice"] = to_cpu(valid_ratios_twice)

    presence_out = decoder.presence_token.weight[None].expand(1, bs, -1)
    debug["decoder.initial_presence_state"] = to_cpu(presence_out)

    intermediate = []
    intermediate_ref_boxes = [reference_boxes]
    intermediate_presence_logits = []

    for layer_idx, layer in enumerate(decoder.layers):
        assert layer_idx == 0, "fixture currently expects a single decoder layer"
        reference_points_input = reference_boxes[:, :, None] * valid_ratios_twice[None, :]
        debug[f"decoder.layer.{layer_idx}.reference_boxes"] = to_cpu(reference_boxes)
        debug[f"decoder.layer.{layer_idx}.reference_points_input"] = to_cpu(reference_points_input)

        query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :], decoder.d_model)
        debug[f"decoder.layer.{layer_idx}.query_sine_embed"] = to_cpu(query_sine_embed)
        query_pos = decoder.ref_point_head(query_sine_embed)
        debug[f"decoder.layer.{layer_idx}.query_pos"] = to_cpu(query_pos)

        cross_attn_mask = decoder._get_rpb_matrix(
            reference_boxes,
            (int(spatial_shapes[0, 0].item()), int(spatial_shapes[0, 1].item())),
        ).flatten(0, 1)
        debug[f"decoder.layer.{layer_idx}.cross_attn_mask_pre_presence"] = to_cpu(cross_attn_mask)

        output, presence_out, layer_debug = run_decoder_layer_with_debug(
            layer=layer,
            tgt=output,
            tgt_query_pos=query_pos,
            memory_text=prompt,
            text_attention_mask=prompt_mask,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            memory_pos=pos_embed,
            cross_attn_mask=cross_attn_mask,
            presence_token=presence_out,
        )
        debug.update(layer_debug)

        normed_queries = decoder.norm(output)
        debug[f"decoder.layer.{layer_idx}.normed_queries"] = to_cpu(normed_queries)
        box_delta = decoder.bbox_embed(normed_queries)
        debug[f"decoder.layer.{layer_idx}.box_delta"] = to_cpu(box_delta)
        pred_boxes = (inverse_sigmoid(reference_boxes) + box_delta).sigmoid()
        debug[f"decoder.layer.{layer_idx}.pred_boxes"] = to_cpu(pred_boxes)

        presence_logits = decoder.presence_token_head(
            decoder.presence_token_out_norm(presence_out)
        ).squeeze(-1)
        if decoder.clamp_presence_logits:
            presence_logits.clamp(
                min=-decoder.clamp_presence_logit_max_val,
                max=decoder.clamp_presence_logit_max_val,
            )
        debug[f"decoder.layer.{layer_idx}.presence_logits"] = to_cpu(presence_logits)

        intermediate.append(normed_queries)
        intermediate_presence_logits.append(presence_logits)

    hs_seq_first = torch.stack(intermediate)
    hs = hs_seq_first.transpose(1, 2)
    pred_logits = dot_prod_scoring(hs, prompt, prompt_mask)[-1]
    pred_boxes = pred_boxes.transpose(0, 1).contiguous()
    pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    final_queries = hs[-1].contiguous()
    final_reference_boxes = intermediate_ref_boxes[-1].transpose(0, 1).contiguous()
    final_presence_logits = intermediate_presence_logits[-1].transpose(0, 1).contiguous()

    debug["decoder.dotprod.prompt_after_mlp"] = to_cpu(
        dot_prod_scoring.prompt_mlp(prompt) if dot_prod_scoring.prompt_mlp is not None else prompt
    )
    pooled_prompt = dot_prod_scoring.mean_pool_text(
        dot_prod_scoring.prompt_mlp(prompt) if dot_prod_scoring.prompt_mlp is not None else prompt,
        prompt_mask,
    )
    debug["decoder.dotprod.pooled_prompt"] = to_cpu(pooled_prompt)
    prompt_proj = dot_prod_scoring.prompt_proj(pooled_prompt)
    debug["decoder.dotprod.prompt_proj"] = to_cpu(prompt_proj)
    query_proj = dot_prod_scoring.hs_proj(final_queries)
    debug["decoder.dotprod.query_proj"] = to_cpu(query_proj)
    scores_pre_clamp = torch.matmul(query_proj, prompt_proj.unsqueeze(-1)) * dot_prod_scoring.scale
    debug["decoder.dotprod.scores_pre_clamp"] = to_cpu(scores_pre_clamp)
    debug["decoder.dotprod.scores"] = to_cpu(pred_logits)

    debug["decoder.final_queries"] = to_cpu(final_queries)
    debug["decoder.final_reference_boxes"] = to_cpu(final_reference_boxes)
    debug["decoder.final_pred_boxes"] = to_cpu(pred_boxes)
    debug["decoder.final_pred_boxes_xyxy"] = to_cpu(pred_boxes_xyxy)
    debug["decoder.final_presence_logits"] = to_cpu(final_presence_logits)
    debug["decoder.pred_logits"] = to_cpu(pred_logits)

    actual_hs, actual_reference_boxes, actual_presence, _presence_feats = decoder(
        tgt=tgt,
        memory=memory,
        memory_key_padding_mask=memory_key_padding_mask,
        pos=pos_embed,
        reference_boxes=None,
        level_start_index=torch.tensor([0], dtype=torch.int64, device=tgt.device),
        spatial_shapes=spatial_shapes,
        valid_ratios=valid_ratios,
        tgt_mask=None,
        memory_text=prompt,
        text_attention_mask=prompt_mask,
        apply_dac=False,
    )
    actual_hs = actual_hs.transpose(1, 2)
    actual_reference_boxes = actual_reference_boxes.transpose(1, 2)
    if actual_presence is not None:
        actual_presence = actual_presence.transpose(1, 2)

    if not torch.allclose(final_queries, actual_hs[-1], atol=1e-6, rtol=1e-6):
        raise RuntimeError("manual decoder queries diverged from upstream decoder output")
    if not torch.allclose(final_reference_boxes, actual_reference_boxes[-1], atol=1e-6, rtol=1e-6):
        raise RuntimeError("manual decoder reference boxes diverged from upstream decoder output")
    if actual_presence is None or not torch.allclose(final_presence_logits, actual_presence[-1], atol=1e-6, rtol=1e-6):
        raise RuntimeError("manual decoder presence logits diverged from upstream decoder output")

    return {
        "queries": final_queries,
        "reference_boxes": final_reference_boxes,
        "pred_boxes": pred_boxes,
        "pred_boxes_xyxy": pred_boxes_xyxy,
        "presence_logits": final_presence_logits,
        "pred_logits": pred_logits,
    }, debug


def main():
    args = parse_args()
    TransformerDecoder = import_sam3_symbol("sam3.model.decoder", "TransformerDecoder")
    TransformerDecoderLayer = import_sam3_symbol(
        "sam3.model.decoder", "TransformerDecoderLayer"
    )
    DotProductScoring = import_sam3_symbol(
        "sam3.model.model_misc", "DotProductScoring"
    )
    MLP = import_sam3_symbol("sam3.model.model_misc", "MLP")
    MultiheadAttention = import_sam3_symbol(
        "sam3.model.model_misc", "MultiheadAttentionWrapper"
    )

    torch.manual_seed(1234)
    device = torch.device("cpu")

    d_model = 8
    num_heads = 2
    dim_feedforward = 16
    num_layers = 1
    num_queries = 2
    height = 2
    width = 2

    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=d_model,
        dim_feedforward=dim_feedforward,
        dropout=0.0,
        cross_attention=MultiheadAttention(
            num_heads=num_heads,
            dropout=0.0,
            embed_dim=d_model,
        ),
        n_heads=num_heads,
        use_text_cross_attention=True,
    )
    decoder = TransformerDecoder(
        d_model=d_model,
        frozen=False,
        interaction_layer=None,
        layer=decoder_layer,
        num_layers=num_layers,
        num_queries=num_queries,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=False,
        boxRPB="log",
        use_act_checkpoint=False,
        presence_token=True,
        resolution=None,
        stride=None,
    ).to(device)
    decoder.eval()

    prompt_mlp = MLP(
        input_dim=d_model,
        hidden_dim=dim_feedforward,
        output_dim=d_model,
        num_layers=2,
        dropout=0.0,
        residual=True,
        out_norm=torch.nn.LayerNorm(d_model),
    )
    dot_prod_scoring = DotProductScoring(
        d_model=d_model,
        d_proj=d_model,
        prompt_mlp=prompt_mlp,
    ).to(device)
    dot_prod_scoring.eval()

    memory = torch.randn(height * width, 1, d_model, device=device)
    pos_embed = torch.randn(height * width, 1, d_model, device=device)
    padding_mask = torch.zeros(height * width, 1, dtype=torch.bool, device=device)
    prompt = torch.randn(3, 1, d_model, device=device)
    prompt_mask = torch.tensor([[False, False, True]], dtype=torch.bool, device=device)
    spatial_shapes = torch.tensor([[height, width]], dtype=torch.int64, device=device)
    valid_ratios = torch.tensor([[[1.0, 1.0]]], dtype=torch.float32, device=device)

    tgt = decoder.query_embed.weight.unsqueeze(1).repeat(1, 1, 1)

    outputs, debug_tensors = run_decoder_with_debug(
        decoder=decoder,
        dot_prod_scoring=dot_prod_scoring,
        tgt=tgt,
        memory=memory,
        memory_key_padding_mask=padding_mask,
        pos_embed=pos_embed,
        prompt=prompt,
        prompt_mask=prompt_mask,
        spatial_shapes=spatial_shapes,
        valid_ratios=valid_ratios,
    )

    fixture_tensors = {
        "inputs/memory": to_cpu(memory),
        "inputs/pos_embed": to_cpu(pos_embed),
        "inputs/padding_mask": to_cpu(padding_mask.to(torch.uint8)),
        "inputs/prompt": to_cpu(prompt),
        "inputs/prompt_mask": to_cpu(prompt_mask.to(torch.uint8)),
        "inputs/spatial_shapes": to_cpu(spatial_shapes.to(torch.uint32)),
        "inputs/level_start_index": to_cpu(torch.tensor([0], dtype=torch.uint32)),
        "inputs/valid_ratios": to_cpu(valid_ratios),
        "decoder.output.queries": to_cpu(outputs["queries"]),
        "decoder.output.reference_boxes": to_cpu(outputs["reference_boxes"]),
        "decoder.output.pred_boxes": to_cpu(outputs["pred_boxes"]),
        "decoder.output.pred_boxes_xyxy": to_cpu(outputs["pred_boxes_xyxy"]),
        "decoder.output.presence_logits": to_cpu(outputs["presence_logits"]),
        "decoder.output.pred_logits": to_cpu(outputs["pred_logits"]),
    }
    fixture_tensors.update(debug_tensors)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(fixture_tensors, str(output_dir / "fixture.safetensors"))
    save_file(
        {key: to_cpu(value) for key, value in decoder.state_dict().items()},
        str(output_dir / "decoder_weights.safetensors"),
    )
    save_file(
        {key: to_cpu(value) for key, value in dot_prod_scoring.state_dict().items()},
        str(output_dir / "score_weights.safetensors"),
    )

    metadata = {
        "d_model": d_model,
        "num_heads": num_heads,
        "dim_feedforward": dim_feedforward,
        "num_layers": num_layers,
        "num_queries": num_queries,
        "height": height,
        "width": width,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"saved fixture bundle to {output_dir}")


if __name__ == "__main__":
    main()
