#!/usr/bin/env python3
import json
import os
import sys
import unittest
from pathlib import Path

import pytest

pytestmark = pytest.mark.full_parity
try:
    import torch
    from safetensors.torch import load_file
except ImportError as exc:
    torch = None
    load_file = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python_debug.sam3_debug.common import (
    import_upstream_symbol,
    require_full_parity_path,
)
from sam3_parity.paths import data_root


FIXTURE_DIR = data_root() / "sam3_segmentation_unit"


def load_upstream_modules():
    PixelDecoder = import_upstream_symbol(
        "sam3.model.maskformer_segmentation", "PixelDecoder"
    )
    UniversalSegmentationHead = import_upstream_symbol(
        "sam3.model.maskformer_segmentation", "UniversalSegmentationHead"
    )
    MultiheadAttention = import_upstream_symbol(
        "sam3.model.model_misc", "MultiheadAttentionWrapper"
    )

    return PixelDecoder, UniversalSegmentationHead, MultiheadAttention


class SegmentationFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if TORCH_IMPORT_ERROR is not None:
            raise unittest.SkipTest(str(TORCH_IMPORT_ERROR))
        require_full_parity_path(
            FIXTURE_DIR / "metadata.json", "segmentation fixture metadata"
        )
        require_full_parity_path(
            FIXTURE_DIR / "fixture.safetensors", "segmentation fixture tensors"
        )
        require_full_parity_path(
            FIXTURE_DIR / "segmentation_weights.safetensors",
            "segmentation fixture weights",
        )
        try:
            pixel_decoder_cls, head_cls, attn_cls = load_upstream_modules()
        except FileNotFoundError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        cls.metadata = json.loads((FIXTURE_DIR / "metadata.json").read_text(encoding="utf-8"))
        cls.fixture = load_file(str(FIXTURE_DIR / "fixture.safetensors"))
        pixel_decoder = pixel_decoder_cls(
            hidden_dim=cls.metadata["hidden_dim"],
            num_upsampling_stages=cls.metadata["upsampling_stages"],
            interpolation_mode="nearest",
        )
        cross_attend_prompt = attn_cls(
            num_heads=8,
            dropout=0.0,
            embed_dim=cls.metadata["hidden_dim"],
        )
        cls.head = head_cls(
            hidden_dim=cls.metadata["hidden_dim"],
            upsampling_stages=cls.metadata["upsampling_stages"],
            pixel_decoder=pixel_decoder,
            aux_masks=False,
            no_dec=False,
            act_ckpt=False,
            presence_head=False,
            dot_product_scorer=None,
            cross_attend_prompt=cross_attend_prompt,
        ).eval()
        state = load_file(str(FIXTURE_DIR / "segmentation_weights.safetensors"))
        missing, unexpected = cls.head.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"segmentation fixture weights mismatch: missing={missing}, unexpected={unexpected}"
            )

    def assert_tensor_close(self, actual, expected, name, atol=1e-5):
        self.assertEqual(tuple(actual.shape), tuple(expected.shape), f"{name} shape mismatch")
        torch.testing.assert_close(
            actual.detach().cpu().float(),
            expected.detach().cpu().float(),
            atol=atol,
            rtol=0.0,
            msg=name,
        )

    def test_mask_predictor_path_matches_fixture(self):
        decoder_queries = self.fixture["inputs/decoder_queries"]
        query_embed = self.head.mask_predictor.mask_embed(decoder_queries)
        self.assert_tensor_close(
            query_embed,
            self.fixture["segmentation.mask_predictor.query_embed"],
            "segmentation.mask_predictor.query_embed",
        )
        instance_embeds = self.head.instance_seg_head(self.fixture["segmentation.pixel_embed"])
        pixel_flat = instance_embeds.reshape(
            instance_embeds.shape[0], instance_embeds.shape[1], -1
        )
        mask_logits = torch.matmul(query_embed, pixel_flat).reshape(
            decoder_queries.shape[0],
            decoder_queries.shape[1],
            instance_embeds.shape[-2],
            instance_embeds.shape[-1],
        )
        self.assert_tensor_close(
            mask_logits,
            self.fixture["segmentation.mask_logits"],
            "segmentation.mask_logits",
        )

    def test_segmentation_mask_logits_matches_fixture(self):
        backbone_feats = []
        for idx in range(3):
            backbone_feats.append(self.fixture[f"inputs/backbone_fpn.{idx}"])
        out = self.head(
            backbone_feats=backbone_feats,
            obj_queries=self.fixture["inputs/decoder_queries"].unsqueeze(0),
            image_ids=torch.tensor([0], dtype=torch.int64),
            encoder_hidden_states=self.fixture["inputs/encoder_hidden_states"],
            prompt=self.fixture["inputs/prompt"],
            prompt_mask=self.fixture["inputs/prompt_mask"].bool(),
        )
        self.assert_tensor_close(
            out["pred_masks"],
            self.fixture["segmentation.mask_logits"],
            "segmentation.mask_logits",
        )
        self.assert_tensor_close(
            out["semantic_seg"],
            self.fixture["segmentation.semantic_logits"],
            "segmentation.semantic_logits",
        )


if __name__ == "__main__":
    unittest.main()
