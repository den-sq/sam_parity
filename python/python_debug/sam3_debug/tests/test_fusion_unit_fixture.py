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
    apply_cpu_safe_upstream_patches,
    ensure_example_sam3_on_path,
    import_upstream_module,
    import_upstream_symbol,
    require_full_parity_path,
    resolve_metadata_path,
)
from sam3_parity.paths import data_root

FIXTURE_DIR = data_root() / "sam3_fusion_unit"


def load_upstream_modules():
    from torchvision.transforms import v2

    sam3_model_builder = import_upstream_module("sam3.model_builder")
    TransformerDecoder = import_upstream_symbol("sam3.model.decoder", "TransformerDecoder")
    Sam3Processor = import_upstream_symbol(
        "sam3.model.sam3_image_processor", "Sam3Processor"
    )

    apply_cpu_safe_upstream_patches(
        sam3_model_builder,
        transformer_decoder_cls=TransformerDecoder,
    )
    return sam3_model_builder, Sam3Processor, v2


class FusionFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if TORCH_IMPORT_ERROR is not None:
            raise unittest.SkipTest(str(TORCH_IMPORT_ERROR))
        require_full_parity_path(FIXTURE_DIR / "metadata.json", "fusion fixture metadata")
        require_full_parity_path(FIXTURE_DIR / "fixture.safetensors", "fusion fixture tensors")
        try:
            ensure_example_sam3_on_path()
            from export_reference import build_preprocessed_image

            cls.build_preprocessed_image = staticmethod(build_preprocessed_image)
            sam3_model_builder, processor_cls, cls.v2 = load_upstream_modules()
        except FileNotFoundError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        cls.metadata = json.loads((FIXTURE_DIR / "metadata.json").read_text(encoding="utf-8"))
        cls.fixture = load_file(str(FIXTURE_DIR / "fixture.safetensors"))

        cls.model = sam3_model_builder.build_sam3_image_model(
            checkpoint_path=str(resolve_metadata_path(FIXTURE_DIR, cls.metadata["checkpoint_path"])),
            bpe_path=str(resolve_metadata_path(FIXTURE_DIR, cls.metadata["bpe_path"])),
            device="cpu",
            eval_mode=True,
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        )
        cls.processor = processor_cls(
            model=cls.model,
            resolution=cls.metadata["image_size"],
            device="cpu",
            confidence_threshold=0.5,
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

    def test_preprocessed_image_matches_fixture(self):
        from PIL import Image

        image = Image.open(resolve_metadata_path(FIXTURE_DIR, self.metadata["image_path"])).convert("RGB")
        image_tensor = self.v2.functional.to_image(image)
        preprocessed = self.build_preprocessed_image(
            self.v2, image_tensor, self.metadata["image_size"]
        )
        self.assert_tensor_close(
            preprocessed,
            self.fixture["inputs.image_preprocessed"],
            "inputs.image_preprocessed",
        )

    def test_fusion_memory_matches_fixture(self):
        visual_features = []
        visual_pos = []
        for idx in range(self.metadata["num_feature_levels"]):
            visual_features.append(self.fixture[f"inputs.backbone_fpn.{idx}"])
            visual_pos.append(self.fixture[f"inputs.vision_pos_enc.{idx}"])
        for idx in range(self.metadata["num_feature_levels"], 4):
            key = f"inputs.backbone_fpn.{idx}"
            if key in self.fixture:
                visual_features.append(self.fixture[key])
                visual_pos.append(self.fixture[f"inputs.vision_pos_enc.{idx}"])

        # Match `_get_img_feats` in upstream by using only the last `num_feature_levels`.
        visual_features = visual_features[-self.metadata["num_feature_levels"] :]
        visual_pos = visual_pos[-self.metadata["num_feature_levels"] :]
        img_feats = [x.flatten(2).permute(2, 0, 1) for x in visual_features]
        img_pos = [x.flatten(2).permute(2, 0, 1) for x in visual_pos]
        feat_sizes = [tuple(x.shape[-2:]) for x in visual_pos]
        prompt = self.fixture["inputs.prompt"]
        prompt_mask = self.fixture["inputs.prompt_mask"].bool()
        out = self.model.transformer.encoder(
            src=img_feats.copy(),
            src_key_padding_mask=None,
            src_pos=img_pos.copy(),
            prompt=prompt,
            prompt_pos=torch.zeros_like(prompt),
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=feat_sizes,
            encoder_extra_kwargs=None,
        )
        self.assert_tensor_close(out["memory"], self.fixture["fusion.memory"], "fusion.memory")
        self.assert_tensor_close(out["pos_embed"], self.fixture["fusion.pos_embed"], "fusion.pos_embed")
        if self.metadata.get("has_padding_mask", False):
            self.assert_tensor_close(
                out["padding_mask"].to(torch.uint8),
                self.fixture["fusion.padding_mask"],
                "fusion.padding_mask",
                atol=0.0,
            )
        else:
            self.assertTrue(
                out["padding_mask"] is None or torch.count_nonzero(out["padding_mask"]) == 0,
                "fusion.padding_mask should be absent or all zeros for this fixture",
            )
        self.assert_tensor_close(
            out["spatial_shapes"], self.fixture["fusion.spatial_shapes"], "fusion.spatial_shapes", atol=0.0
        )
        self.assert_tensor_close(
            out["level_start_index"],
            self.fixture["fusion.level_start_index"],
            "fusion.level_start_index",
            atol=0.0,
        )
        self.assert_tensor_close(
            out["valid_ratios"], self.fixture["fusion.valid_ratios"], "fusion.valid_ratios"
        )


if __name__ == "__main__":
    unittest.main()
