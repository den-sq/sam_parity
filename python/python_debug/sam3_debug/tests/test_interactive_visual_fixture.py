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

FIXTURE_DIR = data_root() / "sam3_interactive_visual_seed"


def load_upstream_modules():
    from torchvision.transforms import v2

    sam3_model_builder = import_upstream_module("sam3.model_builder")
    get_abs_pos = import_upstream_symbol("sam3.model.vitdet", "get_abs_pos")
    window_partition = import_upstream_symbol("sam3.model.vitdet", "window_partition")
    window_unpartition = import_upstream_symbol("sam3.model.vitdet", "window_unpartition")
    apply_cpu_safe_upstream_patches(sam3_model_builder)
    return sam3_model_builder, v2, get_abs_pos, window_partition, window_unpartition


def run_trunk_with_block_outputs(trunk, image_tensor, get_abs_pos, window_partition, window_unpartition):
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
        raise NotImplementedError("interactive visual fixture test does not support retained cls token")

    block_outputs = []
    for block in trunk.blocks:
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
        block_outputs.append(x.permute(0, 3, 1, 2).contiguous())
    return block_outputs


class InteractiveVisualFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if TORCH_IMPORT_ERROR is not None:
            raise unittest.SkipTest(str(TORCH_IMPORT_ERROR))
        require_full_parity_path(
            FIXTURE_DIR / "metadata.json", "interactive visual fixture metadata"
        )
        require_full_parity_path(
            FIXTURE_DIR / "fixture.safetensors", "interactive visual fixture tensors"
        )
        require_full_parity_path(
            FIXTURE_DIR / "vision_backbone_weights.safetensors",
            "interactive visual fixture weights",
        )
        try:
            ensure_example_sam3_on_path()
            from export_reference import build_preprocessed_image

            cls.build_preprocessed_image = staticmethod(build_preprocessed_image)
            (
                sam3_model_builder,
                cls.v2,
                get_abs_pos,
                window_partition,
                window_unpartition,
            ) = load_upstream_modules()
        except FileNotFoundError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        cls.get_abs_pos = staticmethod(get_abs_pos)
        cls.window_partition = staticmethod(window_partition)
        cls.window_unpartition = staticmethod(window_unpartition)
        cls.metadata = json.loads((FIXTURE_DIR / "metadata.json").read_text(encoding="utf-8"))
        cls.fixture = load_file(str(FIXTURE_DIR / "fixture.safetensors"))
        cls.vision_backbone = sam3_model_builder._create_vision_backbone(
            enable_inst_interactivity=False
        ).eval()
        state = load_file(str(FIXTURE_DIR / "vision_backbone_weights.safetensors"))
        missing, unexpected = cls.vision_backbone.load_state_dict(state, strict=False)
        allowed_missing = {
            name
            for name in cls.vision_backbone.state_dict().keys()
            if name.endswith("attn.freqs_cis")
        }
        if set(missing) != allowed_missing or unexpected:
            raise RuntimeError(
                f"interactive visual fixture weights mismatch: missing={missing}, unexpected={unexpected}"
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
        from torchvision.transforms import v2

        image = Image.open(resolve_metadata_path(FIXTURE_DIR, self.metadata["image_path"])).convert("RGB")
        image_tensor = v2.functional.to_image(image)
        preprocessed = self.build_preprocessed_image(
            self.v2, image_tensor, self.metadata["image_size"]
        )
        self.assert_tensor_close(
            preprocessed,
            self.fixture["inputs.image_preprocessed"],
            "inputs.image_preprocessed",
        )

    def test_decoded_image_matches_fixture(self):
        from PIL import Image

        image = Image.open(resolve_metadata_path(FIXTURE_DIR, self.metadata["image_path"])).convert("RGB")
        image_tensor = self.v2.functional.to_image(image)
        self.assert_tensor_close(
            image_tensor,
            self.fixture["inputs.image_decoded_u8"],
            "inputs.image_decoded_u8",
            atol=0.0,
        )

    def test_resized_image_matches_fixture(self):
        resized_u8 = self.v2.functional.resize(
            self.fixture["inputs.image_decoded_u8"],
            [self.metadata["image_size"], self.metadata["image_size"]],
            interpolation=self.v2.InterpolationMode.BILINEAR,
            antialias=True,
        )
        self.assert_tensor_close(
            resized_u8,
            self.fixture["inputs.image_resized_u8"],
            "inputs.image_resized_u8",
            atol=0.0,
        )
        resized_f32 = self.v2.functional.to_dtype(resized_u8, torch.float32, scale=True).unsqueeze(0)
        self.assert_tensor_close(
            resized_f32,
            self.fixture["inputs.image_resized_f32"],
            "inputs.image_resized_f32",
        )
        resized_float_path = (
            self.v2.functional.resize(
                self.fixture["inputs.image_decoded_u8"].to(torch.float32),
                [self.metadata["image_size"], self.metadata["image_size"]],
                interpolation=self.v2.InterpolationMode.BILINEAR,
                antialias=True,
            )
            / 255.0
        ).unsqueeze(0)
        self.assert_tensor_close(
            resized_float_path,
            self.fixture["inputs.image_resized_floatpath_f32"],
            "inputs.image_resized_floatpath_f32",
        )

    def test_trunk_output_matches_fixture(self):
        preprocessed = self.fixture["inputs.image_preprocessed"]
        with torch.inference_mode():
            trunk_last = self.vision_backbone.trunk(preprocessed)[-1]
        self.assert_tensor_close(
            trunk_last,
            self.fixture["vision.trunk.last"],
            "vision.trunk.last",
        )

    def test_trunk_block_outputs_match_fixture(self):
        preprocessed = self.fixture["inputs.image_preprocessed"]
        with torch.inference_mode():
            block_outputs = run_trunk_with_block_outputs(
                self.vision_backbone.trunk,
                preprocessed,
                self.get_abs_pos,
                self.window_partition,
                self.window_unpartition,
            )
        expected_keys = sorted(
            (key for key in self.fixture.keys() if key.startswith("vision.block.")),
            key=lambda key: int(key.rsplit(".", 1)[1]),
        )
        self.assertEqual(len(block_outputs), len(expected_keys))
        for idx, actual in enumerate(block_outputs):
            with self.subTest(block=idx):
                self.assert_tensor_close(
                    actual,
                    self.fixture[f"vision.block.{idx}"],
                    f"vision.block.{idx}",
                )

    def test_neck_last_level_matches_fixture(self):
        preprocessed = self.fixture["inputs.image_preprocessed"]
        with torch.inference_mode():
            trunk_last = self.vision_backbone.trunk(preprocessed)[-1]
            sam3_out, sam3_pos = [], []
            for layer in self.vision_backbone.convs:
                level = layer(trunk_last)
                sam3_out.append(level)
                sam3_pos.append(self.vision_backbone.position_encoding(level).to(level.dtype))
        sam3_out = sam3_out[:-1]
        sam3_pos = sam3_pos[:-1]
        self.assert_tensor_close(
            sam3_out[-1],
            self.fixture["vision.backbone_fpn.last"],
            "vision.backbone_fpn.last",
        )
        self.assert_tensor_close(
            sam3_pos[-1],
            self.fixture["vision.vision_pos_enc.last"],
            "vision.vision_pos_enc.last",
        )


if __name__ == "__main__":
    unittest.main()
