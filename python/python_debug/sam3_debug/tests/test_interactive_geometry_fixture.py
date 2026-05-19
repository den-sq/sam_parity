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
    import_upstream_module,
    import_upstream_symbol,
    require_full_parity_path,
)
from sam3_parity.paths import data_root

FIXTURE_DIR = data_root() / "sam3_interactive_geometry_seed"


def load_upstream_modules():
    sam3_model_builder = import_upstream_module("sam3.model_builder")
    Prompt = import_upstream_symbol("sam3.model.geometry_encoders", "Prompt")
    SequenceGeometryEncoder = import_upstream_symbol(
        "sam3.model.geometry_encoders", "SequenceGeometryEncoder"
    )
    apply_cpu_safe_upstream_patches(
        sam3_model_builder,
        sequence_geometry_encoder_cls=SequenceGeometryEncoder,
    )
    return sam3_model_builder, Prompt


class InteractiveGeometryFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if TORCH_IMPORT_ERROR is not None:
            raise unittest.SkipTest(str(TORCH_IMPORT_ERROR))
        require_full_parity_path(
            FIXTURE_DIR / "metadata.json", "interactive geometry fixture metadata"
        )
        require_full_parity_path(
            FIXTURE_DIR / "fixture.safetensors", "interactive geometry fixture tensors"
        )
        require_full_parity_path(
            FIXTURE_DIR / "weights.safetensors", "interactive geometry fixture weights"
        )
        try:
            sam3_model_builder, prompt_cls = load_upstream_modules()
        except FileNotFoundError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        cls.prompt_cls = prompt_cls
        cls.metadata = json.loads((FIXTURE_DIR / "metadata.json").read_text(encoding="utf-8"))
        cls.fixture = load_file(str(FIXTURE_DIR / "fixture.safetensors"))
        cls.encoder = sam3_model_builder._create_geometry_encoder().eval()
        state = load_file(str(FIXTURE_DIR / "weights.safetensors"))
        missing, unexpected = cls.encoder.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"interactive geometry fixture weights mismatch: missing={missing}, unexpected={unexpected}"
            )

        cls.points_xy = cls.fixture["inputs/points_xy"]
        cls.point_labels = cls.fixture["inputs/point_labels"].long()
        cls.pool_image_features = cls.fixture["inputs/pool_image_features"]
        cls.image_features_bchw = cls.fixture["inputs/image_features"]
        cls.image_pos_bchw = cls.fixture["inputs/image_pos_embeds"]
        batch_size, channels, height, width = cls.image_features_bchw.shape
        cls.image_size_hw = (height, width)
        cls.image_features_seq = cls.image_features_bchw.permute(2, 3, 0, 1).reshape(
            height * width, batch_size, channels
        )
        cls.image_pos_seq = cls.image_pos_bchw.permute(2, 3, 0, 1).reshape(
            height * width, batch_size, channels
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

    def test_point_helpers_match_fixture(self):
        x, y = self.points_xy.unbind(-1)
        enc_x, enc_y = self.encoder.pos_enc._encode_xy(x.flatten(), y.flatten())
        enc_x = enc_x.view(self.points_xy.shape[0], self.points_xy.shape[1], enc_x.shape[-1])
        enc_y = enc_y.view(self.points_xy.shape[0], self.points_xy.shape[1], enc_y.shape[-1])
        point_position = torch.cat([enc_x, enc_y], dim=-1)

        grid = self.points_xy.transpose(0, 1).unsqueeze(2)
        grid = (grid * 2) - 1
        point_sampled = torch.nn.functional.grid_sample(
            self.pool_image_features, grid, align_corners=False
        )
        point_sampled = point_sampled.squeeze(-1).permute(2, 0, 1)

        self.assert_tensor_close(
            point_position, self.fixture["helper/points_position"], "helper/points_position"
        )
        self.assert_tensor_close(
            point_sampled, self.fixture["helper/points_sampled"], "helper/points_sampled"
        )

    def test_point_feature_composition_matches_fixture(self):
        point_label_embed = self.encoder.label_embed(self.point_labels.long())
        point_direct_proj = self.encoder.points_direct_project(self.points_xy)

        grid = self.points_xy.transpose(0, 1).unsqueeze(2)
        grid = (grid * 2) - 1
        point_sampled = torch.nn.functional.grid_sample(
            self.pool_image_features, grid, align_corners=False
        )
        point_sampled = point_sampled.squeeze(-1).permute(2, 0, 1)
        point_pool_proj = self.encoder.points_pool_project(point_sampled)

        x, y = self.points_xy.unbind(-1)
        enc_x, enc_y = self.encoder.pos_enc._encode_xy(x.flatten(), y.flatten())
        enc_x = enc_x.view(self.points_xy.shape[0], self.points_xy.shape[1], enc_x.shape[-1])
        enc_y = enc_y.view(self.points_xy.shape[0], self.points_xy.shape[1], enc_y.shape[-1])
        point_position = torch.cat([enc_x, enc_y], dim=-1)
        point_pos_enc_proj = self.encoder.points_pos_enc_project(point_position)

        point_features = (
            point_label_embed + point_direct_proj + point_pool_proj + point_pos_enc_proj
        )

        self.assert_tensor_close(
            point_label_embed,
            self.fixture["geometry/point_label_embed"],
            "geometry/point_label_embed",
        )
        self.assert_tensor_close(
            point_direct_proj,
            self.fixture["geometry/point_direct_proj"],
            "geometry/point_direct_proj",
        )
        self.assert_tensor_close(
            point_pool_proj,
            self.fixture["geometry/point_pool_proj"],
            "geometry/point_pool_proj",
        )
        self.assert_tensor_close(
            point_pos_enc_proj,
            self.fixture["geometry/point_pos_enc_proj"],
            "geometry/point_pos_enc_proj",
        )
        self.assert_tensor_close(
            point_features, self.fixture["geometry/point_features"], "geometry/point_features"
        )

    def test_encoder_output_matches_fixture(self):
        point_mask = torch.zeros(
            (self.points_xy.shape[1], self.points_xy.shape[0]), dtype=torch.bool
        )
        prompt = self.prompt_cls(
            point_embeddings=self.points_xy,
            point_mask=point_mask,
            point_labels=self.point_labels,
        )
        features, padding_mask = self.encoder(
            geo_prompt=prompt,
            img_feats=[self.image_features_seq],
            img_sizes=[self.image_size_hw],
            img_pos_embeds=[self.image_pos_seq],
        )
        self.assert_tensor_close(
            features, self.fixture["geometry/returned_features"], "geometry/returned_features"
        )
        self.assertTrue(
            torch.equal(
                padding_mask.to(torch.uint8).cpu(), self.fixture["geometry/padding_mask"].cpu()
            ),
            "geometry/padding_mask",
        )


if __name__ == "__main__":
    unittest.main()
