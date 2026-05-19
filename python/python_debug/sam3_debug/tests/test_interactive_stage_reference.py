#!/usr/bin/env python3
import importlib
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
from sam3_parity.paths import bundle_root

BUNDLE_DIR = bundle_root() / "reference_interactive"


def load_upstream_modules():
    from PIL import Image
    from torchvision.transforms import v2

    sam3_model_builder = import_upstream_module("sam3.model_builder")
    TransformerDecoder = import_upstream_symbol("sam3.model.decoder", "TransformerDecoder")
    SequenceGeometryEncoder = import_upstream_symbol(
        "sam3.model.geometry_encoders", "SequenceGeometryEncoder"
    )
    Sam3Processor = import_upstream_symbol(
        "sam3.model.sam3_image_processor", "Sam3Processor"
    )

    apply_cpu_safe_upstream_patches(
        sam3_model_builder,
        transformer_decoder_cls=TransformerDecoder,
        sequence_geometry_encoder_cls=SequenceGeometryEncoder,
    )
    return sam3_model_builder, Sam3Processor, Image, v2


class InteractiveStageReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if TORCH_IMPORT_ERROR is not None:
            raise unittest.SkipTest(str(TORCH_IMPORT_ERROR))
        require_full_parity_path(BUNDLE_DIR / "reference.json", "interactive reference metadata")
        require_full_parity_path(
            BUNDLE_DIR / "reference.safetensors", "interactive reference tensors"
        )
        try:
            ensure_example_sam3_on_path()
            from export_reference import build_preprocessed_image

            cls.build_preprocessed_image = staticmethod(build_preprocessed_image)
            sam3_model_builder, processor_cls, image_cls, cls.v2 = load_upstream_modules()
        except FileNotFoundError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        cls.Image = image_cls
        cls.metadata = json.loads((BUNDLE_DIR / "reference.json").read_text(encoding="utf-8"))
        cls.fixture = load_file(str(BUNDLE_DIR / "reference.safetensors"))

        cls.model = sam3_model_builder.build_sam3_image_model(
            checkpoint_path=str(resolve_metadata_path(BUNDLE_DIR, cls.metadata["checkpoint_path"])),
            bpe_path=str(resolve_metadata_path(BUNDLE_DIR, cls.metadata["bpe_path"])),
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

        image = cls.Image.open(resolve_metadata_path(BUNDLE_DIR, cls.metadata["image_path"])).convert("RGB")
        image_tensor = cls.v2.functional.to_image(image)
        cls.preprocessed_image = cls.build_preprocessed_image(
            cls.v2, image_tensor, cls.metadata["image_size"]
        )
        cls.preprocessed_image = cls.preprocessed_image.to("cpu")
        with torch.inference_mode():
            cls.base_backbone_out = cls.model.backbone.forward_image(cls.preprocessed_image)
            cls.base_backbone_out.update(
                cls.model.backbone.forward_text(["visual"], device="cpu")
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

    def test_fusion_memory_matches_reference_steps(self):
        for step_idx in (0, 2):
            outputs = self._run_step(step_idx)
            self.assert_tensor_close(
                outputs["encoder_hidden_states"],
                self.fixture[f"step.{step_idx}.fusion.memory"],
                f"step.{step_idx}.fusion.memory",
            )

    def test_segmentation_mask_logits_match_reference_steps(self):
        for step_idx in (0, 2):
            outputs = self._run_step(step_idx)
            self.assert_tensor_close(
                outputs["pred_masks"],
                self.fixture[f"step.{step_idx}.segmentation.mask_logits"],
                f"step.{step_idx}.segmentation.mask_logits",
            )

    def _run_step(self, step_idx: int):
        step = self.metadata["steps"][step_idx]
        with torch.inference_mode():
            geometric_prompt = self.model._get_dummy_prompt()
            for point, label in zip(
                step["accumulated_points_xy_normalized"], step["accumulated_point_labels"]
            ):
                points = torch.tensor(point, dtype=torch.float32).view(1, 1, 2)
                labels = torch.tensor([label], dtype=torch.long).view(1, 1)
                geometric_prompt.append_points(points, labels)
            backbone_out = dict(self.base_backbone_out)
            prompt, prompt_mask, backbone_out = self.model._encode_prompt(
                backbone_out=backbone_out,
                find_input=self.processor.find_stage,
                geometric_prompt=geometric_prompt,
                encode_text=False,
            )
            backbone_out, encoder_out, decoder_out = self.model._run_encoder(
                backbone_out=backbone_out,
                find_input=self.processor.find_stage,
                prompt=prompt,
                prompt_mask=prompt_mask,
            )
            out = {"encoder_hidden_states": encoder_out["encoder_hidden_states"]}
            out, hs = self.model._run_decoder(
                pos_embed=encoder_out["pos_embed"],
                memory=out["encoder_hidden_states"],
                src_mask=encoder_out["padding_mask"],
                out=out,
                prompt=prompt,
                prompt_mask=prompt_mask,
                encoder_out=encoder_out,
            )
            self.model._run_segmentation_heads(
                out=out,
                backbone_out=backbone_out,
                img_ids=self.processor.find_stage.img_ids,
                vis_feat_sizes=encoder_out["vis_feat_sizes"],
                encoder_hidden_states=out["encoder_hidden_states"],
                prompt=prompt,
                prompt_mask=prompt_mask,
                hs=hs,
            )
        return {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
            "pred_masks": out["pred_masks"],
        }


if __name__ == "__main__":
    unittest.main()
