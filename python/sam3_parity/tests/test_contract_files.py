import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from sam3_parity.paths import data_root


def test_migration_inventory_exists():
    inventory = REPO_ROOT / "docs" / "MIGRATION_INVENTORY.md"
    assert inventory.exists()
    text = inventory.read_text(encoding="utf-8")
    assert "reference.safetensors" in text
    assert "video_tracker_strict_port_matrix.json" in text


def test_matrix_manifest_is_valid_json():
    manifest_path = REPO_ROOT / "docs" / "video_tracker_strict_port_matrix.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(manifest.get("bundles"), list)
    assert manifest["bundles"]
    for bundle in manifest["bundles"]:
        assert bundle.get("artifact_dir")
        assert not Path(bundle["artifact_dir"]).is_absolute()
        assert ".." not in Path(bundle["artifact_dir"]).parts


def test_seed_fixture_directories_exist():
    root = data_root()
    if not root.exists():
        pytest.skip(f"seed fixture root is not present: {root}")
    expected = {
        "sam3_decoder_unit",
        "sam3_fusion_unit",
        "sam3_geometry_unit",
        "sam3_interactive_geometry_seed",
        "sam3_interactive_visual_seed",
        "sam3_segmentation_unit",
    }
    actual = {path.name for path in root.iterdir() if path.is_dir()}
    assert expected.issubset(actual)
