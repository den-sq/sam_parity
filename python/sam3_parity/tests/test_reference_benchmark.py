import hashlib

import numpy as np
import pytest

from sam3_parity.reference_benchmark import (
    annotation_samples,
    parse_args,
    parse_tiff_description,
    update_mask_digest,
)


def test_parse_tiff_description_requires_object():
    assert parse_tiff_description('{"shape":[2,3,4]}') == {"shape": [2, 3, 4]}
    assert parse_tiff_description(None) == {}
    with pytest.raises(ValueError, match="JSON object"):
        parse_tiff_description("[1,2,3]")


def test_annotation_samples_groups_and_clips_frames():
    points = [
        [10.0, 20.0, -1.0],
        [11.0, 21.0, 0.0],
        [12.0, 22.0, 1.2],
        [13.0, 23.0, 1.4],
        [14.0, 24.0, 4.0],
    ]
    assert annotation_samples(points, frame_count=4) == [
        (0, [[11.0, 21.0]]),
        (1, [[12.0, 22.0], [13.0, 23.0]]),
    ]


def test_output_hash_includes_frame_index_shape_and_binary_bytes():
    first = hashlib.sha256()
    update_mask_digest(first, 0, np.array([[0, 1], [1, 0]], dtype=bool))

    same = hashlib.sha256()
    update_mask_digest(same, 0, np.array([[[0, 1], [1, 0]]], dtype=np.uint8))
    assert first.hexdigest() == same.hexdigest()

    different_frame = hashlib.sha256()
    update_mask_digest(
        different_frame, 1, np.array([[0, 1], [1, 0]], dtype=np.uint8)
    )
    assert first.hexdigest() != different_frame.hexdigest()


def test_math_sdpa_ablation_is_explicit():
    args = parse_args(
        [
            "--fixture",
            "fixture.tif",
            "--checkpoint",
            "checkpoint.pt",
            "--frames-dir",
            "frames",
            "--output-dir",
            "results",
            "--dtype",
            "f32",
            "--sdpa-backend",
            "math",
        ]
    )
    assert args.dtype == "f32"
    assert args.sdpa_backend == "math"
