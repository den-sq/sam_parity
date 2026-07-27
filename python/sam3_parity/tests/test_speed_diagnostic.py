import pytest

from sam3_parity.speed_diagnostic import derived_stage_costs, summarize_samples


def test_summarize_samples_reports_stable_distribution():
    summary = summarize_samples([4.0, 1.0, 3.0, 2.0])
    assert summary["sample_count"] == 4
    assert summary["mean_ms"] == pytest.approx(2.5)
    assert summary["median_ms"] == pytest.approx(2.5)
    assert summary["minimum_ms"] == 1.0
    assert summary["maximum_ms"] == 4.0


def test_derived_stage_costs_partition_frame():
    stages = {
        "image_encoder": {"median_ms": 100.0},
        "tracker_base": {"median_ms": 40.0},
        "tracker_with_history": {"median_ms": 55.0},
        "tracker_full": {"median_ms": 70.0},
    }
    derived = derived_stage_costs(stages)
    assert derived["previous_memory_conditioning_increment_ms"] == 15.0
    assert derived["new_memory_encoder_increment_ms"] == 15.0
    assert derived["estimated_full_frame_ms"] == 170.0
    assert derived["estimated_full_frame_fps"] == pytest.approx(1000.0 / 170.0)
