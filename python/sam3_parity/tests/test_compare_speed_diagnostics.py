from sam3_parity.compare_speed_diagnostics import compare_results


def result(framework: str, image: float, base: float, history: float, full: float):
    return {
        "framework": framework,
        "stages": {
            "image_encoder": {"median_ms": image},
            "tracker_base": {"median_ms": base},
        },
        "derived": {
            "previous_memory_conditioning_increment_ms": history - base,
            "new_memory_encoder_increment_ms": full - history,
            "estimated_full_frame_ms": image + full,
        },
    }


def test_compare_results_attributes_full_gap_to_components():
    candle = result("candle", 300.0, 30.0, 50.0, 80.0)
    facebook = result("facebook-pytorch", 100.0, 10.0, 25.0, 40.0)

    comparison = compare_results(candle, facebook)

    assert comparison["estimated_full_frame"]["gap_ms"] == 240.0
    assert comparison["primary_gap_source"] == "image_encoder"
    assert comparison["components"]["image_encoder"]["gap_ms"] == 200.0
    assert comparison["components"]["tracker_base"]["gap_ms"] == 20.0
    assert (
        sum(
            component["gap_ms"]
            for component in comparison["components"].values()
        )
        == 240.0
    )
