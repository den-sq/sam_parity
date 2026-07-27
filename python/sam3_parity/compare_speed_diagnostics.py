#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


COMPONENTS = (
    "image_encoder",
    "tracker_base",
    "previous_memory_conditioning",
    "new_memory_encoder",
)


def component_medians(result: dict[str, Any]) -> dict[str, float]:
    stages = result["stages"]
    derived = result["derived"]
    return {
        "image_encoder": float(stages["image_encoder"]["median_ms"]),
        "tracker_base": float(stages["tracker_base"]["median_ms"]),
        "previous_memory_conditioning": float(
            derived["previous_memory_conditioning_increment_ms"]
        ),
        "new_memory_encoder": float(derived["new_memory_encoder_increment_ms"]),
    }


def compare_results(
    candle: dict[str, Any],
    facebook: dict[str, Any],
) -> dict[str, Any]:
    candle_components = component_medians(candle)
    facebook_components = component_medians(facebook)
    candle_frame = float(candle["derived"]["estimated_full_frame_ms"])
    facebook_frame = float(facebook["derived"]["estimated_full_frame_ms"])
    total_gap = candle_frame - facebook_frame

    components = {}
    for name in COMPONENTS:
        gap = candle_components[name] - facebook_components[name]
        components[name] = {
            "candle_median_ms": candle_components[name],
            "facebook_median_ms": facebook_components[name],
            "candle_over_facebook": (
                candle_components[name] / facebook_components[name]
                if facebook_components[name] != 0
                else None
            ),
            "gap_ms": gap,
            "share_of_total_gap_percent": 100.0 * gap / total_gap,
        }

    ranked = sorted(COMPONENTS, key=lambda name: components[name]["gap_ms"], reverse=True)
    return {
        "schema_version": 1,
        "status": "completed",
        "candle_framework": candle["framework"],
        "facebook_framework": facebook["framework"],
        "estimated_full_frame": {
            "candle_ms": candle_frame,
            "facebook_ms": facebook_frame,
            "candle_over_facebook": candle_frame / facebook_frame,
            "gap_ms": total_gap,
        },
        "components": components,
        "ranked_gap_sources": ranked,
        "primary_gap_source": ranked[0],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare matched Candle and Facebook SAM3 speed diagnostics."
    )
    parser.add_argument("--candle", required=True, type=Path)
    parser.add_argument("--facebook", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    candle = json.loads(args.candle.read_text(encoding="utf-8"))
    facebook = json.loads(args.facebook.read_text(encoding="utf-8"))
    result = compare_results(candle, facebook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run(parse_args(argv)))


if __name__ == "__main__":
    main()
