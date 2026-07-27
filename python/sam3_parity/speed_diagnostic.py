#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from sam3_parity.reference_benchmark import (
    environment_record,
    sha256_file,
)


STAGE_DESCRIPTIONS = {
    "image_encoder": "image backbone and neck only",
    "tracker_base": "SAM tracker head without prior history or new-memory encoding",
    "tracker_with_history": "tracker head with one prior conditioning state, without new-memory encoding",
    "tracker_full": "tracker head with one prior conditioning state and new-memory encoding",
}


def summarize_samples(samples_ms: list[float]) -> dict[str, Any]:
    if not samples_ms:
        raise ValueError("at least one timing sample is required")
    ordered = sorted(samples_ms)
    return {
        "samples_ms": samples_ms,
        "sample_count": len(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
    }


def derived_stage_costs(stages: dict[str, dict[str, Any]]) -> dict[str, float]:
    medians = {name: float(stats["median_ms"]) for name, stats in stages.items()}
    return {
        "previous_memory_conditioning_increment_ms": (
            medians["tracker_with_history"] - medians["tracker_base"]
        ),
        "new_memory_encoder_increment_ms": (
            medians["tracker_full"] - medians["tracker_with_history"]
        ),
        "estimated_full_frame_ms": medians["image_encoder"] + medians["tracker_full"],
        "estimated_full_frame_fps": 1000.0
        / (medians["image_encoder"] + medians["tracker_full"]),
    }


def measure_cuda_stage(
    operation: Callable[[], Any],
    *,
    warmup: int,
    samples: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    import torch

    for _ in range(warmup):
        value = operation()
        torch.cuda.synchronize()
        del value

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    samples_ms = []
    for _ in range(samples):
        torch.cuda.synchronize()
        started = time.perf_counter()
        value = operation()
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - started) * 1000.0)
        del value
    memory = {
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    }
    return summarize_samples(samples_ms), memory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Partition one steady-state official SAM3 frame into image encoder, "
            "base tracker, previous-memory conditioning, and new-memory encoder costs."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-frame", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--sam3-revision")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.target_frame <= 0:
        raise ValueError("--target-frame must be greater than zero")
    if not torch.cuda.is_available():
        raise RuntimeError("speed diagnostic requires CUDA")

    checkpoint = args.checkpoint.expanduser().resolve()
    frames_dir = args.frames_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if args.target_frame >= len(frame_paths):
        raise ValueError(
            f"target frame {args.target_frame} is out of bounds for "
            f"{len(frame_paths)} JPEG frames"
        )

    from sam3_parity.upstream import import_video_predictor_builder

    build_sam3_predictor = import_video_predictor_builder()
    startup_started = time.perf_counter()
    predictor = build_sam3_predictor(
        version="sam3",
        checkpoint_path=str(checkpoint),
        compile=False,
        async_loading_frames=False,
        apply_temporal_disambiguation=False,
    )
    torch.cuda.synchronize()
    model_startup_seconds = time.perf_counter() - startup_started
    model = predictor.model
    tracker = model.tracker
    if getattr(tracker, "backbone", None) is None:
        tracker.backbone = model.detector.backbone

    upstream_autocast = {
        "enabled": torch.is_autocast_enabled("cuda"),
        "dtype": str(torch.get_autocast_dtype("cuda")),
    }
    context = getattr(tracker, "bf16_context", None)
    if context is not None:
        context.__exit__(None, None, None)
    torch.set_autocast_enabled("cuda", False)

    with torch.inference_mode():
        inference_state = tracker.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
            async_loading_frames=False,
        )
        center = np.asarray(
            [[inference_state["video_width"] / 2, inference_state["video_height"] / 2]],
            dtype=np.float32,
        )
        tracker.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=center,
            labels=np.ones(1, dtype=np.int32),
        )
        tracker.propagate_in_video_preflight(
            inference_state,
            run_mem_encoder=True,
        )
        (
            target_image,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = tracker._get_image_feature(inference_state, args.target_frame, 1)

    empty_output_dict = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }
    history_output_dict = inference_state["output_dict"]
    common_track_args = {
        "frame_idx": args.target_frame,
        "is_init_cond_frame": False,
        "current_vision_feats": current_vision_feats,
        "current_vision_pos_embeds": current_vision_pos_embeds,
        "feat_sizes": feat_sizes,
        "image": target_image,
        "point_inputs": None,
        "mask_inputs": None,
        "num_frames": inference_state["num_frames"],
        "track_in_reverse": False,
        "prev_sam_mask_logits": None,
    }

    operations = {
        "image_encoder": lambda: tracker.forward_image(target_image),
        "tracker_base": lambda: tracker.track_step(
            **common_track_args,
            output_dict=empty_output_dict,
            run_mem_encoder=False,
            use_prev_mem_frame=False,
        ),
        "tracker_with_history": lambda: tracker.track_step(
            **common_track_args,
            output_dict=history_output_dict,
            run_mem_encoder=False,
            use_prev_mem_frame=True,
        ),
        "tracker_full": lambda: tracker.track_step(
            **common_track_args,
            output_dict=history_output_dict,
            run_mem_encoder=True,
            use_prev_mem_frame=True,
        ),
    }

    stages: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for name, operation in operations.items():
            timing, memory = measure_cuda_stage(
                operation,
                warmup=args.warmup,
                samples=args.samples,
            )
            stages[name] = {
                "description": STAGE_DESCRIPTIONS[name],
                **timing,
                **memory,
            }

    sam3_path = Path(__import__("sam3").__file__).resolve().parents[1]
    environment = environment_record(sam3_path)
    if args.sam3_revision:
        environment["sam3"]["commit"] = args.sam3_revision
    result = {
        "schema_version": 1,
        "framework": "facebook-pytorch",
        "status": "completed",
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
        },
        "frames_dir": str(frames_dir),
        "target_frame": args.target_frame,
        "warmup": args.warmup,
        "samples": args.samples,
        "model_startup_seconds": model_startup_seconds,
        "compute": {
            "dtype": "torch.float32",
            "upstream_autocast": upstream_autocast,
            "effective_autocast_enabled": torch.is_autocast_enabled("cuda"),
            "compile": False,
        },
        "environment": environment,
        "stages": stages,
        "derived": derived_stage_costs(stages),
        "interpretation": {
            "tracker_base": "SAM tracker head without previous-frame history",
            "previous_memory_conditioning_increment": (
                "tracker_with_history median minus tracker_base median"
            ),
            "new_memory_encoder_increment": (
                "tracker_full median minus tracker_with_history median"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run(parse_args(argv)))


if __name__ == "__main__":
    main()
