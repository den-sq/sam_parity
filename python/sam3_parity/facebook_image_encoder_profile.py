#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture exactly one warmed-up official Facebook SAM3 image encoder "
            "with CUDA profiler APIs."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--target-frame", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    import torch

    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.target_frame < 0:
        raise ValueError("--target-frame must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("Facebook image-encoder profiling requires CUDA")

    checkpoint = args.checkpoint.expanduser().resolve()
    frames_dir = args.frames_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
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
    model = predictor.model
    tracker = model.tracker
    if getattr(tracker, "backbone", None) is None:
        tracker.backbone = model.detector.backbone

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
        image = (
            inference_state["images"][args.target_frame]
            .cuda()
            .float()
            .unsqueeze(0)
        )
        torch.cuda.synchronize()
        print(
            f"model and input ready in {time.perf_counter() - startup_started:.3f}s; "
            f"running {args.warmup} warmup iteration(s)",
            flush=True,
        )
        for _ in range(args.warmup):
            output = tracker.forward_image(image)
            torch.cuda.synchronize()
            del output

        torch.cuda.synchronize()
        print(
            "starting CUDA profiler capture for one image encoder iteration",
            flush=True,
        )
        with torch.cuda.profiler.profile():
            started = time.perf_counter()
            output = tracker.forward_image(image)
            torch.cuda.synchronize()
            captured_ms = (time.perf_counter() - started) * 1000.0
        del output

    print(
        json.dumps(
            {
                "status": "completed",
                "captured": "image_encoder",
                "framework": "facebook-pytorch",
                "dtype": "torch.float32",
                "autocast_enabled": torch.is_autocast_enabled("cuda"),
                "warmup": args.warmup,
                "captured_ms": captured_ms,
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run(parse_args(argv)))


if __name__ == "__main__":
    main()
