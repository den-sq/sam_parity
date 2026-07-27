#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BACKEND_NAMES = {
    -1: "error",
    0: "math",
    1: "flash_attention",
    2: "efficient_attention",
    3: "cudnn_attention",
    4: "overrideable",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_tiff_description(description: str | bytes | None) -> dict[str, Any]:
    if not description:
        return {}
    if isinstance(description, bytes):
        description = description.decode("utf-8")
    parsed = json.loads(description)
    if not isinstance(parsed, dict):
        raise ValueError("TIFF ImageDescription must be a JSON object")
    return parsed


def annotation_samples(
    annotation_points: list[list[float]] | tuple[tuple[float, ...], ...],
    frame_count: int,
) -> list[tuple[int, list[list[float]]]]:
    grouped: dict[int, list[list[float]]] = {}
    for point in annotation_points:
        if len(point) != 3:
            raise ValueError(f"annotation point must have x, y, z, got {point!r}")
        x, y, z = (float(value) for value in point)
        frame_idx = int(round(z))
        if 0 <= frame_idx < frame_count:
            grouped.setdefault(frame_idx, []).append([x, y])
    return sorted(grouped.items())


def update_mask_digest(digest: Any, frame_idx: int, mask: Any) -> None:
    import numpy as np

    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim == 3 and binary.shape[0] == 1:
        binary = binary[0]
    binary = np.ascontiguousarray(binary)
    digest.update(int(frame_idx).to_bytes(8, byteorder="little", signed=False))
    digest.update(len(binary.shape).to_bytes(1, byteorder="little", signed=False))
    for dimension in binary.shape:
        digest.update(int(dimension).to_bytes(8, byteorder="little", signed=False))
    digest.update(binary.tobytes())


def fixture_details(path: Path) -> dict[str, Any]:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        if not tif.pages:
            raise ValueError(f"{path} has no TIFF pages")
        first = tif.pages[0]
        metadata = parse_tiff_description(first.description)
        height, width = (int(value) for value in first.shape[:2])
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "frame_count": len(tif.pages),
            "height": height,
            "width": width,
            "dtype": str(first.dtype),
            "annotation_points": metadata.get("annotation_points", []),
            "declared_shape": metadata.get("shape"),
        }


def prepare_frames(
    fixture_path: Path,
    frames_dir: Path,
    expected_fixture_sha256: str,
) -> dict[str, Any]:
    import numpy as np
    import tifffile
    from PIL import Image

    manifest_path = frames_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frame_count = int(manifest["frame_count"])
        first_frame = frames_dir / "000000.jpg"
        last_frame = frames_dir / f"{frame_count - 1:06d}.jpg"
        if (
            manifest.get("fixture_sha256") == expected_fixture_sha256
            and first_frame.is_file()
            and last_frame.is_file()
        ):
            return {**manifest, "reused": True}

    if frames_dir.exists():
        for child in frames_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                raise ValueError(
                    f"refusing to replace non-file entry in frame cache: {child}"
                )
    else:
        frames_dir.mkdir(parents=True)

    start = time.perf_counter()
    with tifffile.TiffFile(fixture_path) as tif:
        for frame_idx, page in enumerate(tif.pages):
            frame = page.asarray()
            frame = np.squeeze(frame)
            if frame.ndim != 2:
                raise ValueError(
                    f"frame {frame_idx} has shape {frame.shape}; expected grayscale"
                )
            if np.issubdtype(frame.dtype, np.integer) and frame.dtype.itemsize > 1:
                frame = np.clip(frame // 255, 0, 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            rgb = np.repeat(frame[..., None], 3, axis=2)
            Image.fromarray(rgb, mode="RGB").save(
                frames_dir / f"{frame_idx:06d}.jpg",
                format="JPEG",
                quality=90,
            )
        frame_count = len(tif.pages)

    manifest = {
        "fixture_sha256": expected_fixture_sha256,
        "frame_count": frame_count,
        "preprocessing": (
            "uint16 integer division by 255 with saturation to uint8; "
            "grayscale-to-RGB expansion; Pillow JPEG quality 90"
        ),
        "preparation_seconds": time.perf_counter() - start,
        "reused": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


@dataclass
class Phase:
    value: str = "preflight"


class TelemetrySampler:
    FIELDS = (
        "timestamp_utc",
        "elapsed_seconds",
        "phase",
        "process_pid",
        "process_rss_mib",
        "gpu_utilization_percent",
        "gpu_memory_used_mib",
        "gpu_memory_total_mib",
        "gpu_temperature_c",
        "gpu_pstate",
        "gpu_clock_mhz",
        "gpu_power_w",
        "torch_allocated_mib",
        "torch_reserved_mib",
    )

    def __init__(self, output_path: Path, phase: Phase, interval_seconds: float):
        self.output_path = output_path
        self.phase = phase
        self.interval_seconds = interval_seconds
        self.started = time.perf_counter()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="telemetry", daemon=True)
        self.samples: list[dict[str, Any]] = []

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(5.0, self.interval_seconds * 4))

    def _gpu_values(self) -> list[str]:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,"
            "pstate,clocks.gr,power.draw",
            "--format=csv,noheader,nounits",
        ]
        try:
            output = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
            return [part.strip() for part in output.splitlines()[0].split(",")]
        except (OSError, subprocess.SubprocessError, IndexError):
            return [""] * 7

    def _sample(self) -> dict[str, Any]:
        import psutil
        import torch

        gpu = self._gpu_values()
        allocated = reserved = ""
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            allocated = round(torch.cuda.memory_allocated() / 1024**2, 3)
            reserved = round(torch.cuda.memory_reserved() / 1024**2, 3)
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.perf_counter() - self.started, 3),
            "phase": self.phase.value,
            "process_pid": os.getpid(),
            "process_rss_mib": round(
                psutil.Process(os.getpid()).memory_info().rss / 1024**2, 3
            ),
            "gpu_utilization_percent": gpu[0],
            "gpu_memory_used_mib": gpu[1],
            "gpu_memory_total_mib": gpu[2],
            "gpu_temperature_c": gpu[3],
            "gpu_pstate": gpu[4],
            "gpu_clock_mhz": gpu[5],
            "gpu_power_w": gpu[6],
            "torch_allocated_mib": allocated,
            "torch_reserved_mib": reserved,
        }

    def _run(self) -> None:
        with self.output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
            writer.writeheader()
            while not self.stop_event.is_set():
                sample = self._sample()
                self.samples.append(sample)
                writer.writerow(sample)
                handle.flush()
                self.stop_event.wait(self.interval_seconds)
            sample = self._sample()
            self.samples.append(sample)
            writer.writerow(sample)

    def summary(self) -> dict[str, Any]:
        def numeric(field: str, samples=None) -> list[float]:
            values = []
            for sample in self.samples if samples is None else samples:
                try:
                    values.append(float(sample[field]))
                except (TypeError, ValueError):
                    pass
            return values

        propagation_samples = [
            sample for sample in self.samples if sample["phase"] == "propagation"
        ]
        propagation_utilization = []
        for sample in propagation_samples:
            try:
                propagation_utilization.append(
                    float(sample["gpu_utilization_percent"])
                )
            except (TypeError, ValueError):
                pass
        return {
            "sample_count": len(self.samples),
            "peak_process_rss_mib": max(numeric("process_rss_mib"), default=None),
            "peak_gpu_memory_used_mib": max(
                numeric("gpu_memory_used_mib"), default=None
            ),
            "peak_torch_allocated_mib": max(
                numeric("torch_allocated_mib"), default=None
            ),
            "peak_torch_reserved_mib": max(
                numeric("torch_reserved_mib"), default=None
            ),
            "mean_propagation_gpu_utilization_percent": (
                sum(propagation_utilization) / len(propagation_utilization)
                if propagation_utilization
                else None
            ),
            "minimum_propagation_gpu_clock_mhz": min(
                numeric("gpu_clock_mhz", propagation_samples), default=None
            ),
            "maximum_propagation_gpu_clock_mhz": max(
                numeric("gpu_clock_mhz", propagation_samples), default=None
            ),
            "peak_propagation_gpu_temperature_c": max(
                numeric("gpu_temperature_c", propagation_samples), default=None
            ),
            "observed_propagation_pstates": sorted(
                {
                    str(sample["gpu_pstate"])
                    for sample in propagation_samples
                    if sample["gpu_pstate"]
                }
            ),
            "peak_gpu_temperature_c": max(
                numeric("gpu_temperature_c"), default=None
            ),
            "minimum_gpu_clock_mhz": min(numeric("gpu_clock_mhz"), default=None),
            "maximum_gpu_clock_mhz": max(numeric("gpu_clock_mhz"), default=None),
            "observed_pstates": sorted(
                {
                    str(sample["gpu_pstate"])
                    for sample in self.samples
                    if sample["gpu_pstate"]
                }
            ),
        }


class SdpaRecorder:
    def __init__(self):
        self.records: dict[str, dict[str, Any]] = {}
        self.original = None

    def install(self) -> None:
        import torch
        import torch.nn.functional as functional

        self.original = functional.scaled_dot_product_attention

        def wrapped(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            *,
            scale=None,
            enable_gqa=False,
        ):
            signature = json.dumps(
                {
                    "query_shape": list(query.shape),
                    "key_shape": list(key.shape),
                    "value_shape": list(value.shape),
                    "dtype": str(query.dtype),
                    "device": str(query.device),
                    "has_mask": attn_mask is not None,
                    "dropout_p": float(dropout_p),
                    "is_causal": bool(is_causal),
                    "scale": scale,
                    "enable_gqa": bool(enable_gqa),
                },
                sort_keys=True,
            )
            record = self.records.get(signature)
            if record is None:
                backend_value = int(
                    torch.ops.aten._fused_sdp_choice.default(
                        query,
                        key,
                        value,
                        attn_mask,
                        float(dropout_p),
                        bool(is_causal),
                        scale=scale,
                        enable_gqa=bool(enable_gqa),
                    )
                )
                record = {
                    **json.loads(signature),
                    "selected_backend": BACKEND_NAMES.get(
                        backend_value, f"unknown_{backend_value}"
                    ),
                    "call_count": 0,
                }
                self.records[signature] = record
            record["call_count"] += 1
            return self.original(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )

        functional.scaled_dot_product_attention = wrapped

    def uninstall(self) -> None:
        if self.original is None:
            return
        import torch.nn.functional as functional

        functional.scaled_dot_product_attention = self.original

    def summary(self) -> dict[str, Any]:
        records = list(self.records.values())
        return {
            "selected_backends": sorted(
                {record["selected_backend"] for record in records}
            ),
            "signatures": records,
        }


def git_revision(path: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
            "remote": git("remote", "get-url", "origin"),
        }
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "remote": None}


def environment_record(sam3_path: Path) -> dict[str, Any]:
    import torch

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": properties.name,
        "gpu_total_memory_mib": properties.total_memory // 1024**2,
        "gpu_compute_capability": f"{properties.major}.{properties.minor}",
        "driver_version": subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "sam3": git_revision(sam3_path),
        "sdpa_configuration": {
            "flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "flash_available_in_build": torch.backends.cuda.is_flash_attention_available(),
            "memory_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_enabled": torch.backends.cuda.math_sdp_enabled(),
            "cudnn_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
        },
        "torch_compile": {
            "used": False,
            "disposition": "disabled explicitly for the reference benchmark",
        },
    }


def summarize_state(inference_state: dict[str, Any]) -> dict[str, Any]:
    output_dict = inference_state.get("output_dict", {})
    return {
        "offload_video_to_cpu": bool(
            inference_state.get("offload_video_to_cpu", False)
        ),
        "offload_state_to_cpu": bool(
            inference_state.get("offload_state_to_cpu", False)
        ),
        "storage_device": str(inference_state.get("storage_device")),
        "conditioning_outputs": len(output_dict.get("cond_frame_outputs", {})),
        "non_conditioning_outputs": len(
            output_dict.get("non_cond_frame_outputs", {})
        ),
        "frames_already_tracked": len(
            inference_state.get("frames_already_tracked", {})
        ),
        "cached_features": len(inference_state.get("cached_features", {})),
        "retention": (
            "unbounded per-frame tracker outputs in output_dict; streamed masks "
            "are hashed and discarded by the harness"
        ),
    }


def peak_torch_record() -> dict[str, Any]:
    import torch

    return {
        "max_memory_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "max_memory_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "memory_allocated_mib": torch.cuda.memory_allocated() / 1024**2,
        "memory_reserved_mib": torch.cuda.memory_reserved() / 1024**2,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the official Facebook SAM3 tracker on a private TIFF fixture "
            "with synchronized timing, direct-process RSS, GPU telemetry, and "
            "streamed output hashing."
        )
    )
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--sam3-revision",
        default=None,
        help="Exact upstream SAM3 revision, for runtimes without git installed.",
    )
    parser.add_argument(
        "--sam3-source-url",
        default="https://github.com/facebookresearch/sam3.git",
    )
    parser.add_argument(
        "--dtype",
        choices=("default", "f32"),
        default="default",
        help="Upstream default BF16 autocast, or forced F32.",
    )
    parser.add_argument(
        "--sdpa-backend",
        choices=("auto", "math"),
        default="auto",
        help=(
            "Use PyTorch's automatic SDPA selection, or force math SDPA for a "
            "controlled attention-backend ablation."
        ),
    )
    parser.add_argument(
        "--telemetry-interval-seconds", type=float, default=1.0
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare the matched JPEG frame cache without loading the model.",
    )
    parser.add_argument(
        "--sync-frame-loading",
        action="store_true",
        help=(
            "Load every resized frame synchronously before inference. The default "
            "uses upstream async loading, which defers frame loading but may still "
            "cache resized frames in host memory as propagation advances."
        ),
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    fixture_path = args.fixture.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    frames_dir = args.frames_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fixture = fixture_details(fixture_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    preparation = prepare_frames(fixture_path, frames_dir, fixture["sha256"])
    invocation = " ".join(shlex.quote(value) for value in sys.argv)
    (output_dir / "invocation.txt").write_text(invocation + "\n", encoding="utf-8")
    if args.prepare_only:
        (output_dir / "preparation.json").write_text(
            json.dumps(
                {"fixture": fixture, "frame_preparation": preparation}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("reference benchmark requires CUDA")

    from sam3_parity.upstream import import_video_predictor_builder

    build_sam3_predictor = import_video_predictor_builder()

    sam3_path = Path(__import__("sam3").__file__).resolve().parents[1]
    environment = environment_record(sam3_path)
    if args.sam3_revision is not None:
        environment["sam3"]["commit"] = args.sam3_revision
    environment["sam3"]["source_url"] = args.sam3_source_url
    if args.sdpa_backend == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    environment["sdpa_policy"] = {
        "requested": args.sdpa_backend,
        "effective_configuration": {
            "flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_enabled": torch.backends.cuda.math_sdp_enabled(),
            "cudnn_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
        },
    }
    environment.update(
        {
            "checkpoint_sha256": checkpoint_sha256,
            "fixture_sha256": fixture["sha256"],
            "invocation": invocation,
        }
    )
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )

    phase = Phase()
    telemetry = TelemetrySampler(
        output_dir / "telemetry.csv",
        phase,
        args.telemetry_interval_seconds,
    )
    telemetry.start()
    sdpa = SdpaRecorder()
    result: dict[str, Any] = {
        "status": "running",
        "dtype_arm": args.dtype,
        "fixture": {
            key: value for key, value in fixture.items() if key != "annotation_points"
        },
        "annotation_count": len(fixture["annotation_points"]),
        "checkpoint_sha256": checkpoint_sha256,
        "frame_preparation": preparation,
        "async_loading_frames": not args.sync_frame_loading,
        "offload_video_to_cpu": True,
        "offload_state_to_cpu": False,
        "compile": False,
        "sdpa_policy": args.sdpa_backend,
        "process_pid": os.getpid(),
    }
    timings: dict[str, float] = {}
    output_digest = hashlib.sha256()
    frames_hashed = 0
    last_frame_idx = None
    model = tracker = inference_state = None
    exit_code = 0
    e2e_start = time.perf_counter()

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        phase.value = "model_startup"
        start = time.perf_counter()
        predictor = build_sam3_predictor(
            version="sam3",
            checkpoint_path=str(checkpoint_path),
            compile=False,
            async_loading_frames=False,
            apply_temporal_disambiguation=False,
        )
        torch.cuda.synchronize()
        timings["model_startup_seconds"] = time.perf_counter() - start
        model = predictor.model
        tracker = model.tracker
        if getattr(tracker, "backbone", None) is None:
            tracker.backbone = model.detector.backbone

        default_autocast_enabled = torch.is_autocast_enabled("cuda")
        default_autocast_dtype = str(torch.get_autocast_dtype("cuda"))
        if args.dtype == "f32":
            context = getattr(tracker, "bf16_context", None)
            if context is not None:
                context.__exit__(None, None, None)
            torch.set_autocast_enabled("cuda", False)
        result["compute"] = {
            "upstream_default_autocast_enabled": default_autocast_enabled,
            "upstream_default_autocast_dtype": default_autocast_dtype,
            "effective_autocast_enabled": torch.is_autocast_enabled("cuda"),
            "effective_autocast_dtype": str(torch.get_autocast_dtype("cuda")),
            "model_parameter_dtype": str(next(model.parameters()).dtype),
        }
        sdpa.install()

        phase.value = "session_setup"
        start = time.perf_counter()
        inference_state = tracker.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
            async_loading_frames=not args.sync_frame_loading,
        )
        torch.cuda.synchronize()
        timings["session_setup_seconds"] = time.perf_counter() - start

        samples = annotation_samples(
            fixture["annotation_points"], fixture["frame_count"]
        )
        if not samples:
            samples = [(0, [[fixture["width"] / 2, fixture["height"] / 2]])]
            result["annotation_fallback"] = "center point on frame 0"
        result["annotation_frame_count"] = len(samples)

        phase.value = "prompt_setup"
        start = time.perf_counter()
        with torch.inference_mode():
            for frame_idx, points in samples:
                tracker.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=1,
                    points=np.asarray(points, dtype=np.float32),
                    labels=np.ones(len(points), dtype=np.int32),
                )
        torch.cuda.synchronize()
        timings["prompt_setup_seconds"] = time.perf_counter() - start

        phase.value = "propagation"
        start = time.perf_counter()
        with torch.inference_mode():
            for (
                frame_idx,
                _obj_ids,
                _low_res_masks,
                video_res_masks,
                *_rest,
            ) in tracker.propagate_in_video(
                inference_state,
                start_frame_idx=None,
                max_frame_num_to_track=None,
                reverse=False,
                propagate_preflight=True,
            ):
                last_frame_idx = int(frame_idx)
                mask = (video_res_masks[0] > 0.0).detach().cpu().numpy()
                update_mask_digest(output_digest, last_frame_idx, mask)
                frames_hashed += 1
        torch.cuda.synchronize()
        timings["propagation_seconds"] = time.perf_counter() - start
        result["status"] = "completed"
    except torch.OutOfMemoryError as error:
        exit_code = 3
        timings[f"{phase.value}_seconds_until_failure"] = (
            time.perf_counter() - start
        )
        result.update(
            {
                "status": "oom",
                "failure_phase": phase.value,
                "failure_type": type(error).__name__,
                "failure_message": str(error),
                "memory_summary": torch.cuda.memory_summary(),
            }
        )
    except Exception as error:
        exit_code = 2
        timings[f"{phase.value}_seconds_until_failure"] = (
            time.perf_counter() - start
        )
        result.update(
            {
                "status": "error",
                "failure_phase": phase.value,
                "failure_type": type(error).__name__,
                "failure_message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        sdpa.uninstall()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        timings["measured_end_to_end_seconds"] = time.perf_counter() - e2e_start
        result.update(
            {
                "timings": timings,
                "frames_hashed": frames_hashed,
                "last_frame_idx": last_frame_idx,
                "output_sha256": output_digest.hexdigest(),
                "sdpa": sdpa.summary(),
                "torch_memory": (
                    peak_torch_record()
                    if torch.cuda.is_available() and torch.cuda.is_initialized()
                    else None
                ),
            }
        )
        if inference_state is not None:
            result["retained_state"] = summarize_state(inference_state)
        propagation_seconds = timings.get(
            "propagation_seconds",
            timings.get("propagation_seconds_until_failure"),
        )
        if propagation_seconds:
            result["propagation_fps"] = frames_hashed / propagation_seconds
        e2e_seconds = timings["measured_end_to_end_seconds"]
        result["end_to_end_fps"] = (
            frames_hashed / e2e_seconds if e2e_seconds else None
        )
        phase.value = "finalize"
        telemetry.stop()
        result["telemetry"] = telemetry.summary()
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
    return exit_code


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run(parse_args(argv)))


if __name__ == "__main__":
    main()
