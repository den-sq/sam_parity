from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from sam3_parity.paths import bundle_root as default_bundle_root
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sam3_parity.paths import bundle_root as default_bundle_root


INFORMATIONAL_PATH_KEYS = {
    "bpe_path",
    "checkpoint_path",
    "module_file",
    "package_dir",
    "replay_script_path",
    "source_path",
    "tokenizer_path",
}


@dataclass
class BundleValidation:
    bundle: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate SAM3 reference bundle layout and portable path metadata."
    )
    parser.add_argument(
        "bundles",
        nargs="*",
        help="Bundle directory names to validate. Defaults to every directory under the root.",
    )
    parser.add_argument(
        "--bundle-root",
        default=str(default_bundle_root()),
        help="Reference bundle root. Defaults to SAM3_PARITY_BUNDLE_ROOT or tests/reference-bundles.",
    )
    parser.add_argument(
        "--strict-informational-paths",
        action="store_true",
        help="Treat absolute checkpoint/source/tokenizer metadata paths as errors instead of warnings.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON validation report.",
    )
    return parser.parse_args()


def load_json(path: Path, result: BundleValidation) -> Any | None:
    if not path.exists():
        result.errors.append(f"missing required file: {path.name}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        result.errors.append(f"invalid JSON in {path.name}: {exc}")
        return None


def is_absolute_path_like(value: str) -> bool:
    if value.startswith(("http://", "https://")):
        return False
    return Path(value).expanduser().is_absolute()


def scan_absolute_paths(
    value: Any,
    result: BundleValidation,
    *,
    location: str,
    strict_informational_paths: bool,
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            scan_absolute_paths(
                child,
                result,
                location=f"{location}.{key}",
                strict_informational_paths=strict_informational_paths,
            )
        return
    if isinstance(value, list):
        for idx, child in enumerate(value):
            scan_absolute_paths(
                child,
                result,
                location=f"{location}[{idx}]",
                strict_informational_paths=strict_informational_paths,
            )
        return
    if not isinstance(value, str) or not is_absolute_path_like(value):
        return

    key = location.rsplit(".", 1)[-1]
    message = f"absolute path metadata at {location}: {value}"
    if key in INFORMATIONAL_PATH_KEYS and not strict_informational_paths:
        result.warnings.append(message)
    else:
        result.errors.append(message)


def validate_video_bundle(
    bundle_dir: Path,
    metadata: dict[str, Any],
    result: BundleValidation,
    *,
    strict_informational_paths: bool,
) -> None:
    results_path = metadata.get("results_path") or "video_results.json"
    frames_dir = metadata.get("frames_dir") or "frames"
    masks_dir = metadata.get("masks_dir") or "masks"
    masked_frames_dir = metadata.get("masked_frames_dir") or "masked_frames"

    for relative in (results_path, frames_dir, masks_dir, masked_frames_dir):
        path = bundle_dir / relative
        if not path.exists():
            result.errors.append(f"missing video artifact: {relative}")

    results = load_json(bundle_dir / results_path, result)
    if results is not None:
        scan_absolute_paths(
            results,
            result,
            location="video_results",
            strict_informational_paths=strict_informational_paths,
        )


def validate_non_video_bundle(bundle_dir: Path, result: BundleValidation) -> None:
    tensor_path = bundle_dir / "reference.safetensors"
    if not tensor_path.exists():
        result.errors.append("missing required file: reference.safetensors")


def validate_bundle(
    bundle_dir: Path,
    *,
    strict_informational_paths: bool = False,
) -> BundleValidation:
    result = BundleValidation(bundle=bundle_dir.name)
    if not bundle_dir.exists():
        result.errors.append(f"bundle directory does not exist: {bundle_dir}")
        return result
    if not bundle_dir.is_dir():
        result.errors.append(f"bundle path is not a directory: {bundle_dir}")
        return result

    metadata = load_json(bundle_dir / "reference.json", result)
    if not isinstance(metadata, dict):
        return result

    scan_absolute_paths(
        metadata,
        result,
        location="reference",
        strict_informational_paths=strict_informational_paths,
    )

    mode = str(metadata.get("mode") or "")
    if mode.startswith("video_"):
        validate_video_bundle(
            bundle_dir,
            metadata,
            result,
            strict_informational_paths=strict_informational_paths,
        )
    else:
        validate_non_video_bundle(bundle_dir, result)
        if mode == "interactive_reference" and not metadata.get("steps"):
            result.errors.append("interactive bundle has no steps")

    return result


def discover_bundle_dirs(root: Path, names: list[str]) -> list[Path]:
    if names:
        return [root / name for name in names]
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def main() -> int:
    args = parse_args()
    root = Path(args.bundle_root).expanduser().resolve(strict=False)
    bundle_dirs = discover_bundle_dirs(root, args.bundles)
    if not bundle_dirs:
        report = {
            "bundle_root": str(root),
            "ok": False,
            "errors": [f"no bundles found under {root}"],
            "bundles": [],
        }
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(report["errors"][0])
        return 1

    results = [
        validate_bundle(
            bundle_dir,
            strict_informational_paths=args.strict_informational_paths,
        )
        for bundle_dir in bundle_dirs
    ]
    ok = all(result.ok for result in results)
    report = {
        "bundle_root": str(root),
        "ok": ok,
        "bundles": [
            {
                "name": result.bundle,
                "ok": result.ok,
                "errors": result.errors,
                "warnings": result.warnings,
            }
            for result in results
        ],
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for result in results:
            status = "ok" if result.ok else "failed"
            print(f"{result.bundle}: {status}")
            for warning in result.warnings:
                print(f"  warning: {warning}")
            for error in result.errors:
                print(f"  error: {error}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
