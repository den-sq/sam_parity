from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_env_path(name: str, default: Path | str | None = None) -> Path | None:
    raw = os.environ.get(name)
    if raw:
        path = Path(raw).expanduser()
    elif default is not None:
        path = Path(default).expanduser()
    else:
        return None
    if not path.is_absolute():
        path = repo_root() / path
    return path.resolve(strict=False)


def bundle_root() -> Path:
    return resolve_env_path(
        "SAM3_PARITY_BUNDLE_ROOT", repo_root() / "tests" / "reference-bundles"
    )


def data_root() -> Path:
    return resolve_env_path("SAM3_PARITY_DATA_ROOT", repo_root() / "tests" / "data")


def sam3_checkpoint_path() -> Path | None:
    return resolve_env_path("SAM3_CHECKPOINT")


def sam3_tokenizer_path() -> Path | None:
    return resolve_env_path("SAM3_TOKENIZER")


def resolve_repo_file(path: Path | str, expected_file: str) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = repo_root() / path
    path = path.resolve(strict=False)
    return path / expected_file if path.is_dir() else path


def require_path(path: Path | None, description: str, env_name: str | None = None) -> Path:
    if path is None:
        suffix = f" or set {env_name}" if env_name else ""
        raise FileNotFoundError(f"{description} is required{suffix}")
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def path_for_metadata(path: Path | str | None, base_dir: Path) -> str | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve(strict=False)
    try:
        return resolved.relative_to(base_dir.resolve(strict=False)).as_posix()
    except ValueError:
        return str(resolved)


def bundled_image_path(input_path: Path | str, output_dir: Path) -> Path:
    _ = input_path
    return output_dir / "input_image.png"
