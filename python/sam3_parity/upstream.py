from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path


DEFAULT_SAM3_UPSTREAM_URL = "https://github.com/facebookresearch/sam3.git"
SAM3_DEPENDENCY_HINTS = {
    "einops": "einops",
    "ftfy": "ftfy==6.1.1",
    "huggingface_hub": "huggingface_hub",
    "iopath": "iopath>=0.1.10",
    "pkg_resources": "setuptools<81",
    "pycocotools": "pycocotools",
    "regex": "regex",
    "timm": "timm>=1.0.17",
    "tqdm": "tqdm",
}


def sam3_upstream_url() -> str | None:
    raw = os.environ.get("SAM3_UPSTREAM_URL")
    return raw.strip() if raw and raw.strip() else None


def sam3_upstream_ref() -> str | None:
    raw = os.environ.get("SAM3_UPSTREAM_REF")
    return raw.strip() if raw and raw.strip() else None


def configured_sam3_source(
    explicit_url: str | None = None,
    explicit_ref: str | None = None,
) -> tuple[str, str | None]:
    url = explicit_url or sam3_upstream_url() or DEFAULT_SAM3_UPSTREAM_URL
    ref = explicit_ref or sam3_upstream_ref()
    return url, ref


def git_install_target(url: str, ref: str | None = None) -> str:
    target = url if url.startswith("git+") else f"git+{url}"
    if ref:
        target = f"{target}@{ref}"
    return target


def sam3_install_command(
    explicit_url: str | None = None,
    explicit_ref: str | None = None,
    python_executable: str | None = None,
) -> str:
    url, ref = configured_sam3_source(explicit_url, explicit_ref)
    python_bin = python_executable or sys.executable
    return f"{python_bin} -m pip install '{git_install_target(url, ref)}'"


def describe_sam3_import_error(module_name: str, exc: ModuleNotFoundError) -> str:
    missing_name = exc.name or "<unknown>"
    if missing_name == "sam3":
        return (
            f"installed sam3 package is required to import {module_name}. "
            f"Install it, e.g. {sam3_install_command()}"
        )
    hint = SAM3_DEPENDENCY_HINTS.get(missing_name)
    install_hint = f" (install {hint})" if hint else ""
    return (
        f"installed sam3 package could not import {module_name}: missing module "
        f"'{missing_name}'{install_hint}"
    )


def import_sam3_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise FileNotFoundError(describe_sam3_import_error(module_name, exc)) from exc


def import_sam3_symbol(module_name: str, symbol_name: str):
    module = import_sam3_module(module_name)
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise FileNotFoundError(
            f"installed sam3 package imported {module_name}, but it does not expose "
            f"'{symbol_name}'"
        ) from exc


def import_video_predictor_builder():
    module = import_sam3_module("sam3.model_builder")
    legacy_builder = getattr(module, "build_sam3_predictor", None)
    if callable(legacy_builder):
        return legacy_builder

    video_builder = getattr(module, "build_sam3_video_predictor", None)
    if callable(video_builder):
        def compat_builder(*args, **kwargs):
            # Current upstream exposes the video predictor builder directly and
            # does not accept legacy image/video selector flags.
            kwargs.pop("version", None)
            kwargs.pop("compile", None)
            return video_builder(*args, **kwargs)

        return compat_builder

    raise FileNotFoundError(
        "installed sam3 package imported sam3.model_builder, but it does not expose "
        "'build_sam3_predictor' or 'build_sam3_video_predictor'"
    )


def sam3_package_dir() -> Path:
    module = import_sam3_module("sam3")
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise FileNotFoundError("installed sam3 package does not expose a module file")
    return Path(module_file).resolve().parent


def resolve_sam3_asset_path(relative_path: str) -> Path:
    package_dir = sam3_package_dir()
    candidates = [
        package_dir / relative_path,
        package_dir.parent / relative_path,
    ]
    for asset_path in candidates:
        if asset_path.exists():
            return asset_path
    raise FileNotFoundError(
        "installed sam3 package is missing bundled asset; checked "
        + ", ".join(str(path) for path in candidates)
    )


def resolve_default_bpe_path(explicit_path: str | None = None) -> Path:
    if explicit_path is not None:
        return Path(explicit_path).expanduser().resolve(strict=False)
    return resolve_sam3_asset_path("assets/bpe_simple_vocab_16e6.txt.gz")


def installed_sam3_info(
    explicit_url: str | None = None,
    explicit_ref: str | None = None,
) -> dict[str, str | None]:
    requested_url, requested_ref = configured_sam3_source(explicit_url, explicit_ref)
    info: dict[str, str | None] = {
        "requested_source_url": requested_url,
        "requested_source_ref": requested_ref,
        "module_file": None,
        "package_dir": None,
        "distribution_name": None,
        "distribution_version": None,
    }
    try:
        package_dir = sam3_package_dir()
    except FileNotFoundError:
        return info

    info["package_dir"] = str(package_dir)
    info["module_file"] = str(package_dir / "__init__.py")

    distribution_name = None
    try:
        distribution_name = "sam3"
        info["distribution_version"] = importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        candidates = importlib.metadata.packages_distributions().get("sam3", [])
        if candidates:
            distribution_name = candidates[0]
            try:
                info["distribution_version"] = importlib.metadata.version(distribution_name)
            except importlib.metadata.PackageNotFoundError:
                info["distribution_version"] = None
    info["distribution_name"] = distribution_name
    return info


def main() -> None:
    print(json.dumps(installed_sam3_info(), indent=2))


if __name__ == "__main__":
    main()
