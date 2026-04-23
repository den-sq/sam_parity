from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3_parity.paths import bundle_root, data_root, sam3_checkpoint_path
from sam3_parity.upstream import installed_sam3_info


def _torch_status() -> str:
    try:
        import torch
    except Exception as exc:
        return f"missing ({exc})"
    return f"{torch.__version__}"


def _path_status(path: Path | None, env_name: str) -> str:
    raw = os.environ.get(env_name)
    if path is None:
        return f"{env_name}=unset"
    exists = "exists" if path.exists() else "missing"
    if raw is None:
        return f"{env_name}=default:{path} ({exists})"
    return f"{env_name}={path} ({exists})"


def _full_parity_requested(config: pytest.Config) -> bool:
    markexpr = (config.option.markexpr or "").strip()
    if "full_parity" in markexpr:
        return True
    paths = [str(arg) for arg in config.args]
    return any(
        fragment in path
        for path in paths
        for fragment in (
            "python_debug/sam3_debug/tests",
        )
    )


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    if not _full_parity_requested(config):
        return

    skipped = terminalreporter.stats.get("skipped", [])
    if not skipped:
        return

    terminalreporter.section("full_parity environment")
    terminalreporter.write_line(f"python={sys.executable}")
    terminalreporter.write_line(f"torch={_torch_status()}")
    sam3_info = installed_sam3_info()
    terminalreporter.write_line(
        "sam3="
        + (
            f"{sam3_info['distribution_name'] or 'installed'}"
            f"@{sam3_info['distribution_version'] or 'unknown'} "
            f"({sam3_info['package_dir'] or 'not importable'})"
            if sam3_info["package_dir"] is not None
            else "missing"
        )
    )
    terminalreporter.write_line(
        f"SAM3_UPSTREAM_URL={sam3_info['requested_source_url']}"
    )
    terminalreporter.write_line(
        f"SAM3_UPSTREAM_REF={sam3_info['requested_source_ref'] or 'unset'}"
    )
    terminalreporter.write_line(_path_status(sam3_checkpoint_path(), "SAM3_CHECKPOINT"))
    terminalreporter.write_line(
        f"SAM3_PARITY_BUNDLE_ROOT={bundle_root()} ({'exists' if bundle_root().exists() else 'missing'})"
    )
    terminalreporter.write_line(
        f"SAM3_PARITY_DATA_ROOT={data_root()} ({'exists' if data_root().exists() else 'missing'})"
    )
