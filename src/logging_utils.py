"""Logging + run provenance.

Two jobs:

1. A logger that writes to stdout *and* to logs/<run_id>.log, so a Colab
   session that disconnects still leaves a trace on disk.

2. `provenance()` - the metadata block that goes into every cached artefact's
   JSON sidecar: model id, precision config, dtype,
   seed, layer count, git commit hash, timestamp. Without the commit hash a
   cached .npy is unattributable six weeks later, which in a project whose
   whole output is "these two arrays differ" is fatal.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CONFIGURED = False


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def setup_logging(
    name: str = "quantprobe",
    log_dir: Path | None = None,
    level: int = logging.INFO,
    run_name: str | None = None,
) -> logging.Logger:
    """Configure the root project logger. Idempotent."""
    global _CONFIGURED

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if _CONFIGURED:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{run_name or run_id()}.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.info("logging to %s", log_path)

    logger.propagate = False
    _CONFIGURED = True
    return logger


def git_commit() -> str:
    """Short commit hash, or an explicit marker. Never silently empty."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"

    if result.returncode != 0:
        return "unavailable: not a git repo or no commits yet"
    return result.stdout.strip()


def git_dirty() -> bool | str:
    """True if the working tree has uncommitted changes.

    A cached array produced from a dirty tree is not reproducible from the
    commit hash alone, so this flag is recorded alongside it.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"

    if result.returncode != 0:
        return "unavailable"
    return bool(result.stdout.strip())


def env_versions() -> dict[str, str]:
    """Versions of every library whose behaviour could move a number."""
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for module_name in (
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "numpy",
        "sklearn",
    ):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[module_name] = f"not importable: {type(exc).__name__}"
    return versions


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {"cpu": platform.processor() or "unknown"}
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_memory_gb"] = round(props.total_memory / 1024**3, 2)
            info["cuda_version"] = torch.version.cuda
    except Exception as exc:
        info["cuda_available"] = f"torch unavailable: {type(exc).__name__}"
    return info


def provenance(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """The full metadata block for a cached artefact's sidecar."""
    block: dict[str, Any] = {
        "timestamp_utc": utc_stamp(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "versions": env_versions(),
        "hardware": hardware_info(),
    }
    if extra:
        block.update(extra)
    return block


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write a JSON sidecar, creating parent dirs. Sorted keys for clean diffs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    return path
