"""Central path resolution.

No hardcoded paths. Every location below is resolvable
from an environment variable, with a repo-relative default. This is what makes
the same config file work on this laptop (cache on D:) and on Colab (cache on
mounted Drive) without editing the config.

Environment variables honoured
-----------------------------
QUANTPROBE_CACHE : where extracted hidden states live. Large. Never committed.
HF_HOME          : HuggingFace's own cache root (model weights, datasets).
QUANTPROBE_ROOT  : repo root override, mostly for tests.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Repo root, i.e. the directory containing src/ and configs/."""
    override = os.environ.get("QUANTPROBE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    # src/paths.py -> src/ -> repo root
    return Path(__file__).resolve().parent.parent


def cache_dir() -> Path:
    """Root for cached activations. Big; lives off the repo by default.

    Defaults to <repo>/cache so a fresh clone works with zero setup, but on
    this machine QUANTPROBE_CACHE points at D: because C: has ~11 GB free.
    """
    override = os.environ.get("QUANTPROBE_CACHE")
    path = Path(override).expanduser() if override else repo_root() / "cache"
    return path.resolve()


def hf_home() -> Path | None:
    """HuggingFace cache root, if the user has pinned one."""
    override = os.environ.get("HF_HOME")
    return Path(override).expanduser().resolve() if override else None


def results_dir() -> Path:
    return repo_root() / "results"


def figures_dir() -> Path:
    return results_dir() / "figures"


def metrics_dir() -> Path:
    return results_dir() / "metrics"


def logs_dir() -> Path:
    return repo_root() / "logs"


def ensure_dirs(*paths: Path) -> None:
    """mkdir -p for each path. Called at the top of every script."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def free_space_gb(path: Path) -> float:
    """Free space on the volume holding `path`, in GB.

    Used by the extraction stage to refuse to start a run that cannot fit.
    """
    import shutil

    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / (1024**3)
