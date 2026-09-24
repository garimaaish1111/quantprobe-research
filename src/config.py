"""YAML config loading.

Every script is CLI-driven off a YAML config. No hardcoded paths.

Two features beyond a plain yaml.safe_load:

1. `extends:` - a config may inherit from another and override keys. This lets
   configs/base.yaml hold the things that must be identical across every run
   (seed, token position, dtype-on-save) while model configs hold only what
   differs. If those shared settings were copy-pasted into each model config
   they would drift, and a drifted seed silently invalidates a cross-precision
   comparison.

2. `${ENV_VAR}` interpolation - so a path in a config resolves differently on
   this laptop and on Colab without the file being edited.

Access is attribute-style (cfg.model.hf_id) because it reads better at call
sites and, more usefully, raises a clear KeyError naming the missing key
instead of returning None the way dict.get would.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(Exception):
    """Raised for a malformed or incomplete config. Fails loudly."""


class Config(dict):
    """dict with attribute access, recursively applied to nested dicts."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError:
            raise AttributeError(
                f"config has no key {name!r}; available keys: {sorted(self.keys())}"
            ) from None
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    @classmethod
    def _wrap(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            return cls({k: cls._wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [cls._wrap(v) for v in obj]
        return obj

    def to_plain(self) -> dict:
        """Plain-dict copy, for json.dump into a run sidecar."""

        def unwrap(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: unwrap(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [unwrap(v) for v in obj]
            if isinstance(obj, Path):
                return str(obj)
            return obj

        return unwrap(self)


def _interpolate_env(obj: Any) -> Any:
    """Replace ${VAR} and ${VAR:-default} inside every string in the tree."""
    if isinstance(obj, str):

        def sub(match: re.Match) -> str:
            var, default = match.group(1), match.group(2)
            value = os.environ.get(var)
            if value is None:
                if default is None:
                    raise ConfigError(
                        f"config references ${{{var}}} but that environment "
                        f"variable is not set and no default was given "
                        f"(write it as ${{{var}:-some/default}} if optional)"
                    )
                return default
            return value

        return _ENV_PATTERN.sub(sub, obj)
    if isinstance(obj, dict):
        return {k: _interpolate_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_env(v) for v in obj]
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive merge; `override` wins at the leaves.

    Nested dicts merge key-by-key rather than being replaced wholesale, so a
    model config can override cfg.extract.batch_size without having to restate
    every other key under cfg.extract.
    """
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path, _seen: set[Path] | None = None) -> Config:
    """Load a YAML config, resolving `extends:` chains and ${ENV} refs.

    Args:
        path: path to the YAML file.
        _seen: internal, tracks the extends chain to catch cycles.

    Raises:
        ConfigError: file missing, not a mapping, cyclic extends, or an
            unresolvable ${ENV} reference.
    """
    path = Path(path).expanduser().resolve()
    seen = set() if _seen is None else set(_seen)

    if path in seen:
        raise ConfigError(f"cyclic `extends:` chain, revisited {path}")
    seen.add(path)

    if not path.is_file():
        raise ConfigError(f"config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config {path} must be a mapping at top level, got {type(raw).__name__}")

    parent_ref = raw.pop("extends", None)
    if parent_ref is not None:
        parent_path = (path.parent / str(parent_ref)).resolve()
        parent = load_config(parent_path, _seen=seen)
        raw = _deep_merge(parent.to_plain(), raw)

    raw = _interpolate_env(raw)
    cfg = Config._wrap(raw)
    cfg["_config_path"] = str(path)
    return cfg


def require(cfg: Config, *dotted_keys: str) -> None:
    """Assert that each dotted key exists. Fails loudly at script start.

    Catching a missing config key before a 40-minute extraction run rather
    than after it is the entire point.
    """
    missing = []
    for dotted in dotted_keys:
        node: Any = cfg
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                missing.append(dotted)
                break
    if missing:
        raise ConfigError(
            f"config {cfg.get('_config_path', '<unknown>')} is missing required "
            f"keys: {missing}"
        )
