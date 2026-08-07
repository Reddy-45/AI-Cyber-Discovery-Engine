"""
engine/config.py — Configuration loader.

Loads config.yaml into a plain Python dict.
Access any value with: cfg["pipeline"]["alert_threshold"]

Intentionally simple — no nested Pydantic models, no dataclasses.
Fifteen config keys don't need an object hierarchy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
_cache: dict[str, Any] | None = None


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """
    Load and return the engine configuration as a plain dict.

    Results are cached after the first call — the file is only
    read once per process.

    Args:
        path: Override path to a YAML config file. Defaults to
              project-root config.yaml.

    Returns:
        Configuration dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML is malformed.
    """
    global _cache
    if _cache is not None and path is None:
        return _cache

    config_path = Path(path) if path else _DEFAULT_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed YAML in {config_path}: {exc}") from exc

    if path is None:
        _cache = raw
    return raw
