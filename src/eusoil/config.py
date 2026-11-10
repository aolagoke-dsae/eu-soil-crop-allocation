"""Load ``config/analysis.yaml`` and resolve the project's paths.

All paths in the configuration are relative to the project root, so the
pipeline runs the same from any working directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "analysis.yaml"


@dataclass(frozen=True)
class Paths:
    """Resolved locations of the data layers and source files."""

    root: Path
    raw: Path
    interim: Path
    processed: Path
    reports: Path
    soil: Path
    texture: Path
    landcover_2018: Path
    nuts: Path
    worldclim_30s: Path
    worldclim_2_5m: Path

    def ensure_output_dirs(self) -> None:
        """Create the writable layers; ``raw`` is never created or written."""
        for directory in (
            self.interim,
            self.processed,
            self.reports / "tables",
            self.reports / "figures",
        ):
            directory.mkdir(parents=True, exist_ok=True)


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Read the analysis configuration as a plain dictionary."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    with config_path.open(encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    return config


def resolve_paths(config: dict[str, Any], root: Path | None = None) -> Paths:
    """Turn the relative paths in ``config['paths']`` into absolute paths."""
    base = root if root is not None else PROJECT_ROOT
    p = config["paths"]
    raw = base / p["raw"]
    return Paths(
        root=base,
        raw=raw,
        interim=base / p["interim"],
        processed=base / p["processed"],
        reports=base / p["reports"],
        soil=raw / p["soil"],
        texture=raw / p["texture"],
        landcover_2018=raw / p["landcover_2018"],
        nuts=raw / p["nuts"],
        worldclim_30s=raw / p["worldclim_30s"],
        worldclim_2_5m=raw / p["worldclim_2_5m"],
    )
