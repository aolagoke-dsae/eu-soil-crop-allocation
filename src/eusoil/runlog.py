"""The run log: one row per command run, in ``reports/run_log.csv``.

Each row records what is needed to tie a results table to the code and
settings that produced it: the command and its options, the SHA-256 of the
configuration file, the git commit and whether the working tree had
uncommitted changes, the seed, the Python version and the versions of the
packages the results depend on, the start time, the duration and how the run
ended. Nothing about the data or the machine's paths is written, so the log
can be committed with the tables.
"""

from __future__ import annotations

import csv
import hashlib
import platform
import subprocess
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "shap",
    "xarray",
    "rasterio",
    "pyproj",
    "geopandas",
)
FIELDS = (
    "started_utc",
    "command",
    "options",
    "status",
    "duration_s",
    "git_commit",
    "git_dirty",
    "config_sha256",
    "seed",
    "python",
    "platform",
    "packages",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state(root: Path) -> tuple[str, str]:
    """(commit, dirty) of the repository at ``root``; ("unknown", "unknown") without git."""

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        dirty = "yes" if run("status", "--porcelain", "--untracked-files=no") else "no"
    except (OSError, subprocess.SubprocessError):
        return "unknown", "unknown"
    return commit, dirty


def package_versions(names: tuple[str, ...] = PACKAGES) -> str:
    versions = []
    for name in names:
        try:
            versions.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            continue
    return ";".join(versions)


class RunLog:
    """Context manager that appends one row when the command ends, however it ends."""

    def __init__(
        self,
        log_path: Path,
        root: Path,
        command: str,
        options: dict[str, Any],
        config_path: Path,
        config: dict[str, Any],
    ) -> None:
        self.log_path = log_path
        self.root = root
        self.row: dict[str, Any] = {
            "command": command,
            "options": ";".join(f"{k}={v}" for k, v in sorted(options.items())),
            "config_sha256": file_sha256(config_path),
            "seed": config.get("seed", ""),
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.machine()}",
            "packages": package_versions(),
        }
        self.row["git_commit"], self.row["git_dirty"] = git_state(root)

    def __enter__(self) -> RunLog:
        self.row["started_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            status = "ok"
        elif exc_type is SystemExit and exc.code in (None, 0):
            status = "ok"
        elif exc_type is SystemExit:
            status = f"stopped: {self._without_root(str(exc.code))[:200]}"
        elif exc_type is KeyboardInterrupt:
            status = "interrupted"
        else:
            status = f"error: {exc_type.__name__}"
        self.row["status"] = status
        self.row["duration_s"] = round(time.perf_counter() - self._start, 1)
        self.write()
        return False  # never swallow the exception

    def _without_root(self, text: str) -> str:
        """Messages name files by their path in the project, never the machine's."""
        for root in {str(self.root), self.root.as_posix()}:
            text = text.replace(root + "/", "").replace(root + "\\", "").replace(root, ".")
        return text

    def write(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.log_path.exists()
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if new:
                writer.writeheader()
            writer.writerow({k: self.row.get(k, "") for k in FIELDS})


def options_of(args: Any) -> dict[str, Any]:
    """The command's own options from an argparse namespace (not the config path)."""
    return {k: v for k, v in vars(args).items() if k not in {"command", "config"} and v is not None}


__all__ = ["RunLog", "file_sha256", "git_state", "options_of", "package_versions"]
