from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

import torch


@dataclass(frozen=True)
class ExternalRepoSpec:
    name: str
    path: Path
    git_sha: str | None
    remote_url: str | None

    def as_json(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "path": str(self.path),
            "git_sha": self.git_sha,
            "remote_url": self.remote_url,
        }


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def external_repo_path(name: str) -> Path:
    return project_root() / "external" / name


def _git_output(path: Path, *args: str) -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(path), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return output.strip() or None


def describe_external_repo(name: str) -> ExternalRepoSpec:
    path = external_repo_path(name)
    return ExternalRepoSpec(
        name=name,
        path=path,
        git_sha=_git_output(path, "rev-parse", "HEAD"),
        remote_url=_git_output(path, "remote", "get-url", "origin"),
    )


def ensure_import_path(repo_path: Path, src_subdir: str | None = None) -> None:
    target = repo_path / src_subdir if src_subdir else repo_path
    target_str = str(target)
    if target_str not in sys.path:
        sys.path.insert(0, target_str)


def import_axbench():
    repo_path = external_repo_path("axbench")
    ensure_import_path(repo_path)
    import axbench  # type: ignore

    return axbench


def import_alpaca_eval():
    repo_path = external_repo_path("alpaca_eval")
    ensure_import_path(repo_path, "src")
    import alpaca_eval  # type: ignore

    return alpaca_eval


def ensure_single_process_distributed_compat() -> None:
    if not torch.distributed.is_available() or torch.distributed.is_initialized():
        return

    original_get_rank = torch.distributed.get_rank

    def _safe_get_rank(*args, **kwargs):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return original_get_rank(*args, **kwargs)
        return 0

    torch.distributed.get_rank = _safe_get_rank
