#!/usr/bin/env python3
"""Shared SAM3 source revision guard for native/export workflows."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


PRE31_SAM3_COMMIT = "86ed77094094e5cabb16b0414ec60c5ba9ce0a0f"
PRE31_SAM3_LABEL = "Meta SAM3 pre-3.1, 2026-03-16, before SAM 3.1 Object Multiplex"
SAM3_REPO_URL = "https://github.com/facebookresearch/sam3.git"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _run_git(args: list[str]) -> None:
    subprocess.run(["git", *args], check=True)


def sam3_git_revision(sam3_repo: Path) -> str:
    try:
        return _git(sam3_repo, "rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            f"Could not read SAM3 git revision at {sam3_repo}. "
            "Use a local facebookresearch/sam3 clone."
        ) from exc


def sam3_git_remote(sam3_repo: Path) -> str:
    try:
        return _git(sam3_repo, "config", "--get", "remote.origin.url")
    except (OSError, subprocess.CalledProcessError):
        return ""


def ensure_sam3_revision(
    sam3_repo: Path,
    *,
    expected: str = PRE31_SAM3_COMMIT,
    allow_mismatch: bool = False,
) -> str:
    actual = sam3_git_revision(sam3_repo)
    remote = sam3_git_remote(sam3_repo)
    allowed_by_env = os.getenv("SAM3_ALLOW_REVISION_MISMATCH", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if actual.lower() == expected.lower():
        return actual

    message = (
        "SAM3 source revision mismatch.\n"
        f"  repo     : {sam3_repo}\n"
        f"  remote   : {remote or 'unknown'}\n"
        f"  expected : {expected} ({PRE31_SAM3_LABEL})\n"
        f"  actual   : {actual}\n"
        f"Run: git -C \"{sam3_repo}\" fetch origin main && "
        f"git -C \"{sam3_repo}\" checkout {expected}"
    )
    if allow_mismatch or allowed_by_env:
        print("[WARN] " + message.replace("\n", "\n[WARN] "))
        return actual

    raise SystemExit(message)


def sync_sam3_repo(
    sam3_repo: Path,
    *,
    expected: str = PRE31_SAM3_COMMIT,
    repo_url: str = SAM3_REPO_URL,
) -> str:
    sam3_repo = Path(sam3_repo).resolve()
    if not sam3_repo.exists():
        sam3_repo.parent.mkdir(parents=True, exist_ok=True)
        _run_git(["clone", repo_url, str(sam3_repo)])
    elif not (sam3_repo / ".git").is_dir():
        raise SystemExit(
            f"Cannot sync SAM3 into {sam3_repo}: path exists but is not a Git checkout."
        )

    _run_git(["-C", str(sam3_repo), "fetch", "--tags", repo_url, "main"])
    _run_git(["-C", str(sam3_repo), "checkout", expected])
    return ensure_sam3_revision(sam3_repo, expected=expected)
