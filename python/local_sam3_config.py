from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
_HF_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub"
_SAM3_REPO_CANDIDATE_NAMES = (
    "sam3-3p1",
    "sam3-3.1",
    "sam3_3p1",
    "sam3.1",
    "sam3",
)
_SAM3_MODEL_CACHE_NAME = "models--facebook--sam3"
_SAM31_MODEL_CACHE_NAME = "models--facebook--sam3.1"


def _is_sam3_repo(path: Path) -> bool:
    return path.exists() and (path / "sam3" / "model_builder.py").exists()


def _iter_repo_candidates():
    env_value = os.getenv("SAM3_REPO", "").strip()
    if env_value:
        yield Path(env_value).expanduser()

    parent = REPO_ROOT.parent
    for name in _SAM3_REPO_CANDIDATE_NAMES:
        yield parent / name


def resolve_default_sam3_repo() -> Path:
    for candidate in _iter_repo_candidates():
        resolved = candidate.resolve()
        if _is_sam3_repo(resolved):
            return resolved
    return (REPO_ROOT.parent / "sam3").resolve()


def _iter_checkpoint_candidates(*, env_var: str, model_cache_name: str):
    env_value = os.getenv(env_var, "").strip()
    if env_value:
        yield Path(env_value).expanduser()

    snapshots_dir = _HF_CACHE_ROOT / model_cache_name / "snapshots"
    if not snapshots_dir.exists():
        return
    for checkpoint in sorted(
        snapshots_dir.rglob("sam3*.pt"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ):
        yield checkpoint


def resolve_default_sam3_checkpoint() -> Path:
    for candidate in _iter_checkpoint_candidates(
        env_var="SAM3_CKPT",
        model_cache_name=_SAM3_MODEL_CACHE_NAME,
    ):
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return (
        _HF_CACHE_ROOT
        / _SAM3_MODEL_CACHE_NAME
        / "snapshots"
        / "missing"
        / "sam3.pt"
    ).resolve()


def resolve_default_sam31_checkpoint() -> Path:
    for candidate in _iter_checkpoint_candidates(
        env_var="SAM31_CKPT",
        model_cache_name=_SAM31_MODEL_CACHE_NAME,
    ):
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return (
        _HF_CACHE_ROOT
        / _SAM31_MODEL_CACHE_NAME
        / "snapshots"
        / "missing"
        / "sam3.1_multiplex.pt"
    ).resolve()


DEFAULT_SAM3_REPO = resolve_default_sam3_repo()
DEFAULT_CKPT = resolve_default_sam3_checkpoint()
DEFAULT_SAM31_CKPT = resolve_default_sam31_checkpoint()
