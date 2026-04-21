from __future__ import annotations

from typing import Any

from prompt_spec_utils import prompt_annotations_from_spec


DEFAULT_MAX_MEM_FRAMES = 2
MULTI_ANNOTATION_MAX_MEM_FRAMES = 4
DEFAULT_MAX_OBJ_PTRS = 16


def prompt_annotation_count(prompt_spec: dict[str, Any]) -> int:
    return max(1, len(prompt_annotations_from_spec(prompt_spec)))


def recommended_max_mem_frames(annotation_count: int) -> int:
    return (
        MULTI_ANNOTATION_MAX_MEM_FRAMES
        if int(annotation_count) > 1
        else DEFAULT_MAX_MEM_FRAMES
    )


def resolve_runtime_caps(
    *,
    prompt_spec: dict[str, Any] | None = None,
    annotation_count: int | None = None,
    max_mem_frames: int | None = None,
    max_obj_ptrs: int | None = None,
) -> tuple[int, int]:
    if annotation_count is None:
        if prompt_spec is None:
            raise ValueError("resolve_runtime_caps requires prompt_spec or annotation_count.")
        annotation_count = prompt_annotation_count(prompt_spec)

    resolved_max_mem_frames = (
        recommended_max_mem_frames(annotation_count)
        if max_mem_frames is None
        else int(max_mem_frames)
    )
    resolved_max_obj_ptrs = DEFAULT_MAX_OBJ_PTRS if max_obj_ptrs is None else int(max_obj_ptrs)

    if resolved_max_mem_frames <= 0:
        raise ValueError("max_mem_frames must be positive.")
    if resolved_max_obj_ptrs <= 0:
        raise ValueError("max_obj_ptrs must be positive.")

    return int(resolved_max_mem_frames), int(resolved_max_obj_ptrs)
