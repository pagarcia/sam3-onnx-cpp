from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROMPT_KINDS = {"seed_points", "bounding_box"}


def parse_points_text(text: str) -> tuple[list[tuple[int, int]], list[int]]:
    points, labels = [], []
    if not text.strip():
        return points, labels
    for item in text.split(";"):
        x_str, y_str, label_str = [part.strip() for part in item.split(",")]
        points.append((int(float(x_str)), int(float(y_str))))
        labels.append(int(label_str))
    return points, labels


def parse_box_text(text: str) -> tuple[int, int, int, int]:
    parts = [int(float(part.strip())) for part in text.split(",")]
    if len(parts) != 4:
        raise ValueError("--box expects x1,y1,x2,y2")
    return tuple(parts)


def _normalize_annotation(
    annotation: dict[str, Any],
    *,
    default_frame_idx: int,
) -> dict[str, Any]:
    if not isinstance(annotation, dict):
        raise SystemExit("Prompt annotation entries must be JSON objects.")

    frame_idx = int(annotation.get("frame_idx", default_frame_idx))
    if frame_idx < 0:
        raise SystemExit("Prompt annotation frame_idx values must be non-negative.")

    prompt = annotation.get("prompt")
    if prompt not in PROMPT_KINDS:
        raise SystemExit(
            f"Unsupported prompt kind {prompt!r}. Expected one of: {', '.join(sorted(PROMPT_KINDS))}."
        )

    normalized = {
        "frame_idx": frame_idx,
        "prompt": prompt,
    }
    if prompt == "bounding_box":
        box = annotation.get("box")
        if box is None:
            normalized["box"] = None
        else:
            if len(box) != 4:
                raise SystemExit("Bounding-box prompts must provide four values: x1,y1,x2,y2.")
            normalized["box"] = [int(float(value)) for value in box]
    else:
        points = []
        for item in annotation.get("points", []):
            if len(item) != 3:
                raise SystemExit("Seed-point prompts must provide x,y,label triplets.")
            points.append([int(float(item[0])), int(float(item[1])), int(item[2])])
        normalized["points"] = points
    return normalized


def canonicalize_prompt_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise SystemExit("Prompt JSON must be a JSON object.")

    if "annotations" in spec:
        raw_annotations = spec.get("annotations")
        if not isinstance(raw_annotations, list) or not raw_annotations:
            raise SystemExit("Prompt JSON 'annotations' must be a non-empty list.")
        annotations = [
            _normalize_annotation(annotation, default_frame_idx=idx)
            for idx, annotation in enumerate(raw_annotations)
        ]
    elif "prompt" in spec:
        annotations = [_normalize_annotation(spec, default_frame_idx=0)]
    else:
        raise SystemExit("Prompt JSON must define either 'prompt' or 'annotations'.")

    annotations.sort(key=lambda item: item["frame_idx"])
    seen_frame_indices: set[int] = set()
    for annotation in annotations:
        frame_idx = int(annotation["frame_idx"])
        if frame_idx in seen_frame_indices:
            raise SystemExit(
                f"Prompt JSON contains duplicate annotations for frame {frame_idx}."
            )
        seen_frame_indices.add(frame_idx)

    return {
        "version": 2,
        "annotations": annotations,
    }


def prompt_annotations_from_spec(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return canonicalize_prompt_spec(spec)["annotations"]


def compact_prompt_spec(spec: dict[str, Any]) -> dict[str, Any]:
    canonical = canonicalize_prompt_spec(spec)
    annotations = canonical["annotations"]
    if len(annotations) == 1 and int(annotations[0]["frame_idx"]) == 0:
        compact = dict(annotations[0])
        compact.pop("frame_idx", None)
        return compact
    return canonical


def load_prompt_spec(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        spec = json.load(f)
    return canonicalize_prompt_spec(spec)


def save_prompt_spec(path: Path, spec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(compact_prompt_spec(spec), f, indent=2)

