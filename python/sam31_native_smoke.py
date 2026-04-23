#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
import time
import types
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from local_sam3_config import DEFAULT_SAM31_CKPT, DEFAULT_SAM3_REPO
from prompt_spec_utils import load_prompt_spec, prompt_annotations_from_spec


REPO_ROOT = Path(__file__).resolve().parent.parent


def _configure_console_encoding() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _add_import_paths(sam3_repo: Path) -> None:
    for path in (sam3_repo.resolve(), REPO_ROOT.resolve()):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _install_windows_compat_stubs() -> None:
    if "sam3.model.edt" in sys.modules:
        return

    edt_module = types.ModuleType("sam3.model.edt")

    def edt_triton(data: torch.Tensor) -> torch.Tensor:
        if data.dim() != 3:
            raise ValueError(f"Expected [B,H,W] tensor, got shape {tuple(data.shape)}")

        device = data.device
        data_np = data.detach().to("cpu").numpy()
        out = np.zeros(data_np.shape, dtype=np.float32)
        for idx, mask in enumerate(data_np):
            dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 0)
            out[idx] = dist.astype(np.float32, copy=False)
        return torch.from_numpy(out).to(device=device)

    edt_module.edt_triton = edt_triton
    sys.modules["sam3.model.edt"] = edt_module


def _build_predictor(args: argparse.Namespace):
    from sam3.model_builder import build_sam3_predictor

    kwargs = dict(
        checkpoint_path=str(Path(args.checkpoint).resolve()),
        version="sam3.1",
        compile=False,
        warm_up=False,
        max_num_objects=int(args.max_objects),
        multiplex_count=int(args.multiplex_count),
        use_fa3=bool(args.use_fa3),
        use_rope_real=bool(args.use_rope_real),
        async_loading_frames=False,
    )

    print(f"[INFO] Building SAM 3.1 predictor from checkpoint: {kwargs['checkpoint_path']}")
    if args.verbose_builder:
        predictor = build_sam3_predictor(**kwargs)
        _configure_attention_backend(args.attention_backend)
        return predictor

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        predictor = build_sam3_predictor(**kwargs)

    skipped_missing_key_log = False
    for line in captured.getvalue().splitlines():
        if line.startswith("Missing keys:") or line.startswith("Unexpected keys:"):
            skipped_missing_key_log = True
            continue
        print(line)
    if skipped_missing_key_log:
        print("[INFO] Suppressed expected partial-load key logs from the internal SAM 3.1 tracker build.")
    _configure_attention_backend(args.attention_backend)
    return predictor


def _configure_attention_backend(mode: str) -> None:
    if mode == "optimized":
        print("[INFO] Leaving SAM 3.1 decoder attention on the optimized PyTorch SDPA backend.")
        return

    if mode == "auto":
        if not torch.cuda.is_available():
            _force_safe_sdpa_backend("CUDA is not available")
            return
        props = torch.cuda.get_device_properties(0)
        if int(props.major) >= 8:
            print(
                "[INFO] Leaving SAM 3.1 decoder attention optimized "
                f"for GPU compute capability {props.major}.{props.minor}."
            )
            return
        _force_safe_sdpa_backend(
            f"GPU compute capability {props.major}.{props.minor} is below Ampere 8.0"
        )
        return

    if mode == "safe_math":
        _force_safe_sdpa_backend("safe_math requested")
        return

    raise ValueError(f"Unsupported attention backend mode: {mode!r}")


def _force_safe_sdpa_backend(reason: str) -> None:
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.backends.cuda.enable_flash_sdp(False)
        with contextlib.suppress(Exception):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        with contextlib.suppress(Exception):
            torch.backends.cuda.enable_cudnn_sdp(False)
        with contextlib.suppress(Exception):
            torch.backends.cuda.enable_math_sdp(True)

    with contextlib.suppress(Exception):
        import sam3.model.decoder as decoder

        decoder.sdpa_kernel = lambda *args, **kwargs: contextlib.nullcontext()
        print(
            "[INFO] Patched SAM 3.1 decoder attention to use the safe "
            f"PyTorch math SDP backend ({reason})."
        )


def _extract_video_frames(video_path: Path, frames_dir: Path, max_frames: int) -> tuple[int, int, int]:
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    frame_count = 0
    width = 0
    height = 0
    try:
        while frame_count < max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            height, width = frame_bgr.shape[:2]
            frame_path = frames_dir / f"{frame_count:05d}.jpg"
            if not cv2.imwrite(str(frame_path), frame_bgr):
                raise SystemExit(f"Failed to write extracted frame: {frame_path}")
            frame_count += 1
    finally:
        cap.release()

    if frame_count == 0:
        raise SystemExit(f"No frames could be extracted from: {video_path}")
    return frame_count, width, height


def _tensor_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        data = value.detach()
        return {
            "type": "torch.Tensor",
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "device": str(data.device),
        }
    if isinstance(value, np.ndarray):
        return {
            "type": "numpy.ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "len": len(value),
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": sorted(str(key) for key in value.keys()),
        }
    return {"type": type(value).__name__, "repr": repr(value)[:120]}


def _outputs_summary(outputs: Any) -> dict[str, Any]:
    if not isinstance(outputs, dict):
        return {"outputs": _tensor_summary(outputs)}
    return {str(key): _tensor_summary(value) for key, value in outputs.items()}


def _add_annotation(
    predictor,
    session_id: str,
    annotation: dict[str, Any],
    *,
    width: int,
    height: int,
    obj_id: int,
) -> dict[str, Any]:
    frame_idx = int(annotation["frame_idx"])
    request: dict[str, Any] = {
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": frame_idx,
    }

    if annotation["prompt"] == "seed_points":
        points = []
        labels = []
        for x, y, label in annotation.get("points", []):
            points.append([float(x) / float(width), float(y) / float(height)])
            labels.append(int(label))
        request.update(
            {
                "points": torch.tensor(points, dtype=torch.float32),
                "point_labels": torch.tensor(labels, dtype=torch.int32),
                "obj_id": int(obj_id),
            }
        )
    elif annotation["prompt"] == "bounding_box":
        x1, y1, x2, y2 = [float(v) for v in annotation["box"]]
        request.update(
            {
                "bounding_boxes": torch.tensor(
                    [[x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height]],
                    dtype=torch.float32,
                ),
                "bounding_box_labels": torch.tensor([int(obj_id)], dtype=torch.int32),
            }
        )
    else:
        raise SystemExit(f"Unsupported prompt kind for SAM 3.1 smoke test: {annotation['prompt']!r}")

    t0 = time.time()
    response = predictor.handle_request(request)
    elapsed_ms = (time.time() - t0) * 1000.0
    print(
        f"[INFO] Add prompt frame={frame_idx:03d} kind={annotation['prompt']} "
        f"elapsed={elapsed_ms:.1f} ms"
    )
    return {
        "frame_idx": int(response["frame_index"]),
        "elapsed_ms": elapsed_ms,
        "outputs": _outputs_summary(response.get("outputs")),
    }


def _add_text_prompt(
    predictor,
    session_id: str,
    *,
    frame_idx: int,
    text_prompt: str,
) -> dict[str, Any]:
    request = {
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": int(frame_idx),
        "text": text_prompt,
    }
    t0 = time.time()
    response = predictor.handle_request(request)
    elapsed_ms = (time.time() - t0) * 1000.0
    print(
        f"[INFO] Add text prompt frame={frame_idx:03d} text={text_prompt!r} "
        f"elapsed={elapsed_ms:.1f} ms"
    )
    return {
        "frame_idx": int(response["frame_index"]),
        "text": text_prompt,
        "elapsed_ms": elapsed_ms,
        "outputs": _outputs_summary(response.get("outputs")),
    }


def _propagate(predictor, session_id: str, max_frames: int) -> tuple[list[dict[str, Any]], float]:
    request = {
        "type": "propagate_in_video",
        "session_id": session_id,
        "propagation_direction": "forward",
        "start_frame_index": 0,
        "max_frame_num_to_track": int(max_frames),
    }
    summaries: list[dict[str, Any]] = []
    t0 = time.time()
    for response in predictor.handle_stream_request(request):
        outputs = response.get("outputs")
        frame_idx = int(response["frame_index"])
        summaries.append({"frame_idx": frame_idx, "outputs": _outputs_summary(outputs)})
        print(f"[INFO] Propagated frame={frame_idx:03d}")
    elapsed_ms = (time.time() - t0) * 1000.0
    return summaries, elapsed_ms


def _start_session_compat(predictor, resource_path: Path) -> tuple[str, float]:
    init_kwargs = {
        "resource_path": str(resource_path),
        "offload_video_to_cpu": False,
    }
    if hasattr(predictor, "async_loading_frames"):
        init_kwargs["async_loading_frames"] = predictor.async_loading_frames
    if hasattr(predictor, "video_loader_type"):
        init_kwargs["video_loader_type"] = predictor.video_loader_type

    t0 = time.time()
    inference_state = predictor.model.init_state(**init_kwargs)
    session_id = str(uuid.uuid4())
    predictor._all_inference_states[session_id] = {
        "state": inference_state,
        "session_id": session_id,
        "start_time": time.time(),
        "last_use_time": time.time(),
    }
    elapsed_ms = (time.time() - t0) * 1000.0
    return session_id, elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test the native SAM 3.1 multiplex predictor on a short video clip."
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--prompt_json", required=True, help="Prompt JSON to replay.")
    parser.add_argument("--max_frames", type=int, default=5, help="Frames to extract and test.")
    parser.add_argument(
        "--sam3_repo",
        default=str(DEFAULT_SAM3_REPO),
        help="Path to the local SAM3.1 repo. Defaults to SAM3_REPO or an auto-detected sibling checkout.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_SAM31_CKPT),
        help="Path to sam3.1_multiplex.pt. Defaults to SAM31_CKPT or the local Hugging Face facebook/sam3.1 cache.",
    )
    parser.add_argument(
        "--outdir",
        default=str(REPO_ROOT / "checkpoints" / "sam3" / "sam31_smoke"),
        help="Output directory for extracted frames and summary.json.",
    )
    parser.add_argument("--obj_id", type=int, default=1, help="Object id used for seed-point prompts.")
    parser.add_argument(
        "--text_prompt",
        default="",
        help="Optional SAM 3.1 text prompt to add before replaying point/box annotations.",
    )
    parser.add_argument(
        "--text_frame_idx",
        type=int,
        default=0,
        help="Frame index used for --text_prompt.",
    )
    parser.add_argument("--max_objects", type=int, default=16, help="SAM 3.1 max_num_objects.")
    parser.add_argument("--multiplex_count", type=int, default=16, help="SAM 3.1 multiplex_count.")
    parser.add_argument(
        "--attention_backend",
        default="auto",
        choices=["auto", "optimized", "safe_math"],
        help="SAM 3.1 attention backend policy. auto keeps optimized SDPA on Ampere/A100+ and uses safe math on older GPUs.",
    )
    parser.add_argument("--use_fa3", action="store_true", help="Enable FlashAttention 3 if supported.")
    parser.add_argument("--use_rope_real", action="store_true", help="Enable real-valued RoPE.")
    parser.add_argument("--keep_frames", action="store_true", help="Keep extracted JPEG frames in outdir.")
    parser.add_argument("--verbose_builder", action="store_true", help="Show full SAM 3.1 builder logs.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("SAM 3.1 native predictor currently requires CUDA in the upstream builder.")

    video_path = Path(args.video).resolve()
    prompt_path = Path(args.prompt_json).resolve()
    sam3_repo = Path(args.sam3_repo).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    outdir = Path(args.outdir).resolve()
    frames_dir = outdir / "frames"
    summary_path = outdir / "summary.json"

    if not video_path.exists():
        raise SystemExit(f"Video does not exist: {video_path}")
    if not prompt_path.exists():
        raise SystemExit(f"Prompt JSON does not exist: {prompt_path}")
    if not checkpoint.exists():
        raise SystemExit(f"SAM 3.1 checkpoint does not exist: {checkpoint}")
    if not (sam3_repo / "sam3" / "model_builder.py").exists():
        raise SystemExit(f"SAM3 repo does not look valid: {sam3_repo}")

    outdir.mkdir(parents=True, exist_ok=True)
    _add_import_paths(sam3_repo)
    _install_windows_compat_stubs()

    print(f"[INFO] SAM 3.1 repo : {sam3_repo}")
    print(f"[INFO] Checkpoint   : {checkpoint}")
    print(f"[INFO] Video        : {video_path}")
    print(f"[INFO] Prompt JSON  : {prompt_path}")

    frame_count, width, height = _extract_video_frames(video_path, frames_dir, int(args.max_frames))
    print(f"[INFO] Extracted {frame_count} frames at {width}x{height}: {frames_dir}")

    prompt_spec = load_prompt_spec(prompt_path)
    annotations = [
        annotation
        for annotation in prompt_annotations_from_spec(prompt_spec)
        if int(annotation["frame_idx"]) < frame_count
    ]
    if not annotations:
        raise SystemExit("Prompt JSON has no annotations inside the extracted frame range.")
    print(f"[INFO] Replaying prompt frames: {[int(item['frame_idx']) for item in annotations]}")

    predictor = _build_predictor(args)
    session_id = None
    try:
        session_id, start_session_ms = _start_session_compat(predictor, frames_dir)
        print(f"[INFO] Started SAM 3.1 session in {start_session_ms:.1f} ms")

        add_prompt_summaries = []
        if args.text_prompt.strip():
            add_prompt_summaries.append(
                _add_text_prompt(
                    predictor,
                    session_id,
                    frame_idx=int(args.text_frame_idx),
                    text_prompt=args.text_prompt.strip(),
                )
            )

        add_prompt_summaries.extend([
            _add_annotation(
                predictor,
                session_id,
                annotation,
                width=width,
                height=height,
                obj_id=int(args.obj_id),
            )
            for annotation in annotations
        ])
        propagation_summaries, propagation_ms = _propagate(predictor, session_id, frame_count)
        print(
            f"[INFO] Propagated {len(propagation_summaries)} frames in "
            f"{propagation_ms:.1f} ms ({propagation_ms / max(1, len(propagation_summaries)):.1f} ms/frame)"
        )

        summary = {
            "sam3_repo": str(sam3_repo),
            "checkpoint": str(checkpoint),
            "video": str(video_path),
            "prompt_json": str(prompt_path),
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "start_session_ms": start_session_ms,
            "add_prompts": add_prompt_summaries,
            "propagation_ms": propagation_ms,
            "propagation_frame_count": len(propagation_summaries),
            "propagation_frames": propagation_summaries,
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[INFO] Wrote summary: {summary_path}")
    finally:
        if session_id is not None:
            with contextlib.suppress(Exception):
                predictor.handle_request({"type": "close_session", "session_id": session_id})
        if not args.keep_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    _configure_console_encoding()
    main()
