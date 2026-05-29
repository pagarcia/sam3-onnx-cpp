# sam3-onnx-cpp/python/compare_native_vs_onnx.py
#!/usr/bin/env python3
import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
import types
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from onnx_runtime_policy import resolve_runtime_caps as resolve_onnx_runtime_caps
from prompt_spec_utils import (
    load_prompt_spec as _shared_load_prompt_spec,
    parse_box_text as _shared_parse_box_text,
    parse_points_text as _shared_parse_points_text,
    prompt_annotations_from_spec,
    save_prompt_spec as _shared_save_prompt_spec,
)
from sam3_revision import ensure_sam3_revision


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAM3_REPO = REPO_ROOT.parent / "sam3"
DEFAULT_ONNX_DIR = REPO_ROOT / "checkpoints" / "sam3" / "video_onnx"
DEFAULT_CKPT = Path(
    r"C:\Users\Pablo\.cache\huggingface\hub\models--facebook--sam3\snapshots\3c879f39826c281e95690f02c7821c4de09afae7\sam3.pt"
)
TARGET_SIZE = 1008
DEFAULT_RUN_ORDER = "onnx,native"


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        _sync_cuda()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def _measure_torch(fn):
    _sync_cuda()
    t0 = time.time()
    out = fn()
    _sync_cuda()
    return out, (time.time() - t0) * 1000.0


def _parse_points_text(text: str):
    return _shared_parse_points_text(text)


def _parse_box_text(text: str):
    return _shared_parse_box_text(text)


def _load_prompt_spec(path: Path):
    return _shared_load_prompt_spec(path)


def _save_prompt_spec(path: Path, spec) -> None:
    _shared_save_prompt_spec(path, spec)


def _compute_display_base(frame_bgr: np.ndarray, max_side: int = 1200):
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale == 1.0:
        return frame_bgr.copy(), scale
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA), scale


def _interactive_select_points(first_bgr):
    points, labels = [], []
    base, scale = _compute_display_base(first_bgr)

    def render():
        vis = base.copy()
        for idx, (px, py) in enumerate(points):
            color = (0, 0, 255) if labels[idx] == 1 else (255, 0, 0)
            cv2.circle(vis, (int(px * scale), int(py * scale)), 6, color, -1)
        cv2.imshow("SAM3 Compare Prompt", vis)

    def mouse_cb(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x / scale), int(y / scale)))
            labels.append(1)
            render()
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append((int(x / scale), int(y / scale)))
            labels.append(0)
            render()
        elif event == cv2.EVENT_MBUTTONDOWN:
            points.clear()
            labels.clear()
            render()

    cv2.namedWindow("SAM3 Compare Prompt")
    cv2.setMouseCallback("SAM3 Compare Prompt", mouse_cb)
    render()
    print("[INFO] L-click=FG, R-click=BG, M-click=reset. Press Enter or ESC when done.")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 27):
            break
    cv2.destroyAllWindows()
    return {
        "prompt": "seed_points",
        "points": [[int(px), int(py), int(label)] for (px, py), label in zip(points, labels)],
    }


def _interactive_select_box(first_bgr):
    rect_start = None
    rect_end = None
    drawing = False
    base, scale = _compute_display_base(first_bgr)

    def render():
        vis = base.copy()
        if rect_start is not None and rect_end is not None:
            cv2.rectangle(vis, rect_start, rect_end, (0, 255, 255), 2)
        cv2.imshow("SAM3 Compare Prompt", vis)

    def mouse_cb(event, x, y, _flags, _param):
        nonlocal rect_start, rect_end, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            rect_start = rect_end = (x, y)
            render()
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            rect_end = (x, y)
            render()
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            rect_end = (x, y)
            render()
        elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK):
            rect_start = None
            rect_end = None
            drawing = False
            render()

    cv2.namedWindow("SAM3 Compare Prompt")
    cv2.setMouseCallback("SAM3 Compare Prompt", mouse_cb)
    render()
    print("[INFO] Drag a box. Right click resets. Press Enter or ESC when done.")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 27):
            break
    cv2.destroyAllWindows()
    if rect_start is None or rect_end is None:
        return {"prompt": "bounding_box", "box": None}
    x1, y1 = rect_start[0] / scale, rect_start[1] / scale
    x2, y2 = rect_end[0] / scale, rect_end[1] / scale
    return {
        "prompt": "bounding_box",
        "box": [int(x1), int(y1), int(x2), int(y2)],
    }


def _resolve_prompt(args, first_bgr):
    if args.prompt_json:
        return _load_prompt_spec(Path(args.prompt_json).resolve())
    if args.box:
        return {"prompt": "bounding_box", "box": list(_parse_box_text(args.box))}
    if args.points:
        points, labels = _parse_points_text(args.points)
        return {
            "prompt": "seed_points",
            "points": [[int(px), int(py), int(label)] for (px, py), label in zip(points, labels)],
        }
    if args.prompt == "bounding_box":
        return _interactive_select_box(first_bgr)
    return _interactive_select_points(first_bgr)


def _validate_prompt_annotations(
    annotations: list[dict],
    *,
    frame_count: int,
) -> None:
    if not annotations:
        raise SystemExit("At least one prompt annotation is required.")
    if int(annotations[0]["frame_idx"]) != 0:
        raise SystemExit(
            "Multi-annotation video prompts currently require the first annotation to be on frame 0."
        )
    for annotation in annotations:
        frame_idx = int(annotation["frame_idx"])
        if frame_idx >= frame_count:
            raise SystemExit(
                f"Prompt annotation frame {frame_idx} is outside the loaded video length ({frame_count} frames)."
            )


def _preprocess_frame_native(frame_bgr: np.ndarray) -> torch.Tensor:
    frame_resized = cv2.resize(frame_bgr, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).astype(np.float32) * (1.0 / 255.0)
    frame_rgb = (frame_rgb - 0.5) / 0.5
    chw = np.transpose(frame_rgb, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(chw)).to(torch.float32)


@dataclass
class VideoFrames:
    raw_frames: list[np.ndarray]
    processed_frames: list[torch.Tensor]
    preprocess_ms: list[float]
    fps: float
    width: int
    height: int


def _load_video_frames(video_path: str, max_frames: int) -> VideoFrames:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_frames, processed_frames, preprocess_ms = [], [], []

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if max_frames > 0 and len(raw_frames) >= max_frames:
            break
        raw_frames.append(frame_bgr.copy())
        t0 = time.time()
        processed_frames.append(_preprocess_frame_native(frame_bgr))
        preprocess_ms.append((time.time() - t0) * 1000.0)

    cap.release()
    if not raw_frames:
        raise SystemExit("The selected video is empty.")

    return VideoFrames(
        raw_frames=raw_frames,
        processed_frames=processed_frames,
        preprocess_ms=preprocess_ms,
        fps=fps if fps > 0 else 25.0,
        width=width,
        height=height,
    )


def _mask_to_uint8(mask_logits_high_res: torch.Tensor, width: int, height: int) -> np.ndarray:
    mask_logits = mask_logits_high_res[0, 0].detach().float().cpu().numpy()
    mask_resized = cv2.resize(mask_logits, (width, height), interpolation=cv2.INTER_LINEAR)
    return (mask_resized > 0.0).astype(np.uint8) * 255


def _prepare_native_point_inputs(prompt_spec, width: int, height: int, device: torch.device):
    if prompt_spec["prompt"] == "bounding_box":
        box = prompt_spec.get("box")
        if box is None:
            return None
        x1, y1, x2, y2 = box
        points = torch.tensor(
            [
                [x1 * TARGET_SIZE / float(width), y1 * TARGET_SIZE / float(height)],
                [x2 * TARGET_SIZE / float(width), y2 * TARGET_SIZE / float(height)],
            ],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        labels = torch.tensor([[2, 3]], dtype=torch.int32, device=device)
        return {"point_coords": points, "point_labels": labels}

    raw_points = prompt_spec.get("points", [])
    if not raw_points:
        return None
    points = torch.tensor(
        [
            [item[0] * TARGET_SIZE / float(width), item[1] * TARGET_SIZE / float(height)]
            for item in raw_points
        ],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    labels = torch.tensor([[int(item[2]) for item in raw_points]], dtype=torch.int32, device=device)
    return {"point_coords": points, "point_labels": labels}


def _add_import_paths(sam3_repo: Path) -> None:
    sys.path.insert(0, str(sam3_repo.resolve()))
    sys.path.insert(0, str(REPO_ROOT.resolve()))


def _install_optional_sam3_stubs() -> None:
    if "sam3.model.edt" not in sys.modules:
        edt_module = types.ModuleType("sam3.model.edt")

        def edt_triton(data: torch.Tensor) -> torch.Tensor:
            out = np.zeros(tuple(data.shape), dtype=np.float32)
            for idx, mask in enumerate(data.detach().cpu().numpy()):
                out[idx] = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 0)
            return torch.from_numpy(out).to(device=data.device)

        edt_module.edt_triton = edt_triton
        sys.modules["sam3.model.edt"] = edt_module

    if "sam3.train.data.collator" not in sys.modules:
        collator_module = types.ModuleType("sam3.train.data.collator")

        @dataclass
        class BatchedDatapoint:
            img_batch: object = None
            find_text_batch: object = None
            find_inputs: object = None
            find_targets: object = None
            find_metadatas: object = None
            raw_images: object = None

        collator_module.BatchedDatapoint = BatchedDatapoint
        sys.modules["sam3.train.data.collator"] = collator_module

    if "sam3.model.sam3_video_inference" not in sys.modules:
        module = types.ModuleType("sam3.model.sam3_video_inference")

        class Sam3VideoInferenceWithInstanceInteractivity:
            pass

        module.Sam3VideoInferenceWithInstanceInteractivity = Sam3VideoInferenceWithInstanceInteractivity
        sys.modules["sam3.model.sam3_video_inference"] = module

    if "sam3.model.sam3_video_predictor" not in sys.modules:
        module = types.ModuleType("sam3.model.sam3_video_predictor")

        class Sam3VideoPredictorMultiGPU:
            pass

        module.Sam3VideoPredictorMultiGPU = Sam3VideoPredictorMultiGPU
        sys.modules["sam3.model.sam3_video_predictor"] = module


def _build_native_model(sam3_repo: Path, checkpoint: str):
    ensure_sam3_revision(sam3_repo)
    _add_import_paths(sam3_repo)
    _install_optional_sam3_stubs()
    from sam3.model_builder import build_sam3_image_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(
        checkpoint_path=checkpoint,
        load_from_HF=False,
        device=device,
        eval_mode=True,
        enable_inst_interactivity=True,
    )
    return model


def _compute_native_backbone_features(image_model, tracker, image: torch.Tensor):
    backbone_out = image_model.backbone.forward_image(image)["sam2_backbone_out"].copy()
    backbone_out["backbone_fpn"] = list(backbone_out["backbone_fpn"])
    backbone_out["vision_pos_enc"] = list(backbone_out["vision_pos_enc"])
    backbone_out["backbone_fpn"][0] = tracker.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
    backbone_out["backbone_fpn"][1] = tracker.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
    return tracker._prepare_backbone_features(backbone_out)


@torch.inference_mode()
def _run_native_tracker(
    video: VideoFrames,
    prompt_spec,
    checkpoint: str,
    sam3_repo: Path,
    save_path: Path,
    video_path: str,
):
    print(
        f"[INFO] Building native SAM3 model from checkpoint: {Path(checkpoint).resolve()}",
        flush=True,
    )
    image_model = _build_native_model(sam3_repo, checkpoint)
    tracker = image_model.inst_interactive_predictor.model
    device = next(image_model.parameters()).device
    print(f"[INFO] Native model ready on device: {device}", flush=True)
    annotations = prompt_annotations_from_spec(prompt_spec)
    _validate_prompt_annotations(annotations, frame_count=len(video.raw_frames))
    print(
        f"[INFO] Native prompt frames: {[int(annotation['frame_idx']) for annotation in annotations]}",
        flush=True,
    )
    point_inputs_by_frame = {
        int(annotation["frame_idx"]): _prepare_native_point_inputs(
            annotation,
            video.width,
            video.height,
            device,
        )
        for annotation in annotations
    }
    output_dict = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }

    saved_masks = []
    saved_prep_ms = []
    saved_enc_ms = []
    saved_attn_ms = []
    saved_dec_ms = []
    saved_mem_ms = []
    saved_total_ms = []
    total_frames = len(video.raw_frames)
    run_t0 = time.time()

    try:
        for frame_idx, (frame_bgr, frame_cpu) in enumerate(zip(video.raw_frames, video.processed_frames)):
            prep_ms = float(video.preprocess_ms[frame_idx])
            frame_t0 = time.time()
            image = frame_cpu.to(device, non_blocking=True).unsqueeze(0)

            (_, current_vision_feats, current_vision_pos_embeds, feat_sizes), enc_ms = _measure_torch(
                lambda: _compute_native_backbone_features(image_model, tracker, image)
            )

            if len(current_vision_feats) > 1:
                high_res_features = [
                    x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                    for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
                ]
            else:
                high_res_features = None

            is_init_cond_frame = frame_idx in point_inputs_by_frame
            model_point_inputs = point_inputs_by_frame.get(frame_idx)

            fused_embed, attn_ms = _measure_torch(
                lambda: tracker._prepare_memory_conditioned_features(
                    frame_idx=frame_idx,
                    is_init_cond_frame=is_init_cond_frame,
                    current_vision_feats=current_vision_feats[-1:],
                    current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                    feat_sizes=feat_sizes[-1:],
                    output_dict=output_dict,
                    num_frames=len(video.raw_frames),
                    track_in_reverse=False,
                    use_prev_mem_frame=True,
                )
            )

            multimask_output = tracker._use_multimask(is_init_cond_frame, model_point_inputs)
            sam_outputs, dec_ms = _measure_torch(
                lambda: tracker._forward_sam_heads(
                    backbone_features=fused_embed,
                    point_inputs=model_point_inputs,
                    mask_inputs=None,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                )
            )
            (
                _,
                _high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
            ) = sam_outputs

            mem_out, mem_ms = _measure_torch(
                lambda: tracker._encode_new_memory(
                    image=image,
                    current_vision_feats=current_vision_feats,
                    feat_sizes=feat_sizes,
                    pred_masks_high_res=high_res_masks,
                    object_score_logits=object_score_logits,
                    is_mask_from_pts=is_init_cond_frame,
                    output_dict=output_dict,
                    is_init_cond_frame=is_init_cond_frame,
                )
            )
            maskmem_features, maskmem_pos_enc = mem_out

            current_out = {
                "maskmem_features": maskmem_features,
                "maskmem_pos_enc": maskmem_pos_enc,
                "pred_masks": low_res_masks,
                "obj_ptr": obj_ptr,
                "object_score_logits": object_score_logits,
            }
            if tracker.use_memory_selection:
                iou_score = ious.max(-1)[0]
                current_out["iou_score"] = iou_score
                current_out["eff_iou_score"] = tracker.cal_mem_score(object_score_logits, iou_score)
            if is_init_cond_frame:
                output_dict["cond_frame_outputs"][frame_idx] = current_out
            else:
                output_dict["non_cond_frame_outputs"][frame_idx] = current_out

            mask_uint8 = _mask_to_uint8(high_res_masks, video.width, video.height)
            saved_masks.append(mask_uint8)
            saved_prep_ms.append(prep_ms)
            saved_enc_ms.append(enc_ms)
            saved_attn_ms.append(0.0 if is_init_cond_frame else attn_ms)
            saved_dec_ms.append(dec_ms)
            saved_mem_ms.append(mem_ms)
            total_ms = ((time.time() - frame_t0) * 1000.0) + prep_ms
            saved_total_ms.append(total_ms)

            should_log = (
                frame_idx < 3
                or is_init_cond_frame
                or (frame_idx + 1) == total_frames
                or ((frame_idx + 1) % 10 == 0)
            )
            if should_log:
                elapsed_s = max(0.0, time.time() - run_t0)
                processed = frame_idx + 1
                avg_ms = float(np.mean(saved_total_ms)) if saved_total_ms else 0.0
                remaining = max(0, total_frames - processed)
                eta_s = (avg_ms * remaining) / 1000.0 if avg_ms > 0 else 0.0
                phase = "Cond" if is_init_cond_frame else "Track"
                if is_init_cond_frame:
                    print(
                        f"[INFO] Native Frame {frame_idx:03d} | {phase} | "
                        f"Prep:{prep_ms:.1f} Enc:{enc_ms:.1f} Dec:{dec_ms:.1f} Mem:{mem_ms:.1f} "
                        f"Total:{total_ms:.1f} ms | "
                        f"Elapsed:{elapsed_s:.1f}s ETA:{eta_s:.1f}s",
                        flush=True,
                    )
                else:
                    print(
                        f"[INFO] Native Frame {frame_idx:03d} | {phase} | "
                        f"Prep:{prep_ms:.1f} Enc:{enc_ms:.1f} Attn:{attn_ms:.1f} "
                        f"Dec:{dec_ms:.1f} Mem:{mem_ms:.1f} Total:{total_ms:.1f} ms | "
                        f"Elapsed:{elapsed_s:.1f}s ETA:{eta_s:.1f}s",
                        flush=True,
                    )

            del (
                image,
                current_vision_feats,
                current_vision_pos_embeds,
                feat_sizes,
                high_res_features,
                model_point_inputs,
                fused_embed,
                sam_outputs,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
                mem_out,
                maskmem_features,
                maskmem_pos_enc,
                current_out,
                mask_uint8,
            )

        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            masks=np.stack(saved_masks).astype(np.uint8, copy=False),
            prep_ms=np.asarray(saved_prep_ms, dtype=np.float32),
            enc_ms=np.asarray(saved_enc_ms, dtype=np.float32),
            attn_ms=np.asarray(saved_attn_ms, dtype=np.float32),
            dec_ms=np.asarray(saved_dec_ms, dtype=np.float32),
            mem_ms=np.asarray(saved_mem_ms, dtype=np.float32),
            total_ms=np.asarray(saved_total_ms, dtype=np.float32),
            prompt_json=np.array(json.dumps(prompt_spec)),
            video_path=np.array(str(video_path)),
        )
        print(f"[INFO] Native benchmark dump saved: {save_path}", flush=True)
        return save_path
    finally:
        del output_dict, point_inputs_by_frame, tracker, image_model
        _release_cuda_memory()


def _run_onnx_subprocess(
    video_path: str,
    onnx_dir: Path,
    prompt_json: Path,
    save_path: Path,
    max_frames: int,
    safe: bool,
    onnx_accel: str,
    onnx_max_mem_frames: int,
    onnx_max_obj_ptrs: int,
):
    cmd = [
        sys.executable,
        str(REPO_ROOT / "python" / "onnx_test_video.py"),
        "--video",
        video_path,
        "--onnx_dir",
        str(onnx_dir),
        "--prompt_json",
        str(prompt_json),
        "--save_npz",
        str(save_path),
        "--max_frames",
        str(max_frames),
        "--no_output_video",
    ]
    if safe:
        cmd.append("--safe")
    if onnx_max_mem_frames > 0:
        cmd.extend(["--max_mem_frames", str(onnx_max_mem_frames)])
    if onnx_max_obj_ptrs > 0:
        cmd.extend(["--max_obj_ptrs", str(onnx_max_obj_ptrs)])

    env = os.environ.copy()
    env["SAM3_ORT_ACCEL"] = onnx_accel
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, env=env)


def _run_native_subprocess(
    video_path: str,
    checkpoint: str,
    sam3_repo: Path,
    prompt_json: Path,
    save_path: Path,
    max_frames: int,
):
    cmd = [
        sys.executable,
        str(REPO_ROOT / "python" / "compare_native_vs_onnx.py"),
        "--worker_mode",
        "native",
        "--video",
        video_path,
        "--prompt_json",
        str(prompt_json),
        "--worker_save_npz",
        str(save_path),
        "--max_frames",
        str(max_frames),
        "--checkpoint",
        str(Path(checkpoint).resolve()),
        "--sam3_repo",
        str(Path(sam3_repo).resolve()),
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, env=os.environ.copy())


def _parse_run_order(text: str) -> list[str]:
    raw_items = [item.strip().lower() for item in text.split(",") if item.strip()]
    allowed = {"onnx", "native"}
    if sorted(raw_items) != ["native", "onnx"]:
        raise SystemExit("--run_order must be exactly 'onnx,native' or 'native,onnx'.")
    return raw_items


def _frame_metrics(native_mask: np.ndarray, onnx_mask: np.ndarray):
    native_bool = native_mask > 0
    onnx_bool = onnx_mask > 0
    inter = np.logical_and(native_bool, onnx_bool).sum(dtype=np.int64)
    union = np.logical_or(native_bool, onnx_bool).sum(dtype=np.int64)
    native_area = native_bool.sum(dtype=np.int64)
    onnx_area = onnx_bool.sum(dtype=np.int64)
    iou = 1.0 if union == 0 else float(inter) / float(union)
    dice_den = native_area + onnx_area
    dice = 1.0 if dice_den == 0 else float(2 * inter) / float(dice_den)
    pixel_acc = float((native_bool == onnx_bool).mean())
    return {
        "iou": iou,
        "dice": dice,
        "pixel_acc": pixel_acc,
        "native_area": int(native_area),
        "onnx_area": int(onnx_area),
    }


def _load_npz(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _decode_json_scalar(data: dict, key: str):
    if key not in data:
        return None
    value = data[key]
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not value:
        return None
    return json.loads(value)


def _summarize_timings(prefix: str, data):
    summary = {}
    for key in ("prep_ms", "enc_ms", "attn_ms", "dec_ms", "mem_ms", "total_ms"):
        if key not in data:
            continue
        values = np.asarray(data[key], dtype=np.float32)
        summary[f"{prefix}_mean_{key}"] = float(values.mean())
        summary[f"{prefix}_median_{key}"] = float(np.median(values))
    mean_total = summary[f"{prefix}_mean_total_ms"]
    summary[f"{prefix}_fps"] = float(1000.0 / mean_total) if mean_total > 0 else 0.0
    totals = np.asarray(data["total_ms"], dtype=np.float32)
    if totals.size > 0:
        summary[f"{prefix}_frame0_total_ms"] = float(totals[0])
    steady = totals[1:] if totals.size > 1 else totals
    if steady.size > 0:
        summary[f"{prefix}_steady_mean_total_ms"] = float(steady.mean())
        summary[f"{prefix}_steady_median_total_ms"] = float(np.median(steady))
        steady_mean = summary[f"{prefix}_steady_mean_total_ms"]
        summary[f"{prefix}_steady_fps"] = float(1000.0 / steady_mean) if steady_mean > 0 else 0.0
    return summary


def run_compare(
    *,
    video_path: str,
    onnx_dir: Path,
    checkpoint: str,
    sam3_repo: Path,
    prompt_spec,
    outdir: Path,
    max_frames: int,
    safe: bool,
    onnx_accel: str,
    onnx_max_mem_frames: int,
    onnx_max_obj_ptrs: int,
    run_order: list[str],
    cooldown_sec: float,
    prompt_json_path: Path | None = None,
):
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    video = _load_video_frames(video_path, max_frames)
    prompt_json = (
        Path(prompt_json_path).resolve() if prompt_json_path else outdir / "prompt.json"
    )
    _save_prompt_spec(prompt_json, prompt_spec)

    native_npz = outdir / "native.npz"
    onnx_npz = outdir / "onnx.npz"
    summary_json = outdir / "summary.json"

    print(f"[INFO] Output dir: {outdir}")
    print(f"[INFO] Run order     : {','.join(run_order)}")
    if cooldown_sec > 0.0:
        print(f"[INFO] Cooldown sec  : {cooldown_sec:.1f}")

    for step_idx, target in enumerate(run_order):
        if target == "onnx":
            print("[INFO] Running ONNX tracker in isolated subprocess...")
            _run_onnx_subprocess(
                video_path=video_path,
                onnx_dir=Path(onnx_dir).resolve(),
                prompt_json=prompt_json,
                save_path=onnx_npz,
                max_frames=len(video.raw_frames),
                safe=safe,
                onnx_accel=onnx_accel,
                onnx_max_mem_frames=onnx_max_mem_frames,
                onnx_max_obj_ptrs=onnx_max_obj_ptrs,
            )
        else:
            print("[INFO] Running native PyTorch tracker in isolated subprocess...")
            _run_native_subprocess(
                video_path=video_path,
                checkpoint=str(Path(checkpoint).resolve()),
                sam3_repo=Path(sam3_repo).resolve(),
                prompt_json=prompt_json,
                save_path=native_npz,
                max_frames=len(video.raw_frames),
            )
        if cooldown_sec > 0.0 and step_idx + 1 < len(run_order):
            print(f"[INFO] Cooling down for {cooldown_sec:.1f}s before the next run...")
            time.sleep(cooldown_sec)

    native = _load_npz(native_npz)
    onnx = _load_npz(onnx_npz)
    onnx_runtime = _decode_json_scalar(onnx, "runtime_json")
    native_frame_count = len(native["masks"])
    onnx_frame_count = len(onnx["masks"])
    frame_count = min(native_frame_count, onnx_frame_count)
    frame_summaries = [
        _frame_metrics(native["masks"][idx], onnx["masks"][idx]) for idx in range(frame_count)
    ]
    if frame_summaries:
        mean_iou = float(np.mean([item["iou"] for item in frame_summaries]))
        min_iou = float(np.min([item["iou"] for item in frame_summaries]))
        mean_dice = float(np.mean([item["dice"] for item in frame_summaries]))
        mean_pixel_acc = float(np.mean([item["pixel_acc"] for item in frame_summaries]))
        worst_frame_idx = int(np.argmin([item["iou"] for item in frame_summaries]))
        worst_frame = {"frame_idx": worst_frame_idx, **frame_summaries[worst_frame_idx]}
    else:
        mean_iou = 0.0
        min_iou = 0.0
        mean_dice = 0.0
        mean_pixel_acc = 0.0
        worst_frame = None

    summary = {
        "video": str(video_path),
        "frame_count": frame_count,
        "native_frame_count": native_frame_count,
        "onnx_frame_count": onnx_frame_count,
        "prompt": prompt_spec,
        "run_order": list(run_order),
        "cooldown_sec": float(cooldown_sec),
        "onnx_mode": onnx_runtime.get("mode", "default") if isinstance(onnx_runtime, dict) else "default",
        "onnx_max_mem_frames": int(onnx_max_mem_frames),
        "onnx_max_obj_ptrs": int(onnx_max_obj_ptrs),
        "mean_iou": mean_iou,
        "min_iou": min_iou,
        "mean_dice": mean_dice,
        "mean_pixel_acc": mean_pixel_acc,
        "worst_frame": worst_frame,
        "frame_metrics": frame_summaries,
    }
    if onnx_runtime is not None:
        summary["onnx_runtime"] = onnx_runtime
    summary.update(_summarize_timings("native", native))
    summary.update(_summarize_timings("onnx", onnx))

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "summary": summary,
        "summary_json": summary_json,
        "prompt_json": prompt_json,
        "native_npz": native_npz,
        "onnx_npz": onnx_npz,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare native SAM3 tracker propagation against the ONNX wrapper."
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument(
        "--onnx_dir",
        default=str(DEFAULT_ONNX_DIR),
        help="Directory containing the exported ONNX tracker files.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CKPT),
        help="Path to the SAM3 checkpoint.",
    )
    parser.add_argument(
        "--sam3_repo",
        default=str(DEFAULT_SAM3_REPO),
        help="Path to the local SAM3 repo.",
    )
    parser.add_argument(
        "--prompt",
        default="seed_points",
        choices=["seed_points", "bounding_box"],
    )
    parser.add_argument("--points", default="", help="Prompt points as x,y,label;x,y,label")
    parser.add_argument("--box", default="", help="Prompt box as x1,y1,x2,y2")
    parser.add_argument(
        "--prompt_json",
        default="",
        help="Optional prompt JSON to replay. Supports a single prompt object or a multi-frame 'annotations' list.",
    )
    parser.add_argument("--save_prompt_json", default="", help="Optional output path for the prompt JSON.")
    parser.add_argument("--max_frames", type=int, default=20, help="Number of frames to compare.")
    parser.add_argument(
        "--onnx_accel",
        default=os.getenv("SAM3_ORT_ACCEL", "cuda"),
        choices=["auto", "cpu", "cuda", "trt"],
        help="Execution provider choice for the ONNX subprocess.",
    )
    parser.add_argument(
        "--onnx_max_mem_frames",
        type=int,
        default=None,
        help="Optional ONNX spatial-memory cap. When omitted, the runtime auto-selects 2 for single-annotation and 4 for multi-annotation.",
    )
    parser.add_argument(
        "--onnx_max_obj_ptrs",
        type=int,
        default=None,
        help="Optional ONNX object-pointer cap. Defaults to 16 when omitted.",
    )
    parser.add_argument(
        "--outdir",
        default="",
        help="Optional output directory for the benchmark dumps and summary.",
    )
    parser.add_argument(
        "--safe",
        action="store_true",
        help="Disable ORT graph optimizations in the ONNX subprocess.",
    )
    parser.add_argument(
        "--run_order",
        default=DEFAULT_RUN_ORDER,
        help="Comma-separated execution order for the isolated runs: onnx,native or native,onnx.",
    )
    parser.add_argument(
        "--cooldown_sec",
        type=float,
        default=0.0,
        help="Optional sleep inserted between the isolated ONNX/native runs.",
    )
    parser.add_argument(
        "--worker_mode",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker_save_npz",
        default="",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.worker_mode:
        if args.worker_mode != "native":
            raise SystemExit(f"Unsupported worker mode: {args.worker_mode}")
        if not args.prompt_json:
            raise SystemExit("--worker_mode native requires --prompt_json.")
        if not args.worker_save_npz:
            raise SystemExit("--worker_mode native requires --worker_save_npz.")
        prompt_spec = _load_prompt_spec(Path(args.prompt_json).resolve())
        video = _load_video_frames(args.video, args.max_frames)
        _run_native_tracker(
            video=video,
            prompt_spec=prompt_spec,
            checkpoint=str(Path(args.checkpoint).resolve()),
            sam3_repo=Path(args.sam3_repo).resolve(),
            save_path=Path(args.worker_save_npz).resolve(),
            video_path=args.video,
        )
        return

    outdir = Path(args.outdir).resolve() if args.outdir else Path(
        tempfile.mkdtemp(prefix="sam3_compare_", dir=str(REPO_ROOT / "checkpoints" / "sam3"))
    )
    outdir.mkdir(parents=True, exist_ok=True)

    video = _load_video_frames(args.video, args.max_frames)
    prompt_spec = _resolve_prompt(args, video.raw_frames[0])
    onnx_max_mem_frames, onnx_max_obj_ptrs = resolve_onnx_runtime_caps(
        prompt_spec=prompt_spec,
        max_mem_frames=args.onnx_max_mem_frames,
        max_obj_ptrs=args.onnx_max_obj_ptrs,
    )
    run_order = _parse_run_order(args.run_order)
    cooldown_sec = max(0.0, float(args.cooldown_sec))
    run = run_compare(
        video_path=args.video,
        onnx_dir=Path(args.onnx_dir).resolve(),
        checkpoint=str(Path(args.checkpoint).resolve()),
        sam3_repo=Path(args.sam3_repo).resolve(),
        prompt_spec=prompt_spec,
        outdir=outdir,
        max_frames=len(video.raw_frames),
        safe=args.safe,
        onnx_accel=args.onnx_accel,
        onnx_max_mem_frames=onnx_max_mem_frames,
        onnx_max_obj_ptrs=onnx_max_obj_ptrs,
        run_order=run_order,
        cooldown_sec=cooldown_sec,
        prompt_json_path=Path(args.save_prompt_json).resolve() if args.save_prompt_json else None,
    )
    summary = run["summary"]
    summary_json = run["summary_json"]
    native_npz = run["native_npz"]
    onnx_npz = run["onnx_npz"]
    mean_iou = summary["mean_iou"]
    min_iou = summary["min_iou"]
    mean_dice = summary["mean_dice"]
    mean_pixel_acc = summary["mean_pixel_acc"]

    print(f"[INFO] Mean IoU      : {mean_iou:.4f}")
    print(f"[INFO] Min IoU       : {min_iou:.4f}")
    print(f"[INFO] Mean Dice     : {mean_dice:.4f}")
    print(f"[INFO] Mean PixelAcc : {mean_pixel_acc:.4f}")
    runtime = summary.get("onnx_runtime", {})
    if isinstance(runtime, dict):
        print(
            f"[INFO] ONNX runtime  : mode={runtime.get('mode', 'default')} "
            f"graph={runtime.get('graph_profile', 'default')} "
            f"iobinding={'on' if runtime.get('uses_iobinding') else 'off'}"
        )
    print(
        f"[INFO] Native total  : {summary['native_mean_total_ms']:.1f} ms/frame "
        f"({summary['native_fps']:.2f} fps) | steady={summary.get('native_steady_mean_total_ms', summary['native_mean_total_ms']):.1f} ms"
    )
    print(
        f"[INFO] Native stage  : prep={summary.get('native_mean_prep_ms', 0.0):.1f} "
        f"enc={summary['native_mean_enc_ms']:.1f} "
        f"attn={summary['native_mean_attn_ms']:.1f} dec={summary['native_mean_dec_ms']:.1f} "
        f"mem={summary['native_mean_mem_ms']:.1f}"
    )
    print(
        f"[INFO] ONNX total    : {summary['onnx_mean_total_ms']:.1f} ms/frame "
        f"({summary['onnx_fps']:.2f} fps) | steady={summary.get('onnx_steady_mean_total_ms', summary['onnx_mean_total_ms']):.1f} ms"
    )
    print(
        f"[INFO] ONNX stage    : prep={summary.get('onnx_mean_prep_ms', 0.0):.1f} "
        f"enc={summary['onnx_mean_enc_ms']:.1f} "
        f"attn={summary['onnx_mean_attn_ms']:.1f} dec={summary['onnx_mean_dec_ms']:.1f} "
        f"mem={summary['onnx_mean_mem_ms']:.1f}"
    )
    print(f"[INFO] Summary JSON  : {summary_json}")
    print(f"[INFO] Native dump   : {native_npz}")
    print(f"[INFO] ONNX dump     : {onnx_npz}")


if __name__ == "__main__":
    main()
