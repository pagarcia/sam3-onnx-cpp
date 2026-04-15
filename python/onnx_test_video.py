# sam3-onnx-cpp/python/onnx_test_video.py
#!/usr/bin/env python3
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from onnx_test_utils import (
    PrepInfo,
    compute_display_base,
    green_overlay,
    make_session,
    preprocess_image_bgr,
    print_system_info,
    set_cv2_threads,
)
from PyQt5 import QtWidgets

NUM_MASKMEM = 7
MAX_OBJ_PTRS = 16


def _as_f32c(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def _as_i32c(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr.astype(np.int32, copy=False))


def _empty_prompt():
    return np.zeros((1, 0, 2), np.float32), np.zeros((1, 0), np.int32)


def _prepare_prompt_points(points_xy, labels, info: PrepInfo):
    if not points_xy:
        return _empty_prompt()
    points = np.asarray(points_xy, dtype=np.float32)
    point_labels = np.asarray(labels, dtype=np.int32)
    points[:, 0] *= info.scale_x
    points[:, 1] *= info.scale_y
    return np.ascontiguousarray(points[None, ...]), np.ascontiguousarray(point_labels[None, ...])


def _prepare_prompt_box(rect_xyxy, info: PrepInfo):
    if rect_xyxy is None:
        return _empty_prompt()
    x1, y1, x2, y2 = rect_xyxy
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
    points[:, 0] *= info.scale_x
    points[:, 1] *= info.scale_y
    labels = np.array([2, 3], dtype=np.int32)
    return np.ascontiguousarray(points[None, ...]), np.ascontiguousarray(labels[None, ...])


def _run_encoder(session, pixel_values):
    input_name = session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]
    values = session.run(None, {input_name: _as_f32c(pixel_values)})
    return dict(zip(output_names, values))


def _normalize_encoder_outputs(raw_outputs, constants):
    if "image_embeddings" in raw_outputs:
        return raw_outputs

    current_vision_feat = raw_outputs["image_embeddings.2"]
    return {
        "image_embeddings": current_vision_feat + constants["no_mem_embed_bchw"],
        "high_res_features0": raw_outputs["image_embeddings.0"],
        "high_res_features1": raw_outputs["image_embeddings.1"],
        "current_vision_feat": current_vision_feat,
        "current_vision_pos_embed": constants["current_vision_pos_embed"],
    }


def _run_decoder(session, point_coords, point_labels, image_embed, high_res_0, high_res_1):
    if point_coords is None or point_labels is None:
        point_coords, point_labels = _empty_prompt()

    feed = {
        "point_coords": _as_f32c(point_coords),
        "point_labels": _as_i32c(point_labels),
        "image_embed": _as_f32c(image_embed),
        "high_res_feats_0": _as_f32c(high_res_0),
        "high_res_feats_1": _as_f32c(high_res_1),
    }
    values = session.run(None, feed)
    names = [out.name for out in session.get_outputs()]
    return dict(zip(names, values))


def _run_memory_attention(
    session,
    current_vision_feat,
    current_vision_pos_embed,
    memory_obj_ptrs,
    memory_obj_tpos,
    memory_mask_feats,
    memory_mask_pos,
    memory_mask_tpos_idx,
):
    feed = {
        "current_vision_feat": _as_f32c(current_vision_feat),
        "current_vision_pos_embed": _as_f32c(current_vision_pos_embed),
        "memory_obj_ptrs": _as_f32c(memory_obj_ptrs),
        "memory_obj_tpos": _as_f32c(memory_obj_tpos),
        "memory_mask_feats": _as_f32c(memory_mask_feats),
        "memory_mask_pos": _as_f32c(memory_mask_pos),
        "memory_mask_tpos_idx": np.ascontiguousarray(
            memory_mask_tpos_idx.astype(np.int64, copy=False)
        ),
    }
    return session.run(None, feed)[0]


def _run_memory_encoder(
    session,
    pred_mask_high_res,
    current_vision_feat,
    object_score_logits,
    is_mask_from_points,
):
    feed = {
        "pred_mask_high_res": _as_f32c(pred_mask_high_res),
        "current_vision_feat": _as_f32c(current_vision_feat),
        "object_score_logits": _as_f32c(object_score_logits),
        "is_mask_from_points": np.ascontiguousarray(
            np.array([1.0 if is_mask_from_points else 0.0], dtype=np.float32)
        ),
    }
    values = session.run(None, feed)
    names = [out.name for out in session.get_outputs()]
    return dict(zip(names, values))


def _mask_to_overlay(frame_bgr: np.ndarray, mask_logits_high_res: np.ndarray, info: PrepInfo):
    mask_uint8 = _mask_to_uint8(mask_logits_high_res, info)
    return green_overlay(frame_bgr, mask_uint8, alpha=0.5)


def _mask_to_uint8(mask_logits_high_res: np.ndarray, info: PrepInfo) -> np.ndarray:
    mask_logits = mask_logits_high_res[0, 0]
    mask_resized = cv2.resize(
        mask_logits,
        (info.orig_hw[1], info.orig_hw[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return (mask_resized > 0.0).astype(np.uint8) * 255


def _parse_points_text(text: str):
    points, labels = [], []
    if not text.strip():
        return points, labels
    for item in text.split(";"):
        x_str, y_str, label_str = [part.strip() for part in item.split(",")]
        points.append((int(float(x_str)), int(float(y_str))))
        labels.append(int(label_str))
    return points, labels


def _parse_box_text(text: str):
    parts = [int(float(part.strip())) for part in text.split(",")]
    if len(parts) != 4:
        raise ValueError("--box expects x1,y1,x2,y2")
    return tuple(parts)


def _load_prompt_spec(path: Path):
    with path.open("r", encoding="utf-8") as f:
        spec = json.load(f)
    if "prompt" not in spec:
        raise SystemExit(f"Prompt JSON is missing 'prompt': {path}")
    return spec


def _save_prompt_spec(path: Path, spec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)


def _interactive_select_points(first_bgr, sess_enc, sess_dec, constants):
    pixel_values, info = preprocess_image_bgr(first_bgr, target_size=1008)
    enc = _normalize_encoder_outputs(_run_encoder(sess_enc, pixel_values), constants)

    base, scale = compute_display_base(first_bgr, max_side=1200)
    points, labels = [], []

    def render(mask_logits=None):
        vis = base.copy()
        if mask_logits is not None:
            overlay = _mask_to_overlay(first_bgr, mask_logits, info)
            vis = cv2.resize(overlay, (base.shape[1], base.shape[0]))
        for idx, (px, py) in enumerate(points):
            color = (0, 0, 255) if labels[idx] == 1 else (255, 0, 0)
            cv2.circle(vis, (int(px * scale), int(py * scale)), 6, color, -1)
        cv2.imshow("SAM3 ONNX Video", vis)

    def update_preview():
        if not points:
            render()
            return
        prompt_points, prompt_labels = _prepare_prompt_points(points, labels, info)
        out = _run_decoder(
            sess_dec,
            prompt_points,
            prompt_labels,
            enc["image_embeddings"],
            enc["high_res_features0"],
            enc["high_res_features1"],
        )
        render(out["pred_mask_high_res"])

    def mouse_cb(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x / scale), int(y / scale)))
            labels.append(1)
            update_preview()
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append((int(x / scale), int(y / scale)))
            labels.append(0)
            update_preview()
        elif event == cv2.EVENT_MBUTTONDOWN:
            points.clear()
            labels.clear()
            update_preview()

    cv2.namedWindow("SAM3 ONNX Video")
    cv2.setMouseCallback("SAM3 ONNX Video", mouse_cb)
    update_preview()
    print("[INFO] L-click=FG, R-click=BG, M-click=reset. Press Enter or ESC when done.")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 27):
            break
    cv2.destroyAllWindows()

    prompt_points, prompt_labels = _prepare_prompt_points(points, labels, info)
    prompt_spec = {
        "prompt": "seed_points",
        "points": [[int(px), int(py), int(label)] for (px, py), label in zip(points, labels)],
    }
    return enc, info, prompt_points, prompt_labels, prompt_spec


def _interactive_select_box(first_bgr, sess_enc, sess_dec, constants):
    pixel_values, info = preprocess_image_bgr(first_bgr, target_size=1008)
    enc = _normalize_encoder_outputs(_run_encoder(sess_enc, pixel_values), constants)

    base, scale = compute_display_base(first_bgr, max_side=1200)
    rect_start = None
    rect_end = None
    drawing = False

    def current_box():
        if rect_start is None or rect_end is None:
            return None
        x1d, y1d = rect_start
        x2d, y2d = rect_end
        x1 = int(x1d / scale)
        y1 = int(y1d / scale)
        x2 = int(x2d / scale)
        y2 = int(y2d / scale)
        return (x1, y1, x2, y2)

    def render(mask_logits=None):
        vis = base.copy()
        if mask_logits is not None:
            overlay = _mask_to_overlay(first_bgr, mask_logits, info)
            vis = cv2.resize(overlay, (base.shape[1], base.shape[0]))
        if rect_start is not None and rect_end is not None:
            cv2.rectangle(vis, rect_start, rect_end, (0, 255, 255), 2)
        cv2.imshow("SAM3 ONNX Video", vis)

    def update_preview():
        box = current_box()
        if box is None:
            render()
            return
        prompt_points, prompt_labels = _prepare_prompt_box(box, info)
        out = _run_decoder(
            sess_dec,
            prompt_points,
            prompt_labels,
            enc["image_embeddings"],
            enc["high_res_features0"],
            enc["high_res_features1"],
        )
        render(out["pred_mask_high_res"])

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
            update_preview()
        elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK):
            rect_start = None
            rect_end = None
            drawing = False
            render()

    cv2.namedWindow("SAM3 ONNX Video")
    cv2.setMouseCallback("SAM3 ONNX Video", mouse_cb)
    render()
    print("[INFO] Drag a box on the first frame. Press Enter or ESC when done.")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 27):
            break
    cv2.destroyAllWindows()

    box = current_box()
    prompt_points, prompt_labels = _prepare_prompt_box(box, info)
    prompt_spec = {
        "prompt": "bounding_box",
        "box": list(box) if box is not None else None,
    }
    return enc, info, prompt_points, prompt_labels, prompt_spec


def _prepare_prompt_from_spec(first_bgr, sess_enc, constants, prompt_spec):
    pixel_values, info = preprocess_image_bgr(first_bgr, target_size=1008)
    enc = _normalize_encoder_outputs(_run_encoder(sess_enc, pixel_values), constants)
    prompt_kind = prompt_spec["prompt"]
    if prompt_kind == "bounding_box":
        box = prompt_spec.get("box")
        prompt_points, prompt_labels = _prepare_prompt_box(tuple(box) if box is not None else None, info)
    elif prompt_kind == "seed_points":
        raw_points = prompt_spec.get("points", [])
        points = [(int(item[0]), int(item[1])) for item in raw_points]
        labels = [int(item[2]) for item in raw_points]
        prompt_points, prompt_labels = _prepare_prompt_points(points, labels, info)
    else:
        raise SystemExit(f"Unsupported prompt kind in prompt JSON: {prompt_kind}")
    return enc, info, prompt_points, prompt_labels


def _select_memory_inputs(frame_idx, cond_states, non_cond_states):
    spatial_items = []
    if 0 in cond_states and frame_idx > 0:
        spatial_items.append((cond_states[0], NUM_MASKMEM - 1))

    for t_pos in range(1, NUM_MASKMEM):
        prev_idx = frame_idx - (NUM_MASKMEM - t_pos)
        if prev_idx < 0:
            continue
        state = non_cond_states.get(prev_idx)
        if state is None:
            continue
        spatial_items.append((state, NUM_MASKMEM - t_pos - 1))

    if not spatial_items:
        raise RuntimeError("No spatial memory was available for memory attention.")

    memory_mask_feats = np.concatenate(
        [item[0]["maskmem_features"] for item in spatial_items], axis=0
    )
    memory_mask_pos = np.concatenate(
        [item[0]["maskmem_pos_enc"] for item in spatial_items], axis=0
    )
    memory_mask_tpos_idx = np.array([item[1] for item in spatial_items], dtype=np.int64)

    pointer_items = []
    if 0 in cond_states and frame_idx > 0:
        pointer_items.append((cond_states[0], float(frame_idx)))
    for t_diff in range(1, MAX_OBJ_PTRS):
        prev_idx = frame_idx - t_diff
        if prev_idx < 0:
            break
        state = non_cond_states.get(prev_idx)
        if state is not None:
            pointer_items.append((state, float(t_diff)))

    if pointer_items:
        memory_obj_ptrs = np.concatenate([item[0]["obj_ptr"] for item in pointer_items], axis=0)
        memory_obj_tpos = np.array([item[1] for item in pointer_items], dtype=np.float32)
    else:
        memory_obj_ptrs = np.zeros((0, 256), np.float32)
        memory_obj_tpos = np.zeros((0,), np.float32)

    return {
        "memory_obj_ptrs": memory_obj_ptrs,
        "memory_obj_tpos": memory_obj_tpos,
        "memory_mask_feats": memory_mask_feats,
        "memory_mask_pos": memory_mask_pos,
        "memory_mask_tpos_idx": memory_mask_tpos_idx,
    }


def _resolve_video(args):
    if args.video:
        return args.video

    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QtWidgets.QApplication(sys.argv)

    video_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None,
        "Select Video",
        "",
        "Video files (*.mp4 *.mkv *.avi *.mov *.m4v);;All files (*.*)",
    )
    if owns_app:
        app.quit()
    if not video_path:
        raise SystemExit("No video selected.")
    return video_path


def _load_video_constants(onnx_dir: Path):
    constants_path = onnx_dir / "video_constants.npz"
    if not constants_path.exists():
        raise SystemExit(
            f"Missing {constants_path}. Run export\\onnx_export.py first to generate the video constants bundle."
        )

    with np.load(constants_path) as data:
        constants = {key: np.ascontiguousarray(data[key]) for key in data.files}

    global NUM_MASKMEM, MAX_OBJ_PTRS
    NUM_MASKMEM = int(constants.get("num_maskmem", np.array([NUM_MASKMEM]))[0])
    MAX_OBJ_PTRS = int(constants.get("max_obj_ptrs", np.array([MAX_OBJ_PTRS]))[0])
    return constants


def _resolve_encoder_path(onnx_dir: Path) -> Path:
    custom_encoder = onnx_dir / "image_encoder.onnx"
    if custom_encoder.exists():
        return custom_encoder

    repo_root = Path(__file__).resolve().parent.parent
    shared_dir = repo_root / "checkpoints" / "sam3" / "onnx"
    for candidate in (
        shared_dir / "vision_encoder.onnx",
        shared_dir / "vision_encoder_fp16.onnx",
    ):
        # The bundled encoder models use external data sidecars. Skip any
        # half-downloaded .onnx file so we can fall back to a complete variant.
        sidecar = candidate.with_name(candidate.name + "_data")
        if candidate.exists() and sidecar.exists():
            return candidate

    raise SystemExit(
        "Could not find a complete encoder ONNX pair. Expected image_encoder.onnx in the video export dir or vision_encoder*.onnx plus .onnx_data under checkpoints/sam3/onnx."
    )


def main():
    print_system_info()
    set_cv2_threads(1)

    parser = argparse.ArgumentParser(
        description="SAM3 tracker ONNX video demo for point/box propagation."
    )
    parser.add_argument("--video", default="", help="Optional input video path.")
    parser.add_argument(
        "--onnx_dir",
        default=str(Path(__file__).resolve().parent.parent / "checkpoints" / "sam3" / "video_onnx"),
        help="Directory containing image_encoder.onnx, image_decoder.onnx, memory_attention.onnx, memory_encoder.onnx.",
    )
    parser.add_argument(
        "--prompt",
        default="seed_points",
        choices=["seed_points", "bounding_box"],
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Optional frame limit. 0 processes the entire video.",
    )
    parser.add_argument(
        "--max_mem_frames",
        type=int,
        default=0,
        help="Optional cap on the number of spatial memory frames used by ONNX attention.",
    )
    parser.add_argument(
        "--max_obj_ptrs",
        type=int,
        default=0,
        help="Optional cap on the number of object pointers used by ONNX attention.",
    )
    parser.add_argument(
        "--points",
        default="",
        help="Noninteractive prompt points as x,y,label;x,y,label",
    )
    parser.add_argument(
        "--box",
        default="",
        help="Noninteractive box prompt as x1,y1,x2,y2",
    )
    parser.add_argument(
        "--prompt_json",
        default="",
        help="Optional JSON file describing the prompt to replay.",
    )
    parser.add_argument(
        "--save_prompt_json",
        default="",
        help="Optional JSON file where the chosen prompt will be written.",
    )
    parser.add_argument(
        "--save_npz",
        default="",
        help="Optional .npz output with frame masks and timings for benchmarking.",
    )
    parser.add_argument(
        "--no_output_video",
        action="store_true",
        help="Skip writing the overlay video and only run inference.",
    )
    parser.add_argument(
        "--safe",
        action="store_true",
        help="Disable ORT graph optimizations for all sessions.",
    )
    args = parser.parse_args()

    video_path = _resolve_video(args)
    print(f"[INFO] Video: {video_path}")

    onnx_dir = Path(args.onnx_dir).resolve()
    constants = _load_video_constants(onnx_dir)
    global NUM_MASKMEM, MAX_OBJ_PTRS
    if args.max_mem_frames > 0:
        NUM_MASKMEM = max(1, min(NUM_MASKMEM, int(args.max_mem_frames)))
    if args.max_obj_ptrs > 0:
        MAX_OBJ_PTRS = max(1, min(MAX_OBJ_PTRS, int(args.max_obj_ptrs)))
    enc_path = _resolve_encoder_path(onnx_dir)
    dec_path = onnx_dir / "image_decoder.onnx"
    mat_path = onnx_dir / "memory_attention.onnx"
    men_path = onnx_dir / "memory_encoder.onnx"
    for path in (enc_path, dec_path, mat_path, men_path):
        if not path.exists():
            raise SystemExit(f"Missing ONNX file: {path}")

    sess_enc = make_session(str(enc_path), tag="video_image_encoder", safe=args.safe)
    sess_dec = make_session(str(dec_path), tag="video_image_decoder", safe=args.safe)
    sess_mat = make_session(str(mat_path), tag="video_memory_attention", safe=args.safe)
    sess_men = make_session(str(men_path), tag="video_memory_encoder", safe=args.safe)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit("Could not open the selected video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ok, first_frame = cap.read()
    if not ok:
        raise SystemExit("The selected video is empty.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    prompt_spec = None
    if args.prompt_json:
        prompt_spec = _load_prompt_spec(Path(args.prompt_json).resolve())
        enc0, info0, init_points, init_labels = _prepare_prompt_from_spec(
            first_frame, sess_enc, constants, prompt_spec
        )
    elif args.box:
        prompt_spec = {"prompt": "bounding_box", "box": list(_parse_box_text(args.box))}
        enc0, info0, init_points, init_labels = _prepare_prompt_from_spec(
            first_frame, sess_enc, constants, prompt_spec
        )
    elif args.points:
        points, labels = _parse_points_text(args.points)
        prompt_spec = {
            "prompt": "seed_points",
            "points": [[int(px), int(py), int(label)] for (px, py), label in zip(points, labels)],
        }
        enc0, info0, init_points, init_labels = _prepare_prompt_from_spec(
            first_frame, sess_enc, constants, prompt_spec
        )
    elif args.prompt == "bounding_box":
        enc0, info0, init_points, init_labels, prompt_spec = _interactive_select_box(
            first_frame, sess_enc, sess_dec, constants
        )
    else:
        enc0, info0, init_points, init_labels, prompt_spec = _interactive_select_points(
            first_frame, sess_enc, sess_dec, constants
        )

    if args.save_prompt_json:
        _save_prompt_spec(Path(args.save_prompt_json).resolve(), prompt_spec)

    cond_states = {}
    non_cond_states = {}

    output_path = str(Path(video_path).with_name(Path(video_path).stem + "_sam3_onnx_overlay.mkv"))
    writer = None
    if not args.no_output_video:
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"XVID"),
            fps if fps > 0 else 25.0,
            (orig_width, orig_height),
        )
        if not writer.isOpened():
            raise SystemExit("Could not create the output video writer.")

    saved_masks = []
    saved_enc_ms = []
    saved_attn_ms = []
    saved_dec_ms = []
    saved_mem_ms = []
    saved_total_ms = []

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break
        t_total = time.time()

        if frame_idx == 0:
            enc = enc0
            info = info0
            fused_embed = enc["image_embeddings"]
            prompt_points = init_points
            prompt_labels = init_labels
            is_mask_from_points = True
            attn_ms = 0.0
            enc_ms = 0.0
        else:
            pixel_values, info = preprocess_image_bgr(frame_bgr, target_size=1008)
            t_enc = time.time()
            enc = _normalize_encoder_outputs(_run_encoder(sess_enc, pixel_values), constants)
            enc_ms = (time.time() - t_enc) * 1000.0

            mem_inputs = _select_memory_inputs(frame_idx, cond_states, non_cond_states)
            t_attn = time.time()
            fused_embed = _run_memory_attention(
                sess_mat,
                enc["current_vision_feat"],
                enc["current_vision_pos_embed"],
                mem_inputs["memory_obj_ptrs"],
                mem_inputs["memory_obj_tpos"],
                mem_inputs["memory_mask_feats"],
                mem_inputs["memory_mask_pos"],
                mem_inputs["memory_mask_tpos_idx"],
            )
            attn_ms = (time.time() - t_attn) * 1000.0
            prompt_points, prompt_labels = _empty_prompt()
            is_mask_from_points = False

        t_dec = time.time()
        dec = _run_decoder(
            sess_dec,
            prompt_points,
            prompt_labels,
            fused_embed,
            enc["high_res_features0"],
            enc["high_res_features1"],
        )
        dec_ms = (time.time() - t_dec) * 1000.0

        t_mem = time.time()
        mem = _run_memory_encoder(
            sess_men,
            dec["pred_mask_high_res"],
            enc["current_vision_feat"],
            dec["object_score_logits"],
            is_mask_from_points=is_mask_from_points,
        )
        mem_ms = (time.time() - t_mem) * 1000.0

        state = {
            "maskmem_features": mem["maskmem_features"],
            "maskmem_pos_enc": mem["maskmem_pos_enc"],
            "obj_ptr": dec["obj_ptr"],
            "pred_mask_high_res": dec["pred_mask_high_res"],
            "object_score_logits": dec["object_score_logits"],
        }
        if frame_idx == 0:
            cond_states[frame_idx] = state
        else:
            non_cond_states[frame_idx] = state

        mask_uint8 = _mask_to_uint8(dec["pred_mask_high_res"], info)
        if writer is not None:
            writer.write(green_overlay(frame_bgr, mask_uint8, alpha=0.5))

        saved_masks.append(mask_uint8)
        saved_enc_ms.append(enc_ms)
        saved_attn_ms.append(attn_ms)
        saved_dec_ms.append(dec_ms)
        saved_mem_ms.append(mem_ms)
        saved_total_ms.append((time.time() - t_total) * 1000.0)

        if frame_idx == 0:
            print(
                f"Frame {frame_idx:03d} | Dec:{dec_ms:.1f} ms | MemEnc:{mem_ms:.1f} ms"
            )
        else:
            print(
                f"Frame {frame_idx:03d} | Enc:{enc_ms:.1f} ms | Attn:{attn_ms:.1f} ms | Dec:{dec_ms:.1f} ms | MemEnc:{mem_ms:.1f} ms"
            )

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[INFO] Wrote {frame_idx} frames to {output_path}")
    else:
        print(f"[INFO] Processed {frame_idx} frames without writing a video.")

    if args.save_npz:
        save_path = Path(args.save_npz).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            masks=np.stack(saved_masks).astype(np.uint8, copy=False),
            enc_ms=np.asarray(saved_enc_ms, dtype=np.float32),
            attn_ms=np.asarray(saved_attn_ms, dtype=np.float32),
            dec_ms=np.asarray(saved_dec_ms, dtype=np.float32),
            mem_ms=np.asarray(saved_mem_ms, dtype=np.float32),
            total_ms=np.asarray(saved_total_ms, dtype=np.float32),
            prompt_json=np.array(json.dumps(prompt_spec)),
            video_path=np.array(str(video_path)),
        )
        print(f"[INFO] Saved benchmark dump: {save_path}")


if __name__ == "__main__":
    main()
