#!/usr/bin/env python3
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt5 import QtWidgets

from onnx_test_utils import compute_display_base, green_overlay, print_system_info, set_cv2_threads
from sam3_onnx_session import (
    FAST_DEFAULT_MAX_MEM_FRAMES,
    FAST_DEFAULT_MAX_OBJ_PTRS,
    Sam3OnnxTrackerSession,
    VARIANT_PRESET_CAPS,
    load_prompt_spec,
    mask_to_overlay,
    parse_box_text,
    parse_points_text,
    prepare_prompt_box,
    prepare_prompt_points,
    save_prompt_spec,
)


def _resolve_runtime_caps(args) -> tuple[str, int, int]:
    preset = args.preset.strip().lower()
    max_mem_frames = args.max_mem_frames
    max_obj_ptrs = args.max_obj_ptrs
    if preset:
        caps = VARIANT_PRESET_CAPS[preset]
        if max_mem_frames is None:
            max_mem_frames = caps["max_mem_frames"]
        if max_obj_ptrs is None:
            max_obj_ptrs = caps["max_obj_ptrs"]
    if max_mem_frames is None:
        max_mem_frames = FAST_DEFAULT_MAX_MEM_FRAMES
    if max_obj_ptrs is None:
        max_obj_ptrs = FAST_DEFAULT_MAX_OBJ_PTRS
    return preset, int(max_mem_frames), int(max_obj_ptrs)


def _interactive_select_points(first_bgr, tracker: Sam3OnnxTrackerSession):
    prepared = tracker.prepare_frame(first_bgr)
    base, scale = compute_display_base(first_bgr, max_side=1200)
    points, labels = [], []

    def render(mask_logits=None):
        vis = base.copy()
        if mask_logits is not None:
            overlay = mask_to_overlay(first_bgr, mask_logits, prepared.info)
            vis = cv2.resize(overlay, (base.shape[1], base.shape[0]))
        for idx, (px, py) in enumerate(points):
            color = (0, 0, 255) if labels[idx] == 1 else (255, 0, 0)
            cv2.circle(vis, (int(px * scale), int(py * scale)), 6, color, -1)
        cv2.imshow("SAM3 ONNX Video", vis)

    def update_preview():
        if not points:
            render()
            return
        prompt_points, prompt_labels = prepare_prompt_points(points, labels, prepared.info)
        render(tracker.preview_prompt_mask(prepared, prompt_points, prompt_labels))

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

    prompt_points, prompt_labels = prepare_prompt_points(points, labels, prepared.info)
    prompt_spec = {
        "prompt": "seed_points",
        "points": [[int(px), int(py), int(label)] for (px, py), label in zip(points, labels)],
    }
    return prepared, prompt_points, prompt_labels, prompt_spec


def _interactive_select_box(first_bgr, tracker: Sam3OnnxTrackerSession):
    prepared = tracker.prepare_frame(first_bgr)
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
            overlay = mask_to_overlay(first_bgr, mask_logits, prepared.info)
            vis = cv2.resize(overlay, (base.shape[1], base.shape[0]))
        if rect_start is not None and rect_end is not None:
            cv2.rectangle(vis, rect_start, rect_end, (0, 255, 255), 2)
        cv2.imshow("SAM3 ONNX Video", vis)

    def update_preview():
        box = current_box()
        if box is None:
            render()
            return
        prompt_points, prompt_labels = prepare_prompt_box(box, prepared.info)
        render(tracker.preview_prompt_mask(prepared, prompt_points, prompt_labels))

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
    prompt_points, prompt_labels = prepare_prompt_box(box, prepared.info)
    prompt_spec = {
        "prompt": "bounding_box",
        "box": list(box) if box is not None else None,
    }
    return prepared, prompt_points, prompt_labels, prompt_spec


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
        "--onnx_variant",
        default=os.getenv("SAM3_ONNX_VARIANT", "").strip(),
        help="Optional tracker ONNX variant suffix such as fp16, fast, quality, or quality_fp16.",
    )
    parser.add_argument(
        "--preset",
        default="",
        choices=["", *sorted(VARIANT_PRESET_CAPS.keys())],
        help="Optional named runtime preset. When set, it fills in missing caps and prefers the matching exported variant.",
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
        default=None,
        help="Cap on spatial memory frames used by ONNX attention. Defaults to the selected preset, or the fast preset when omitted.",
    )
    parser.add_argument(
        "--max_obj_ptrs",
        type=int,
        default=None,
        help="Cap on object pointers used by ONNX attention. Defaults to the selected preset, or the fast preset when omitted.",
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
    preset, max_mem_frames, max_obj_ptrs = _resolve_runtime_caps(args)

    tracker = Sam3OnnxTrackerSession(
        Path(args.onnx_dir),
        safe=args.safe,
        max_mem_frames=max_mem_frames,
        max_obj_ptrs=max_obj_ptrs,
        variant=args.onnx_variant,
        preset=preset,
    )
    runtime = tracker.runtime_metadata
    print(
        f"[INFO] ONNX runtime: preset={runtime['requested_preset'] or 'auto'} "
        f"variant={runtime['resolved_variant'] or 'default'} "
        f"device={runtime['device_type']}"
    )
    print(
        f"[INFO] Models: enc={runtime['model_names']['encoder']} "
        f"dec={runtime['model_names']['decoder']} "
        f"attn={runtime['model_names']['memory_attention']} "
        f"mem={runtime['model_names']['memory_encoder']}"
    )
    print(
        f"[INFO] ONNX runtime caps: max_mem_frames={tracker.num_maskmem}, "
        f"max_obj_ptrs={tracker.max_obj_ptrs}, "
        f"static_mem={tracker.static_num_mem_frames}, static_obj_ptrs={tracker.static_num_obj_ptrs}, "
        f"iobinding={'on' if tracker.uses_iobinding else 'off'}"
    )

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

    if args.prompt_json:
        prompt_spec = load_prompt_spec(Path(args.prompt_json).resolve())
        prepared0, init_points, init_labels = tracker.prepare_prompt_from_spec(first_frame, prompt_spec)
    elif args.box:
        prompt_spec = {"prompt": "bounding_box", "box": list(parse_box_text(args.box))}
        prepared0, init_points, init_labels = tracker.prepare_prompt_from_spec(first_frame, prompt_spec)
    elif args.points:
        points, labels = parse_points_text(args.points)
        prompt_spec = {
            "prompt": "seed_points",
            "points": [[int(px), int(py), int(label)] for (px, py), label in zip(points, labels)],
        }
        prepared0, init_points, init_labels = tracker.prepare_prompt_from_spec(first_frame, prompt_spec)
    elif args.prompt == "bounding_box":
        prepared0, init_points, init_labels, prompt_spec = _interactive_select_box(first_frame, tracker)
    else:
        prepared0, init_points, init_labels, prompt_spec = _interactive_select_points(first_frame, tracker)

    if args.save_prompt_json:
        save_prompt_spec(Path(args.save_prompt_json).resolve(), prompt_spec)

    tracker.reset()

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
    saved_prep_ms = []
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

        if frame_idx == 0:
            result = tracker.process_frame(
                frame_idx,
                frame_bgr,
                prepared=prepared0,
                prompt_points=init_points,
                prompt_labels=init_labels,
            )
        else:
            result = tracker.process_frame(frame_idx, frame_bgr)

        if writer is not None:
            writer.write(green_overlay(frame_bgr, result.mask_uint8, alpha=0.5))

        saved_masks.append(result.mask_uint8)
        saved_prep_ms.append(result.timings.prep_ms)
        saved_enc_ms.append(result.timings.enc_ms)
        saved_attn_ms.append(result.timings.attn_ms)
        saved_dec_ms.append(result.timings.dec_ms)
        saved_mem_ms.append(result.timings.mem_ms)
        frame_total_ms = result.timings.total_ms
        if frame_idx == 0:
            frame_total_ms += result.timings.prep_ms + result.timings.enc_ms
        saved_total_ms.append(frame_total_ms)

        if frame_idx == 0:
            print(
                f"Frame {frame_idx:03d} | Prep:{result.timings.prep_ms:.1f} ms | "
                f"Enc:{result.timings.enc_ms:.1f} ms | Dec:{result.timings.dec_ms:.1f} ms | "
                f"MemEnc:{result.timings.mem_ms:.1f} ms | Total:{frame_total_ms:.1f} ms"
            )
        else:
            print(
                f"Frame {frame_idx:03d} | Prep:{result.timings.prep_ms:.1f} ms | "
                f"Enc:{result.timings.enc_ms:.1f} ms | "
                f"Attn:{result.timings.attn_ms:.1f} ms | Dec:{result.timings.dec_ms:.1f} ms | "
                f"MemEnc:{result.timings.mem_ms:.1f} ms | Total:{frame_total_ms:.1f} ms"
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
            prep_ms=np.asarray(saved_prep_ms, dtype=np.float32),
            enc_ms=np.asarray(saved_enc_ms, dtype=np.float32),
            attn_ms=np.asarray(saved_attn_ms, dtype=np.float32),
            dec_ms=np.asarray(saved_dec_ms, dtype=np.float32),
            mem_ms=np.asarray(saved_mem_ms, dtype=np.float32),
            total_ms=np.asarray(saved_total_ms, dtype=np.float32),
            runtime_json=np.array(json.dumps(runtime)),
            prompt_json=np.array(json.dumps(prompt_spec)),
            video_path=np.array(str(video_path)),
        )
        print(f"[INFO] Saved benchmark dump: {save_path}")


if __name__ == "__main__":
    main()
