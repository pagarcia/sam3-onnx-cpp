# sam3-onnx-cpp/python/onnx_test_image.py
#!/usr/bin/env python3
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import onnxruntime as ort
from PyQt5 import QtWidgets

from onnx_test_utils import (
    compute_display_base,
    empty_boxes,
    empty_points,
    green_overlay,
    make_session,
    pick_best_mask,
    postprocess_mask_to_original,
    prepare_boxes,
    prepare_points,
    preprocess_image_bgr,
    print_system_info,
    run_decoder,
    run_encoder,
    set_cv2_threads,
)


def _resolve_image_path_macos() -> str:
    if sys.platform != "darwin":
        return ""

    script = """
try
    POSIX path of (choose file with prompt "Select an Image" of type {"public.image"})
on error number -128
    return ""
end try
"""
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _resolve_image_path(arg_value: str) -> str:
    if arg_value:
        image_path = Path(arg_value).expanduser().resolve()
        if not image_path.exists():
            sys.exit(f"ERROR: Image file does not exist: {image_path}")
        return str(image_path)

    img_path = _resolve_image_path_macos()
    if img_path:
        return img_path

    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QtWidgets.QApplication(sys.argv)
    img_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None, "Select an Image", "", "Images (*.jpg *.jpeg *.png *.bmp);;All files (*)"
    )
    if owns_app:
        app.quit()
    if not img_path:
        sys.exit("No image selected - exiting.")
    return img_path


def main():
    print_system_info()
    set_cv2_threads(1)

    ap = argparse.ArgumentParser(description="SAM3-Tracker ONNX (image) - seed points / bounding box")
    ap.add_argument("--prompt", default="seed_points", choices=["seed_points", "bounding_box"])
    ap.add_argument("--image", default="", help="Optional image path. If omitted, a file picker is shown.")
    ap.add_argument("--safe", action="store_true", help="Disable ORT graph optimizations (more conservative).")
    args = ap.parse_args()
    mode_bbox = args.prompt == "bounding_box"
    print(f"[INFO] Prompt mode: {'bounding_box' if mode_bbox else 'seed_points'}")

    img_path = _resolve_image_path(args.image)
    print(f"[INFO] Selected image: {img_path}")

    repo_root = Path(__file__).resolve().parent.parent
    onnx_dir = repo_root / "checkpoints" / "sam3" / "onnx"

    av = ort.get_available_providers()
    cuda_available = "CUDAExecutionProvider" in av

    requested = os.getenv("SAM3_ONNX_VARIANT", "").strip().lower()
    if requested not in ("fp16", "fp32"):
        accel = os.getenv("SAM3_ORT_ACCEL", "auto").strip().lower()
        requested = "fp16" if (accel == "cuda" or (accel == "auto" and cuda_available)) else "fp32"

    def pick(primary: Path, fallback: Path) -> Path:
        return primary if primary.exists() else fallback

    if requested == "fp16":
        enc_path = pick(onnx_dir / "vision_encoder_fp16.onnx", onnx_dir / "vision_encoder.onnx")
        dec_path = pick(
            onnx_dir / "prompt_encoder_mask_decoder_fp16.onnx",
            onnx_dir / "prompt_encoder_mask_decoder.onnx",
        )
    else:
        enc_path = pick(onnx_dir / "vision_encoder.onnx", onnx_dir / "vision_encoder_fp16.onnx")
        dec_path = pick(
            onnx_dir / "prompt_encoder_mask_decoder.onnx",
            onnx_dir / "prompt_encoder_mask_decoder_fp16.onnx",
        )

    if not enc_path.exists() or not dec_path.exists():
        sys.exit(
            f"ERROR: Missing encoder/decoder ONNX in {onnx_dir}\n"
            f"Tip: run .\\fetch_onnx_models.bat fp32 or fp16"
        )

    effective = "fp16" if "fp16" in enc_path.name.lower() else "fp32"
    if effective != requested:
        print(f"[WARN] Requested variant={requested} but using {effective} because of file availability.")

    print(f"[INFO] ORT providers available: {av}")
    print(f"[INFO] ONNX variant requested: {requested} | effective: {effective}")
    print(f"[INFO] Encoder ONNX: {enc_path.name}")
    print(f"[INFO] Decoder ONNX: {dec_path.name}")

    sess_enc = make_session(str(enc_path), tag="vision_encoder", safe=args.safe)
    sess_dec = make_session(str(dec_path), tag="prompt_encoder_mask_decoder", safe=args.safe)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        sys.exit("ERROR: Could not read image.")

    pixel_values, info = preprocess_image_bgr(img_bgr, target_size=1008)

    t0 = time.time()
    try:
        enc_out = run_encoder(sess_enc, pixel_values)
    except Exception as e:
        msg = str(e)
        if "no kernel image is available" in msg or "CUDNN_STATUS_EXECUTION_FAILED" in msg:
            print("\n[HINT] This looks like a CUDA/cuDNN kernel support issue for your GPU.")
            print("[HINT] If you have Pascal/Volta, use the README 'Stable stack (Pascal/Volta-friendly)'.\n")
        raise

    print(f"[INFO] Encoder time: {(time.time() - t0) * 1000:.1f} ms")
    print(
        f"[INFO] orig_hw={info.orig_hw} target={info.target_size} "
        f"scale_x={info.scale_x:.6f} scale_y={info.scale_y:.6f}"
    )

    disp_base, disp_scale = compute_display_base(img_bgr, max_side=1200)

    points: list[tuple[int, int]] = []
    labels: list[int] = []
    rect_start = rect_end = None
    drawing = False

    def run_with_points():
        if not points:
            cv2.imshow("SAM3 ONNX Demo", disp_base)
            return

        in_pts, in_lbl = prepare_points(points, labels, info)

        t = time.time()
        dec_out = run_decoder(
            sess_dec,
            enc_out,
            input_points=in_pts,
            input_labels=in_lbl,
            input_boxes=empty_boxes(),
        )
        dt = (time.time() - t) * 1000.0

        mask2d, best_score = pick_best_mask(dec_out["pred_masks"], dec_out["iou_scores"], which_prompt=0)
        mask255 = postprocess_mask_to_original(mask2d, info)
        overlay = green_overlay(img_bgr, mask255, alpha=0.5)

        vis = overlay.copy()
        for i, (px, py) in enumerate(points):
            col = (0, 0, 255) if labels[i] == 1 else (255, 0, 0)
            cv2.circle(vis, (px, py), 6, col, -1)

        cv2.imshow("SAM3 ONNX Demo", cv2.resize(vis, (disp_base.shape[1], disp_base.shape[0])))

        iou_vec = dec_out["iou_scores"][0, 0]
        print(f"[INFO] Decoder time: {dt:.1f} ms | iou={iou_vec} | best={best_score:.3f} | points={len(points)}")

    def run_with_box():
        nonlocal rect_start, rect_end
        if rect_start is None or rect_end is None:
            cv2.imshow("SAM3 ONNX Demo", disp_base)
            return

        x1d, y1d = rect_start
        x2d, y2d = rect_end

        x1, y1 = int(x1d / disp_scale), int(y1d / disp_scale)
        x2, y2 = int(x2d / disp_scale), int(y2d / disp_scale)

        boxes = prepare_boxes((x1, y1, x2, y2), info)
        pts0, lbl0 = empty_points()

        t = time.time()
        dec_out = run_decoder(sess_dec, enc_out, input_points=pts0, input_labels=lbl0, input_boxes=boxes)
        dt = (time.time() - t) * 1000.0

        mask2d, best_score = pick_best_mask(dec_out["pred_masks"], dec_out["iou_scores"], which_prompt=0)
        mask255 = postprocess_mask_to_original(mask2d, info)
        overlay = green_overlay(img_bgr, mask255, alpha=0.5)

        disp = cv2.resize(overlay, (disp_base.shape[1], disp_base.shape[0]))
        cv2.rectangle(disp, rect_start, rect_end, (0, 255, 255), 2)
        cv2.imshow("SAM3 ONNX Demo", disp)

        iou_vec = dec_out["iou_scores"][0, 0]
        print(f"[INFO] Decoder time: {dt:.1f} ms | iou={iou_vec} | best={best_score:.3f}")

    def mouse_cb(event, x, y, flags, param):
        nonlocal rect_start, rect_end, drawing

        if not mode_bbox:
            if event == cv2.EVENT_MBUTTONDOWN:
                points.clear()
                labels.clear()
                cv2.imshow("SAM3 ONNX Demo", disp_base)
            elif event == cv2.EVENT_LBUTTONDOWN:
                points.append((int(x / disp_scale), int(y / disp_scale)))
                labels.append(1)
                run_with_points()
            elif event == cv2.EVENT_RBUTTONDOWN:
                points.append((int(x / disp_scale), int(y / disp_scale)))
                labels.append(0)
                run_with_points()
            return

        if event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK):
            rect_start = rect_end = None
            cv2.imshow("SAM3 ONNX Demo", disp_base)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            rect_start = rect_end = (x, y)
            cv2.imshow("SAM3 ONNX Demo", disp_base)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            rect_end = (x, y)
            vis = disp_base.copy()
            cv2.rectangle(vis, rect_start, rect_end, (0, 255, 255), 2)
            cv2.imshow("SAM3 ONNX Demo", vis)
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            rect_end = (x, y)
            run_with_box()

    cv2.namedWindow("SAM3 ONNX Demo", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("SAM3 ONNX Demo", mouse_cb)
    cv2.imshow("SAM3 ONNX Demo", disp_base)

    print("[INFO] Interactive mode ready. ESC to quit.")
    print("[INFO] seed_points: left=pos, right=neg, middle=clear")
    print("[INFO] bounding_box: drag LMB to draw, RMB or double-click to clear")

    while True:
        if cv2.waitKey(20) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
