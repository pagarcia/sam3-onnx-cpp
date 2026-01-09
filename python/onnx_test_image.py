# sam3-onnx-cpp/python/onnx_test_image.py
#!/usr/bin/env python3

import os
# Limit thread pools early
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
from PyQt5 import QtWidgets

from onnx_test_utils import (
    print_system_info, set_cv2_threads,
    make_session,
    preprocess_image_bgr,
    prepare_points, prepare_box_as_points,
    run_encoder, run_decoder,
    pick_best_mask, postprocess_mask_to_original,
    compute_display_base, green_overlay,
)

def main():
    print_system_info()
    set_cv2_threads(1)

    ap = argparse.ArgumentParser(description="SAM3-Tracker ONNX (image) – seed points / bounding box")
    ap.add_argument("--prompt", default="seed_points", choices=["seed_points", "bounding_box"])
    ap.add_argument("--safe", action="store_true", help="Disable ORT graph optimizations (more conservative).")
    args = ap.parse_args()
    mode_bbox = args.prompt == "bounding_box"
    print(f"[INFO] Prompt mode: {'bounding_box' if mode_bbox else 'seed_points'}")

    app = QtWidgets.QApplication(sys.argv)
    img_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None, "Select an Image", "", "Images (*.jpg *.jpeg *.png *.bmp);;All files (*)"
    )
    if not img_path:
        sys.exit("No image selected – exiting.")
    print(f"[INFO] Selected image: {img_path}")

    REPO_ROOT = Path(__file__).resolve().parent.parent
    onnx_dir = REPO_ROOT / "checkpoints" / "sam3" / "onnx"

    enc_path = onnx_dir / "vision_encoder.onnx"
    dec_path = onnx_dir / "prompt_encoder_mask_decoder.onnx"
    if not enc_path.exists():
        sys.exit(f"ERROR: Missing {enc_path}")
    if not dec_path.exists():
        sys.exit(f"ERROR: Missing {dec_path}")

    sess_enc = make_session(str(enc_path), tag="vision_encoder", safe=args.safe)
    sess_dec = make_session(str(dec_path), tag="prompt_encoder_mask_decoder", safe=args.safe)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        sys.exit("ERROR: Could not read image.")
    H_org, W_org = img_bgr.shape[:2]

    # Preprocess + encode
    pixel_values, info = preprocess_image_bgr(img_bgr, target_size=1008)
    t0 = time.time()
    enc_out = run_encoder(sess_enc, pixel_values)
    print(f"[INFO] Encoder time: {(time.time()-t0)*1000:.1f} ms")
    print(f"[INFO] orig_hw={info.orig_hw} resized_hw={info.resized_hw} target={info.target_size} scale={info.scale:.5f}")

    # Display base
    disp_base, disp_scale = compute_display_base(img_bgr, max_side=1200)

    # Interactive state
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
        dec_out = run_decoder(sess_dec, enc_out, in_pts, in_lbl)
        dt = (time.time() - t) * 1000.0

        # Expect outputs include pred_masks + iou_scores
        if "pred_masks" not in dec_out or "iou_scores" not in dec_out:
            print("[ERROR] Decoder outputs missing 'pred_masks' or 'iou_scores'. Got:", list(dec_out.keys()))
            return

        mask2d, best_score = pick_best_mask(dec_out["pred_masks"], dec_out["iou_scores"])
        mask255 = postprocess_mask_to_original(mask2d, info)
        overlay = green_overlay(img_bgr, mask255, alpha=0.5)

        vis = overlay.copy()
        for i, (px, py) in enumerate(points):
            col = (0, 0, 255) if labels[i] == 1 else (255, 0, 0)
            cv2.circle(vis, (px, py), 6, col, -1)

        vis_disp = cv2.resize(vis, (disp_base.shape[1], disp_base.shape[0]))
        cv2.imshow("SAM3 ONNX Demo", vis_disp)

        print(f"[INFO] Decoder time: {dt:.1f} ms | best_score={best_score:.3f} | points={len(points)}")

    def run_with_box():
        nonlocal rect_start, rect_end
        if rect_start is None or rect_end is None:
            cv2.imshow("SAM3 ONNX Demo", disp_base)
            return

        x1d, y1d = rect_start
        x2d, y2d = rect_end

        # Display coords -> original coords
        x1, y1 = int(x1d / disp_scale), int(y1d / disp_scale)
        x2, y2 = int(x2d / disp_scale), int(y2d / disp_scale)

        in_pts, in_lbl = prepare_box_as_points((x1, y1, x2, y2), info)

        t = time.time()
        dec_out = run_decoder(sess_dec, enc_out, in_pts, in_lbl)
        dt = (time.time() - t) * 1000.0

        if "pred_masks" not in dec_out or "iou_scores" not in dec_out:
            print("[ERROR] Decoder outputs missing 'pred_masks' or 'iou_scores'. Got:", list(dec_out.keys()))
            return

        mask2d, best_score = pick_best_mask(dec_out["pred_masks"], dec_out["iou_scores"])
        mask255 = postprocess_mask_to_original(mask2d, info)
        overlay = green_overlay(img_bgr, mask255, alpha=0.5)

        disp = cv2.resize(overlay, (disp_base.shape[1], disp_base.shape[0]))
        cv2.rectangle(disp, rect_start, rect_end, (0, 255, 255), 2)
        cv2.imshow("SAM3 ONNX Demo", disp)

        print(f"[INFO] Decoder time: {dt:.1f} ms | best_score={best_score:.3f}")

    def mouse_cb(event, x, y, flags, param):
        nonlocal rect_start, rect_end, drawing

        if not mode_bbox:
            # seed points mode
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

        # bbox mode
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
    print("[INFO] bounding_box: drag LMB to draw, RMB to clear")

    while True:
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
