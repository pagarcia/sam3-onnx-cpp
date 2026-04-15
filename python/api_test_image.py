# sam3-onnx-cpp/python/api_test_image.py
#!/usr/bin/env python3
"""
SAM3 native API interactive demo (image-only), similar UX to onnx_test_image.py.

Assumptions:
- The facebookresearch/sam3 repo is cloned next to sam3-onnx-cpp:
    ../sam3
- You have access to SAM3 checkpoints (HF gated) and are authenticated if needed.

Controls:
- seed_points mode:
    Left click  = positive point
    Right click = negative point
    Middle click= clear points
- bounding_box mode:
    Drag LMB to draw box
    RMB or double-click to clear box
- ESC to quit

Tips:
- Use --device auto (default), or --device cuda / mps / cpu
- If predict_inst is missing, you likely need enable_inst_interactivity=True at model build.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PyQt5 import QtWidgets
from PIL import Image


def _add_sam3_repo_to_syspath():
    """
    Add ../sam3 to sys.path so `import sam3` works without needing a pip install.
    """
    repo_root = Path(__file__).resolve().parent.parent  # sam3-onnx-cpp/
    sam3_repo = repo_root.parent / "sam3"
    if sam3_repo.exists():
        sys.path.insert(0, str(sam3_repo))
        return sam3_repo
    return None


def _choose_device(requested: str) -> str:
    """
    Choose torch device string: 'cuda' | 'mps' | 'cpu'
    """
    requested = requested.lower().strip()
    if requested != "auto":
        return requested

    import torch  # local import so script can error nicely if torch isn't installed

    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon / Metal
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _to_numpy_masks_scores(masks, scores):
    """
    Convert SAM3 outputs to (masks_np [K,H,W], scores_np [K]).
    Works with torch tensors or numpy arrays / lists.
    """
    import torch

    if isinstance(masks, torch.Tensor):
        masks_np = masks.detach().cpu().numpy()
    else:
        masks_np = np.asarray(masks)

    if isinstance(scores, torch.Tensor):
        scores_np = scores.detach().cpu().numpy()
    else:
        scores_np = np.asarray(scores)

    # Normalize mask dims
    # Common possibilities: [K,1,H,W] or [K,H,W] or [1,K,H,W] etc.
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0, :, :]
    elif masks_np.ndim == 4 and masks_np.shape[0] == 1:
        masks_np = masks_np[0]
        if masks_np.ndim == 3 and masks_np.shape[0] == 1:
            masks_np = masks_np[0]
    elif masks_np.ndim == 2:
        masks_np = masks_np[None, :, :]

    # Normalize scores
    scores_np = scores_np.reshape(-1)

    return masks_np, scores_np


def _best_mask(masks_np: np.ndarray, scores_np: np.ndarray) -> Tuple[np.ndarray, float]:
    if masks_np.size == 0 or scores_np.size == 0:
        return np.zeros((1, 1), dtype=np.uint8), 0.0
    k = int(np.argmax(scores_np))
    best_score = float(scores_np[k])
    m = masks_np[k]
    # if bool mask: keep; if float/logits: threshold at 0
    if m.dtype == np.bool_:
        mask255 = (m.astype(np.uint8) * 255)
    else:
        mask255 = ((m > 0.0).astype(np.uint8) * 255)
    return mask255, best_score


def _green_overlay(bgr: np.ndarray, mask255: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    fg = mask255 > 0
    color = np.zeros_like(bgr)
    color[fg] = (0, 255, 0)
    return cv2.addWeighted(bgr, 1.0, color, alpha, 0)


def _compute_display_base(img_bgr: np.ndarray, max_side: int = 1200) -> Tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(w, h))
    disp = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    return disp, scale


def main():
    # Ensure local sam3 repo can be imported
    sam3_repo = _add_sam3_repo_to_syspath()

    ap = argparse.ArgumentParser(description="SAM3 native API demo (image) – seed points / bounding box")
    ap.add_argument("--prompt", default="seed_points", choices=["seed_points", "bounding_box"])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--checkpoint", default="", help="Optional local checkpoint path for SAM3 (if you don't want HF download).")
    ap.add_argument("--multimask", action="store_true", help="Ask model for multiple masks; pick best by score.")
    args = ap.parse_args()

    mode_bbox = args.prompt == "bounding_box"
    print(f"[INFO] Prompt mode: {'bounding_box' if mode_bbox else 'seed_points'}")
    if sam3_repo:
        print(f"[INFO] Using local sam3 repo: {sam3_repo}")

    # Import torch + sam3 after sys.path adjustment
    try:
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except Exception as e:
        print("\n[ERROR] Could not import SAM3 API.")
        print("Make sure:")
        print("  1) ../sam3 exists next to this repo, OR")
        print("  2) you installed sam3 with: pip install -e ../sam3")
        print("Also ensure torch + dependencies are installed.")
        raise

    device = _choose_device(args.device)
    print(f"[INFO] Device: {device}")

    # Build model (enable_inst_interactivity is required for SAM1/SAM2-style prompts like predict_inst)
    # This flag is used in the official codebase and issues. :contentReference[oaicite:2]{index=2}
    build_kwargs = dict(
        device=device,
        enable_inst_interactivity=True,
        eval_mode=True,
    )
    if args.checkpoint.strip():
        build_kwargs["checkpoint_path"] = args.checkpoint.strip()
        build_kwargs["load_from_HF"] = False

    print("[INFO] Building SAM3 image model...")
    model = build_sam3_image_model(**build_kwargs)
    model.eval()

    processor = Sam3Processor(model)

    # Pick image via Qt
    app = QtWidgets.QApplication(sys.argv)
    img_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None, "Select an Image", "", "Images (*.jpg *.jpeg *.png *.bmp);;All files (*)"
    )
    if not img_path:
        sys.exit("No image selected – exiting.")
    print(f"[INFO] Selected image: {img_path}")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        sys.exit("ERROR: Could not read image with OpenCV.")
    h0, w0 = img_bgr.shape[:2]

    # Use PIL for SAM3 processor (avoids numpy-order pitfalls and matches repo examples) :contentReference[oaicite:3]{index=3}
    image_pil = Image.open(img_path).convert("RGB")

    print("[INFO] Encoding image (set_image)...")
    t0 = time.time()
    inference_state = processor.set_image(image_pil)
    print(f"[INFO] set_image time: {(time.time()-t0)*1000:.1f} ms")

    disp_base, disp_scale = _compute_display_base(img_bgr, max_side=1200)

    # Interactive state
    points: List[Tuple[int, int]] = []
    labels: List[int] = []
    rect_start = None
    rect_end = None
    drawing = False

    def run_with_points():
        if not points:
            cv2.imshow("SAM3 API Demo", disp_base)
            return

        # SAM3 expects coordinates in original pixel space (processor handles internal transforms).
        pt_coords = [[float(x), float(y)] for (x, y) in points]
        pt_labels = [int(v) for v in labels]

        kwargs = dict(
            point_coords=pt_coords,
            point_labels=pt_labels,
            multimask_output=bool(args.multimask),
        )

        with torch.inference_mode():
            t = time.time()
            out = model.predict_inst(inference_state, **kwargs)
            dt = (time.time() - t) * 1000.0

        # predict_inst returns (masks, scores, logits)
        masks, scores, _ = out
        masks_np, scores_np = _to_numpy_masks_scores(masks, scores)
        mask255, best_score = _best_mask(masks_np, scores_np)

        overlay = _green_overlay(img_bgr, mask255, alpha=0.5)
        vis = overlay.copy()
        for i, (px, py) in enumerate(points):
            col = (0, 0, 255) if labels[i] == 1 else (255, 0, 0)
            cv2.circle(vis, (px, py), 6, col, -1)

        cv2.imshow("SAM3 API Demo", cv2.resize(vis, (disp_base.shape[1], disp_base.shape[0])))
        print(f"[INFO] predict_inst time: {dt:.1f} ms | best_score={best_score:.3f} | points={len(points)}")

    def run_with_box():
        nonlocal rect_start, rect_end
        if rect_start is None or rect_end is None:
            cv2.imshow("SAM3 API Demo", disp_base)
            return

        x1d, y1d = rect_start
        x2d, y2d = rect_end

        # Display coords -> original coords
        x1, y1 = int(x1d / disp_scale), int(y1d / disp_scale)
        x2, y2 = int(x2d / disp_scale), int(y2d / disp_scale)

        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        kwargs = dict(
            box=[float(x1), float(y1), float(x2), float(y2)],
            multimask_output=bool(args.multimask),
        )

        with torch.inference_mode():
            t = time.time()
            out = model.predict_inst(inference_state, **kwargs)
            dt = (time.time() - t) * 1000.0

        masks, scores, _ = out
        masks_np, scores_np = _to_numpy_masks_scores(masks, scores)
        mask255, best_score = _best_mask(masks_np, scores_np)

        overlay = _green_overlay(img_bgr, mask255, alpha=0.5)
        disp = cv2.resize(overlay, (disp_base.shape[1], disp_base.shape[0]))
        cv2.rectangle(disp, rect_start, rect_end, (0, 255, 255), 2)
        cv2.imshow("SAM3 API Demo", disp)
        print(f"[INFO] predict_inst time: {dt:.1f} ms | best_score={best_score:.3f}")

    def mouse_cb(event, x, y, flags, param):
        nonlocal rect_start, rect_end, drawing

        if not mode_bbox:
            # seed points
            if event == cv2.EVENT_MBUTTONDOWN:
                points.clear()
                labels.clear()
                cv2.imshow("SAM3 API Demo", disp_base)

            elif event == cv2.EVENT_LBUTTONDOWN:
                px = int(x / disp_scale)
                py = int(y / disp_scale)
                points.append((px, py))
                labels.append(1)
                run_with_points()

            elif event == cv2.EVENT_RBUTTONDOWN:
                px = int(x / disp_scale)
                py = int(y / disp_scale)
                points.append((px, py))
                labels.append(0)
                run_with_points()
            return

        # bbox mode
        if event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK):
            rect_start = rect_end = None
            cv2.imshow("SAM3 API Demo", disp_base)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            rect_start = rect_end = (x, y)
            cv2.imshow("SAM3 API Demo", disp_base)

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            rect_end = (x, y)
            vis = disp_base.copy()
            cv2.rectangle(vis, rect_start, rect_end, (0, 255, 255), 2)
            cv2.imshow("SAM3 API Demo", vis)

        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            rect_end = (x, y)
            run_with_box()

    cv2.namedWindow("SAM3 API Demo", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("SAM3 API Demo", mouse_cb)
    cv2.imshow("SAM3 API Demo", disp_base)

    print("[INFO] Interactive mode ready. ESC to quit.")
    print("[INFO] seed_points: left=pos, right=neg, middle=clear")
    print("[INFO] bounding_box: drag LMB to draw, RMB or double-click to clear")

    while True:
        if cv2.waitKey(20) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
