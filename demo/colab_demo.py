"""
Colab + Gradio demo for your repo (sam2_realtime_with_kf)

Flow:
  1) Open Gradio UI and stream frames from the webcam.
  2) Click **Detect & Track Person**: we run YOLO to find a person, pick the largest person bbox,
     initialize the tracker with that bbox, and start tracking in real-time.
  3) Click **Stop** to stop tracking.

Notes:
  - This file is designed to live at: demo/colab_gradio_person_track.py
  - It installs its Python deps if missing (ultralytics for YOLO + gradio + opencv).
  - "SAM2" integration points are clearly marked below. I included a wrapper class that should work
    with typical APIs from "sam2_realtime" repos. If names differ in your fork, only adjust the few
    lines marked with "### EDIT HERE if your API differs".
  - For Colab: make sure your checkpoints are downloaded to the repo's ./checkpoints directory.

If you need me to make this fit *exactly* to your tracker class/method names,
share the short signature (or paste the first ~50 lines) of `sam2/sam2_object_tracker.py`, and I’ll lock it in.
"""
from __future__ import annotations

import os
import sys
import time
import threading
import queue
from dataclasses import dataclass
from typing import Optional, Tuple, List

# --- light auto-install for Colab convenience ---------------------------------
try:
    import gradio as gr  # type: ignore
except Exception:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio>=4.44.0"])  # recent gradio w/ webcam streaming
    import gradio as gr

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics>=8.2.0"])  # YOLOv8
    from ultralytics import YOLO

try:
    import cv2
except Exception:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless>=4.9.0.80"])  # headless for Colab/Gradio
    import cv2

import numpy as np

# --- SAM2 imports (from your repo) --------------------------------------------
# We keep these in a try/except because class/module names can differ between forks.
SAM2_IMPORT_ERROR = None
try:
    # Most forks expose a builder function for the real-time video predictor
    # and/or a tracker class that wraps it.
    from sam2.build_sam import build_sam2_video_predictor  # type: ignore
except Exception as e:
    SAM2_IMPORT_ERROR = e

# Some forks place the tracker here
try:
    # ### EDIT HERE if your tracker lives elsewhere or has a different class name
    from sam2.sam2_object_tracker import SAM2ObjectTracker  # type: ignore
except Exception:
    SAM2ObjectTracker = None  # will handle below

# ----------------------------------------------------------------------------
# Configuration knobs (change these if needed)
# ----------------------------------------------------------------------------
CHECKPOINT_PATH = os.environ.get("SAM2_CKPT", "checkpoints/sam2.1_hiera_tiny.pt")
DEVICE = os.environ.get("SAM2_DEVICE", "cuda" if (os.environ.get("COLAB_GPU") or os.environ.get("CUDA_VISIBLE_DEVICES")) else "cpu")
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolov8n.pt")  # small + fast
PERSON_CLASS_ID = 0  # COCO person

# Tracking UI/logic params
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

# ----------------------------------------------------------------------------
# Utility drawing helpers
# ----------------------------------------------------------------------------

def draw_bbox(img: np.ndarray, bbox: Tuple[int, int, int, int], color=(0, 255, 0), label: Optional[str] = None) -> np.ndarray:
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img


def np_uint8(img: np.ndarray) -> np.ndarray:
    img = np.ascontiguousarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# ----------------------------------------------------------------------------
# SAM2 wrapper (adapt if API differs in your fork)
# ----------------------------------------------------------------------------

class RealtimeSAM2:
    """Minimal wrapper so the demo code doesn’t depend on exact class names.

    Expected behavior:
      - Create/hold a stateful tracker
      - `init_with_bbox(frame, bbox)` to initialize target
      - `track(frame)` -> returns dict with maybe 'mask'/'bbox'/'polygons'.
    """

    def __init__(self, checkpoint_path: str = CHECKPOINT_PATH, device: str = DEVICE, num_objects: int = 1):
        if SAM2_IMPORT_ERROR is not None:
            raise RuntimeError(
                f"Could not import SAM2 predictor builder (sam2.build_sam). Error: {SAM2_IMPORT_ERROR}\n"
                "Make sure your repo is installed (pip install -e .) and checkpoints are present."
            )
        self.predictor = build_sam2_video_predictor(checkpoint_path, device=device)
        self.num_objects = num_objects

        # Try to prefer a provided tracker class if available
        if SAM2ObjectTracker is not None:
            # ### EDIT HERE if your tracker takes different args
            self.tracker = SAM2ObjectTracker(self.predictor, num_objects=num_objects)
            self.use_direct_predictor = False
        else:
            # Fallback: track directly with `predictor` if your build returns a stateful predictor.
            self.tracker = None
            self.use_direct_predictor = True

        # Internal state that some APIs require (e.g., memory/keyframes)
        self._initialized = False

    def init_with_bbox(self, frame_bgr: np.ndarray, bbox_xyxy: Tuple[int, int, int, int]):
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        bbox_xyxy = (x1, y1, x2, y2)

        if not self.use_direct_predictor:
            # ### EDIT HERE if your tracker has a different init method
            self.tracker.start_sequence(frame_bgr, [bbox_xyxy])
        else:
            # Common direct predictor pattern in real-time forks
            # 1) start a new video sequence / reset state
            try:
                self.state = self.predictor.start_video()
            except AttributeError:
                # Some forks do `init_state` with the first frame
                self.state = self.predictor.init_state(frame_bgr)
            # 2) Add/init target by bbox
            try:
                self.predictor.add_target_box(self.state, bbox_xyxy)
            except Exception:
                # Another common API uses prompt points/boxes functions
                self.predictor.add_new_points_or_box(
                    self.state,
                    box=bbox_xyxy,
                    frame=frame_bgr,
                )
        self._initialized = True

    def track(self, frame_bgr: np.ndarray):
        if not self._initialized:
            return {"frame": frame_bgr}

        if not self.use_direct_predictor:
            # ### EDIT HERE if your tracker exposes another method name
            out = self.tracker.track(frame_bgr)
            # We expect something like {"bboxes": [...], "masks": [...], ...}
            return out
        else:
            # Direct predictor step
            try:
                step_out = self.predictor.track(self.state, frame_bgr)
            except Exception:
                # Older forks may use this
                step_out = self.predictor.step(self.state, frame_bgr)
            return step_out

# ----------------------------------------------------------------------------
# YOLO person detector helper
# ----------------------------------------------------------------------------

class YOLOPerson:
    def __init__(self, weights: str = YOLO_WEIGHTS, device: str = DEVICE):
        self.model = YOLO(weights)
        # The Ultralytics interface handles device automatically; we keep device string for clarity.
        self.device = device

    def largest_person_bbox(self, frame_bgr: np.ndarray, conf: float = CONF_THRESHOLD, iou: float = IOU_THRESHOLD) -> Optional[Tuple[int, int, int, int]]:
        # YOLO expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.model.predict(source=frame_rgb, conf=conf, iou=iou, verbose=False)
        # results is a list; take first
        if not results:
            return None
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None
        # Filter to person class
        bboxes = []
        for b in r.boxes:
            cls = int(b.cls.item()) if hasattr(b.cls, 'item') else int(b.cls)
            if cls == PERSON_CLASS_ID:
                xyxy = b.xyxy.squeeze().tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                area = max(0, x2 - x1) * max(0, y2 - y1)
                bboxes.append(((x1, y1, x2, y2), area))
        if not bboxes:
            return None
        # Largest area
        bboxes.sort(key=lambda t: t[1], reverse=True)
        return bboxes[0][0]

# ----------------------------------------------------------------------------
# Gradio real-time app
# ----------------------------------------------------------------------------

@dataclass
class AppState:
    tracker: Optional[RealtimeSAM2] = None
    detecting: bool = False
    tracking: bool = False
    init_bbox: Optional[Tuple[int, int, int, int]] = None


app_state = AppState()
yolo = YOLOPerson()


def arm_detection():
    app_state.detecting = True
    return gr.update(value="Armed: next frame will auto-detect the largest person and start tracking."), None


def stop_tracking():
    app_state.tracking = False
    app_state.detecting = False
    app_state.tracker = None
    app_state.init_bbox = None
    return gr.update(value="Stopped."), None


def process_frame(frame: np.ndarray, mirror: bool = True):
    """Main per-frame callback. Receives an RGB image from the webcam.
    Returns an RGB image to display.
    """
    if frame is None:
        return None
    # Gradio gives RGB; convert to BGR for cv2-style processing and SAM2 (usually expects BGR)
    # Optional horizontal mirror to match webcam selfie view
    if mirror:
        frame = cv2.flip(frame, 1)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # If user clicked Detect & Track, run YOLO once to initialize
    if app_state.detecting and not app_state.tracking:
        bbox = yolo.largest_person_bbox(frame_bgr)
        if bbox is not None:
            app_state.init_bbox = bbox
            if app_state.tracker is None:
                try:
                    app_state.tracker = RealtimeSAM2(CHECKPOINT_PATH, DEVICE, num_objects=1)
                except Exception as e:
                    # Display the error as overlay text
                    overlay = frame_bgr.copy()
                    cv2.putText(overlay, f"SAM2 init failed: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    return cv2.cvtColor(np_uint8(overlay), cv2.COLOR_BGR2RGB)
            try:
                app_state.tracker.init_with_bbox(frame_bgr, bbox)
                app_state.tracking = True
                app_state.detecting = False
            except Exception as e:
                overlay = frame_bgr.copy()
                cv2.putText(overlay, f"Tracker init error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                return cv2.cvtColor(np_uint8(overlay), cv2.COLOR_BGR2RGB)
        else:
            # No person found
            overlay = frame_bgr.copy()
            cv2.putText(overlay, "No person detected. Try again.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return cv2.cvtColor(np_uint8(overlay), cv2.COLOR_BGR2RGB)

    # If tracking is on, call SAM2
    if app_state.tracking and app_state.tracker is not None:
        try:
            out = app_state.tracker.track(frame_bgr)
        except Exception as e:
            overlay = frame_bgr.copy()
            cv2.putText(overlay, f"Track step error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return cv2.cvtColor(np_uint8(overlay), cv2.COLOR_BGR2RGB)

        # Try to visualize from a few common output formats
        vis = frame_bgr.copy()
        # 1) If bbox is returned
        bbox = None
        if isinstance(out, dict):
            if "bbox" in out and out["bbox"] is not None:
                bbox = out["bbox"]
            elif "bboxes" in out and out["bboxes"]:
                bbox = out["bboxes"][0]
            # If masks are present, overlay the first
            mask = out.get("mask") or (out.get("masks")[0] if out.get("masks") else None)
            if mask is not None:
                # Ensure mask is HxW bool/0-1
                m = mask.astype(np.uint8)
                if m.max() == 1:
                    m = (m * 255).astype(np.uint8)
                color = (0, 255, 0)
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, color, 2)
        if bbox is None and app_state.init_bbox is not None:
            # fallback to initial bbox
            bbox = app_state.init_bbox
        if bbox is not None:
            vis = draw_bbox(vis, bbox, color=(0, 255, 0), label="person")
        return cv2.cvtColor(np_uint8(vis), cv2.COLOR_BGR2RGB)

    # Default: idle view; show armed message if needed
    vis = frame_bgr.copy()
    msg = "Click 'Detect & Track Person' to start"
    if app_state.detecting:
        msg = "Armed: waiting for next frame to detect a person…"
    cv2.putText(vis, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return cv2.cvtColor(np_uint8(vis), cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------------------
# Build Gradio UI
# ----------------------------------------------------------------------------

def build_demo() -> gr.Blocks:
    with gr.Blocks(title="SAM2 Realtime Person Tracking (Colab)") as demo:
        gr.Markdown(
            "# SAM2 Realtime Person Tracking\n"
            "This demo uses YOLO to initialize a person bbox and then tracks with your SAM2 realtime pipeline.\n"
            f"**Checkpoint:** `{CHECKPOINT_PATH}`  ·  **Device:** `{DEVICE}`  ·  **YOLO:** `{YOLO_WEIGHTS}`\n"
        )

        with gr.Row():
            cam = gr.Image(sources=["webcam"], streaming=True, label="Webcam")
            out = gr.Image(label="Output", interactive=False)

        with gr.Row():
            mirror_cb = gr.Checkbox(label="Mirror webcam", value=True)
            status = gr.Textbox(label="Status", value="Idle.", interactive=False)
        with gr.Row():
            btn_arm = gr.Button("🔎 Detect & Track Person")
            btn_stop = gr.Button("⏹️ Stop")

        # Wire events
        cam.stream(process_frame, inputs=[cam, mirror_cb], outputs=out)
        btn_arm.click(arm_detection, inputs=None, outputs=[status, out])
        btn_stop.click(stop_tracking, inputs=None, outputs=[status, out])

        gr.Markdown(
            "**Tip:** If you see an import error mentioning `sam2.build_sam` or the tracker class,\n"
            "double-check that you've run `pip install -e .` from the repo root in this Colab runtime\n"
            "and that your SAM2 checkpoints exist under `./checkpoints`.\n"
        )
    return demo


if __name__ == "__main__":
    demo = build_demo()
    # In Colab we generally want share=True to expose a public URL for the webcam UI.
    demo.queue().launch(share=True)