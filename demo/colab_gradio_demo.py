import os
import cv2
import time
import json
import numpy as np
import torch
import gradio as gr
import traceback

# Optional perf knobs (safe on CPU too)
if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# --------------------------
# Import tracker (your repo)
# --------------------------
tracker_module = None
last_import_error = None
for mod in ["sam2.sam2_object_tracker", "sam2.tools.sam2_object_tracker"]:
    try:
        tracker_module = __import__(mod, fromlist=["SAM2ObjectTracker"])
        break
    except Exception as e:
        last_import_error = e

if tracker_module is None:
    raise ImportError(
        "Couldn't import SAM2ObjectTracker. Expected sam2.sam2_object_tracker. "
        f"Last error: {last_import_error}"
    )

SAM2ObjectTracker = getattr(tracker_module, "SAM2ObjectTracker")

# --------------------------
# YOLO (Ultralytics)
# --------------------------
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# --------------------------
# Utilities
# --------------------------
def list_checkpoints(ckpt_dir="checkpoints"):
    if not os.path.isdir(ckpt_dir):
        return []
    files = []
    for root, _, fnames in os.walk(ckpt_dir):
        for f in fnames:
            if f.endswith((".pt", ".pth")):
                files.append(os.path.join(root, f))
    return sorted(files)

def overlay_masks_rgb(frame_rgb, masks_tensor, alpha=0.45):
    """
    frame_rgb: uint8 RGB
    masks_tensor: torch.Tensor or np.ndarray of shape [B, 1, H, W] or [B, H, W] (logits or probs)
    returns: uint8 RGB
    """
    if masks_tensor is None:
        return frame_rgb
    try:
        if torch.is_tensor(masks_tensor):
            masks = masks_tensor.detach().float().cpu().numpy()
        else:
            masks = masks_tensor
    except Exception:
        masks = masks_tensor

    if masks.ndim == 4:  # [B, 1, H, W]
        masks = masks[:, 0]  # -> [B, H, W]

    out = frame_rgb.copy()
    H, W = out.shape[:2]
    overlay = np.zeros_like(out, dtype=np.uint8)

    B = masks.shape[0] if masks is not None else 0
    for i in range(B):
        m = masks[i]
        # logits or probs → bin mask
        m_bin = (m > 0.5).astype(np.uint8)
        if m_bin.shape[:2] != (H, W):
            m_bin = cv2.resize(m_bin, (W, H), interpolation=cv2.INTER_NEAREST)

        rng = np.random.default_rng((i + 1) * 2654435761 % (2**32))
        color = tuple(int(x) for x in rng.integers(60, 255, size=3))  # RGB
        colored = np.zeros_like(out, dtype=np.uint8)
        colored[:, :] = color

        mask3 = np.repeat(m_bin[:, :, None], 3, axis=2)
        overlay = np.where(mask3 > 0, colored, overlay)

        ys, xs = np.where(m_bin > 0)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.circle(out, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(out, f"ID {i}", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out

def run_yolo_boxes_rgb(model, rgb_frame, conf=0.25, classes=None, max_det=100):
    """
    Returns boxes as np.ndarray (n,2,2) in ABSOLUTE pixels [[(x1,y1),(x2,y2)], ...]
    """
    if model is None or rgb_frame is None:
        return np.zeros((0, 2, 2), dtype=np.float32)
    res = model.predict(rgb_frame, conf=float(conf), classes=classes, max_det=int(max_det), verbose=False)
    if not res or len(res[0].boxes) == 0:
        return np.zeros((0, 2, 2), dtype=np.float32)
    xyxy = res[0].boxes.xyxy.cpu().numpy()
    out = []
    for x1, y1, x2, y2 in xyxy:
        out.append([[float(x1), float(y1)], [float(x2), float(y2)]])
    return np.array(out, dtype=np.float32)

def _try_open_writer(base_path, size, fps):
    """Try multiple codecs; return (writer, final_path)."""
    w, h = size
    attempts = [("mp4v", ".mp4"), ("avc1", ".mp4"), ("XVID", ".avi"), ("MJPG", ".avi")]
    base, _ = os.path.splitext(base_path)
    for fourcc_str, ext in attempts:
        path = base + ext
        writer = cv2.VideoWriter(cv2.VideoWriter_fourcc(*fourcc_str), fps, (w, h))
        # compat with OpenCV sig changes
        if not writer.isOpened():
            writer.release()
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc_str), fps, (w, h))
        if writer.isOpened():
            return writer, path
        writer.release()
    return None, None

# --------------------------
# App State (dict)
# --------------------------
state = {
    # user options
    "ckpt": None,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_objects": 4,
    "yolo_on": True,
    "yolo_model_name": "yolov8n.pt",
    "yolo_conf": 0.25,
    "yolo_classes": "",  # "" or "0,2"
    "reinject_every": 60,  # frames; 0=off

    # live session
    "tracker": None,
    "yolo": None,
    "first_frame_seeded": False,
    "tracking": False,
    "frame_idx": 0,

    # proposals
    "cands": [],
    "selected_idx": 0,
    "last_frame": None,

    # saving (video tab)
    "saving_enabled": False,
    "save_name": "segmented_output",
    "save_fps": 30.0,
    "writer": None,
    "writer_size": None,
    "save_path": None,
}

def _parse_classes(text):
    if not text or not str(text).strip():
        return None
    return [int(x.strip()) for x in str(text).split(",") if x.strip().isdigit()]

def _ensure_yolo():
    if not state["yolo_on"]:
        state["yolo"] = None
        return
    if YOLO is None:
        raise RuntimeError("Ultralytics is not installed. Please `pip install ultralytics`.")
    if state["yolo"] is None or getattr(state["yolo"], "model_name", None) != state["yolo_model_name"]:
        m = YOLO(state["yolo_model_name"])
        m.model_name = state["yolo_model_name"]
        state["yolo"] = m

def _build_tracker():
    # try checkpoint= then checkpoint_path=
    kwargs = dict(num_objects=int(state["num_objects"]), device=state["device"])
    t = None
    try:
        t = SAM2ObjectTracker(checkpoint=state["ckpt"], **kwargs)
    except Exception as e1:
        try:
            t = SAM2ObjectTracker(checkpoint_path=state["ckpt"], **kwargs)
        except Exception as e2:
            raise RuntimeError(f"Failed to init SAM2ObjectTracker: {e1} / {e2}")
    return t

def _reset_session():
    # release writer
    if state["writer"] is not None:
        try:
            state["writer"].release()
        except Exception:
            pass
    state.update({
        "tracker": None,
        "yolo": None,
        "first_frame_seeded": False,
        "tracking": False,
        "frame_idx": 0,
        "cands": [],
        "selected_idx": 0,
        "last_frame": None,
        "saving_enabled": False,
        "writer": None,
        "writer_size": None,
        "save_path": None,
    })

# --------------------------
# LIVE: per-frame handler
# --------------------------
@torch.inference_mode()
def process_live_frame(rgb_frame, ckpt, num_objects, device, yolo_on, yolo_model, yolo_conf, yolo_classes, reinject_every):
    """
    Receives frames from gr.Image(sources=["webcam"], streaming=True)
    """
    # update settings
    state["ckpt"] = ckpt
    state["device"] = device
    state["num_objects"] = int(num_objects)
    state["yolo_on"] = bool(yolo_on)
    state["yolo_model_name"] = yolo_model or "yolov8n.pt"
    state["yolo_conf"] = float(yolo_conf)
    state["yolo_classes"] = yolo_classes or ""
    state["reinject_every"] = int(reinject_every) if reinject_every not in (None, "", 0) else 0

    if rgb_frame is None or ckpt is None:
        return None

    state["last_frame"] = rgb_frame

    # lazy init tracker
    if state["tracker"] is None:
        state["tracker"] = _build_tracker()
        state["first_frame_seeded"] = False
        state["tracking"] = False
        state["frame_idx"] = 0

    # YOLO proposals (always draw when not tracking; optional while tracking)
    try:
        _ensure_yolo()
    except Exception as e:
        # YOLO optional; ignore errors until user installs
        print("[YOLO init warning]", e)
        state["yolo"] = None

    # If tracking: track and overlay
    if state["tracking"]:
        pred = state["tracker"].track_all_objects(rgb_frame)  # accepts np.ndarray (RGB) → preprocesses inside
        vis = overlay_masks_rgb(rgb_frame, pred.get("pred_masks_high_res"))

        # periodic reinjection (add new objects if capacity left)
        if state["yolo"] is not None and state["reinject_every"] > 0 and (state["frame_idx"] % state["reinject_every"] == 0):
            can_add = max(0, state["num_objects"] - state["tracker"].curr_obj_idx)
            if can_add > 0:
                boxes = run_yolo_boxes_rgb(state["yolo"], rgb_frame, conf=state["yolo_conf"], classes=_parse_classes(state["yolo_classes"]))
                if boxes.shape[0] > 0:
                    state["tracker"].track_new_object(rgb_frame, box=boxes[:can_add])

        state["frame_idx"] += 1
        return vis

    # Not tracking yet: show proposals overlay on raw frame
    draw = rgb_frame
    if state["yolo"] is not None:
        cands = state["cands"] = run_yolo_boxes_rgb(state["yolo"], rgb_frame, conf=state["yolo_conf"], classes=_parse_classes(state["yolo_classes"]))
        bgr = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR).copy()
        if len(cands) > 0:
            state["selected_idx"] = max(0, min(state["selected_idx"], len(cands)-1))
            for j, ((x1,y1),(x2,y2)) in enumerate(cands):
                color = (0,255,0) if j == state["selected_idx"] else (0,200,255)
                thick = 3 if j == state["selected_idx"] else 1
                cv2.rectangle(bgr, (int(x1),int(y1)), (int(x2),int(y2)), color, thick)
        draw = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return draw

def ui_next():
    if state["cands"]:
        state["selected_idx"] = (state["selected_idx"] + 1) % len(state["cands"])
    return gr.update()

def ui_prev():
    if state["cands"]:
        state["selected_idx"] = (state["selected_idx"] - 1) % len(state["cands"])
    return gr.update()

def ui_accept():
    """
    Add the selected YOLO box as a new object (allowed before or during tracking).
    """
    if state["last_frame"] is None or not state["cands"]:
        return "No candidate available."
    if state["tracker"] is None:
        return "Tracker not initialized yet."

    idx = state["selected_idx"]
    idx = max(0, min(idx, len(state["cands"])-1))
    (x1,y1),(x2,y2) = state["cands"][idx]
    bbox = np.array([[[x1, y1], [x2, y2]]], dtype=np.float32)

    # Seed object (this also builds/updates memory & increments curr_obj_idx)
    out = state["tracker"].track_new_object(state["last_frame"], box=bbox)
    state["first_frame_seeded"] = True
    return f"Added object (curr_obj_idx={state['tracker'].curr_obj_idx}). You can add more; then press Start Tracking."

def ui_toggle_yolo():
    state["yolo_on"] = not state["yolo_on"]
    if not state["yolo_on"]:
        state["yolo"] = None
    return f"YOLO proposals: {'ON' if state['yolo_on'] else 'OFF'}"

def ui_start_tracking():
    if state["tracker"] is None or state["last_frame"] is None:
        return "Tracker not ready."
    if state["tracker"].curr_obj_idx == 0:
        return "No objects added yet. Accept at least one YOLO box."
    state["tracking"] = True
    state["frame_idx"] = 0
    return f"Tracking started with {state['tracker'].curr_obj_idx} object(s)."

def ui_reset():
    _reset_session()
    return "Reset done."

# --------------------------
# VIDEO mode (generator)
# --------------------------
def start_video(video_file, ckpt, num_objects, device, yolo_on, yolo_model, yolo_conf, yolo_classes, reinject_every, save_name):
    _reset_session()
    state["ckpt"] = ckpt
    state["device"] = device
    state["num_objects"] = int(num_objects)
    state["yolo_on"] = bool(yolo_on)
    state["yolo_model_name"] = yolo_model or "yolov8n.pt"
    state["yolo_conf"] = float(yolo_conf)
    state["yolo_classes"] = yolo_classes or ""
    state["reinject_every"] = int(reinject_every) if reinject_every not in (None, "", 0) else 0
    state["save_name"] = (save_name or "").strip() or "segmented_output"

    if video_file is None:
        yield None, None, "Provide a video file."
        return

    path = video_file if isinstance(video_file, str) else getattr(video_file, "name", None)
    if not path or not os.path.exists(path):
        yield None, None, "Invalid video."
        return

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        yield None, None, "Cannot open video."
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    state["save_fps"] = float(fps)
    state["saving_enabled"] = True

    # init tracker + yolo
    try:
        state["tracker"] = _build_tracker()
    except Exception as e:
        yield None, None, f"Tracker init failed: {e}"
        cap.release()