# demo/colab_gradio_demo.py

import os
import cv2
import time
import numpy as np
import torch
import gradio as gr
import traceback

from ultralytics import YOLO

# -------- Performance knobs --------
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ========= BUILD TRACKER (repo-correct) =========
# We use your SAM2ObjectTracker class and build a SAM2 model under the hood.
# Config names here match THIS repo (July SAM2 configs, Sept 2.1 checkpoints).
from importlib import import_module
import inspect

# Fixed paths (no UI)
CKPT = "checkpoints/sam2.1_hiera_small.pt"
CFG_BASENAME = "sam2_hiera_s.yaml"
CFG_CANDIDATES = [
    f"sam2/configs/sam2/{CFG_BASENAME}",
    f"configs/sam2/{CFG_BASENAME}",
]

# Pull tracker class
_tracker_mod = None
_last_err = None
for mod in ["sam2.sam2_object_tracker", "sam2.tools.sam2_object_tracker"]:
    try:
        _tracker_mod = import_module(mod)
        break
    except Exception as e:
        _last_err = e
if _tracker_mod is None:
    raise ImportError(f"Couldn't import SAM2ObjectTracker (last error: {_last_err})")
SAM2ObjectTracker = getattr(_tracker_mod, "SAM2ObjectTracker")

def _first_exist(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

try:
    from omegaconf import OmegaConf
except Exception:
    OmegaConf = None

def _load_cfg_for_builder():
    cfg_path = _first_exist(CFG_CANDIDATES)
    if cfg_path and OmegaConf is not None:
        try:
            return OmegaConf.load(cfg_path), os.path.basename(cfg_path), cfg_path
        except Exception:
            pass
    # Fall back to just basename; many forks resolve internally.
    return None, CFG_BASENAME, _first_exist(CFG_CANDIDATES) or CFG_BASENAME

def _discover_and_build_sam2():
    """
    Import sam2.build_sam and try common builder signatures so this works
    across forks (build_sam2, build_model, build_*predictor, etc.).
    Returns a model exposing the encoders/decoders SAM2ObjectTracker expects.
    """
    try:
        mod = import_module("sam2.build_sam")
    except Exception as e:
        raise ImportError(f"Couldn't import sam2.build_sam: {e}")

    cfg_obj, cfg_base, cfg_path_print = _load_cfg_for_builder()
    print(f"[sam2] cfg → {os.path.abspath(cfg_path_print) if os.path.exists(str(cfg_path_print)) else cfg_base}")

    trials = []
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if "build" not in name.lower():
            continue
        trials += [
            (lambda f=fn: f(cfg_obj, CKPT), f"{name}(cfg_obj, ckpt)"),
            (lambda f=fn: f(config=cfg_obj, checkpoint=CKPT), f"{name}(config=cfg_obj, checkpoint=...)"),
            (lambda f=fn: f(cfg_base, CKPT), f"{name}('{cfg_base}', ckpt)"),
            (lambda f=fn: f(config=cfg_base, checkpoint=CKPT), f"{name}(config='{cfg_base}', checkpoint=...)"),
            (lambda f=fn: f(cfg_obj), f"{name}(cfg_obj)"),
            (lambda f=fn: f(cfg_base), f"{name}('{cfg_base}')"),
        ]

    need = ["image_encoder", "memory_attention", "memory_encoder", "prompt_encoder", "mask_decoder"]
    last_err = None
    for caller, desc in trials:
        try:
            built = caller()
        except TypeError:
            continue
        except Exception as e:
            last_err = e
            continue

        cands = [built]
        for attr in ["model", "sam", "sam2", "module", "net", "backbone"]:
            if hasattr(built, attr):
                cands.append(getattr(built, attr))
        for m in cands:
            if m is not None and all(hasattr(m, k) for k in need):
                print(f"[sam2] using builder → {desc}")
                return m

    raise RuntimeError(f"Could not build a SAM2 model exposing encoders (last error: {last_err})")

def build_tracker():
    if not os.path.exists(CKPT):
        raise RuntimeError(f"Checkpoint not found: {CKPT}")
    model = _discover_and_build_sam2()
    return SAM2ObjectTracker(
        image_encoder=model.image_encoder,
        memory_attention=model.memory_attention,
        memory_encoder=model.memory_encoder,
        prompt_encoder=model.prompt_encoder,
        mask_decoder=model.mask_decoder,
        num_objects=10,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )

# Create the tracker once at import-time (same as your predictor pattern)
tracker = build_tracker()

# --- robust SAMURAI/KF toggler (works if attrs exist; safe if not) ---
def _maybe_set_attr(obj, name, value):
    try:
        if hasattr(obj, name):
            setattr(obj, name, value)
            return True
    except Exception:
        pass
    return False

def set_samurai_mode(tracker_obj, enable: bool):
    """
    Try sensible locations for KF-like flags. If your build doesn't expose them,
    this is a no-op (keeps API parity with your reference demo).
    """
    hit = []
    candidates = [
        tracker_obj,
        getattr(tracker_obj, "model", None),
        getattr(tracker_obj, "module", None),
        getattr(getattr(tracker_obj, "module", None), "model", None),
    ]
    candidates = [c for c in candidates if c is not None]

    for c in candidates:
        if _maybe_set_attr(c, "samurai_mode", bool(enable)):
            hit.append(f"{c.__class__.__name__}.samurai_mode")

    if not enable:
        for c in candidates:
            if _maybe_set_attr(c, "stable_frames_threshold", 0):
                hit.append(f"{c.__class__.__name__}.stable_frames_threshold=0")
            if _maybe_set_attr(c, "kf_score_weight", 0.0):
                hit.append(f"{c.__class__.__name__}.kf_score_weight=0.0")
            # clear KF state if present
            _maybe_set_attr(c, "kf_mean", None)
            _maybe_set_attr(c, "kf_covariance", None)
            _maybe_set_attr(c, "stable_frames", 0)
            _maybe_set_attr(c, "frame_cnt", 0)

    if hit:
        print(("SAMURAI mode: ON" if enable else "SAMURAI mode: OFF") + " | " + ", ".join(hit))
    else:
        print("Warning: couldn't set any samurai_mode/KF attrs (ok if your build hides them).")
    return enable

# Default: assume single-person until user adds more
set_samurai_mode(tracker, True)

# YOLO for proposals
yolo_model = YOLO("yolov8s.pt")

# ---------- small utils ----------
def _writable_dir():
    return "/tmp"

def _resolve_video_path(video_input):
    if isinstance(video_input, str):
        return video_input
    if isinstance(video_input, dict) and "name" in video_input:
        return video_input["name"]
    return None

def _try_open_writer(base_path, size, fps):
    """Try multiple codecs; return (writer, final_path)."""
    w, h = size
    attempts = [("mp4v", ".mp4"), ("avc1", ".mp4"), ("XVID", ".avi"), ("MJPG", ".avi")]
    base, _ = os.path.splitext(base_path)
    for fourcc_str, ext in attempts:
        test_path = base + ext
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(test_path, fourcc, fps, (w, h))
        if writer.isOpened():
            return writer, test_path
        writer.release()
    return None, None

# -------- Helpers (vision) --------
def yolo_person_bboxes(rgb_frame, model, conf_thres=0.25):
    if rgb_frame is None:
        return []
    res = model(rgb_frame, verbose=False, conf=conf_thres)[0]
    out = []
    for det in res.boxes:
        if int(det.cls) == 0:  # person
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            conf = float(det.conf[0].item()) if det.conf is not None else 0.0
            out.append((x1, y1, x2, y2, conf))
    out.sort(key=lambda t: t[4], reverse=True)
    return out

def _count_objs(pred_masks_or_ids):
    if pred_masks_or_ids is None:
        return 0
    # our tracker returns masks list/np/tensor; count along first dim
    if isinstance(pred_masks_or_ids, (list, tuple)):
        return len(pred_masks_or_ids)
    if torch.is_tensor(pred_masks_or_ids):
        return int(pred_masks_or_ids.shape[0]) if pred_masks_or_ids.ndim >= 1 else int(pred_masks_or_ids.numel())
    if hasattr(pred_masks_or_ids, "shape"):
        return int(pred_masks_or_ids.shape[0])
    return 0

def draw_mask_overlay(rgb_frame, pred_masks_high_res):
    if rgb_frame is None:
        return None
    n = _count_objs(pred_masks_high_res)
    if n == 0:
        return rgb_frame
    h, w = rgb_frame.shape[:2]
    all_mask = np.zeros((h, w, 3), dtype=np.uint8)
    all_mask[..., 1] = 255  # saturation
    for i in range(n):
        logits_i = pred_masks_high_res[i]
        if torch.is_tensor(logits_i):
            m = (logits_i > 0).detach().cpu().numpy().astype(np.uint8)
        else:
            m = (np.asarray(logits_i) > 0).astype(np.uint8)
        if m.ndim == 3:
            m = m.squeeze()
        hue = int((i + 3) / (n + 3) * 255)
        sel = m.astype(bool)
        all_mask[sel, 0] = hue
        all_mask[sel, 2] = 255
    all_mask = cv2.cvtColor(all_mask, cv2.COLOR_HSV2RGB)
    return cv2.addWeighted(rgb_frame, 1.0, all_mask, 0.5, 0.0)

# -------- App state --------
state = {
    # session & seeding
    "first_frame_loaded": False,   # (kept for parity; tracker doesn’t need it)
    "seeded_any": False,
    "tracking": False,

    # proposals
    "yolo_enabled": True,
    "selected_idx": 0,
    "cands": [],
    "last_frame": None,

    # bookkeeping
    "added_count": 0,  # how many objects we added (used to toggle samurai_mode)

    # video & auto-save
    "video_path": None,
    "video_fps": 30.0,
    "saving_enabled": False,
    "save_name": "segmented_output",
    "save_fps": 30.0,
    "writer": None,
    "writer_size": None,
    "save_path": None,
}

# ---- writer helpers ----
def _maybe_open_writer_on_first_segmented(frame_rgb):
    if not state["saving_enabled"] or state["writer"] is not None or frame_rgb is None:
        return
    h, w = frame_rgb.shape[:2]
    base_dir = _writable_dir()
    base_path = os.path.join(base_dir, state["save_name"])
    writer, final_path = _try_open_writer(base_path, (w, h), state["save_fps"])
    if writer is None:
        print("[save] Failed to open writer.")
        state["saving_enabled"] = False
        return
    state["writer"] = writer
    state["writer_size"] = (w, h)
    state["save_path"] = final_path
    print(f"[save] Writer opened: {final_path} @ {state['save_fps']:.2f} FPS")

def _write_segmented_frame(frame_rgb):
    if not state["saving_enabled"] or state["writer"] is None or frame_rgb is None:
        return
    w, h = state["writer_size"]
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if (bgr.shape[1], bgr.shape[0]) != (w, h):
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    state["writer"].write(bgr)

def _finalize_writer():
    if state["writer"] is not None:
        try:
            state["writer"].release()
        except Exception:
            pass
    path = state["save_path"]
    state["writer"] = None
    state["writer_size"] = None
    state["saving_enabled"] = False
    return path if path and os.path.exists(path) else None

# -------- Core (webcam & video) --------
@torch.inference_mode()
def process_frame(rgb_frame):
    """
    Webcam:
      - While tracking=False you can Accept several people (all on the current frame).
      - You CAN also Accept mid-stream (adds new object immediately).
    Video:
      - Pauses on the first frame to accept several; Start Tracking begins playback.
    """
    if rgb_frame is None:
        return None
    state["last_frame"] = rgb_frame

    # 1) If tracking, run tracker and draw masks onto base
    base = rgb_frame
    if state["tracking"]:
        try:
            # repo tracker API
            pred = tracker.track_all_objects(rgb_frame)
            masks = pred.get("pred_masks_high_res")
            base = draw_mask_overlay(rgb_frame, masks)
        except Exception as e:
            print("[error] track_all_objects() failed:", repr(e))
            print(traceback.format_exc())
            base = rgb_frame
        _maybe_open_writer_on_first_segmented(base)
        _write_segmented_frame(base)

    # 2) (Optional) draw YOLO proposals on top
    if state["yolo_enabled"]:
        cands = yolo_person_bboxes(rgb_frame, yolo_model, conf_thres=0.25)
        state["cands"] = cands
        bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR).copy()
        if cands:
            state["selected_idx"] = max(0, min(state["selected_idx"], len(cands)-1))
            for j, (x1,y1,x2,y2,conf) in enumerate(cands):
                color = (0,255,0) if j == state["selected_idx"] else (0,200,255)
                thick = 3 if j == state["selected_idx"] else 1
                cv2.rectangle(bgr, (x1,y1), (x2,y2), color, thick)
                cv2.putText(bgr, f"{conf:.2f}", (x1, max(0,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            hint = "[Accept]=add person  [Next]/[Prev]=cycle  [Toggle YOLO]=hide/show"
        else:
            hint = "No person found."
        cv2.putText(bgr, hint, (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        base = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return base

# -------- Controls --------
def on_next():
    if state["yolo_enabled"] and state["cands"]:
        state["selected_idx"] = (state["selected_idx"] + 1) % len(state["cands"])
    return None

def on_prev():
    if state["yolo_enabled"] and state["cands"]:
        state["selected_idx"] = (state["selected_idx"] - 1) % len(state["cands"])
    return None

def on_toggle_yolo():
    state["yolo_enabled"] = not state["yolo_enabled"]
    return f"YOLO proposals: {'ON' if state['yolo_enabled'] else 'OFF'}"

def on_accept():
    """
    Add the selected bbox as a new object.
    Works both BEFORE and DURING tracking.
    """
    if not state["cands"] or state["last_frame"] is None:
        return "No candidate available."

    x1, y1, x2, y2, conf = state["cands"][state["selected_idx"]]
    # repo tracker expects shape [N,2,2] with absolute pixels
    bbox = np.array([[[x1, y1], [x2, y2]]], dtype=np.float32)

    try:
        tracker.track_new_object(state["last_frame"], box=bbox)
    except Exception as e:
        return f"Add failed: {e}"

    state["seeded_any"] = True
    state["added_count"] += 1

    # If >1 objects, disable KF/SAMURAI-like behavior (if the build exposes it)
    if state["added_count"] > 1:
        set_samurai_mode(tracker, False)

    return f"Added object #{state['added_count']} (conf={conf:.2f}). " \
           f"You can keep adding{' (even mid-stream)' if state['tracking'] else ''} or press 'Start Tracking'."

def on_start_tracking():
    """
    Begin per-frame tracking; decide SAMURAI mode based on how many objects were seeded.
    """
    # repo tracker exposes curr_obj_idx
    if getattr(tracker, "curr_obj_idx", 0) == 0:
        return "No objects added yet. Accept at least one person first."

    # Single object → enable SAMURAI(KF) if present; Multi → disable
    set_samurai_mode(tracker, enable=(state["added_count"] == 1))

    state["tracking"] = True
    return f"Tracking started. (objects={state['added_count']}, " \
           f"samurai_mode={'ON' if state['added_count']==1 else 'OFF'})"

def on_reset():
    global tracker
    tracker = build_tracker()              # rebuild fresh model+tracker
    set_samurai_mode(tracker, True)        # default back to single-object assumption

    _finalize_writer()
    state.update({
        "first_frame_loaded": False,
        "seeded_any": False,
        "tracking": False,
        "yolo_enabled": True,
        "selected_idx": 0,
        "cands": [],
        "last_frame": None,
        "added_count": 0,
        "video_path": None,
        "video_fps": 30.0,
        "saving_enabled": False,
        "save_name": "segmented_output",
        "save_fps": 30.0,
        "writer": None,
        "writer_size": None,
        "save_path": None,
    })
    return "Reset done."

# -------- Video (pause to seed; then tracking + auto-save) --------
def start_video(video_input, save_basename):
    on_reset()  # fresh session

    state["save_name"] = (save_basename or "").strip() or "segmented_output"
    path = _resolve_video_path(video_input)
    state["video_path"] = path
    if not path or not os.path.exists(path):
        yield None, None
        return

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        yield None, None
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    state["video_fps"] = float(fps)
    state["save_fps"]  = float(fps)
    state["saving_enabled"] = True

    delay = 1.0 / state["video_fps"]

    ok, bgr = cap.read()
    if not ok:
        cap.release()
        yield None, None
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    state["last_frame"] = rgb

    # show first frame (can Accept multiple here)
    frame0 = process_frame(rgb)
    yield frame0, None

    # Wait until user presses Start Tracking
    while not state["tracking"]:
        time.sleep(0.05)
        yield process_frame(state["last_frame"]), None

    # Playback with tracking on (+ auto save)
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        state["last_frame"] = rgb
        out = process_frame(rgb)
        yield out, None
        time.sleep(delay)

    cap.release()
    file_path = _finalize_writer()
    yield None, file_path

# -------- UI --------
with gr.Blocks() as demo:
    gr.Markdown("## SAM2 realtime (this repo) — Add people before **or** during tracking (Webcam or Video)")

    src = gr.Radio(["Webcam", "Video"], value="Webcam", label="Source")
    cam = gr.Image(sources=["webcam"], streaming=True, visible=True, label="Webcam", type="numpy")
    vid = gr.File(label="Video file", visible=False, type="filepath", file_types=["video"])
    save_name = gr.Textbox(label="Output base filename (no extension)", value="segmented_output", visible=False)

    out = gr.Image(label="Output", type="numpy")
    download = gr.File(label="Download (appears after video ends)")

    with gr.Row():
        btn_prev   = gr.Button("Prev")
        btn_accept = gr.Button("Accept (add person)")
        btn_next   = gr.Button("Next")
        btn_toggle = gr.Button("Toggle YOLO")
        btn_start  = gr.Button("Start Tracking")
        btn_reset  = gr.Button("Reset")
        btn_start_vid = gr.Button("Start video")

    status = gr.Markdown("Status: waiting…")

    def toggle_src(choice):
        on_reset()
        return (
            gr.update(visible=(choice=="Webcam")),
            gr.update(visible=(choice=="Video")),
            gr.update(visible=(choice=="Video")),
        )

    src.change(fn=toggle_src, inputs=src, outputs=[cam, vid, save_name])

    cam.stream(fn=process_frame, inputs=cam, outputs=out)

    btn_next.click(fn=on_next, inputs=None, outputs=None)
    btn_prev.click(fn=on_prev, inputs=None, outputs=None)
    btn_accept.click(fn=on_accept, inputs=None, outputs=status)
    btn_toggle.click(fn=on_toggle_yolo, inputs=None, outputs=status)
    btn_start.click(fn=on_start_tracking, inputs=None, outputs=status)
    btn_reset.click(fn=on_reset, inputs=None, outputs=status)

    btn_start_vid.click(fn=start_video, inputs=[vid, save_name], outputs=[out, download])

    gr.Markdown("""
**How to use:**
- **Webcam:** YOLO ON, press **Accept** for each person (you can add many) — you can also keep adding **mid-stream**.
  Then **Start Tracking**.
- **Video:** Upload file → **Start video**. On the first frame Accept several, then **Start Tracking**.
  When it finishes, a download appears.
""")

demo.launch(share=True)