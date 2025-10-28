import os
import cv2
import time
import numpy as np
import torch
import gradio as gr

# ========= Perf knobs (safe on CPU too) =========
if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ========= Import tracker from your repo =========
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

# ========= YOLO (Ultralytics) for proposals =========
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # We'll handle gracefully

# ========= Defaults (hidden from UI) =========
DEFAULT_NUM_OBJECTS = 10          # tracker capacity
DEFAULT_YOLO_MODEL = "yolov8n.pt" # small, downloads automatically
DEFAULT_YOLO_CONF = 0.25          # proposals threshold

# ========= Utilities =========
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
    masks_tensor: torch.Tensor or np.ndarray [B,1,H,W] or [B,H,W] (logits/probs)
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
        masks = masks[:, 0]

    out = frame_rgb.copy()
    H, W = out.shape[:2]
    overlay = np.zeros_like(out, dtype=np.uint8)

    B = masks.shape[0] if masks is not None else 0
    for i in range(B):
        m = masks[i]
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

def run_yolo_boxes_rgb(model, rgb_frame, conf=DEFAULT_YOLO_CONF, classes=None, max_det=100):
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
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if writer.isOpened():
            return writer, path
        writer.release()
    return None, None

# ---- checkpoint loader (post-construction) ----
def _smart_load_ckpt(tracker, ckpt_path, device="cuda"):
    if not ckpt_path:
        return "No checkpoint provided; using tracker defaults."

    map_loc = (device if device in ("cpu", "cuda") else "cpu")
    try:
        sd = torch.load(ckpt_path, map_location=map_loc)
    except Exception as e:
        return f"WARNING: could not read weights file: {e}"

    candidates = []
    if isinstance(sd, dict):
        candidates.extend([sd.get("model"), sd.get("state_dict"), sd])
    else:
        candidates.append(sd)

    targets = [
        ("tracker", tracker),
        ("tracker.model", getattr(tracker, "model", None)),
        ("tracker.module", getattr(tracker, "module", None)),
    ]

    last_err = None
    for payload in candidates:
        if not isinstance(payload, dict):
            continue
        for name, tgt in targets:
            if tgt is None or not hasattr(tgt, "load_state_dict"):
                continue
            try:
                missing, unexpected = tgt.load_state_dict(payload, strict=False)
                return f"Loaded weights into {name} (missing={len(missing)}, unexpected={len(unexpected)})."
            except Exception as e:
                last_err = e
                continue

    for meth in ("load_checkpoint", "load_weights", "load_from_checkpoint"):
        if hasattr(tracker, meth):
            try:
                getattr(tracker, meth)(ckpt_path)
                return f"Loaded weights via tracker.{meth}()."
            except Exception as e:
                last_err = e
                continue

    return f"WARNING: could not load weights from {ckpt_path} (last error: {last_err})"

# ========= App State =========
state = {
    # hidden options / defaults
    "ckpt": None,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_objects": DEFAULT_NUM_OBJECTS,

    # runtime
    "tracker": None,
    "yolo": None,
    "tracking": False,
    "frame_idx": 0,

    # proposals
    "cands": [],
    "selected_idx": 0,
    "last_frame": None,

    # saving (video tab)
    "save_name": "segmented_output",
    "writer": None,
    "save_path": None,
    "save_fps": 30.0,
}

def _ensure_yolo():
    if YOLO is None and state["yolo"] is None:
        raise RuntimeError("Ultralytics not installed. `pip install ultralytics`.")
    if state["yolo"] is None:
        m = YOLO(DEFAULT_YOLO_MODEL)
        m.model_name = DEFAULT_YOLO_MODEL
        state["yolo"] = m

def _build_tracker():
    # Construct WITHOUT checkpoint kwargs (your base doesn't accept them)
    try:
        t = SAM2ObjectTracker(num_objects=int(state["num_objects"]))
    except TypeError:
        t = SAM2ObjectTracker()
    # Post-load checkpoint
    msg = _smart_load_ckpt(t, state["ckpt"], device=state["device"])
    print("[ckpt]", msg)
    return t

def _reset_session():
    if state["writer"] is not None:
        try:
            state["writer"].release()
        except Exception:
            pass
    state.update({
        "tracker": None,
        "yolo": None,
        "tracking": False,
        "frame_idx": 0,
        "cands": [],
        "selected_idx": 0,
        "last_frame": None,
        "writer": None,
        "save_path": None,
    })

# ========= LIVE: per-frame handler (user-driven prompts only) =========
@torch.inference_mode()
def process_live_frame(rgb_frame, ckpt):
    """
    Receives frames from gr.Image(sources=["webcam"], streaming=True)
    User only decides when to press 'Accept (add object)' or 'Start Tracking'.
    """
    state["ckpt"] = ckpt

    if rgb_frame is None or ckpt is None:
        return None

    state["last_frame"] = rgb_frame

    # lazy init
    if state["tracker"] is None:
        state["tracker"] = _build_tracker()
        state["tracking"] = False
        state["frame_idx"] = 0

    # proposals (always visible to allow manual Accept any time)
    try:
        _ensure_yolo()
        cands = state["cands"] = run_yolo_boxes_rgb(state["yolo"], rgb_frame, conf=DEFAULT_YOLO_CONF)
    except Exception as e:
        print("[YOLO warning]", e)
        cands = state["cands"] = []

    if state["tracking"]:
        pred = state["tracker"].track_all_objects(rgb_frame)
        vis = overlay_masks_rgb(rgb_frame, pred.get("pred_masks_high_res"))

        # draw current proposals subtly (so user can add mid-stream)
        if len(cands) > 0:
            bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR).copy()
            state["selected_idx"] = max(0, min(state["selected_idx"], len(cands)-1))
            for j, ((x1,y1),(x2,y2)) in enumerate(cands):
                color = (0,200,255) if j != state["selected_idx"] else (0,255,0)
                cv2.rectangle(bgr, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
            vis = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        state["frame_idx"] += 1
        return vis

    # Not tracking yet → show proposals on raw frame
    draw = rgb_frame
    if len(cands) > 0:
        bgr = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR).copy()
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

    state["tracker"].track_new_object(state["last_frame"], box=bbox)
    return f"Added object (curr_obj_idx={state['tracker'].curr_obj_idx}). You can add more whenever you want."

def ui_start_tracking():
    if state["tracker"] is None or state["last_frame"] is None:
        return "Tracker not ready."
    if state["tracker"].curr_obj_idx == 0:
        return "Add at least one object first (press Accept)."
    state["tracking"] = True
    state["frame_idx"] = 0
    return f"Tracking started with {state['tracker'].curr_obj_idx} object(s)."

def ui_reset():
    _reset_session()
    return "Reset done."

# ========= VIDEO mode (generator) =========
def start_video(video_file, ckpt, save_name):
    _reset_session()
    state["ckpt"] = ckpt
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

    # init tracker + yolo
    try:
        state["tracker"] = _build_tracker()
    except Exception as e:
        yield None, None, f"Tracker init failed: {e}"
        cap.release()
        return
    try:
        _ensure_yolo()
    except Exception as e:
        print("[YOLO warning]", e)
        state["yolo"] = None

    # read first frame, show proposals (paused)
    ok, bgr = cap.read()
    if not ok:
        cap.release()
        yield None, None, "Empty video."
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    state["last_frame"] = rgb

    # prepare writer
    w, h = rgb.shape[1], rgb.shape[0]
    writer, path_out = _try_open_writer(os.path.join("/tmp", state["save_name"]), (w, h), fps)
    state["writer"], state["save_path"] = writer, path_out

    # show first frame with proposals; user can Accept multiple then press start
    draw = process_live_frame(rgb, ckpt)
    yield draw, None, "Paused on first frame. Press **Accept** to add objects, then **Start Tracking (video)**."

    # wait for tracking flag
    while not state["tracking"]:
        time.sleep(0.05)
        yield process_live_frame(state["last_frame"], ckpt), None, "Waiting…"

    # playback with tracking
    delay = 1.0 / float(fps)
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = process_live_frame(rgb, ckpt)
        if state["writer"] is not None and out is not None:
            bgrw = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            state["writer"].write(bgrw)
        yield out, None, f"Tracking… frame {state['frame_idx']}"
        time.sleep(max(0.0, delay * 0.8))

    cap.release()
    if state["writer"] is not None:
        try:
            state["writer"].release()
        except Exception:
            pass
    yield None, state.get("save_path", None), "Done."

# ========= UI (minimal knobs) =========
CKPTS = list_checkpoints()

with gr.Blocks(title="SAM2 Realtime + KF — Live & Video (YOLO-assisted, user prompts only)") as demo:
    gr.Markdown("### SAM2 Realtime + Kalman Filter — add YOLO boxes whenever you want (before **or** during tracking)")

    with gr.Tabs():
        # LIVE TAB
        with gr.Tab("Live (webcam frames)"):
            ckpt = gr.Dropdown(choices=CKPTS, value=CKPTS[0] if CKPTS else None, label="Checkpoint (.pt/.pth)")
            cam = gr.Image(sources=["webcam"], streaming=True, label="Webcam", type="numpy")
            out_live = gr.Image(label="Output (live)")

            with gr.Row():
                btn_prev   = gr.Button("Prev proposal")
                btn_accept = gr.Button("Accept (add object)")
                btn_next   = gr.Button("Next proposal")
                btn_start  = gr.Button("Start Tracking")
                btn_reset  = gr.Button("Reset")

            live_status = gr.Markdown("Status: waiting…")

            cam.stream(fn=process_live_frame, inputs=[cam, ckpt], outputs=out_live)
            btn_next.click(fn=ui_next, inputs=None, outputs=None)
            btn_prev.click(fn=ui_prev, inputs=None, outputs=None)
            btn_accept.click(fn=ui_accept, inputs=None, outputs=live_status)
            btn_start.click(fn=ui_start_tracking, inputs=None, outputs=live_status)
            btn_reset.click(fn=ui_reset, inputs=None, outputs=live_status)

        # VIDEO TAB
        with gr.Tab("Video file"):
            ckpt_v = gr.Dropdown(choices=CKPTS, value=CKPTS[0] if CKPTS else None, label="Checkpoint (.pt/.pth)")
            vid = gr.File(label="Video file", file_types=["video"])
            save_name = gr.Textbox(value="segmented_output", label="Output base filename (/tmp)")

            out_vid = gr.Image(label="Output (video)")
            download = gr.File(label="Download segmented video (appears when finished)")
            status_v = gr.Markdown("")

            btn_start_vid = gr.Button("Start Tracking (video)")
            btn_start_vid.click(
                fn=start_video,
                inputs=[vid, ckpt_v, save_name],
                outputs=[out_vid, download, status_v]
            )

    gr.Markdown("""
**How to use**
- **Live:** pick a checkpoint → YOLO proposals appear → press **Accept** to add an object whenever you want → **Start Tracking**.  
  You can keep pressing **Accept** mid-stream to add more (capacity is 10 by default).
- **Video:** upload clip → first frame pauses → **Accept** a few → **Start Tracking (video)** → a download link appears when done.
""")

if __name__ == "__main__":
    app, local_url, share_url = demo.queue().launch(
        share=True,
        debug=True,
        show_error=True,
        server_name="0.0.0.0",
        server_port=7860,
        prevent_thread_lock=True,
    )
    print("Local URL:", local_url)
    print("Share URL:", share_url)
    while True:
        time.sleep(60)