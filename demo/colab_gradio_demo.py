import os, cv2, time, inspect
import numpy as np
import torch
import gradio as gr
from typing import Optional

# ===== Perf knobs (safe on CPU) =====
if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ===== Import tracker =====
tracker_module = None
_last_import_error = None
for mod in ["sam2.sam2_object_tracker", "sam2.tools.sam2_object_tracker"]:
    try:
        tracker_module = __import__(mod, fromlist=["SAM2ObjectTracker"])
        break
    except Exception as e:
        _last_import_error = e
if tracker_module is None:
    raise ImportError(
        "Couldn't import SAM2ObjectTracker. Expected sam2.sam2_object_tracker. "
        f"Last error: {_last_import_error}"
    )
SAM2ObjectTracker = getattr(tracker_module, "SAM2ObjectTracker")

# ===== YOLO (proposals) =====
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # optional

# ===== New: Hydra/OmegaConf loader =====
from omegaconf import OmegaConf

# ===== Defaults =====
DEFAULT_NUM_OBJECTS = 10
DEFAULT_YOLO_MODEL = "yolov8n.pt"
DEFAULT_YOLO_CONF = 0.25

CFG_CANDIDATES = {
    "tiny": [
        "sam2/configs/sam2/sam2_hiera_t.yaml",
        "configs/sam2.1_hiera_t.yaml", "configs/samurai/sam2.1_hiera_t.yaml",
    ],
    "small": [
        "sam2/configs/sam2/sam2_hiera_s.yaml",
        "configs/sam2.1_hiera_s.yaml", "configs/samurai/sam2.1_hiera_s.yaml",
    ],
    "base_plus": [
        "sam2/configs/sam2/sam2_hiera_b+.yaml",
        "configs/sam2.1_hiera_b+.yaml", "configs/sam2.1_hiera_base_plus.yaml",
        "configs/samurai/sam2.1_hiera_b+.yaml",
    ],
    "large": [
        "sam2/configs/sam2/sam2_hiera_l.yaml",
        "configs/sam2.1_hiera_l.yaml", "configs/samurai/sam2.1_hiera_l.yaml",
    ],
}

def _guess_cfg_for_ckpt(ckpt_path: str) -> tuple[str, str]:
    """Return (cfg_path_to_load, cfg_basename_for_builder)."""
    name = os.path.basename(ckpt_path).lower()
    if "tiny" in name or name.endswith("_t.pt"):
        group = "tiny"
    elif "small" in name or name.endswith("_s.pt"):
        group = "small"
    elif "base_plus" in name or "base-plus" in name or "b_plus" in name or "b+.pt" in name:
        group = "base_plus"
    elif "large" in name or name.endswith("_l.pt"):
        group = "large"
    else:
        group = "small"
    for cand in CFG_CANDIDATES[group]:
        if os.path.exists(cand):
            return cand, os.path.basename(cand)
    # fallback to first candidate (basename may resolve inside builder)
    cand = CFG_CANDIDATES[group][0]
    return cand, os.path.basename(cand)

# ===== Utils =====
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
    if masks_tensor is None:
        return frame_rgb
    try:
        if torch.is_tensor(masks_tensor):
            masks = masks_tensor.detach().float().cpu().numpy()
        else:
            masks = masks_tensor
    except Exception:
        masks = masks_tensor
    if getattr(masks, "ndim", 0) == 4:  # [B,1,H,W]
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
        color = tuple(int(x) for x in rng.integers(60, 255, size=3))
        colored = np.zeros_like(out, dtype=np.uint8); colored[:, :] = color
        mask3 = np.repeat(m_bin[:, :, None], 3, axis=2)
        overlay = np.where(mask3 > 0, colored, overlay)
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out

def run_yolo_boxes_rgb(model, rgb_frame, conf=DEFAULT_YOLO_CONF, classes=None, max_det=100):
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

# ===== Discover & build SAM2 model safely =====
def _load_cfg_object(cfg_path_or_basename: str):
    """Try OmegaConf.load on path; if that fails, try to locate under ./ or ./sam2/configs/sam2/."""
    try:
        if os.path.exists(cfg_path_or_basename):
            return OmegaConf.load(cfg_path_or_basename)
    except Exception:
        pass
    # try common roots
    for root in ["", "sam2/configs/sam2", "configs", "configs/samurai", "sam2/configs"]:
        p = os.path.join(root, cfg_path_or_basename)
        if os.path.exists(p):
            try:
                return OmegaConf.load(p)
            except Exception:
                continue
    return None  # builder may still handle just the basename

def _discover_and_build_sam2(cfg_path_to_load: str, cfg_basename_for_builder: str, ckpt_path: str):
    try:
        mod = __import__("sam2.build_sam", fromlist=["*"])
    except Exception as e:
        raise ImportError(f"Couldn't import sam2.build_sam: {e}")

    cfg_obj = _load_cfg_object(cfg_path_to_load)  # may be None
    trials = []  # (callable, description)
    # Most likely signatures across forks:
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        lower = name.lower()
        if "build" not in lower:
            continue
        # pass cfg object first
        trials += [
            (lambda f=fn: f(cfg_obj, ckpt_path), f"{name}(cfg_obj, ckpt)"),
            (lambda f=fn: f(config=cfg_obj, checkpoint=ckpt_path), f"{name}(config=cfg_obj, checkpoint=...)"),
            # pass basename (Hydra-style resolver)
            (lambda f=fn: f(cfg_basename_for_builder, ckpt_path), f"{name}('basename', ckpt)"),
            (lambda f=fn: f(config=cfg_basename_for_builder, checkpoint=ckpt_path), f"{name}(config='basename', checkpoint=...)"),
            # some builders accept only cfg and then you load state later; try it anyway
            (lambda f=fn: f(cfg_obj), f"{name}(cfg_obj)"),
            (lambda f=fn: f(cfg_basename_for_builder), f"{name}('basename')"),
        ]

    last_err = None
    for caller, desc in trials:
        try:
            built = caller()
        except TypeError:
            continue
        except Exception as e:
            last_err = e
            continue

        # unwrap common holders
        candidates = [built]
        for attr in ["model", "sam", "sam2", "module", "net"]:
            if hasattr(built, attr):
                candidates.append(getattr(built, attr))

        def _has_parts(m):
            needed = ["image_encoder", "memory_attention", "memory_encoder", "prompt_encoder", "mask_decoder"]
            return m is not None and all(hasattr(m, k) for k in needed)

        for m in candidates:
            if _has_parts(m):
                print(f"[sam2] using builder → {desc}")
                return m

    raise RuntimeError(f"Could not build a SAM2 model exposing encoders (last error: {last_err})")

def _build_tracker_with_model(ckpt_path: str):
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise RuntimeError("Choose a valid checkpoint (.pt/.pth).")
    cfg_path, cfg_base = _guess_cfg_for_ckpt(ckpt_path)
    print(f"[sam2] cfg → {os.path.abspath(cfg_path)}")
    model = _discover_and_build_sam2(cfg_path, cfg_base, ckpt_path)
    tracker = SAM2ObjectTracker(
        image_encoder=model.image_encoder,
        memory_attention=model.memory_attention,
        memory_encoder=model.memory_encoder,
        prompt_encoder=model.prompt_encoder,
        mask_decoder=model.mask_decoder,
        num_objects=DEFAULT_NUM_OBJECTS,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    return tracker

# ===== App state =====
state = {
    "ckpt": None,
    "tracker": None,
    "yolo": None,
    "tracking": False,
    "frame_idx": 0,
    "cands": [], "selected_idx": 0, "last_frame": None,
    "save_name": "segmented_output", "writer": None, "save_path": None, "save_fps": 30.0,
}

def _ensure_yolo():
    if YOLO is None and state["yolo"] is None:
        raise RuntimeError("Ultralytics not installed. `pip install ultralytics`.")
    if state["yolo"] is None:
        m = YOLO(DEFAULT_YOLO_MODEL); m.model_name = DEFAULT_YOLO_MODEL
        state["yolo"] = m

def _reset_session():
    if state["writer"] is not None:
        try: state["writer"].release()
        except Exception: pass
    state.update({
        "tracker": None, "yolo": None, "tracking": False, "frame_idx": 0,
        "cands": [], "selected_idx": 0, "last_frame": None,
        "writer": None, "save_path": None,
    })

# ===== Live handler (manual prompts) =====
@torch.inference_mode()
def process_live_frame(rgb_frame, ckpt):
    state["ckpt"] = ckpt
    if rgb_frame is None or ckpt is None:
        return None
    state["last_frame"] = rgb_frame
    if state["tracker"] is None:
        state["tracker"] = _build_tracker_with_model(state["ckpt"])
        state["tracking"] = False; state["frame_idx"] = 0
    try:
        _ensure_yolo()
        cands = state["cands"] = run_yolo_boxes_rgb(state["yolo"], rgb_frame, conf=DEFAULT_YOLO_CONF)
    except Exception as e:
        print("[YOLO warning]", e); cands = state["cands"] = []
    if state["tracking"]:
        pred = state["tracker"].track_all_objects(rgb_frame)
        vis = overlay_masks_rgb(rgb_frame, pred.get("pred_masks_high_res"))
        if len(cands) > 0:
            bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR).copy()
            state["selected_idx"] = max(0, min(state["selected_idx"], len(cands)-1))
            for j, ((x1,y1),(x2,y2)) in enumerate(cands):
                color = (0,200,255) if j != state["selected_idx"] else (0,255,0)
                cv2.rectangle(bgr, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
            vis = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        state["frame_idx"] += 1
        return vis
    # not tracking yet → draw proposals
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
    if state["last_frame"] is None or not state["cands"]:
        return "No candidate available."
    if state["tracker"] is None:
        return "Tracker not initialized."
    idx = max(0, min(state["selected_idx"], len(state["cands"])-1))
    (x1,y1),(x2,y2) = state["cands"][idx]
    bbox = np.array([[[x1, y1], [x2, y2]]], dtype=np.float32)
    state["tracker"].track_new_object(state["last_frame"], box=bbox)
    return f"Added object (curr_obj_idx={state['tracker'].curr_obj_idx})."

def ui_start_tracking():
    if state["tracker"] is None or state["last_frame"] is None:
        return "Tracker not ready."
    if getattr(state["tracker"], "curr_obj_idx", 0) == 0:
        return "Add at least one object first (press Accept)."
    state["tracking"] = True; state["frame_idx"] = 0
    return f"Tracking started with {state['tracker'].curr_obj_idx} object(s)."

def ui_reset():
    _reset_session()
    return "Reset done."

# ===== Video mode =====
def start_video(video_file, ckpt, save_name):
    _reset_session()
    state["ckpt"] = ckpt
    state["save_name"] = (save_name or "").strip() or "segmented_output"
    if video_file is None:
        yield None, None, "Provide a video file."; return
    path = video_file if isinstance(video_file, str) else getattr(video_file, "name", None)
    if not path or not os.path.exists(path):
        yield None, None, "Invalid video."; return
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        yield None, None, "Cannot open video."; return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    state["save_fps"] = float(fps)
    try:
        state["tracker"] = _build_tracker_with_model(state["ckpt"])
    except Exception as e:
        yield None, None, f"Tracker init failed: {e}"; cap.release(); return
    try:
        _ensure_yolo()
    except Exception as e:
        print("[YOLO warning]", e); state["yolo"] = None
    ok, bgr = cap.read()
    if not ok:
        cap.release(); yield None, None, "Empty video."; return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    state["last_frame"] = rgb
    w, h = rgb.shape[1], rgb.shape[0]
    writer, path_out = _try_open_writer(os.path.join("/tmp", state["save_name"]), (w, h), fps)
    state["writer"], state["save_path"] = writer, path_out
    draw = process_live_frame(rgb, ckpt)
    yield draw, None, "Paused on first frame. Press **Accept** to add objects, then **Start Tracking (video)**."
    while not state["tracking"]:
        time.sleep(0.05)
        yield process_live_frame(state["last_frame"], ckpt), None, "Waiting…"
    delay = 1.0 / float(fps)
    while True:
        ok, bgr = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = process_live_frame(rgb, ckpt)
        if state["writer"] is not None and out is not None:
            state["writer"].write(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        yield out, None, f"Tracking… frame {state['frame_idx']}"
        time.sleep(max(0.0, delay * 0.8))
    cap.release()
    if state["writer"] is not None:
        try: state["writer"].release()
        except Exception: pass
    yield None, state.get("save_path", None), "Done."

# ===== UI =====
CKPTS = list_checkpoints()
with gr.Blocks(title="SAM2 Realtime + KF — Live & Video (user prompts only)") as demo:
    gr.Markdown("### Add boxes whenever you want (before **or** during tracking)")
    with gr.Tabs():
        with gr.Tab("Live (webcam)"):
            ckpt = gr.Dropdown(choices=CKPTS, value=CKPTS[0] if CKPTS else None, label="Checkpoint (.pt/.pth)")
            cam = gr.Image(sources=["webcam"], streaming=True, label="Webcam", type="numpy")
            out_live = gr.Image(label="Output (live)")
            with gr.Row():
                btn_prev   = gr.Button("Prev")
                btn_accept = gr.Button("Accept (add object)")
                btn_next   = gr.Button("Next")
                btn_start  = gr.Button("Start Tracking")
                btn_reset  = gr.Button("Reset")
            status = gr.Markdown("Status: waiting…")
            cam.stream(fn=process_live_frame, inputs=[cam, ckpt], outputs=out_live)
            btn_next.click(fn=ui_next, inputs=None, outputs=None)
            btn_prev.click(fn=ui_prev, inputs=None, outputs=None)
            btn_accept.click(fn=ui_accept, inputs=None, outputs=status)
            btn_start.click(fn=ui_start_tracking, inputs=None, outputs=status)
            btn_reset.click(fn=ui_reset, inputs=None, outputs=status)
        with gr.Tab("Video file"):
            ckpt_v = gr.Dropdown(choices=CKPTS, value=CKPTS[0] if CKPTS else None, label="Checkpoint (.pt/.pth)")
            vid = gr.File(label="Video file", file_types=["video"])
            save_name = gr.Textbox(value="segmented_output", label="Output base filename (/tmp)")
            out_vid = gr.Image(label="Output (video)")
            download = gr.File(label="Download (appears when finished)")
            status_v = gr.Markdown("")
            btn_start_vid = gr.Button("Start Tracking (video)")
            btn_start_vid.click(fn=start_video, inputs=[vid, ckpt_v, save_name], outputs=[out_vid, download, status_v])

    gr.Markdown("""
**Quick use**
1) Pick a checkpoint.  
2) Webcam: proposals appear → click **Accept** to add one/more → **Start Tracking**.  
3) Video: upload → first frame pauses → **Accept** some → **Start Tracking (video)** → download appears when done.
""")

if __name__ == "__main__":
    app, local_url, share_url = demo.queue().launch(
        share=True, debug=True, show_error=True,
        server_name="0.0.0.0", server_port=7860, prevent_thread_lock=True
    )
    print("Local URL:", local_url)
    print("Share URL:", share_url)
    while True:
        time.sleep(60)