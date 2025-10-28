import os
import cv2
import gradio as gr
import tempfile
import time
import ffmpeg
import numpy as np

# YOLO (Ultralytics)
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None

# --------------------------
# Import your tracker class
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
        "Couldn't import SAM2ObjectTracker. Expected sam2.sam2_object_tracker."
        f" Last error: {last_import_error}"
    )

SAM2ObjectTracker = getattr(tracker_module, "SAM2ObjectTracker")

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

def overlay_masks(frame_bgr, masks_tensor, alpha=0.45):
    """
    masks_tensor: torch.Tensor or np.ndarray with shape [B, 1, H, W] or [B, H, W]
    """
    if masks_tensor is None:
        return frame_bgr
    try:
        import torch
        if torch.is_tensor(masks_tensor):
            masks = masks_tensor.detach().float().cpu().numpy()
        else:
            masks = masks_tensor
    except Exception:
        masks = masks_tensor

    if masks.ndim == 4:  # [B, 1, H, W]
        masks = masks[:, 0]  # -> [B, H, W]

    out = frame_bgr.copy()
    H, W = out.shape[:2]
    overlay = np.zeros_like(out, dtype=np.uint8)

    B = masks.shape[0] if masks is not None else 0
    for i in range(B):
        m = masks[i]
        if m.shape[0] != H or m.shape[1] != W:
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        m_bin = (m > 0.5).astype(np.uint8)

        # deterministic color per index
        rng = np.random.default_rng((i + 1) * 2654435761 % (2**32))
        color = tuple(int(x) for x in rng.integers(60, 255, size=3))
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

def grab_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to read first frame.")
    return frame

def run_yolo_boxes(model, img_bgr, conf=0.25, classes=None, max_det=100):
    """
    Returns boxes as np.ndarray shape (n,2,2) with absolute pixels [[(x1,y1),(x2,y2)],...]
    """
    if model is None:
        return np.zeros((0, 2, 2), dtype=np.float32)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = model.predict(img_rgb, conf=conf, classes=classes, max_det=max_det, verbose=False)
    if not res or len(res[0].boxes) == 0:
        return np.zeros((0, 2, 2), dtype=np.float32)
    xyxy = res[0].boxes.xyxy.cpu().numpy()  # (n,4)
    boxes = []
    for x1, y1, x2, y2 in xyxy:
        boxes.append([[float(x1), float(y1)], [float(x2), float(y2)]])
    return np.array(boxes, dtype=np.float32)

def encode_video(frames, fps, out_path):
    H, W = frames[0].shape[:2]
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f"{W}x{H}", r=fps)
        .output(out_path, vcodec='libx264', pix_fmt='yuv420p', r=fps, movflags='+faststart', video_bitrate='3M')
        .overwrite_output()
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )
    for f in frames:
        process.stdin.write(f.astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()

# --------------------------
# Core: process a video with YOLO seeding & mid-stream reinjection
# --------------------------
def track_video(
    video_path: str,
    ckpt_path: str,
    num_objects: int = 4,
    device: str = "cuda",
    out_fps: float | None = None,
    yolo_enable: bool = True,
    yolo_model_name: str = "yolov8n.pt",
    yolo_conf: float = 0.25,
    yolo_classes: str = "",   # e.g. "0,2" or "" for all
    reinject_every: int = 60, # frames; 0 disables mid-stream reinjection
):
    """
    Initializes objects on the first frame via YOLO boxes, then runs `track_all_objects`.
    Optionally re-injects YOLO boxes mid-stream every `reinject_every` frames to add new objects.
    """
    # Load YOLO if requested
    yolo_model = None
    yolo_cls_list = None
    if yolo_enable:
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed. `pip install ultralytics`")
        yolo_model = YOLO(yolo_model_name)
        yolo_cls_list = None
        if yolo_classes.strip():
            yolo_cls_list = [int(x.strip()) for x in yolo_classes.split(",") if x.strip().isdigit()]

    # Open video + meta
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = out_fps or in_fps
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create tracker
    # `SAM2ObjectTracker` takes **kwargs forwarded to SAM2Base; try `checkpoint` then fallback to `checkpoint_path`
    tracker = None
    err = None
    try:
        tracker = SAM2ObjectTracker(num_objects=num_objects, device=device, checkpoint=ckpt_path)
    except Exception as e1:
        err = e1
        try:
            tracker = SAM2ObjectTracker(num_objects=num_objects, device=device, checkpoint_path=ckpt_path)
            err = None
        except Exception as e2:
            err = (e1, e2)
    if tracker is None:
        raise RuntimeError(f"Failed to init SAM2ObjectTracker with checkpoint={ckpt_path}. Errors: {err}")

    # Grab & seed on first frame
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Failed to read first frame.")

    frames_vis = []
    initialized = 0
    if yolo_model is not None:
        boxes = run_yolo_boxes(yolo_model, first, conf=yolo_conf, classes=yolo_cls_list)
        if boxes.shape[0] > 0:
            # Add up to capacity
            can_add = max(0, num_objects - tracker.curr_obj_idx)
            if can_add > 0:
                boxes = boxes[:can_add]
                out = tracker.track_new_object(first, box=boxes)
                initialized += boxes.shape[0]
                vis0 = overlay_masks(first, out.get("pred_masks_high_res"))
                frames_vis.append(vis0)
        else:
            # No boxes -> just run once to create memory
            pred = tracker.track_all_objects(first)
            vis0 = overlay_masks(first, pred.get("pred_masks_high_res"))
            frames_vis.append(vis0)
    else:
        pred = tracker.track_all_objects(first)
        vis0 = overlay_masks(first, pred.get("pred_masks_high_res"))
        frames_vis.append(vis0)

    # Iterate remaining frames
    frame_idx = 1
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Optional mid-stream YOLO reinjection
        if yolo_model is not None and reinject_every and frame_idx % reinject_every == 0:
            can_add = max(0, num_objects - tracker.curr_obj_idx)
            if can_add > 0:
                boxes = run_yolo_boxes(yolo_model, frame, conf=yolo_conf, classes=yolo_cls_list)
                if boxes.shape[0] > 0:
                    boxes = boxes[:can_add]
                    tracker.track_new_object(frame, box=boxes)

        pred = tracker.track_all_objects(frame)
        vis = overlay_masks(frame, pred.get("pred_masks_high_res"))
        frames_vis.append(vis)
        frame_idx += 1

    cap.release()
    dt = time.time() - t0

    # Encode to MP4
    tmpdir = tempfile.mkdtemp(prefix="sam2rt_")
    out_mp4 = os.path.join(tmpdir, "tracked.mp4")
    encode_video(frames_vis, fps, out_mp4)
    return out_mp4, f"YOLO init: {initialized} objs | Frames: {frame_idx} | {dt:.1f}s (~{frame_idx/max(dt,1e-6):.1f} FPS)"

# --------------------------
# Gradio UI
# --------------------------
CKPTS = list_checkpoints()

with gr.Blocks(title="SAM2 Realtime + KF (Colab Demo)") as demo:
    gr.Markdown(
        "## SAM2 Realtime + Kalman Filter + YOLO seeding (Colab)\n"
        "Upload a video (or record a short webcam clip), choose a SAM2 checkpoint, and optionally enable YOLO to add bounding boxes\n"
        "both on the first frame and periodically mid-stream.\n"
        "**Note:** The tracker’s API uses `track_new_object(img, box|points|mask)` once per new object, then `track_all_objects(img)` per frame."
    )

    with gr.Tabs():
        with gr.Tab("Upload video"):
            video_in = gr.Video(label="Input video (MP4/MOV/AVI)")
            with gr.Row():
                ckpt = gr.Dropdown(choices=CKPTS, value=CKPTS[0] if CKPTS else None, label="Checkpoint (.pt/.pth)")
                num_obj = gr.Slider(1, 16, value=4, step=1, label="Max objects")
                device = gr.Radio(choices=["cuda", "cpu"], value="cuda", label="Device")
                outfps = gr.Number(value=None, label="Output FPS (optional)")
            with gr.Accordion("YOLO settings", open=True):
                yolo_enable = gr.Checkbox(value=True, label="Use YOLO for boxes")
                yolo_model = gr.Textbox(value="yolov8n.pt", label="YOLO model name/path")
                yolo_conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="YOLO conf")
                yolo_classes = gr.Textbox(value="", label="Class IDs (comma-separated, empty=all)")
                reinject_every = gr.Number(value=60, label="Re-inject every N frames (0=off)")

            go_btn = gr.Button("Run tracking")
            video_out = gr.Video(label="Output (annotated MP4)")
            log = gr.Markdown()

        with gr.Tab("Webcam"):
            cam = gr.Video(sources=["webcam"], streaming=True, label="Webcam (record a short clip)")
            with gr.Row():
                ckpt2 = gr.Dropdown(choices=CKPTS, value=CKPTS[0] if CKPTS else None, label="Checkpoint (.pt/.pth)")
                num_obj2 = gr.Slider(1, 16, value=4, step=1, label="Max objects")
                device2 = gr.Radio(choices=["cuda", "cpu"], value="cuda", label="Device")
                outfps2 = gr.Number(value=None, label="Output FPS (optional)")
            with gr.Accordion("YOLO settings", open=True):
                yolo_enable2 = gr.Checkbox(value=True, label="Use YOLO for boxes")
                yolo_model2 = gr.Textbox(value="yolov8n.pt", label="YOLO model name/path")
                yolo_conf2 = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="YOLO conf")
                yolo_classes2 = gr.Textbox(value="", label="Class IDs (comma-separated, empty=all)")
                reinject_every2 = gr.Number(value=60, label="Re-inject every N frames (0=off)")

            go_btn2 = gr.Button("Run on captured clip")
            video_out2 = gr.Video(label="Output (annotated MP4)")
            log2 = gr.Markdown()

    def run_upload(vid, ckpt_path, k, dev, ofps, ye, ym, yc, ycls, reinj):
        if vid is None:
            return None, "Please provide a video.", None
        if ckpt_path is None:
            return None, "No checkpoint selected/found.", None
        out_path, msg = track_video(
            video_path=vid,
            ckpt_path=ckpt_path,
            num_objects=int(k),
            device=dev,
            out_fps=ofps if ofps not in (None, "") else None,
            yolo_enable=bool(ye),
            yolo_model_name=ym,
            yolo_conf=float(yc),
            yolo_classes=ycls or "",
            reinject_every=int(reinj) if reinj not in (None, "", 0) else 0,
        )
        return out_path, msg, None

    def run_webcam(vid, ckpt_path, k, dev, ofps, ye, ym, yc, ycls, reinj):
        if vid is None:
            return None, "Record a short clip first.", None
        if ckpt_path is None:
            return None, "No checkpoint selected/found.", None
        out_path, msg = track_video(
            video_path=vid,
            ckpt_path=ckpt_path,
            num_objects=int(k),
            device=dev,
            out_fps=ofps if ofps not in (None, "") else None,
            yolo_enable=bool(ye),
            yolo_model_name=ym,
            yolo_conf=float(yc),
            yolo_classes=ycls or "",
            reinject_every=int(reinj) if reinj not in (None, "", 0) else 0,
        )
        return out_path, msg, None

    go_btn.click(
        run_upload,
        inputs=[video_in, ckpt, num_obj, device, outfps, yolo_enable, yolo_model, yolo_conf, yolo_classes, reinject_every],
        outputs=[video_out, log, gr.State()],
    )
    go_btn2.click(
        run_webcam,
        inputs=[cam, ckpt2, num_obj2, device2, outfps2, yolo_enable2, yolo_model2, yolo_conf2, yolo_classes2, reinject_every2],
        outputs=[video_out2, log2, gr.State()],
    )

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)