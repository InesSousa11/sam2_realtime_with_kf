import queue
import threading

import cv2
import numpy as np
import torch
from IPython.display import clear_output, display
from PIL import Image


class Visualizer:
    def __init__(self,
                 video_width,
                 video_height,
                 num_masks,
                 save_video=False,
                 output_notebook=False,
                 output_path='tracked_video.mp4'
                 ):

        if not output_notebook:
            self.frame_queue = queue.Queue()
            self.stop_flag = threading.Event()
            self.player_thread = threading.Thread(target=self.play, args=())
            self.player_thread.start()

        self.colors = self.get_mask_colors(num_masks=num_masks)
        self.video_width = video_width
        self.video_height = video_height
        self.save_video = save_video
        self.output_notebook = output_notebook
        self.output_path = output_path

        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.output_path, fourcc, 30,
                                                (self.video_width, self.video_height))
        else:
            self.video_writer = None

    def get_mask_colors(self, num_masks: int = 0):
        colors = np.linspace(0, 1, num_masks)
        colors = np.array([np.array([np.cos(2 * np.pi * color), np.sin(2 * np.pi * color), 1.0]) for color in colors])
        colors = np.clip((colors + 1) / 2 * 255, 0, 255).astype(np.uint8)

        return colors

    def play(self):
        cv2.namedWindow("Video Player", cv2.WINDOW_NORMAL)
        while not self.stop_flag.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Video Player", bgr_frame)

                if self.save_video and self.video_writer is not None:
                    self.video_writer.write(bgr_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except queue.Empty:
                continue

        cv2.destroyAllWindows()

    def resize_mask(self, mask):
        mask = torch.tensor(mask, device='cpu')
        mask = torch.nn.functional.interpolate(mask,
                                               size=(self.video_height, self.video_width),
                                               mode="bilinear",
                                               align_corners=False,
                                               )

        return mask

    def add_frame(self, frame, mask=None, object_id=None):
        frame = frame.copy()
        frame = cv2.resize(frame, (self.video_width, self.video_height))

        if mask is not None:
            mask = self.resize_mask(mask=mask)

            mask = (mask > 0.0).numpy()

            if object_id is not None:
                for idx, id in enumerate(object_id):
                    obj_mask = mask[idx, 0, :, :]
                    frame[obj_mask] = self.colors[id % len(self.colors)]

            else:
                for i in range(mask.shape[0]):
                    obj_mask = mask[i, 0, :, :]
                    frame[obj_mask] = self.colors[i % len(self.colors)]

        if not self.output_notebook:
            if not self.stop_flag.is_set():
                self.frame_queue.put(frame)

        else:
            self.display_frame_in_notebook(frame)

    def display_frame_in_notebook(self, frame):
        rgb_frame = Image.fromarray(frame)
        clear_output(wait=True)
        display(rgb_frame)

    def stop(self):
        if not self.output_notebook:
            self.stop_flag.set()
            self.player_thread.join()

        if self.video_writer is not None:
            self.video_writer.release()