"""Camera management."""
from __future__ import annotations
import logging, time
from typing import Optional
import cv2
from config import AppConfig

logger = logging.getLogger(__name__)

class Camera:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps = 0.0
        self._fps_count = 0
        self._fps_timer = time.perf_counter()

    @property
    def is_open(self): return self._cap is not None and self._cap.isOpened()
    @property
    def current_fps(self): return self._fps

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.config.camera_index)
        if not self._cap.isOpened(): return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        logger.info("Camera: %dx%d", int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return True

    def read(self) -> Optional[cv2.ndarray]:
        if not self.is_open: return None
        ret, frame = self._cap.read()
        if not ret or frame is None: return None
        self._fps_count += 1
        if time.perf_counter() - self._fps_timer >= 1.0:
            self._fps = self._fps_count
            self._fps_count = 0
            self._fps_timer = time.perf_counter()
        return frame

    def close(self):
        if self._cap: self._cap.release(); self._cap = None
