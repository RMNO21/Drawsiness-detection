"""Simple, robust eye detector — threshold + hysteresis + time tracking."""
from __future__ import annotations
import logging, time
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
import numpy as np
from config import AppConfig
from face_detector import FaceResult

logger = logging.getLogger(__name__)


class EyeState(IntEnum):
    UNKNOWN = 0
    OPEN = 1
    CLOSING = 2
    CLOSED = 3
    OPENING = 4
    BLINKING = 5


class DrowsinessLevel(IntEnum):
    NORMAL = 0
    WARNING = 1
    ALARM = 2
    EMERGENCY = 3


@dataclass
class BlinkEvent:
    start_ts: float = 0.0
    end_ts: float = 0.0
    min_ear: float = 1.0
    duration_ms: float = 0.0


@dataclass
class EyeMetrics:
    left_ear: float = 0.0
    right_ear: float = 0.0
    avg_ear: float = 0.0
    smoothed_ear: float = 0.0
    ear_velocity: float = 0.0
    eye_state: EyeState = EyeState.UNKNOWN
    blink_count: int = 0
    blink_rate: float = 0.0
    eyes_closed_duration_sec: float = 0.0
    drowsiness_level: DrowsinessLevel = DrowsinessLevel.NORMAL
    calibration_done: bool = False
    threshold: float = 0.0
    drowsiness_score: float = 0.0
    state_machine_state: str = "UNKNOWN"
    confidence: float = 0.0
    consecutive_closed_frames: int = 0
    head_pitch: float = 0.0
    head_yaw: float = 0.0


class AdaptiveEyeDetector:
    """
    Simple algorithm:
      1. Compute EAR
      2. Smooth it
      3. Compare to threshold (with hysteresis)
      4. Track how long eyes have been closed
      5. Trigger alarm based on duration
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config

        # --- Calibration ---
        self._phase = "warmup"
        self._warmup_frames = 15
        self._frame_counter = 0
        self._calibration_vals: list[float] = []
        self._calibration_target = config.ear_calibration_frames

        # --- Threshold (set by calibration) ---
        self._open_ear = 0.35       # median EAR with eyes open
        self._threshold = 0.25      # below this = eyes closed
        self._hysteresis = 0.03     # threshold + hysteresis = eyes open again
        self._sensitivity = 0.5

        # --- Smoothing ---
        self._alpha = 0.5
        self._smoothed: Optional[float] = None

        # --- State ---
        self._eyes_are_closed = False
        self._closed_since: Optional[float] = None
        self._closed_duration = 0.0

        # --- Blink counting ---
        self._blink_count = 0
        self._blink_ts: deque[float] = deque(maxlen=200)

        # --- Drowsiness ---
        self._drowsiness_score = 0.0

        # --- Graph history ---
        self._ear_history: deque[float] = deque(maxlen=config.graph_history_length)
        self._thr_history: deque[float] = deque(maxlen=config.graph_history_length)

        # --- Timing ---
        self._last_ts: Optional[float] = None

        self._apply_sensitivity()
        logger.info("EyeDetector ready (thr=%.3f, hyst=%.3f)", self._threshold, self._hysteresis)

    @property
    def ear_history(self): return list(self._ear_history)
    @property
    def threshold(self): return self._threshold
    @property
    def threshold_history(self): return list(self._thr_history)
    @property
    def is_calibrated(self): return self._phase == "adaptive"
    @property
    def sensitivity(self): return self._sensitivity
    @property
    def blink_markers(self): return []

    def set_sensitivity(self, v: float) -> None:
        self._sensitivity = max(0.0, min(1.0, v))
        self._apply_sensitivity()

    def _apply_sensitivity(self):
        """Adjust threshold and hysteresis based on sensitivity.

        Sensitivity 0% = conservative: high threshold, narrow gap on graph
        Sensitivity 50% = balanced
        Sensitivity 100% = aggressive: low threshold, wide gap on graph
        """
        if self._open_ear <= 0:
            return
        # Threshold: 0.85x down to 0.55x of open_ear
        # sens=0 -> 0.85x (high, close to open EAR)
        # sens=0.5 -> 0.70x (balanced)
        # sens=1 -> 0.55x (low, far from open EAR)
        thr_factor = 0.85 - self._sensitivity * 0.30
        self._threshold = self._open_ear * thr_factor

        # Hysteresis: 0.02x to 0.10x of open_ear
        # sens=0 -> 0.02x (easy to reopen)
        # sens=0.5 -> 0.06x (balanced)
        # sens=1 -> 0.10x (hard to reopen)
        hyst_factor = 0.02 + self._sensitivity * 0.08
        self._hysteresis = self._open_ear * hyst_factor

        logger.info("Sens=%.2f -> threshold=%.3f (%.2fx), hyst=%.3f (open=%.3f)",
                     self._sensitivity, self._threshold, thr_factor,
                     self._hysteresis, self._open_ear)

    def process(self, face: FaceResult, fps: float = 30.0) -> EyeMetrics:
        now = face.frame_timestamp if face.frame_timestamp > 0 else time.perf_counter()
        dt = min(now - (self._last_ts or now), 0.5) if self._last_ts else 0.0
        self._last_ts = now

        m = EyeMetrics(
            threshold=self._threshold,
            calibration_done=self._phase == "adaptive",
            confidence=face.confidence,
            head_pitch=face.head_pitch,
            head_yaw=face.head_yaw,
        )

        # --- Face not detected ---
        if not face.detected:
            self._handle_face_lost(m)
            return m

        # --- Compute EAR ---
        raw_ear = face.avg_ear

        # Outlier rejection: clamp to reasonable range
        raw_ear = max(0.05, min(0.60, raw_ear))

        # Smooth for display only — use raw for detection (faster response)
        if self._smoothed is not None:
            self._smoothed = self._alpha * raw_ear + (1 - self._alpha) * self._smoothed
        else:
            self._smoothed = raw_ear

        ear = raw_ear  # Use RAW for threshold comparison
        m.avg_ear = raw_ear
        m.smoothed_ear = self._smoothed

        # --- Warmup ---
        if self._phase == "warmup":
            self._frame_counter += 1
            self._ear_history.append(ear)
            self._thr_history.append(self._threshold)
            if self._frame_counter >= self._warmup_frames:
                self._phase = "calibrating"
                logger.info("Warmup done, calibrating...")
            return m

        # --- Calibration ---
        if self._phase == "calibrating":
            self._calibration_vals.append(ear)
            self._ear_history.append(ear)
            self._thr_history.append(self._threshold)
            if len(self._calibration_vals) >= self._calibration_target:
                self._calibrate()
            return m

        # --- Normal operation ---
        m.calibration_done = True

        # Head pose compensation: lower threshold when looking down
        adj_threshold = self._threshold
        if self.config.head_pose_adjustment and abs(face.head_pitch) > self.config.head_pitch_threshold:
            factor = max(0.80, 1.0 - (abs(face.head_pitch) - self.config.head_pitch_threshold) / 50.0)
            adj_threshold *= factor

        # --- Core detection: compare EAR to threshold ---
        if not self._eyes_are_closed:
            # Eyes were open — check if they just closed
            if ear < adj_threshold:
                self._eyes_are_closed = True
                self._closed_since = now
                m.eye_state = EyeState.CLOSING
        else:
            # Eyes were closed — check if they opened (use hysteresis)
            if ear > adj_threshold + self._hysteresis:
                # Eyes opened!
                self._eyes_are_closed = False
                dur = now - (self._closed_since or now)
                # Count as blink if it was short (< 500ms)
                if dur < 0.5:
                    self._blink_count += 1
                    self._blink_ts.append(now)
                    m.eye_state = EyeState.BLINKING
                else:
                    m.eye_state = EyeState.OPENING
                self._closed_since = None
                self._closed_duration = 0.0
            else:
                # Still closed
                self._closed_duration = now - (self._closed_since or now)
                m.eye_state = EyeState.CLOSED

        # If eyes are open and not in a special state
        if not self._eyes_are_closed and m.eye_state not in (EyeState.BLINKING, EyeState.OPENING):
            m.eye_state = EyeState.OPEN

        # --- Duration tracking ---
        m.eyes_closed_duration_sec = self._closed_duration

        # --- Drowsiness level based on duration ---
        if self._closed_duration >= self.config.drowsiness_emergency_sec:
            m.drowsiness_level = DrowsinessLevel.EMERGENCY
        elif self._closed_duration >= self.config.drowsiness_alarm_sec:
            m.drowsiness_level = DrowsinessLevel.ALARM
        elif self._closed_duration >= self.config.drowsiness_warning_sec:
            m.drowsiness_level = DrowsinessLevel.WARNING

        # --- Drowsiness score (accumulates while closed, decays while open) ---
        if self._eyes_are_closed:
            self._drowsiness_score = min(self._drowsiness_score + 60.0 * dt, 200)
        else:
            self._drowsiness_score = max(self._drowsiness_score - 30.0 * dt, 0)
        m.drowsiness_score = self._drowsiness_score

        # Also trigger based on score if duration didn't catch it
        if m.drowsiness_level == DrowsinessLevel.NORMAL:
            if self._drowsiness_score >= 50:
                m.drowsiness_level = DrowsinessLevel.WARNING
            elif self._drowsiness_score >= 100:
                m.drowsiness_level = DrowsinessLevel.ALARM
            elif self._drowsiness_score >= 150:
                m.drowsiness_level = DrowsinessLevel.EMERGENCY

        # --- Blinks ---
        m.blink_count = self._blink_count
        m.blink_rate = self._blink_rate()
        m.state_machine_state = "CLOSED" if self._eyes_are_closed else "OPEN"
        m.consecutive_closed_frames = int(self._closed_duration * (fps or 30)) if self._eyes_are_closed else 0

        self._ear_history.append(self._smoothed)
        self._thr_history.append(adj_threshold)

        return m

    def _handle_face_lost(self, m: EyeMetrics):
        """Keep drowsiness state but reset detection."""
        m.eye_state = EyeState.UNKNOWN
        m.drowsiness_score = self._drowsiness_score
        m.state_machine_state = "UNKNOWN"

    def _calibrate(self):
        """Set threshold based on calibration data."""
        vals = np.array(self._calibration_vals)

        # IQR outlier rejection
        q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
        iqr = q3 - q1
        lower = max(q1 - 1.5 * iqr, np.percentile(vals, 5))
        clean = vals[vals >= lower]
        if len(clean) < 10:
            clean = vals

        self._open_ear = float(np.median(clean))
        self._apply_sensitivity()
        self._phase = "adaptive"

        logger.info("Calibrated: open_ear=%.4f, threshold=%.4f (from %d/%d samples)",
                     self._open_ear, self._threshold, len(clean), len(vals))

    def _blink_rate(self):
        now = time.perf_counter()
        self._blink_ts = deque(
            t for t in self._blink_ts if now - t <= self.config.blink_rate_window_sec)
        if len(self._blink_ts) < 2:
            return 0.0
        span = self._blink_ts[-1] - self._blink_ts[0]
        return (len(self._blink_ts) - 1) / max(span, 0.5) * 60

    def reset(self):
        self.__init__(self.config)
