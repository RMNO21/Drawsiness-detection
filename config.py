"""Configuration."""
from __future__ import annotations
import json, logging
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class AppConfig:
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30
    face_detection_confidence: float = 0.6
    yawning_threshold: float = 0.75
    blink_rate_window_sec: float = 60.0
    alarm_volume: float = 0.7
    graph_history_length: int = 100
    log_level: str = "INFO"
    log_dir: str = "logs"
    sounds_dir: str = "assets/sounds"
    models_dir: str = "assets/models"

    # EAR detection
    ear_threshold: float = 0.21
    ear_calibration_frames: int = 60
    ear_smoothing_alpha: float = 0.5
    ear_min_valid: float = 0.05
    ear_max_valid: float = 0.5

    # Blink detection
    blink_min_duration_ms: float = 60.0
    blink_max_duration_ms: float = 500.0
    blink_consecutive_frames: int = 2

    # Drowsiness thresholds (in seconds of eyes closed)
    drowsiness_warning_sec: float = 0.8
    drowsiness_alarm_sec: float = 1.5
    drowsiness_emergency_sec: float = 2.5

    # Landmark smoothing
    landmark_ema_alpha: float = 0.4

    # Head pose compensation
    head_pose_adjustment: bool = True
    head_pitch_threshold: float = 15.0
    head_yaw_threshold: float = 20.0

    # Performance
    skip_frames: int = 0
    show_landmarks: bool = True
    show_ear_graph: bool = True
    show_cpu_usage: bool = True

    @classmethod
    def load(cls, path: str = "settings.json") -> AppConfig:
        config = cls()
        try:
            with open(path) as f:
                for k, v in json.load(f).items():
                    if hasattr(config, k):
                        setattr(config, k, type(getattr(config, k))(v))
            logger.info("Config loaded from %s", path)
        except FileNotFoundError:
            config.save(path)
        except Exception as e:
            logger.warning("Config error: %s", e)
        return config

    def save(self, path: str = "settings.json") -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
