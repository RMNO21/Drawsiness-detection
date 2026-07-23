"""Main GUI application with enhanced visualization."""
from __future__ import annotations
import logging, time, sys, cv2
from typing import Optional
from PySide6.QtCore import Qt, QTimer, Slot, QSize
from PySide6.QtGui import QImage, QPixmap, QFont, QColor
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QGridLayout, QSlider, QComboBox,
    QSplitter, QFrame, QStatusBar, QMessageBox, QFileDialog)

from config import AppConfig
from camera import Camera
from face_detector import FaceDetector
from eye_detector import AdaptiveEyeDetector, EyeState, DrowsinessLevel
from alarm import AlarmManager
from statistics import Statistics
from ear_graph import EARGraphWidget

logger = logging.getLogger(__name__)

_STATE_COLORS = {
    "OPEN": (0, 200, 0),
    "CLOSING": (0, 165, 255),
    "CLOSED": (0, 0, 255),
    "OPENING": (0, 200, 200),
    "UNKNOWN": (128, 128, 128),
}

class DrowsinessApp(QMainWindow):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self._running = False
        self._paused = False
        self._frame_count = 0
        self._camera = Camera(config)
        self._face_detector: Optional[FaceDetector] = None
        self._eye_detector: Optional[AdaptiveEyeDetector] = None
        self._alarm = AlarmManager(config)
        self._stats = Statistics()
        self._process_timer = QTimer(self)
        self._process_timer.timeout.connect(self._process_frame)
        self._video_target = QSize(640, 480)
        self._build_ui()
        self.setWindowTitle("Drowsiness Detection System")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 750)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)
        main.setContentsMargins(8, 8, 8, 8)

        # Left: video
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self._video = QLabel()
        self._video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video.setMinimumSize(640, 480)
        self._video.setSizePolicy(
            self._video.sizePolicy().horizontalPolicy(),
            self._video.sizePolicy().verticalPolicy())
        self._video.setStyleSheet("background:#1a1a1e; border-radius:8px;")
        self._video.setScaledContents(False)
        ll.addWidget(self._video, stretch=1)

        # Controls
        ctrl = QHBoxLayout()
        self._start_btn = QPushButton("Start")
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.setEnabled(False)
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.clicked.connect(self._on_pause)
        self._pause_btn.setEnabled(False)
        for b in [self._start_btn, self._stop_btn, self._pause_btn]:
            b.setMinimumHeight(32)
            b.setMinimumWidth(80)
            ctrl.addWidget(b)
        ll.addLayout(ctrl)

        # Right panel
        right = QWidget()
        right.setMinimumWidth(300)
        right.setMaximumWidth(340)
        rl = QVBoxLayout(right)
        rl.setSpacing(6)

        # Status
        sg = QGroupBox("Status")
        sg.setStyleSheet("QGroupBox { font-weight: bold; }")
        sl = QGridLayout(sg)
        sl.setSpacing(4)
        self._face_ind = QLabel("  Face  ")
        self._face_ind.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._face_ind.setMinimumHeight(28)
        self._face_ind.setStyleSheet(
            "background:#2a2a32; border-radius:4px; padding:4px; color:#888; font-size:11px;")
        sl.addWidget(self._face_ind, 0, 0)
        self._state_lbl = QLabel("OPEN")
        self._state_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._state_lbl.setMinimumHeight(28)
        self._state_lbl.setStyleSheet(
            "font-size:14px; font-weight:bold; padding:4px; background:#2a2a32; "
            "border-radius:4px; color:#00cc44;")
        sl.addWidget(self._state_lbl, 0, 1)
        self._status_lbl = QLabel("Ready")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setMinimumHeight(36)
        self._status_lbl.setStyleSheet(
            "font-size:16px; font-weight:bold; padding:6px; background:#2a2a32; "
            "border-radius:4px; color:#888;")
        sl.addWidget(self._status_lbl, 1, 0, 1, 2)
        rl.addWidget(sg)

        # Metrics - use a simple 2-column grid with fixed row heights
        mg = QGroupBox("Metrics")
        mg.setStyleSheet("QGroupBox { font-weight: bold; }")
        ml = QGridLayout(mg)
        ml.setSpacing(4)
        ml.setContentsMargins(8, 8, 8, 8)
        self._metrics = {}
        metric_defs = [
            ("FPS", "0"), ("EAR", "0.000"), ("Threshold", "0.000"),
            ("Velocity", "0.000"), ("Confidence", "0.00"),
            ("Blinks", "0"), ("Closure", "0.0s"), ("Score", "0.0"),
            ("Level", "NORMAL"), ("Consec", "0"), ("Pitch", "0"),
            ("Yaw", "0"), ("Uptime", "00:00:00"),
        ]
        for i, (name, init) in enumerate(metric_defs):
            row, col = divmod(i, 2)
            f = QFrame()
            f.setMinimumHeight(42)
            f.setStyleSheet(
                "QFrame { background:#2a2a32; border-radius:4px; }")
            fl = QVBoxLayout(f)
            fl.setContentsMargins(8, 4, 8, 4)
            fl.setSpacing(1)
            nl = QLabel(name)
            nl.setStyleSheet("color:#999; font-size:10px; border:none; background:transparent;")
            vl = QLabel(init)
            vl.setStyleSheet("color:#fff; font-size:14px; font-weight:bold; border:none; background:transparent;")
            vl.setMinimumHeight(18)
            fl.addWidget(nl)
            fl.addWidget(vl)
            self._metrics[name] = vl
            ml.addWidget(f, row, col)
        rl.addWidget(mg)

        # Graph
        self._graph = EARGraphWidget(history_length=self.config.graph_history_length)
        self._graph.setMinimumHeight(120)
        rl.addWidget(self._graph)

        # Sensitivity
        sg2 = QGroupBox("Sensitivity")
        sg2.setStyleSheet("QGroupBox { font-weight: bold; }")
        sgl = QHBoxLayout(sg2)
        sgl.addWidget(QLabel("Low"))
        self._sens_slider = QSlider(Qt.Orientation.Horizontal)
        self._sens_slider.setRange(0, 100)
        self._sens_slider.setValue(50)
        self._sens_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._sens_slider.setTickInterval(10)
        self._sens_slider.valueChanged.connect(self._on_sensitivity)
        sgl.addWidget(self._sens_slider)
        sgl.addWidget(QLabel("High"))
        self._sens_lbl = QLabel("50%")
        self._sens_lbl.setStyleSheet("color:#00ccff; font-weight:bold; min-width:35px;")
        self._sens_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sgl.addWidget(self._sens_lbl)
        rl.addWidget(sg2)

        # Volume
        vg = QGroupBox("Volume")
        vg.setStyleSheet("QGroupBox { font-weight: bold; }")
        vgl = QHBoxLayout(vg)
        self._mute_btn = QPushButton("Mute")
        self._mute_btn.setCheckable(True)
        self._mute_btn.clicked.connect(
            lambda: self._alarm.toggle_mute() or self._mute_btn.setText(
                "Unmute" if self._alarm.is_muted else "Mute"))
        vgl.addWidget(self._mute_btn)
        self._vol = QSlider(Qt.Orientation.Horizontal)
        self._vol.setRange(0, 100)
        self._vol.setValue(int(self.config.alarm_volume * 100))
        self._vol.valueChanged.connect(lambda v: setattr(self._alarm, 'volume', v / 100.0))
        vgl.addWidget(self._vol)
        rl.addWidget(vg)

        rl.addStretch()

        # Splitter
        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.addWidget(left)
        sp.addWidget(right)
        sp.setStretchFactor(0, 3)
        sp.setStretchFactor(1, 1)
        main.addWidget(sp)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

    def _on_sensitivity(self, v):
        self._sens_lbl.setText(f"{v}%")
        if self._eye_detector:
            self._eye_detector.set_sensitivity(v / 100.0)

    def _on_start(self):
        if self._running:
            return
        try:
            self._face_detector = FaceDetector(self.config)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return
        self._eye_detector = AdaptiveEyeDetector(self.config)
        if not self._camera.open():
            QMessageBox.warning(self, "Camera Error", "Cannot open camera")
            return
        self._running = True
        self._paused = False
        self._frame_count = 0
        self._stats.reset()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._pause_btn.setEnabled(True)
        self._process_timer.start(max(1, int(1000 / self.config.target_fps)))
        self._status_bar.showMessage("Monitoring active")
        logger.info("Detection started")

    def _on_stop(self):
        if not self._running:
            return
        self._running = False
        self._process_timer.stop()
        self._camera.close()
        self._alarm.close()
        if self._face_detector:
            self._face_detector.close()
            self._face_detector = None
        self._eye_detector = None
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._pause_btn.setEnabled(False)
        self._video.clear()
        self._video.setText("Detection stopped")
        self._status_bar.showMessage("Stopped")

    def _on_pause(self):
        if not self._running:
            return
        self._paused = not self._paused
        self._pause_btn.setText("Resume" if self._paused else "Pause")

    @Slot()
    def _process_frame(self):
        if not self._running or self._paused:
            return
        ts = time.perf_counter()
        frame = self._camera.read()
        if frame is None:
            return

        h, w = frame.shape[:2]
        if w > 640 or h > 480:
            frame = cv2.resize(frame, (640, 480))

        assert self._face_detector is not None
        try:
            face = self._face_detector.detect(frame, ts)
        except Exception as e:
            logger.warning("Detection error: %s", e)
            from face_detector import FaceResult
            face = FaceResult(frame_timestamp=ts)

        assert self._eye_detector is not None
        metrics = self._eye_detector.process(face, self._camera.current_fps)
        self._stats.update_frame(
            face.detected, metrics.blink_count, metrics.drowsiness_level,
            metrics.eyes_closed_duration_sec, self._camera.current_fps)
        self._alarm.update(metrics.drowsiness_level)

        # Draw overlay
        vis = frame.copy()
        if face.detected:
            vis = self._face_detector.draw_landmarks(vis, face)
        vis = self._draw_overlay(vis, face, metrics)

        # Display - scale to FIXED target size to avoid layout feedback loop
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                       rgb.strides[0], QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        # Scale to the video label's current size, but only if it changed
        label_size = self._video.size()
        if label_size != self._video_target:
            self._video_target = label_size
        scaled = pix.scaled(self._video_target,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        self._video.setPixmap(scaled)

        # Update metrics panel (only update text, avoid full stylesheet reapply)
        self._set_metric("FPS", f"{self._camera.current_fps:.1f}")
        ear_color = "#00cc44" if metrics.smoothed_ear > self._eye_detector.threshold else "#ff4444"
        self._set_metric("EAR", f"{metrics.smoothed_ear:.3f}", ear_color)
        self._set_metric("Threshold", f"{metrics.threshold:.3f}")

        vel = metrics.ear_velocity
        vel_color = "#ff4444" if vel < -0.5 else ("#00cc44" if vel > 0.5 else "#888")
        self._set_metric("Velocity", f"{vel:+.3f}", vel_color)
        self._set_metric("Confidence", f"{metrics.confidence:.2f}")
        self._set_metric("Blinks", str(metrics.blink_count))
        self._set_metric("Closure", f"{metrics.eyes_closed_duration_sec:.1f}s")
        self._set_metric("Score", f"{metrics.drowsiness_score:.1f}")
        self._set_metric("Pitch", f"{metrics.head_pitch:.0f}")
        self._set_metric("Yaw", f"{metrics.head_yaw:.0f}")

        lc = {
            DrowsinessLevel.NORMAL: "#00cc44",
            DrowsinessLevel.WARNING: "#ffcc00",
            DrowsinessLevel.ALARM: "#ff8800",
            DrowsinessLevel.EMERGENCY: "#ff3333",
        }
        ln = {
            DrowsinessLevel.NORMAL: "NORMAL",
            DrowsinessLevel.WARNING: "WARNING",
            DrowsinessLevel.ALARM: "ALARM",
            DrowsinessLevel.EMERGENCY: "EMERGENCY",
        }
        self._set_metric("Level", ln.get(metrics.drowsiness_level, ""), lc.get(metrics.drowsiness_level, "#fff"))
        self._set_metric("Consec", str(metrics.consecutive_closed_frames))
        self._set_metric("Uptime", self._stats.uptime())

        # Face indicator
        fc = '#0a3a0a' if face.detected else '#2a2a32'
        ftc = '#00cc44' if face.detected else '#888'
        self._face_ind.setStyleSheet(
            f"background:{fc}; border-radius:4px; padding:4px; color:{ftc}; font-size:11px;")

        # State machine label
        sm_state = metrics.state_machine_state
        sm_color = _STATE_COLORS.get(sm_state, (128, 128, 128))
        sc_hex = f"#{sm_color[0]:02x}{sm_color[1]:02x}{sm_color[2]:02x}"
        self._state_lbl.setText(sm_state)
        self._state_lbl.setStyleSheet(
            f"font-size:14px; font-weight:bold; padding:4px; background:#2a2a32; "
            f"border-radius:4px; color:{sc_hex};")

        # Status text
        if metrics.drowsiness_level == DrowsinessLevel.EMERGENCY:
            self._status_lbl.setText("WAKE UP!")
            self._status_lbl.setStyleSheet(
                "font-size:16px; font-weight:bold; padding:6px; background:#3a0a0a; "
                "border-radius:4px; color:#ff3333;")
        elif metrics.drowsiness_level == DrowsinessLevel.ALARM:
            self._status_lbl.setText("ALARM - Eyes Closed!")
            self._status_lbl.setStyleSheet(
                "font-size:16px; font-weight:bold; padding:6px; background:#3a2a0a; "
                "border-radius:4px; color:#ff8800;")
        elif metrics.drowsiness_level == DrowsinessLevel.WARNING:
            self._status_lbl.setText("WARNING")
            self._status_lbl.setStyleSheet(
                "font-size:16px; font-weight:bold; padding:6px; background:#3a3a0a; "
                "border-radius:4px; color:#ffcc00;")
        elif not face.detected:
            self._status_lbl.setText("No Face")
            self._status_lbl.setStyleSheet(
                "font-size:16px; font-weight:bold; padding:6px; background:#2a2a32; "
                "border-radius:4px; color:#888;")
        else:
            self._status_lbl.setText("Monitoring")
            self._status_lbl.setStyleSheet(
                "font-size:16px; font-weight:bold; padding:6px; background:#0a2a0a; "
                "border-radius:4px; color:#00cc44;")

        # Video overlay text
        vc = _STATE_COLORS.get(metrics.state_machine_state, (180, 180, 180))
        cv2.putText(vis, f"FPS:{self._camera.current_fps:.0f}", (10, 28),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.putText(vis, f"EAR:{metrics.smoothed_ear:.3f}", (120, 28),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.putText(vis, f"State:{metrics.state_machine_state}", (260, 28),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, vc, 2)
        cv2.putText(vis, f"VEL:{metrics.ear_velocity:+.3f}", (10, 55),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(vis, f"Blinks:{metrics.blink_count}", (10, h - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        if metrics.eyes_closed_duration_sec > 0:
            cd_color = (0, 0, 255) if metrics.eyes_closed_duration_sec > 2.0 else (0, 165, 255)
            cv2.putText(vis, f"Closed:{metrics.eyes_closed_duration_sec:.1f}s",
                         (w - 200, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cd_color, 2)

        self._graph.update_data(self._eye_detector.ear_history, self._eye_detector.threshold)

    def _draw_overlay(self, frame, face, metrics):
        h, w = frame.shape[:2]
        if metrics.drowsiness_level == DrowsinessLevel.EMERGENCY:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
            cv2.putText(frame, "WAKE UP!", (w // 2 - 150, h // 2),
                         cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        elif metrics.drowsiness_level == DrowsinessLevel.ALARM:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 140, 255), 4)
        elif metrics.drowsiness_level == DrowsinessLevel.WARNING:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 200, 255), 2)
        return frame

    def _set_metric(self, name, value, color="#fff"):
        if name in self._metrics:
            lbl = self._metrics[name]
            if lbl.text() != value:
                lbl.setText(value)
            old = lbl.styleSheet()
            target = f"color:{color}; font-size:14px; font-weight:bold; border:none; background:transparent;"
            if old != target:
                lbl.setStyleSheet(target)

    def closeEvent(self, event):
        self._on_stop()
        self._stats.export_csv("statistics.csv")
        event.accept()
