"""Statistics tracking."""
from __future__ import annotations
import time
from eye_detector import DrowsinessLevel

class Statistics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.perf_counter()
        self.total_frames = 0
        self.face_frames = 0
        self.blinks = 0
        self.warnings = 0
        self.alarms = 0
        self.emergencies = 0
        self.fps_samples = []
        self.longest_closure = 0.0

    def update_frame(self, has_face, blink_count, drowsiness_level, closure_sec, fps):
        self.total_frames += 1
        if has_face:
            self.face_frames += 1
        if blink_count > self.blinks:
            self.blinks = blink_count
        if closure_sec > self.longest_closure:
            self.longest_closure = closure_sec
        if fps > 0:
            self.fps_samples.append(fps)
        if drowsiness_level == DrowsinessLevel.WARNING:
            self.warnings += 1
        elif drowsiness_level == DrowsinessLevel.ALARM:
            self.alarms += 1
        elif drowsiness_level == DrowsinessLevel.EMERGENCY:
            self.emergencies += 1

    def avg_fps(self):
        recent = self.fps_samples[-30:]
        return sum(recent) / max(len(recent), 1)

    def uptime(self):
        s = int(time.perf_counter() - self.start)
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def export_csv(self, path="statistics.csv"):
        import csv
        with open(path, "w", newline="") as f:
            csv.writer(f).writerows([
                ["Metric", "Value"],
                ["Uptime", self.uptime()],
                ["Frames", self.total_frames],
                ["Face detected", self.face_frames],
                ["Blinks", self.blinks],
                ["Warnings", self.warnings],
                ["Alarms", self.alarms],
                ["Emergencies", self.emergencies],
                ["Avg FPS", f"{self.avg_fps():.1f}"],
                ["Longest closure", f"{self.longest_closure:.1f}s"],
            ])
