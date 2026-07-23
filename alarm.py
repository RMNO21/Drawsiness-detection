"""Alarm system with progressive alert levels."""
from __future__ import annotations
import logging, time
from collections import deque
from eye_detector import DrowsinessLevel

logger = logging.getLogger(__name__)

class AlarmManager:
    def __init__(self, config):
        self._volume = config.alarm_volume
        self._muted = False
        self._level = DrowsinessLevel.NORMAL
        self._last_alarm = 0.0
        self._alarm_count = 0
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            self._avail = True
        except Exception:
            self._avail = False

    @property
    def volume(self): return self._volume
    @volume.setter
    def volume(self, v): self._volume = max(0, min(1, v))
    @property
    def is_muted(self): return self._muted
    def mute(self):
        self._muted = True
        self._stop()
    def unmute(self): self._muted = False
    def toggle_mute(self): self.unmute() if self._muted else self.mute()

    def update(self, level: DrowsinessLevel):
        if self._muted or level == DrowsinessLevel.NORMAL:
            self._stop()
            return

        now = time.perf_counter()
        # Always play on escalation (level increase), respect interval only for same level
        is_escalation = level > self._level
        min_interval = {DrowsinessLevel.WARNING: 2.0, DrowsinessLevel.ALARM: 1.0,
                         DrowsinessLevel.EMERGENCY: 0.5}.get(level, 1.5)

        if not is_escalation and now - self._last_alarm < min_interval:
            return

        self._level = level
        self._last_alarm = now
        self._alarm_count += 1

        if not self._avail:
            return
        try:
            import pygame, io, wave, struct, math
            freq = {DrowsinessLevel.WARNING: 600, DrowsinessLevel.ALARM: 800,
                     DrowsinessLevel.EMERGENCY: 1000}[level]
            dur = {DrowsinessLevel.WARNING: 200, DrowsinessLevel.ALARM: 400,
                    DrowsinessLevel.EMERGENCY: 800}[level]
            vol = {DrowsinessLevel.WARNING: 0.3, DrowsinessLevel.ALARM: 0.5,
                    DrowsinessLevel.EMERGENCY: 0.8}[level]
            n = int(44100 * dur / 1000)
            samples = [int(math.sin(2 * math.pi * freq * i / 44100) * vol * self._volume * 32767)
                        for i in range(n)]
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                wf.writeframes(struct.pack(f'<{len(samples)}h', *samples))
            buf.seek(0)
            snd = pygame.mixer.Sound(buf)
            snd.set_volume(self._volume)
            snd.play()
        except Exception as e:
            logger.warning("Alarm: %s", e)

    def _stop(self):
        if self._avail:
            try:
                import pygame
                pygame.mixer.stop()
            except Exception:
                pass

    def close(self): self._stop()
