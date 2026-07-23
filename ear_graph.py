"""EAR history graph widget."""
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QFont, QBrush
from PySide6.QtWidgets import QWidget

class EARGraphWidget(QWidget):
    def __init__(self, parent=None, history_length=100):
        super().__init__(parent)
        self._data = []
        self._threshold = 0.21
        self._max = history_length
        self.setMinimumSize(200, 120)

    def update_data(self, data, threshold):
        self._data = data[-self._max:]
        self._threshold = threshold
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h, m = self.width(), self.height(), 5
        pw, ph = w-2*m, h-2*m
        p.fillRect(self.rect(), QColor(30, 30, 30))

        if not self._data:
            p.end(); return

        mx = max(max(self._data), self._threshold*2, 0.4)
        def yp(v): return m + int(ph*(1-(v/max(mx,0.01))))
        def xp(i): return m + i*pw/max(self._max-1,1)

        # Threshold zone
        ty = yp(self._threshold)
        p.fillRect(QRectF(m, ty, pw, h-m-ty), QBrush(QColor(255,50,50,25)))
        p.fillRect(QRectF(m, m, pw, ty-m), QBrush(QColor(50,200,50,12)))
        pen = QPen(QColor(255,200,0), 1, Qt.PenStyle.DashDotLine)
        p.setPen(pen); p.drawLine(m, ty, w-m, ty)

        # Data line
        if len(self._data) > 1:
            path = QPainterPath()
            off = self._max - len(self._data)
            for i, v in enumerate(self._data):
                x, y = xp(off+i), yp(v)
                path.moveTo(x,y) if i==0 else path.lineTo(x,y)
            c = QColor(255,80,80) if self._data[-1] < self._threshold else QColor(0,200,255)
            p.setPen(QPen(c, 2)); p.drawPath(path)

        # Labels
        p.setPen(QColor(255,255,255))
        p.setFont(QFont("Consolas", 8))
        p.drawText(w-m-70, m+12, f"EAR:{self._data[-1]:.3f}" if self._data else "EAR:--")
        p.setPen(QColor(255,200,0))
        p.drawText(m+2, ty-4, f"Thr:{self._threshold:.3f}")
        p.end()
