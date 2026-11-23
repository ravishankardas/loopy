"""Floating desktop mascot: a paperclip character drawn with QPainter.

Public API:
    mascot.set_state(State.LISTENING)   # idle / listening / working / speaking
    mascot.show_bubble("Opening Chrome", 3000)  # auto-hides after ms (None = stay)
    mascot.set_level(0.0..1.0)          # mic level, drives the listening ring
    mascot.set_status("Thinking...", Activity.BUSY)  # status chip under the character
    mascot.set_thinking(True)           # keep the "thinking" dots going (e.g. while talking)
    mascot.speak_requested              # signal(str): user asked it to say something
"""

import math
import random
import time
from enum import Enum

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QGuiApplication,
    QColor,
    QCursor,
    QFont,
    QFontMetrics,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QApplication, QMenu, QWidget


def system_drag(widget: QWidget) -> bool:
    """On Wayland, apps can't move their own windows (Loopy just wouldn't
    budge), so hand the drag to the compositor. Returns True if it did; then
    the window moves on its own and partners can't follow. Elsewhere (Windows,
    macOS, X11/XWayland) Loopy moves itself, partners and all."""
    if QGuiApplication.platformName() == "wayland" and widget.windowHandle() is not None:
        return widget.windowHandle().startSystemMove()
    return False


def move_together(lead: QWidget, top_left: QPoint, partners: list[QWidget] = ()) -> None:
    """Move `lead` to `top_left` and its visible partners by the same amount,
    so they travel as one unit, but never past the edge of the screen: the
    whole group stops at the edge (and stays above the taskbar)."""
    group = [lead] + [w for w in partners if w is not lead and w.isVisible()]
    delta = top_left - lead.pos()
    box = group[0].frameGeometry()
    for w in group[1:]:
        box = box.united(w.frameGeometry())
    box.translate(delta)
    screen = (lead.screen() or QApplication.primaryScreen()).availableGeometry()
    dx = max(screen.left() - box.left(), min(0, screen.right() - box.right()))
    dy = max(screen.top() - box.top(), min(0, screen.bottom() - box.bottom()))
    delta += QPoint(dx, dy)
    for w in group:
        w.move(w.pos() + delta)


class State(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    WORKING = "working"
    SPEAKING = "speaking"
    SLEEPING = "sleeping"


class Activity(Enum):
    """Which animated icon the status chip shows."""
    IDLE = "idle"            # softly breathing dot
    LISTENING = "listening"  # dot that pulses with the mic level
    BUSY = "busy"            # spinning ring
    SPEAKING = "speaking"    # equalizer bars
    SLEEPING = "sleeping"    # dim, slow pulse


# Window size and where the clip sits inside it (clip is drawn in a 46x150 box).
WIN_W, WIN_H = 240, 330
CHIP_Y = 304          # vertical centre of the status chip
SCALE = 1.25
CLIP_X = (WIN_W - 46 * SCALE) / 2
CLIP_Y = 100

OUTLINE = QColor(52, 60, 78)
PUPIL = QColor(28, 32, 44)
ACCENT = QColor(0, 184, 169)
BUBBLE_BG = QColor(255, 252, 228)

FPS = 30


def _clip_path() -> QPainterPath:
    """Classic three-loop paperclip, in a 46 x 150 coordinate box."""
    path = QPainterPath(QPointF(14, 110))
    path.lineTo(14, 40)                                   # inner left leg
    path.arcTo(QRectF(14, 30, 20, 20), 180, -180)         # inner top loop
    path.lineTo(34, 130)                                  # middle leg
    path.arcTo(QRectF(0, 113, 34, 34), 0, -180)           # bottom loop
    path.lineTo(0, 20)                                    # outer left leg
    path.arcTo(QRectF(0, -3, 46, 46), 180, -180)          # outer top loop
    path.lineTo(46, 110)                                  # outer right leg
    return path


class Mascot(QWidget):
    speak_requested = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(WIN_W, WIN_H)

        # Optional callback(menu) to add app-specific items to the right-click menu.
        self.menu_extender = None

        self._state = State.IDLE
        self._level = 0.0
        self._bubble_text = ""
        self._bubble_until = 0.0
        self._drag_offset = None
        self.partners: list[QWidget] = []  # windows that move with Loopy (e.g. the game board)
        self._status_text = ""
        self._activity = Activity.IDLE
        self._thinking = False

        self._t0 = time.monotonic()
        self._blink_start = -1.0
        self._next_blink = 2.0

        self._clip = _clip_path()
        self._font = QFont("Segoe UI", 10)
        self._chip_font = QFont("Segoe UI", 8)
        self._chip_font.setWeight(QFont.DemiBold)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(1000 // FPS)

        self._move_to_corner()

    # ------------------------------------------------------------------ API

    def set_state(self, state: State):
        self._state = state

    def state(self) -> State:
        return self._state

    def show_bubble(self, text: str, ms: int | None = 3500):
        self._bubble_text = text
        self._bubble_until = (self._now() + ms / 1000) if ms else math.inf

    def set_level(self, level: float):
        self._level = max(0.0, min(1.0, level))

    def set_status(self, text: str, activity: "Activity"):
        """Always-visible status chip under the character."""
        self._status_text = text
        self._activity = activity

    def set_thinking(self, thinking: bool):
        """Show the bouncing 'thinking' dots regardless of state, e.g. while
        Loopy is already talking but the model is still writing."""
        self._thinking = thinking

    # ------------------------------------------------------------ internals

    def _now(self) -> float:
        return time.monotonic() - self._t0

    def _move_to_corner(self):
        screen = self.screen().availableGeometry()
        self.move(screen.right() - WIN_W - 20, screen.bottom() - WIN_H - 10)

    def _blink_openness(self, t: float) -> float:
        if t >= self._next_blink:
            self._blink_start = t
            self._next_blink = t + random.uniform(2.5, 6.0)
        dt = t - self._blink_start
        if 0 <= dt < 0.16:
            return 1 - math.sin(math.pi * dt / 0.16)
        return 1.0

    def _look_offset(self, t: float, bob: float) -> QPointF:
        """Where the pupils point, in clip coordinates (max ~3.5)."""
        if self._state == State.WORKING:  # glance up and around while busy
            return QPointF(2.5 * math.cos(t * 1.7), -2.5 + math.sin(t * 2.3))
        local = self.mapFromGlobal(QCursor.pos())
        cx = (local.x() - CLIP_X) / SCALE - 24
        cy = (local.y() - CLIP_Y - bob) / SCALE - 55
        dist = math.hypot(cx, cy)
        if dist < 1e-3:
            return QPointF(0, 0)
        mag = min(3.5, dist / 40)
        return QPointF(cx / dist * mag, cy / dist * mag)

    # ------------------------------------------------------------- painting

    def paintEvent(self, _event):
        t = self._now()
        if self._bubble_text and t > self._bubble_until:
            self._bubble_text = ""

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        if self._state == State.SLEEPING:
            bob = math.sin(t * 1.1) * 1.5  # slow, shallow "breathing"
        else:
            amp = 4.5 if self._state == State.SPEAKING else 3.0
            bob = math.sin(t * 2.2) * amp

        if self._bubble_text:
            self._draw_bubble(p)

        p.save()
        p.translate(CLIP_X, CLIP_Y + bob)
        p.scale(SCALE, SCALE)

        # Near-invisible fill so the whole character is clickable/draggable
        # (fully transparent pixels are click-through on Windows).
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 2))
        p.drawEllipse(QRectF(-14, -12, 74, 165))

        if self._state == State.LISTENING:
            self._draw_listen_ring(p, t)
        self._draw_clip(p)
        self._draw_face(p, t, bob)
        if self._state == State.WORKING or self._thinking:
            self._draw_working_dots(p, t)
        if self._state == State.SLEEPING:
            self._draw_zzz(p, t)
        p.restore()

        if self._status_text:
            self._draw_status_chip(p, t)

    def _draw_status_chip(self, p: QPainter, t: float):
        p.setFont(self._chip_font)
        fm = QFontMetrics(self._chip_font)
        icon_w, pad = 16, 9
        w = min(WIN_W - 8, icon_w + 6 + fm.horizontalAdvance(self._status_text) + pad * 2)
        h = 22
        rect = QRectF((WIN_W - w) / 2, CHIP_Y - h / 2, w, h)

        sleeping = self._activity == Activity.SLEEPING
        bg = QColor(30, 36, 50, 150 if sleeping else 215)
        p.setPen(QPen(QColor(255, 255, 255, 40), 1))
        p.setBrush(bg)
        p.drawRoundedRect(rect, h / 2, h / 2)

        icon = QPointF(rect.left() + pad + icon_w / 2, CHIP_Y)
        self._draw_activity_icon(p, icon, t)

        p.setPen(QColor(235, 240, 248, 150 if sleeping else 240))
        text_rect = QRectF(icon.x() + icon_w / 2 + 6, rect.top(), rect.right() - pad - icon.x() - icon_w / 2 - 6, h)
        p.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft,
                   fm.elidedText(self._status_text, Qt.ElideRight, int(text_rect.width())))

    def _draw_activity_icon(self, p: QPainter, c: QPointF, t: float):
        a = self._activity
        p.setBrush(Qt.NoBrush)
        if a == Activity.BUSY:                       # spinning ring
            p.setPen(QPen(QColor(255, 255, 255, 50), 2))
            p.drawEllipse(c, 5.5, 5.5)
            p.setPen(QPen(ACCENT, 2, Qt.SolidLine, Qt.RoundCap))
            start = int(-t * 360 * 1.4) % 360
            p.drawArc(QRectF(c.x() - 5.5, c.y() - 5.5, 11, 11), start * 16, 110 * 16)
        elif a == Activity.SPEAKING:                 # equalizer bars
            p.setPen(Qt.NoPen)
            p.setBrush(ACCENT)
            for i in range(4):
                bar = 2.5 + 7 * abs(math.sin(t * (7 + i * 1.7) + i * 1.3))
                p.drawRoundedRect(QRectF(c.x() - 7 + i * 4, c.y() - bar / 2, 2.6, bar), 1.2, 1.2)
        elif a == Activity.LISTENING:                # dot that pulses with your voice
            r = 3.5 + 3.5 * self._level + 0.8 * math.sin(t * 8)
            halo = QColor(255, 90, 90, 70)
            p.setPen(Qt.NoPen)
            p.setBrush(halo)
            p.drawEllipse(c, r + 2.5, r + 2.5)
            p.setBrush(QColor(255, 95, 95))
            p.drawEllipse(c, r, r)
        elif a == Activity.SLEEPING:                 # dim, slow pulse
            alpha = int(70 + 50 * (0.5 + 0.5 * math.sin(t * 1.2)))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(160, 170, 200, alpha))
            p.drawEllipse(c, 3.5, 3.5)
        else:                                        # idle: softly breathing dot
            k = 0.5 + 0.5 * math.sin(t * 2.0)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(70, 210, 120, int(60 * k)))
            p.drawEllipse(c, 5.5 + k, 5.5 + k)
            p.setBrush(QColor(70, 210, 120))
            p.drawEllipse(c, 3.5, 3.5)

    def _draw_clip(self, p: QPainter):
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(OUTLINE, 7, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawPath(self._clip)

        metal = QLinearGradient(0, 0, 46, 0)
        metal.setColorAt(0.0, QColor(150, 160, 178))
        metal.setColorAt(0.45, QColor(236, 241, 248))
        metal.setColorAt(1.0, QColor(130, 141, 160))
        p.setPen(QPen(metal, 4.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawPath(self._clip)

        p.save()
        p.translate(-0.9, -0.9)
        p.setPen(QPen(QColor(255, 255, 255, 140), 1.2, Qt.SolidLine, Qt.RoundCap))
        p.drawPath(self._clip)
        p.restore()

    def _draw_face(self, p: QPainter, t: float, bob: float):
        if self._state == State.SLEEPING:
            self._draw_sleeping_face(p)
            return
        listening = self._state == State.LISTENING
        working = self._state == State.WORKING

        # Eyebrows
        lift = -4 if listening else 0
        brow_tilt = (-3, 1) if working else (0, 0)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(OUTLINE, 2.4, Qt.SolidLine, Qt.RoundCap))
        for cx, tilt in zip((13, 35), brow_tilt):
            y = 41 + lift + tilt
            brow = QPainterPath(QPointF(cx - 7, y + 1))
            brow.quadTo(cx, y - 3, cx + 7, y + 1)
            p.drawPath(brow)

        # Eyes
        r = 10.5 if listening else 9.5
        openness = self._blink_openness(t)
        look = self._look_offset(t, bob)
        for cx in (13, 35):
            self._draw_eye(p, cx, 55, r, openness, look)

        # Mouth
        self._draw_mouth(p, t)

    def _draw_sleeping_face(self, p: QPainter):
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(OUTLINE, 2.2, Qt.SolidLine, Qt.RoundCap))
        for cx in (13, 35):
            p.setPen(QPen(OUTLINE, 2.2, Qt.SolidLine, Qt.RoundCap))
            p.setBrush(Qt.NoBrush)
            brow = QPainterPath(QPointF(cx - 6, 44))       # relaxed, lower brows
            brow.quadTo(cx, 42, cx + 6, 44)
            p.drawPath(brow)
            p.setPen(QPen(OUTLINE, 1.3))                   # closed eye: white lid...
            p.setBrush(Qt.white)
            p.drawEllipse(QPointF(cx, 55), 9.5, 6.5)
            p.setPen(QPen(OUTLINE, 2.2, Qt.SolidLine, Qt.RoundCap))
            p.setBrush(Qt.NoBrush)
            lid = QPainterPath(QPointF(cx - 7, 55.5))      # ...with a sleepy lash line
            lid.quadTo(cx, 60, cx + 7, 55.5)
            p.drawPath(lid)
        p.setPen(QPen(OUTLINE, 1.6))
        p.setBrush(QColor(150, 40, 55))
        p.drawEllipse(QPointF(24, 77), 2.0, 1.6)            # tiny open mouth

    def _draw_zzz(self, p: QPainter, t: float):
        font = QFont("Segoe UI", 1)
        font.setBold(True)
        for i in range(3):
            phase = (t * 0.3 + i / 3) % 1.0               # each z drifts up and fades
            font.setPointSizeF(6 + phase * 6)
            p.setFont(font)
            color = QColor(ACCENT)
            color.setAlphaF(max(0.0, 1 - phase) * 0.9)
            p.setPen(color)
            p.drawText(QPointF(48 + phase * 14 + math.sin(t * 2 + i) * 2, 22 - phase * 30), "z")

    def _draw_eye(self, p, cx, cy, r, openness, look: QPointF):
        p.setPen(QPen(OUTLINE, 1.3))
        p.setBrush(Qt.white)
        p.drawEllipse(QPointF(cx, cy), r, max(r * openness, 0.7))
        if openness < 0.35:
            return
        px, py = cx + look.x(), cy + look.y() * openness
        p.setPen(Qt.NoPen)
        p.setBrush(PUPIL)
        p.drawEllipse(QPointF(px, py), 4.2, 4.2 * min(1.0, openness * 1.2))
        p.setBrush(Qt.white)
        p.drawEllipse(QPointF(px + 1.4, py - 1.4), 1.3, 1.3)

    def _draw_mouth(self, p: QPainter, t: float):
        cx, cy = 24, 77
        state = self._state
        if state == State.SPEAKING:
            h = 1.2 + 4.5 * abs(math.sin(t * 13)) * (0.6 + 0.4 * math.sin(t * 3.1))
            p.setPen(QPen(OUTLINE, 1.4))
            p.setBrush(QColor(150, 40, 55))
            p.drawEllipse(QPointF(cx, cy), 5.5, h)
        elif state == State.LISTENING:
            p.setPen(QPen(OUTLINE, 1.6))
            p.setBrush(QColor(150, 40, 55))
            p.drawEllipse(QPointF(cx, cy), 2.4, 3.0)
        elif state == State.WORKING:
            p.setPen(QPen(OUTLINE, 2.0, Qt.SolidLine, Qt.RoundCap))
            p.setBrush(Qt.NoBrush)
            mouth = QPainterPath(QPointF(cx - 5, cy))
            mouth.quadTo(cx - 2, cy - 2, cx, cy)
            mouth.quadTo(cx + 2, cy + 2, cx + 5, cy)
            p.drawPath(mouth)
        else:
            p.setPen(QPen(OUTLINE, 2.0, Qt.SolidLine, Qt.RoundCap))
            p.setBrush(Qt.NoBrush)
            mouth = QPainterPath(QPointF(cx - 5.5, cy - 1))
            mouth.quadTo(cx, cy + 4.5, cx + 5.5, cy - 1)
            p.drawPath(mouth)

    def _draw_listen_ring(self, p: QPainter, t: float):
        center = QPointF(23, 70)
        for i in range(2):
            phase = (t * 0.9 + i * 0.5) % 1.0
            radius = 48 + phase * 18 + self._level * 14
            color = QColor(ACCENT)
            color.setAlphaF(0.55 * (1 - phase))
            p.setPen(QPen(color, 2.5))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(center, radius, radius)

    def _draw_working_dots(self, p: QPainter, t: float):
        p.setPen(Qt.NoPen)
        p.setBrush(ACCENT)
        for i in range(3):
            y = 18 - abs(math.sin(t * 6 - i * 0.7)) * 5
            p.drawEllipse(QPointF(56 + i * 7, y), 2.2, 2.2)

    def _draw_bubble(self, p: QPainter):
        p.setFont(self._font)
        fm = QFontMetrics(self._font)
        inner_w = WIN_W - 44
        text_rect = fm.boundingRect(
            QRect(0, 0, inner_w, 1000), Qt.TextWordWrap | Qt.AlignCenter,
            self._bubble_text,
        )
        h = min(text_rect.height() + 20, 80)
        rect = QRectF(10, 82 - h, WIN_W - 20, h)

        cx = CLIP_X + 23 * SCALE
        shape = QPainterPath()
        shape.addRoundedRect(rect, 12, 12)
        tail = QPainterPath(QPointF(cx - 8, rect.bottom() - 1))
        tail.lineTo(cx + 2, rect.bottom() + 11)
        tail.lineTo(cx + 8, rect.bottom() - 1)
        tail.closeSubpath()
        shape = shape.united(tail)

        p.setPen(QPen(OUTLINE, 1.5))
        p.setBrush(BUBBLE_BG)
        p.drawPath(shape)
        p.setPen(PUPIL)
        p.drawText(
            rect.adjusted(12, 8, -12, -8),
            Qt.TextWordWrap | Qt.AlignCenter,
            self._bubble_text,
        )

    # ---------------------------------------------------------- interaction

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if system_drag(self):  # Wayland: the compositor moves the window
                return
            self._drag_offset = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )

    def mouseMoveEvent(self, event):
        if self._drag_offset is not None and event.buttons() & Qt.LeftButton:
            move_together(self, event.globalPosition().toPoint() - self._drag_offset, self.partners)

    def mouseReleaseEvent(self, _event):
        self._drag_offset = None

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        for state in State:
            action = QAction(state.value.capitalize(), menu, checkable=True)
            action.setChecked(state == self._state)
            action.triggered.connect(lambda _=False, s=state: self.set_state(s))
            menu.addAction(action)
        menu.addSeparator()
        menu.addAction("Say hello", lambda: self.speak_requested.emit("Hi! I'm Loopy, nice to meet you."))
        if self.menu_extender:
            self.menu_extender(menu)
        menu.addSeparator()
        menu.addAction("Quit", QApplication.quit)
        menu.exec(event.globalPos())
