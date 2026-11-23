"""Tic-tac-toe with Loopy: the rules (TicTacToe) and the board window.

Cells are numbered like a phone keypad, so they're easy to say:

    1 | 2 | 3
    ---------
    4 | 5 | 6
    ---------
    7 | 8 | 9

The user is X, Loopy is O. Loopy's moves come from the language model;
the code here is the referee (is the cell free? who won?).

    game = TicTacToe()
    game.place(5, "X")
    game.is_free(5)   -> False
    game.winner       -> None / "X" / "O" / "draw"

    board = BoardWindow()
    board.open_next_to(mascot)
    board.place(5, "X")                  # drawn stroke by stroke
    board.set_status("Loopy is thinking...")
    board.show_win((0, 4, 8))            # line through cells 1, 5, 9 (0-based)
    board.set_scores(you=1, draws=0, loopy=0)
"""

import math
import random
import time

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF, QRadialGradient
from PySide6.QtWidgets import QWidget

# The board: black ink on white paper.
CARD = QColor(255, 255, 255)
CARD_EDGE = QColor(0, 0, 0, 45)
INK = QColor(20, 20, 24)            # grid, numbers, X's and O's, win line
TEXT = QColor(20, 20, 24)
SOFT_TEXT = QColor(20, 20, 24, 150)  # status line, separators
NUMBER = QColor(20, 20, 24, 200)     # numbers in empty cells
GRID = INK
# The celebration stays colorful: coral for you, teal for Loopy.
X_COLOR = QColor(255, 120, 110)
O_COLOR = QColor(0, 200, 185)

WIN_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
CELL_NAMES = {1: "top-left", 2: "top-middle", 3: "top-right", 4: "middle-left", 5: "the center",
              6: "middle-right", 7: "bottom-left", 8: "bottom-middle", 9: "bottom-right"}

DRAW_TIME = 0.35   # seconds to draw an X or O
LINE_TIME = 0.45   # seconds to draw the winning line
OPEN_TIME = 0.30   # pop-in animation


class TicTacToe:
    """The rules. No Qt here, so it's easy to test."""

    def __init__(self):
        self.cells: list[str | None] = [None] * 9
        self.winner: str | None = None             # "X", "O" or "draw"
        self.line: tuple[int, int, int] | None = None  # winning cells, 0-based

    @property
    def over(self) -> bool:
        return self.winner is not None

    def free(self) -> list[int]:
        """Free cells, 1-9."""
        return [i + 1 for i, c in enumerate(self.cells) if c is None]

    def is_free(self, cell: int | None) -> bool:
        return cell is not None and 1 <= cell <= 9 and self.cells[cell - 1] is None and not self.over

    def why_not(self, cell: int | None) -> str:
        """Why `cell` can't be played, in words."""
        if cell is None or not 1 <= cell <= 9:
            return "that's not a cell (cells are 1 to 9)"
        if self.over:
            return "the game is over"
        return f"cell {cell} is taken by {self.cells[cell - 1]}"

    def place(self, cell: int, mark: str) -> None:
        if not self.is_free(cell):
            raise ValueError(self.why_not(cell))
        self.cells[cell - 1] = mark
        for line in WIN_LINES:
            if self.cells[line[0]] is not None and all(self.cells[i] == self.cells[line[0]] for i in line):
                self.winner, self.line = self.cells[line[0]], line
                return
        if None not in self.cells:
            self.winner = "draw"

    def describe(self) -> str:
        """The board as text for the model, free cells shown as their number."""
        show = [c or str(i + 1) for i, c in enumerate(self.cells)]
        return "\n".join(" | ".join(show[r * 3:r * 3 + 3]) for r in range(3))


def _ease_out_back(p: float) -> float:
    c = 1.7
    return 1 + (c + 1) * (p - 1) ** 3 + c * (p - 1) ** 2


class BoardWindow(QWidget):
    W, H = 300, 392
    CELL = 80
    GRID_X, GRID_Y = 30, 84

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(self.W, self.H)
        self._title = QFont("Segoe UI", 13, QFont.Weight.Bold)
        self._small = QFont("Segoe UI", 9)
        self._number = QFont("Segoe UI", 22, QFont.Weight.Bold)
        self._corner = QFont("Segoe UI", 7)
        self._drag = None
        self.partners: list[QWidget] = []  # Loopy, while the board is open: they move together
        self.reset()
        self.status = "Your turn · you're X"
        self.scores = (0, 0, 0)
        self._opened = time.monotonic()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(33)

    # ------------------------------------------------------------------ API

    def reset(self) -> None:
        self.cells: list[str | None] = [None] * 9
        self._placed = [0.0] * 9          # when each mark appeared (animation)
        self._win: tuple[int, int, int] | None = None
        self._win_at = 0.0

    def place(self, cell: int, mark: str) -> None:
        """Put "X" or "O" in `cell` (1-9)."""
        self.cells[cell - 1] = mark
        self._placed[cell - 1] = time.monotonic()

    def show_win(self, line: tuple[int, int, int]) -> None:
        """Draw the winning line through three cells (0-based indexes)."""
        self._win, self._win_at = line, time.monotonic()

    def set_status(self, text: str) -> None:
        self.status = text

    def set_scores(self, you: int, draws: int, loopy: int) -> None:
        self.scores = (you, draws, loopy)

    def open_next_to(self, widget: QWidget) -> None:
        """Show the board beside Loopy (left, or right if there's no room),
        bottom-aligned, and tie the two together: dragging either moves both,
        and neither goes past the edge of the screen."""
        from loopy.mascot import move_together, tie

        geo = widget.frameGeometry()
        screen = (widget.screen() or self.screen()).availableGeometry()
        x = geo.left() - self.W - 8
        if x < screen.left():                       # no room on the left: go right
            x = geo.right() + 8
        self.move(x, geo.bottom() - self.H - 30)
        tie(self, widget)
        self._opened = time.monotonic()
        self.show()
        move_together(self, self.pos(), self.partners)  # pull the pair on screen if needed

    def celebrate(self, winner: str) -> None:
        """Balloons, confetti and the winner's name ("X" = you, "O" = Loopy)."""
        if not hasattr(self, "_party"):
            self._party = Celebration()
        text, color = ("You win!", X_COLOR) if winner == "X" else ("Loopy wins!", O_COLOR)
        self._party.play(text, color, self.frameGeometry().center())

    def hide(self) -> None:
        if hasattr(self, "_party"):
            self._party.stop()
        from loopy.mascot import untie

        untie(self)  # Loopy moves on its own again
        super().hide()

    # ------------------------------------------------------------- painting

    def _cell_rect(self, i: int) -> QRectF:
        row, col = divmod(i, 3)
        return QRectF(self.GRID_X + col * self.CELL, self.GRID_Y + row * self.CELL, self.CELL, self.CELL)

    def paintEvent(self, _event):
        now = time.monotonic()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Pop-in: scale up from 85% with a little overshoot, fading in.
        k = min(1.0, (now - self._opened) / OPEN_TIME)
        scale = 0.85 + 0.15 * _ease_out_back(k)
        p.setOpacity(min(1.0, k * 1.5))
        p.translate(self.W / 2, self.H / 2)
        p.scale(scale, scale)
        p.translate(-self.W / 2, -self.H / 2)

        card = QRectF(4, 4, self.W - 8, self.H - 8)
        p.setPen(QPen(CARD_EDGE, 1.5))
        p.setBrush(CARD)
        p.drawRoundedRect(card, 18, 18)

        p.setPen(TEXT)
        p.setFont(self._title)
        p.drawText(QPointF(24, 40), "Tic-Tac-Toe")
        p.setFont(self._small)
        p.setPen(SOFT_TEXT)
        p.drawText(QPointF(24, 62), self.status)

        self._draw_grid(p)
        for i in range(9):
            self._draw_cell(p, i, now)
        if self._win:
            self._draw_win_line(p, now)
        self._draw_scores(p)

    def _draw_grid(self, p: QPainter):
        p.setPen(QPen(GRID, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        size = self.CELL * 3
        for n in (1, 2):
            x = self.GRID_X + n * self.CELL
            y = self.GRID_Y + n * self.CELL
            p.drawLine(QPointF(x, self.GRID_Y + 6), QPointF(x, self.GRID_Y + size - 6))
            p.drawLine(QPointF(self.GRID_X + 6, y), QPointF(self.GRID_X + size - 6, y))

    def _draw_cell(self, p: QPainter, i: int, now: float):
        rect = self._cell_rect(i)
        mark = self.cells[i]
        if mark is None:  # big number, so you know what to say
            p.setFont(self._number)
            p.setPen(NUMBER)
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(i + 1))
            return
        p.setFont(self._corner)
        p.setPen(SOFT_TEXT)
        p.drawText(rect.adjusted(7, 5, 0, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, str(i + 1))

        progress = min(1.0, (now - self._placed[i]) / DRAW_TIME)
        c, r = rect.center(), self.CELL * 0.27
        if mark == "X":
            p.setPen(QPen(INK, 7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            a = min(1.0, progress * 2)            # first stroke, then the second
            b = max(0.0, progress * 2 - 1)
            p.drawLine(QPointF(c.x() - r, c.y() - r), QPointF(c.x() - r + 2 * r * a, c.y() - r + 2 * r * a))
            if b > 0:
                p.drawLine(QPointF(c.x() + r, c.y() - r), QPointF(c.x() + r - 2 * r * b, c.y() - r + 2 * r * b))
        else:
            p.setPen(QPen(INK, 7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawArc(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r), 90 * 16, int(-360 * 16 * progress))

    def _draw_win_line(self, p: QPainter, now: float):
        assert self._win is not None  # only called once there is a winning line
        a, _, b = self._win
        start, end = self._cell_rect(a).center(), self._cell_rect(b).center()
        # Extend a little past the outer cells.
        dx, dy = end.x() - start.x(), end.y() - start.y()
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        start = QPointF(start.x() - ux * 26, start.y() - uy * 26)
        end = QPointF(end.x() + ux * 26, end.y() + uy * 26)
        k = min(1.0, (now - self._win_at) / LINE_TIME)
        tip = QPointF(start.x() + (end.x() - start.x()) * k, start.y() + (end.y() - start.y()) * k)
        p.setPen(QPen(QColor(0, 0, 0, 40), 16, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))  # soft gray glow
        p.drawLine(start, tip)
        p.setPen(QPen(INK, 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(start, tip)

    def _draw_scores(self, p: QPainter):
        you, draws, loopy = self.scores
        y = self.GRID_Y + 3 * self.CELL + 34
        p.setFont(self._small)
        parts = [(f"You {you}", TEXT), ("  ·  ", SOFT_TEXT), (f"Draws {draws}", TEXT),
                 ("  ·  ", SOFT_TEXT), (f"Loopy {loopy}", TEXT)]
        widths = [p.fontMetrics().horizontalAdvance(t) for t, _ in parts]
        x = (self.W - sum(widths)) / 2
        for (text, color), w in zip(parts, widths, strict=True):
            p.setPen(color)
            p.drawText(QPointF(x, y), text)
            x += w

    # ---------------------------------------------------------- dragging

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            from loopy.mascot import system_drag

            if system_drag(self):  # Wayland: the compositor moves the window
                return
            self._drag = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self._drag is not None and event.buttons() & Qt.MouseButton.LeftButton:
            from loopy.mascot import move_together

            move_together(self, event.globalPosition().toPoint() - self._drag, self.partners)

    def mouseReleaseEvent(self, _event):
        self._drag = None


PARTY_COLORS = [X_COLOR, O_COLOR, QColor(255, 205, 70), QColor(170, 125, 255),
                QColor(255, 125, 200), QColor(95, 175, 255)]


class Celebration(QWidget):
    """A see-through, click-through window over the board: balloons float up,
    confetti and glitter burst out, and the winner's name pops in."""

    W, H = 600, 660
    DURATION = 4.5
    FADE = 0.7
    GRAVITY = 520.0

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
                            | Qt.WindowType.WindowTransparentForInput)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedSize(self.W, self.H)
        self._banner = QFont("Segoe UI", 34, QFont.Weight.Black)
        self._sub = QFont("Segoe UI", 11, QFont.Weight.DemiBold)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._t0 = 0.0
        self._text, self._color = "", X_COLOR
        self.balloons: list[dict] = []
        self.confetti: list[dict] = []
        self.sparkles: list[dict] = []

    def play(self, text: str, color: QColor, center: QPoint) -> None:
        self._text, self._color = text, color
        self.move(center.x() - self.W // 2, center.y() - self.H // 2)
        self._make_particles(random.Random())
        self._t0 = time.monotonic()
        self.show()
        self._timer.start(16)  # ~60 fps while it plays

    def stop(self) -> None:
        self._timer.stop()
        super().hide()

    def _make_particles(self, rng: random.Random) -> None:
        w, h = self.W, self.H
        self.balloons = [{
            "x": rng.uniform(40, w - 40), "y": h + rng.uniform(20, 260),
            "speed": rng.uniform(120, 190), "sway": rng.uniform(8, 22), "phase": rng.uniform(0, 6.3),
            "r": rng.uniform(22, 32), "color": rng.choice(PARTY_COLORS),
        } for _ in range(11)]
        cx, cy = w / 2, h / 2 + 40
        self.confetti = []
        for _ in range(150):
            angle = rng.uniform(-math.pi * 0.92, -math.pi * 0.08)  # burst upwards and outwards
            speed = rng.uniform(260, 560)
            glitter = rng.random() < 0.35
            self.confetti.append({
                "x": cx, "y": cy, "vx": math.cos(angle) * speed, "vy": math.sin(angle) * speed,
                "rot": rng.uniform(0, 360), "spin": rng.uniform(-540, 540), "phase": rng.uniform(0, 6.3),
                "size": rng.uniform(2, 3.2) if glitter else rng.uniform(5, 9), "glitter": glitter,
                "color": QColor(255, 245, 200) if glitter and rng.random() < 0.5 else rng.choice(PARTY_COLORS),
                "delay": rng.uniform(0, 0.25),
            })
        self.sparkles = [{
            "x": rng.uniform(60, w - 60), "y": rng.uniform(90, h - 120),
            "start": rng.uniform(0.2, self.DURATION - 1.0), "life": rng.uniform(0.5, 0.9),
            "size": rng.uniform(6, 12), "color": rng.choice([QColor(255, 240, 180), QColor(255, 255, 255)]),
        } for _ in range(26)]

    def paintEvent(self, _event):
        t = time.monotonic() - self._t0
        if t > self.DURATION:
            self.stop()
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setOpacity(min(1.0, (self.DURATION - t) / self.FADE))
        for b in self.balloons:
            self._draw_balloon(p, b, t)
        for c in self.confetti:
            self._draw_confetto(p, c, t)
        for s in self.sparkles:
            self._draw_sparkle(p, s, t)
        self._draw_banner(p, t)

    def _draw_balloon(self, p: QPainter, b: dict, t: float):
        x = b["x"] + b["sway"] * math.sin(t * 1.6 + b["phase"])
        y = b["y"] - b["speed"] * t
        r = b["r"]
        if y < -r * 4:
            return
        # String: a gentle wiggle hanging below.
        string = QPainterPath(QPointF(x, y + r * 1.2))
        for i in range(1, 5):
            string.lineTo(x + 4 * math.sin(t * 4 + i + b["phase"]), y + r * 1.2 + i * 14)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 140), 1.2))
        p.drawPath(string)
        # Balloon body with a soft shine.
        grad = QRadialGradient(QPointF(x - r * 0.35, y - r * 0.45), r * 1.4)
        grad.setColorAt(0.0, b["color"].lighter(150))
        grad.setColorAt(1.0, b["color"].darker(115))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(grad)
        p.drawEllipse(QPointF(x, y), r * 0.86, r)
        knot = QPolygonF([QPointF(x - 4, y + r * 1.18), QPointF(x + 4, y + r * 1.18), QPointF(x, y + r * 0.95)])
        p.setBrush(b["color"].darker(120))
        p.drawPolygon(knot)
        p.setBrush(QColor(255, 255, 255, 120))
        p.drawEllipse(QPointF(x - r * 0.32, y - r * 0.42), r * 0.18, r * 0.3)

    def _draw_confetto(self, p: QPainter, c: dict, t: float):
        t = t - c["delay"]
        if t < 0:
            return
        x = c["x"] + c["vx"] * t + 14 * math.sin(t * 5 + c["phase"])   # flutter
        y = c["y"] + c["vy"] * t + 0.5 * self.GRAVITY * t * t
        if y > self.H + 20:
            return
        color = QColor(c["color"])
        if c["glitter"]:  # twinkles on and off
            color.setAlpha(int(110 + 145 * (0.5 + 0.5 * math.sin(t * 18 + c["phase"]))))
        p.save()
        p.translate(x, y)
        p.rotate(c["rot"] + c["spin"] * t)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        s = c["size"]
        if c["glitter"]:
            p.drawEllipse(QPointF(0, 0), s, s)
        else:  # a paper strip seen edge-on as it tumbles
            p.drawRect(QRectF(-s / 2, -s * 0.3 * abs(math.cos(t * 6 + c["phase"])) - 1, s, s * 0.6))
        p.restore()

    def _draw_sparkle(self, p: QPainter, s: dict, t: float):
        k = (t - s["start"]) / s["life"]
        if not 0 <= k <= 1:
            return
        size = s["size"] * math.sin(math.pi * k)  # grows, then shrinks
        x, y = s["x"], s["y"]
        star = QPolygonF([QPointF(x, y - size), QPointF(x + size * 0.25, y - size * 0.25),
                          QPointF(x + size, y), QPointF(x + size * 0.25, y + size * 0.25),
                          QPointF(x, y + size), QPointF(x - size * 0.25, y + size * 0.25),
                          QPointF(x - size, y), QPointF(x - size * 0.25, y - size * 0.25)])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(s["color"])
        p.drawPolygon(star)

    def _draw_banner(self, p: QPainter, t: float):
        k = min(1.0, t / 0.45)
        scale = max(0.01, _ease_out_back(k)) if k < 1 else 1 + 0.03 * math.sin(t * 5)  # pop, then breathe
        p.save()
        p.translate(self.W / 2, self.H / 2 - 30)
        p.scale(scale, scale)
        p.setFont(self._banner)
        width = p.fontMetrics().horizontalAdvance(self._text) + 56
        pill = QRectF(-width / 2, -42, width, 84)
        p.setPen(QPen(self._color, 3))
        p.setBrush(QColor(20, 24, 36, 225))
        p.drawRoundedRect(pill, 42, 42)
        p.setPen(self._color)
        text_area = QRectF(pill.left(), pill.top() - 6, pill.width(), pill.height())
        p.drawText(text_area, Qt.AlignmentFlag.AlignCenter, self._text)
        p.setFont(self._sub)
        p.setPen(QColor(235, 240, 248, 200))
        p.drawText(QRectF(pill.left(), 12, pill.width(), 26), Qt.AlignmentFlag.AlignCenter, "Three in a row!")
        p.restore()
