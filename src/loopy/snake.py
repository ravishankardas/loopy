"""Snake: the user plays with the keyboard; Loopy cheers.

    game = SnakeGame()
    game.turn("left")        # arrow key / WASD
    event = game.step()      # call every tick: None, "ate" or "died"
    game.score, game.snake, game.food

    window = SnakeWindow(best=12)
    window.finished.connect(lambda score: ...)
    window.open_next_to(mascot)

Controls: arrow keys or WASD to steer, Space to pause, Esc to close. Going
off one edge brings the snake back on the opposite side; only running into
yourself ends the game. The snake speeds up as it grows. Clicking away
pauses it.
"""

import random
import time
from collections import deque

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

DIRECTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}

IDLE_WRAPS = 3          # through an edge this many times without a key press: nobody's playing
START_INTERVAL = 0.16   # seconds per step at the start...
FASTEST = 0.07          # ...down to this
SPEEDUP = 0.004         # faster by this much per food eaten


class SnakeGame:
    """The rules. No Qt here, so it's easy to test."""

    def __init__(self, width: int = 20, height: int = 20, rng: random.Random | None = None):
        self.width, self.height = width, height
        self.rng = rng or random.Random()
        mid = (width // 2, height // 2)
        self.snake: deque[tuple[int, int]] = deque([mid, (mid[0] - 1, mid[1]), (mid[0] - 2, mid[1])])  # head first
        self.direction = "right"
        self._turns: deque[str] = deque()  # quick key presses, applied one per step
        self.score = 0
        self.over = False
        self.wraps_untouched = 0  # edges crossed since the player last turned
        self.food = self._place_food()

    @property
    def head(self) -> tuple[int, int]:
        return self.snake[0]

    @property
    def idle(self) -> bool:
        """Nobody's steering: it just keeps going in one edge and out the other."""
        return self.wraps_untouched >= IDLE_WRAPS

    @property
    def interval(self) -> float:
        """Seconds per step: faster as the snake grows."""
        return max(FASTEST, START_INTERVAL - SPEEDUP * self.score)

    def turn(self, direction: str) -> None:
        """Queue a turn. Turning straight back into yourself is ignored."""
        if direction not in DIRECTIONS or self.over:
            return
        last = self._turns[-1] if self._turns else self.direction
        if direction != last and direction != OPPOSITE[last] and len(self._turns) < 3:
            self._turns.append(direction)

    def step(self) -> str | None:
        """Move one cell. Returns "ate", "died" or None."""
        if self.over:
            return None
        if self._turns:
            self.direction = self._turns.popleft()
            self.wraps_untouched = 0
        dx, dy = DIRECTIONS[self.direction]
        # Off one edge, back on the opposite side.
        straight_on = (self.head[0] + dx, self.head[1] + dy)
        new = (straight_on[0] % self.width, straight_on[1] % self.height)
        if new != straight_on:
            self.wraps_untouched += 1
        grows = new == self.food
        body = list(self.snake) if grows else list(self.snake)[:-1]  # the tail moves away this step
        if new in body:
            self.over = True
            return "died"
        self.snake.appendleft(new)
        if grows:
            self.score += 1
            self.food = self._place_food()
            return "ate"
        self.snake.pop()
        return None

    def _place_food(self) -> tuple[int, int] | None:
        free = [(x, y) for x in range(self.width) for y in range(self.height) if (x, y) not in self.snake]
        return self.rng.choice(free) if free else None


# --------------------------------------------------------------- the window

CARD = QColor(255, 255, 255)
CARD_EDGE = QColor(0, 0, 0, 45)
INK = QColor(20, 20, 24)
SOFT = QColor(20, 20, 24, 150)
GRID_DOT = QColor(20, 20, 24, 28)
FOOD = QColor(235, 70, 70)

KEYS = {Qt.Key_Up: "up", Qt.Key_W: "up", Qt.Key_Down: "down", Qt.Key_S: "down",
        Qt.Key_Left: "left", Qt.Key_A: "left", Qt.Key_Right: "right", Qt.Key_D: "right"}


class SnakeWindow(QWidget):
    """The snake game: a white card like the tic-tac-toe board."""

    finished = Signal(int)     # game over: the score
    ate = Signal(int)          # food eaten: the new score
    closed = Signal()          # Esc or closed
    idle = Signal()            # paused itself: nobody's steering
    resumed = Signal()         # playing again after an idle pause

    CELL = 16
    COLS = ROWS = 20
    PAD = 20
    TOP = 66                   # room for title + score
    W = PAD * 2 + COLS * CELL
    H = TOP + ROWS * CELL + PAD + 22

    def __init__(self, best: int = 0):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFocusPolicy(Qt.StrongFocus)  # it needs the keyboard
        self.setFixedSize(self.W, self.H)
        self.best = best
        self.partners: list[QWidget] = []    # Loopy: they move together
        self._title = QFont("Segoe UI", 13, QFont.Bold)
        self._small = QFont("Segoe UI", 9)
        self._big = QFont("Segoe UI", 16, QFont.Bold)
        self._drag = None
        self._tick = QTimer(self)
        self._tick.timeout.connect(self._step)
        self._repaint = QTimer(self)
        self._repaint.timeout.connect(self.update)
        self._repaint.start(33)
        self.new_game()

    # ------------------------------------------------------------------ API

    def new_game(self) -> None:
        self.game = SnakeGame(self.COLS, self.ROWS)
        self.paused = False
        self.idle_paused = False   # paused because nobody was steering
        self._ready_prompt = False # asked to keep playing: waiting for the first key
        self._died_at = 0.0
        self._started = False  # waits for the first key press
        self._tick.stop()

    def open_next_to(self, widget: QWidget) -> None:
        """Show beside Loopy (left, or right if there's no room), tied to it."""
        from loopy.mascot import move_together

        geo = widget.frameGeometry()
        screen = (widget.screen() or self.screen()).availableGeometry()
        x = geo.left() - self.W - 8
        if x < screen.left():
            x = geo.right() + 8
        self.move(x, geo.bottom() - self.H - 30)
        self.partners = [widget]
        if hasattr(widget, "partners"):
            widget.partners = [self]
        self.show()
        move_together(self, self.pos(), self.partners)
        self.grab_keyboard()

    def get_ready(self) -> None:
        """The player said they want to keep going: wait for their first key
        (so the snake doesn't crash while their hands are off the keyboard)."""
        self._ready_prompt = True
        self.grab_keyboard()

    def grab_keyboard(self) -> None:
        self.raise_()
        self.activateWindow()
        self.setFocus()

    def hide(self) -> None:
        self._tick.stop()
        for partner in self.partners:
            if getattr(partner, "partners", None) is not None and self in partner.partners:
                partner.partners.remove(self)
        self.partners = []
        super().hide()

    # ----------------------------------------------------------- game loop

    def _step(self) -> None:
        event = self.game.step()
        if event == "ate":
            self.ate.emit(self.game.score)
            self._tick.setInterval(int(self.game.interval * 1000))
        elif event == "died":
            self._tick.stop()
            self._died_at = time.monotonic()
            self.best = max(self.best, self.game.score)
            self.finished.emit(self.game.score)
            return
        if self.game.idle and not self.paused:
            # In one edge and out the other, again and again, untouched:
            # nobody's playing. Pause and let Loopy ask.
            self.paused = self.idle_paused = True
            self._tick.stop()
            self.idle.emit()

    def _run(self) -> None:
        if self.idle_paused:
            self.idle_paused = False
            self.resumed.emit()
        self._ready_prompt = False
        self.game.wraps_untouched = 0
        self._started, self.paused = True, False
        self._tick.start(int(self.game.interval * 1000))

    # ------------------------------------------------------------- keyboard

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Escape:
            self.hide()
            self.closed.emit()
            return
        if self.game.over:
            if key in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
                self.new_game()
            return
        if key == Qt.Key_Space and self._started:
            if self.paused:
                self._run()
            else:
                self.paused = True
                self._tick.stop()
            return
        if key in KEYS:
            self.game.turn(KEYS[key])
            self.game.wraps_untouched = 0  # any arrow key means someone's here
            if not self._started or self.paused:
                self._run()

    def focusOutEvent(self, event):
        # Clicked away: pause, so the snake doesn't crash unattended.
        if self._started and not self.game.over and not self.paused:
            self.paused = True
            self._tick.stop()
        super().focusOutEvent(event)

    # ------------------------------------------------------------- painting

    def _cell(self, x: int, y: int, inset: float = 1.0) -> QRectF:
        return QRectF(self.PAD + x * self.CELL + inset, self.TOP + y * self.CELL + inset,
                      self.CELL - 2 * inset, self.CELL - 2 * inset)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(CARD_EDGE, 1.5))
        p.setBrush(CARD)
        p.drawRoundedRect(QRectF(4, 4, self.W - 8, self.H - 8), 18, 18)

        p.setPen(INK)
        p.setFont(self._title)
        p.drawText(QPointF(24, 40), "Snake")
        p.setFont(self._small)
        p.setPen(SOFT)
        p.drawText(QRectF(0, 26, self.W - 24, 20), Qt.AlignRight | Qt.AlignVCenter,
                   f"Score {self.game.score}   ·   Best {max(self.best, self.game.score)}")

        # Play area: a faint dot grid inside a dashed frame (dashed: the edges
        # aren't walls, the snake comes back on the other side).
        area = QRectF(self.PAD, self.TOP, self.COLS * self.CELL, self.ROWS * self.CELL)
        p.setPen(QPen(SOFT, 1.5, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(area.adjusted(-2, -2, 2, 2), 6, 6)
        p.setPen(Qt.NoPen)
        p.setBrush(GRID_DOT)
        for x in range(self.COLS):
            for y in range(self.ROWS):
                p.drawEllipse(self._cell(x, y).center(), 1.2, 1.2)

        if self.game.food:
            p.setBrush(FOOD)
            p.drawEllipse(self._cell(*self.game.food, inset=2.5))

        flash = self.game.over and int((time.monotonic() - self._died_at) * 6) % 2 == 0 \
            and time.monotonic() - self._died_at < 1.2
        p.setBrush(QColor(235, 70, 70) if flash else INK)
        for i, (x, y) in enumerate(self.game.snake):
            p.drawRoundedRect(self._cell(x, y, inset=1.0 if i == 0 else 1.6), 4, 4)
        hx, hy = self.game.head  # two little eyes on the head
        p.setBrush(CARD)
        dx, dy = DIRECTIONS[self.game.direction]
        c = self._cell(hx, hy).center()
        for side in (-1, 1):
            p.drawEllipse(QPointF(c.x() + dx * 3 + side * dy * 3.2, c.y() + dy * 3 + side * dx * 3.2), 1.8, 1.8)

        message = None
        if self.game.over:
            message = (f"Game over! Score {self.game.score}", "Space to play again · Esc to close")
        elif not self._started and not self.hasFocus():
            # Opened by voice while another app had the keyboard: Windows
            # doesn't let it take over, so ask for a click first.
            message = ("Click here to play", "then an arrow key or WASD")
        elif not self._started:
            message = ("Press an arrow key", "or WASD to start · Esc closes")
        elif self._ready_prompt:
            message = ("Ready?", "Press an arrow key to keep going")
        elif self.idle_paused:
            message = ("Still there?", "Press an arrow key to keep playing")
        elif self.paused:
            message = ("Paused", "Click here, then an arrow key or Space")
        if message:
            self._draw_message(p, area, *message)

        p.setFont(self._small)
        p.setPen(SOFT)
        p.drawText(QRectF(0, self.H - 30, self.W, 20), Qt.AlignCenter, "Arrows / WASD  ·  Space pause  ·  Esc close")

    def _draw_message(self, p: QPainter, area: QRectF, title: str, hint: str) -> None:
        box = QRectF(area.center().x() - 130, area.center().y() - 36, 260, 72)
        p.setPen(QPen(INK, 1.5))
        p.setBrush(QColor(255, 255, 255, 235))
        p.drawRoundedRect(box, 12, 12)
        p.setPen(INK)
        p.setFont(self._big)
        p.drawText(box.adjusted(0, 6, 0, -30), Qt.AlignCenter, title)
        p.setFont(self._small)
        p.setPen(SOFT)
        p.drawText(box.adjusted(0, 38, 0, -6), Qt.AlignCenter, hint)

    # ---------------------------------------------------------- dragging

    def mousePressEvent(self, event):
        self.grab_keyboard()
        if event.button() == Qt.LeftButton:
            from loopy.mascot import system_drag

            if system_drag(self):  # Wayland: the compositor moves the window
                return
            self._drag = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self._drag is not None and event.buttons() & Qt.LeftButton:
            from loopy.mascot import move_together

            move_together(self, event.globalPosition().toPoint() - self._drag, self.partners)

    def mouseReleaseEvent(self, _event):
        self._drag = None
