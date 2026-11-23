"""While a game is open, Loopy and the board move as one unit and never go
past the edge of the screen. Runs on Qt's offscreen platform."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from loopy.game import BoardWindow  # noqa: E402
from loopy.mascot import Mascot, move_together  # noqa: E402

app = QApplication.instance() or QApplication([])


class MoveTogether(unittest.TestCase):
    def setUp(self):
        self.mascot = Mascot()
        self.mascot.show()
        self.board = BoardWindow()
        self.screen = self.mascot.screen().availableGeometry()
        self.addCleanup(self.board.close)
        self.addCleanup(self.mascot.close)

    def drag(self, widget, to: QPoint):
        """What a mouse drag does: the widget asks to go to `to`."""
        move_together(widget, to, widget.partners)

    def gap(self) -> QPoint:
        return self.mascot.pos() - self.board.pos()

    def inside_screen(self, *widgets):
        for w in widgets:
            self.assertTrue(self.screen.contains(w.frameGeometry()), f"{w} is past the screen edge")

    def test_board_opens_beside_loopy_and_they_are_tied(self):
        self.board.open_next_to(self.mascot)
        self.assertLess(self.board.x(), self.mascot.x())          # on the left
        self.assertEqual(self.mascot.partners, [self.board])
        self.assertEqual(self.board.partners, [self.mascot])
        self.inside_screen(self.mascot, self.board)

    def test_dragging_loopy_moves_the_board_too(self):
        self.board.open_next_to(self.mascot)
        gap = self.gap()
        self.drag(self.mascot, self.mascot.pos() - QPoint(20, 30))
        self.assertEqual(self.gap(), gap)                          # same distance apart

    def test_dragging_the_board_moves_loopy_too(self):
        self.board.open_next_to(self.mascot)
        gap = self.gap()
        before = self.mascot.pos()
        self.drag(self.board, self.board.pos() - QPoint(15, 25))
        self.assertEqual(self.gap(), gap)
        self.assertEqual(self.mascot.pos(), before - QPoint(15, 25))

    def test_the_pair_stops_at_every_edge(self):
        self.board.open_next_to(self.mascot)
        gap = self.gap()
        for far in (QPoint(-5000, 0), QPoint(5000, 0), QPoint(0, -5000), QPoint(0, 5000)):
            self.drag(self.mascot, self.mascot.pos() + far)
            self.inside_screen(self.mascot, self.board)            # never past the edge...
            self.assertEqual(self.gap(), gap)                      # ...and still together

    def test_it_reaches_the_edge_exactly(self):
        self.board.open_next_to(self.mascot)
        self.drag(self.board, QPoint(-5000, self.board.y()))
        self.assertEqual(self.board.frameGeometry().left(), self.screen.left())

    def test_board_opens_on_the_right_when_theres_no_room_on_the_left(self):
        self.mascot.move(self.screen.left(), self.mascot.y())
        self.board.open_next_to(self.mascot)
        self.assertGreater(self.board.x(), self.mascot.x())
        self.inside_screen(self.mascot, self.board)

    def test_after_the_game_loopy_moves_alone_and_still_stops_at_the_edge(self):
        self.board.open_next_to(self.mascot)
        self.board.hide()
        self.assertEqual(self.mascot.partners, [])
        board_at = self.board.pos()
        self.drag(self.mascot, self.mascot.pos() + QPoint(-5000, 0))
        self.assertEqual(self.board.pos(), board_at)               # the board didn't come along
        self.inside_screen(self.mascot)


if __name__ == "__main__":
    unittest.main()
