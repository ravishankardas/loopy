"""The snake window pauses itself when nobody's steering, and Loopy asks
whether they're still playing. Runs on Qt's offscreen platform."""

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from fakes import FakeLlama, RecordedActions  # noqa: E402
from loopy import actions  # noqa: E402
from loopy.brain import Brain  # noqa: E402
from loopy.snake import SnakeWindow  # noqa: E402

app = QApplication.instance() or QApplication([])


class WindowPausesItself(unittest.TestCase):
    def setUp(self):
        self.w = SnakeWindow()
        self.addCleanup(self.w.close)
        self.idle, self.resumed = [], []
        self.w.idle.connect(lambda: self.idle.append(True))
        self.w.resumed.connect(lambda: self.resumed.append(True))
        self.w._run()                            # playing, nobody touching the keys
        self.w.game.food = (0, 0)

    def run_untouched(self, steps=200):
        for _ in range(steps):
            if self.w.paused:
                break
            self.w._step()

    def test_pauses_and_says_so_after_three_untouched_edges(self):
        self.run_untouched()
        self.assertEqual(self.idle, [True])
        self.assertTrue(self.w.paused)
        self.assertTrue(self.w.idle_paused)
        self.assertFalse(self.w.game.over)        # paused, not dead

    def test_an_arrow_key_carries_on(self):
        self.run_untouched()
        QTest.keyClick(self.w, Qt.Key.Key_Up)
        self.assertEqual(self.resumed, [True])
        self.assertFalse(self.w.paused)
        self.assertEqual(self.w.game.wraps_untouched, 0)

    def test_steering_now_and_then_never_pauses(self):
        for i in range(200):
            if i % 15 == 0:
                QTest.keyClick(self.w, Qt.Key.Key_Up if (i // 15) % 2 else Qt.Key.Key_Right)
            self.w._step()
            if self.w.game.over:
                break
        self.assertEqual(self.idle, [])

    def test_get_ready_waits_for_a_key(self):
        self.run_untouched()
        self.w.get_ready()                        # "yes, keep playing" by voice
        self.assertTrue(self.w.paused)            # still waiting for the player's hands
        QTest.keyClick(self.w, Qt.Key.Key_Left)
        self.assertFalse(self.w.paused)


class BrainReadsTheAnswer(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(actions, "run", RecordedActions())
        p.start()
        self.addCleanup(p.stop)
        self.brain = Brain("unused.gguf")

    def turn(self, user, *replies):
        self.brain._llm = FakeLlama(*replies)
        said, events = [], []
        self.brain.respond(user, said.append, lambda *e: events.append(e))
        return "".join(said).strip(), events

    def snake_paused(self):
        self.turn("let's play snake", "[game: snake] Ooh!")
        question = self.brain.snake_idle()
        self.assertIn("still playing", question)

    def test_the_model_is_told_how_to_read_yes_and_no(self):
        self.snake_paused()
        self.assertIn("[game: snake]", self.brain._history[-1]["content"])
        self.assertIn("[game: stop]", self.brain._history[-1]["content"])

    def test_yes_carries_on_without_restarting(self):
        self.snake_paused()
        said, events = self.turn("yeah", "[game: snake] Yay, back to it!")
        self.assertEqual(events, [("snake_resume",)])
        self.assertTrue(self.brain.snake_open)
        self.assertEqual(said, "Yay, back to it! Press an arrow key when you're ready.")  # not "back to it" twice

    def test_no_closes_it(self):
        self.snake_paused()
        said, events = self.turn("no, I'm done", "[game: stop] Okay, good game!")
        self.assertEqual(events, [("snake_stop",)])
        self.assertFalse(self.brain.snake_open)


if __name__ == "__main__":
    unittest.main()
