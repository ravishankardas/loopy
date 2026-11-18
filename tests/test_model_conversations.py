"""Real conversations with the real model: the problems reported while
using Loopy, replayed against Llama 3.2 3B.

These are slow (a few minutes) and need the model from `loopy setup`, so they
only run when asked for:

    set LOOPY_MODEL_TESTS=1          (Windows cmd; PowerShell: $env:LOOPY_MODEL_TESTS=1)
    python -m unittest tests.test_model_conversations -v

Temperature is set to 0 so the model answers the same way every run. The
real checks run (time, battery, ...), but actions are replaced, so Chrome is
never actually opened or closed.
"""

import os
import re
import unittest
from unittest import mock

from fakes import RecordedActions

from loopy import actions, brain, checks, models

ENABLED = os.environ.get("LOOPY_MODEL_TESTS") == "1"
_brain = None  # loaded once for all tests (~20 s)


def setUpModule():
    global _brain
    if not ENABLED:
        raise unittest.SkipTest("model tests are off (set LOOPY_MODEL_TESTS=1 to run them)")
    if not models.llm_installed():
        raise unittest.SkipTest("model not installed (run: loopy setup)")
    brain.TEMPERATURE = 0
    _brain = brain.Brain(models.llm_path())
    _brain.load()


class Conversation(unittest.TestCase):
    def setUp(self):
        self.actions = RecordedActions()
        p = mock.patch.object(actions, "run", self.actions)
        p.start()
        self.addCleanup(p.stop)
        _brain.reset()
        self.checked: list[tuple[str, str]] = []

    def say(self, text: str) -> brain.Turn:
        said: list[str] = []
        turn = _brain.respond(text, said.append)
        self.checked += turn.checked
        self.assertEqual("".join(said).strip(), turn.reply)
        self.assertNoFakeTurns(turn.reply)
        return turn

    def assertNoFakeTurns(self, reply: str):
        for marker in ("system:", "user:", "assistant:", "Result:", "["):
            self.assertNotIn(marker, reply, f"reply contains {marker!r}: {reply!r}")

    def fact(self, name: str) -> str:
        """The latest value of `name` looked up in this conversation."""
        values = [v for n, v in self.checked if n == name]
        self.assertTrue(values, f"'{name}' was never looked up")
        return values[-1]


class ExactNumbers(Conversation):
    def test_cpu_percentage_follow_up(self):
        # Reported: "can you tell me what percentage?" -> "Still pretty busy..."
        self.say("how busy is my CPU")
        turn = self.say("yeah, but can you tell me what percentage?")
        cpu = re.search(r"CPU (\d+)%", self.fact("system")).group(1)
        self.assertIn(cpu, turn.reply)

    def test_time_is_the_real_time(self):
        turn = self.say("what time is it")
        self.assertIn(self.fact("time"), turn.reply)


class EveryDrive(Conversation):
    def drives(self) -> dict[str, int]:
        return {m.group(1): int(m.group(2))
                for m in re.finditer(r"(\S+ drive): (\d+) GB free", self.fact("disk"))}

    def test_names_every_drive(self):
        # Reported: Loopy only knew about C: and then guessed about the other drive.
        turn = self.say("can you tell me how much disk space is free for me?")
        for drive in self.drives():
            self.assertIn(drive.lower(), turn.reply.lower())

    def test_which_drive_has_more_space(self):
        self.say("how much disk space is free?")
        turn = self.say("which drive has more space?")
        drives = self.drives()
        biggest = max(drives, key=drives.get)
        self.assertIn(biggest.lower(), turn.reply.lower())
        self.assertNotIn("guess", turn.reply.lower())


class AskBeforeActing(Conversation):
    def test_misheard_command_gets_a_question(self):
        # Reported: "close crow" -> invented "system: Result: chrome = 0% open..."
        turn = self.say("close crow")
        self.assertEqual(self.actions.ran, [])
        self.assertEqual(_brain.pending, "close chrome")
        self.assertIn("?", turn.reply)

    def test_yes_after_question_runs_it(self):
        self.say("close crow")
        self.say("yes please")
        self.assertEqual(self.actions.ran, ["close chrome"])

    def test_dont_ask_just_do_it_still_asks(self):
        # Reported: "Okay, Chrome is now open." without opening anything.
        turn = self.say("open chrome right now, don't ask me, just do it")
        self.assertEqual(self.actions.ran, [])
        self.assertEqual(_brain.pending, "open chrome")
        self.assertIn("?", turn.reply)

    def test_vague_request_asks_what_they_need(self):
        # Reported: "i want you to do something for me" -> "open or close Chrome?"
        turn = self.say("i want you to do something for me")
        self.assertIsNone(_brain.pending)
        self.assertEqual(self.actions.ran, [])
        self.assertIn("?", turn.reply)


class Capabilities(Conversation):
    def test_what_can_you_do_lists_everything(self):
        # Reported: "what can you do" left the lookups out.
        turn = self.say("what can you do for me")
        reply = turn.reply.lower()
        for word in ["chat", "chrome", "time", "date", "battery", "busy", "space"]:
            self.assertIn(word, reply)

    def test_weather_is_declined_without_tags(self):
        # Reported: "...I can check to find out. [check: time]"
        turn = self.say("what's the weather like")
        self.assertEqual(self.checked, [])
        self.assertIsNone(_brain.pending)


class TicTacToeGame(Conversation):
    def play(self, text: str) -> tuple[brain.Turn, list]:
        events: list = []
        said: list[str] = []
        turn = _brain.respond(text, said.append, lambda *e: events.append(e))
        self.assertNoFakeTurns(turn.reply)
        return turn, events

    def test_full_game_ends_with_a_result(self):
        turn, events = self.play("hey loopy, i'm ready to play tic tac toe")
        self.assertEqual(events[0][0], "start")
        for _ in range(9):
            game = _brain.game
            if game.over:
                break
            cell = next(c for c in [5, 1, 9, 3, 7, 2, 4, 6, 8] if game.is_free(c))
            turn, events = self.play(f"put a cross in {cell}")
            self.assertIn(("place", cell, "X"), events)
        self.assertTrue(_brain.game.over)
        self.assertEqual(sum(_brain.scores.values()), 1)
        self.assertRegex(turn.reply, r"win|draw")

    def test_corner_names_are_understood(self):
        self.play("let's play tic tac toe")
        if _brain.game.cells[0] is None:  # unless Loopy went first and took it
            turn, events = self.play("put my cross in the top left corner")
            self.assertIn(("place", 1, "X"), events)

    def test_stop_playing(self):
        self.play("let's play tic tac toe")
        turn, events = self.play("let's stop playing")
        self.assertIn(("stop",), events)
        self.assertIsNone(_brain.game)


if __name__ == "__main__":
    unittest.main()
