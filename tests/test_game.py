"""Tic-tac-toe: the rules, and the game flow with a fake model.

The fake model plays exactly the moves scripted here, including illegal
ones, to check that the code is a strict referee: illegal moves are rejected,
their comments are never spoken, the model is told why, and the game always
ends with the right winner.
"""

import unittest
from unittest import mock

from fakes import FakeLlama, RecordedActions

from loopy import actions, brain
from loopy.brain import Brain
from loopy.game import WIN_LINES, TicTacToe


def board(layout: str) -> TicTacToe:
    """'XX-O-----' -> X in 1 and 2, O in 4."""
    game = TicTacToe()
    for i, c in enumerate(layout):
        if c != "-":
            game.place(i + 1, c)
    return game


class Rules(unittest.TestCase):
    def test_every_line_wins(self):
        for line in WIN_LINES:
            game = TicTacToe()
            for i in line:
                game.cells[i] = "O"
            game.cells[line[0]] = None
            game.place(line[0] + 1, "O")  # completing the line decides it
            self.assertEqual(game.winner, "O")
            self.assertEqual(game.line, line)

    def test_draw(self):
        game = TicTacToe()
        for cell, mark in zip([1, 2, 3, 5, 4, 6, 8, 7, 9], "XOXOXOXOX"):
            game.place(cell, mark)
        self.assertEqual(game.winner, "draw")
        self.assertIsNone(game.line)

    def test_taken_and_invalid_cells(self):
        game = board("X--------")
        self.assertFalse(game.is_free(1))
        self.assertFalse(game.is_free(0))
        self.assertFalse(game.is_free(10))
        self.assertFalse(game.is_free(None))
        self.assertEqual(game.why_not(1), "cell 1 is taken by X")
        self.assertIn("not a cell", game.why_not(10))
        with self.assertRaises(ValueError):
            game.place(1, "O")

    def test_no_moves_after_the_game_is_over(self):
        game = TicTacToe()
        for cell, mark in [(1, "X"), (5, "O"), (2, "X"), (6, "O"), (3, "X")]:
            game.place(cell, mark)
        self.assertTrue(game.over)
        self.assertFalse(game.is_free(4))

    def test_description_shows_free_cells_as_numbers(self):
        self.assertEqual(board("X---O----").describe(), "X | 2 | 3\n4 | O | 6\n7 | 8 | 9")


class GameTest(unittest.TestCase):
    """A Brain with a fake model; every turn records speech and board events."""

    def setUp(self):
        p = mock.patch.object(actions, "run", RecordedActions())
        p.start()
        self.addCleanup(p.stop)
        self.brain = Brain("unused.gguf")
        self.fake = None

    def turn(self, user: str, *model_replies: str):
        self.fake = FakeLlama(*model_replies)
        self.brain._llm = self.fake
        said, events = [], []
        turn = self.brain.respond(user, said.append, lambda *e: events.append(e))
        self.assertEqual(self.fake.replies, [], "the model wasn't asked as often as expected")
        return "".join(said).strip(), events

    def start(self, *loopy_moves: str):
        return self.turn("i'm ready to play tic tac toe", "[game: start] Yay!", *loopy_moves)

    def last_system_message(self, call: int) -> str:
        return [m for m in self.fake.calls[call] if m["role"] == "system"][-1]["content"]


class StartingAndStopping(GameTest):
    def test_start_opens_the_board_and_you_go_first(self):
        said, events = self.start()
        self.assertEqual(events, [("start", False)])
        self.assertIn("You go first", said)
        self.assertIsNotNone(self.brain.game)

    def test_players_take_turns_going_first(self):
        self.start()
        self.turn("put a cross in 1", "[move: 1]", "[play: 4] Hi!")
        self.turn("2", "[move: 2]", "[play: 5] Hmm.")
        self.turn("3", "[move: 3]")                       # X wins: 1-2-3
        said, events = self.start("[play: 5] Middle, me first!")
        self.assertEqual(events[0], ("start", True))
        self.assertIn(("place", 5, "O"), events)
        self.assertIn("I'll go first", said)

    def test_loopys_words_are_spaced_after_the_announcement(self):
        # Reported: "I'll go first!I'll place my O in the middle!"
        self.start()
        self.turn("1", "[move: 1]", "[play: 4] a")
        self.turn("2", "[move: 2]", "[play: 5] b")
        self.turn("3", "[move: 3]")
        said, _ = self.start("[play: 5] I'll place my O in the middle!")
        self.assertIn("I'll go first! I'll place my O", said)

    def test_a_board_drawn_in_text_starts_the_real_game(self):
        # Reported: "Play that TikTok toe game" -> the model played in words:
        # "I'll start a new game. Here's the board: 1 | 2 | 3 4 | 5 | 6 ..."
        said, events = self.turn("Play that TikTok toe game",
                                 "[none] I think you mean Tic Tac Toe! I'll start a new game. Here's the board: "
                                 "1 | 2 | 3 4 | 5 | 6 7 | 8 | 9 You play as X. To make a move, type the number.")
        self.assertEqual(events[0], ("start", False))
        self.assertIsNotNone(self.brain.game)
        self.assertNotIn("|", said)
        self.assertNotIn("type the number", said)
        self.assertIn("You go first", said)

    def test_a_text_board_during_a_game_is_just_cut(self):
        self.start()
        said, events = self.turn("what's the board", "[none] Here it is: X | 2 | 3 ...")
        self.assertEqual(events, [])
        self.assertNotIn("|", said)

    def test_a_move_without_a_game_starts_one(self):
        said, events = self.turn("put a cross in 5", "[move: 5]", "[play: 1] Corner!")
        self.assertEqual(events[0], ("start", False))
        self.assertIn(("place", 5, "X"), events)

    def test_stop_closes_the_board(self):
        self.start()
        said, events = self.turn("let's stop playing", "[game: stop] Okay, good game!")
        self.assertEqual(events, [("stop",)])
        self.assertIsNone(self.brain.game)
        self.assertIn("good game", said)

    def test_falling_asleep_ends_the_game_but_keeps_the_score(self):
        self.start()
        self.turn("1", "[move: 1]", "[play: 4] Hi!")
        self.turn("2", "[move: 2]", "[play: 5] Hmm.")
        self.turn("3", "[move: 3]")
        self.brain.reset()
        self.assertIsNone(self.brain.game)
        self.assertEqual(self.brain.scores["you"], 1)


class StartingWithoutAsking(GameTest):
    """`loopy tic-tac-toe` and the menu start a game without a spoken request."""

    def new_game(self, *model_replies):
        self.fake = FakeLlama(*model_replies)
        self.brain._llm = self.fake
        said, events = [], []
        self.brain.new_game(said.append, lambda *e: events.append(e))
        return "".join(said).strip(), events

    def test_new_game_opens_the_board_and_you_go_first(self):
        said, events = self.new_game()
        self.assertEqual(events, [("start", False)])
        self.assertIn("Yay, let's play!", said)
        self.assertIn("You go first", said)
        self.assertEqual(self.fake.calls, [])          # no model call needed to start

    def test_the_model_knows_a_game_is_on(self):
        self.new_game()
        roles = [(m["role"], m["content"][:30]) for m in self.brain._history]
        self.assertEqual(roles[0], ("user", "Let's play tic-tac-toe!"))
        self.assertTrue(any("game started" in c.lower() for _r, c in roles))

    def test_then_moves_work_as_usual(self):
        self.new_game()
        said, events = self.turn("put a cross in 5", "[move: 5]", "[play: 1] Corner!")
        self.assertIn(("place", 5, "X"), events)
        self.assertIn(("place", 1, "O"), events)


class Command(unittest.TestCase):
    def test_loopy_tic_tac_toe_starts_with_a_game(self):
        from loopy import cli
        for command in ("tic-tac-toe", "ttt"):
            with mock.patch("loopy.app.run", return_value=0) as run:
                cli.main([command])
            run.assert_called_once_with(start_with_game=True)

    def test_plain_loopy_says_hello_instead(self):
        from loopy import cli
        with mock.patch("loopy.app.run", return_value=0) as run:
            cli.main([])
        run.assert_called_once_with(start_with_game=False)


class Moves(GameTest):
    def test_your_move_then_loopys_move(self):
        self.start()
        said, events = self.turn("put a cross in 5", "[move: 5]", "[play: 1] I'll take the corner!")
        self.assertEqual(events, [("place", 5, "X"), ("thinking",), ("place", 1, "O"), ("your_turn",)])
        self.assertEqual(said, "I'll take the corner!")
        self.assertEqual(self.brain.game.cells[0], "O")

    def test_a_silent_move_is_asked_for_again(self):
        # Seen with the real model: "[play: 9]" with no comment, so Loopy said nothing.
        self.start()
        said, events = self.turn("put a cross in the middle", "[move: 5]", "[play: 9]", "[play: 3] Corner for me!")
        self.assertEqual(said, "Corner for me!")
        self.assertIn(("place", 3, "O"), events)
        self.assertNotIn(("place", 9, "O"), events)  # the silent attempt wasn't placed
        retry_note = self.fake.calls[2][-1]["content"]
        self.assertIn("You picked 9 but said nothing", retry_note)

    def test_a_move_silent_every_time_is_announced(self):
        self.start()
        said, events = self.turn("put a cross in the middle", "[move: 5]", "[play: 9]", "[play: 9]", "[play: 9]")
        self.assertEqual(said, "I'll take 9, bottom-right.")
        self.assertIn(("place", 9, "O"), events)

    def test_loopy_is_shown_the_board_and_free_cells(self):
        self.start()
        self.turn("put a cross in 5", "[move: 5]", "[play: 1] Corner!")
        board_msg = self.last_system_message(call=1)
        self.assertIn("4 | X | 6", board_msg)
        self.assertIn("Free cells: 1, 2, 3, 4, 6, 7, 8, 9", board_msg)

    def test_your_move_on_a_taken_cell_is_refused(self):
        self.start()
        self.turn("put a cross in 5", "[move: 5]", "[play: 1] Corner!")
        said, events = self.turn("put a cross in 1", "[move: 1]")   # Loopy's O is there
        self.assertIn("cell 1 is taken", said)
        self.assertEqual(events, [])                                  # nothing placed, Loopy doesn't move
        self.assertEqual(self.brain.game.cells[0], "O")


class IllegalMovesByLoopy(GameTest):
    def test_illegal_move_is_rejected_and_loopy_picks_again(self):
        self.start()
        said, events = self.turn("put a cross in 5", "[move: 5]",
                                 "[play: 5] Taking the middle!",      # taken by X: illegal
                                 "[play: 3] Oops! Then 3 for me.")
        self.assertIn(("illegal", 5, "cell 5 is taken by X"), events)
        self.assertIn(("place", 3, "O"), events)
        self.assertNotIn("Taking the middle", said)                   # never spoken
        self.assertIn("Then 3 for me", said)

    def test_model_is_told_why_its_move_was_illegal(self):
        self.start()
        self.turn("put a cross in 5", "[move: 5]", "[play: 5] Middle!", "[play: 3] Okay, 3.")
        retry_msg = self.last_system_message(call=2)
        self.assertIn("Your move 5 isn't allowed: cell 5 is taken by X", retry_msg)
        self.assertNotIn("5,", retry_msg.split("Free cells:")[1])  # 5 isn't offered again

    def test_not_a_cell_at_all(self):
        self.start()
        said, events = self.turn("5", "[move: 5]", "[play: 12] Twelve!", "[play: middle] Hmm",
                                 "[play: 7] Fine, 7.")
        self.assertEqual([e[0] for e in events].count("illegal"), 2)
        self.assertIn(("place", 7, "O"), events)

    def test_chatting_instead_of_moving_counts_as_a_miss(self):
        self.start()
        said, events = self.turn("5", "[move: 5]", "[none] Ooh, tough one.", "[play: 1] Corner!")
        self.assertIn(("place", 1, "O"), events)

    def test_after_three_misses_loopy_takes_a_random_free_cell(self):
        self.start()
        said, events = self.turn("5", "[move: 5]", "[play: 5] A", "[play: 5] B", "[play: 5] C")
        placed = [e for e in events if e[0] == "place" and e[2] == "O"]
        self.assertEqual(len(placed), 1)
        self.assertTrue(1 <= placed[0][1] <= 9 and placed[0][1] != 5)
        self.assertIn("I'll just take", said)
        for comment in ("A", "B", "C"):
            self.assertNotIn(f" {comment}", f" {said}")

    def test_play_tag_outside_a_game_changes_nothing(self):
        said, events = self.turn("hello", "[play: 5] Hello there!")
        self.assertEqual(events, [])
        self.assertIsNone(self.brain.game)


class Endings(GameTest):
    def play(self, *rounds):
        """rounds: (your cell, Loopy's reply or None when the game ends on your move)."""
        self.start()
        said = events = None
        for cell, loopy in rounds:
            replies = [f"[move: {cell}]"] + ([loopy] if loopy else [])
            said, events = self.turn(str(cell), *replies)
        return said, events

    def test_you_win(self):
        said, events = self.play((1, "[play: 4] a"), (2, "[play: 5] b"), (3, None))
        self.assertIn(("over", "X", (0, 1, 2), {"you": 1, "draws": 0, "loopy": 0}), events)
        self.assertIn("you win", said)

    def test_loopy_wins(self):
        said, events = self.play((1, "[play: 4] a"), (2, "[play: 5] b"), (9, "[play: 6] Got you!"))
        self.assertEqual(events[-1], ("over", "O", (3, 4, 5), {"you": 0, "draws": 0, "loopy": 1}))
        self.assertIn("I win", said)

    def test_draw(self):
        said, events = self.play((1, "[play: 2] a"), (3, "[play: 5] b"), (4, "[play: 7] c"),
                                 (6, "[play: 9] d"), (8, None))
        self.assertEqual(events[-1][:2], ("over", "draw"))
        self.assertIn("draw", said)

    def test_no_after_a_game_closes_the_board(self):
        # Reported: after winning, "No, close the game" (heard as "No closed again")
        # was taken as "close Chrome".
        self.play((1, "[play: 4] a"), (2, "[play: 5] b"), (3, None))
        said, events = self.turn("No closed again", "[game: stop] Okay, good game!")
        self.assertEqual(events, [("stop",)])
        self.assertIsNone(self.brain.pending)  # no Chrome question pending

    def test_anything_but_a_rematch_closes_the_board(self):
        # Seen with the real model: "no thanks" got a chatty reply but the board stayed open.
        self.play((1, "[play: 4] a"), (2, "[play: 5] b"), (3, None))
        said, events = self.turn("no thanks", "[none] No worries, I'm here if you change your mind!")
        self.assertEqual(events, [("stop",)])
        self.assertIsNone(self.brain.game)

    def test_chatting_during_a_game_keeps_the_board(self):
        self.start()
        self.turn("5", "[move: 5]", "[play: 1] Corner!")
        said, events = self.turn("this is fun", "[none] It is! Your move.")
        self.assertEqual(events, [])
        self.assertIsNotNone(self.brain.game)

    def test_model_is_told_how_to_close_after_a_game(self):
        self.play((1, "[play: 4] a"), (2, "[play: 5] b"), (3, None))
        over = [m["content"] for m in self.brain._history if m["content"].startswith("Game over")][-1]
        self.assertIn("[game: stop]", over)
        self.assertIn("close the game", over)

    def test_rematch_keeps_the_score(self):
        self.play((1, "[play: 4] a"), (2, "[play: 5] b"), (3, None))
        self.start("[play: 5] Me first this time!")
        self.assertEqual(self.brain.scores, {"you": 1, "draws": 0, "loopy": 0})


class CellNames(unittest.TestCase):
    def test_prompt_names_every_cell(self):
        # Reported: Loopy played 8 (bottom-middle) and said "top right corner".
        prompt = brain._system_prompt()
        for name in ["1 top-left", "2 top-middle", "3 top-right", "4 middle-left", "5 center",
                     "6 middle-right", "7 bottom-left", "8 bottom-middle", "9 bottom-right"]:
            self.assertIn(name, prompt)


class LongConversations(GameTest):
    def test_history_stays_inside_the_context_window(self):
        # A long chat must never overflow the model's context (it would crash).
        for i in range(300):
            self.turn(f"message number {i} " + "blah " * 20, "[none] " + "ok " * 20)
        tokens = sum(len(m["content"].split()) + 5 for m in self.brain._history)
        self.assertLess(tokens, brain.N_CTX)
        self.assertEqual(self.brain._history[0]["role"], "user")


if __name__ == "__main__":
    unittest.main()
