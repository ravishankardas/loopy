"""Snake rules: moving, turning, eating, growing, dying, speeding up."""

import random
import unittest

from loopy.snake import FASTEST, START_INTERVAL, SnakeGame


def game(**kw) -> SnakeGame:
    g = SnakeGame(10, 10, rng=random.Random(1), **kw)
    g.food = (0, 0)  # out of the way unless a test puts it somewhere
    return g


class Moving(unittest.TestCase):
    def test_starts_in_the_middle_moving_right(self):
        g = game()
        self.assertEqual(g.head, (5, 5))
        self.assertEqual(len(g.snake), 3)
        g.step()
        self.assertEqual(g.head, (6, 5))
        self.assertEqual(len(g.snake), 3)

    def test_turning(self):
        g = game()
        g.turn("up")
        g.step()
        self.assertEqual(g.head, (5, 4))

    def test_cant_turn_straight_back_into_itself(self):
        g = game()
        g.turn("left")            # moving right: reversing is ignored
        g.step()
        self.assertEqual(g.head, (6, 5))
        self.assertFalse(g.over)

    def test_quick_presses_are_applied_one_per_step(self):
        g = game()
        g.turn("up")
        g.turn("left")            # up-then-left in one tick: a quick U-turn
        g.step()
        self.assertEqual(g.head, (5, 4))
        g.step()
        self.assertEqual(g.head, (4, 4))
        self.assertFalse(g.over)


class Eating(unittest.TestCase):
    def test_eating_grows_scores_and_moves_the_food(self):
        g = game()
        g.food = (6, 5)
        self.assertEqual(g.step(), "ate")
        self.assertEqual(g.score, 1)
        self.assertEqual(len(g.snake), 4)
        self.assertNotEqual(g.food, (6, 5))
        self.assertNotIn(g.food, g.snake)

    def test_food_never_appears_on_the_snake(self):
        g = SnakeGame(4, 4, rng=random.Random(3))
        for _ in range(50):
            self.assertNotIn(g._place_food(), g.snake)

    def test_it_speeds_up_as_it_grows(self):
        g = game()
        self.assertEqual(g.interval, START_INTERVAL)
        g.score = 10
        self.assertLess(g.interval, START_INTERVAL)
        g.score = 1000
        self.assertEqual(g.interval, FASTEST)


class Dying(unittest.TestCase):
    def test_going_off_an_edge_comes_back_on_the_other_side(self):
        g = game()
        for _ in range(4):
            g.step()
        self.assertEqual(g.head, (9, 5))     # at the right edge of a 10-wide field...
        self.assertIsNone(g.step())
        self.assertEqual(g.head, (0, 5))     # ...and back in on the left
        self.assertFalse(g.over)

    def test_wrapping_works_in_every_direction(self):
        for direction, start, expected in [("up", (5, 0), (5, 9)), ("down", (5, 9), (5, 0)),
                                           ("left", (0, 5), (9, 5)), ("right", (9, 5), (0, 5))]:
            g = game()
            g.snake = type(g.snake)([start])
            g.direction = direction
            g.step()
            self.assertEqual(g.head, expected, direction)

    def test_nothing_happens_after_game_over(self):
        g = game()
        g.over = True
        self.assertIsNone(g.step())

    def test_hitting_itself(self):
        g = game()
        # Head at (5,5) going up; (4,5) to its left is the middle of its body.
        g.snake = type(g.snake)([(5, 5), (5, 6), (4, 6), (4, 5), (4, 4), (3, 4)])
        g.direction = "up"
        g.turn("left")
        self.assertEqual(g.step(), "died")
        self.assertTrue(g.over)

    def test_moving_into_the_tails_old_spot_is_fine(self):
        g = game()
        g.snake = type(g.snake)([(5, 5), (5, 6), (4, 6), (4, 5)])  # tail at (4,5), right behind-left
        g.direction = "up"
        g.turn("left")
        self.assertIsNone(g.step())          # the tail moves away in the same step
        self.assertFalse(g.over)


class NobodySteering(unittest.TestCase):
    """Requested: a snake that just goes in one edge and out the other,
    untouched, means nobody's playing."""

    def test_untouched_wraps_are_counted(self):
        g = game()
        for _ in range(4 + 10 * 2):               # cross the right edge twice
            g.step()
        self.assertEqual(g.wraps_untouched, 2)
        self.assertFalse(g.idle)
        for _ in range(10):
            g.step()
        self.assertEqual(g.wraps_untouched, 3)
        self.assertTrue(g.idle)

    def test_turning_resets_the_count(self):
        g = game()
        for _ in range(4 + 10 * 2):
            g.step()
        g.turn("up")
        g.step()
        self.assertEqual(g.wraps_untouched, 0)


if __name__ == "__main__":
    unittest.main()
