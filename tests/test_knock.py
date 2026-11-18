"""Knock detection: hitting the laptop should trigger it; talking, music,
typing-level sounds and shouts should not."""

import unittest

import numpy as np

from loopy import listener as L

RATE, N = L.SAMPLE_RATE, L.FRAME
t = np.arange(N) / RATE
rng = np.random.default_rng(0)


def quiet():
    return (rng.normal(0, 0.003, N)).astype(np.float32)          # a quiet room


def impact(peak=0.9):
    # A hit: a sharp click that dies away within a few milliseconds.
    return (peak * np.exp(-t / 0.003) * np.sin(2 * np.pi * 900 * t)).astype(np.float32)


def hit(peak=0.9):
    return impact(peak) + quiet()


def tone(level):
    return (level * np.sin(2 * np.pi * 220 * t)).astype(np.float32)  # steady sound, e.g. a voice or music


class KnockDetection(unittest.TestCase):
    def setUp(self):
        self.knocks = []
        self.mic = L.Listener(on_utterance=lambda _a: None, on_knock=self.knocks.append)
        for _ in range(40):                     # ~1.3 s to learn the room first
            self.mic._check_knock(quiet(), 0.0)

    def frames(self, *frames, speech=0.0):
        for f in frames:
            self.mic._check_knock(f, speech)

    def test_a_hit_is_a_knock(self):
        self.frames(hit())
        self.assertEqual(len(self.knocks), 1)
        self.assertGreater(self.knocks[0], 0.5)

    def test_normal_speech_level_is_not(self):
        self.frames(*[tone(0.25)] * 10, speech=0.9)
        self.assertEqual(self.knocks, [])

    def test_loud_steady_sound_is_not(self):
        self.frames(*[tone(0.8)] * 10)          # loud, but not a spike
        self.assertEqual(self.knocks, [])

    def test_a_shout_is_not(self):
        self.frames(hit(), speech=0.9)          # spiky but the voice detector says it's speech
        self.assertEqual(self.knocks, [])

    def test_a_moderate_hit_counts(self):
        # Reported: real hits didn't trigger. A short spike inside a 32 ms frame
        # used to be averaged away, and a fixed 0.5 peak was far too strict.
        self.frames(hit(peak=0.25))
        self.assertEqual(len(self.knocks), 1)

    def test_a_short_click_inside_a_quiet_frame_counts(self):
        frame = quiet()
        click = hit(peak=0.4)[:64]                  # 4 ms of impact...
        frame[200:264] += click                     # ...in the middle of 32 ms of quiet room
        self.frames(frame)
        self.assertEqual(len(self.knocks), 1)

    def test_soft_tap_barely_above_the_room_is_not(self):
        self.frames(hit(peak=0.01))                 # only a few times louder than the room
        self.assertEqual(self.knocks, [])

    def test_real_hits_from_a_quiet_laptop_mic(self):
        # Measured on the laptop this was tuned on: a very quiet room (~0.0005)
        # and hits peaking at only 0.01-0.04, but 20-86x louder than the room.
        mic = L.Listener(on_utterance=lambda _a: None, on_knock=self.knocks.append)
        silent_room = lambda: (rng.normal(0, 0.0005, N)).astype(np.float32)
        for _ in range(40):
            mic._check_knock(silent_room(), 0.0)
        mic._check_knock(impact(peak=0.04) + silent_room(), 0.0)
        self.assertEqual(len(self.knocks), 1)

    def test_the_dying_away_part_of_a_hit_is_not_a_second_knock(self):
        self.frames(hit(peak=0.04 * 20))            # hit...
        self.frames(*[quiet() + hit(peak=0.02) for _ in range(3)])   # ...ringing on
        self.assertEqual(len(self.knocks), 1)

    def test_one_hit_triggers_once(self):
        self.frames(hit(), quiet(), hit(), quiet(), hit())   # bounces within ~0.2 s
        self.assertEqual(len(self.knocks), 1)

    def test_another_hit_after_the_cooldown_counts(self):
        self.frames(hit())
        self.frames(*[quiet()] * int(L.KNOCK_COOLDOWN * RATE / N + 2))
        self.frames(hit())
        self.assertEqual(len(self.knocks), 2)

    def test_nothing_before_the_room_is_learned(self):
        fresh = L.Listener(on_utterance=lambda _a: None, on_knock=self.knocks.append)
        fresh._check_knock(hit(), 0.0)
        self.assertEqual(self.knocks, [])

    def test_sensitivity_can_be_changed(self):
        self.frames(hit(peak=0.01))
        self.assertEqual(self.knocks, [])
        self.mic.knock_above_noise = 1.3        # very sensitive, e.g. from settings.json "knock_above_noise"
        self.frames(*[quiet()] * int(L.KNOCK_COOLDOWN * RATE / N + 2))
        self.frames(hit(peak=0.01))
        self.assertEqual(len(self.knocks), 1)

    def test_background_adapts_to_a_noisy_room(self):
        # In a loud room, the same moderate hit is not "much louder than the room".
        for _ in range(120):
            self.mic._check_knock((rng.normal(0, 0.08, N)).astype(np.float32), 0.0)
        self.knocks.clear()                     # the room suddenly getting loud may count once
        self.frames(*[(rng.normal(0, 0.08, N)).astype(np.float32) for _ in range(5)])
        self.frames(hit(peak=0.25) + (rng.normal(0, 0.08, N)).astype(np.float32))
        self.assertEqual(self.knocks, [])


if __name__ == "__main__":
    unittest.main()
