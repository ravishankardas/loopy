"""Knock detection: hitting the laptop should trigger it; talking, music,
typing-level sounds and shouts should not."""

import unittest

import numpy as np

from loopy import listener as L

RATE, N = L.SAMPLE_RATE, L.FRAME
t = np.arange(N) / RATE
rng = np.random.default_rng(0)


def quiet():
    # A quiet room, as measured on the laptop this was tuned on (~0.0005).
    return (rng.normal(0, 0.0005, N)).astype(np.float32)


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

    def test_off_while_the_model_loads(self):
        # Requested: no knock reactions during startup, while the model loads.
        self.mic.knocks_enabled = False
        self.frames(hit())
        self.assertEqual(self.knocks, [])
        self.mic.enable_knocks(after=1.0)       # loading finished
        self.frames(hit())                      # still settling: ignored
        self.assertEqual(self.knocks, [])
        self.frames(*[quiet()] * int(1.0 * RATE / N + 2))
        self.frames(hit())
        self.assertEqual(len(self.knocks), 1)

    def test_tiny_clicks_are_not(self):
        # Reported false alarms: "knock!" at peak 0.01 without any hit.
        self.frames(hit(peak=0.01))
        self.assertEqual(self.knocks, [])

    def test_real_firm_hits(self):
        # Measured with `loopy knock-test`: firm hits peaked at 0.09-0.18.
        for peak in (0.091, 0.151, 0.18):
            self.frames(*[quiet()] * int(L.KNOCK_COOLDOWN * RATE / N + 2))
            self.frames(hit(peak=peak))
        self.assertEqual(len(self.knocks), 3)

    def test_audio_glitch_from_a_busy_cpu_is_not(self):
        # Reported: two "knocks" while the model was loading (CPU flat out).
        # When the mic drops audio, the gap clicks; that must not count.
        self.mic._glitch = True                 # what the audio callback sets on input overflow
        self.frames(hit(peak=0.11))
        self.assertEqual(self.knocks, [])
        self.frames(*[quiet()] * int(L.KNOCK_GLITCH_SKIP * RATE / N + 2))
        self.frames(hit(peak=0.2))              # a real hit afterwards still counts
        self.assertEqual(len(self.knocks), 1)

    def test_right_after_loopy_speaks_is_not(self):
        # Reported: "knock!" just after Loopy finished talking (its voice's tail end).
        # The mic is paused while Loopy talks...
        self.mic.pause()
        self.mic.feed(quiet())
        self.mic.resume()                       # ...and comes back on
        self.mic._vad = lambda frame: 0.0       # no voice model needed for this test
        self.mic.feed(hit(peak=0.09))
        self.assertEqual(self.knocks, [])
        for _ in range(int(L.KNOCK_RESUME_SKIP * RATE / N + 2)):
            self.mic.feed(quiet())
        self.mic.feed(hit(peak=0.2))            # once settled, a real hit counts
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
        self.frames(hit(peak=0.02))
        self.assertEqual(self.knocks, [])
        self.mic.knock_peak = 0.01              # more sensitive, from settings.json:
        self.mic.knock_above_noise = 10.0       # "knock_peak" and "knock_above_noise"
        self.frames(*[quiet()] * int(L.KNOCK_COOLDOWN * RATE / N + 2))
        self.frames(hit(peak=0.02))
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
