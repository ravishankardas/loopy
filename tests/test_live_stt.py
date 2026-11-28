"""Streaming speech to text: the listener sends the audio so far while the
user talks, and when they stop, the text from the pass that already heard
everything is used, with no second pass."""

import threading
import unittest

import numpy as np

from loopy.listener import END_FRAMES, FRAME, PAUSE_FRAMES, Listener
from loopy.stt import LiveTranscription


class FakeWhisper:
    """'Transcribes' audio into how many frames it had; can be held up."""

    def __init__(self):
        self.calls: list[int] = []
        self.gate = threading.Event()
        self.gate.set()
        self.started = threading.Event()

    def transcribe(self, audio):
        self.started.set()
        self.gate.wait(5)
        self.calls.append(len(audio))
        return f"{len(audio) // FRAME} frames"


class Live(unittest.TestCase):
    def setUp(self):
        self.whisper = FakeWhisper()
        self.partials: list[str] = []
        self.live = LiveTranscription(self.whisper, on_partial=self.partials.append)

    def audio(self, frames):
        return np.zeros(frames * FRAME, dtype=np.float32)

    def test_a_pass_that_heard_everything_is_the_final_text(self):
        self.live.update(self.audio(40), speech_end=34 * FRAME)
        text = self.live.finish(self.audio(56))           # + the silence the listener waited for
        self.assertEqual(text, "40 frames")
        self.assertEqual(self.whisper.calls, [40 * FRAME])  # no second pass
        self.assertEqual(self.live.reused, 1)

    def test_speech_after_the_pass_means_a_fresh_pass(self):
        self.live.update(self.audio(40), speech_end=34 * FRAME)
        self.live.finish(self.audio(40))                   # wait for it
        self.live.update(self.audio(20), speech_end=20 * FRAME)
        self.whisper.gate.clear()
        self.live.update(self.audio(30), speech_end=50 * FRAME)  # speech went on past the audio sent
        self.whisper.gate.set()
        self.assertEqual(self.live.finish(self.audio(70)), "70 frames")

    def test_only_the_newest_audio_waiting_is_transcribed(self):
        self.whisper.gate.clear()
        self.live.update(self.audio(10), speech_end=10 * FRAME)
        self.assertTrue(self.whisper.started.wait(5))      # worker busy with the first one
        self.live.update(self.audio(20), speech_end=20 * FRAME)
        self.live.update(self.audio(30), speech_end=30 * FRAME)
        self.whisper.gate.set()
        self.assertEqual(self.live.finish(self.audio(50)), "30 frames")
        self.assertNotIn(20 * FRAME, self.whisper.calls)

    def test_partial_text_is_shown_while_talking(self):
        self.live.update(self.audio(16), speech_end=16 * FRAME)
        self.live.finish(self.audio(16))
        self.assertEqual(self.partials, ["16 frames"])

    def test_a_dropped_utterance_does_not_leak_into_the_next(self):
        self.live.update(self.audio(40), speech_end=40 * FRAME)
        self.live.cancel()
        self.assertEqual(self.live.finish(self.audio(12)), "12 frames")

    def test_a_failed_pass_falls_back_to_the_whole_utterance(self):
        calls = []

        class Flaky:
            def transcribe(self, audio):
                calls.append(len(audio))
                if len(calls) == 1:
                    raise RuntimeError("boom")
                return "fine"

        live = LiveTranscription(Flaky())
        live.update(self.audio(40), speech_end=40 * FRAME)
        self.assertEqual(live.finish(self.audio(56)), "fine")


class ScriptedVoice:
    """Stands in for the voice detector: one speech probability per frame."""

    def __init__(self, pattern):
        self._probs = iter(pattern)

    def __call__(self, frame):
        return next(self._probs)

    def reset(self):
        pass


class ListenerSendsAudioSoFar(unittest.TestCase):
    def run_listener(self, pattern, partial_every=None):
        """pattern: one probability per frame from the voice detector."""
        sent, done = [], []
        mic = Listener(on_utterance=done.append, on_partial=lambda a, end: sent.append((len(a), end)),
                       partial_every=partial_every)
        mic._vad = ScriptedVoice(pattern)
        mic.knocks_enabled = False
        for _ in pattern:
            mic.feed(np.full(FRAME, 0.01, dtype=np.float32))
        return sent, done

    def test_a_pause_sends_everything_said(self):
        sent, done = self.run_listener([0.9] * 20 + [0.0] * (END_FRAMES + 2))
        self.assertEqual(len(done), 1)
        self.assertEqual(len(sent), 1)
        frames, speech_end = sent[0]
        self.assertEqual(frames, (20 + PAUSE_FRAMES) * FRAME)
        self.assertEqual(speech_end, 20 * FRAME)
        self.assertGreaterEqual(frames, speech_end)

    def test_the_last_send_before_the_end_has_all_the_speech(self):
        # talk, short pause, talk again, then stop
        sent, done = self.run_listener([0.9] * 20 + [0.0] * 10 + [0.9] * 15 + [0.0] * (END_FRAMES + 2))
        self.assertEqual(len(done), 1)
        frames, speech_end = sent[-1]
        self.assertEqual(speech_end, 45 * FRAME)
        self.assertGreaterEqual(frames, speech_end)

    def test_live_passes_while_talking(self):
        sent, _ = self.run_listener([0.9] * 40 + [0.0] * (END_FRAMES + 2), partial_every=0.5)
        self.assertGreaterEqual(len(sent), 3)   # every ~16 frames, then the pause

    def test_without_the_hook_nothing_changes(self):
        done = []
        mic = Listener(on_utterance=done.append)
        mic._vad = ScriptedVoice([0.9] * 20 + [0.0] * (END_FRAMES + 2))
        for _ in range(20 + END_FRAMES + 2):
            mic.feed(np.full(FRAME, 0.01, dtype=np.float32))
        self.assertEqual(len(done), 1)


if __name__ == "__main__":
    unittest.main()
