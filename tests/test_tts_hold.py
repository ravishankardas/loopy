"""While a sound clip plays, Loopy's own voice holds: nothing is spoken until
release(), and nothing is lost either. Uses a fake voice and a fake speaker
device, so no audio is played."""

import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from loopy import tts


class FakeVoice:
    def synthesize(self, text):
        yield SimpleNamespace(sample_rate=16000, sample_channels=1,
                              audio_int16_array=np.zeros(1600, dtype=np.int16))


class FakeOutput:
    """Records when audio is written, instead of playing it."""
    writes: list[float] = []

    def __init__(self, **_kw): pass
    def start(self): pass
    def stop(self): pass
    def abort(self): pass
    def close(self): pass

    def write(self, _audio):
        FakeOutput.writes.append(time.monotonic())
        return False


class HoldWhileAClipPlays(unittest.TestCase):
    def setUp(self):
        FakeOutput.writes = []
        patches = [mock.patch.object(tts.PiperVoice, "load", lambda *_a, **_k: FakeVoice()),
                   mock.patch.object(tts.sd, "OutputStream", FakeOutput)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.speaker = tts.Speaker("fake.onnx")
        self.addCleanup(self.speaker.close)

    def test_nothing_is_spoken_while_held(self):
        started = threading.Event()
        self.speaker.hold()
        self.speaker.say("Hello there!", on_start=lambda _t: started.set())
        time.sleep(0.6)
        self.assertEqual(FakeOutput.writes, [])          # quiet while the clip plays
        self.assertFalse(started.is_set())

        released_at = time.monotonic()
        self.speaker.release()
        self.assertTrue(started.wait(2))                  # ...and spoken afterwards
        time.sleep(0.2)
        self.assertTrue(FakeOutput.writes)
        self.assertGreaterEqual(FakeOutput.writes[0], released_at)

    def test_speaks_normally_when_not_held(self):
        done = threading.Event()
        self.speaker.say("Hi!", on_done=done.set)
        self.assertTrue(done.wait(2))
        self.assertTrue(FakeOutput.writes)

    def test_stop_while_held_doesnt_hang(self):
        self.speaker.hold()
        self.speaker.say("This will be cut off.")
        time.sleep(0.2)
        self.speaker.stop()
        self.speaker.release()
        self.speaker.close(timeout=2)
        self.assertIsNone(self.speaker._worker)


if __name__ == "__main__":
    unittest.main()
