"""The saved warm-up: later starts load it instead of re-reading the prompt,
and any change to the prompt (or model) makes it redo and re-save it.

Uses a fake llama.cpp model, so nothing heavy runs."""

import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import llama_cpp

from loopy import brain
from loopy.brain import Brain


class FakeModel:
    """Counts warm-ups and remembers what was saved/restored."""
    instances: list["FakeModel"] = []

    def __init__(self, path, **_kwargs):
        self.warmups = 0
        self.restored = None
        FakeModel.instances.append(self)

    def create_chat_completion(self, messages, max_tokens=None, **_kwargs):
        self.warmups += 1
        self.last_prompt = messages
        return iter(())

    def save_state(self):
        return {"prompt": self.last_prompt}       # picklable, like LlamaState

    def load_state(self, state):
        self.restored = state


class SavedWarmup(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.model = self.dir / "model.gguf"
        self.model.write_bytes(b"fake model")
        FakeModel.instances = []
        p = mock.patch.object(llama_cpp, "Llama", FakeModel)
        p.start()
        self.addCleanup(p.stop)

    def start(self) -> tuple[Brain, FakeModel]:
        b = Brain(self.model)
        b.load()
        return b, FakeModel.instances[-1]

    def saved_files(self):
        return sorted(p.name for p in self.dir.glob("warmup-*.state"))

    def test_first_start_warms_up_and_saves(self):
        b, model = self.start()
        self.assertEqual(model.warmups, 1)
        self.assertEqual(b.warmup_source, "fresh")
        self.assertEqual(len(self.saved_files()), 1)

    def test_next_start_uses_the_saved_warmup(self):
        self.start()
        b, model = self.start()
        self.assertEqual(model.warmups, 0)            # no 80-second warm-up
        self.assertIsNotNone(model.restored)
        self.assertEqual(b.warmup_source, "saved")

    def test_a_prompt_change_redoes_and_replaces_it(self):
        # Requested: when the prompt changes, the saved state updates by itself.
        self.start()
        old = self.saved_files()
        with mock.patch.object(brain, "_system_prompt", lambda: "a different prompt"):
            b, model = self.start()
        self.assertEqual(model.warmups, 1)            # warmed up again...
        self.assertEqual(b.warmup_source, "fresh")
        new = self.saved_files()
        self.assertEqual(len(new), 1)                 # ...saved, and the old one is gone
        self.assertNotEqual(new, old)

    def test_new_examples_or_abilities_count_as_a_change(self):
        self.start()
        with mock.patch.object(brain, "_examples", lambda: [{"role": "user", "content": "new example"}]):
            b, model = self.start()
        self.assertEqual(model.warmups, 1)

    def test_a_different_model_file_counts_as_a_change(self):
        self.start()
        self.model.write_bytes(b"a newer, bigger fake model")
        os.utime(self.model, (time.time() + 10, time.time() + 10))
        b, model = self.start()
        self.assertEqual(model.warmups, 1)

    def test_a_broken_file_falls_back_to_a_normal_warmup(self):
        self.start()
        (self.dir / self.saved_files()[0]).write_bytes(b"not a saved state")
        b, model = self.start()
        self.assertEqual(model.warmups, 1)
        self.assertEqual(b.warmup_source, "fresh")
        b2, model2 = self.start()                     # and it's fixed for next time
        self.assertEqual(model2.warmups, 0)

    def test_failing_to_save_doesnt_break_loading(self):
        with mock.patch.object(FakeModel, "save_state", side_effect=MemoryError):
            b, model = self.start()
        self.assertEqual(model.warmups, 1)
        self.assertEqual(self.saved_files(), [])
        self.assertEqual(list(self.dir.glob("*.tmp")), [])   # no half-written leftovers


if __name__ == "__main__":
    unittest.main()
