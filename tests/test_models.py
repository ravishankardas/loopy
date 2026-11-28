"""Choosing the conversation model (Llama or Gemma), and what changes with it."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import llama_cpp  # noqa: E402

from loopy import brain, models, settings  # noqa: E402
from loopy.brain import Brain, _fold_system  # noqa: E402

LLAMA, GEMMA = models.get_llm("llama-3.2-3b"), models.get_llm("gemma-3-4b")


class Choosing(unittest.TestCase):
    def test_both_models_are_listed(self):
        self.assertIsNotNone(LLAMA)
        self.assertIsNotNone(GEMMA)
        self.assertEqual(models.DEFAULT_LLM, LLAMA)
        self.assertTrue(GEMMA.url.startswith("https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF/"))

    def test_the_setting_picks_the_model(self):
        with mock.patch.object(settings, "load", lambda: {"llm": "gemma-3-4b"}):
            self.assertEqual(models.active_llm(), GEMMA)
            self.assertEqual(models.llm_path(), GEMMA.path)
        with mock.patch.object(settings, "load", lambda: {}):
            self.assertEqual(models.active_llm(), LLAMA)
        with mock.patch.object(settings, "load", lambda: {"llm": "no-such-model"}):
            self.assertEqual(models.active_llm(), LLAMA)

    def test_the_brain_knows_which_format_to_use(self):
        self.assertTrue(Brain(GEMMA.path)._fold_system)
        self.assertFalse(Brain(LLAMA.path)._fold_system)
        self.assertFalse(Brain("some-other.gguf")._fold_system)


class GemmaMessages(unittest.TestCase):
    """Gemma has no "system" role and needs user/model turns to alternate."""

    def messages(self) -> list[dict]:
        return Brain(GEMMA.path)._messages([
            {"role": "user", "content": "how's my battery"},
            {"role": "assistant", "content": "[check: battery]"},
            {"role": "system", "content": "Result: battery = 57%"},
        ])

    def test_only_the_prompt_is_a_system_message(self):
        msgs = self.messages()
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], brain._system_prompt())
        self.assertNotIn("system", [m["role"] for m in msgs[1:]])

    def test_turns_alternate(self):
        roles = [m["role"] for m in self.messages()[1:]]
        self.assertEqual(roles[0], "user")
        self.assertEqual(roles[-1], "user")                  # the model always answers a user turn
        self.assertTrue(all(a != b for a, b in zip(roles, roles[1:], strict=False)), roles)

    def test_nothing_is_lost(self):
        text = "\n".join(m["content"] for m in self.messages())
        for part in ["how's my battery", "[check: battery]", "Result: battery = 57%", "The examples are over"]:
            self.assertIn(part, text)
        self.assertIn("(Note from Loopy's app, not said by the user) Result: battery = 57%", text)

    def test_folding_rules(self):
        folded = _fold_system([
            {"role": "system", "content": "prompt"},
            {"role": "system", "content": "game started"},
            {"role": "user", "content": "put a cross in 2"},
            {"role": "assistant", "content": "[move: 2]"},
            {"role": "system", "content": "board"},
        ])
        self.assertEqual([m["role"] for m in folded], ["system", "user", "assistant", "user"])
        self.assertEqual(folded[1]["content"],
                         "(Note from Loopy's app, not said by the user) game started\n\nput a cross in 2")

    def test_llama_keeps_system_messages(self):
        msgs = Brain(LLAMA.path)._messages([{"role": "system", "content": "board"}])
        self.assertEqual(msgs[-1], {"role": "system", "content": "board"})


class FakeModel:
    def __init__(self, path, **_kwargs):
        pass

    def create_chat_completion(self, messages, **_kwargs):
        return iter(())

    def save_state(self):
        return {"state": 1}

    def load_state(self, state):
        pass


class WarmupPerModel(unittest.TestCase):
    """Switching models must not throw away the other model's saved warm-up."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        p = mock.patch.object(llama_cpp, "Llama", FakeModel)
        p.start()
        self.addCleanup(p.stop)

    def model(self, name: str) -> Path:
        path = self.dir / name
        path.write_bytes(b"fake " + name.encode())
        return path

    def test_each_model_keeps_its_own(self):
        llama, gemma = self.model(LLAMA.file), self.model(GEMMA.file)
        Brain(llama).load()
        Brain(gemma).load()
        self.assertTrue(Brain(llama).warmup_ready())
        self.assertTrue(Brain(gemma).warmup_ready())
        self.assertEqual(len(list(self.dir.glob("warmup-*.state"))), 2)

    def test_old_unnamed_warmups_are_cleaned_up(self):
        old = self.dir / "warmup-0123456789abcdef.state"
        old.write_bytes(b"from before warm-ups were named per model")
        Brain(self.model(LLAMA.file)).load()
        self.assertFalse(old.exists())


if __name__ == "__main__":
    unittest.main()
