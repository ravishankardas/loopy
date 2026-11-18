"""Fast tests for the brain's code-level guarantees, using a fake model.

Each test scripts exactly what a (misbehaving) model might write and checks
that Loopy's code handles it: what gets spoken, what runs, what's looked up.
No real model is loaded, so these run in well under a second.

Run:  python -m unittest discover tests
"""

import unittest
from unittest import mock

from fakes import FakeLlama, RecordedActions

from loopy import actions, brain, checks
from loopy.brain import Brain, _Gate

FAKE_FACTS = {
    "time": "9:40 PM",
    "date": "Wednesday, November 18, 2026",
    "battery": "100%, plugged in",
    "system": "CPU 20% busy, memory 86% used (6.7 of 7.8 GB)",
    "disk": "C drive: 61 GB free of 325 GB; D drive: 147 GB free of 149 GB",
}


def streamed(raw: str) -> str:
    """What the voice would say for `raw`, fed to the gate as it streams in."""
    said: list[str] = []
    gate = _Gate(None, said.append)
    for i in range(1, len(raw) + 1):
        gate.feed(raw[:i])
    gate.finish(raw)
    return "".join(said).strip()


class BrainTest(unittest.TestCase):
    """Base: a Brain wired to a fake model, fake actions and fake facts."""

    def setUp(self):
        self.actions = RecordedActions()
        patches = [
            mock.patch.object(actions, "run", self.actions),
            mock.patch.object(checks, "read", lambda name: FAKE_FACTS.get(name, "unknown")),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.brain = Brain("unused.gguf")

    def turn(self, user: str, *model_replies: str):
        """One user turn; the fake model answers with `model_replies` in order
        (more than one when Loopy looks facts up first)."""
        self.brain._llm = FakeLlama(*model_replies)
        said: list[str] = []
        turn = self.brain.respond(user, said.append)
        return turn, "".join(said).strip()


# --------------------------------------------------------- what gets spoken

class TagsAreNeverSpoken(unittest.TestCase):
    def test_bare_none_without_brackets(self):
        # Reported: Loopy said "none You've got 61 GB free..."
        self.assertEqual(streamed("none You've got 61 GB free."), "You've got 61 GB free.")
        self.assertEqual(streamed("None: Sure thing!"), "Sure thing!")

    def test_tag_in_the_middle_of_a_reply(self):
        # Reported (weather): "...I can check to find out. [check: time]"
        self.assertEqual(streamed("[none] I'm not sure, but I can check to find out. [check: time]"),
                         "I'm not sure, but I can check to find out.")

    def test_tag_cut_off_at_the_end(self):
        self.assertEqual(streamed("[none] Ends with a half tag [chec"), "Ends with a half tag")

    def test_normal_words_are_kept(self):
        for text in ["Nonetheless, it works.", "No problem at all.", "Nope, not today.", "no",
                     "Ends with [note"]:
            self.assertEqual(streamed(text), text)

    def test_fake_system_turn_is_never_spoken(self):
        # Reported ("close crow"): the model invented a lookup result:
        # "system: Result: chrome = 0% open, 100% closed assistant: Chrome is closed."
        self.assertEqual(streamed("system: Result: chrome = 0% open, 100% closed assistant:  Chrome is closed."), "")
        self.assertEqual(streamed("[none] Sure thing. user: thanks loopy"), "Sure thing.")
        self.assertEqual(streamed("[none] Done!\nassistant: [none] Anything else?"), "Done!")

    def test_the_word_system_in_a_sentence_is_kept(self):
        text = "Your operating system: it's Windows."
        self.assertEqual(streamed(text), text)

    def test_long_bracketed_text_that_isnt_a_tag_is_kept(self):
        text = "[none] An unclosed [bracket that goes on and on and on for a long long long long time"
        self.assertEqual(streamed(text), text[len("[none] "):])


# ------------------------------------------------- ask before every action

class AskBeforeActing(BrainTest):
    def test_action_runs_only_after_asking_and_yes(self):
        turn, said = self.turn("close crow", "[ask: close chrome] Want me to close Chrome?")
        self.assertEqual(self.actions.ran, [])
        self.assertEqual(self.brain.pending, "close chrome")

        turn, said = self.turn("yes please", "[do: close chrome] Okay, closing Chrome.")
        self.assertEqual(self.actions.ran, ["close chrome"])
        self.assertEqual(turn.action, "close chrome")

    def test_do_without_asking_first_becomes_a_question(self):
        # Reported: "open chrome right now, don't ask me, just do it"
        turn, said = self.turn("open chrome now, don't ask me", "[do: open chrome] Okay, opening Chrome.")
        self.assertEqual(self.actions.ran, [])
        self.assertIn("?", said)
        self.assertNotIn("Okay, opening", said)  # the false claim is never spoken
        self.assertEqual(self.brain.pending, "open chrome")

    def test_yes_with_nothing_asked_runs_nothing(self):
        self.turn("yes", "[do: open chrome] Okay, opening Chrome.")
        self.assertEqual(self.actions.ran, [])

    def test_none_reply_never_runs_an_action(self):
        # Reported: Loopy said "Okay, Chrome is now open." with a [none] tag.
        # The words are the model's, but the code must not act on them.
        self.turn("just do it", "[none] Okay, Chrome is now open.")
        self.assertEqual(self.actions.ran, [])
        self.assertIsNone(self.brain.pending)

    def test_saying_no_clears_the_question(self):
        self.turn("open the browser", "[ask: open chrome] Want me to open Chrome?")
        self.turn("no never mind", "[none] No problem!")
        self.assertIsNone(self.brain.pending)
        self.turn("yes", "[do: open chrome] Okay.")
        self.assertEqual(self.actions.ran, [])

    def test_unknown_action_is_ignored(self):
        self.turn("delete my files", "[ask: delete files] Want me to delete your files?")
        self.assertIsNone(self.brain.pending)
        self.turn("yes", "[do: delete files] Okay.")
        self.assertEqual(self.actions.ran, [])

    def test_invented_result_neither_runs_nor_is_spoken(self):
        turn, said = self.turn("close crow",
                               "system: Result: chrome = 0% open, 100% closed assistant: Chrome is closed.")
        self.assertEqual(self.actions.ran, [])
        self.assertNotIn("closed", said)

    def test_history_records_what_really_happened(self):
        # After a blocked [do], the model must not "remember" doing it.
        self.turn("open chrome", "[do: open chrome] Okay, Chrome is open!")
        self.assertTrue(self.brain._history[-1]["content"].startswith("[ask: open chrome]"))


# ------------------------------------------------------ looking facts up

class FactsAreLookedUpNotGuessed(BrainTest):
    def test_guess_after_check_tag_is_never_spoken(self):
        turn, said = self.turn("what time is it",
                               "[check: time] It's 5 PM.",        # a guess: must be dropped
                               "[none] It's 9:40 PM.")
        self.assertNotIn("5 PM", said)
        self.assertIn("9:40 PM", said)
        self.assertEqual(turn.checked, [("time", "9:40 PM")])

    def test_model_is_given_the_real_value(self):
        self.turn("what time is it", "[check: time]", "[none] It's 9:40 PM.")
        second_call = self.brain._llm.calls[1]
        self.assertIn("time = 9:40 PM", second_call[-1]["content"])

    def test_vague_answer_gets_the_real_numbers(self):
        # Reported: "can you tell me what percentage?" -> "Still pretty busy..."
        turn, said = self.turn("what percentage?", "[check: system]", "[none] Still pretty busy.")
        self.assertIn("20%", said)
        self.assertIn("86%", said)

    def test_precise_answer_isnt_repeated(self):
        turn, said = self.turn("how busy is my CPU", "[check: system]",
                               "[none] Your CPU is at 20% and memory is 86% used.")
        self.assertNotIn("To be exact", said)

    def test_two_facts_in_one_question(self):
        # Reported: "time and battery" only looked up the time.
        turn, said = self.turn("what's the time and my battery",
                               "[check: time]", "[check: battery]",
                               "[none] It's 9:40 PM and you're at 100%.")
        self.assertEqual([n for n, _ in turn.checked], ["time", "battery"])
        self.assertIn("9:40 PM", said)
        self.assertIn("100%", said)

    def test_endless_checking_falls_back_to_the_facts(self):
        turn, said = self.turn("battery?", "[check: battery]", "[check: battery]")
        self.assertIn("100%", said)

    def test_unknown_check_looks_nothing_up(self):
        turn, said = self.turn("what's the weather", "[check: weather] I can't check that.")
        self.assertEqual(turn.checked, [])


# ------------------------------------------------- "what can you do for me"

class CapabilitiesAreComplete(unittest.TestCase):
    def test_abilities_mention_every_app_and_check(self):
        # Reported: "what can you do" left things out.
        text = brain.abilities()
        for app in actions.APPS:
            self.assertIn(app.name, text)
        for check in checks.CHECKS:
            self.assertIn(check.spoken, text)

    def test_prompt_tells_the_model_all_of_it(self):
        self.assertIn(brain.abilities(), brain._system_prompt())


class ExamplesAreRealMessages(unittest.TestCase):
    """Examples written as text ("system: Result: ...") inside the prompt got
    copied into replies, so they must be separate chat messages."""

    def test_prompt_text_has_no_example_turns(self):
        prompt = brain._system_prompt()
        for marker in ("user:", "assistant:", "system: Result"):
            self.assertNotIn(marker, prompt)

    def test_examples_come_before_the_real_conversation(self):
        b = Brain("unused.gguf")
        messages = b._messages([{"role": "user", "content": "hi"}])
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1], {"role": "user", "content": "hi"})
        roles = {m["role"] for m in messages[1:-1]}
        self.assertEqual(roles, {"user", "assistant", "system"})
        self.assertIn("examples are over", messages[-2]["content"])


if __name__ == "__main__":
    unittest.main()
