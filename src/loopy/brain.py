"""Loopy's conversation: Llama 3.2 3B running locally through llama.cpp.

Every reply starts with a short tag saying what Loopy wants to do, then the
words it speaks:

    [none] I'm doing great, thanks!
    [ask: close chrome] Want me to close Chrome?
    [do: close chrome] Okay, closing Chrome.

The tag comes first so the code can decide what's allowed before anything is
spoken; the words after it are passed on piece by piece as the model writes
them (to the voice, which starts talking after the first few words).

Safety rule (enforced here, not trusted to the model): an action only runs if
Loopy asked about that exact action on the previous turn and the model then
marks it [do: ...] (i.e. the user said yes). Anything else just becomes a
question. A malformed tag is treated as plain chat.
"""

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from loopy import actions

N_CTX = 2048           # tokens of conversation the model can see
N_THREADS = 4          # physical cores
MAX_REPLY_TOKENS = 120
# Dropping old messages makes llama.cpp re-read the whole conversation (slow
# on a laptop CPU), so trim rarely and in one big chunk.
TRIM_AT, TRIM_TO = 40, 16

TextOut = Callable[[str], None]  # receives reply text piece by piece


def _system_prompt() -> str:
    acts = ", ".join(actions.names())
    return f"""You are Loopy, a friendly paperclip who lives on the user's desktop and talks with them by voice.
You're warm, curious and a little playful, like a good friend. You enjoy chatting about anything: their day, ideas, jokes, questions.
Be a real conversation partner: react to what they say, show interest, and often ask a short follow-up question.
If a request is vague, ask what they need. Never guess an action they didn't ask for.
Their words come from speech recognition and may have mistakes, so guess what they meant ("close crow" means "close chrome").
You're speaking aloud: keep it to one or two short, natural sentences. No lists, markdown or emojis.

On the computer you can only: {acts}. Bring these up only when the user asks for them.
If asked for something you can't do, say so kindly and stay in the conversation.

Start every reply with a tag:
[none] for conversation (almost always).
[ask: ACTION] only when the user clearly asks for one of your actions. Always ask a yes/no question first.
[do: ACTION] only when the user just said yes to your question about that action.

Example:
user: i want you to do something for me
assistant: [none] Sure, I'm all ears! What do you need?
user: can you close crome
assistant: [ask: close chrome] You got it, want me to close Chrome?
user: yeah go ahead
assistant: [do: close chrome] Okay, closing Chrome.
user: i'm so tired today
assistant: [none] Oh no, long day? What wore you out?
user: what can you do
assistant: [none] Mostly I love to chat, and I can open or close Chrome for you. What are you up to?
user: can you set a timer
assistant: [none] I wish I could, but timers aren't something I can do yet. What's the timer for?
user: open the browser
assistant: [ask: open chrome] Want me to open Chrome?
user: no never mind
assistant: [none] No problem! So, what's on your mind?"""


@dataclass
class Turn:
    reply: str                              # everything Loopy said this turn
    action: str | None = None               # action carried out, if any
    result: actions.Result | None = None    # its outcome


class Brain:
    def __init__(self, model_path: Path):
        self._model_path = Path(model_path)
        self._llm = None
        self._lock = threading.Lock()
        self._history: list[dict] = []
        self.pending: str | None = None     # action Loopy asked about last turn

    def load(self):
        """Load the model and pre-read the system prompt (~20 s on a laptop
        CPU); called automatically, but best started at app launch."""
        with self._lock:
            if self._llm is None:
                from llama_cpp import Llama

                self._llm = Llama(str(self._model_path), n_ctx=N_CTX, n_threads=N_THREADS, verbose=False)
                # llama.cpp reuses the shared start of the prompt between
                # calls, so the system prompt is only processed this once.
                self._llm.create_chat_completion(self._messages([{"role": "user", "content": "hi"}]), max_tokens=1)
            return self._llm

    def reset(self) -> None:
        """Forget the conversation (Loopy does this when it falls asleep)."""
        with self._lock:
            self._history.clear()
            self.pending = None

    def respond(self, text: str, on_text: TextOut) -> Turn:
        """Reply to `text`. The reply is passed to `on_text` piece by piece
        as the model writes it; a confirmed action is carried out before
        returning."""
        llm = self.load()
        with self._lock:
            if len(self._history) >= TRIM_AT:
                self._history = _trim(self._history, TRIM_TO)
            self._history.append({"role": "user", "content": text})
            gate = _Gate(self.pending, on_text)
            raw = ""
            for chunk in llm.create_chat_completion(
                self._messages(self._history),
                max_tokens=MAX_REPLY_TOKENS,
                temperature=0.6,
                stream=True,
            ):
                piece = chunk["choices"][0]["delta"].get("content") or ""
                if piece:
                    raw += piece
                    gate.feed(raw)
            gate.finish(raw)
            return self._decide(gate)

    # ------------------------------------------------------------ internals

    def _decide(self, gate: "_Gate") -> Turn:
        # History records what Loopy actually said and did, not what the
        # model wanted to: otherwise, after a blocked "[do: ...]", the model
        # would believe it had already done it.
        if gate.kind == "do" and gate.allowed_to_run:
            self.pending = None
            self._remember(f"do: {gate.action}", gate.spoken)
            result = actions.run(gate.action)
            self._history.append({"role": "system", "content": f"Action done: {gate.action}. Result: {result.message}"})
            return Turn(gate.spoken, gate.action, result)

        if gate.kind in ("ask", "do"):
            # Asking (or claiming a yes Loopy never asked for): make sure it
            # really asks, and remember the question for next turn.
            self.pending = gate.action
            if not gate.spoken.rstrip().endswith("?"):
                gate.say(f"Do you want me to {gate.action}?")
            self._remember(f"ask: {gate.action}", gate.spoken)
            return Turn(gate.spoken)

        self.pending = None
        self._remember("none", gate.spoken)
        return Turn(gate.spoken)

    def _remember(self, tag: str, spoken: str) -> None:
        self._history.append({"role": "assistant", "content": f"[{tag}] {spoken}"})

    def _messages(self, history: list[dict]) -> list[dict]:
        return [{"role": "system", "content": _system_prompt()}, *history]


class _Gate:
    """Reads the tag at the start of the reply, decides whether the words
    after it may be spoken as-is, then passes them on as they arrive."""

    _TAG = re.compile(r"^\s*\[\s*(none|ask|do)\s*(?::\s*([^\]]*?))?\s*\]\s*", re.IGNORECASE)
    MAX_TAG_CHARS = 40   # no closing "]" by then: it's not a tag

    def __init__(self, pending: str | None, on_text: TextOut):
        self._pending = pending
        self._on_text = on_text
        self.kind = "none"
        self.action: str | None = None
        self.allowed_to_run = False
        self.spoken = ""
        self._body_start: int | None = None
        self._pos = 0
        self._suppress = False

    def feed(self, raw: str) -> None:
        if self._body_start is None and not self._read_tag(raw):
            return
        piece = raw[self._pos:]
        self._pos = len(raw)
        if piece and not self._suppress:
            piece = piece.replace("\r", " ").replace("\n", " ")
            if not self.spoken:
                piece = piece.lstrip()
            if piece:
                self.spoken += piece
                self._on_text(piece)

    def finish(self, raw: str) -> None:
        if self._body_start is None:  # never found a tag: treat it all as chat
            self._body_start = self._pos = 0
            self.feed(raw)
        self.spoken = self.spoken.strip()

    def say(self, text: str) -> None:
        """Add words of Loopy's own (not from the model)."""
        piece = f" {text}" if self.spoken else text
        self.spoken += piece
        self._on_text(piece)

    def _read_tag(self, raw: str) -> bool:
        m = self._TAG.match(raw)
        if m is None:
            stripped = raw.lstrip()
            if stripped and not stripped.startswith("["):
                self._body_start = self._pos = 0  # no tag at all: plain chat
                return True
            if "]" in raw:
                self._body_start = self._pos = raw.index("]") + 1  # unknown tag: skip it
                return True
            if len(raw) > self.MAX_TAG_CHARS:
                self._body_start = self._pos = 0
                return True
            return False  # tag still arriving
        kind, action = m.group(1).lower(), (m.group(2) or "").strip().lower()
        if kind != "none" and action in actions.names():
            self.kind, self.action = kind, action
            if kind == "do":
                if action == self._pending:
                    self.allowed_to_run = True
                else:
                    # About to say "Okay, doing X" without having asked: don't.
                    self._suppress = True
                    self.say(f"Do you want me to {action}?")
        self._body_start = self._pos = m.end()
        return True


def _trim(history: list[dict], keep: int) -> list[dict]:
    """Keep the last `keep` messages, starting at a user turn."""
    kept = history[-keep:]
    while kept and kept[0]["role"] != "user":
        kept.pop(0)
    return kept
