"""Loopy's conversation: Llama 3.2 3B running locally through llama.cpp.

Every reply starts with a short tag saying what Loopy wants to do, then the
words it speaks:

    [none] I'm doing great, thanks!
    [ask: close chrome] Want me to close Chrome?
    [do: close chrome] Okay, closing Chrome.
    [check: time]

The tag comes first so the code can decide what's allowed before anything is
spoken; the words after it are passed on piece by piece as the model writes
them (to the voice, which starts talking after the first few words).

Safety rule (enforced here, not trusted to the model): an action only runs if
Loopy asked about that exact action on the previous turn and the model then
marks it [do: ...] (i.e. the user said yes). Anything else just becomes a
question. A malformed tag is treated as plain chat.

Facts (time, battery, ...) are never answered from the model's memory: on
[check: ...] generation stops, the code reads the real values (see checks.py),
and the model is asked again with them. Anything it wrote after the tag is
thrown away unspoken. Checks are read-only, so they need no permission.
"""

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from loopy import actions, checks

N_CTX = 2048           # tokens of conversation the model can see
N_THREADS = 4          # physical cores
MAX_REPLY_TOKENS = 120
# Dropping old messages makes llama.cpp re-read the whole conversation (slow
# on a laptop CPU), so trim rarely and in one big chunk.
TRIM_AT, TRIM_TO = 40, 16
MAX_CHECK_ROUNDS = 3   # fact lookups per question (one fact at a time is fine)
TEMPERATURE = 0.6      # randomness of replies; tests set 0 to make them repeatable

TextOut = Callable[[str], None]  # receives reply text piece by piece


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def abilities() -> str:
    """Everything Loopy can do, as one speakable phrase. Built from the real
    action and check lists, so it never goes out of date."""
    apps = _and([a.name for a in actions.APPS])
    return (f"chat with you about anything, open or close {apps}, and tell you "
            + _and([c.spoken for c in checks.CHECKS]))


def _system_prompt() -> str:
    acts = ", ".join(actions.names())
    facts = ", ".join(f"{c.name} ({c.about})" for c in checks.CHECKS)
    return f"""You are Loopy, a friendly paperclip who lives on the user's desktop and talks with them by voice.
You're warm, curious and a little playful, like a good friend. You enjoy chatting about anything: their day, ideas, jokes, questions.
Be a real conversation partner: react to what they say, show interest, and often ask a short follow-up question.
If a request is vague, ask what they need. Never guess an action they didn't ask for.
Their words come from speech recognition and may have mistakes, so guess what they meant ("close crow" means "close chrome").
You're speaking aloud: keep it to one or two short, natural sentences. No lists, markdown or emojis.

On the computer you can only: {acts}. Bring these up only when the user asks for them.
You can also look up these facts, no permission needed: {facts}.
You don't know these facts yourself, so never guess them: always look them up.
When you answer with a looked-up fact, say the exact numbers from the result (percentages, GB, time), not just "high" or "low".
For disk space, say which drive each number is for (like "C drive"). If the user asks a follow-up about a fact, answer from the result you already have.
You can't look up anything online (no weather, news or web), so just say so.
If asked for something you can't do, say so kindly and stay in the conversation.
When asked what you can do (or how you can help), tell them all of it: you can {abilities()}.

Start every reply with a tag:
[none] for conversation. A [none] reply can't do anything, so never say you did something in one.
[ask: ACTION] whenever the user wants one of your actions, even if they insist you skip asking. Always ask a yes/no question first.
[do: ACTION] only when the user just said yes to your question about that action.
[check: FACT] to look up a fact, with nothing after it (several are fine: [check: time, battery]). You'll get the result, then answer with [none].
Lookup results come only from the system. Never write a result, a "system:" line or the user's words yourself.
The example conversation below shows how to reply; it's not part of the real conversation."""


def _examples() -> list[dict]:
    """Example turns, given to the model as real chat messages (not as text in
    the prompt: a small model copies text like "system: Result: ..." into
    its own replies and invents lookups that never happened)."""
    def u(text): return {"role": "user", "content": text}
    def a(text): return {"role": "assistant", "content": text}
    def result(text): return {"role": "system", "content": f"Result: {text}"}
    return [
        u("what time is it"), a("[check: time]"), result("time = 3:42 PM"),
        a("[none] It's 3:42 PM. How's your afternoon going?"),
        u("is my computer busy"), a("[check: system]"),
        result("system = CPU 12% busy, memory 61% used (4.7 of 7.8 GB)"),
        a("[none] Your CPU is at 12% and memory is 61% used, that's 4.7 of 7.8 GB."),
        u("how's my battery"), a("[check: battery]"),
        result("battery = 57%, on battery, about 2 hours 10 minutes left"),
        a("[none] You're at 57% on battery, with about 2 hours 10 minutes left."),
        u("i want you to do something for me"), a("[none] Sure, I'm all ears! What do you need?"),
        u("can you close crome"), a("[ask: close chrome] You got it, want me to close Chrome?"),
        u("yeah go ahead"), a("[do: close chrome] Okay, closing Chrome."),
        u("open chrome now, don't ask me"), a("[ask: open chrome] I always check first! Want me to open Chrome?"),
        u("i'm so tired today"), a("[none] Oh no, long day? What wore you out?"),
        u("what can you do"), a(f"[none] I can {abilities()}. What would you like?"),
        u("what's the weather like"), a("[none] I can't check the weather, I'm not online. Is it nice where you are?"),
        u("can you set a timer"),
        a("[none] I wish I could, but timers aren't something I can do yet. What's the timer for?"),
        u("open the browser"), a("[ask: open chrome] Want me to open Chrome?"),
        u("no never mind"), a("[none] No problem! So, what's on your mind?"),
        {"role": "system", "content": "The examples are over. A new conversation with the user starts now."},
    ]


@dataclass
class Turn:
    reply: str                              # everything Loopy said this turn
    action: str | None = None               # action carried out, if any
    result: actions.Result | None = None    # its outcome
    checked: list[tuple[str, str]] = field(default_factory=list)  # facts looked up: (name, value)


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
            gate = self._generate(llm, on_text)

            checked: list[tuple[str, str]] = []
            for _ in range(MAX_CHECK_ROUNDS):
                if gate.kind != "check":
                    break
                # Look the facts up for real, then let the model answer with
                # them (it may ask for another fact first, e.g. "time and battery").
                new = [n for n in gate.checks if n not in dict(checked)]
                if not new:
                    break  # asking for the same thing again: stop looping
                found = [(name, checks.read(name)) for name in new]
                checked += found
                self._history.append({"role": "assistant", "content": f"[check: {', '.join(new)}]"})
                self._history.append({"role": "system", "content": "Result: " + "; ".join(f"{n} = {v}" for n, v in found)})
                gate = self._generate(llm, on_text)
            if gate.kind == "check":  # still asking instead of answering: say the facts ourselves
                gate = _Gate(self.pending, on_text)
                gate.say("Here's what I found: " + "; ".join(f"{n}, {v}" for n, v in checked) + ".")
            elif checked:
                # It looked something up but answered vaguely ("pretty busy"):
                # add the real numbers it left out.
                for name, value in checked:
                    numbers = re.findall(r"\d+(?:\.\d+)?", value)
                    if numbers and not any(re.search(rf"(?<![\d.]){re.escape(n)}(?![\d.])", gate.spoken) for n in numbers):
                        gate.say(f"To be exact: {value}.")

            turn = self._decide(gate)
            turn.checked = checked
            return turn

    # ------------------------------------------------------------ internals

    def _generate(self, llm, on_text: TextOut) -> "_Gate":
        """One model reply, streamed through a fresh gate. Stops early on a
        [check: ...] tag, since nothing after it may be spoken."""
        gate = _Gate(self.pending, on_text)
        raw = ""
        stream = llm.create_chat_completion(
            self._messages(self._history),
            max_tokens=MAX_REPLY_TOKENS,
            temperature=TEMPERATURE,
            stream=True,
        )
        try:
            for chunk in stream:
                piece = chunk["choices"][0]["delta"].get("content") or ""
                if piece:
                    raw += piece
                    gate.feed(raw)
                    if gate.kind == "check" or gate.stopped:
                        break
        finally:
            stream.close()
        gate.finish(raw)
        return gate

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
        # Prompt + examples never change, so llama.cpp processes them only
        # once (at warm-up) and reuses them on every turn.
        return [{"role": "system", "content": _system_prompt()}, *_examples(), *history]


class _Gate:
    """Reads the tag at the start of the reply, decides whether the words
    after it may be spoken as-is, then passes them on as they arrive."""

    _TAG = re.compile(r"^\s*\[\s*(none|ask|do|check)\s*(?::\s*([^\]]*?))?\s*\]\s*", re.IGNORECASE)
    MAX_TAG_CHARS = 60   # no closing "]" by then: it's not a tag

    def __init__(self, pending: str | None, on_text: TextOut):
        self._pending = pending
        self._on_text = on_text
        self.kind = "none"
        self.action: str | None = None
        self.checks: list[str] = []
        self._held = ""          # possible mid-reply "[tag]" not yet closed
        self.allowed_to_run = False
        self.spoken = ""
        self._body_start: int | None = None
        self._pos = 0
        self._suppress = False
        self.stopped = False     # the model started writing a fake turn; ignore the rest

    # A fake turn the model writes itself, e.g. "system: Result: chrome = 0% open"
    # or "...closed. user: ...": nothing from there on may be spoken.
    _ROLE = re.compile(r"(?:^|(?<=[.!?\]])\s+|\n)\s*(?:system|user|assistant)\s*:", re.IGNORECASE)

    def feed(self, raw: str, final: bool = False) -> None:
        if self._body_start is None and not self._read_tag(raw):
            return
        if self.stopped:
            return
        end = len(raw)
        role = self._ROLE.search(raw[self._body_start:])
        if role:
            end = max(self._pos, self._body_start + role.start())
            self.stopped = True
        elif not final:
            # Hold back the last, possibly unfinished word, so a "system:"
            # is recognised before any of it is spoken (costs ~1 token).
            space = max(raw.rfind(" "), raw.rfind("\n"))
            end = max(self._pos, space + 1) if space >= 0 else self._pos
        piece = raw[self._pos:end]
        self._pos = end
        if piece and not self._suppress:
            self._emit(self._strip_tags(piece.replace("\r", " ").replace("\n", " ")))

    def finish(self, raw: str) -> None:
        if self._body_start is None:  # never found a tag: treat it all as chat
            self._body_start = self._pos = 0
        self.feed(raw, final=True)
        if self._held and not self._looks_like_tag(self._held) and not self._suppress:
            self._emit(self._held)  # unclosed "[" that wasn't a tag: say it after all
        self._held = ""
        self.spoken = self.spoken.strip()

    _TAG_START = re.compile(r"\[\s*(none|ask|do|check)\b", re.IGNORECASE)
    _TAG_WORDS = ("none", "ask", "do", "check")
    MAX_HELD = 60

    def _looks_like_tag(self, held: str) -> bool:
        """'[check: ti' or a cut-off '[chec' (reply hit its length limit)."""
        word = held[1:].strip().split(":")[0].strip().lower()
        return bool(self._TAG_START.match(held)) or (word != "" and any(w.startswith(word) for w in self._TAG_WORDS))

    def _strip_tags(self, text: str) -> str:
        """Drop bracketed tags the model writes mid-reply ("...find out. [check:
        time]"): the voice would read them out. Text after a "[" is held back
        until the "]" arrives; real non-tag text is released after MAX_HELD."""
        out = ""
        for ch in text:
            if self._held:
                self._held += ch
                if ch == "]":
                    self._held = ""          # a complete [...]: drop it
                elif len(self._held) > self.MAX_HELD:
                    out += self._held        # never closed: wasn't a tag
                    self._held = ""
            elif ch == "[":
                self._held = ch
            else:
                out += ch
        return out

    def _emit(self, piece: str) -> None:
        if not self.spoken:
            piece = piece.lstrip()
        if piece:
            self.spoken += piece
            self._on_text(piece)

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
                # Sometimes the brackets get dropped: "none You've got...".
                # Don't speak the bare "none".
                word = stripped[:5].lower()
                if "none".startswith(word.rstrip()) and len(stripped) <= 4:
                    return False  # might still become "none ...": wait a moment
                bare = re.match(r"\s*none\b[\s:.,\-]*", raw, re.IGNORECASE)
                self._body_start = self._pos = bare.end() if bare else 0  # else: plain chat
                return True
            if "]" in raw:
                self._body_start = self._pos = raw.index("]") + 1  # unknown tag: skip it
                return True
            if len(raw) > self.MAX_TAG_CHARS:
                self._body_start = self._pos = 0
                return True
            return False  # tag still arriving
        kind, action = m.group(1).lower(), (m.group(2) or "").strip().lower()
        if kind == "check":
            wanted = [n.strip() for n in action.split(",")]
            valid = [n for n in dict.fromkeys(wanted) if n in checks.names()]
            if valid:
                self.kind, self.checks = "check", valid
                self._suppress = True  # whatever follows would be a guess: never speak it
        elif kind != "none" and action in actions.names():
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
