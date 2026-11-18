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

Tic-tac-toe: the model plays its own moves, the code is the referee.
[game: start] opens the board, [move: N] is the user's X, and when it's
Loopy's turn the model is shown the board and answers [play: N]. An illegal
move is rejected (its comment is never spoken), the model is told why, and it
picks again; after MAX_PLAY_TRIES misses Loopy takes a random free cell.
"""

import random
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from loopy import actions, checks
from loopy.game import CELL_NAMES, TicTacToe

N_CTX = 4096           # tokens the model can see: prompt + examples + conversation
N_THREADS = 4          # physical cores
MAX_REPLY_TOKENS = 120
# Dropping old messages makes llama.cpp re-read the whole conversation (slow
# on a laptop CPU), so trim rarely, and then keep only the most recent part.
HISTORY_KEEP = 900     # tokens of recent conversation kept after a trim
MAX_CHECK_ROUNDS = 3   # fact lookups per question (one fact at a time is fine)
MAX_PLAY_TRIES = 3     # illegal tic-tac-toe moves before Loopy picks at random
EMPTY_RETRIES = 2      # a reply with no words is asked for again this many times
TEMPERATURE = 0.6      # randomness of replies; tests set 0 to make them repeatable

TextOut = Callable[[str], None]   # receives reply text piece by piece
GameEvent = Callable[..., None]   # ("start", loopy_first) ("place", cell, mark) ("thinking",)
                                  # ("illegal", cell, why) ("your_turn",) ("over", winner, line, scores) ("stop",)


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def abilities() -> str:
    """Everything Loopy can do, as one speakable phrase. Built from the real
    action and check lists, so it never goes out of date."""
    apps = _and([a.name for a in actions.APPS])
    return (f"chat with you about anything, play tic-tac-toe, open or close {apps}, and tell you "
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

You can play tic-tac-toe with the user on a board that appears on screen (no permission needed).
Cells are numbered 1 to 9: 1 top-left, 2 top-middle, 3 top-right, 4 middle-left, 5 center, 6 middle-right,
7 bottom-left, 8 bottom-middle, 9 bottom-right. The user is X, you are O. When you name a cell in a comment, use these names.
Try to win: get three O's in a row, and block the user when they have two X's in a row.

Start every reply with one tag, then your words:
[none] for conversation, like "[none] I'm great, thanks!". A [none] reply can't do anything, so never say you did something in one.
[ask: close chrome] (or another action from the list) whenever the user wants one of your actions, even if they insist you skip asking. Always ask a yes/no question first.
[do: close chrome] (the same action) only when the user just said yes to your question about it.
[check: time] (or another fact; several are fine: [check: time, battery]) to look up a fact, with nothing after it. You'll get the result, then answer with [none].
[game: start] when the user wants to play tic-tac-toe or a rematch.
[game: stop] when they want to stop playing, close the game or the board, or say no to a rematch.
While a tic-tac-toe board is open, "close it" or "close the game" means [game: stop], never Chrome.
[move: 5] (the cell number) when the user says where their X goes ("put a cross in 2" is [move: 2], "the middle" is [move: 5]), with nothing after it.
[play: 3] (a free cell number) for your own move, only when the system says it's your turn, then a short playful comment.
Lookup results come only from the system. Never write a result, a "system:" line or the user's words yourself.
Every [none] reply must have words after the tag.
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
        # A short tic-tac-toe game, including a rejected move.
        u("i'm ready to play tic tac toe"), a("[game: start] Yay, game time!"),
        {"role": "system", "content": "Tic-tac-toe game started. The user is X, you are O. The user goes first."},
        u("put a cross in 2"), a("[move: 2]"),
        {"role": "system", "content": _board_message(_example_board("-X-------"), "")},
        a("[play: 5] I'm taking the middle!"),
        u("the top left corner"), a("[move: 1]"),
        {"role": "system", "content": _board_message(_example_board("XX--O----"), "")},
        a("[play: 2] Mine now!"),
        {"role": "system", "content": _board_message(_example_board("XX--O----"),
                                                     "Your move 2 isn't allowed: cell 2 is taken by X.")},
        a("[play: 3] Oops! Then 3, blocked you!"),
        u("let's stop playing"), a("[game: stop] Okay, good game!"),
        # After a game: declining the rematch, or "close", closes the board (not Chrome).
        u("let's play tic tac toe"), a("[game: start] Yay!"),
        {"role": "system", "content": "Tic-tac-toe game started. The user is X, you are O. The user goes first."},
        u("put a cross in 7"), a("[move: 7]"),
        {"role": "system", "content": "Game over: the user won. Score: user 1, Loopy 0, draws 0. "
                                      "The board is still open. For a rematch reply with [game: start]; "
                                      "if they say no, or want to close the game, reply with [game: stop]."},
        u("no, close it"), a("[game: stop] Okay! That was fun, good game."),
        {"role": "system", "content": "The examples are over. A new conversation with the user starts now."},
    ]


def _example_board(layout: str) -> TicTacToe:
    """'XX--O----' -> a board with X in 1 and 2, O in 5."""
    game = TicTacToe()
    game.cells = [None if c == "-" else c for c in layout]
    return game


def _board_message(game: TicTacToe, problem: str) -> str:
    """What the model is shown when it's Loopy's turn."""
    free = ", ".join(map(str, game.free()))
    return (f"{problem + ' ' if problem else ''}Tic-tac-toe board (you are O, the user is X):\n"
            f"{game.describe()}\n"
            f"It's your turn. Free cells: {free}. Pick one of them: reply with the play tag and its number, then a short playful comment.")


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
        self._prompt_tokens: int | None = None
        # Tic-tac-toe
        self.game: TicTacToe | None = None
        self.scores = {"you": 0, "draws": 0, "loopy": 0}
        self._loopy_starts_next = False     # players take turns going first
        # Per-turn plumbing, set by respond()
        self._said: list[str] = []
        self._on_text: TextOut = lambda _p: None
        self._on_event: GameEvent = lambda *_e: None

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
        """Forget the conversation (Loopy does this when it falls asleep).
        Any game in progress ends; the score is kept."""
        with self._lock:
            self._history.clear()
            self.pending = None
            self.game = None

    def respond(self, text: str, on_text: TextOut, on_event: GameEvent | None = None) -> Turn:
        """Reply to `text`. The reply is passed to `on_text` piece by piece
        as the model writes it; a confirmed action is carried out before
        returning. Tic-tac-toe happenings are reported to `on_event` as they
        happen, so a board can show them live."""
        llm = self.load()
        with self._lock:
            self._said = []
            self._on_text = on_text
            self._on_event = on_event or (lambda *_e: None)
            on_text = self._out  # everything spoken this turn goes through _out
            self._trim(llm)
            self._history.append({"role": "user", "content": text})
            gate = self._generate_with_words(llm, on_text)

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
                gate = self._generate_with_words(llm, on_text)
            if gate.kind == "check":  # still asking instead of answering: say the facts ourselves
                gate = _Gate(self.pending, on_text)
                gate.say("Here's what I found: " + "; ".join(f"{n}, {v}" for n, v in checked) + ".")
            elif checked and not gate.spoken:
                # Looked something up, then said nothing: say the facts ourselves.
                gate.say("Here's what I found: " + "; ".join(f"{n}, {v}" for n, v in checked) + ".")
            elif checked:
                # It looked something up but answered vaguely ("pretty busy"):
                # add the real numbers it left out.
                for name, value in checked:
                    numbers = re.findall(r"\d+(?:\.\d+)?", value)
                    if numbers and not any(re.search(rf"(?<![\d.]){re.escape(n)}(?![\d.])", gate.spoken) for n in numbers):
                        gate.say(f"To be exact: {value}.")

            if gate.kind in ("none", "ask", "do") and not gate.spoken and not checked:
                # Still nothing after every retry: Loopy must never go silent.
                gate.say("Sorry, I lost my train of thought. Could you say that again?")

            if gate.kind in ("game", "move"):
                self.pending = None
                self._game_turn(llm, gate)
                turn = Turn("")
            else:
                if self.game is not None and self.game.over:
                    # The game ended and this isn't a rematch ("no thanks",
                    # or just chatting on): close the leftover board.
                    self.game = None
                    self._on_event("stop")
                turn = self._decide(gate)
            turn.reply = "".join(self._said).strip()
            turn.checked = checked
            return turn

    # ---------------------------------------------------------- tic-tac-toe

    def _game_turn(self, llm, gate: "_Gate") -> None:
        if gate.kind == "game":
            if gate.game_cmd == "stop":
                if not gate.spoken:
                    gate.say("Okay, good game!")
                self._remember("game: stop", gate.spoken)
                if self.game is not None:
                    self.game = None
                    self._on_event("stop")
                return
            self._remember("game: start", gate.spoken)
            self._start_game(llm, announce=True)
            return

        # The user's move.
        cell = gate.cell
        self._history.append({"role": "assistant", "content": f"[move: {cell}]"})
        if self.game is None or self.game.over:
            self._start_game(llm, announce=False, user_first=True)
        if not self.game.is_free(cell):
            why = self.game.why_not(cell)
            self._say(f"Oops, {why}! You can pick {_and([str(c) for c in self.game.free()])}.")
            self._history.append({"role": "system", "content": f"The user's move didn't count: {why}."})
            return
        self.game.place(cell, "X")
        self._on_event("place", cell, "X")
        if self.game.over:
            self._game_over()
        else:
            self._loopy_turn(llm)

    def _start_game(self, llm, announce: bool, user_first: bool = False) -> None:
        loopy_first = self._loopy_starts_next and not user_first
        self._loopy_starts_next = not loopy_first
        self.game = TicTacToe()
        self._on_event("start", loopy_first)
        if announce:
            self._say("You're X and I'm O. " + ("I'll go first!" if loopy_first
                                                 else "You go first, just say a number from 1 to 9."))
        self._history.append({"role": "system", "content":
                              "Tic-tac-toe game started. The user is X, you are O. "
                              + ("You go first." if loopy_first else "The user goes first.")})
        if loopy_first:
            self._loopy_turn(llm)

    def _loopy_turn(self, llm) -> None:
        """The model picks Loopy's move; the code checks it and, if it's
        illegal, says why and lets the model pick again."""
        game = self.game
        self._on_event("thinking")
        problem = ""
        for _ in range(MAX_PLAY_TRIES):
            self._history.append({"role": "system", "content": _board_message(game, problem)})
            gate = self._generate_with_words(llm, self._out, validate_play=game.is_free)
            cell = gate.cell if gate.kind == "play" else None
            if game.is_free(cell):
                if not gate.spoken:  # still silent after the retries: say where it went
                    gate.say(f"I'll take {cell}, {CELL_NAMES[cell]}.")
                self._remember(f"play: {cell}", gate.spoken)
                game.place(cell, "O")
                self._on_event("place", cell, "O")
                break
            # Illegal (or no move at all): its comment was never spoken.
            self._history.append({"role": "assistant", "content": f"[play: {cell}]" if cell else f"[none] {gate.spoken}"})
            why = game.why_not(cell) if cell else "you didn't pick a cell"
            self._on_event("illegal", cell, why)
            problem = f"Your move {cell} isn't allowed: {why}." if cell else "You must pick a free cell."
        else:
            cell = random.choice(game.free())
            self._say(f"Hmm, I'll just take {cell}.")
            self._remember(f"play: {cell}", f"Hmm, I'll just take {cell}.")
            game.place(cell, "O")
            self._on_event("place", cell, "O")

        if game.over:
            self._game_over()
        else:
            self._on_event("your_turn")

    def _game_over(self) -> None:
        winner = self.game.winner
        if winner == "X":
            self.scores["you"] += 1
            self._say("Three in a row, you win! Want a rematch?")
            outcome = "the user won"
        elif winner == "O":
            self.scores["loopy"] += 1
            self._say("Three in a row, I win! Want a rematch?")
            outcome = "you (Loopy) won"
        else:
            self.scores["draws"] += 1
            self._say("It's a draw! One more game?")
            outcome = "it's a draw"
        s = self.scores
        self._history.append({"role": "system", "content":
                              f"Game over: {outcome}. Score: user {s['you']}, Loopy {s['loopy']}, draws {s['draws']}. "
                              "The board is still open. For a rematch reply with [game: start]; "
                              "if they say no, or want to close the game, reply with [game: stop]."})
        self._on_event("over", winner, self.game.line, dict(s))

    def _out(self, piece: str) -> None:
        self._said.append(piece)
        self._on_text(piece)

    def _say(self, text: str) -> None:
        """Words of Loopy's own (not from the model), added to this turn."""
        self._out(f" {text}" if "".join(self._said).strip() else text)

    # ------------------------------------------------------------ internals

    def _trim(self, llm) -> None:
        """Keep the conversation inside the model's context window. Trimming
        means re-reading everything that follows (slow), so it only happens
        when needed, and then cuts back to the most recent HISTORY_KEEP tokens."""
        def size(message: dict) -> int:
            return len(llm.tokenize(message["content"].encode("utf-8"), add_bos=False)) + 5

        if self._prompt_tokens is None:
            self._prompt_tokens = sum(size(m) for m in self._messages([]))
        budget = N_CTX - self._prompt_tokens - MAX_REPLY_TOKENS - 300  # room for this turn
        if sum(size(m) for m in self._history) <= budget:
            return
        kept, total = [], 0
        for message in reversed(self._history):
            total += size(message)
            if total > min(HISTORY_KEEP, budget):
                break
            kept.append(message)
        kept.reverse()
        while kept and kept[0]["role"] != "user":
            kept.pop(0)
        self._history = kept

    def _generate_with_words(self, llm, on_text: TextOut,
                             validate_play: Callable[[int | None], bool] | None = None) -> "_Gate":
        """Like _generate, but a reply that should have words and has none
        (e.g. only "[ask: ACTION]", or a move with no comment) is asked for
        again, up to EMPTY_RETRIES more times."""
        for attempt in range(1 + EMPTY_RETRIES):
            gate = self._generate(llm, on_text, validate_play)
            wordless = gate.kind in ("check", "move", "game")      # these have nothing after the tag
            illegal = gate.kind == "play" and not (validate_play and validate_play(gate.cell))
            if gate.spoken or wordless or illegal or attempt == EMPTY_RETRIES:
                return gate
            if gate.kind == "play":  # a fine move, but no comment: say exactly what's missing
                note = (f"You picked {gate.cell} but said nothing. Reply again with [play: {gate.cell}] "
                        "and a short playful comment, like \"[play: 5] I'm taking the middle!\".")
            else:
                note = "Your last reply had no words. Reply again: the tag, then what you want to say."
            self._history.append({"role": "system", "content": note})
        return gate

    def _generate(self, llm, on_text: TextOut, validate_play: Callable[[int | None], bool] | None = None) -> "_Gate":
        """One model reply, streamed through a fresh gate. Stops early when
        nothing more may be spoken (a [check], a [move], an illegal [play])."""
        first = [True]

        def spaced(piece: str) -> None:
            # This reply may follow words already said this turn ("I'll go
            # first!" + "I'll take the middle!"): keep a space between them.
            if first[0] and self._said and not "".join(self._said).endswith(" ") and not piece[:1].isspace():
                piece = " " + piece
            first[0] = False
            on_text(piece)

        gate = _Gate(self.pending, spaced, validate_play)
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
                    if gate.halt:
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

    _TAG = re.compile(r"^\s*\[\s*(none|ask|do|check|game|move|play)\s*(?::\s*([^\]]*?))?\s*\]\s*", re.IGNORECASE)
    MAX_TAG_CHARS = 60   # no closing "]" by then: it's not a tag

    def __init__(self, pending: str | None, on_text: TextOut,
                 validate_play: Callable[[int | None], bool] | None = None):
        self._pending = pending
        self._on_text = on_text
        self._validate_play = validate_play  # set only when it's Loopy's turn to play
        self.kind = "none"
        self.action: str | None = None
        self.checks: list[str] = []
        self.cell: int | None = None         # [move: N] / [play: N]
        self.game_cmd: str | None = None     # [game: start|stop]
        self._held = ""          # possible mid-reply "[tag]" not yet closed
        self.allowed_to_run = False
        self.spoken = ""
        self._body_start: int | None = None
        self._pos = 0
        self._suppress = False
        self.stopped = False     # the model started writing a fake turn; ignore the rest

    @property
    def halt(self) -> bool:
        """Nothing more may be spoken, so stop generating."""
        return (self.stopped or self.kind in ("check", "move")
                or (self.kind == "play" and self._suppress))

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

    _TAG_START = re.compile(r"\[\s*(none|ask|do|check|game|move|play)\b", re.IGNORECASE)
    _TAG_WORDS = ("none", "ask", "do", "check", "game", "move", "play")
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
        elif kind == "game":
            self.kind = "game"
            self.game_cmd = "stop" if re.search(r"stop|end|quit|exit", action) else "start"
        elif kind in ("move", "play"):
            number = re.search(r"\d+", action)
            cell = int(number.group()) if number else None
            if kind == "move":
                if cell is not None and 1 <= cell <= 9:
                    self.kind, self.cell = "move", cell
                    self._suppress = True  # the reply comes after Loopy's own move
            elif self._validate_play is not None:  # only counts on Loopy's turn
                self.kind, self.cell = "play", cell
                if not self._validate_play(cell):
                    self._suppress = True  # illegal: its comment must not be spoken
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
