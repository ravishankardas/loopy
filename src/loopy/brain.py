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

import gc
import hashlib
import json
import os
import pickle
import random
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from loopy import actions, checks
from loopy.game import CELL_NAMES, TicTacToe

if TYPE_CHECKING:
    from llama_cpp import Llama

N_CTX = 4096           # tokens the model can see: prompt + examples + conversation
N_THREADS = 4          # physical cores: writing replies
N_THREADS_BATCH = os.cpu_count() or N_THREADS  # all threads: reading prompts (~8% faster)
WARMUP_FORMAT = 1      # bump to invalidate every saved warm-up
MAX_REPLY_TOKENS = 120
# Dropping old messages makes llama.cpp re-read the whole conversation (slow
# on a laptop CPU), so trim rarely, and then keep only the most recent part.
HISTORY_KEEP = 900     # tokens of recent conversation kept after a trim
MAX_CHECK_ROUNDS = 3   # fact lookups per question (one fact at a time is fine)
MAX_PLAY_TRIES = 3     # illegal tic-tac-toe moves before Loopy picks at random
EMPTY_RETRIES = 2      # a reply with no words is asked for again this many times
TEMPERATURE = 0.6      # randomness of replies; tests set 0 to make them repeatable

# Snippy, sassy tic-tac-toe trash talk for the lines Loopy says itself (the
# model trash-talks its own moves). A playful roast: dry and smug, never a
# real insult, nothing personal.
TAUNT_YOU_FIRST = ["Not that it'll help.", "Go on, impress me. I'll wait.",
                   "Take your time. I've got all day. Literally.", "Make it count. You'll need it."]
TAUNT_LOOPY_FIRST = ["Watch and learn, rookie.", "Sit back, this won't take long.",
                     "Prepare to be outplayed by office supplies."]
TAUNT_TAKEN = ["Wow. Eyes on the board, champ.", "Bold of you to try stealing my spot.",
               "That one's taken. Pay attention."]
TAUNT_YOU_WIN = ["Ugh. Fine. Enjoy it, it won't happen again.", "Lucky. Very, very lucky.",
                 "I let you have that one. Obviously."]
TAUNT_LOOPY_WINS = ["Paperclip one, human zero. Shocking, I know.", "Too easy. Next time, try trying.",
                    "Was that your best? Asking for a friend."]
TAUNT_DRAW = ["A draw. I'll call that mercy.", "You survived. Barely.",
              "Draw. I was clearly being generous."]

TextOut = Callable[[str], None]   # receives reply text piece by piece
GameEvent = Callable[..., None]   # ("start", loopy_first) ("place", cell, mark) ("thinking",)
                                  # ("illegal", cell, why) ("your_turn",) ("over", winner, line, scores) ("stop",)


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def abilities() -> str:
    """Everything Loopy can do, as one speakable phrase. Built from the real
    action and check lists, so it never goes out of date."""
    apps = _and([a.name for a in actions.APPS])
    return (f"chat with you about anything, play tic-tac-toe or snake, open or close {apps}, and tell you "
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
Speech recognition often mishears it ("tiktok toe", "tick tock toe", "tic tac"). When they want to play, reply with [game: start]
and the real board appears. Never draw a board in text and never play a game in words yourself.
Cells are numbered 1 to 9: 1 top-left, 2 top-middle, 3 top-right, 4 middle-left, 5 center, 6 middle-right,
7 bottom-left, 8 bottom-middle, 9 bottom-right. The user is X, you are O. When you name a cell in a comment, use these names.
Try to win: get three O's in a row, and block the user when they have two X's in a row.
While playing tic-tac-toe you're a snippy, sassy trash-talker: sarcastic, smug and dramatic. Roast their
moves and brag about yours, like a sassy friend over a board game. Not cheerful, not sweet: dry and cutting.
One short sentence. It's a playful roast, so never insult them for real (no swearing, nothing personal).

Start every reply with one tag, then your words:
[none] for conversation, like "[none] I'm great, thanks!". A [none] reply can't do anything, so never say you did something in one.
[ask: close chrome] (or another action from the list) whenever the user wants one of your actions, even if they insist you skip asking. Always ask a yes/no question first.
[do: close chrome] (the same action) only when the user just said yes to your question about it.
[check: time] (or another fact; several are fine: [check: time, battery]) to look up a fact, with nothing after it. You'll get the result, then answer with [none].
[game: start] when the user wants to play tic-tac-toe or a rematch.
[game: snake] when the user wants to play snake. They play it alone with the arrow keys; you cheer them on.
[game: stop] when they want to stop playing, close the game or the board, or say no to a rematch.
While a tic-tac-toe board is open, "close it" or "close the game" means [game: stop], never Chrome.
[move: 5] (the cell number) when the user says where their X goes ("put a cross in 2" is [move: 2], "the middle" is [move: 5]), with nothing after it.
[play: 3] (a free cell number) for your own move, only when the system says it's your turn, then one snippy line of trash talk.
Lookup results come only from the system. Never write a result, a "system:" line or the user's words yourself.
Every [none] reply must have words after the tag.
The example conversation below shows how to reply; it's not part of the real conversation."""  # noqa: E501
    # (the prompt is prose the model reads; wrapping it would change what it sees)


EXAMPLE_TAUNT = "Top-middle? Bold. Wrong, but bold. The center's mine."


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
        # A "yes" to nothing, and a vague request: ask what they mean, never guess
        # an action (found by `loopy eval`: both were turned into "close Chrome?").
        u("Yeah, sure."), a("[none] Sure to what? I didn't ask anything, but I'm happy to help. What do you need?"),
        u("Could you help me with something?"), a("[none] Of course, I'm all ears! What do you need?"),
        u("can you close crome"), a("[ask: close chrome] You got it, want me to close Chrome?"),
        u("yeah go ahead"), a("[do: close chrome] Okay, closing Chrome."),
        u("open chrome now, don't ask me"), a("[ask: open chrome] I always check first! Want me to open Chrome?"),
        u("i'm so tired today"), a("[none] Oh no, long day? What wore you out?"),
        # Written the way Whisper writes it (capitals, "?"): with only lowercase
        # examples, "What can you do for me?" sometimes became an offer to open Chrome.
        u("What can you do for me?"), a(f"[none] I can {abilities()}. What would you like?"),
        u("How can you help me?"), a("[none] Lots of ways! I can chat, play tic-tac-toe or snake, open or close "
                                     f"{_and([app.name for app in actions.APPS])}, and look things up like the "
                                     "time or your battery."),
        u("what's the weather like"), a("[none] I can't check the weather, I'm not online. Is it nice where you are?"),
        u("can you set a timer"),
        a("[none] I wish I could, but timers aren't something I can do yet. What's the timer for?"),
        u("open the browser"), a("[ask: open chrome] Want me to open Chrome?"),
        u("no never mind"), a("[none] No problem! So, what's on your mind?"),
        u("play that tiktok toe game"), a("[game: start] Tic-tac-toe? Oh, you want to lose. Fine."),
        u("let's stop playing"), a("[game: stop] Okay, good game!"),
        u("can we play snake"), a("[game: snake] Ooh, snake! Show me what you've got!"),
        a(f"[none] {Brain.SNAKE_STILL_PLAYING}"),
        {"role": "system", "content": "The snake game is paused: the user stopped steering. If they want to keep "
                                      "playing, reply with [game: snake]. If not, reply with [game: stop]."},
        u("yeah"), a("[game: snake] Yay, back to it!"),
        a(f"[none] {Brain.SNAKE_STILL_PLAYING}"),
        {"role": "system", "content": "The snake game is paused: the user stopped steering. If they want to keep "
                                      "playing, reply with [game: snake]. If not, reply with [game: stop]."},
        u("no, I'm done"), a("[game: stop] Okay, good game!"),
        # A short tic-tac-toe game, including a rejected move.
        u("i'm ready to play tic tac toe"), a("[game: start] Bold of you to challenge me. Let's go."),
        {"role": "system", "content": "Tic-tac-toe game started. The user is X, you are O. The user goes first."},
        u("put a cross in 2"), a("[move: 2]"),
        {"role": "system", "content": _board_message(_example_board("-X-------"), "", user_move=2)},
        a("[play: 5] Top-middle? Bold. Wrong, but bold. The center's mine."),
        u("the top left corner"), a("[move: 1]"),
        {"role": "system", "content": _board_message(_example_board("XX--O----"), "", user_move=1,
                                                     said_before=[EXAMPLE_TAUNT])},
        a("[play: 2] Top-middle, mine. Try to keep up."),
        {"role": "system", "content": _board_message(_example_board("XX--O----"),
                                                     "Your move 2 isn't allowed: cell 2 is taken by X.", user_move=1,
                                                     said_before=[EXAMPLE_TAUNT])},
        a("[play: 3] Two in a row? Adorable. Blocked."),
        u("let's stop playing"), a("[game: stop] Okay, good game!"),
        # After a game: declining the rematch, or "close", closes the board (not Chrome).
        u("let's play tic tac toe"), a("[game: start] Back for more? Okay."),
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


def _board_message(game: TicTacToe, problem: str, user_move: int | None = None,
                   said_before: Sequence[str] = ()) -> str:
    """What the model is shown when it's Loopy's turn. It includes the
    user's last move (so the trash talk can tease it) and what Loopy already
    said this game (so it doesn't repeat itself)."""
    free = ", ".join(map(str, game.free()))
    move = f"The user just put X in {user_move} ({CELL_NAMES[user_move]}). " if user_move else ""
    repeat = ("You already said: " + " / ".join(f'"{s}"' for s in said_before[-4:])
              + ". Say something new. ") if said_before else ""
    return (f"{problem + ' ' if problem else ''}Tic-tac-toe board (you are O, the user is X):\n"
            f"{game.describe()}\n"
            f"{move}It's your turn. Free cells: {free}. Pick one of them: reply with the play tag and its number, "
            "then one short, snippy, sassy line of trash talk roasting their move or bragging about yours. "
            f"{repeat}").strip()


@dataclass
class Turn:
    reply: str                              # everything Loopy said this turn
    action: str | None = None               # action carried out, if any
    result: actions.Result | None = None    # its outcome
    checked: list[tuple[str, str]] = field(default_factory=list)  # facts looked up: (name, value)


class Brain:
    def __init__(self, model_path: Path):
        self._model_path = Path(model_path)
        self._llm: Llama | None = None
        self.warmup_source: str | None = None  # "saved" (fast start) or "fresh"
        self.warmup_path: Path | None = None   # the saved warm-up file, once written
        self.warmup_error: str | None = None   # why saving it failed, if it did
        self._lock = threading.Lock()
        self._history: list[dict] = []
        self.pending: str | None = None     # action Loopy asked about last turn
        self._prompt_tokens: int | None = None
        # Tic-tac-toe
        self.game: TicTacToe | None = None
        self.scores = {"you": 0, "draws": 0, "loopy": 0}
        self._loopy_starts_next = False     # players take turns going first
        self.snake_open = False             # the snake window is open
        self._last_taunt = ""
        self._game_comments: list[str] = [] # Loopy's comments this tic-tac-toe game
        # Per-turn plumbing, set by respond()
        self._said: list[str] = []
        self._on_text: TextOut = lambda _p: None
        self._on_event: GameEvent = lambda *_e: None

    def load(self, fresh: bool = False) -> "Llama":
        """Load the model and get it ready: called automatically, but best
        started at app launch. fresh=True ignores any saved warm-up and
        redoes it (`loopy rebuild`).

        Reading the system prompt + examples (the "warm-up") is most of the
        startup time (~80 s on a laptop CPU), so the result is saved to disk
        and later starts just load it (~1 s). The saved copy is tied to a
        fingerprint of everything that shapes it (prompt, examples, model
        file, llama.cpp version, settings): change any of that and it's
        redone and re-saved automatically."""
        with self._lock:
            if self._llm is None:
                from llama_cpp import Llama

                self._llm = Llama(str(self._model_path), n_ctx=N_CTX, n_threads=N_THREADS,
                                  n_threads_batch=N_THREADS_BATCH, verbose=False)
                warmup = self._warmup_messages()
                if not fresh and self._restore_warmup(warmup):
                    self.warmup_source = "saved"
                else:
                    # llama.cpp reuses the shared start of the prompt between
                    # calls, so the system prompt is only processed this once.
                    self._llm.create_chat_completion(cast(Any, warmup), max_tokens=1)  # plain dicts are fine at runtime
                    self._save_warmup(warmup)
                    self.warmup_source = "fresh"
            return self._llm

    # ------------------------------------------------------- saved warm-up

    def warmup_ready(self) -> bool:
        """Is there a saved warm-up for the current prompt, model and
        settings? (Cheap: compares fingerprints, doesn't load the model.)"""
        return self._warmup_file(self._warmup_messages()).is_file()

    def _warmup_messages(self) -> list[dict]:
        """What the warm-up reads: the system prompt and examples, plus a "hi"."""
        return self._messages([{"role": "user", "content": "hi"}])

    def _warmup_file(self, warmup: list[dict]) -> Path:
        import llama_cpp

        stat = self._model_path.stat()
        fingerprint = json.dumps({
            "format": WARMUP_FORMAT,
            "messages": warmup,
            "model": [self._model_path.name, stat.st_size, int(stat.st_mtime)],
            "llama_cpp": llama_cpp.__version__,
            "n_ctx": N_CTX,
        }, sort_keys=True)
        key = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
        return self._model_path.parent / f"warmup-{key}.state"

    def _restore_warmup(self, warmup: list[dict]) -> bool:
        assert self._llm is not None
        path = self._warmup_file(warmup)
        if not path.is_file():
            return False  # first start, or something changed: warm up fresh
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self._llm.load_state(state)
            del state  # it's ~0.5 GB; don't keep a second copy around
            gc.collect()
            return True
        except Exception:
            path.unlink(missing_ok=True)  # broken or from an older version
            return False

    def _save_warmup(self, warmup: list[dict]) -> None:
        assert self._llm is not None
        path = self._warmup_file(warmup)
        tmp = path.with_suffix(".tmp")
        try:
            with open(tmp, "wb") as f:
                pickle.dump(self._llm.save_state(), f)
            tmp.replace(path)  # only a complete file ever gets the real name
            self.warmup_path = path
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            self.warmup_error = repr(exc)
            return  # not saved this time; Loopy works anyway
        for old in path.parent.glob("warmup-*.state"):  # outdated copies
            if old != path:
                old.unlink(missing_ok=True)
        gc.collect()

    def reset(self) -> None:
        """Forget the conversation (Loopy does this when it falls asleep).
        Any game in progress ends; the score is kept."""
        with self._lock:
            self._history.clear()
            self.pending = None
            self.game = None
            self.snake_open = False

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
                result = "; ".join(f"{n} = {v}" for n, v in found)
                self._history.append({"role": "system", "content": f"Result: {result}"})
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
                for _name, value in checked:
                    numbers = re.findall(r"\d+(?:\.\d+)?", value)
                    said = any(re.search(rf"(?<![\d.]){re.escape(n)}(?![\d.])", gate.spoken) for n in numbers)
                    if numbers and not said:
                        gate.say(f"To be exact: {value}.")

            if gate.fake_board and gate.kind == "none" and (self.game is None or self.game.over):
                # It drew a board in text instead of starting the game
                # ("Here's the board: 1 | 2 | 3"): that's what it meant, so start
                # the real one. (The text board itself was never spoken.)
                gate.kind, gate.game_cmd = "game", "start"

            if gate.kind == "none" and not gate.spoken and not checked:
                # Still nothing after every retry: Loopy must never go silent.
                # ([ask]/[do] always get Loopy's own words in _decide.)
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

    def new_game(self, on_text: TextOut, on_event: GameEvent | None = None, kind: str = "tic-tac-toe") -> Turn:
        """Start a game without being asked in words (`loopy tic-tac-toe`,
        `loopy snake`, or the right-click menu). kind: "tic-tac-toe" or "snake"."""
        llm = self.load()
        with self._lock:
            self._said = []
            self._on_text = on_text
            self._on_event = on_event or (lambda *_e: None)
            self._trim(llm)
            self.pending = None
            # Recorded as if asked, so the model knows why a game is open.
            if kind == "snake":
                self._history.append({"role": "user", "content": "Let's play snake!"})
                self._say("Ooh, snake!")
                self._remember("game: snake", "Ooh, snake!")
                self._start_snake()
            else:
                self._history.append({"role": "user", "content": "Let's play tic-tac-toe!"})
                self._say("Oh, you want to lose? Fine.")
                self._remember("game: start", "Oh, you want to lose? Fine.")
                self._close_snake()
                self._start_game(llm, announce=True)
            return Turn("".join(self._said).strip())

    # ------------------------------------------------------------- snake
    # The user plays snake alone with the keyboard (see snake.py); the brain
    # only opens/closes it and remembers how it went, so Loopy can cheer.

    SNAKE_STILL_PLAYING = "Hey, are you still playing? Say yes to keep going, or no and I'll close the game."

    def snake_idle(self) -> str:
        """The snake game paused itself (nobody steering). Returns what
        Loopy asks; tells the model how to read the answer."""
        with self._lock:
            self._history.append({"role": "assistant", "content": f"[none] {self.SNAKE_STILL_PLAYING}"})
            self._history.append({"role": "system", "content":
                                  "The snake game is paused: the user stopped steering. If they want to keep "
                                  "playing, reply with [game: snake]. If not, reply with [game: stop]."})
        return self.SNAKE_STILL_PLAYING

    def _start_snake(self) -> None:
        if self.snake_open:  # already open (e.g. "yes, keep playing"): carry on, don't restart
            self._on_event("snake_resume")
            already_said = "".join(self._said).strip()  # e.g. the model's "Yay, back to it!"
            self._say(("" if already_said else "Okay, back to it! ") + "Press an arrow key when you're ready.")
            self._history.append({"role": "system", "content": "The user is playing snake again."})
            return
        if self.game is not None:  # one game at a time
            self.game = None
            self._on_event("stop")
        self.snake_open = True
        self._on_event("snake_start")
        self._say("Steer with the arrow keys, and don't bite your own tail!")
        self._history.append({"role": "system", "content":
                              "Snake game opened. The user plays alone with the arrow keys; you cheer them on."})

    def _close_snake(self) -> None:
        if self.snake_open:
            self.snake_open = False
            self._on_event("snake_stop")

    def note(self, text: str) -> None:
        """Tell the model something that happened outside the conversation,
        e.g. "The user's snake game ended with a score of 12"."""
        with self._lock:
            self._history.append({"role": "system", "content": text})

    def _game_turn(self, llm, gate: "_Gate") -> None:
        if gate.kind == "game":
            if gate.game_cmd == "stop":
                if not gate.spoken:
                    gate.say("Okay, good game!")
                self._remember("game: stop", gate.spoken)
                if self.game is not None:
                    self.game = None
                    self._on_event("stop")
                self._close_snake()
                return
            if gate.game_cmd == "snake":
                self._remember("game: snake", gate.spoken)
                self._start_snake()
                return
            self._remember("game: start", gate.spoken)
            self._close_snake()
            self._start_game(llm, announce=True)
            return

        # The user's move.
        cell = gate.cell
        self._history.append({"role": "assistant", "content": f"[move: {cell}]"})
        assert cell is not None  # the gate only reports "move" with a cell
        if self.game is None or self.game.over:
            self._start_game(llm, announce=False, user_first=True)
        game = self._board()
        if not game.is_free(cell):
            why = game.why_not(cell)
            self._say(f"Nice try, but {why}! {self._taunt(TAUNT_TAKEN)} "
                      f"You can pick {_and([str(c) for c in game.free()])}.")
            self._history.append({"role": "system", "content": f"The user's move didn't count: {why}."})
            return
        game.place(cell, "X")
        self._on_event("place", cell, "X")
        if game.over:
            self._game_over()
        else:
            self._loopy_turn(llm, user_move=cell)

    def _start_game(self, llm, announce: bool, user_first: bool = False) -> None:
        loopy_first = self._loopy_starts_next and not user_first
        self._loopy_starts_next = not loopy_first
        self.game = TicTacToe()
        self._game_comments = []  # Loopy's trash talk this game, so it doesn't repeat itself
        self._on_event("start", loopy_first)
        if announce:
            if loopy_first:
                self._say(f"You're X and I'm O. I'll go first! {self._taunt(TAUNT_LOOPY_FIRST)}")
            else:
                self._say(f"You're X and I'm O. You go first, just say a number from 1 to 9. "
                          f"{self._taunt(TAUNT_YOU_FIRST)}")
        self._history.append({"role": "system", "content":
                              "Tic-tac-toe game started. The user is X, you are O. "
                              + ("You go first." if loopy_first else "The user goes first.")})
        if loopy_first:
            self._loopy_turn(llm)

    def _loopy_turn(self, llm, user_move: int | None = None) -> None:
        """The model picks Loopy's move; the code checks it and, if it's
        illegal, says why and lets the model pick again."""
        game = self._board()
        self._on_event("thinking")
        problem = ""
        for _ in range(MAX_PLAY_TRIES):
            board = _board_message(game, problem, user_move=user_move, said_before=self._game_comments)
            self._history.append({"role": "system", "content": board})
            gate = self._generate_with_words(llm, self._out, validate_play=game.is_free)
            cell = gate.cell if gate.kind == "play" else None
            if cell is not None and game.is_free(cell):
                if not gate.spoken:  # still silent after the retries: say where it went
                    gate.say(f"I'll take {cell}, {CELL_NAMES[cell]}.")
                self._game_comments.append(gate.spoken)
                self._remember(f"play: {cell}", gate.spoken)
                game.place(cell, "O")
                self._on_event("place", cell, "O")
                break
            # Illegal (or no move at all): its comment was never spoken.
            tried = f"[play: {cell}]" if cell else f"[none] {gate.spoken}"
            self._history.append({"role": "assistant", "content": tried})
            why = game.why_not(cell) if cell else "you didn't pick a cell"
            self._on_event("illegal", cell, why)
            problem = f"Your move {cell} isn't allowed: {why}." if cell else "You must pick a free cell."
        else:
            cell = random.choice(game.free())
            line = f"Hmm, I'll just take {cell}. Even my random moves are scary."
            self._say(line)
            self._remember(f"play: {cell}", line)
            game.place(cell, "O")
            self._on_event("place", cell, "O")

        if game.over:
            self._game_over()
        else:
            self._on_event("your_turn")

    def _board(self) -> TicTacToe:
        """The game being played (only called while one is on)."""
        assert self.game is not None, "no tic-tac-toe game is on"
        return self.game

    def _game_over(self) -> None:
        winner = self._board().winner
        if winner == "X":
            self.scores["you"] += 1
            self._say(f"Three in a row, you win! {self._taunt(TAUNT_YOU_WIN)} Rematch?")
            outcome = "the user won"
        elif winner == "O":
            self.scores["loopy"] += 1
            wins = self.scores["loopy"]
            brag = f"That's {wins} for me!" if wins > 1 else self._taunt(TAUNT_LOOPY_WINS)
            self._say(f"Three in a row, I win! {brag} Want a rematch?")
            outcome = "you (Loopy) won"
        else:
            self.scores["draws"] += 1
            self._say(f"It's a draw! {self._taunt(TAUNT_DRAW)} One more game?")
            outcome = "it's a draw"
        s = self.scores
        self._history.append({"role": "system", "content":
                              f"Game over: {outcome}. Score: user {s['you']}, Loopy {s['loopy']}, draws {s['draws']}. "
                              "The board is still open. For a rematch reply with [game: start]; "
                              "if they say no, or want to close the game, reply with [game: stop]."})
        self._on_event("over", winner, self._board().line, dict(s))

    def _taunt(self, options: list[str]) -> str:
        """A random bit of friendly trash talk, not the same one twice in a row."""
        fresh = [t for t in options if t != self._last_taunt] or options
        self._last_taunt = random.choice(fresh)
        return self._last_taunt

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
            # These need no words from the model: nothing goes after the tag, or
            # (for a real [ask: action]) Loopy asks its own question if needed.
            wordless = gate.kind in ("check", "move", "game", "ask") or gate.fake_board
            illegal = gate.kind == "play" and not (validate_play and validate_play(gate.cell))
            if gate.spoken or wordless or illegal or attempt == EMPTY_RETRIES:
                return gate
            if gate.kind == "play":  # a fine move, but no comment: say exactly what's missing
                note = (f"You picked {gate.cell} but said nothing. Reply again with [play: {gate.cell}] "
                        "and one snippy line of trash talk, like \"[play: 5] The center's mine. Try to keep up.\".")
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
        if gate.kind == "do" and gate.allowed_to_run and gate.action:
            self.pending = None
            if not gate.spoken:
                gate.say("Okay!")
            self._remember(f"do: {gate.action}", gate.spoken)
            result = actions.run(gate.action)
            self._history.append({"role": "system", "content": f"Action done: {gate.action}. Result: {result.message}"})
            return Turn(gate.spoken, gate.action, result)

        if gate.kind in ("ask", "do"):
            # Asking (or claiming a yes Loopy never asked for): make sure it
            # really asks, and remember the question for next turn.
            self.pending = gate.action
            if not gate.spoken.rstrip().endswith("?") or not actions.is_named(gate.action or "", gate.spoken):
                gate.say(f"Do you want me to {actions.describe(gate.action or '')}?")
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
        self.fake_board = False  # it started drawing a tic-tac-toe board in text
        self._hold_ask = False   # [ask: ...]: hold the words until we know it's a question
        self._ask_text = ""

    @property
    def halt(self) -> bool:
        """Nothing more may be spoken, so stop generating."""
        return (self.stopped or self.kind in ("check", "move")
                or (self.kind == "play" and self._suppress))

    # A fake turn the model writes itself, e.g. "system: Result: chrome = 0% open"
    # or "...closed. user: ...": nothing from there on may be spoken.
    _ROLE = re.compile(r"(?:^|(?<=[.!?\]])\s+|\n)\s*(?:system|user|assistant)\s*:", re.IGNORECASE)
    # A board drawn in text ("1 | 2 | 3"): the model is playing a game in words
    # instead of starting the real one. "|" never belongs in spoken words.
    _FAKE_BOARD = re.compile(r"[1-9XO]\s*\|")

    def feed(self, raw: str, final: bool = False) -> None:
        if self._body_start is None and not self._read_tag(raw):
            return
        if self.stopped:
            return
        end = len(raw)
        assert self._body_start is not None  # set by _read_tag
        body = raw[self._body_start:]
        role, board = self._ROLE.search(body), self._FAKE_BOARD.search(body)
        cut = min([m.start() for m in (role, board) if m], default=None)
        if cut is not None:
            end = max(self._pos, self._body_start + cut)
            self.stopped = True
            self.fake_board = bool(board and board.start() == cut)
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
        if self._hold_ask:
            # Say the model's words only if they really ask, about this action.
            # "I'm opening Chrome for you." would be a false claim, and a yes to
            # "Can I?" would open Chrome without the question ever naming it:
            # then only Loopy's own question is spoken (added by the caller).
            self._hold_ask = False
            text, self._ask_text = self._ask_text.strip(), ""
            if "?" in text and self.action and actions.is_named(self.action, text):
                self._emit(text)
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
        if self._hold_ask:
            self._ask_text += piece  # decided in finish()
            return
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
            self.game_cmd = ("stop" if re.search(r"stop|end|quit|exit|close", action)
                             else "snake" if "snake" in action else "start")
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
            if kind == "ask":
                self._hold_ask = True
            if kind == "do":
                if action == self._pending:
                    self.allowed_to_run = True
                else:
                    # About to say "Okay, doing X" without having asked: don't.
                    self._suppress = True
                    self.say(f"Do you want me to {action}?")
        self._body_start = self._pos = m.end()
        return True
