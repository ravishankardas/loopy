"""Desktop app: wires the mascot, ears (listener + Whisper), brain (Llama)
and voice together.

Flow:
    asleep --"Hey Loopy"--> awake --(no activity for a while)--> asleep
While awake, everything you say goes to Loopy's brain, which chats back and
asks before doing anything on the computer. "Hey Loopy, how are you?" works
in one go, asleep or awake. Falling asleep ends the conversation.
"""

import os
import random
import signal
import sys
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication, QInputDialog, QMenu

from loopy import models, settings, wake
from loopy.game import X_COLOR, BoardWindow, Celebration
from loopy.listener import KNOCK_ABOVE_NOISE, KNOCK_PEAK
from loopy.log import logger, setup_logging
from loopy.mascot import Activity, Mascot, State
from loopy.models import Voice
from loopy.snake import SnakeWindow

if TYPE_CHECKING:  # loaded lazily at runtime (heavy imports)
    from loopy.brain import Brain
    from loopy.sounds import Clip
    from loopy.stt import LiveTranscription, Transcriber
    from loopy.tts import Speaker

DEFAULT_SLEEP_AFTER = 60  # seconds without activity before Loopy dozes off
GAME_SLEEP_AFTER = 300    # ...longer during a game, so you have time to think
# Seconds between live transcription passes while the user talks (partial text
# in the bubble), or None for only the pass when they pause. See LiveTranscription.
LIVE_PASS_EVERY: float | None = None
SNAKE_ANSWER_SECONDS = 30 # "still playing snake?" with no answer: close it


def _log(text: str = "") -> None:
    """One line of the conversation: in the terminal and the log file."""
    logger.opt(depth=1).info(text)  # depth=1: report the caller's file:line, not this helper's


def _describe_game_event(event: tuple) -> str:
    kind, *args = event
    if kind == "start":
        return "── tic-tac-toe started: " + ("Loopy goes first" if args[0] else "you go first") + " ──"
    if kind == "place":
        cell, mark = args
        return f"[{'you' if mark == 'X' else 'Loopy'}: {mark} in {cell}]"
    if kind == "illegal":
        cell, why = args
        return f"[Loopy tried {cell}: {why}, picking again]"
    if kind == "over":
        winner, _line, s = args
        result = {"X": "you win", "O": "Loopy wins"}.get(winner, "draw")
        return f"[game over: {result}. Score: you {s['you']}, Loopy {s['loopy']}, draws {s['draws']}]"
    if kind == "stop":
        return "── tic-tac-toe closed ──"
    return ""  # thinking / your_turn: only shown on the board


class Assistant(QObject):
    # Emitted from worker threads; Qt delivers them on the UI thread.
    _speech_started = Signal(str)            # first words of an utterance
    _speech_chunk = Signal(str)              # each piece as it's spoken
    _speech_finished = Signal()
    _download_progress = Signal(str, int)   # voice label, percent
    _download_done = Signal(str)             # voice id
    _download_failed = Signal(str)           # error message
    _turn_done = Signal(object)              # brain: Turn (or Exception)
    _thought = Signal(str)                   # brain text, when there's no voice
    _speech_detected = Signal()              # someone started talking
    _utterance = Signal(object)              # np.ndarray of audio
    _mic_level = Signal(float)               # 0..1 while talking
    _heard = Signal(str)                     # transcription result
    _hearing_so_far = Signal(str)            # partial transcription while the user talks
    _brain_ready = Signal(str)               # model finished loading ("" = ok, else the error)
    _knock = Signal(float)                   # someone hit the laptop (peak level)
    _game_event = Signal(object)             # tic-tac-toe event tuple, see brain.GameEvent

    def __init__(self, mascot: Mascot):
        super().__init__()
        self.mascot = mascot
        self.awake = True
        # What Loopy is doing right now; _refresh_status() turns these into
        # the always-visible status chip.
        self._speaking = False
        self._thinking = False
        self._transcribing = False
        self._hearing = False                # someone is talking right now
        self._brain_loaded = False
        self.brain_error = ""
        self.start_with_game: str | None = None  # "tic-tac-toe"/"snake": a game instead of hello
        self.snake: SnakeWindow | None = None
        self._party: Celebration | None = None   # new-best celebration for snake
        self._snake_answer_timer = QTimer(self)  # "still playing?" unanswered -> close
        self._snake_answer_timer.setSingleShot(True)
        self._snake_answer_timer.timeout.connect(self._on_snake_no_answer)
        self._spoken = ""                    # what the bubble shows while talking
        self.speaker: Speaker | None = None
        self.voice: Voice | None = settings.active_voice()
        self._downloading: Voice | None = None
        if self.voice:
            self._make_speaker(self.voice)

        self._speech_started.connect(self._on_speech_started)
        self._speech_chunk.connect(self._on_speech_chunk)
        self._speech_finished.connect(self._on_speech_finished)
        self._download_progress.connect(self._on_download_progress)
        self._download_done.connect(self._on_download_done)
        self._download_failed.connect(self._on_download_failed)
        self._turn_done.connect(self._on_turn_done)
        self._thought.connect(self._on_thought)
        self._speech_detected.connect(self._on_speech_detected)
        self._utterance.connect(self._on_utterance)
        self._mic_level.connect(mascot.set_level)
        self._heard.connect(self._on_heard)
        self._hearing_so_far.connect(self._on_hearing_so_far)
        self._brain_ready.connect(self._on_brain_ready)
        self._game_event.connect(self._on_game_event)
        self.board: BoardWindow | None = None
        self._knock.connect(self._on_knock)
        self._clip: Clip | None = None     # created on the first knock
        self._clip_playing = False
        self._clip_id = 0
        self._state_before_clip = State.IDLE
        mascot.speak_requested.connect(self.say)
        mascot.menu_extender = self._extend_menu

        self._sleep_after = settings.load().get("sleep_after_seconds", DEFAULT_SLEEP_AFTER)
        self._sleep_timer = QTimer(self)
        self._sleep_timer.setSingleShot(True)
        self._sleep_timer.setInterval(int(self._sleep_after * 1000))
        self._sleep_timer.timeout.connect(self.go_to_sleep)

        # Brain: loading takes ~20 s, so start it now in the background.
        self.brain: Brain | None = None
        if models.llm_installed():
            from loopy import brain as brain_module

            self.brain = brain = brain_module.Brain(models.llm_path())

            def load_brain():
                _log(f"Loading Loopy's brain ({models.active_llm().name})...")
                started = time.monotonic()
                try:
                    brain.load()
                    self._load_seconds = time.monotonic() - started
                except Exception as exc:
                    self._brain_ready.emit(repr(exc))
                else:
                    self._brain_ready.emit("")

            threading.Thread(target=load_brain, daemon=True).start()

        # Ears: always-on listener -> Whisper.
        self.transcriber: Transcriber | None = None
        self.live: LiveTranscription | None = None
        self.listener = None
        self.mic_error = None
        if models.stt_installed():
            from loopy import stt
            from loopy.listener import Listener

            self.transcriber = stt.Transcriber(models.stt_path())
            threading.Thread(target=self.transcriber.load, daemon=True).start()  # warm up
            # Streaming: transcribe while the user talks, so the text is ready
            # (nearly) as soon as they stop. See stt.LiveTranscription.
            self.live = stt.LiveTranscription(self.transcriber, on_partial=self._hearing_so_far.emit)
            self.listener = Listener(
                on_utterance=self._utterance.emit,
                on_partial=self.live.update,
                partial_every=LIVE_PASS_EVERY,
                on_speech_start=self._speech_detected.emit,
                on_level=self._mic_level.emit,
                on_knock=self._knock.emit,
                knock_peak=settings.load().get("knock_peak", KNOCK_PEAK),
                knock_above_noise=settings.load().get("knock_above_noise", KNOCK_ABOVE_NOISE),
            )
            # No knock detection while the model loads: the CPU is flat out,
            # the mic drops audio, and the gaps click like a hit.
            self.listener.knocks_enabled = self.brain is None
            try:
                self.listener.start()
            except Exception as exc:
                self.listener = None
                self.mic_error = str(exc)
        self._refresh_status()

    @property
    def ready(self) -> bool:
        return all((self.speaker, self.transcriber, self.brain))

    # --------------------------------------------------------------- status

    def _refresh_status(self) -> None:
        """Make the status chip say what Loopy is doing, always. Called after
        every change of state; the order below is the priority."""
        m = self.mascot
        m.set_thinking(self._thinking and self._speaking)  # dots keep going while it talks
        if self._clip_playing:
            status = ("Ho yeah!", Activity.SPEAKING)
        elif self._downloading:
            status = (f"Downloading {self._downloading.label.split(' (')[0]}'s voice...", Activity.BUSY)
        elif self.brain_error:
            status = ("Brain failed to load (see terminal)", Activity.IDLE)
        elif not self.ready:
            status = ("Setup needed: run 'loopy setup'", Activity.IDLE)
        elif self.listener is None:
            status = ("Microphone unavailable", Activity.IDLE)
        elif not self.awake:
            if self._transcribing:
                status = ("Was that my name?", Activity.BUSY)
            elif self._hearing:
                status = ("Hmm? Listening...", Activity.LISTENING)
            else:
                status = ('Asleep · say "Hey Loopy"', Activity.SLEEPING)
        elif self._speaking:
            status = ("Talking · still thinking" if self._thinking else "Talking", Activity.SPEAKING)
        elif self._thinking:
            status = ("Thinking..." if self._brain_loaded else "Waking up my brain...", Activity.BUSY)
        elif self._transcribing:
            status = ("Catching that...", Activity.BUSY)
        elif self._hearing:
            status = ("Listening...", Activity.LISTENING)
        elif not self._brain_loaded:
            status = ("Getting ready...", Activity.BUSY)
        elif self._snake_on and self.snake and self.snake.idle_paused:
            status = ("Snake paused · still playing?", Activity.IDLE)
        elif self._snake_on and self.snake:
            status = (f"Snake · score {self.snake.game.score}", Activity.IDLE)
        elif self._game_on:
            status = ("Tic-tac-toe · your turn", Activity.IDLE)
        else:
            status = ("Awake · just talk", Activity.IDLE)
        m.set_status(*status)

    def _on_brain_ready(self, error: str) -> None:
        if self.listener:
            self.listener.enable_knocks(after=1.0)  # loading is over; give the CPU a second to settle
        if error:
            _log(f"       [brain failed to load: {error}]")
            self.brain_error = error
            self.mascot.show_bubble("My brain didn't load, sorry. The terminal says why.", 10000)
            self._refresh_status()
            return
        self._brain_loaded = True
        how = {"saved": "used the saved warm-up",
               "fresh": "warmed up and saved it, next start will be fast"}.get(
                   (self.brain.warmup_source if self.brain else None) or "", "")
        _log(f"Loopy is ready ({getattr(self, '_load_seconds', 0):.0f} s{', ' + how if how else ''}).")
        self._refresh_status()
        # Only now, with the model loaded, does Loopy say hello (unless you
        # already asked something while it was loading: then it just answers).
        if self.start_with_game and not self._thinking:
            self.play_game(self.start_with_game)  # `loopy tic-tac-toe` / `loopy snake`: a game instead of hello
        elif self.listener is not None and self.speaker and self.transcriber and not self._thinking:
            self.say("Hi! I'm Loopy. Say hey Loopy whenever you want to chat.")
        self._keep_awake()

    # -------------------------------------------------------- awake / asleep

    def wake_up(self, greet: bool = True) -> None:
        if not self.awake:
            _log("── Loopy woke up ──")
        self.awake = True
        if self.mascot.state() == State.SLEEPING:
            self.mascot.set_state(State.IDLE)
        if greet:
            self.say("Yes?")  # countdown starts once it finishes speaking
        else:
            self._keep_awake()
        self._refresh_status()

    def go_to_sleep(self) -> None:
        if self.awake:
            _log("── Loopy fell asleep ──")
        self.awake = False
        self._hearing = False
        self._sleep_timer.stop()
        if self.speaker:
            self.speaker.stop()
        if self.brain:
            self.brain.reset()  # a new wake-up starts a new conversation (and ends any game)
        if self.board:
            self.board.hide()
        if self.snake:
            self.snake.hide()
        self.mascot.set_state(State.SLEEPING)
        self.mascot.show_bubble("", 0)
        self._refresh_status()

    def _keep_awake(self) -> None:
        """Restart the sleep countdown. Called whenever Loopy goes back to
        idle; while it's listening, thinking or speaking the countdown is
        paused (see _busy), so it can't doze off mid-answer."""
        if self.brain and not self._brain_loaded and not self.brain_error:
            return  # still loading: don't count that as being idle
        if self.awake and not self._thinking and not self._speaking:
            seconds = max(self._sleep_after, GAME_SLEEP_AFTER) if self._game_on else self._sleep_after
            self._sleep_timer.start(int(seconds * 1000))

    @property
    def _game_on(self) -> bool:
        tic_tac_toe = bool(self.brain and self.brain.game and not self.brain.game.over)
        return tic_tac_toe or self._snake_on

    def _busy(self) -> None:
        self._sleep_timer.stop()

    # ------------------------------------------------------------ listening

    def _on_speech_detected(self) -> None:
        self._hearing = True
        if not self.awake:
            self._refresh_status()  # asleep: show it heard something
            return
        if self.mascot.state() == State.IDLE:
            self.mascot.set_state(State.LISTENING)
            self.mascot.show_bubble("", 0)
            self._busy()
        self._refresh_status()

    def _on_utterance(self, audio: np.ndarray) -> None:
        self._hearing = False
        live = self.live
        assert live is not None, "speech only reaches here once the transcriber exists"
        if len(audio) == 0:  # too short to be speech
            live.cancel()
            self._back_to_rest()
            return
        self._transcribing = True
        if self.awake:
            self._busy()
            self.mascot.set_state(State.WORKING)
        self._refresh_status()
        threading.Thread(target=lambda: self._heard.emit(live.finish(audio)), daemon=True).start()

    def _on_hearing_so_far(self, text: str) -> None:
        """Show what Loopy has heard so far while the user is still talking."""
        if self.awake and self._hearing and self.mascot.state() == State.LISTENING:
            self.mascot.show_bubble(f"{text}…", 0)

    def _on_heard(self, text: str) -> None:
        self._transcribing = False
        self._refresh_status()
        woke, rest = wake.split_wake(text)
        if text:
            if self.awake or woke:
                _log(f"You:   {text}")
            else:
                _log(f'       (asleep, ignored: "{text}")')
        if not self.awake:
            if woke:  # "Hey Loopy" or "Hey Loopy, <anything>"
                self.wake_up(greet=not rest)
                if rest:
                    self.think(rest)
            return  # asleep: ignore everything else

        message = rest if woke else text
        if woke and not rest:
            self.mascot.set_state(State.IDLE)
            self.say("Yes?")
        elif not message:
            self._back_to_rest()
        else:
            self.think(message)

    def _back_to_rest(self) -> None:
        if self.awake:
            if self.mascot.state() in (State.LISTENING, State.WORKING) and not self._thinking:
                self.mascot.set_state(State.IDLE)
                self.mascot.show_bubble("", 0)
            self._keep_awake()
        self._refresh_status()

    # ---------------------------------------------------------------- brain

    def play_game(self, kind: str = "tic-tac-toe") -> None:
        """Start a game right away (`loopy tic-tac-toe` / `loopy snake`, or
        the menu). kind: "tic-tac-toe" or "snake"."""
        if not self.awake:
            self.wake_up(greet=False)
        self.think("", new_game=kind)

    def think(self, text: str, new_game: str | None = None) -> None:
        """Send what the user said to the brain (or, with new_game, start
        that game). Its words go straight into the voice as the model writes
        them. Runs off the UI thread."""
        text = text.strip()
        if not text and not new_game:
            return
        if self.brain is None:
            self.mascot.set_state(State.IDLE)
            self.mascot.show_bubble("I can't think yet. Run 'loopy setup' to install my brain.", 6000)
            return
        self._busy()
        self._thinking = True
        self.mascot.set_state(State.WORKING)
        bubble = {"tic-tac-toe": "Tic-tac-toe time!", "snake": "Snake time!"}.get(new_game or "", f'"{text}"')
        self.mascot.show_bubble(bubble, None)  # what Loopy heard
        self._refresh_status()

        stream = self._open_speech() if self.speaker else None
        speak = stream.feed if stream else self._thought.emit
        self._spoken = ""

        line: list[str] = []  # Loopy's words so far; logged as one line once the reply (or a part) ends

        def out(piece: str) -> None:  # to the voice as it's written, and to the log
            speak(piece)
            line.append(piece)

        def end_line():
            if text := "".join(line).strip():
                _log(f"Loopy: {text}")
            line.clear()

        def on_event(*event):  # logged here, in order with the speech, then shown on the board
            if description := _describe_game_event(event):
                end_line()
                _log(f"       {description}")
            self._game_event.emit(event)

        brain = self.brain
        assert brain is not None, "think() is only called once the brain is installed"

        def work():
            try:
                if new_game:
                    turn = brain.new_game(on_text=out, on_event=on_event, kind=new_game)
                else:
                    turn = brain.respond(text, on_text=out, on_event=on_event)
                if turn.result is not None and not turn.result.ok:
                    out(f" {turn.result.message}")  # e.g. "Chrome isn't open."
                end_line()
                for name, value in turn.checked:
                    _log(f"       [checked {name}: {value}]")
                if turn.action and turn.result is not None:
                    _log(f"       [{'done' if turn.result.ok else 'failed'}: {turn.result.message}]")
            except Exception as exc:  # keep Loopy alive whatever happens
                out(" Sorry, my brain got a bit tangled. Can you say that again?")
                end_line()
                logger.exception(f"       [brain error: {exc!r}]")
                turn = None
            finally:
                if stream:
                    stream.close()
            self._turn_done.emit(turn)

        threading.Thread(target=work, daemon=True).start()

    def _on_turn_done(self, turn) -> None:
        self._thinking = False
        if self.mascot.state() == State.WORKING and not self._transcribing:  # nothing (more) to say
            self.mascot.set_state(State.IDLE)
            self.mascot.show_bubble(self._spoken or (turn.reply if turn else ""), 2500)
        self._keep_awake()
        self._refresh_status()

    # ---------------------------------------------------------- tic-tac-toe

    def _on_game_event(self, event: tuple) -> None:
        kind, *args = event
        if kind == "snake_start":
            self._open_snake()
        elif kind == "snake_resume":  # "yes, keep playing"
            self._snake_answer_timer.stop()
            if self.snake and self.snake.isVisible():
                self.snake.get_ready()
            else:
                self._open_snake()
        elif kind == "snake_stop":
            self._snake_answer_timer.stop()
            if self.snake:
                self.snake.hide()
        elif kind == "start":
            (loopy_first,) = args
            if self.board is None:
                self.board = BoardWindow()
            self.board.reset()
            s = self.brain.scores if self.brain else {"you": 0, "draws": 0, "loopy": 0}
            self.board.set_scores(s["you"], s["draws"], s["loopy"])
            self.board.set_status("Loopy goes first..." if loopy_first else "Your turn · you're X")
            self.board.open_next_to(self.mascot)
        elif self.board is None:
            return
        elif kind == "place":
            cell, mark = args
            self.board.place(cell, mark)
        elif kind == "thinking":
            self.board.set_status("Loopy is thinking...")
        elif kind == "illegal":
            cell, _why = args
            self.board.set_status(f"Loopy tried {cell or '?'}, nope! Rethinking...")
        elif kind == "your_turn":
            self.board.set_status("Your turn · say a number")
        elif kind == "over":
            winner, line, scores = args
            if line:
                self.board.show_win(line)
            if winner in ("X", "O"):
                board = self.board
                QTimer.singleShot(450, lambda w=winner: board.celebrate(w))  # after the win line draws
            self.board.set_scores(scores["you"], scores["draws"], scores["loopy"])
            self.board.set_status({"X": "You win! \U0001F389", "O": "Loopy wins!"}.get(winner, "It's a draw!"))
        elif kind == "stop":
            self.board.hide()
        self._refresh_status()

    # ---------------------------------------------------------------- snake

    def _open_snake(self) -> None:
        if self.snake is None:
            self.snake = SnakeWindow(best=settings.load().get("snake_best", 0))
            self.snake.ate.connect(self._on_snake_ate)
            self.snake.finished.connect(self._on_snake_over)
            self.snake.closed.connect(self._on_snake_closed)
            self.snake.idle.connect(self._on_snake_idle)
            self.snake.resumed.connect(self._snake_answer_timer.stop)
        self.snake.new_game()
        self.snake.open_next_to(self.mascot)
        self._keep_awake()

    def _on_snake_idle(self) -> None:
        """The snake kept going in one edge and out the other with nobody
        steering: the game paused itself. Ask whether they're still playing;
        "yes" carries on, "no" (or no answer) closes it."""
        _log("       [snake paused: nobody's steering]")
        if self.brain is None:
            return
        question = self.brain.SNAKE_STILL_PLAYING
        threading.Thread(target=self.brain.snake_idle, daemon=True).start()  # tells the model how to read the answer
        self.say(question)
        self._snake_answer_timer.start(SNAKE_ANSWER_SECONDS * 1000)
        self._refresh_status()

    def _on_snake_no_answer(self) -> None:
        if self.snake and self.snake.isVisible() and self.snake.idle_paused:
            self.snake.hide()
            if self.brain:
                self.brain.snake_open = False
                threading.Thread(target=self.brain.note, daemon=True,
                                 args=("Nobody answered, so Loopy closed the snake game.",)).start()
            _log("       ── snake closed: no answer ──")
            self.say("No answer, so I closed the snake game. Say let's play snake anytime!")
            self._refresh_status()

    def _on_snake_ate(self, score: int) -> None:
        self._keep_awake()  # playing counts as activity: no dozing off mid-game
        self.mascot.show_bubble(f"Yum! {score}", 900)
        if score % 5 == 0:  # cheer now and then, not on every bite
            self.say(random.choice([f"{score}! Keep going!", f"Nice, {score}!", f"{score} already, wow!"]))
        self._refresh_status()

    def _on_snake_over(self, score: int) -> None:
        best_before = settings.load().get("snake_best", 0)
        if score > best_before:
            settings.save(snake_best=score)
            self.say(f"Game over! You scored {score}, that's a new best!")
            if self._party is None:
                self._party = Celebration()
            if self.snake is not None:
                self._party.play(f"New best: {score}!", X_COLOR, self.snake.frameGeometry().center())
        elif score == 0:
            self.say("Game over! Zero this time, give it another go!")
        else:
            self.say(f"Game over! You scored {score}. Your best is {best_before}.")
        _log(f"       [snake over: score {score}, best {max(score, best_before)}]")
        best = max(score, best_before)
        if self.brain is None:
            return
        threading.Thread(target=self.brain.note, daemon=True, args=(
            f"The user's snake game ended with a score of {score} (their best is {best}). "
            "The window is still open: Space plays again.",)).start()
        self._keep_awake()
        self._refresh_status()

    def _on_snake_closed(self) -> None:
        """Closed with Esc (not by asking Loopy)."""
        self._snake_answer_timer.stop()
        if self.brain:
            self.brain.snake_open = False
            threading.Thread(target=self.brain.note, daemon=True,
                             args=("The user closed the snake game.",)).start()
        _log("       ── snake closed ──")
        self._keep_awake()
        self._refresh_status()

    @property
    def _snake_on(self) -> bool:
        return bool(self.snake and self.snake.isVisible())

    def _ask_by_typing(self) -> None:
        text, ok = QInputDialog.getText(self.mascot, "Loopy", "Say something to Loopy:")
        if ok and text.strip():
            _log(f"You:   {text.strip()}   (typed)")
            if not self.awake:
                self.wake_up(greet=False)
            self.think(text)

    def _extend_menu(self, menu: QMenu) -> None:
        menu.addAction("Type to Loopy...", self._ask_by_typing)
        if self._brain_loaded and not self._game_on and not self._thinking:
            menu.addAction("Play tic-tac-toe", lambda: self.play_game("tic-tac-toe"))
            menu.addAction("Play snake", lambda: self.play_game("snake"))
        if self.awake:
            menu.addAction("Go to sleep", self.go_to_sleep)
        else:
            menu.addAction("Wake up", lambda: self.wake_up())
        self._add_voice_menu(menu)

    # -------------------------------------------------------------- speech

    def say(self, text: str) -> None:
        """Speak `text` aloud (or just show it, if no voice is installed)."""
        _log(f"Loopy: {text}")
        if self.speaker is None:
            self.mascot.show_bubble(text)
            self._keep_awake()
            return
        stream = self._open_speech()
        stream.feed(text)
        stream.close()

    def _open_speech(self):
        assert self.speaker is not None, "only called when a voice is set up"
        return self.speaker.stream(
            on_start=self._speech_started.emit,
            on_chunk=self._speech_chunk.emit,
            on_done=self._speech_finished.emit,
        )

    def _make_speaker(self, voice: Voice) -> None:
        from loopy.tts import Speaker

        if self.speaker is None:
            self.speaker = Speaker(voice.path)
        else:
            self.speaker.stop()
            self.speaker.set_model(voice.path)

    def _on_speech_started(self, _first: str) -> None:
        self._busy()
        self._speaking = True
        self._spoken = ""
        if self.listener:
            self.listener.pause()  # so Loopy doesn't hear (and answer) itself
        self._hearing = False
        self.mascot.set_state(State.SPEAKING)
        self._refresh_status()

    def _on_speech_chunk(self, text: str) -> None:
        self._spoken = f"{self._spoken} {text.strip()}".strip()
        shown = self._spoken if len(self._spoken) <= 140 else "..." + self._spoken[-137:]
        self.mascot.show_bubble(shown, None)

    def _on_thought(self, piece: str) -> None:
        """No voice installed: show the reply as it's written."""
        self._spoken = (self._spoken + piece).lstrip()
        self.mascot.show_bubble(self._spoken[-140:], None)

    def _on_speech_finished(self) -> None:
        self._speaking = False
        QTimer.singleShot(300, self._resume_listening)  # let the room go quiet first
        if self.mascot.state() == State.SPEAKING:
            # Still thinking: the model is writing more, so look busy, not idle.
            self.mascot.set_state(State.WORKING if self._thinking else State.IDLE)
            self.mascot.show_bubble(self._spoken[-140:], None if self._thinking else 2000)
        self._keep_awake()
        self._refresh_status()

    def _resume_listening(self) -> None:
        # Another sentence (or the clip) may have started in the last 300 ms; stay muted then.
        if self.listener and not self._speaking and not self._clip_playing:
            self.listener.resume()

    # ------------------------------------------------------- knock knock

    def _on_knock(self, peak: float) -> None:
        """Someone hit the laptop: "Can I get a ho yeah!" """
        if self._clip_playing or (self.brain and not self._brain_loaded and not self.brain_error):
            return  # already playing, or still starting up
        from loopy.sounds import HO_YEAH, Clip

        if self._clip is None:
            self._clip = Clip(HO_YEAH)
            self._clip.finished.connect(self._on_clip_finished)
        clip = self._clip
        info = self.listener.last_knock if self.listener else None
        detail = f"peak {info[0]:.3f}, {info[1]:.0f}x the room, sharpness {info[2]:.1f}" if info else f"peak {peak:.3f}"
        _log(f"       [knock! ({detail}) -> can I get a ho yeah!]")
        self._clip_playing = True
        self._clip_id += 1
        if self.speaker:
            self.speaker.hold()    # Loopy's own voice waits until the clip is over
            # Safety net if "finished" never comes; only for this clip, not a later one.
            this_clip = self._clip_id

            def safety_net() -> None:
                if this_clip == self._clip_id:
                    self._release_voice()

            QTimer.singleShot(10_000, safety_net)
        if self.listener:
            self.listener.pause()  # don't hear the clip as speech (or as more knocks)
        self._state_before_clip = self.mascot.state()
        self.mascot.set_state(State.SPEAKING)
        self.mascot.show_bubble("Can I get a ho yeah?! \U0001F64C", 3800)
        self._refresh_status()
        clip.play()

    def _release_voice(self) -> None:
        if self.speaker:
            self.speaker.release()

    def _on_clip_finished(self) -> None:
        self._clip_playing = False
        self._release_voice()  # anything Loopy wanted to say now plays
        QTimer.singleShot(300, self._resume_listening)
        if self.mascot.state() == State.SPEAKING and not self._speaking:
            if not self.awake:
                self.mascot.set_state(State.SLEEPING)
            elif self._thinking:
                self.mascot.set_state(State.WORKING)
            else:
                self.mascot.set_state(State.IDLE)
        self._refresh_status()

    def shutdown(self) -> None:
        self._sleep_timer.stop()
        if self._clip:
            self._clip.stop()
        self._release_voice()
        if self.listener:
            self.listener.stop()
        if self.speaker:
            self.speaker.close()

    # -------------------------------------------------------------- voices

    def _add_voice_menu(self, menu: QMenu) -> None:
        sub = menu.addMenu("Voice")
        for voice in models.VOICES:
            label = voice.label if voice.installed else f"{voice.label}  - download"
            action = sub.addAction(label)
            action.setCheckable(True)
            action.setChecked(voice == self.voice)
            action.setEnabled(self._downloading is None)
            action.triggered.connect(lambda _=False, v=voice: self.choose_voice(v))

    def choose_voice(self, voice: Voice) -> None:
        if voice == self.voice or self._downloading:
            return
        if not self.awake:
            self.wake_up(greet=False)
        if voice.installed:
            self._switch_to(voice)
            return
        self._downloading = voice
        self._busy()
        self.mascot.set_state(State.WORKING)
        self.mascot.show_bubble(f"Downloading {voice.label}...", None)
        self._refresh_status()
        threading.Thread(target=self._download, args=(voice,), daemon=True).start()

    def _download(self, voice: Voice) -> None:
        def progress(done: int, total: int) -> None:
            if total:
                self._download_progress.emit(voice.label, done * 100 // total)

        try:
            models.install_voice(voice, progress=progress)
        except OSError as exc:
            self._download_failed.emit(str(exc))
        else:
            self._download_done.emit(voice.id)

    def _on_download_progress(self, label: str, percent: int) -> None:
        self.mascot.show_bubble(f"Downloading {label}... {percent}%", None)
        self.mascot.set_status(f"Downloading voice... {percent}%", Activity.BUSY)

    def _on_download_done(self, voice_id: str) -> None:
        self._downloading = None
        self._refresh_status()
        self.mascot.set_state(State.IDLE)
        voice = models.get_voice(voice_id)
        if voice is not None:
            self._switch_to(voice)

    def _on_download_failed(self, error: str) -> None:
        self._downloading = None
        self._refresh_status()
        self.mascot.set_state(State.IDLE)
        self.mascot.show_bubble(f"Couldn't download that voice. Check your internet.\n({error})", 6000)
        self._keep_awake()

    def _switch_to(self, voice: Voice) -> None:
        self.voice = voice
        settings.save(voice=voice.id)
        self._make_speaker(voice)
        name = voice.label.split(" (")[0]
        self.say(f"Hi, I'm Loopy. This is the {name} voice.")


def _quit(app: QApplication, why: str) -> None:
    """Quit, saying why in the log (a silent exit is a mystery otherwise)."""
    logger.info(f"── Quitting ({why}) ──")
    app.quit()


def _quit_on_ctrl_c(app: QApplication) -> QTimer:
    """Make Ctrl+C in the terminal quit the app.

    Python only runs signal handlers when it gets control back, which never
    happens while Qt's event loop is idle, so a timer pokes it every 200 ms.
    """
    for sig in (signal.SIGINT, getattr(signal, "SIGBREAK", None)):  # SIGBREAK = Ctrl+Break on Windows
        if sig is not None:
            signal.signal(sig, lambda *_: _quit(app, "Ctrl+C in the terminal"))
    wake_timer = QTimer()
    wake_timer.timeout.connect(lambda: None)
    wake_timer.start(200)
    return wake_timer  # caller must keep a reference


def pick_display_backend(environ=os.environ, platform: str = sys.platform,
                         find_library=None) -> str | None:
    """On a Linux Wayland desktop (Ubuntu's default), apps can't move their
    own windows, so Loopy couldn't be dragged. Run through XWayland instead,
    where they can: dragging, games moving with Loopy and staying on top all
    work like on Windows. Qt's X11 backend needs libxcb-cursor, so only switch
    when it's there. Returns a hint to show if it isn't."""
    if not platform.startswith("linux") or "QT_QPA_PLATFORM" in environ:
        return None  # not Linux, or the user chose a backend themselves
    if not environ.get("WAYLAND_DISPLAY") and environ.get("XDG_SESSION_TYPE") != "wayland":
        return None  # plain X11: windows can be moved already
    if find_library is None:
        import ctypes.util
        find_library = ctypes.util.find_library
    if find_library("xcb-cursor"):
        environ["QT_QPA_PLATFORM"] = "xcb"  # XWayland
        return None
    return ("Running on Wayland: you can drag Loopy, but the game windows can't move with it. "
            "For the full experience: sudo apt install libxcb-cursor0")


def run(start_with_game: str | None = None) -> int:
    """Start Loopy. With start_with_game ("tic-tac-toe" or "snake", from
    `loopy tic-tac-toe` / `loopy snake`), it jumps straight into that game
    once its brain has loaded, instead of saying hello."""
    setup_logging()  # terminal + log file; also logs crashes in any thread
    logger.debug(f"Loopy starting: Python {sys.version.split()[0]}, {sys.platform}")
    # Sound clips: on Windows use its own media engine; Qt's default (FFmpeg)
    # prints decoder chatter into the terminal, cluttering the conversation log.
    if sys.platform == "win32":
        os.environ.setdefault("QT_MEDIA_BACKEND", "windows")
    os.environ.setdefault("QT_LOGGING_RULES", "qt.multimedia*=false")
    if hint := pick_display_backend():
        _log(f"[note] {hint}")
    app = QApplication(sys.argv)
    mascot = Mascot()
    mascot.quit_requested.connect(lambda: _quit(app, "Quit from the menu"))
    assistant = Assistant(mascot)
    assistant.start_with_game = start_with_game  # read once the brain is ready (inside app.exec)
    app.aboutToQuit.connect(assistant.shutdown)
    _wake = _quit_on_ctrl_c(app)
    mascot.show()

    if not assistant.ready:
        mascot.show_bubble("I'm not fully set up yet. Run 'loopy setup' in a terminal.", 10000)
    elif assistant.listener is None:
        mascot.show_bubble(f"I can't open the microphone.\n({assistant.mic_error})", 10000)
    # Otherwise Loopy stays quiet ("Getting ready...") until its brain has
    # loaded, then says hello (see Assistant._on_brain_ready).
    assistant._keep_awake()
    return app.exec()
