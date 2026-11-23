"""Desktop app: wires the mascot, ears (listener + Whisper), brain (Llama)
and voice together.

Flow:
    asleep --"Hey Loopy"--> awake --(no activity for a while)--> asleep
While awake, everything you say goes to Loopy's brain, which chats back and
asks before doing anything on the computer. "Hey Loopy, how are you?" works
in one go, asleep or awake. Falling asleep ends the conversation.
"""

import os
import signal
import sys
import threading
import time

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication, QInputDialog, QMenu

from loopy import models, settings, wake
from loopy.game import BoardWindow
from loopy.listener import KNOCK_ABOVE_NOISE, KNOCK_PEAK
from loopy.mascot import Activity, Mascot, State
from loopy.models import Voice

DEFAULT_SLEEP_AFTER = 60  # seconds without activity before Loopy dozes off
GAME_SLEEP_AFTER = 300    # ...longer during a game, so you have time to think


def _log(text: str = "", end: str = "\n") -> None:
    """The conversation, printed in the terminal Loopy was started from."""
    print(text, end=end, flush=True)


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
        self._spoken = ""                    # what the bubble shows while talking
        self.speaker = None
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
        self._brain_ready.connect(self._on_brain_ready)
        self._game_event.connect(self._on_game_event)
        self.board: BoardWindow | None = None
        self._knock.connect(self._on_knock)
        self._clip = None                    # created on the first knock
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
        self.brain = None
        if models.llm_installed():
            from loopy.brain import Brain

            self.brain = Brain(models.llm_path())

            def load_brain():
                _log("Loading Loopy's brain (Llama 3.2 3B)...")
                started = time.monotonic()
                try:
                    self.brain.load()
                    self._load_seconds = time.monotonic() - started
                except Exception as exc:
                    self._brain_ready.emit(repr(exc))
                else:
                    self._brain_ready.emit("")

            threading.Thread(target=load_brain, daemon=True).start()

        # Ears: always-on listener -> Whisper.
        self.transcriber = None
        self.listener = None
        self.mic_error = None
        if models.stt_installed():
            from loopy.listener import Listener
            from loopy.stt import Transcriber

            self.transcriber = Transcriber(models.stt_path())
            threading.Thread(target=self.transcriber.load, daemon=True).start()  # warm up
            self.listener = Listener(
                on_utterance=self._utterance.emit,
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
               "fresh": "warmed up and saved it, next start will be fast"}.get(self.brain.warmup_source, "")
        _log(f"Loopy is ready ({getattr(self, '_load_seconds', 0):.0f} s{', ' + how if how else ''}).")
        self._refresh_status()
        # Only now, with the model loaded, does Loopy say hello (unless you
        # already asked something while it was loading: then it just answers).
        if self.listener is not None and self.speaker and self.transcriber and not self._thinking:
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
        return bool(self.brain and self.brain.game and not self.brain.game.over)

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
        if len(audio) == 0:  # too short to be speech
            self._back_to_rest()
            return
        self._transcribing = True
        if self.awake:
            self._busy()
            self.mascot.set_state(State.WORKING)
        self._refresh_status()
        threading.Thread(target=lambda: self._heard.emit(self.transcriber.transcribe(audio)), daemon=True).start()

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

    def think(self, text: str) -> None:
        """Send what the user said to the brain. Its words go straight into
        the voice as the model writes them. Runs off the UI thread."""
        text = text.strip()
        if not text:
            return
        if self.brain is None:
            self.mascot.set_state(State.IDLE)
            self.mascot.show_bubble("I can't think yet. Run 'loopy setup' to install my brain.", 6000)
            return
        self._busy()
        self._thinking = True
        self.mascot.set_state(State.WORKING)
        self.mascot.show_bubble(f'"{text}"', None)  # what Loopy heard
        self._refresh_status()

        stream = self._open_speech() if self.speaker else None
        speak = stream.feed if stream else self._thought.emit
        self._spoken = ""

        line_open = [False]  # a "Loopy: ..." line is being printed

        def out(piece: str) -> None:  # to the voice and the terminal, as it's written
            speak(piece)
            if not line_open[0]:
                _log("Loopy: ", end="")
                piece = piece.lstrip()
                line_open[0] = True
            _log(piece, end="")

        def end_line():
            if line_open[0]:
                _log()
                line_open[0] = False

        def on_event(*event):  # logged here, in order with the speech, then shown on the board
            if description := _describe_game_event(event):
                end_line()
                _log(f"       {description}")
            self._game_event.emit(event)

        def work():
            try:
                turn = self.brain.respond(text, on_text=out, on_event=on_event)
                if turn.result is not None and not turn.result.ok:
                    out(f" {turn.result.message}")  # e.g. "Chrome isn't open."
                end_line()
                for name, value in turn.checked:
                    _log(f"       [checked {name}: {value}]")
                if turn.action:
                    _log(f"       [{'done' if turn.result.ok else 'failed'}: {turn.result.message}]")
            except Exception as exc:  # keep Loopy alive whatever happens
                out(" Sorry, my brain got a bit tangled. Can you say that again?")
                end_line()
                _log(f"       [brain error: {exc!r}]")
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
        if kind == "start":
            (loopy_first,) = args
            if self.board is None:
                self.board = BoardWindow()
            self.board.reset()
            s = self.brain.scores
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
                QTimer.singleShot(450, lambda w=winner: self.board.celebrate(w))  # after the win line draws
            self.board.set_scores(scores["you"], scores["draws"], scores["loopy"])
            self.board.set_status({"X": "You win! \U0001F389", "O": "Loopy wins!"}.get(winner, "It's a draw!"))
        elif kind == "stop":
            self.board.hide()
        self._refresh_status()

    def _ask_by_typing(self) -> None:
        text, ok = QInputDialog.getText(self.mascot, "Loopy", "Say something to Loopy:")
        if ok and text.strip():
            _log(f"You:   {text.strip()}   (typed)")
            if not self.awake:
                self.wake_up(greet=False)
            self.think(text)

    def _extend_menu(self, menu: QMenu) -> None:
        menu.addAction("Type to Loopy...", self._ask_by_typing)
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
        info = self.listener.last_knock if self.listener else None
        detail = f"peak {info[0]:.3f}, {info[1]:.0f}x the room, sharpness {info[2]:.1f}" if info else f"peak {peak:.3f}"
        _log(f"       [knock! ({detail}) -> can I get a ho yeah!]")
        self._clip_playing = True
        self._clip_id += 1
        if self.speaker:
            self.speaker.hold()    # Loopy's own voice waits until the clip is over
            # Safety net if "finished" never comes; only for this clip, not a later one.
            QTimer.singleShot(10_000, lambda clip=self._clip_id: clip == self._clip_id and self._release_voice())
        if self.listener:
            self.listener.pause()  # don't hear the clip as speech (or as more knocks)
        self._state_before_clip = self.mascot.state()
        self.mascot.set_state(State.SPEAKING)
        self.mascot.show_bubble("Can I get a ho yeah?! \U0001F64C", 3800)
        self._refresh_status()
        self._clip.play()

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
        self._switch_to(models.get_voice(voice_id))

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


def _quit_on_ctrl_c(app: QApplication) -> QTimer:
    """Make Ctrl+C in the terminal quit the app.

    Python only runs signal handlers when it gets control back, which never
    happens while Qt's event loop is idle, so a timer pokes it every 200 ms.
    """
    for sig in (signal.SIGINT, getattr(signal, "SIGBREAK", None)):  # SIGBREAK = Ctrl+Break on Windows
        if sig is not None:
            signal.signal(sig, lambda *_: app.quit())
    wake_timer = QTimer()
    wake_timer.timeout.connect(lambda: None)
    wake_timer.start(200)
    return wake_timer  # caller must keep a reference


def run() -> int:
    # The Windows console can't encode some characters a model may write
    # (curly quotes, accents...); print a "?" instead of crashing.
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(errors="replace")
    # Sound clips: on Windows use its own media engine; Qt's default (FFmpeg)
    # prints decoder chatter into the terminal, cluttering the conversation log.
    if sys.platform == "win32":
        os.environ.setdefault("QT_MEDIA_BACKEND", "windows")
    os.environ.setdefault("QT_LOGGING_RULES", "qt.multimedia*=false")
    app = QApplication(sys.argv)
    mascot = Mascot()
    assistant = Assistant(mascot)
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
