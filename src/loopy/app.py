"""Desktop app: wires the mascot, ears (listener + Whisper), brain (Llama)
and voice together.

Flow:
    asleep --"Hey Loopy"--> awake --(no activity for a while)--> asleep
While awake, everything you say goes to Loopy's brain, which chats back and
asks before doing anything on the computer. "Hey Loopy, how are you?" works
in one go, asleep or awake. Falling asleep ends the conversation.
"""

import signal
import sys
import threading

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication, QInputDialog, QMenu

from loopy import models, settings, wake
from loopy.mascot import Activity, Mascot, State
from loopy.models import Voice

DEFAULT_SLEEP_AFTER = 60  # seconds without activity before Loopy dozes off


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
    _brain_ready = Signal()                  # model finished loading

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
        mascot.speak_requested.connect(self.say)
        mascot.menu_extender = self._extend_menu

        sleep_after = settings.load().get("sleep_after_seconds", DEFAULT_SLEEP_AFTER)
        self._sleep_timer = QTimer(self)
        self._sleep_timer.setSingleShot(True)
        self._sleep_timer.setInterval(int(sleep_after * 1000))
        self._sleep_timer.timeout.connect(self.go_to_sleep)

        # Brain: loading takes ~20 s, so start it now in the background.
        self.brain = None
        if models.llm_installed():
            from loopy.brain import Brain

            self.brain = Brain(models.llm_path())

            def load_brain():
                try:
                    self.brain.load()
                finally:
                    self._brain_ready.emit()

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
            )
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
        if self._downloading:
            status = (f"Downloading {self._downloading.label.split(' (')[0]}'s voice...", Activity.BUSY)
        elif not self.ready:
            status = ("Setup needed: run 'loopy setup'", Activity.IDLE)
        elif self.listener is None:
            status = ("Microphone unavailable", Activity.IDLE)
        elif not self.awake:
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
        else:
            status = ("Awake · just talk", Activity.IDLE)
        m.set_status(*status)

    def _on_brain_ready(self) -> None:
        self._brain_loaded = True
        self._refresh_status()

    # -------------------------------------------------------- awake / asleep

    def wake_up(self, greet: bool = True) -> None:
        self.awake = True
        if self.mascot.state() == State.SLEEPING:
            self.mascot.set_state(State.IDLE)
        if greet:
            self.say("Yes?")  # countdown starts once it finishes speaking
        else:
            self._keep_awake()
        self._refresh_status()

    def go_to_sleep(self) -> None:
        self.awake = False
        self._hearing = False
        self._sleep_timer.stop()
        if self.speaker:
            self.speaker.stop()
        if self.brain:
            self.brain.reset()  # a new wake-up starts a new conversation
        self.mascot.set_state(State.SLEEPING)
        self.mascot.show_bubble("", 0)
        self._refresh_status()

    def _keep_awake(self) -> None:
        """Restart the sleep countdown. Called whenever Loopy goes back to
        idle; while it's listening, thinking or speaking the countdown is
        paused (see _busy), so it can't doze off mid-answer."""
        if self.awake and not self._thinking and not self._speaking:
            self._sleep_timer.start()

    def _busy(self) -> None:
        self._sleep_timer.stop()

    # ------------------------------------------------------------ listening

    def _on_speech_detected(self) -> None:
        if not self.awake:
            return
        self._hearing = True
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
        if self.awake:
            self._busy()
            self._transcribing = True
            self.mascot.set_state(State.WORKING)
        self._refresh_status()
        threading.Thread(target=lambda: self._heard.emit(self.transcriber.transcribe(audio)), daemon=True).start()

    def _on_heard(self, text: str) -> None:
        self._transcribing = False
        self._refresh_status()
        woke, rest = wake.split_wake(text)
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
        out = stream.feed if stream else self._thought.emit
        self._spoken = ""

        def work():
            try:
                turn = self.brain.respond(text, on_text=out)
                if turn.result is not None and not turn.result.ok:
                    out(f" {turn.result.message}")  # e.g. "Chrome isn't open."
            except Exception as exc:  # keep Loopy alive whatever happens
                print(f"[loopy] brain error: {exc}")
                out(" Sorry, my brain got a bit tangled. Can you say that again?")
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

    def _ask_by_typing(self) -> None:
        text, ok = QInputDialog.getText(self.mascot, "Loopy", "Say something to Loopy:")
        if ok:
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
        # Another sentence may have started in the last 300 ms; stay muted then.
        if self.listener and not self._speaking:
            self.listener.resume()

    def shutdown(self) -> None:
        self._sleep_timer.stop()
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
    else:
        assistant.say("Hi! I'm Loopy. Say hey Loopy whenever you want to chat.")
    assistant._keep_awake()
    return app.exec()
