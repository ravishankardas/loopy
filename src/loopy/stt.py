"""Speech-to-text with faster-whisper.

    text = Transcriber(models.stt_path()).transcribe(audio)   # 16 kHz mono float32
"""

import threading
from pathlib import Path

import numpy as np
from loguru import logger

SAMPLE_RATE = 16_000

# Things Whisper tends to "hear" in noise or breathing.
_HALLUCINATIONS = {"", "you", "thank you", "thanks for watching", "bye", "uh", "um", "hmm"}


class Transcriber:
    def __init__(self, model_dir: Path):
        self._model_dir = Path(model_dir)
        self._model = None
        self._lock = threading.Lock()

    def load(self):
        """Load the model (a second or two); called automatically."""
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(str(self._model_dir), device="cpu", compute_type="int8")
            return self._model

    def transcribe(self, audio: np.ndarray) -> str:
        if len(audio) == 0:
            return ""
        model = self.load()
        with self._lock:  # one transcription at a time
            segments, _info = model.transcribe(
                audio,
                language="en",
                beam_size=1,                     # greedy: much faster, fine for short commands
                without_timestamps=True,
                condition_on_previous_text=False,
                hotwords="Loopy",                # nudge it to spell the name consistently
            )
            text = " ".join(s.text.strip() for s in segments if s.no_speech_prob < 0.6).strip()
        if text.lower().strip(" .!?,") in _HALLUCINATIONS:
            return ""
        return text


class LiveTranscription:
    """Transcribes while you talk, so the text is (nearly) ready when you stop.

    The listener sends the audio so far with update(): every so often while
    you speak (partial text for the speech bubble) and when you pause. The
    pause pass runs during the silence the listener waits through anyway, and
    if you don't speak again its result is the final text: finish() returns
    it with no second pass. Only the newest audio is transcribed; one pass at
    a time, on a background thread.

        live = LiveTranscription(transcriber, on_partial=show)
        live.update(audio_so_far, speech_end)   # from the listener, often
        text = live.finish(whole_utterance)     # blocks until the text is ready
    """

    def __init__(self, transcriber: Transcriber, on_partial=None):
        self._transcriber = transcriber
        self._on_partial = on_partial
        self._cond = threading.Condition()
        self._utterance = 0           # which utterance the state below belongs to
        self._next: tuple[int, np.ndarray] | None = None   # (utterance, audio) waiting for a pass
        self._busy = False
        self._done: tuple[int, int, str] | None = None     # (utterance, samples covered, text)
        self._speech_end = 0          # samples: where the speech in this utterance ends so far
        self.reused = 0               # finished utterances that needed no extra pass (for tests/benchmarks)
        threading.Thread(target=self._run, name="loopy-live-stt", daemon=True).start()

    def update(self, audio: np.ndarray, speech_end: int) -> None:
        """Transcribe this (the utterance so far) when the worker is free; an
        older snapshot still waiting is replaced, as this one covers more."""
        with self._cond:
            self._next = (self._utterance, audio)
            self._speech_end = speech_end
            self._cond.notify_all()

    def cancel(self) -> None:
        """The utterance was dropped (a cough): forget it."""
        with self._cond:
            self._new_utterance()

    def finish(self, audio: np.ndarray) -> str:
        """The final text for the whole utterance."""
        with self._cond:
            while self._next is not None or self._busy:   # a pass on newer audio may be coming
                self._cond.wait()
            done, speech_end, utterance = self._done, self._speech_end, self._utterance
            self._new_utterance()
        if done is not None and done[0] == utterance and done[1] >= speech_end > 0:
            self.reused += 1
            return done[2]             # that pass already heard everything said
        return self._transcriber.transcribe(audio)

    def _new_utterance(self) -> None:
        self._utterance += 1
        self._next, self._done, self._speech_end = None, None, 0

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._next is None:
                    self._cond.wait()
                utterance, audio = self._next
                self._next, self._busy = None, True
            try:
                text = self._transcriber.transcribe(audio)
            except Exception:
                logger.exception("[live transcription failed; the whole utterance is transcribed at the end]")
                text = None
            with self._cond:
                self._busy = False
                current = utterance == self._utterance
                if current and text is not None:
                    self._done = (utterance, len(audio), text)
                self._cond.notify_all()
            if current and text and self._on_partial:
                self._on_partial(text)
