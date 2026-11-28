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
    """Makes the final text while you pause, so it's ready when you stop.

    The listener sends the audio so far with update() when you pause. That
    pass runs during the silence the listener waits through anyway, and if
    you don't speak again its result is the final text: finish() returns it
    with no second pass. Only the newest audio is transcribed; one pass at a
    time, on a background thread. (The words shown while you talk come from
    Preview, a smaller model.)

        live = LiveTranscription(transcriber)
        live.update(audio_so_far, speech_end)   # from the listener, on a pause
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


def load_streaming_model(model_dir: Path, threads: int = 1):
    """The live-preview model (see models.preview_path), run by sherpa-onnx."""
    import sherpa_onnx

    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(model_dir / "tokens.txt"), encoder=str(model_dir / "encoder.int8.onnx"),
        decoder=str(model_dir / "decoder.int8.onnx"), joiner=str(model_dir / "joiner.int8.onnx"),
        num_threads=threads, sample_rate=SAMPLE_RATE, feature_dim=80, decoding_method="greedy_search")


class Preview:
    """The words so far, shown while you talk, from a streaming model: it
    decodes each new bit of audio as it arrives (~50 ms of work per 160 ms
    of speech, on one core), so words flow in about a second behind the
    voice without slowing the accurate final text. Only a preview: the
    final text (Whisper) replaces it and is what Loopy acts on.

        preview = Preview(lambda: load_streaming_model(models.preview_path()), on_text=show)
        preview.update(audio_so_far)   # from the listener while you talk; only the new part is decoded
        preview.cancel()               # you paused or stopped: start afresh next time
    """

    def __init__(self, load, on_text):
        self._load = load             # () -> recognizer; called on the worker thread (takes ~1 s)
        self._on_text = on_text
        self._cond = threading.Condition()
        self._generation = 0          # bumped by cancel(): a new stream, older results dropped
        self._fed = 0                 # samples of this utterance already queued
        self._queue: list[tuple[int, np.ndarray]] = []
        threading.Thread(target=self._run, name="loopy-preview-stt", daemon=True).start()

    def update(self, audio: np.ndarray) -> None:
        with self._cond:
            if len(audio) > self._fed:
                self._queue.append((self._generation, audio[self._fed:]))
                self._fed = len(audio)
                self._cond.notify_all()

    def cancel(self) -> None:
        with self._cond:
            self._generation += 1
            self._fed = 0
            self._queue.clear()

    def _run(self) -> None:
        try:
            recognizer = self._load()
        except Exception:
            logger.exception("[live preview unavailable; speech recognition still works]")
            return
        stream, stream_generation, shown = None, -1, ""
        while True:
            with self._cond:
                while not self._queue:
                    self._cond.wait()
                generation, samples = self._queue.pop(0)
            try:
                if stream is None or generation != stream_generation:      # a new utterance
                    stream, stream_generation, shown = recognizer.create_stream(), generation, ""
                stream.accept_waveform(SAMPLE_RATE, samples)
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)
                text = recognizer.get_result(stream).strip()
            except Exception:
                logger.exception("[live preview failed; the final text is unaffected]")
                continue
            with self._cond:
                current = generation == self._generation
            if current and text and text != shown:
                shown = text
                self._on_text(text[0].upper() + text[1:])
