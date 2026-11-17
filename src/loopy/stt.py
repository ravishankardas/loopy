"""Speech-to-text with faster-whisper.

    text = Transcriber(models.stt_path()).transcribe(audio)   # 16 kHz mono float32
"""

import threading
from pathlib import Path

import numpy as np

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
