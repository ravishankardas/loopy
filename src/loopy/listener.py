"""Always-on microphone listening, split into utterances.

A tiny voice-activity model (Silero, bundled with faster-whisper) checks every
32 ms of audio for speech. When someone talks, audio is collected until they
pause, then handed to `on_utterance`. Silence and background noise never
reach Whisper, so idle CPU use stays near zero.

    listener = Listener(on_utterance=handle_audio)
    listener.start()
    listener.pause()    # e.g. while Loopy is talking, so it can't hear itself
    listener.resume()
    listener.stop()
"""

import collections
import os
import queue
import threading
from typing import Callable

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000
FRAME = 512                  # 32 ms: the chunk size the VAD model expects
START_PROB, END_PROB = 0.5, 0.35
START_FRAMES = 3             # ~100 ms of voice before it counts as speech
END_FRAMES = 22              # ~700 ms of quiet ends an utterance
PREROLL_FRAMES = 15          # keep ~480 ms from before speech was detected
MAX_SECONDS = 15
MIN_SECONDS = 0.4


class VoiceDetector:
    """Silero VAD, streaming: call with one 512-sample frame, get P(speech)."""

    _CONTEXT = 64

    def __init__(self):
        import onnxruntime
        from faster_whisper.utils import get_assets_path

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self._session = onnxruntime.InferenceSession(
            os.path.join(get_assets_path(), "silero_vad_v6.onnx"),
            providers=["CPUExecutionProvider"], sess_options=opts,
        )
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(self._CONTEXT, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self._context, frame])[None, :]
        prob, self._h, self._c = self._session.run(None, {"input": x, "h": self._h, "c": self._c})
        self._context = frame[-self._CONTEXT:]
        return float(prob.reshape(-1)[0])


class Listener:
    def __init__(
        self,
        on_utterance: Callable[[np.ndarray], None],
        on_speech_start: Callable[[], None] | None = None,
        on_level: Callable[[float], None] | None = None,
    ):
        """Callbacks run on the listener thread. `on_utterance` gets mono
        float32 audio at 16 kHz, or an empty array if the speech turned out
        too short to be a command (so callers can undo on_speech_start)."""
        self._on_utterance = on_utterance
        self._on_speech_start = on_speech_start
        self._on_level = on_level
        self._vad: VoiceDetector | None = None
        self._paused = threading.Event()
        self._frames: queue.Queue[np.ndarray | None] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._thread: threading.Thread | None = None
        self._reset()

    # -------------------------------------------------------------- control

    def start(self) -> None:
        """Open the microphone. Raises if there is no usable input device."""
        self._vad = VoiceDetector()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=FRAME, callback=self._callback,
        )
        self._thread = threading.Thread(target=self._run, name="loopy-listen", daemon=True)
        self._thread.start()
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._thread is not None:
            self._frames.put(None)
            self._thread.join(2)
            self._thread = None

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    # ------------------------------------------------------------- pipeline

    def _callback(self, indata, _frames, _time, _status) -> None:
        self._frames.put(indata[:, 0].copy())  # keep the audio thread light

    def _run(self) -> None:
        pending = np.zeros(0, dtype=np.float32)
        while (chunk := self._frames.get()) is not None:
            pending = np.concatenate([pending, chunk])
            while len(pending) >= FRAME:
                self.feed(pending[:FRAME])
                pending = pending[FRAME:]

    def _reset(self) -> None:
        self._speaking = False
        self._voiced_run = 0
        self._quiet_run = 0
        self._voiced_total = 0
        self._preroll: collections.deque = collections.deque(maxlen=PREROLL_FRAMES)
        self._utterance: list[np.ndarray] = []
        if self._vad is not None:
            self._vad.reset()

    def feed(self, frame: np.ndarray) -> None:
        """Process one 512-sample frame (public so tests can drive it)."""
        if self._paused.is_set():
            if self._speaking or self._preroll:
                self._reset()  # drop anything half-heard
            return

        prob = self._vad(frame)

        if not self._speaking:
            self._preroll.append(frame)
            self._voiced_run = self._voiced_run + 1 if prob >= START_PROB else 0
            if self._voiced_run >= START_FRAMES:
                self._speaking = True
                self._utterance = list(self._preroll)
                self._voiced_total = self._voiced_run
                self._quiet_run = 0
                if self._on_speech_start:
                    self._on_speech_start()
            return

        self._utterance.append(frame)
        if self._on_level:
            self._on_level(min(1.0, float(np.sqrt(np.mean(frame**2))) * 12))
        if prob < END_PROB:
            self._quiet_run += 1
        else:
            self._quiet_run = 0
            self._voiced_total += 1

        too_long = len(self._utterance) * FRAME >= MAX_SECONDS * SAMPLE_RATE
        if self._quiet_run >= END_FRAMES or too_long:
            audio = np.concatenate(self._utterance)
            if self._voiced_total * FRAME < MIN_SECONDS * SAMPLE_RATE:
                audio = np.zeros(0, dtype=np.float32)  # a cough or click, not a command
            self._reset()
            if self._on_level:
                self._on_level(0.0)
            self._on_utterance(audio)
