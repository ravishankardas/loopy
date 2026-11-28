"""Always-on microphone listening, split into utterances.

A tiny voice-activity model (Silero, bundled with faster-whisper) checks every
32 ms of audio for speech. When someone talks, audio is collected until they
pause, then handed to `on_utterance`. Silence and background noise never
reach speech-to-text, so idle CPU use stays near zero.

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
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
from loguru import logger

SAMPLE_RATE = 16_000
FRAME = 512                  # 32 ms: the chunk size the VAD model expects
START_PROB, END_PROB = 0.5, 0.35
START_FRAMES = 3             # ~100 ms of voice before it counts as speech
END_FRAMES = 22              # ~700 ms of quiet ends an utterance
PREROLL_FRAMES = 15          # keep ~480 ms from before speech was detected
MAX_SECONDS = 15
MIN_SECONDS = 0.4
PAUSE_FRAMES = 6             # ~200 ms of quiet: probably done, so transcribe what was said (see on_partial)

# Knock detection: hitting the laptop makes a spike in the mic that is short
# (a few ms), sharp and much louder than the room. Same idea as the first two
# filters of github.com/NithinVarma50/spankurlaptop: check short blocks (a
# 32 ms frame would dilute a 5 ms spike), compare with a rolling 2 s
# background level, and require a spike shape rather than a steady sound.
KNOCK_BLOCK = 128            # samples per block (8 ms at 16 kHz)
# Mic levels differ a lot between laptops (noise suppression, mic volume). On
# the laptop this was tuned on, firm hits peaked at 0.09-0.18 and were
# 112-464x louder than the very quiet room; false alarms (CPU-load glitches,
# Loopy's own voice ending) peaked at 0.01-0.11.
KNOCK_ABOVE_NOISE = 40.0     # block must be this many times louder than the room
KNOCK_SHARPNESS = 2.0        # peak / average level: a spike (steady sounds are ~1.4)
KNOCK_PEAK = 0.05            # quietest peak that counts
KNOCK_GLITCH_SKIP = 0.3      # seconds to ignore after the mic dropped audio (busy CPU -> clicks)
KNOCK_RESUME_SKIP = 1.0      # seconds to ignore after the mic comes back on (Loopy's voice/clip ending)
KNOCK_BACKGROUND_SECONDS = 2.0
KNOCK_COOLDOWN = 2.0         # seconds; one hit triggers once


@dataclass(frozen=True)
class KnockReading:
    """How one frame scored on the knock tests (see Listener(on_sound=...))."""
    peak: float
    above_noise: float      # how many times louder than the room
    sharpness: float        # peak / average level
    speech_prob: float
    knocked: bool
    learning: bool = False  # still learning the room's level (first second)
    cooling_down: bool = False  # right after another knock


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
        on_knock: Callable[[float], None] | None = None,
        knock_peak: float = KNOCK_PEAK,
        knock_above_noise: float = KNOCK_ABOVE_NOISE,
        on_sound: Callable[[KnockReading], None] | None = None,
        on_partial: Callable[[np.ndarray, int], None] | None = None,
        on_progress: Callable[[np.ndarray], None] | None = None,
        progress_every: float = 0.5,
    ):
        """Callbacks run on the listener thread. `on_utterance` gets mono
        float32 audio at 16 kHz, or an empty array if the speech turned out
        too short to be a command (so callers can undo on_speech_start).
        `on_knock` gets the peak level when someone hits the laptop.
        `on_partial` gets the utterance so far and where its speech ends (in
        samples) when the user pauses (PAUSE_FRAMES of quiet), so the final
        text can be made during the silence the listener waits for; the last
        call before on_utterance always has all the speech.
        `on_progress` gets the utterance so far every `progress_every`
        seconds while the user speaks, for a live preview of the words."""
        self._on_utterance = on_utterance
        self._on_partial = on_partial
        self._on_progress = on_progress
        self._progress_every = max(1, int(progress_every * SAMPLE_RATE / FRAME))
        self._on_speech_start = on_speech_start
        self._on_level = on_level
        self._on_knock = on_knock
        self._on_sound = on_sound        # every frame's knock reading (for `loopy knock-test`)
        self.knock_peak = knock_peak
        self.knock_above_noise = knock_above_noise   # the main sensitivity knob
        # Background: the room's level over the last 2 s (per 8 ms block).
        self._background_size = int(KNOCK_BACKGROUND_SECONDS * SAMPLE_RATE / KNOCK_BLOCK)
        self._background: collections.deque[float] = collections.deque(maxlen=self._background_size)
        self._last_knock = -1e9
        self._frames_seen = 0
        self._glitch = False             # set by the audio thread when input was dropped
        self._knock_skip = 0             # frames to ignore before knocks count again
        self._was_paused = False
        self.last_knock: tuple[float, float, float] | None = None  # (peak, x room, sharpness)
        self.knocks_enabled = True       # off e.g. while the model loads (CPU flat out -> clicks)
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

    def enable_knocks(self, after: float = 0.0) -> None:
        """Start reacting to knocks, `after` seconds from now."""
        self._knock_skip = max(self._knock_skip, int(after * SAMPLE_RATE / FRAME))
        self.knocks_enabled = True

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    # ------------------------------------------------------------- pipeline

    def _callback(self, indata, _frames, _time, status) -> None:
        if status and status.input_overflow:
            self._glitch = True  # audio was dropped (busy CPU): the gap sounds like a click
        self._frames.put(indata[:, 0].copy())  # keep the audio thread light

    def _run(self) -> None:
        pending = np.zeros(0, dtype=np.float32)
        while (chunk := self._frames.get()) is not None:
            pending = np.concatenate([pending, chunk])
            while len(pending) >= FRAME:
                try:
                    self.feed(pending[:FRAME])
                except Exception as exc:
                    # Never let one bad frame kill the thread: Loopy would go deaf.
                    logger.exception(f"[listener error, still listening: {exc!r}]")
                    self._reset()
                pending = pending[FRAME:]

    def _reset(self) -> None:
        self._speaking = False
        self._voiced_run = 0
        self._quiet_run = 0
        self._voiced_total = 0
        self._speech_end = 0             # frames: the last voiced frame of the utterance so far
        self._preroll: collections.deque = collections.deque(maxlen=PREROLL_FRAMES)
        self._utterance: list[np.ndarray] = []
        if self._vad is not None:
            self._vad.reset()

    def knock_score(self, frame: np.ndarray) -> tuple[float, float, float]:
        """For the most spiky 8 ms block of a frame: (peak, how many times
        louder than the room's background, sharpness = peak / average)."""
        best = (0.0, 0.0, 0.0)
        background = self._background_level()
        for block in frame.reshape(-1, KNOCK_BLOCK):
            peak = float(np.max(np.abs(block)))
            rms = float(np.sqrt(np.mean(block**2))) + 1e-9
            score = (peak, rms / background, peak / rms)
            if score[1] > best[1]:
                best = score
        return best

    def _background_level(self) -> float:
        if not self._background:
            return 1e-3
        return max(float(np.median(self._background)), 1e-4)  # median: a few loud blocks don't skew it

    def _check_knock(self, frame: np.ndarray, speech_prob: float) -> None:
        peak, above_noise, sharpness = self.knock_score(frame)
        self._frames_seen += 1
        now = self._frames_seen * FRAME / SAMPLE_RATE
        if self._glitch:
            self._glitch = False
            self._knock_skip = max(self._knock_skip, int(KNOCK_GLITCH_SKIP * SAMPLE_RATE / FRAME))
        skipping = self._knock_skip > 0
        if skipping:
            self._knock_skip -= 1
        learned = len(self._background) >= self._background_size // 2   # ~1 s of room first
        is_knock = (
            self.knocks_enabled
            and learned
            and not skipping
            and peak >= self.knock_peak
            and above_noise >= self.knock_above_noise
            and sharpness >= KNOCK_SHARPNESS
            and speech_prob < START_PROB           # a shout isn't a knock
            and now - self._last_knock >= KNOCK_COOLDOWN
        )
        if self._on_sound:
            self._on_sound(KnockReading(peak, above_noise, sharpness, speech_prob, is_knock,
                                        learning=not learned, cooling_down=not is_knock and
                                        now - self._last_knock < KNOCK_COOLDOWN))
        if is_knock:
            self._last_knock = now
            self.last_knock = (peak, above_noise, sharpness)
            if self._on_knock:
                self._on_knock(peak)
        elif above_noise < self.knock_above_noise:
            # Only ordinary blocks teach the background, never the spikes.
            for block in frame.reshape(-1, KNOCK_BLOCK):
                self._background.append(float(np.sqrt(np.mean(block**2))))

    def feed(self, frame: np.ndarray) -> None:
        """Process one 512-sample frame (public so tests can drive it)."""
        if self._paused.is_set():
            self._was_paused = True
            if self._speaking or self._preroll:
                self._reset()  # drop anything half-heard
            return
        if self._was_paused:
            # Just back on after Loopy's voice or a clip: its tail end (or the
            # speaker switching off) can click, so don't count knocks yet.
            self._was_paused = False
            self._knock_skip = max(self._knock_skip, int(KNOCK_RESUME_SKIP * SAMPLE_RATE / FRAME))

        assert self._vad is not None, "start() loads the voice detector first"
        prob = self._vad(frame)
        self._check_knock(frame, prob)

        if not self._speaking:
            self._preroll.append(frame)
            self._voiced_run = self._voiced_run + 1 if prob >= START_PROB else 0
            if self._voiced_run >= START_FRAMES:
                self._speaking = True
                self._utterance = list(self._preroll)
                self._speech_end = len(self._utterance)
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
            self._speech_end = len(self._utterance)

        if self._on_partial and self._quiet_run == PAUSE_FRAMES:
            self._on_partial(np.concatenate(self._utterance), self._speech_end * FRAME)
        elif self._on_progress and self._quiet_run == 0 and len(self._utterance) % self._progress_every == 0:
            self._on_progress(np.concatenate(self._utterance))

        too_long = len(self._utterance) * FRAME >= MAX_SECONDS * SAMPLE_RATE
        if self._quiet_run >= END_FRAMES or too_long:
            audio = np.concatenate(self._utterance)
            if self._voiced_total * FRAME < MIN_SECONDS * SAMPLE_RATE:
                audio = np.zeros(0, dtype=np.float32)  # a cough or click, not a command
            self._reset()
            if self._on_level:
                self._on_level(0.0)
            self._on_utterance(audio)
