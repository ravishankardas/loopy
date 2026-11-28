"""Text-to-speech with Piper, played on the default output device.

    speaker = Speaker(models.voice_path())
    speaker.speak("Hello")                          # blocking
    speaker.say("Hello", on_start=..., on_done=...) # queued, background thread

    stream = speaker.stream(on_start=..., on_done=...)  # text arriving bit by bit,
    stream.feed("Sure, ")                                # e.g. from a language model
    stream.feed("want me to ...")
    stream.close()

    speaker.stop()                                  # cut off current + queued speech

Streaming: speech starts as soon as the first few words exist. After that,
each time the voice is ready for more it takes everything that has arrived
since (cut at the last punctuation if there is one), so it talks while the
text is still being written, in chunks big enough to sound natural. One
utterance plays through a single audio stream, and the next chunk is
synthesized while the current one is still playing, so there are no gaps.
"""

import queue
import re
import threading
from collections.abc import Callable
from pathlib import Path

import sounddevice as sd
from loguru import logger
from piper import PiperVoice

TextCallback = Callable[[str], None] | None
Callback = Callable[[], None] | None

import time

MIN_CHUNK_WORDS = 3       # never speak fewer words than this (unless it's the end)
FIRST_PUNCT_WAIT = 0.5    # first chunk: once 3 words exist, wait this long for a , or .
PUNCT_WAIT = 0.2          # later chunks: same, but shorter than OUTPUT_LATENCY so the
                          # audio already queued covers the wait (no gap)
OUTPUT_LATENCY = 0.4      # seconds of audio buffered ahead; hides synthesis time
_BOUNDARY = re.compile(r"[.!?;:,](?=\s)")


class SpeechStream:
    """Text for one utterance, fed in pieces from any thread."""

    def __init__(self, on_start: TextCallback = None, on_chunk: TextCallback = None,
                 on_done: Callback = None):
        self.on_start = on_start    # (first chunk) when audio is about to begin
        self.on_chunk = on_chunk    # (chunk) each piece as it's spoken, incl. the first
        self.on_done = on_done      # when playback ends (only if it started)
        self._buf = ""
        self._closed = False
        self._cancelled = False
        self._cond = threading.Condition()
        self.first_audio_at: float | None = None   # time.monotonic() when sound first went out

    def feed(self, text: str) -> None:
        with self._cond:
            self._buf += text
            self._cond.notify()

    def close(self) -> None:
        """No more text is coming; the rest will still be spoken."""
        with self._cond:
            self._closed = True
            self._cond.notify()

    def cancel(self) -> None:
        with self._cond:
            self._cancelled = True
            self._cond.notify()

    def take(self, first: bool) -> str | None:
        """Block until there's something worth saying; None when finished."""
        since = time.monotonic()
        wait_for_punct = FIRST_PUNCT_WAIT if first else PUNCT_WAIT
        with self._cond:
            while True:
                if self._cancelled:
                    return None
                # Wait briefly for punctuation so phrases aren't split mid-way.
                # For the first chunk the clock starts once 3 words exist.
                if first and len(self._buf.split()) < MIN_CHUNK_WORDS:
                    since = time.monotonic()
                patient = time.monotonic() - since < wait_for_punct
                cut = self._cut(allow_mid_phrase=not patient)
                if cut:
                    chunk, self._buf = self._buf[:cut], self._buf[cut:]
                    if chunk.strip():
                        return chunk.strip()
                    continue
                if self._closed:
                    return None
                self._cond.wait(0.03)

    def _cut(self, allow_mid_phrase: bool) -> int:
        """How much of the buffer to speak now (0 = wait for more)."""
        buf = self._buf
        if self._closed:
            return len(buf)
        # Prefer the last punctuation that leaves a chunk of at least 3 words.
        for m in reversed(list(_BOUNDARY.finditer(buf))):
            if len(buf[:m.end()].split()) >= MIN_CHUNK_WORDS:
                return m.end()
            break  # earlier boundaries only make it shorter
        if not allow_mid_phrase:
            return 0
        # No usable punctuation: take whole words (ones followed by a space).
        last_space = buf.rfind(" ")
        if last_space <= 0:
            return 0
        return last_space if len(buf[:last_space].split()) >= MIN_CHUNK_WORDS else 0


class Speaker:
    def __init__(self, model_path: Path):
        self._model_path = Path(model_path)
        self._voice: PiperVoice | None = None
        self._load_lock = threading.Lock()
        self._stop = threading.Event()
        self._queue: queue.Queue[SpeechStream | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._current: SpeechStream | None = None
        self.underflows = 0  # times playback ran dry mid-utterance (audible gaps)
        self._free = threading.Event()  # cleared by hold(): speech waits instead of playing
        self._free.set()

    def set_model(self, model_path: Path) -> None:
        """Switch voices; the new one loads on the next utterance."""
        with self._load_lock:
            self._model_path = Path(model_path)
            self._voice = None

    def load(self) -> PiperVoice:
        """Load the voice model (about a second); called automatically."""
        with self._load_lock:
            if self._voice is None:
                self._voice = PiperVoice.load(self._model_path)
            return self._voice

    # ------------------------------------------------------------------ API

    def speak(self, text: str) -> None:
        """Speak `text`, returning when playback ends."""
        s = SpeechStream()
        s.feed(text)
        s.close()
        self._play(s)

    def say(self, text: str, on_start: TextCallback = None, on_done: Callback = None,
            on_chunk: TextCallback = None) -> None:
        """Queue complete `text` to be spoken on the background thread."""
        s = self.stream(on_start=on_start, on_chunk=on_chunk, on_done=on_done)
        s.feed(text)
        s.close()

    def stream(self, on_start: TextCallback = None, on_chunk: TextCallback = None,
               on_done: Callback = None) -> SpeechStream:
        """Queue an utterance whose text will arrive bit by bit."""
        self._ensure_worker()
        s = SpeechStream(on_start, on_chunk, on_done)
        self._queue.put(s)
        return s

    def hold(self) -> None:
        """Keep quiet (e.g. while a sound clip plays): nothing new starts and
        speech in progress pauses before its next piece. Nothing is lost:
        it carries on after release()."""
        self._free.clear()

    def release(self) -> None:
        self._free.set()

    @property
    def held(self) -> bool:
        return not self._free.is_set()

    def stop(self) -> None:
        """Cut off what's playing and drop everything queued."""
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                item.cancel()
        self._stop.set()
        if self._current is not None:
            self._current.cancel()

    def close(self, timeout: float = 2.0) -> None:
        """Stop and shut the worker down (call before exiting, so audio isn't
        still playing while Python tears down PortAudio)."""
        self.stop()
        if self._worker is not None:
            self._queue.put(None)
            self._worker.join(timeout)
            self._worker = None

    # ------------------------------------------------------------ internals

    def _ensure_worker(self) -> None:
        if self._worker is None:
            self._worker = threading.Thread(target=self._run, name="loopy-tts", daemon=True)
            self._worker.start()

    def _run(self) -> None:
        self.load()  # warm up before the first request arrives
        while True:
            item = self._queue.get()
            if item is None:  # close() sentinel
                return
            self._current = item
            try:
                self._play(item)
            except Exception as exc:  # keep the worker alive
                logger.exception(f"[speech failed: {exc}]")
            finally:
                self._current = None

    def _play(self, s: SpeechStream) -> None:
        voice = self.load()
        self._stop.clear()
        out = None
        started = False
        try:
            first = True
            while (text := s.take(first)) is not None:
                first = False
                while not self._free.wait(0.05):   # held (a clip is playing): wait
                    if self._stop.is_set() or s._cancelled:
                        break
                if self._stop.is_set():
                    break
                if not started:
                    started = True
                    if s.on_start:
                        s.on_start(text)
                if s.on_chunk:
                    s.on_chunk(text)
                for chunk in voice.synthesize(text):
                    if self._stop.is_set():
                        break
                    if out is None:
                        out = sd.OutputStream(
                            samplerate=chunk.sample_rate, channels=chunk.sample_channels,
                            dtype="int16", latency=OUTPUT_LATENCY,
                        )
                        out.start()
                    audio = chunk.audio_int16_array.reshape(-1, chunk.sample_channels)
                    block = chunk.sample_rate // 10  # 100 ms, so stop() is snappy
                    for i in range(0, len(audio), block):
                        if self._stop.is_set():
                            break
                        if s.first_audio_at is None:
                            s.first_audio_at = time.monotonic()
                        if out.write(audio[i:i + block]):  # True = it had run dry
                            self.underflows += 1
        finally:
            if out is not None:
                if self._stop.is_set():
                    out.abort()  # drop buffered audio
                else:
                    out.stop()   # let buffered audio finish
                out.close()
            if started and s.on_done:
                s.on_done()
