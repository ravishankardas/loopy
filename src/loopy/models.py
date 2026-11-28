"""Model files that are too big to ship in the package.

They are downloaded (by `loopy setup`, or from the Voice menu) into the
user's app-data folder, e.g. %LOCALAPPDATA%\\loopy on Windows or
~/.local/share/loopy on Linux.
"""

import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_data_path

DATA_DIR = user_data_path("loopy", appauthor=False)

_PIPER_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/{family}/{lang}/{speaker}/{quality}/{file}"

Progress = Callable[[int, int], None]  # (bytes_done, bytes_total)


@dataclass(frozen=True)
class Voice:
    id: str      # Piper voice id, e.g. "en_US-lessac-medium"
    label: str   # shown in menus

    @property
    def path(self) -> Path:
        return DATA_DIR / "voices" / f"{self.id}.onnx"

    @property
    def files(self) -> list[Path]:
        return [self.path, self.path.with_name(self.path.name + ".json")]

    @property
    def installed(self) -> bool:
        return all(f.is_file() for f in self.files)

    def url(self, file: str) -> str:
        lang, speaker, quality = self.id.split("-")
        return _PIPER_URL.format(
            family=lang.split("_")[0], lang=lang, speaker=speaker, quality=quality, file=file
        )


VOICES = [
    Voice("en_US-lessac-medium", "Lessac (US, female)"),
    Voice("en_US-amy-medium", "Amy (US, female)"),
    Voice("en_US-kristin-medium", "Kristin (US, female)"),
    Voice("en_US-ryan-medium", "Ryan (US, male)"),
    Voice("en_US-joe-medium", "Joe (US, male)"),
    Voice("en_US-norman-medium", "Norman (US, male)"),
    Voice("en_GB-cori-medium", "Cori (UK, female)"),
    Voice("en_GB-jenny_dioco-medium", "Jenny (UK, female)"),
    Voice("en_GB-alan-medium", "Alan (UK, male)"),
    Voice("en_GB-northern_english_male-medium", "Northern English (UK, male)"),
]
DEFAULT_VOICE = VOICES[0]


def get_voice(voice_id: str) -> Voice | None:
    return next((v for v in VOICES if v.id == voice_id), None)


@dataclass(frozen=True)
class LLM:
    """A conversation model: a 4-bit GGUF file run by llama.cpp."""
    id: str            # used in settings and on the command line
    name: str          # shown to people
    repo: str          # Hugging Face repo with the GGUF file
    file: str
    size: str          # download size, for `loopy setup`
    system_role: bool = True   # False: the chat format has no "system" role (Gemma)

    @property
    def path(self) -> Path:
        return DATA_DIR / "llm" / self.file

    @property
    def installed(self) -> bool:
        return self.path.is_file()

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{self.file}"


LLMS = [
    LLM("llama-3.2-3b", "Llama 3.2 3B Instruct (Q4_K_M)", "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf", "1.9 GB"),
    LLM("gemma-3-4b", "Gemma 3 4B Instruct (Q4_K_M)", "bartowski/google_gemma-3-4b-it-GGUF",
        "google_gemma-3-4b-it-Q4_K_M.gguf", "2.5 GB", system_role=False),
    # Google's quantization-aware-trained build: made for 4 bits, and Q4_0 is
    # the format llama.cpp's fastest CPU kernels are written for.
    LLM("gemma-3-4b-qat", "Gemma 3 4B Instruct QAT (Q4_0)", "bartowski/google_gemma-3-4b-it-qat-GGUF",
        "google_gemma-3-4b-it-qat-Q4_0.gguf", "2.4 GB", system_role=False),
]
DEFAULT_LLM = LLMS[0]


def get_llm(llm_id: str | None) -> LLM | None:
    return next((m for m in LLMS if m.id == llm_id), None)


def llm_for_file(path: Path) -> LLM | None:
    return next((m for m in LLMS if m.file == Path(path).name), None)


def active_llm() -> LLM:
    """The model chosen with `loopy setup --model` (settings "llm"), or the default."""
    from loopy import settings  # settings imports this module
    return get_llm(settings.load().get("llm")) or DEFAULT_LLM


def llm_path() -> Path:
    return active_llm().path


def llm_installed() -> bool:
    return active_llm().installed


def install_llm(llm: LLM | None = None, force: bool = False, progress: Progress | None = None) -> None:
    llm = llm or active_llm()
    if llm.installed and not force:
        return
    _download(llm.url, llm.path, progress)


# Speech-to-text: faster-whisper "base.en", int8 on CPU (~145 MB on disk).
STT_MODEL = "base.en"


def stt_path() -> Path:
    return DATA_DIR / "stt" / STT_MODEL


def stt_installed() -> bool:
    return (stt_path() / "model.bin").is_file()


def install_stt(force: bool = False) -> None:
    """Download the Whisper model (shows its own progress bars)."""
    if stt_installed() and not force:
        return
    from faster_whisper import download_model

    download_model(STT_MODEL, output_dir=str(stt_path()))


# The live preview of the words while you talk: a streaming model (NVIDIA NeMo
# FastConformer transducer, 80 ms look-ahead, int8, ~137 MB) run by sherpa-onnx.
# It decodes 160 ms chunks as they arrive, on ~30% of one core, so it doesn't
# slow the accurate final text (Whisper). Only shown; never acted on.
PREVIEW_NAME = "NeMo FastConformer streaming (80 ms)"
_PREVIEW_ARCHIVE = "sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms-int8"
_PREVIEW_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{_PREVIEW_ARCHIVE}.tar.bz2"
PREVIEW_FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")


def preview_path() -> Path:
    return DATA_DIR / "stt" / "preview-nemo-80ms"


def preview_installed() -> bool:
    return all((preview_path() / name).is_file() for name in PREVIEW_FILES)


def install_preview(force: bool = False, progress: Progress | None = None) -> None:
    """Download the release archive and keep only the model files."""
    if preview_installed() and not force:
        return
    import tarfile

    archive = DATA_DIR / "stt" / f"{_PREVIEW_ARCHIVE}.tar.bz2"
    _download(_PREVIEW_URL, archive, progress)
    try:
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                name = Path(member.name).name
                if name in PREVIEW_FILES and member.isfile():
                    source = tar.extractfile(member)
                    assert source is not None
                    dest = preview_path() / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(source.read())
    finally:
        archive.unlink(missing_ok=True)


def install_voice(voice: Voice, force: bool = False, progress: Progress | None = None) -> None:
    for dest in voice.files:
        if dest.is_file() and not force:
            continue
        _download(voice.url(dest.name), dest, progress)


def _download(url: str, dest: Path, progress: Progress | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "loopy-setup"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done, last = 0, -1
            while chunk := resp.read(1 << 16):
                out.write(chunk)
                done += len(chunk)
                # Report only when the visible percent (or MB) changes.
                step = done * 100 // total if total else done >> 20
                if progress and step != last:
                    last = step
                    progress(done, total)
        tmp.replace(dest)  # only appears under its real name once complete
    finally:
        tmp.unlink(missing_ok=True)
