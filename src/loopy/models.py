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


# Conversation model: Llama 3.2 3B Instruct, 4-bit (~1.9 GB on disk, ~2.4 GB RAM).
LLM_NAME = "Llama 3.2 3B Instruct (Q4_K_M)"
LLM_FILE = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
_LLM_URL = f"https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/{LLM_FILE}"


def llm_path() -> Path:
    return DATA_DIR / "llm" / LLM_FILE


def llm_installed() -> bool:
    return llm_path().is_file()


def install_llm(force: bool = False, progress: Progress | None = None) -> None:
    if llm_installed() and not force:
        return
    _download(_LLM_URL, llm_path(), progress)


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
