"""User settings, stored as JSON next to the models."""

import json

from loopy.models import DATA_DIR, DEFAULT_VOICE, VOICES, Voice, get_voice

_FILE = DATA_DIR / "settings.json"


def load() -> dict:
    try:
        return json.loads(_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save(**changes) -> None:
    data = load() | changes
    _FILE.parent.mkdir(parents=True, exist_ok=True)
    _FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def selected_voice() -> Voice:
    """The voice the user picked (may not be downloaded yet)."""
    return get_voice(load().get("voice", "")) or DEFAULT_VOICE


def active_voice() -> Voice | None:
    """The voice to speak with: the selected one, else any installed one."""
    chosen = selected_voice()
    if chosen.installed:
        return chosen
    return next((v for v in VOICES if v.installed), None)
