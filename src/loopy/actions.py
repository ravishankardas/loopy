"""What Loopy can do on the computer.

The conversation model picks an action by name from names(); Loopy's code
(not the model) makes sure the user said yes first, then calls run().

Add an app by adding an entry to APPS; "open <app>" and "close <app>"
become available automatically.
"""

import os
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class App:
    name: str                    # display name, e.g. "Chrome"
    key: str                     # used in action names, e.g. "open chrome"
    win_exe: str                 # Windows image name (also its App Paths key)
    mac_name: str                # macOS application name
    linux_cmds: tuple[str, ...]  # Linux executables to try, in order


APPS = [
    App(
        name="Chrome",
        key="chrome",
        win_exe="chrome.exe",
        mac_name="Google Chrome",
        linux_cmds=("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"),
    ),
]


@dataclass(frozen=True)
class Result:
    ok: bool
    message: str


def names() -> list[str]:
    """Every action the model may propose, e.g. ["open chrome", "close chrome"]."""
    return [f"{verb} {app.key}" for app in APPS for verb in ("open", "close")]


def describe(action: str) -> str:
    """How to say an action: "open chrome" -> "open Chrome"."""
    verb, _, key = action.partition(" ")
    app = next((a for a in APPS if a.key == key), None)
    return f"{verb} {app.name}" if app else action


def is_named(action: str, text: str) -> bool:
    """Does `text` say which action it's about (the verb and the app)? A yes
    only counts for a question that said what it was for: "Can I?" must not
    turn a "yes" into opening Chrome."""
    verb, _, key = action.partition(" ")
    app = next((a for a in APPS if a.key == key), None)
    words = text.lower()
    return verb in words and (key in words or (app is not None and app.name.lower() in words))


def run(action: str) -> Result:
    verb, _, key = action.partition(" ")
    app = next((a for a in APPS if a.key == key), None)
    if app is None or verb not in ("open", "close"):
        return Result(False, f"Sorry, I can't {action}.")
    try:
        return _open(app) if verb == "open" else _close_app(app)
    except Exception as exc:  # never let an action crash the assistant
        return Result(False, f"Sorry, something went wrong: {exc}")


def _open(app: App) -> Result:
    try:
        _launch(app)
    except OSError:
        return Result(False, f"Sorry, I couldn't find {app.name} on this computer.")
    return Result(True, f"Opened {app.name}.")


def _close_app(app: App) -> Result:
    if not _is_running(app):
        return Result(False, f"{app.name} isn't open.")
    _close(app)
    return Result(True, f"Closed {app.name}.")


# ----------------------------------------------------- platform specifics

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console flash on Windows


def _launch(app: App) -> None:
    if sys.platform == "win32":
        os.startfile(app.win_exe)  # resolves via the App Paths registry, like Win+R
    elif sys.platform == "darwin":
        if subprocess.run(["open", "-a", app.mac_name]).returncode != 0:
            raise FileNotFoundError(app.mac_name)
    else:
        for cmd in app.linux_cmds:
            try:
                subprocess.Popen([cmd], start_new_session=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except FileNotFoundError:
                continue
        raise FileNotFoundError(app.name)


def _is_running(app: App) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {app.win_exe}", "/NH"],
            capture_output=True, text=True, creationflags=_NO_WINDOW,
        ).stdout
        return app.win_exe.lower() in out.lower()
    if sys.platform == "darwin":
        return subprocess.run(["pgrep", "-xq", app.mac_name]).returncode == 0
    return any(
        subprocess.run(["pgrep", "-x", cmd], capture_output=True).returncode == 0
        for cmd in app.linux_cmds
    )


def _close(app: App, timeout: float = 5.0) -> None:
    """Close politely (like clicking X, so tabs can be restored), then
    force-quit only if it's still running after `timeout` seconds."""
    if sys.platform == "win32":
        polite = ["taskkill", "/IM", app.win_exe]
        force = ["taskkill", "/F", "/IM", app.win_exe]
    elif sys.platform == "darwin":
        polite = ["osascript", "-e", f'quit app "{app.mac_name}"']
        force = ["pkill", "-x", app.mac_name]
    else:
        polite = ["pkill", "-TERM", "-x", "|".join(app.linux_cmds)]
        force = ["pkill", "-KILL", "-x", "|".join(app.linux_cmds)]

    kw = {"capture_output": True, "creationflags": _NO_WINDOW} if sys.platform == "win32" \
        else {"capture_output": True}
    subprocess.run(polite, **kw)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_running(app):
            return
        time.sleep(0.25)
    subprocess.run(force, **kw)
