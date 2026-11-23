"""Read-only facts Loopy can look up: they change nothing on the computer,
so (unlike actions) they don't need the user's permission.

The model must never answer these from memory (a 3B model would just make
up the time); it asks for them with a [check: ...] tag and gets the real
value back from read().

Add one by writing a function that returns a short phrase and listing it in
CHECKS.
"""

import datetime
import sys
from collections.abc import Callable
from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class Check:
    name: str                  # used in the tag, e.g. [check: time]
    about: str                 # tells the model when to use it
    read: Callable[[], str]    # returns a short phrase with the real value
    spoken: str                # how Loopy describes it aloud, e.g. "the time"


def _time() -> str:
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p").lstrip("0")                 # "3:42 PM"


def _date() -> str:
    now = datetime.datetime.now()
    return f"{now:%A}, {now:%B} {now.day}, {now.year}"          # "Wednesday, November 18, 2026"


def _battery() -> str:
    b = psutil.sensors_battery()
    if b is None:
        return "no battery found (this is probably a desktop PC, always plugged in)"
    text = f"{round(b.percent)}%, " + ("plugged in" if b.power_plugged else "on battery")
    if not b.power_plugged and b.secsleft not in (psutil.POWER_TIME_UNKNOWN, psutil.POWER_TIME_UNLIMITED):
        hours, minutes = divmod(b.secsleft // 60, 60)
        text += f", about {hours} hours {minutes} minutes left" if hours else f", about {minutes} minutes left"
    return text


def _system() -> str:
    cpu = psutil.cpu_percent(interval=0.3)
    mem = psutil.virtual_memory()
    return (f"CPU {cpu:.0f}% busy, memory {mem.percent:.0f}% used "
            f"({mem.used / 2**30:.1f} of {mem.total / 2**30:.1f} GB)")


def _disk() -> str:
    """Free space on every drive, e.g. "C drive: 61 GB free of 325 GB; D drive: ..."."""
    drives = [p for p in psutil.disk_partitions(all=False) if "cdrom" not in p.opts]
    return "; ".join(_drive_space(p.mountpoint) for p in drives) or "no drives found"


def _drive_space(mountpoint: str) -> str:
    # "C:\\" -> "C drive" on Windows; "/home" stays "/home" elsewhere.
    label = f"{mountpoint.rstrip(chr(92)).rstrip(':')} drive" if sys.platform == "win32" else mountpoint
    try:
        d = psutil.disk_usage(mountpoint)
    except OSError:  # e.g. an empty card reader
        return f"{label}: not readable"
    return f"{label}: {d.free / 2**30:.0f} GB free of {d.total / 2**30:.0f} GB"


CHECKS = [
    Check("time", "the current time", _time, "the time"),
    Check("date", "today's date and day of the week", _date, "the date"),
    Check("battery", "battery level and whether it's charging", _battery, "your battery level"),
    Check("system", "how busy the computer is (CPU and memory use)", _system, "how busy your computer is"),
    Check("disk", "free space on each drive", _disk, "how much space is free on your drives"),
]


def names() -> list[str]:
    return [c.name for c in CHECKS]


def read(name: str) -> str:
    check = next((c for c in CHECKS if c.name == name), None)
    if check is None:
        return "unknown"
    try:
        return check.read()
    except Exception as exc:  # never let a check crash the assistant
        return f"couldn't read it ({exc})"
