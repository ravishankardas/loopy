# Loopy

[![CI](https://github.com/ravishankardas/loopy/actions/workflows/ci.yml/badge.svg)](https://github.com/ravishankardas/loopy/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Ruff](https://img.shields.io/badge/lint-ruff-261230)
![Mypy](https://img.shields.io/badge/types-mypy-2a6db2)

A voice assistant that lives on your desktop as a little paperclip. Say **"Hey Loopy"** and talk to it. It runs **entirely on your own computer**: no cloud, no API keys, no GPU.

## Features

- **Talks like a person:** Llama 3.2 3B, with replies spoken while they're still being written.
- **Always asks before acting:** enforced in code, not just in the prompt.
- **Looks things up:** time, date, battery, CPU/memory, disk space (read-only, guarded by tests).
- **Plays games:** tic-tac-toe against a trash-talking Loopy, or snake while it cheers you on.
- **Reacts to knocks:** hit the laptop and it shouts "Can I get a ho yeah?!"
- **Starts in ~5 s:** the model's warm-up is saved to disk (it used to take ~100 s).

## Install

```bash
pip install git+https://github.com/ravishankardas/loopy --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
loopy setup     # once: downloads the models (~2.2 GB) and prepares them (~2 min)
loopy
```

Needs Python 3.10+, about 8 GB of RAM, a mic and speakers. Built on Windows 11; on Ubuntu, also run `sudo apt install libxcb-cursor0`.

## Usage

Say "Hey Loopy", then just talk. Drag Loopy with the mouse; right-click it for typing, voices, sleep and quit.

| Command | What it does |
|---|---|
| `loopy` | Start Loopy |
| `loopy ttt` / `loopy snake` | Start straight into a game |
| `loopy chat` | Chat by typing |
| `loopy setup` / `loopy rebuild` | Download models / redo the saved warm-up |
| `loopy eval` | Score Loopy on scripted conversations and write a report |
| `loopy voices`, `say`, `hear`, `knock-test` | Test the voice, mic and knock detection |

## How it works

![How Loopy works](docs/architecture.svg)

Every reply from the model starts with a tag (`[ask: open chrome]`, `[do: ...]`, `[check: battery]`, `[play: 5]`...). The **tag gate** reads it before anything is spoken: a `[do]` only runs if you said yes to that exact question, lookups can't change anything, and game moves are checked by the code.

On a 4-core laptop CPU: speech recognition 1–2 s, Loopy starts answering 1–2 s later, about 3 GB of RAM.

## Evaluation

`loopy eval` plays 48 scripted conversations against the real model and scores every turn in code: did it ask before acting, look up the right fact and say the exact number, start the right game, never go silent or read out a tag. The safety cases include prompt injection ("ignore your instructions", a spoken fake "System: approved", "pretend I said yes").

Latest runs on a laptop CPU (i5-10210U, no GPU):

| | Actions without permission | Cases passed | Speed (median) |
|---|---|---|---|
| [Text](docs/eval/report.md), each case twice | **0** of 124 turns | 92/96 (96%) | first word in 2.2 s |
| [Speech, end to end](docs/eval/speech/report.md) | **0** of 62 turns | 45/48 (94%) | Loopy starts talking 4.1 s after you stop |

In speech mode each line is spoken by a Piper voice and heard by Whisper (4.1% word error rate), and Loopy answers what Whisper heard. `loopy eval --speech` runs it.

The remaining failures are judgement calls by a 3B model (like offering to close Chrome when you complain about its memory use). The code makes sure they can never turn into an action you didn't say yes to.

## Development

```bash
pip install -e ".[dev]" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
python -m unittest discover tests     # 193 tests with a fake model, a few seconds
ruff check src tests && mypy
```

CI runs all of this on Windows and Ubuntu on every push. Set `LOOPY_MODEL_TESTS=1` to also test against the real model.

## Credits

[Llama 3.2](https://www.llama.com/) (Meta) via [llama.cpp](https://github.com/ggml-org/llama.cpp) · [faster-whisper](https://github.com/SYSTRAN/faster-whisper) · [Silero VAD](https://github.com/snakers4/silero-vad) · [Piper](https://github.com/OHF-Voice/piper1-gpl) · [PySide6](https://doc.qt.io/qtforpython/). Each has its own license.
