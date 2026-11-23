# Loopy

[![CI](https://github.com/ravishankardas/loopy/actions/workflows/ci.yml/badge.svg)](https://github.com/ravishankardas/loopy/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Ruff](https://img.shields.io/badge/lint-ruff-261230)
![Mypy](https://img.shields.io/badge/types-mypy-2a6db2)

A desktop voice assistant that runs entirely on your own computer. Loopy is a little paperclip who floats on your desktop: say **"Hey Loopy"**, talk to it naturally, and it answers out loud. It can also do things on your computer, and it always asks before it does.

No cloud and no API keys. Speech recognition, the language model and the voice all run locally on your CPU. Internet is only needed once, to download the models.

## Features

- **Wake word:** say "Hey Loopy" (or "Hey Loopy, how are you?" in one go). After a minute of quiet it falls asleep and waits for the wake word again.
- **Real conversation:** powered by Llama 3.2 3B. It remembers the conversation while awake and copes with misheard words ("close crow" → "close Chrome?").
- **Asks before acting:** Loopy can only *propose* an action. It runs only after you say yes to its question, and this rule is enforced in code, not just in the prompt.
- **Looks things up, read-only:** time, date, battery, CPU and memory use, and free space on every drive. These lookups can never change anything, and tests enforce that.
- **Plays games:** tic-tac-toe against Loopy, where Llama picks its own moves and the code referees, with plenty of sassy trash talk. Or snake, where you play and Loopy cheers you on.
- **Knock knock:** hit the laptop and Loopy shouts "Can I get a ho yeah?!"
- **Streaming speech:** Loopy starts talking while the model is still writing its reply.
- **Fast start:** about 5 seconds, by saving the model's warm-up to disk (it was about 100 seconds before).
- **Always shows what it's doing:** a status chip under the mascot shows Listening, Catching that, Thinking, Talking or Asleep, each with its own animation.
- **10 voices:** US and UK English, male and female, switchable from the right-click menu.

## Requirements

- Python 3.10 or newer
- About 8 GB of RAM (Loopy itself uses about 3 GB)
- About 2.2 GB of disk space for the models
- A microphone and speakers
- Built and used on Windows 11. The test suite also runs on Ubuntu in CI; the desktop app on Linux is lightly tested, and macOS is untested.

### Ubuntu / Linux

Ubuntu's default Wayland desktop doesn't let apps move their own windows, so Loopy runs itself through XWayland there. That needs one system library:

```bash
sudo apt install libxcb-cursor0
```

Without it Loopy still starts on Wayland, and you can still drag it, but the game windows can't move along with it.

## Install

```bash
pip install git+https://github.com/ravishankardas/loopy --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
loopy setup
```

> The `--extra-index-url` gives you a prebuilt `llama-cpp-python`. Without it, pip tries to compile it from source, which needs a C++ compiler.

`loopy setup` downloads the models (once) into your user data folder:

| Model | Size | Used for |
|---|---|---|
| Piper `en_US-lessac-medium` | 60 MB | Loopy's voice |
| faster-whisper `base.en` | 140 MB | Speech recognition |
| Llama 3.2 3B Instruct (Q4_K_M) | 1.9 GB | Conversation |

They're stored in `%LOCALAPPDATA%\loopy` on Windows, `~/Library/Application Support/loopy` on macOS, and `~/.local/share/loopy` on Linux.

## Usage

```bash
loopy
```

Loopy appears in the bottom-right corner of your screen. It takes about 5 seconds to load its brain, and the status chip says "Getting ready..." until then. The very first start (and the first one after an update that changes Loopy's prompt) takes about 2 minutes, because the model reads its instructions once and saves the result.

- **Talk to it:** say "Hey Loopy", then just talk. While it's awake you don't need to repeat the wake word.
- **Drag it:** use the left mouse button. An open game board moves along with it.
- **Right-click it** for:
  - Type to Loopy...: chat by typing
  - Go to sleep / Wake up
  - Voice: switch voices; new ones download automatically
  - Quit
- **Ctrl+C** in the terminal also quits.

### Commands

| Command | What it does |
|---|---|
| `loopy` | Start the assistant |
| `loopy tic-tac-toe` (or `loopy ttt`) | Start straight into a game of tic-tac-toe |
| `loopy snake` | Start straight into a game of snake |
| `loopy setup` | Download the models |
| `loopy setup --voice ID` | Download a specific voice and use it |
| `loopy rebuild` | Redo the saved warm-up now, so the next start is fast |
| `loopy voices` | List the available voices |
| `loopy say "text"` | Speak a line, to test the voice |
| `loopy chat` | Chat with Loopy by typing (add `--speak` to hear the replies) |
| `loopy hear` | Microphone test: prints what Loopy hears, live |
| `loopy knock-test` | Knock test: hit the laptop and see how each sound scores |

## What Loopy can do

Ask it "what can you do?" and it tells you.

### Actions (always asks first)

Loopy can **open and close Chrome**:

> **You:** Hey Loopy, close Chrome
> **Loopy:** Do you want me to close Chrome?
> **You:** Yes
> **Loopy:** Okay, closing Chrome.

Chrome is closed politely, like clicking the X, so your tabs can be restored.

To add another app, add an entry to `APPS` in [`src/loopy/actions.py`](src/loopy/actions.py). Loopy then understands "open ..." and "close ..." for it automatically.

### Lookups (no need to ask)

"What time is it?", "How's my battery?", "Is my computer busy?", "How much space is left?" Loopy looks up the real value and says the exact numbers. Lookups take no arguments from the model and may only call read-only functions: [`tests/test_checks_readonly.py`](tests/test_checks_readonly.py) fails if anyone adds a write, delete, process or shell call to [`checks.py`](src/loopy/checks.py).

### Tic-tac-toe

Say "let's play tic-tac-toe" (or "tiktok toe", which is how speech recognition usually hears it). A board appears next to Loopy. You're X: say a cell number, 1 (top-left) to 9 (bottom-right), or a name like "the center". Loopy is O. The model chooses Loopy's move and the code checks it: an illegal move is rejected with the reason, and the model picks again. Expect trash talk ("Two in a row? Adorable. Blocked."). Winners get balloons, and Loopy keeps score. Say "rematch" to play again, or "close the game" to stop.

### Snake

Say "let's play snake". You play with the arrow keys or WASD (Space pauses, Esc closes), and Loopy cheers every 5 points and celebrates a new best score. The snake wraps through the walls. If it passes through them 3 times with nobody steering, the game pauses and Loopy asks if you're still playing; no answer within 30 seconds closes it.

### Knocks

Hit or tap the laptop and Loopy plays "Can I get a ho yeah?!". It listens for a short, sharp spike that is much louder than the room, ignores speech, and is switched off while the model loads (a busy CPU makes the mic click). If hits are missed or typing counts as a knock, run `loopy knock-test` and adjust the settings below.

## Settings

Settings are stored in `settings.json` in Loopy's data folder:

| Key | Default | Meaning |
|---|---|---|
| `voice` | `en_US-lessac-medium` | The selected voice (also set from the Voice menu) |
| `sleep_after_seconds` | `60` | How long Loopy stays awake without activity |
| `knock_above_noise` | `40` | How many times louder than the room a knock must be. Lower it if hits are missed |
| `knock_peak` | `0.05` | The quietest peak that counts as a knock |
| `snake_best` | `0` | Your best snake score (saved automatically) |

## How it works

```
Microphone ─► Silero VAD ─► faster-whisper ─► "Hey Loopy"? ─► Llama 3.2 3B ─► tag gate ─► Piper ─► Speakers
              (is someone     (speech to       (only checked     (reply starts    (code decides    (speaks while
               talking?)       text)            while asleep)     with a tag)      what happens)    the model writes)
                                                                                       │
                     [do] runs only after a yes to [ask] · [check] reads a fact · [game]/[move]/[play] drive the games
```

- **Listening:** a tiny voice-activity model checks each 32 ms of audio, so Whisper only runs when someone has actually spoken. Idle CPU use is about 1%. The mic is muted while Loopy talks, so it can't hear itself.
- **Thinking:** every reply starts with a short tag: `[none]` for chat, `[ask: open chrome]` to ask permission, `[do: open chrome]` to act, `[check: battery]` to look something up, and `[game: ...]`, `[move: N]` and `[play: N]` for the games. The code reads the tag *before* anything is spoken. A `[do]` only runs if Loopy asked about that exact action on the previous turn; otherwise it's turned into a question. The code also stops the model from speaking tags, inventing "system:" lines or drawing a board in text, and if a reply comes back empty, the model is asked again, so Loopy never goes silent.
- **Starting fast:** reading the system prompt and examples is most of the startup work (about 80 seconds on a laptop CPU), so the model's state after it is saved to disk (about 0.5 GB) and loaded on later starts. The saved file is tied to a fingerprint of the prompt, the examples, the model file, the llama.cpp version and the settings, so any change redoes it automatically.
- **Speaking:** speech starts after the first few words. Each next piece is whatever text has arrived since, cut at punctuation where possible, and played through one continuous audio stream.

Measured on a laptop with an Intel i5-10210U (4 cores, no GPU):

| Step | Time |
|---|---|
| Start, with the saved warm-up | about 5 s |
| First start (or after a prompt change) | about 2 min |
| Speech recognition | 1–2 s |
| Loopy starts talking, after recognition | 1–2 s |
| RAM in use | about 3 GB |

## Limitations

- English only.
- Like any small model, Llama 3.2 3B sometimes makes up facts. Don't rely on Loopy for facts.
- You can't interrupt Loopy by talking, because the mic is muted while it speaks.
- Knock detection depends on your laptop's microphone; use `loopy knock-test` to tune it.
- The desktop app is mostly tested on Windows.

## Project layout

```
src/loopy/
  app.py        the desktop app: ties everything together, wake/sleep, status
  mascot.py     the paperclip character, its animations, windows that move together
  listener.py   always-on mic, voice activity detection and knock detection
  stt.py        speech-to-text (faster-whisper)
  wake.py       "Hey Loopy" detection
  brain.py      conversation (Llama 3.2 3B via llama.cpp), the tag gate, ask-before-acting,
                game turns, the saved warm-up
  actions.py    what Loopy can do on the computer (asks first)
  checks.py     read-only lookups: time, date, battery, CPU/memory, disks
  game.py       tic-tac-toe rules, the board window and the celebration
  snake.py      the snake game and its window
  sounds.py     short sound clips (the knock reaction)
  tts.py        streaming text-to-speech (Piper)
  models.py     model downloads
  settings.py   user settings
  cli.py        the `loopy` command
  data/         bundled sound files
tests/          fast tests with a fake model, plus opt-in real-model tests
.github/        CI: lint, type check and tests on Windows and Ubuntu
```

### Development

```bash
git clone https://github.com/ravishankardas/loopy
cd loopy
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -e ".[dev]" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
loopy setup
loopy
```

### Tests

```bash
python -m unittest discover tests        # fast tests, a few seconds, no model needed
ruff check src tests                     # lint
mypy                                     # type check
```

GitHub Actions runs all three on every push: lint and type checks, plus the tests on Windows and Ubuntu (Python 3.11 and 3.12). See `.github/workflows/ci.yml`.

The 168 fast tests use a fake model that replies with scripted, misbehaving output. They check that Loopy's code still asks before every action, never speaks tags or invented "system:" lines, never goes silent, reports real numbers, keeps the lookups read-only, referees the games correctly and doesn't mistake speech or clicks for knocks.

To also replay real conversations against Llama 3.2 3B (a few minutes, needs `loopy setup`):

```bash
set LOOPY_MODEL_TESTS=1                  # PowerShell: $env:LOOPY_MODEL_TESTS=1
python -m unittest discover tests -v
```

## Credits

Loopy is built on these open-source projects and models. Each has its own license:

- [Llama 3.2](https://www.llama.com/) by Meta (Llama 3.2 Community License), GGUF quantization by [bartowski](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) / [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and OpenAI [Whisper](https://github.com/openai/whisper)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Piper](https://github.com/OHF-Voice/piper1-gpl) and the [Piper voices](https://huggingface.co/rhasspy/piper-voices) (licenses vary by voice; see each voice's model card)
- [PySide6](https://doc.qt.io/qtforpython/) (Qt for Python)
