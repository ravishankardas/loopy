# Loopy

A desktop voice assistant that runs entirely on your own computer. Loopy is a little paperclip who floats on your desktop: say **"Hey Loopy"**, talk to it naturally, and it answers out loud. It can also do things on your computer, and it always asks before it does.

No cloud and no API keys. Speech recognition, the language model and the voice all run locally on your CPU. Internet is only needed once, to download the models.

## Features

- **Wake word:** say "Hey Loopy" (or "Hey Loopy, how are you?" in one go). After a minute of quiet it falls asleep and waits for the wake word again.
- **Real conversation:** powered by Llama 3.2 3B. It remembers the conversation while awake and copes with misheard words ("close crow" → "close Chrome?").
- **Asks before acting:** Loopy can only *propose* an action. It runs only after you say yes to its question, and this rule is enforced in code, not just in the prompt.
- **Streaming speech:** Loopy starts talking while the model is still writing its reply.
- **Always shows what it's doing:** a status chip under the mascot shows Listening, Catching that, Thinking, Talking or Asleep, each with its own animation.
- **10 voices:** US and UK English, male and female, switchable from the right-click menu.

## Requirements

- Python 3.10 or newer
- About 8 GB of RAM (Loopy itself uses about 3 GB)
- About 2.2 GB of disk space for the models
- A microphone and speakers
- Tested on Windows 11. The macOS and Linux code paths exist but are untested.

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

Loopy appears in the bottom-right corner of your screen. It takes about 15 seconds to load its brain; the status chip says "Getting ready..." until then.

- **Talk to it:** say "Hey Loopy", then just talk. While it's awake you don't need to repeat the wake word.
- **Drag it:** use the left mouse button.
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
| `loopy setup` | Download the models |
| `loopy setup --voice ID` | Download a specific voice and use it |
| `loopy voices` | List the available voices |
| `loopy say "text"` | Speak a line, to test the voice |
| `loopy chat` | Chat with Loopy by typing (add `--speak` to hear the replies) |
| `loopy hear` | Microphone test: prints what Loopy hears, live |

## What Loopy can do

Right now Loopy can **open and close Chrome**, and it will always ask first:

> **You:** Hey Loopy, close Chrome
> **Loopy:** Do you want me to close Chrome?
> **You:** Yes
> **Loopy:** Okay, closing Chrome.

Chrome is closed politely, like clicking the X, so your tabs can be restored.

To add another app, add an entry to `APPS` in [`src/loopy/actions.py`](src/loopy/actions.py). Loopy then understands "open ..." and "close ..." for it automatically.

## Settings

Settings are stored in `settings.json` in Loopy's data folder:

| Key | Default | Meaning |
|---|---|---|
| `voice` | `en_US-lessac-medium` | The selected voice (also set from the Voice menu) |
| `sleep_after_seconds` | `60` | How long Loopy stays awake without activity |

## How it works

```
Microphone ─► Silero VAD ─► faster-whisper ─► "Hey Loopy"? ─► Llama 3.2 3B ─► Piper ─► Speakers
              (is someone     (speech to       (only checked     (reply + tag:      (speaks while
               talking?)       text)            while asleep)     [none]/[ask]/[do]) the model writes)
                                                                       │
                                                       action runs only after a yes to [ask]
```

- **Listening:** a tiny voice-activity model checks each 32 ms of audio, so Whisper only runs when someone has actually spoken. Idle CPU use is about 1%. The mic is muted while Loopy talks, so it can't hear itself.
- **Thinking:** every reply starts with a short tag: `[none]` for chat, `[ask: open chrome]` to ask permission, `[do: open chrome]` to act. The code reads the tag *before* anything is spoken. A `[do]` only runs if Loopy asked about that exact action on the previous turn; otherwise it's turned into a question.
- **Speaking:** speech starts after the first few words. Each next piece is whatever text has arrived since, cut at punctuation where possible, and played through one continuous audio stream.

Measured on a laptop with an Intel i5-10210U (4 cores, no GPU):

| Step | Time |
|---|---|
| Speech recognition | 1–2 s |
| Loopy starts talking, after recognition | 1–2 s |
| RAM in use | about 3 GB |

## Limitations

- English only.
- Like any small model, Llama 3.2 3B sometimes makes up facts. Don't rely on Loopy for facts.
- You can't interrupt Loopy by talking, because the mic is muted while it speaks.
- Only tested on Windows so far.

## Project layout

```
src/loopy/
  app.py        the desktop app: ties everything together, wake/sleep, status
  mascot.py     the paperclip character and its animations
  listener.py   always-on mic + voice activity detection
  stt.py        speech-to-text (faster-whisper)
  wake.py       "Hey Loopy" detection
  brain.py      conversation (Llama 3.2 3B via llama.cpp) + ask-before-acting rule
  actions.py    what Loopy can do on the computer
  tts.py        streaming text-to-speech (Piper)
  models.py     model downloads
  settings.py   user settings
  cli.py        the `loopy` command
```

### Development

```bash
git clone https://github.com/ravishankardas/loopy
cd loopy
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -e . --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
loopy setup
loopy
```

## Credits

Loopy is built on these open-source projects and models. Each has its own license:

- [Llama 3.2](https://www.llama.com/) by Meta (Llama 3.2 Community License), GGUF quantization by [bartowski](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) / [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and OpenAI [Whisper](https://github.com/openai/whisper)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Piper](https://github.com/OHF-Voice/piper1-gpl) and the [Piper voices](https://huggingface.co/rhasspy/piper-voices) (licenses vary by voice; see each voice's model card)
- [PySide6](https://doc.qt.io/qtforpython/) (Qt for Python)
