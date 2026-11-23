"""The `loopy` command.

    loopy                     start the assistant
    loopy tic-tac-toe         start the assistant straight into a game of tic-tac-toe
    loopy snake               start the assistant straight into a game of snake
    loopy setup               download the models (run once after pip install)
    loopy rebuild             redo the saved warm-up now (after a prompt change), so starts are fast
    loopy setup --voice ID    download a specific voice
    loopy voices              list available voices
    loopy say TEXT            speak a line, to test the voice
    loopy chat                talk to Loopy by typing (same brain, no mic needed)
    loopy hear                test the microphone, wake word and speech recognition
    loopy knock-test          hit the laptop and see whether Loopy notices
"""

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="loopy", description="Local desktop voice assistant.")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="start Loopy (the default)")
    sub.add_parser("tic-tac-toe", aliases=["ttt"], help="start Loopy straight into a game of tic-tac-toe")
    sub.add_parser("snake", help="start Loopy straight into a game of snake (you play, Loopy cheers)")
    setup = sub.add_parser("setup", help="download the models Loopy needs")
    setup.add_argument("--voice", help="voice id to download and use (see 'loopy voices')")
    setup.add_argument("--force", action="store_true", help="re-download even if already installed")
    sub.add_parser("rebuild", help="redo the saved warm-up now (e.g. after changing the prompt), so starts are fast")
    sub.add_parser("voices", help="list the available voices")
    say = sub.add_parser("say", help="speak a line of text, to test the voice")
    say.add_argument("text", nargs="+")
    say.add_argument("--voice", help="voice id to use")
    chat = sub.add_parser("chat", help="talk to Loopy by typing (no mic needed)")
    chat.add_argument("--speak", action="store_true", help="also speak Loopy's replies aloud")
    sub.add_parser("hear", help="test the microphone: prints what Loopy hears, live")
    sub.add_parser("knock-test", help="hit your laptop and see if Loopy notices (for tuning knock_peak)")
    args = parser.parse_args(argv)

    if args.command == "setup":
        return _setup(args.voice, args.force)
    if args.command == "rebuild":
        return _rebuild()
    if args.command == "voices":
        return _voices()
    if args.command == "say":
        return _say(" ".join(args.text), args.voice)
    if args.command == "chat":
        return _chat(args.speak)
    if args.command == "hear":
        return _hear()
    if args.command == "knock-test":
        return _knock_test()

    from loopy.app import run

    games = {"tic-tac-toe": "tic-tac-toe", "ttt": "tic-tac-toe", "snake": "snake"}
    return run(start_with_game=games.get(args.command))


def _rebuild() -> int:
    """Redo the saved warm-up now, so the next `loopy` start is fast."""
    import time

    from loopy import models
    from loopy.brain import Brain

    if not models.llm_installed():
        print("The conversation model isn't installed yet. Run: loopy setup")
        return 1
    print("Rebuilding Loopy's saved warm-up. This reads the whole prompt once,")
    print("which takes 1-2 minutes on a laptop CPU...", flush=True)
    started = time.monotonic()
    brain = Brain(models.llm_path())
    brain.load(fresh=True)
    seconds = time.monotonic() - started
    if brain.warmup_path is None:
        print(f"\nCouldn't save the warm-up ({brain.warmup_error or 'unknown error'}).")
        print("Loopy still works, it just starts slowly. Free some RAM or disk space and try again.")
        return 1
    size = brain.warmup_path.stat().st_size / 2**30
    print(f"\nDone in {seconds:.0f} s. Saved {brain.warmup_path.name} ({size:.2f} GB).")
    print("The next `loopy` start will be fast.")
    return 0


def _setup(voice_id: str | None, force: bool) -> int:
    from loopy import models, settings

    voice = models.get_voice(voice_id) if voice_id else settings.selected_voice()
    if voice is None:
        print(f"Unknown voice '{voice_id}'. See: loopy voices")
        return 1

    print(f"Installing models into {models.DATA_DIR}")
    print(f"Voice: {voice.label}")
    if voice.installed and not force:
        print("  already installed")
    else:
        try:
            models.install_voice(voice, force=force, progress=_print_progress)
            print()
        except OSError as exc:
            print(f"\nDownload failed: {exc}\nCheck your internet connection and run 'loopy setup' again.")
            return 1
    if voice_id:
        settings.save(voice=voice.id)

    print(f"Speech recognition (Whisper {models.STT_MODEL}):")
    if models.stt_installed() and not force:
        print("  already installed")
    else:
        try:
            models.install_stt(force=force)
        except Exception as exc:  # huggingface_hub raises its own error types
            print(f"\nDownload failed: {exc}\nCheck your internet connection and run 'loopy setup' again.")
            return 1

    print(f"Conversation model ({models.LLM_NAME}, ~1.9 GB):")
    if models.llm_installed() and not force:
        print("  already installed")
    else:
        try:
            models.install_llm(force=force, progress=_print_progress)
            print()
        except OSError as exc:
            print(f"\nDownload failed: {exc}\nCheck your internet connection and run 'loopy setup' again.")
            return 1
    print("Done. Start Loopy with: loopy")
    return 0


def _chat(speak: bool) -> int:
    from loopy import models, settings

    if not models.llm_installed():
        print("The conversation model isn't installed yet. Run: loopy setup")
        return 1
    from loopy.brain import Brain

    speaker = None
    if speak and (voice := settings.active_voice()):
        from loopy.tts import Speaker

        speaker = Speaker(voice.path)

    print("Loading Loopy's brain...")
    brain = Brain(models.llm_path())
    brain.load()
    print("Type to chat with Loopy. Ctrl+C to quit.\n")
    try:
        while True:
            text = input("you> ").strip()
            if not text:
                continue
            print("loopy> ", end="", flush=True)
            stream = speaker.stream() if speaker else None

            def on_text(piece: str, stream=stream) -> None:  # printed (and spoken) as it's written
                print(piece, end="", flush=True)
                if stream:
                    stream.feed(piece)

            def on_event(kind, *args) -> None:  # tic-tac-toe: show the board as text
                if kind == "place" and brain.game is not None:
                    cell, mark = args
                    who = "you" if mark == "X" else "Loopy"
                    board = "\n".join(f"        {row}" for row in brain.game.describe().splitlines())
                    print(f"\n  [{who}: {mark} in {cell}]\n{board}\n", end="loopy> ", flush=True)
                elif kind == "illegal":
                    print(f"\n  [Loopy tried {args[0]}: {args[1]}, picking again]", end="\nloopy> ", flush=True)

            turn = brain.respond(text, on_text, on_event)
            if turn.result and not turn.result.ok:
                on_text(f" {turn.result.message}")
            if stream:
                stream.close()
            print()
            for name, value in turn.checked:
                print(f"  [checked {name}: {value}]")
            if turn.result:
                print(f"  [{'done' if turn.result.ok else 'failed'}: {turn.result.message}]")
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
    finally:
        if speaker:
            speaker.close()
    return 0


def _hear() -> int:
    from loopy import models

    if not models.stt_installed():
        print("Speech recognition isn't installed yet. Run: loopy setup")
        return 1
    import queue

    from loopy import wake
    from loopy.listener import Listener
    from loopy.stt import Transcriber

    transcriber = Transcriber(models.stt_path())
    transcriber.load()
    utterances: queue.Queue = queue.Queue()
    listener = Listener(on_utterance=utterances.put, on_speech_start=lambda: print("  (hearing speech...)"))
    listener.start()
    print('Listening. Try "Hey Loopy, how are you?". Press Ctrl+C to stop.\n')
    try:
        while True:
            try:
                audio = utterances.get(timeout=0.2)  # timeout keeps Ctrl+C responsive
            except queue.Empty:
                continue
            text = transcriber.transcribe(audio)
            if not text:
                print("  (too short / just noise)")
                continue
            woke, rest = wake.split_wake(text)
            print(f'  heard: "{text}"   wake word: {"yes" if woke else "no"}'
                  + (f'   -> Loopy gets: "{rest}"' if woke and rest else ""))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        listener.stop()
    return 0


def _knock_test() -> int:
    import os
    import signal

    if sys.platform == "win32":
        os.environ.setdefault("QT_MEDIA_BACKEND", "windows")  # no decoder chatter in the output
    from PySide6.QtCore import QCoreApplication, QObject, QTimer, Signal

    from loopy import listener as L
    from loopy import settings
    from loopy.sounds import HO_YEAH, Clip

    app = QCoreApplication([])
    clip = Clip(HO_YEAH)

    class Bridge(QObject):  # knocks are detected on the mic thread; the clip plays on Qt's
        knocked = Signal()

    bridge = Bridge()

    s = settings.load()
    peak_needed = s.get("knock_peak", L.KNOCK_PEAK)
    above_needed = s.get("knock_above_noise", L.KNOCK_ABOVE_NOISE)
    knocks: list[float] = []

    def on_knock(peak: float) -> None:
        knocks.append(peak)
        bridge.knocked.emit()

    class SoundReport:
        """Groups frames into sounds; prints one line per sound (its strongest
        moment) and why it did or didn't count as a knock."""

        def __init__(self) -> None:
            self.reset()

        def reset(self) -> None:
            self.best: L.KnockReading | None = None
            self.knocked = self.learning = self.cooling = False
            self.quiet = 0

        def __call__(self, r: L.KnockReading) -> None:
            if r.above_noise >= 4:  # clearly louder than the room: part of a sound
                if self.best is None or r.above_noise > self.best.above_noise:
                    self.best = r
                self.knocked |= r.knocked
                self.learning |= r.learning
                self.cooling |= r.cooling_down
                self.quiet = 0
            elif self.best is not None:
                self.quiet += 1
                if self.quiet >= 4:  # ~130 ms of calm: that sound is over
                    self.print_line(self.best)
                    self.reset()

        def print_line(self, b: L.KnockReading) -> None:
            why = [w for w, bad in [
                ("not loud enough vs the room", b.above_noise < above_needed),
                ("peak too low", b.peak < peak_needed),
                ("not sharp", b.sharpness < L.KNOCK_SHARPNESS),
                ("sounded like speech", b.speech_prob >= L.START_PROB)] if bad]
            if not why:
                why = ["still learning the room, first second"] if self.learning else \
                      ["cooldown: just after another knock"]
            verdict = "KNOCK! -> ho yeah" if self.knocked else f"not a knock ({', '.join(why)})"
            print(f"  {b.above_noise:6.1f}x louder than the room (need {above_needed:g})"
                  f"   peak {b.peak:.3f} (need {peak_needed:g})"
                  f"   sharpness {b.sharpness:.1f} (need {L.KNOCK_SHARPNESS:g})"
                  f"   speech {b.speech_prob:.2f}   {verdict}", flush=True)

    mic = L.Listener(on_utterance=lambda _a: None, on_knock=on_knock, on_sound=SoundReport(),
                     knock_peak=peak_needed, knock_above_noise=above_needed)

    def play() -> None:
        if clip.play():
            mic.pause()  # so the clip itself isn't heard as another knock

    bridge.knocked.connect(play)
    clip.finished.connect(lambda: QTimer.singleShot(300, mic.resume))
    mic.start()
    print("Hit or tap your laptop a few times, then also talk, type and clap. One line per sound;")
    print("every knock plays the clip, like Loopy does. Ctrl+C to stop.")
    print('Sensitivity is "knock_above_noise" in settings.json: lower it if hits are missed,')
    print("raise it if typing or talking counts as a knock.\n")

    signal.signal(signal.SIGINT, lambda *_: app.quit())  # Ctrl+C while Qt's loop runs
    wake = QTimer()
    wake.timeout.connect(lambda: None)  # lets Python notice Ctrl+C
    wake.start(200)
    try:
        app.exec()
    finally:
        clip.stop()
        mic.stop()
    print(f"\nStopped. {len(knocks)} knock(s) detected.")
    return 0


def _voices() -> int:
    from loopy import models, settings

    current = settings.active_voice()
    for v in models.VOICES:
        mark = "*" if v == current else " "
        state = "installed" if v.installed else "not downloaded"
        print(f" {mark} {v.id:<36} {v.label:<28} {state}")
    print("\n* = current voice. Download one with: loopy setup --voice <id>")
    print("You can also switch voices from Loopy's right-click menu.")
    return 0


def _say(text: str, voice_id: str | None) -> int:
    from loopy import models, settings

    voice = models.get_voice(voice_id) if voice_id else settings.active_voice()
    if voice is None or not voice.installed:
        print("That voice isn't installed yet. Run: loopy setup" + (f" --voice {voice_id}" if voice_id else ""))
        return 1
    from loopy.tts import Speaker

    Speaker(voice.path).speak(text)
    return 0


def _print_progress(done: int, total: int) -> None:
    mb = done / 1_048_576
    if total:
        frac = done / total
        line = f"\r  [{'#' * int(frac * 30):<30}] {frac:4.0%} {mb:5.1f} MB"
    else:
        line = f"\r  {mb:5.1f} MB"
    print(line, end="", flush=True)


if __name__ == "__main__":
    sys.exit(main())
