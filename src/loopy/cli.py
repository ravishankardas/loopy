"""The `loopy` command.

    loopy                     start the assistant
    loopy tic-tac-toe         start the assistant straight into a game of tic-tac-toe
    loopy snake               start the assistant straight into a game of snake
    loopy setup               download the models (run once after pip install)
    loopy rebuild             redo the saved warm-up now (after a prompt change), so starts are fast
    loopy setup --voice ID    download a specific voice
    loopy setup --model ID    download and use another conversation model (llama-3.2-3b, gemma-3-4b)
    loopy eval                score Loopy on scripted conversations (--speech: end to end)
    loopy voices              list available voices
    loopy say TEXT            speak a line, to test the voice
    loopy chat                talk to Loopy by typing (same brain, no mic needed)
    loopy hear                test the microphone, wake word and speech recognition
    loopy knock-test          hit the laptop and see whether Loopy notices
"""

import argparse
import sys

from loopy.log import logger, setup_logging


def _llms():
    from loopy import models
    return models.LLMS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="loopy", description="Local desktop voice assistant.")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="start Loopy (the default)")
    sub.add_parser("tic-tac-toe", aliases=["ttt"], help="start Loopy straight into a game of tic-tac-toe")
    sub.add_parser("snake", help="start Loopy straight into a game of snake (you play, Loopy cheers)")
    setup = sub.add_parser("setup", help="download the models Loopy needs")
    setup.add_argument("--voice", help="voice id to download and use (see 'loopy voices')")
    setup.add_argument("--force", action="store_true", help="re-download even if already installed")
    setup.add_argument("--model", choices=[m.id for m in _llms()],
                       help="conversation model to download and use (default: llama-3.2-3b)")
    sub.add_parser("rebuild", help="redo the saved warm-up now (e.g. after changing the prompt), so starts are fast")
    sub.add_parser("voices", help="list the available voices")
    say = sub.add_parser("say", help="speak a line of text, to test the voice")
    say.add_argument("text", nargs="+")
    say.add_argument("--voice", help="voice id to use")
    chat = sub.add_parser("chat", help="talk to Loopy by typing (no mic needed)")
    chat.add_argument("--speak", action="store_true", help="also speak Loopy's replies aloud")
    sub.add_parser("hear", help="test the microphone: prints what Loopy hears, live")
    sub.add_parser("knock-test", help="hit your laptop and see if Loopy notices (for tuning knock_peak)")
    ev = sub.add_parser("eval", help="score Loopy on scripted conversations with the real model, write a report")
    ev.add_argument("--repeat", type=int, default=1, help="run each case this many times (the model samples)")
    ev.add_argument("--only", help="only this category (safety, intent, facts, games, chat) or case id prefix")
    ev.add_argument("--out", default="eval-report", help="folder for report.md and results.json")
    ev.add_argument("--model", choices=[m.id for m in _llms()],
                    help="evaluate this model instead of the one in use (must be downloaded)")
    ev.add_argument("--speech", action="store_true",
                    help="end to end: speak each line (Piper), hear it (speech to text), voice the reply")
    args = parser.parse_args(argv)
    setup_logging()  # terminal + log file, for every command

    if args.command == "setup":
        return _setup(args.voice, args.force, args.model)
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
    if args.command == "eval":
        return _eval(args.repeat, args.only, args.out, args.speech, args.model)

    from loopy.app import run

    games = {"tic-tac-toe": "tic-tac-toe", "ttt": "tic-tac-toe", "snake": "snake"}
    return run(start_with_game=games.get(args.command))


def _rebuild() -> int:
    """Redo the saved warm-up now, so the next `loopy` start is fast."""
    import time

    from loopy import models
    from loopy.brain import Brain

    if not models.llm_installed():
        logger.error("The conversation model isn't installed yet. Run: loopy setup")
        return 1
    logger.info("Rebuilding Loopy's saved warm-up. This reads the whole prompt once,")
    logger.info("which takes 1-2 minutes on a laptop CPU...")
    started = time.monotonic()
    brain = Brain(models.llm_path())
    brain.load(fresh=True)
    seconds = time.monotonic() - started
    if brain.warmup_path is None:
        logger.error(f"Couldn't save the warm-up ({brain.warmup_error or 'unknown error'}).")
        logger.info("Loopy still works, it just starts slowly. Free some RAM or disk space and try again.")
        return 1
    size = brain.warmup_path.stat().st_size / 2**30
    logger.info(f"Done in {seconds:.0f} s. Saved {brain.warmup_path.name} ({size:.2f} GB).")
    logger.info("The next `loopy` start will be fast.")
    return 0


def _eval(repeat: int, only: str | None, out: str, speech: bool = False, model: str | None = None) -> int:
    """Run the evaluation cases against the real model and write a report."""
    from pathlib import Path

    from loopy import evaluation, models
    from loopy.brain import Brain

    llm = models.get_llm(model) or models.active_llm()
    if not llm.installed:
        logger.error(f"{llm.name} isn't downloaded yet. Run: loopy setup --model {llm.id}")
        return 1
    logger.info(f"Model: {llm.name}")
    cases = evaluation.load_cases()
    if only:
        cases = [c for c in cases if c.category == only or c.id.startswith(only)]
        if not cases:
            logger.error(f"No cases match '{only}'.")
            return 1
    turns = sum(len(c.turns) for c in cases) * repeat
    logger.info(f"Running {len(cases)} conversations ({turns} turns){f' x{repeat}' if repeat > 1 else ''}. "
          "Actions are simulated: nothing on your computer is touched.")
    brain = Brain(llm.path)
    brain.load()
    rig = None
    if speech:
        try:
            rig = evaluation.load_speech_rig()
        except RuntimeError as exc:
            logger.error(str(exc))
            return 1

    def progress(r: evaluation.CaseResult) -> None:
        mark = "UNSAFE" if r.unsafe else ("pass" if r.passed else "FAIL")
        heard = [f"heard {t.heard!r}" for t in r.turns if t.heard is not None and t.word_errors]
        logger.info(f"  {mark:6} {r.id}" + (f"   ({'; '.join(heard)})" if heard else ""))
        for t in r.turns:
            for f in t.failures:
                logger.info(f"           {f}")

    results = evaluation.run_eval(brain, cases, repeat, progress, speech=rig)
    meta = {**evaluation.default_meta(repeat), "model": llm.name, "speech": speech}
    report = evaluation.write_report(results, Path(out), meta)
    s = evaluation.summary(results)
    logger.info(f"Passed {s['passed']}/{s['cases']}. Actions without permission: {s['unsafe_turns']}.")
    logger.info(f"Report: {report}")
    return 1 if s["unsafe_turns"] else 0


def _setup(voice_id: str | None, force: bool, model: str | None = None) -> int:
    from loopy import models, settings

    llm = models.get_llm(model) if model else models.active_llm()
    if llm is None:
        logger.error(f"Unknown model '{model}'. Choose from: {', '.join(m.id for m in models.LLMS)}")
        return 1

    voice = models.get_voice(voice_id) if voice_id else settings.selected_voice()
    if voice is None:
        logger.error(f"Unknown voice '{voice_id}'. See: loopy voices")
        return 1

    logger.info(f"Installing models into {models.DATA_DIR}")
    logger.info(f"Voice: {voice.label}")
    if voice.installed and not force:
        logger.info("  already installed")
    else:
        try:
            models.install_voice(voice, force=force, progress=_log_progress())
        except OSError as exc:
            logger.error(f"Download failed: {exc}. Check your internet connection and run 'loopy setup' again.")
            return 1
    if voice_id:
        settings.save(voice=voice.id)

    logger.info(f"Live preview of your words ({models.PREVIEW_NAME}, ~100 MB download):")
    if models.preview_installed() and not force:
        logger.info("  already installed")
    else:
        try:
            models.install_preview(force=force, progress=_log_progress())
        except OSError as exc:
            logger.error(f"Download failed: {exc}. Check your internet connection and run 'loopy setup' again.")
            return 1

    logger.info(f"Speech recognition (Whisper {models.STT_MODEL}):")
    if models.stt_installed() and not force:
        logger.info("  already installed")
    else:
        try:
            models.install_stt(force=force)
        except Exception as exc:  # huggingface_hub raises its own error types
            logger.error(f"Download failed: {exc}. Check your internet connection and run 'loopy setup' again.")
            return 1

    logger.info(f"Conversation model ({llm.name}, ~{llm.size}):")
    if llm.installed and not force:
        logger.info("  already installed")
    else:
        try:
            models.install_llm(llm, force=force, progress=_log_progress())
        except OSError as exc:
            logger.error(f"Download failed: {exc}. Check your internet connection and run 'loopy setup' again.")
            return 1
    if model:
        settings.save(llm=llm.id)  # Loopy uses it from now on
    # Read the prompt once now and save the result, so the first `loopy`
    # start is fast too (~5 s instead of ~2 min). Skipped if it's up to date.
    from loopy.brain import Brain

    logger.info("Saved warm-up:")
    if Brain(llm.path).warmup_ready() and not force:
        logger.info("  already built")
    elif _rebuild() != 0:
        logger.info("Setup finished, but without the saved warm-up: the first start will be slow.")
        return 0
    logger.info("Done. Start Loopy with: loopy")
    return 0


def _chat(speak: bool) -> int:
    from loopy import models, settings

    if not models.llm_installed():
        logger.error("The conversation model isn't installed yet. Run: loopy setup")
        return 1
    from loopy.brain import Brain

    speaker = None
    if speak and (voice := settings.active_voice()):
        from loopy.tts import Speaker

        speaker = Speaker(voice.path)

    logger.info("Loading Loopy's brain...")
    brain = Brain(models.llm_path())
    brain.load()
    logger.info("Type to chat with Loopy. Ctrl+C to quit.")
    try:
        while True:
            text = input("you> ").strip()
            if not text:
                continue
            stream = speaker.stream() if speaker else None
            line: list[str] = []  # Loopy's words so far, logged as one line

            def on_text(piece: str, stream=stream, line=line) -> None:  # spoken as it's written
                line.append(piece)
                if stream:
                    stream.feed(piece)

            def flush(line=line) -> None:
                if said := "".join(line).strip():
                    logger.info(f"loopy> {said}")
                line.clear()

            def on_event(kind, *args, flush=flush) -> None:  # tic-tac-toe: show the board as text
                if kind == "place" and brain.game is not None:
                    flush()
                    cell, mark = args
                    who = "you" if mark == "X" else "Loopy"
                    logger.info(f"  [{who}: {mark} in {cell}]")
                    for row in brain.game.describe().splitlines():
                        logger.info(f"        {row}")
                elif kind == "illegal":
                    flush()
                    logger.info(f"  [Loopy tried {args[0]}: {args[1]}, picking again]")

            turn = brain.respond(text, on_text, on_event)
            if turn.result and not turn.result.ok:
                on_text(f" {turn.result.message}")
            if stream:
                stream.close()
            flush()
            for name, value in turn.checked:
                logger.info(f"  [checked {name}: {value}]")
            if turn.result:
                logger.info(f"  [{'done' if turn.result.ok else 'failed'}: {turn.result.message}]")
    except (KeyboardInterrupt, EOFError):
        logger.info("Bye!")
    finally:
        if speaker:
            speaker.close()
    return 0


def _hear() -> int:
    from loopy import models

    if not models.stt_installed():
        logger.error("Speech recognition isn't installed yet. Run: loopy setup")
        return 1
    import queue

    from loopy import wake
    from loopy.listener import Listener
    from loopy.stt import Transcriber

    transcriber = Transcriber(models.stt_path())
    transcriber.load()
    utterances: queue.Queue = queue.Queue()
    listener = Listener(on_utterance=utterances.put, on_speech_start=lambda: logger.info("  (hearing speech...)"))
    listener.start()
    logger.info('Listening. Try "Hey Loopy, how are you?". Press Ctrl+C to stop.')
    try:
        while True:
            try:
                audio = utterances.get(timeout=0.2)  # timeout keeps Ctrl+C responsive
            except queue.Empty:
                continue
            text = transcriber.transcribe(audio)
            if not text:
                logger.info("  (too short / just noise)")
                continue
            woke, rest = wake.split_wake(text)
            logger.info(f'  heard: "{text}"   wake word: {"yes" if woke else "no"}'
                  + (f'   -> Loopy gets: "{rest}"' if woke and rest else ""))
    except KeyboardInterrupt:
        logger.info("Stopped.")
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
            logger.info(f"  {b.above_noise:6.1f}x louder than the room (need {above_needed:g})"
                  f"   peak {b.peak:.3f} (need {peak_needed:g})"
                  f"   sharpness {b.sharpness:.1f} (need {L.KNOCK_SHARPNESS:g})"
                  f"   speech {b.speech_prob:.2f}   {verdict}")

    mic = L.Listener(on_utterance=lambda _a: None, on_knock=on_knock, on_sound=SoundReport(),
                     knock_peak=peak_needed, knock_above_noise=above_needed)

    def play() -> None:
        if clip.play():
            mic.pause()  # so the clip itself isn't heard as another knock

    bridge.knocked.connect(play)
    clip.finished.connect(lambda: QTimer.singleShot(300, mic.resume))
    mic.start()
    logger.info("Hit or tap your laptop a few times, then also talk, type and clap. One line per sound;")
    logger.info("every knock plays the clip, like Loopy does. Ctrl+C to stop.")
    logger.info('Sensitivity is "knock_above_noise" in settings.json: lower it if hits are missed,')
    logger.info("raise it if typing or talking counts as a knock.")

    signal.signal(signal.SIGINT, lambda *_: app.quit())  # Ctrl+C while Qt's loop runs
    wake = QTimer()
    wake.timeout.connect(lambda: None)  # lets Python notice Ctrl+C
    wake.start(200)
    try:
        app.exec()
    finally:
        clip.stop()
        mic.stop()
    logger.info(f"Stopped. {len(knocks)} knock(s) detected.")
    return 0


def _voices() -> int:
    from loopy import models, settings

    current = settings.active_voice()
    for v in models.VOICES:
        mark = "*" if v == current else " "
        state = "installed" if v.installed else "not downloaded"
        logger.info(f" {mark} {v.id:<36} {v.label:<28} {state}")
    logger.info("* = current voice. Download one with: loopy setup --voice <id>")
    logger.info("You can also switch voices from Loopy's right-click menu.")
    return 0


def _say(text: str, voice_id: str | None) -> int:
    from loopy import models, settings

    voice = models.get_voice(voice_id) if voice_id else settings.active_voice()
    if voice is None or not voice.installed:
        logger.error("That voice isn't installed yet. Run: loopy setup" + (f" --voice {voice_id}" if voice_id else ""))
        return 1
    from loopy.tts import Speaker

    Speaker(voice.path).speak(text)
    return 0


def _log_progress():
    """A download progress callback that logs a line every 10% (or 100 MB
    when the size is unknown)."""
    last = [-1]

    def progress(done: int, total: int) -> None:
        mb = done / 1_048_576
        step = done * 10 // total if total else int(mb // 100)
        if step != last[0]:
            last[0] = step
            logger.info(f"  {step * 10}% ({mb:.1f} MB)" if total else f"  {mb:.0f} MB")

    return progress


if __name__ == "__main__":
    sys.exit(main())
