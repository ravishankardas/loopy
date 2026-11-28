"""Logging with loguru: every line has the time and the file:line that
logged it, goes to the terminal and a log file, and crashes are logged."""

import io
import os
import re
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from loopy import app  # noqa: E402
from loopy.log import logger, setup_logging  # noqa: E402

LINE = re.compile(r"^\d\d:\d\d:\d\d\.\d{3} \| INFO +\| (?P<where>\S+?):(?P<line>\d+) *\| (?P<msg>.*)$")


class Logging(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.terminal = io.StringIO()
        self.log_file = self.dir / "logs" / "loopy.log"
        hooks = (sys.excepthook, threading.excepthook)
        self.addCleanup(lambda: (setattr(sys, "excepthook", hooks[0]), setattr(threading, "excepthook", hooks[1])))
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.addCleanup(logger.remove)                     # release the log file before deleting it
        setup_logging(self.log_file, terminal=self.terminal)

    def terminal_lines(self) -> list[str]:
        return [line for line in self.terminal.getvalue().splitlines() if line]

    def file_text(self) -> str:
        logger.complete()                                  # the file sink writes in the background
        return self.log_file.read_text(encoding="utf-8")

    def test_lines_have_the_time_and_where_they_came_from(self):
        logger.info("hello")
        here = sys._getframe().f_lineno - 1
        m = LINE.match(self.terminal_lines()[-1])
        self.assertIsNotNone(m, self.terminal_lines()[-1])
        self.assertEqual((m["where"], int(m["line"]), m["msg"]), ("test_logging.py", here, "hello"))

    def test_conversation_lines_name_the_caller_not_the_helper(self):
        app._log("You:   hello Loopy")
        here = sys._getframe().f_lineno - 1
        m = LINE.match(self.terminal_lines()[-1])
        self.assertEqual((m["where"], int(m["line"])), ("test_logging.py", here))

    def test_the_log_file_has_the_date_too(self):
        logger.info("written to the file")
        text = self.file_text()
        self.assertRegex(text, r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3} \| INFO +\| test_logging\.py:\d+ \(\w+\) "
                               r"\| written to the file")

    def test_a_crash_in_a_background_thread_is_logged(self):
        def boom():
            raise RuntimeError("thread went bang")

        t = threading.Thread(target=boom, name="loopy-test")
        t.start()
        t.join()
        self.assertIn("[error in background thread loopy-test]", self.terminal.getvalue())
        self.assertIn("RuntimeError: thread went bang", self.file_text())

    def test_commands_log_through_loguru_too(self):
        import contextlib

        from loopy import cli
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch.dict(os.environ, {"LOOPY_LOG_FILE": ""}):
            self.assertEqual(cli.main(["voices"]), 0)
        lines = [line for line in out.getvalue().splitlines() if line]
        self.assertTrue(lines)
        for line in lines:
            self.assertRegex(line, r"^\d\d:\d\d:\d\d\.\d{3} \| INFO +\| cli\.py:\d+ +\| ")

    def test_no_log_file_when_switched_off(self):
        from loopy import log
        with mock.patch.dict(os.environ, {"LOOPY_LOG_FILE": ""}):
            self.assertIsNone(log.default_log_file())
        with mock.patch.dict(os.environ, {"LOOPY_LOG_FILE": str(self.dir / "x.log")}):
            self.assertEqual(log.default_log_file(), self.dir / "x.log")

    def test_quitting_says_why(self):
        fake_app = mock.Mock()
        app._quit(fake_app, "Ctrl+C in the terminal")
        fake_app.quit.assert_called_once()
        self.assertIn("── Quitting (Ctrl+C in the terminal) ──", self.terminal_lines()[-1])


if __name__ == "__main__":
    unittest.main()
