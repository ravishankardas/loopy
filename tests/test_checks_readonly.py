"""Guards that Loopy's checks stay read-only.

Checks run WITHOUT asking the user first, so they must never be able to
change anything on the computer. These tests fail if someone adds a write,
delete, process or shell call to checks.py, imports something that could do
that, or lets the model pass arguments into a check.

Run:  python -m unittest discover tests
"""

import ast
import inspect
import unittest
from pathlib import Path

from loopy import checks

SOURCE = Path(checks.__file__).read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)

# The only modules checks.py may use.
ALLOWED_IMPORTS = {"datetime", "os", "sys", "dataclasses", "typing", "collections", "psutil"}

# Every function/method/attribute call checks.py may make. Anything not listed
# here (open, os.remove, subprocess.run, psutil.Process(...).kill, ...) fails.
ALLOWED_CALLS = {
    # reading the clock
    "datetime.datetime.now", "now.strftime",
    # reading system state
    "psutil.sensors_battery", "psutil.cpu_percent", "psutil.virtual_memory", "psutil.disk_usage",
    "psutil.disk_partitions",  # lists drives; changes nothing
    "os.environ.get",
    # formatting and plumbing
    "round", "divmod", "next", "chr", "dataclass", "Check", "check.read", "_drive_space",
    "str.lstrip", "lstrip", "rstrip", "join",
}


def _dotted(node: ast.AST) -> str:
    """'psutil.disk_usage' for psutil.disk_usage(...), '...lstrip' for chained calls."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        inner = _dotted(node.value)
        return f"{inner}.{node.attr}" if inner else node.attr
    return ""


class ChecksAreReadOnly(unittest.TestCase):
    def test_only_allowed_imports(self):
        for node in ast.walk(TREE):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            for name in names:
                self.assertIn(name, ALLOWED_IMPORTS,
                              f"checks.py imports '{name}', which isn't on the read-only allowlist")

    def test_only_allowed_calls(self):
        for node in ast.walk(TREE):
            if not isinstance(node, ast.Call):
                continue
            name = _dotted(node.func)
            short = name.split(".")[-1]
            self.assertTrue(
                name in ALLOWED_CALLS or short in {"lstrip", "rstrip", "join"},
                f"checks.py line {node.lineno} calls '{name or ast.unparse(node.func)}', "
                "which isn't on the read-only allowlist",
            )

    def test_checks_take_no_arguments(self):
        # The model only picks a check by name; it must not be able to pass
        # in a path, command or anything else.
        for check in checks.CHECKS:
            params = inspect.signature(check.read).parameters
            self.assertEqual(len(params), 0, f"check '{check.name}' takes arguments: {list(params)}")

    def test_unknown_names_do_nothing(self):
        for bogus in ["delete", "rm -rf /", "../../etc", "time; shutdown", ""]:
            self.assertEqual(checks.read(bogus), "unknown")

    def test_every_check_returns_text(self):
        for name in checks.names():
            value = checks.read(name)
            self.assertIsInstance(value, str)
            self.assertTrue(value)


if __name__ == "__main__":
    unittest.main()
