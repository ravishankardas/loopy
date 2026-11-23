"""Reported: on Ubuntu, Loopy couldn't be dragged. Ubuntu runs Wayland,
where apps can't move their own windows; Loopy switches to XWayland (when
Qt's X11 backend can load) or lets the compositor do the drag."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from loopy.app import pick_display_backend  # noqa: E402
from loopy.mascot import Mascot, system_drag  # noqa: E402

app = QApplication.instance() or QApplication([])
has_lib = lambda name: "/usr/lib/libxcb-cursor.so.0"  # noqa: E731
no_lib = lambda name: None  # noqa: E731


class PickDisplayBackend(unittest.TestCase):
    def test_ubuntu_wayland_switches_to_xwayland(self):
        env = {"WAYLAND_DISPLAY": "wayland-0", "XDG_SESSION_TYPE": "wayland"}
        self.assertIsNone(pick_display_backend(env, "linux", has_lib))
        self.assertEqual(env["QT_QPA_PLATFORM"], "xcb")

    def test_without_libxcb_cursor_it_stays_on_wayland_with_a_hint(self):
        env = {"WAYLAND_DISPLAY": "wayland-0"}
        hint = pick_display_backend(env, "linux", no_lib)
        self.assertNotIn("QT_QPA_PLATFORM", env)       # Qt's xcb backend wouldn't load: don't break startup
        self.assertIn("libxcb-cursor0", hint)

    def test_plain_x11_is_left_alone(self):
        env = {"DISPLAY": ":0", "XDG_SESSION_TYPE": "x11"}
        self.assertIsNone(pick_display_backend(env, "linux", has_lib))
        self.assertNotIn("QT_QPA_PLATFORM", env)

    def test_windows_and_mac_are_left_alone(self):
        for platform in ("win32", "darwin"):
            env = {}
            self.assertIsNone(pick_display_backend(env, platform, has_lib))
            self.assertEqual(env, {})

    def test_a_backend_the_user_picked_wins(self):
        env = {"WAYLAND_DISPLAY": "wayland-0", "QT_QPA_PLATFORM": "wayland"}
        pick_display_backend(env, "linux", has_lib)
        self.assertEqual(env["QT_QPA_PLATFORM"], "wayland")


class SystemDrag(unittest.TestCase):
    def test_only_wayland_hands_the_drag_to_the_compositor(self):
        m = Mascot()
        m.show()
        self.addCleanup(m.close)
        self.assertFalse(system_drag(m))   # offscreen (like Windows/X11): Loopy moves itself


if __name__ == "__main__":
    unittest.main()
