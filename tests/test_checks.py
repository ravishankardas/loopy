"""Tests for what each read-only check reports, with fake system readings."""

import unittest
from collections import namedtuple
from unittest import mock

import psutil

from loopy import checks

Partition = namedtuple("Partition", "device mountpoint fstype opts")
Usage = namedtuple("Usage", "total used free percent")
Battery = namedtuple("Battery", "percent secsleft power_plugged")
GB = 2**30


class DiskCoversEveryDrive(unittest.TestCase):
    def setUp(self):
        self.parts = [
            Partition("C:\\", "C:\\", "NTFS", "rw,fixed"),
            Partition("D:\\", "D:\\", "NTFS", "rw,fixed"),
        ]
        self.usage = {"C:\\": Usage(325 * GB, 264 * GB, 61 * GB, 81.0),
                      "D:\\": Usage(149 * GB, 2 * GB, 147 * GB, 1.3)}

    def disk(self):
        with mock.patch.object(psutil, "disk_partitions", lambda all=False: self.parts), \
             mock.patch.object(psutil, "disk_usage", lambda m: self.usage[m]), \
             mock.patch.object(checks.sys, "platform", "win32"):
            return checks._disk()

    def test_reports_every_drive_with_its_letter(self):
        # Reported: Loopy only knew about C: and then guessed about the other drive.
        text = self.disk()
        self.assertIn("C drive: 61 GB free of 325 GB", text)
        self.assertIn("D drive: 147 GB free of 149 GB", text)

    def test_skips_cd_drives(self):
        self.parts.append(Partition("E:\\", "E:\\", "", "cdrom"))
        self.assertNotIn("E drive", self.disk())

    def test_unreadable_drive_doesnt_break_the_others(self):
        self.parts.append(Partition("F:\\", "F:\\", "", "rw,removable"))

        def usage(m):
            if m == "F:\\":
                raise PermissionError("no media")
            return self.usage[m]

        with mock.patch.object(psutil, "disk_partitions", lambda all=False: self.parts), \
             mock.patch.object(psutil, "disk_usage", usage), \
             mock.patch.object(checks.sys, "platform", "win32"):
            text = checks._disk()
        self.assertIn("C drive: 61 GB", text)
        self.assertIn("F drive: not readable", text)


class Battery_(unittest.TestCase):
    def battery(self, value):
        with mock.patch.object(psutil, "sensors_battery", lambda: value):
            return checks._battery()

    def test_plugged_in(self):
        self.assertEqual(self.battery(Battery(100, psutil.POWER_TIME_UNLIMITED, True)), "100%, plugged in")

    def test_on_battery_with_time_left(self):
        self.assertEqual(self.battery(Battery(57, 7800, False)),
                         "57%, on battery, about 2 hours 10 minutes left")

    def test_desktop_without_battery(self):
        self.assertIn("no battery", self.battery(None))


class SystemAndTime(unittest.TestCase):
    def test_system_has_real_percentages(self):
        mem = namedtuple("Mem", "total used percent")(8 * GB, 6 * GB, 75.0)
        with mock.patch.object(psutil, "cpu_percent", lambda interval=None: 20.0), \
             mock.patch.object(psutil, "virtual_memory", lambda: mem):
            self.assertEqual(checks._system(), "CPU 20% busy, memory 75% used (6.0 of 8.0 GB)")

    def test_time_has_no_leading_zero(self):
        self.assertRegex(checks._time(), r"^[1-9]\d?:\d\d [AP]M$")


if __name__ == "__main__":
    unittest.main()
