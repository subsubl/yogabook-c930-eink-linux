"""Real-time HID raw & evdev listener for Yoga Book C930 E-Ink touch & panel buttons.

Monitors /dev/hidraw* nodes belonging to 048d:8951:
  - Report 0x90: single-contact E-Ink touch (11-byte records, Lenovo USBHIDAPI.cpp format)
  - Report 0x10 / 0x20: vendor panel/Homebar/settings button reports on hidraw0
  - Report 0x0c: multitouch contact frames
Also monitors special/Fn/Settings keys on the ITE T-CON Keyboard evdev device.
"""

from __future__ import annotations

import glob
import os
import select
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .protocol import PANEL_HEIGHT, PANEL_WIDTH

REPORT_ID_TOUCH = 0x90
REPORT_ID_BTN_10 = 0x10
REPORT_ID_BTN_20 = 0x20
REPORT_ID_MT = 0x0C

# Linux struct input_event on 64-bit
INPUT_EVENT_FMT = "llHHi"
INPUT_EVENT_SIZE = struct.calcsize(INPUT_EVENT_FMT)
EV_KEY = 0x01


@dataclass
class EinkTouchEvent:
    """Parsed single-contact touch event in 1920x1080 E-Ink panel coordinates."""

    panel_x: int
    panel_y: int
    raw_x: int
    raw_y: int
    pressure: int
    is_down: bool
    is_up: bool
    is_new_action: bool
    contact_id: int


@dataclass
class EinkVendorEvent:
    """Vendor HID button/status event (e.g. report 0x10 or 0x20 from hidraw0)."""

    device: str
    report_id: int
    payload: bytes


def find_yogabook_hidraw_devices(vendor_only: bool = True) -> list[tuple[str, str]]:
    """Return [(dev_path, phys)] for hidraw nodes belonging to 048d:8951.

    By default (vendor_only=True), returns only the vendor E-Ink touch/button interface
    (input1 / hidraw0, which carries Report IDs 0x90, 0x10, 0x20) so we never flood
    on normal mouse/keyboard/touchpad traffic from input2/input3.
    """
    results: list[tuple[str, str]] = []
    for sys_dir in sorted(glob.glob("/sys/class/hidraw/hidraw*")):
        uevent_path = os.path.join(sys_dir, "device", "uevent")
        try:
            text = open(uevent_path, "r", encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        if "048D" not in text.upper() or "8951" not in text.upper():
            continue
        phys = ""
        for line in text.splitlines():
            if line.startswith("HID_PHYS="):
                phys = line.split("=", 1)[1]
                break
        if vendor_only and not phys.endswith("/input1"):
            continue
        dev_path = f"/dev/{os.path.basename(sys_dir)}"
        results.append((dev_path, phys))
    return results


def parse_report_0x90(
    data: bytes,
    swap_axes: bool = False,
    flip_x: bool = False,
    flip_y: bool = False,
) -> list[EinkTouchEvent]:
    """Parse 11-byte 0x90 single-contact touch records from hidraw0.

    Verified live on Yoga Book C930 (YB-J912):
      - Raw coordinates are in 4x native touch space (7680x4320 -> 1920x1080)
      - Top-right Settings icon reports raw=(~7467, ~73) -> panel=(~1866, ~18)
      - byte[4] == 1 on press/hold, byte[4] == 0 on release
    """
    events: list[EinkTouchEvent] = []
    rec_len = 11
    n_records = len(data) // rec_len
    if n_records == 0 and len(data) >= 11 and data[0] == REPORT_ID_TOUCH:
        n_records = 1

    for i in range(n_records):
        off = i * rec_len
        rec = data[off : off + rec_len]
        if len(rec) < 11 or rec[0] != REPORT_ID_TOUCH:
            continue
        contact_id = rec[2]
        mode_byte = rec[3]
        action_marker = rec[4]
        raw_x, raw_y, z = struct.unpack_from("<HHH", rec, 5)

        # Auto-scale if controller reports 4x native touch space (7680x4320)
        sx, sy = raw_x, raw_y
        if sx > PANEL_WIDTH + 64 or sy > PANEL_HEIGHT + 64:
            sx = sx // 4
            sy = sy // 4

        if swap_axes:
            px = (PANEL_WIDTH - sy) if flip_x else sy
            py = (PANEL_HEIGHT - sx) if flip_y else sx
        else:
            px = (PANEL_WIDTH - sx) if flip_x else sx
            py = (PANEL_HEIGHT - sy) if flip_y else sy

        px = max(0, min(PANEL_WIDTH - 1, px))
        py = max(0, min(PANEL_HEIGHT - 1, py))

        is_up = (mode_byte == 0x02) or (action_marker == 0)
        is_down = (mode_byte == 0x01) and (action_marker != 0)

        events.append(
            EinkTouchEvent(
                panel_x=px,
                panel_y=py,
                raw_x=raw_x,
                raw_y=raw_y,
                pressure=z,
                is_down=is_down,
                is_up=is_up,
                is_new_action=(action_marker == 0),
                contact_id=contact_id,
            )
        )
    return events


class EinkHidListener:
    """Background listener for E-Ink HID touch reports (0x90) and vendor buttons (0x10/0x20)."""

    def __init__(
        self,
        on_touch: Optional[Callable[[EinkTouchEvent], None]] = None,
        on_vendor: Optional[Callable[[EinkVendorEvent], None]] = None,
        on_raw: Optional[Callable[[str, bytes], None]] = None,
    ) -> None:
        self.on_touch = on_touch
        self.on_vendor = on_vendor
        self.on_raw = on_raw
        self.swap_axes = False
        self.flip_x = False
        self.flip_y = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_tap_ts = 0.0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> tuple[bool, str]:
        if self.is_running:
            return True, "HID listener already running."

        devs = find_yogabook_hidraw_devices()
        if not devs:
            return False, "No Yoga Book C930 hidraw devices (048d:8951) found."

        readable: list[tuple[str, str]] = []
        denied: list[str] = []
        for path, phys in devs:
            if os.access(path, os.R_OK):
                readable.append((path, phys))
            else:
                denied.append(path)

        if not readable:
            return (
                False,
                f"Permission denied on {', '.join(denied)}. "
                "Re-run ./install.sh to install the updated hidraw udev rule.",
            )

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(readable,), daemon=True, name="eink-hid-listener"
        )
        self._thread.start()
        names = ", ".join(os.path.basename(p) for p, _ in readable)
        return True, f"Listening on {names}"

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.6)
        self._thread = None

    def _run(self, devices: list[tuple[str, str]]) -> None:
        fds: dict[int, tuple[str, str]] = {}
        try:
            for path, phys in devices:
                try:
                    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
                    fds[fd] = (path, phys)
                except OSError:
                    continue

            if not fds:
                return

            while not self._stop.is_set():
                try:
                    ready, _, _ = select.select(list(fds.keys()), [], [], 0.2)
                except (OSError, ValueError):
                    break
                for fd in ready:
                    path, _phys = fds[fd]
                    try:
                        data = os.read(fd, 256)
                    except BlockingIOError:
                        continue
                    except OSError:
                        continue
                    if not data:
                        continue
                    self._handle_packet(path, data)
        finally:
            for fd in fds:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def _handle_packet(self, path: str, data: bytes) -> None:
        if self.on_raw:
            self.on_raw(os.path.basename(path), data)

        rid = data[0]
        if rid == REPORT_ID_TOUCH and self.on_touch:
            for ev in parse_report_0x90(
                data,
                swap_axes=self.swap_axes,
                flip_x=self.flip_x,
                flip_y=self.flip_y,
            ):
                # Debounce rapid duplicate down events slightly when needed by callers
                self.on_touch(ev)
        elif rid in (REPORT_ID_BTN_10, REPORT_ID_BTN_20) and self.on_vendor:
            now = time.monotonic()
            if now - self._last_tap_ts > 0.15:
                self._last_tap_ts = now
                self.on_vendor(
                    EinkVendorEvent(
                        device=os.path.basename(path),
                        report_id=rid,
                        payload=data[1:],
                    )
                )
