"""Shared opaque-box fixtures, USB BOT wire simulator, C oracles, and GTK3 harness."""

from __future__ import annotations

import atexit
import base64
import ctypes
import functools
import inspect
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from PIL import Image, ImageDraw

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

CLI_PATH = os.path.join(REPO_ROOT, "yogabook-eink-cli")
GUI_PATH = os.path.join(REPO_ROOT, "yogabook-eink-gui")
GLIDER_DIR = os.path.join(REPO_ROOT, "references", "Glider", "fw", "User")
GLIDER_TONE_LUT_C = os.path.join(GLIDER_DIR, "tone_lut.c")

from yogabook_eink import config as _config_mod  # noqa: E402
from yogabook_eink import daemon as _daemon_mod  # noqa: E402
from yogabook_eink import hid_listener as _hid_mod  # noqa: E402
from yogabook_eink import key_sound as _key_sound_mod  # noqa: E402
from yogabook_eink import protocol as _protocol_mod  # noqa: E402
from yogabook_eink import usb_lib as _usb_lib_mod  # noqa: E402
from yogabook_eink.protocol import (  # noqa: E402
    IMAGE_BUF_BASE,
    OP_DPY_AREA,
    OP_DYNAMIC_SETTING,
    OP_GET_DPY_STATUS,
    OP_GET_SYS,
    OP_LD_IMG_AREA2,
    OP_MODE_HELPER_AE,
    OP_READ_MEM,
    OP_READ_REG,
    OP_SCENARIO,
    OP_SET_HANDWR_REGION,
    OP_SET_TP_AREA,
    OP_SET_WAVEFORM,
    OP_WRITE_REG,
    PANEL_HEIGHT,
    PANEL_WIDTH,
    REG_DISPLAY_CFG,
    REG_PANEL_MODE,
    REG_PANEL_MODE_READY,
    USBC_HEADER,
    USBS_MAGIC,
)
from yogabook_eink.usb_lib import EP_IN, EP_OUT, LIBUSB_ERROR_TIMEOUT, UsbError  # noqa: E402


# ---------------------------------------------------------------------------
# Glider C Tone LUT Oracle (gcc -shared -fPIC)
# ---------------------------------------------------------------------------

_C_ORACLE_TMPDIR: Optional[str] = None
_C_ORACLE_LIB: Optional[ctypes.CDLL] = None


def _cleanup_c_oracle() -> None:
    global _C_ORACLE_TMPDIR
    if _C_ORACLE_TMPDIR and os.path.isdir(_C_ORACLE_TMPDIR):
        shutil.rmtree(_C_ORACLE_TMPDIR, ignore_errors=True)
        _C_ORACLE_TMPDIR = None


atexit.register(_cleanup_c_oracle)


def get_glider_c_oracle() -> Callable[[int, int], bytes]:
    """Compile references/Glider/fw/User/tone_lut.c once and return C oracle fn."""
    global _C_ORACLE_TMPDIR, _C_ORACLE_LIB
    if _C_ORACLE_LIB is None:
        tmpdir = tempfile.mkdtemp(prefix="yb_e2e_glider_")
        _C_ORACLE_TMPDIR = tmpdir
        so_path = os.path.join(tmpdir, "libtone_lut.so")
        cmd = [
            "gcc",
            "-shared",
            "-fPIC",
            "-O2",
            f"-I{GLIDER_DIR}",
            GLIDER_TONE_LUT_C,
            "-o",
            so_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        lib = ctypes.CDLL(so_path)
        lib.tone_lut_build_y.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
        ]
        lib.tone_lut_build_y.restype = None
        _C_ORACLE_LIB = lib

    lib = _C_ORACLE_LIB

    def _call(lightness: int, contrast: int) -> bytes:
        buf = (ctypes.c_uint8 * 256)()
        lib.tone_lut_build_y(int(lightness), int(contrast), buf)
        return bytes(buf)

    return _call


def float32_tone_lut_reference(lightness: int, contrast: int) -> bytes:
    """Pure-Python IEEE 754 float32 reference matching tone_lut.c."""
    f32 = lambda x: ctypes.c_float(x).value
    l_lut = [0.0, 0.3, 0.6, 1.0, 1.3, 1.6, 2.0]
    c_lut = [0.8, 1.0, 1.35, 1.8, 2.7, 5.0, 12.0, 255.0]
    l_idx = 0 if lightness < -3 else (6 if lightness > 3 else lightness + 3)
    c_idx = 0 if contrast < -1 else (7 if contrast > 6 else contrast + 1)
    lf = f32(l_lut[l_idx])
    cf = f32(c_lut[c_idx])
    bias = f32(lf - f32(1.0))
    out = bytearray(256)
    for i in range(256):
        norm = f32(f32(i) / f32(255.0))
        contrasted = f32(f32(0.5) + f32(f32(norm - f32(0.5)) * cf))
        if contrasted < 0.0:
            contrasted = f32(0.0)
        elif contrasted > 1.0:
            contrasted = f32(1.0)
        c2 = f32(contrasted * contrasted)
        shaped = f32(contrasted + f32(bias * f32(contrasted - c2)))
        if shaped < 0.0:
            shaped = f32(0.0)
        elif shaped > 1.0:
            shaped = f32(1.0)
        out[i] = int(f32(f32(shaped * f32(255.0)) + f32(0.5))) & 0xFF
    return bytes(out)


# ---------------------------------------------------------------------------
# Wire-Level USB BOT Recorder & MockLibUsbDevice
# ---------------------------------------------------------------------------


@dataclass
class UsbcCommandRecord:
    opcode: int
    dir_in: bool
    payload_sz: int
    addr: int
    arg1: int
    arg2: int
    arg3: int
    arg4: int
    raw_cbw: bytes
    payload: bytes = b""
    thread_id: int = field(default_factory=threading.get_ident)
    ts: float = field(default_factory=time.monotonic)


@dataclass
class LdImgRecord:
    addr: int
    buf_offset: int
    width: int
    height: int
    payload: bytes
    ts: float = field(default_factory=time.monotonic)


@dataclass
class DpyAreaRecord:
    mem_addr: int
    waveform: int
    x: int
    y: int
    width: int
    height: int
    pre_b1: bool
    post_b1: bool = False
    had_a9_between: bool = False
    fb_region_bytes: bytes = b""
    thread_id: int = field(default_factory=threading.get_ident)
    ts: float = field(default_factory=time.monotonic)


@dataclass
class WaveformSetRecord:
    arg1: int
    waveform: int
    ts: float = field(default_factory=time.monotonic)


class UsbBotRecorder:
    """Thread-safe wire-level ITE8951 USB BOT simulator and packet recorder."""

    def __init__(self, busy_polls_per_wait: int = 0, usbs_mode: str = "separate") -> None:
        self._lock = threading.RLock()
        self._busy_remaining = busy_polls_per_wait
        self.usbs_mode = usbs_mode  # "separate" | "coalesced" | "split"

        self.registers: dict[int, int] = {
            REG_PANEL_MODE: REG_PANEL_MODE_READY,
            REG_DISPLAY_CFG: 0x00260000,
        }
        self.scenario: int = 1
        self.framebuffer = bytearray([0xFF] * (PANEL_WIDTH * PANEL_HEIGHT))

        self.commands: list[UsbcCommandRecord] = []
        self.ld_img_ops: list[LdImgRecord] = []
        self.dpy_area_ops: list[DpyAreaRecord] = []
        self.waveform_set_ops: list[WaveformSetRecord] = []
        self.b1_count: int = 0
        self.scenario_history: list[tuple[int, int]] = []
        self.enter_app_count: int = 0
        self.exit_kb_count: int = 0
        self.open_count: int = 0
        self.close_count: int = 0

        self._error_hooks: dict[int, tuple[Optional[int], Callable[[int], Exception]]] = {}
        self._last_cmd: Optional[UsbcCommandRecord] = None
        self._a9_since_b1_or_a8: bool = False
        self._last_op_was_b1: bool = False
        self._pending_post_b1_dpy: Optional[DpyAreaRecord] = None
        self._active_transfer_thread: Optional[int] = None
        self.interleaved_transfer_detected: bool = False

    @property
    def busy_polls_per_wait(self) -> int:
        return self._busy_remaining

    @busy_polls_per_wait.setter
    def busy_polls_per_wait(self, value: int) -> None:
        self._busy_remaining = int(value)

    def inject_error_on_opcode(
        self,
        opcode: int,
        err: Optional[Exception | Callable[[int], Exception]] = None,
        count: Optional[int] = None,
    ) -> None:
        with self._lock:
            if err is None:
                factory = lambda op: UsbError(f"Simulated timeout on 0x{op:02x}", LIBUSB_ERROR_TIMEOUT)
            elif isinstance(err, Exception):
                factory = lambda _op, e=err: e
            else:
                factory = err
            self._error_hooks[opcode] = (count, factory)

    def clear_error_hooks(self) -> None:
        with self._lock:
            self._error_hooks.clear()

    def _maybe_raise_hook(self, opcode: int) -> None:
        hook = self._error_hooks.get(opcode)
        if hook is None:
            return
        rem, factory = hook
        if rem is not None:
            if rem <= 1:
                del self._error_hooks[opcode]
            else:
                self._error_hooks[opcode] = (rem - 1, factory)
        raise factory(opcode)

    def snapshot_counts(self) -> dict[str, int]:
        with self._lock:
            return {
                "commands": len(self.commands),
                "ld_img": len(self.ld_img_ops),
                "dpy_area": len(self.dpy_area_ops),
                "waveform_set": len(self.waveform_set_ops),
                "b1": self.b1_count,
            }

    def snapshot_region(self, x: int, y: int, w: int, h: int) -> bytes:
        """Read back packed bytes stored for region (x, y, w, h) row-by-row."""
        with self._lock:
            if x == 0 and w == PANEL_WIDTH:
                start = y * PANEL_WIDTH
                return bytes(self.framebuffer[start : start + w * h])
            buf = bytearray()
            fb_len = len(self.framebuffer)
            for dy in range(h):
                row_start = (y + dy) * PANEL_WIDTH + x
                if 0 <= row_start and row_start + w <= fb_len:
                    buf.extend(self.framebuffer[row_start : row_start + w])
            return bytes(buf)


class MockLibUsbDevice:
    """In-memory mock replacement for yogabook_eink.usb_lib.LibUsbDevice."""

    def __init__(self, recorder: UsbBotRecorder) -> None:
        self.recorder = recorder
        self._is_open = False
        self._pending_cbw: Optional[UsbcCommandRecord] = None
        self._in_queue: list[bytes] = []

    def open(self) -> None:
        if not self._is_open:
            self._is_open = True
            with self.recorder._lock:
                self.recorder.open_count += 1

    def close(self) -> None:
        if self._is_open:
            self._is_open = False
            with self.recorder._lock:
                self.recorder.close_count += 1

    @property
    def is_open(self) -> bool:
        return self._is_open

    def clear_halt(self, endpoint: int) -> None:
        pass

    def reset_device(self) -> None:
        self._pending_cbw = None
        self._in_queue.clear()

    def __enter__(self) -> "MockLibUsbDevice":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _make_usbs() -> bytes:
        return USBS_MAGIC + b"\x61\x89\x51\x89\x00\x00\x00\x00\x00"

    @staticmethod
    def _build_sys_info() -> bytes:
        # >10I (40B) + 48B padding + >2I (8B) + <4B (4B) + B (1B) + 11B = 112B
        part1 = struct.pack(
            ">10I",
            1,
            1,
            0x38393531,
            0x00010000,
            PANEL_WIDTH,
            PANEL_HEIGHT,
            0x00100000,
            IMAGE_BUF_BASE,
            25,
            0,
        )
        part2 = b"\x00" * 48
        part3 = struct.pack(">2I", 7680, 4320)
        part4 = struct.pack("<4B", 0, 2, 0, 72)
        part5 = bytes([1]) + b"\x00" * 11
        raw = part1 + part2 + part3 + part4 + part5
        assert len(raw) == 112
        return raw

    def _enqueue_in_response(self, data: bytes) -> None:
        usbs = self._make_usbs()
        mode = self.recorder.usbs_mode
        if data:
            if mode == "coalesced":
                self._in_queue.append(data + usbs)
                return
            self._in_queue.append(data)
        if mode == "split":
            # First read returns < 13 bytes not starting with USBS, second returns USBS
            self._in_queue.append(b"\x00\x00")
            self._in_queue.append(usbs)
        else:
            self._in_queue.append(usbs)

    def _finalize_command(self, rec: UsbcCommandRecord) -> None:
        r = self.recorder
        r.commands.append(rec)
        op = rec.opcode

        if op == OP_GET_SYS:
            self._enqueue_in_response(self._build_sys_info())
        elif op == OP_GET_DPY_STATUS:
            r.b1_count += 1
            busy = 0
            if r._busy_remaining > 0:
                busy = 1
                r._busy_remaining -= 1
            buf = bytearray(136)
            struct.pack_into("<H", buf, 4, busy)
            self._enqueue_in_response(bytes(buf))
            if busy == 0:
                r._last_op_was_b1 = True
                r._a9_since_b1_or_a8 = False
                if r._pending_post_b1_dpy is not None:
                    r._pending_post_b1_dpy.post_b1 = True
                    r._pending_post_b1_dpy = None
        elif op == OP_READ_REG:
            val = r.registers.get(rec.addr, 0)
            self._enqueue_in_response(struct.pack(">I", val & 0xFFFFFFFF))
        elif op == OP_WRITE_REG:
            if len(rec.payload) >= 4:
                val = struct.unpack(">I", rec.payload[:4])[0]
                r.registers[rec.addr] = val
            self._enqueue_in_response(b"")
        elif op == OP_SCENARIO:
            prev = r.scenario
            if rec.addr in (0x01010000, 0x01000000):
                r.scenario = 1
                r.exit_kb_count += 1
            elif rec.addr == 0x03000000:
                r.scenario = 0
                r.enter_app_count += 1
            elif rec.addr == 0x01030100:
                r.scenario = 3
            r.scenario_history.append((rec.addr, r.scenario))
            resp = bytes([0, prev if rec.addr == 0 else r.scenario, 0, 0])
            self._enqueue_in_response(resp)
        elif op == OP_SET_WAVEFORM:
            wf = (rec.arg1 >> 8) if rec.arg1 > 0xFF else rec.arg1
            r.waveform_set_ops.append(WaveformSetRecord(arg1=rec.arg1, waveform=wf))
            r._a9_since_b1_or_a8 = True
            r._last_op_was_b1 = False
            self._enqueue_in_response(b"\x00")
        elif op == OP_LD_IMG_AREA2:
            buf_offset = (rec.addr - IMAGE_BUF_BASE) & 0xFFFFFFFF
            width = rec.arg3
            height = rec.arg4
            if width > 0 and height > 0 and width * height == len(rec.payload):
                last_end = buf_offset + (height - 1) * PANEL_WIDTH + width
                if last_end > len(r.framebuffer):
                    raise AssertionError(
                        f"0xA8 out of T-CON RAM bounds: offset={buf_offset}, w={width}, h={height}, end={last_end}"
                    )
                for dy in range(height):
                    row_off = buf_offset + dy * PANEL_WIDTH
                    r.framebuffer[row_off : row_off + width] = rec.payload[
                        dy * width : (dy + 1) * width
                    ]
            else:
                if buf_offset + len(rec.payload) > len(r.framebuffer):
                    raise AssertionError(
                        f"0xA8 out of T-CON RAM bounds: offset={buf_offset}, len={len(rec.payload)}"
                    )
                r.framebuffer[buf_offset : buf_offset + len(rec.payload)] = rec.payload
            r.ld_img_ops.append(
                LdImgRecord(
                    addr=rec.addr,
                    buf_offset=buf_offset,
                    width=rec.arg3,
                    height=rec.arg4,
                    payload=rec.payload,
                )
            )
            r._a9_since_b1_or_a8 = False
            r._last_op_was_b1 = False
            self._enqueue_in_response(b"")
        elif op == OP_DPY_AREA:
            if len(rec.payload) != 24:
                raise AssertionError(f"0x94 payload must be 24 bytes, got {len(rec.payload)}")
            mem_addr, waveform, x, y, width, height = struct.unpack(">6I", rec.payload)
            fb_slice = r.snapshot_region(x, y, width, height)
            dpy = DpyAreaRecord(
                mem_addr=mem_addr,
                waveform=waveform,
                x=x,
                y=y,
                width=width,
                height=height,
                pre_b1=r._last_op_was_b1,
                post_b1=False,
                had_a9_between=r._a9_since_b1_or_a8,
                fb_region_bytes=fb_slice,
                thread_id=rec.thread_id,
            )
            r.dpy_area_ops.append(dpy)
            r._pending_post_b1_dpy = dpy
            r._last_op_was_b1 = False
            self._enqueue_in_response(b"")
        else:
            if rec.dir_in and rec.payload_sz > 0:
                self._enqueue_in_response(b"\x00" * rec.payload_sz)
            else:
                self._enqueue_in_response(b"")

    def bulk_write(self, data: bytes, endpoint: int = EP_OUT, timeout_ms: int = 5000) -> int:
        if not self._is_open:
            raise UsbError("USB device is not open")
        if endpoint != EP_OUT:
            raise AssertionError(f"Expected EP_OUT (0x02), got 0x{endpoint:02x}")

        tid = threading.get_ident()
        with self.recorder._lock:
            if self._pending_cbw is None:
                if len(data) != 31 or data[:8] != USBC_HEADER:
                    raise AssertionError(
                        f"Expected 31-byte USBC CBW, got len={len(data)} header={data[:8]!r}"
                    )
                _hdr, payload_sz, direction, _z, cdb_len, scsi_op, lun = struct.unpack_from(
                    "<8sIBBBBB", data, 0
                )
                if cdb_len != 0x10 or scsi_op != 0xFE or lun != 0x00:
                    raise AssertionError(f"Malformed USBC CDB metadata: {(cdb_len, scsi_op, lun)}")
                addr, opcode, arg1, arg2, arg3, arg4, _pad = struct.unpack_from(
                    ">IBHHHHB", data, 17
                )
                self.recorder._maybe_raise_hook(opcode)

                if (
                    self.recorder._active_transfer_thread is not None
                    and self.recorder._active_transfer_thread != tid
                ):
                    self.recorder.interleaved_transfer_detected = True
                self.recorder._active_transfer_thread = tid

                rec = UsbcCommandRecord(
                    opcode=opcode,
                    dir_in=(direction == 0x80),
                    payload_sz=payload_sz,
                    addr=addr,
                    arg1=arg1,
                    arg2=arg2,
                    arg3=arg3,
                    arg4=arg4,
                    raw_cbw=data,
                    thread_id=tid,
                )
                if not rec.dir_in and payload_sz > 0:
                    self._pending_cbw = rec
                else:
                    self._finalize_command(rec)
                    self.recorder._active_transfer_thread = None
                return len(data)
            else:
                rec = self._pending_cbw
                self._pending_cbw = None
                if len(data) != rec.payload_sz:
                    raise AssertionError(
                        f"Payload length mismatch for 0x{rec.opcode:02x}: expected {rec.payload_sz}, got {len(data)}"
                    )
                rec.payload = data
                self._finalize_command(rec)
                self.recorder._active_transfer_thread = None
                return len(data)

    def bulk_read(self, length: int, endpoint: int = EP_IN, timeout_ms: int = 5000) -> bytes:
        if not self._is_open:
            raise UsbError("USB device is not open")
        if endpoint != EP_IN:
            raise AssertionError(f"Expected EP_IN (0x81), got 0x{endpoint:02x}")
        with self.recorder._lock:
            if not self._in_queue:
                return b""
            pkt = self._in_queue.pop(0)
            return pkt[:length]


def assert_r5_protocol_invariants(
    test: unittest.TestCase,
    recorder: UsbBotRecorder,
    *,
    allow_standalone_0xa9: bool = True,
) -> None:
    """Verify every R5 invariant on recorded wire-level USB BOT transactions."""
    test.assertGreater(len(recorder.dpy_area_ops), 0, "Expected at least one 0x94 DPY_AREA blit")
    for idx, dpy in enumerate(recorder.dpy_area_ops):
        test.assertEqual(
            dpy.mem_addr,
            IMAGE_BUF_BASE,
            f"Blit #{idx} mem_addr=0x{dpy.mem_addr:08x}, expected 0x{IMAGE_BUF_BASE:08x}",
        )
        test.assertTrue(dpy.pre_b1, f"Blit #{idx} missing pre-blit 0xB1 wait_display_ready()")
        test.assertTrue(dpy.post_b1, f"Blit #{idx} missing post-blit 0xB1 wait_display_ready()")
        test.assertFalse(dpy.had_a9_between, f"Blit #{idx} had forbidden 0xA9 inside blit()")
        test.assertGreaterEqual(dpy.x, 0)
        test.assertGreaterEqual(dpy.y, 0)
        test.assertGreater(dpy.width, 0)
        test.assertGreater(dpy.height, 0)
        test.assertLessEqual(dpy.x + dpy.width, PANEL_WIDTH)
        test.assertLessEqual(dpy.y + dpy.height, PANEL_HEIGHT)
    if not allow_standalone_0xa9:
        test.assertEqual(
            len(recorder.waveform_set_ops),
            0,
            f"Unexpected 0xA9 SET_WAVEFORM calls: {recorder.waveform_set_ops}",
        )
    test.assertFalse(
        recorder.interleaved_transfer_detected,
        "Concurrent USB transfer interleaving detected across threads!",
    )


# ---------------------------------------------------------------------------
# In-Process Isolation Environment
# ---------------------------------------------------------------------------


class IsolatedEinkEnvironment:
    """Context manager isolating config, XDG_RUNTIME_DIR, USB, HID, key_sound, and Popen."""

    def __init__(self, busy_polls_per_wait: int = 0) -> None:
        self._tmpdir_obj: Optional[tempfile.TemporaryDirectory] = None
        self.tmpdir: str = ""
        self.home_dir: str = ""
        self.runtime_dir: str = ""
        self.config_dir: str = ""
        self.settings_path: str = ""
        self.usb_recorder = UsbBotRecorder(busy_polls_per_wait=busy_polls_per_wait)

        self.listeners: list[Any] = []
        self.daemon_popens: list[list[str]] = []
        self.sound_popens: list[list[str]] = []

        self._saved: dict[str, Any] = {}

    def __enter__(self) -> "IsolatedEinkEnvironment":
        self._tmpdir_obj = tempfile.TemporaryDirectory(prefix="yb_e2e_env_")
        self.tmpdir = self._tmpdir_obj.name
        self.home_dir = os.path.join(self.tmpdir, "home")
        self.runtime_dir = os.path.join(self.tmpdir, "run")
        os.makedirs(self.home_dir, exist_ok=True)
        os.makedirs(self.runtime_dir, exist_ok=True)
        self.config_dir = os.path.join(self.home_dir, ".config", "yogabook-eink")
        self.settings_path = os.path.join(self.config_dir, "settings.json")

        self._saved["HOME"] = os.environ.get("HOME")
        self._saved["XDG_RUNTIME_DIR"] = os.environ.get("XDG_RUNTIME_DIR")
        os.environ["HOME"] = self.home_dir
        os.environ["XDG_RUNTIME_DIR"] = self.runtime_dir

        # 1. Config isolation (both module globals and EinkSettings.load/save defaults)
        self._saved["CONFIG_DIR"] = _config_mod.CONFIG_DIR
        self._saved["CONFIG_PATH"] = _config_mod.CONFIG_PATH
        _config_mod.CONFIG_DIR = self.config_dir
        _config_mod.CONFIG_PATH = self.settings_path

        orig_load = _config_mod.EinkSettings.load
        orig_save = _config_mod.EinkSettings.save
        self._saved["EinkSettings.load"] = orig_load
        self._saved["EinkSettings.save"] = orig_save

        env_self = self
        load_sig = inspect.signature(orig_load)
        save_sig = inspect.signature(orig_save)

        @classmethod
        def _isolated_load(cls, *args, **kwargs):
            if "path" in load_sig.parameters and not args and "path" not in kwargs:
                kwargs["path"] = env_self.settings_path
            return orig_load.__func__(cls, *args, **kwargs)

        def _isolated_save(inst, *args, **kwargs):
            if "path" in save_sig.parameters and not args and "path" not in kwargs:
                kwargs["path"] = env_self.settings_path
            return orig_save(inst, *args, **kwargs)

        _config_mod.EinkSettings.load = _isolated_load  # type: ignore[assignment]
        _config_mod.EinkSettings.save = _isolated_save  # type: ignore[assignment]

        # 2. USB isolation
        self._saved["usb_lib.LibUsbDevice"] = _usb_lib_mod.LibUsbDevice
        self._saved["protocol.LibUsbDevice"] = _protocol_mod.LibUsbDevice

        def _usb_factory(*_args, **_kwargs) -> MockLibUsbDevice:
            return MockLibUsbDevice(env_self.usb_recorder)

        _usb_lib_mod.LibUsbDevice = _usb_factory  # type: ignore[misc]
        _protocol_mod.LibUsbDevice = _usb_factory  # type: ignore[misc]

        # 3. HID isolation
        self._saved["find_hidraw"] = _hid_mod.find_yogabook_hidraw_devices
        self._saved["gui_find_hidraw"] = None
        _hid_mod.find_yogabook_hidraw_devices = lambda vendor_only=True: []
        if "yogabook_eink.gui" in sys.modules:
            gui_mod = sys.modules["yogabook_eink.gui"]
            if hasattr(gui_mod, "find_yogabook_hidraw_devices"):
                self._saved["gui_find_hidraw"] = gui_mod.find_yogabook_hidraw_devices
                gui_mod.find_yogabook_hidraw_devices = lambda vendor_only=True: []

        self._saved["hid_start"] = _hid_mod.EinkHidListener.start
        self._saved["hid_stop"] = _hid_mod.EinkHidListener.stop
        self._saved["hid_is_running"] = _hid_mod.EinkHidListener.is_running

        def _mock_hid_start(listener_self) -> tuple[bool, str]:
            if listener_self not in env_self.listeners:
                env_self.listeners.append(listener_self)
            listener_self._mock_running = True
            return True, "Listening on mock_hidraw0"

        def _mock_hid_stop(listener_self) -> None:
            listener_self._mock_running = False
            listener_self._stop.set()

        @property
        def _mock_is_running(listener_self) -> bool:
            return bool(getattr(listener_self, "_mock_running", False))

        _hid_mod.EinkHidListener.start = _mock_hid_start  # type: ignore[assignment]
        _hid_mod.EinkHidListener.stop = _mock_hid_stop  # type: ignore[assignment]
        _hid_mod.EinkHidListener.is_running = _mock_is_running  # type: ignore[assignment]

        # 4. Key sound isolation (replace module attribute, never mutate global subprocess.Popen)
        self._saved["find_ite_kb"] = _key_sound_mod.find_ite_keyboard_event_path
        self._saved["key_sound_subproc"] = _key_sound_mod.subprocess
        self._saved["sound_start"] = _key_sound_mod.KeyboardSoundDaemon.start
        self._saved["sound_stop"] = _key_sound_mod.KeyboardSoundDaemon.stop
        _key_sound_mod.find_ite_keyboard_event_path = lambda: None
        if "yogabook_eink.gui" in sys.modules:
            gui_mod = sys.modules["yogabook_eink.gui"]
            if hasattr(gui_mod, "find_ite_keyboard_event_path"):
                self._saved["gui_find_ite_kb"] = gui_mod.find_ite_keyboard_event_path
                gui_mod.find_ite_keyboard_event_path = lambda: None

        def _mock_sound_start(sound_self) -> tuple[bool, str]:
            sound_self._mock_running = True
            return True, "Mock key sound listener active"

        def _mock_sound_stop(sound_self) -> None:
            sound_self._mock_running = False
            sound_self._stop_event.set()
            if getattr(sound_self, "_thread", None) and sound_self._thread.is_alive():
                sound_self._thread.join(timeout=0.5)
            sound_self._thread = None

        _key_sound_mod.KeyboardSoundDaemon.start = _mock_sound_start  # type: ignore[assignment]
        _key_sound_mod.KeyboardSoundDaemon.stop = _mock_sound_stop  # type: ignore[assignment]

        def _mock_sound_popen(cmd, *args, **kwargs):
            env_self.sound_popens.append(list(cmd))

            class _DummyProc:
                def poll(self):
                    return 0

                def wait(self, timeout=None):
                    return 0

            return _DummyProc()

        class _SubprocProxy:
            def __init__(self, popen_fn):
                self.Popen = popen_fn

            def __getattr__(self, name):
                return getattr(subprocess, name)

        _key_sound_mod.subprocess = _SubprocProxy(_mock_sound_popen)  # type: ignore[assignment]

        # 5. Daemon Popen isolation (replace module attribute, never mutate global subprocess.Popen)
        self._saved["daemon_subproc"] = _daemon_mod.subprocess

        def _mock_daemon_popen(cmd, *args, **kwargs):
            env_self.daemon_popens.append(list(cmd))

            class _DummyProc:
                pid = 99999

                def poll(self):
                    return 0

            return _DummyProc()

        _daemon_mod.subprocess = _SubprocProxy(_mock_daemon_popen)  # type: ignore[assignment]

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for listener in list(self.listeners):
            try:
                listener.stop()
            except Exception:
                pass
        self.listeners.clear()

        # Restore patches in reverse
        _daemon_mod.subprocess = self._saved["daemon_subproc"]
        _key_sound_mod.subprocess = self._saved["key_sound_subproc"]
        _key_sound_mod.KeyboardSoundDaemon.stop = self._saved["sound_stop"]
        _key_sound_mod.KeyboardSoundDaemon.start = self._saved["sound_start"]
        _key_sound_mod.find_ite_keyboard_event_path = self._saved["find_ite_kb"]
        if "gui_find_ite_kb" in self._saved and "yogabook_eink.gui" in sys.modules:
            sys.modules["yogabook_eink.gui"].find_ite_keyboard_event_path = self._saved["gui_find_ite_kb"]

        _hid_mod.EinkHidListener.start = self._saved["hid_start"]
        _hid_mod.EinkHidListener.stop = self._saved["hid_stop"]
        _hid_mod.EinkHidListener.is_running = self._saved["hid_is_running"]
        _hid_mod.find_yogabook_hidraw_devices = self._saved["find_hidraw"]
        if self._saved.get("gui_find_hidraw") is not None and "yogabook_eink.gui" in sys.modules:
            sys.modules["yogabook_eink.gui"].find_yogabook_hidraw_devices = self._saved["gui_find_hidraw"]

        _protocol_mod.LibUsbDevice = self._saved["protocol.LibUsbDevice"]
        _usb_lib_mod.LibUsbDevice = self._saved["usb_lib.LibUsbDevice"]

        _config_mod.EinkSettings.load = self._saved["EinkSettings.load"]
        _config_mod.EinkSettings.save = self._saved["EinkSettings.save"]
        _config_mod.CONFIG_DIR = self._saved["CONFIG_DIR"]
        _config_mod.CONFIG_PATH = self._saved["CONFIG_PATH"]

        for k in ("HOME", "XDG_RUNTIME_DIR"):
            if self._saved[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = self._saved[k]

        if self._tmpdir_obj is not None:
            self._tmpdir_obj.cleanup()
            self._tmpdir_obj = None

    def write_settings_json(self, data: dict[str, Any] | str) -> str:
        os.makedirs(self.config_dir, exist_ok=True)
        if isinstance(data, str):
            with open(self.settings_path, "w", encoding="utf-8") as f:
                f.write(data)
        else:
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        return self.settings_path

    def read_settings_json(self) -> dict[str, Any]:
        with open(self.settings_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def make_touch_packet_0x90(
        self,
        panel_x: int,
        panel_y: int,
        *,
        is_down: bool = True,
        pressure: int = 40,
        contact_id: int = 0,
    ) -> bytes:
        # Match parse_report_0x90 4x scaling threshold (> PANEL_WIDTH + 64 or > PANEL_HEIGHT + 64)
        if panel_x * 4 > PANEL_WIDTH + 64 or panel_y * 4 > PANEL_HEIGHT + 64:
            rx, ry = panel_x * 4, panel_y * 4
        else:
            rx, ry = panel_x, panel_y
        mode_byte = 0x01 if is_down else 0x02
        action_marker = 0x01 if is_down else 0x00
        return struct.pack("<BBBBBHHH", 0x90, 0x00, contact_id, mode_byte, action_marker, rx, ry, pressure)

    def inject_touch(
        self,
        panel_x: int,
        panel_y: int,
        *,
        is_down: bool = True,
        pressure: int = 40,
        contact_id: int = 0,
        listener: Any = None,
    ) -> None:
        pkt = self.make_touch_packet_0x90(
            panel_x, panel_y, is_down=is_down, pressure=pressure, contact_id=contact_id
        )
        targets = [listener] if listener is not None else [
            l for l in self.listeners if l.is_running
        ]
        for l in targets:
            l._handle_packet("/dev/hidraw0", pkt)

    def inject_vendor(self, report_id: int = 0x10, payload: bytes = b"\x01\x00", listener: Any = None) -> None:
        pkt = bytes([report_id]) + payload
        targets = [listener] if listener is not None else [
            l for l in self.listeners if l.is_running
        ]
        for l in targets:
            l._handle_packet("/dev/hidraw0", pkt)


# ---------------------------------------------------------------------------
# Subprocess CLI / GUI Isolation via sitecustomize.py
# ---------------------------------------------------------------------------

_SITECUSTOMIZE_CODE = r'''
import base64
import json
import os
import struct
import sys

_USB_LOG = os.environ.get("YOGABOOK_E2E_USB_LOG")
_FAIL_OP = os.environ.get("YOGABOOK_E2E_FAIL_OPCODE")
_GUI_AUTOQUIT_MS = os.environ.get("YOGABOOK_E2E_GUI_AUTOQUIT_MS")

if _USB_LOG:
    import yogabook_eink.usb_lib as usb_lib
    import yogabook_eink.protocol as protocol

    _USBC_HEADER = b"USBC\x61\x89\x51\x89"
    _USBS_MAGIC = b"USBS"
    _IMAGE_BUF_BASE = 0x00382F30
    _PANEL_W = 1920
    _PANEL_H = 1080
    _STATE = {
        "scenario": 1,
        "regs": {0x18001224: 0x80000000, 0x18001138: 0x00260000},
        "fail_used": False,
    }

    def _log_op(rec):
        with open(_USB_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    class _SubprocMockUsb:
        def __init__(self, *args, **kwargs):
            self._open = False
            self._pending = None
            self._inq = []

        def open(self):
            self._open = True

        def close(self):
            self._open = False

        @property
        def is_open(self):
            return self._open

        def clear_halt(self, ep):
            pass

        def reset_device(self):
            self._inq.clear()

        def __enter__(self):
            self.open()
            return self

        def __exit__(self, *a):
            self.close()

        def _usbs(self):
            return _USBS_MAGIC + b"\x61\x89\x51\x89\x00\x00\x00\x00\x00"

        def _finish(self, rec):
            op = rec["op"]
            if _FAIL_OP and not _STATE["fail_used"]:
                target_op = int(_FAIL_OP, 0)
                if op == target_op or target_op == -1:
                    _STATE["fail_used"] = True
                    raise usb_lib.UsbError(f"Simulated timeout on 0x{op:02x}", usb_lib.LIBUSB_ERROR_TIMEOUT)

            payload = rec["payload"]
            log_entry = {
                "op": op,
                "addr": rec["addr"],
                "arg1": rec["arg1"],
                "arg2": rec["arg2"],
                "arg3": rec["arg3"],
                "arg4": rec["arg4"],
                "payload_len": len(payload),
            }
            if op == 0x80:
                p1 = struct.pack(">10I", 1, 1, 0x38393531, 0x10000, _PANEL_W, _PANEL_H, 0x100000, _IMAGE_BUF_BASE, 25, 0)
                raw = p1 + b"\x00" * 48 + struct.pack(">2I", 7680, 4320) + struct.pack("<4B", 0, 2, 0, 72) + bytes([1]) + b"\x00" * 11
                self._inq.extend([raw, self._usbs()])
            elif op == 0xB1:
                self._inq.extend([b"\x00" * 136, self._usbs()])
            elif op == 0x83:
                val = _STATE["regs"].get(rec["addr"], 0)
                self._inq.extend([struct.pack(">I", val), self._usbs()])
            elif op == 0x84:
                if len(payload) >= 4:
                    _STATE["regs"][rec["addr"]] = struct.unpack(">I", payload[:4])[0]
                self._inq.append(self._usbs())
            elif op == 0xA6:
                addr = rec["addr"]
                prev = _STATE["scenario"]
                if addr in (0x01010000, 0x01000000):
                    _STATE["scenario"] = 1
                elif addr == 0x03000000:
                    _STATE["scenario"] = 0
                elif addr == 0x01030100:
                    _STATE["scenario"] = 3
                log_entry["scenario"] = _STATE["scenario"]
                self._inq.extend([bytes([0, prev if addr == 0 else _STATE["scenario"], 0, 0]), self._usbs()])
            elif op == 0xA8:
                log_entry["buf_offset"] = (rec["addr"] - _IMAGE_BUF_BASE) & 0xFFFFFFFF
                log_entry["payload_b64"] = base64.b64encode(payload).decode("ascii")
                self._inq.append(self._usbs())
            elif op == 0x94:
                mem_addr, wf, x, y, w, h = struct.unpack(">6I", payload)
                log_entry["dpy"] = {"mem_addr": mem_addr, "waveform": wf, "x": x, "y": y, "w": w, "h": h}
                self._inq.append(self._usbs())
            elif op == 0xA9:
                log_entry["waveform"] = (rec["arg1"] >> 8) if rec["arg1"] > 0xFF else rec["arg1"]
                self._inq.extend([b"\x00", self._usbs()])
            else:
                if rec["dir_in"] and rec["payload_sz"] > 0:
                    self._inq.extend([b"\x00" * rec["payload_sz"], self._usbs()])
                else:
                    self._inq.append(self._usbs())
            _log_op(log_entry)

        def bulk_write(self, data, endpoint=0x02, timeout_ms=5000):
            if not self._open:
                raise usb_lib.UsbError("not open")
            if self._pending is None:
                _hdr, payload_sz, direction, _z, _cl, _so, _lun = struct.unpack_from("<8sIBBBBB", data, 0)
                addr, op, a1, a2, a3, a4, _p = struct.unpack_from(">IBHHHHB", data, 17)
                rec = {"op": op, "dir_in": (direction == 0x80), "payload_sz": payload_sz, "addr": addr, "arg1": a1, "arg2": a2, "arg3": a3, "arg4": a4, "payload": b""}
                if not rec["dir_in"] and payload_sz > 0:
                    self._pending = rec
                else:
                    self._finish(rec)
                return len(data)
            else:
                rec = self._pending
                self._pending = None
                rec["payload"] = data
                self._finish(rec)
                return len(data)

        def bulk_read(self, length, endpoint=0x81, timeout_ms=5000):
            if not self._open:
                raise usb_lib.UsbError("not open")
            if not self._inq:
                return b""
            return self._inq.pop(0)[:length]

    usb_lib.LibUsbDevice = _SubprocMockUsb
    protocol.LibUsbDevice = _SubprocMockUsb

if _GUI_AUTOQUIT_MS:
    import yogabook_eink.hid_listener as _hl
    import yogabook_eink.key_sound as _ks
    _hl.find_yogabook_hidraw_devices = lambda vendor_only=True: []
    _ks.find_ite_keyboard_event_path = lambda: None
    import gi
    gi.require_version("Gtk", "3.0")
    from gi.repository import GLib, Gtk
    _orig_gtk_main = Gtk.main

    def _smoke_main():
        ms = int(_GUI_AUTOQUIT_MS)
        def _quit():
            for w in Gtk.Window.list_toplevels():
                w.destroy()
            return False
        GLib.timeout_add(ms, _quit)
        _orig_gtk_main()

    Gtk.main = _smoke_main
'''


@dataclass
class CliRunResult:
    returncode: int
    stdout: str
    stderr: str
    ops: list[dict[str, Any]]
    dpy_area_ops: list[dict[str, Any]]
    ld_img_ops: list[dict[str, Any]]
    waveform_set_ops: list[dict[str, Any]]
    b1_ops: list[dict[str, Any]]
    framebuffer: bytes


def run_cli_isolated(
    args: list[str],
    home_dir: str,
    *,
    runtime_dir: Optional[str] = None,
    fail_opcode: Optional[int] = None,
    extra_env: Optional[dict[str, str]] = None,
    timeout: float = 60.0,
) -> CliRunResult:
    """Execute yogabook-eink-cli in an isolated subprocess with mocked libusb BOT shim."""
    with tempfile.TemporaryDirectory(prefix="yb_e2e_cli_") as shim_dir:
        with open(os.path.join(shim_dir, "sitecustomize.py"), "w", encoding="utf-8") as f:
            f.write(_SITECUSTOMIZE_CODE)
        trace_path = os.path.join(shim_dir, "usb_trace.jsonl")
        env = dict(os.environ)
        env["HOME"] = home_dir
        env["XDG_CONFIG_HOME"] = os.path.join(home_dir, ".config")
        env["XDG_RUNTIME_DIR"] = runtime_dir or os.path.join(home_dir, "run")
        os.makedirs(env["XDG_RUNTIME_DIR"], exist_ok=True)
        env["PYTHONPATH"] = f"{shim_dir}{os.pathsep}{REPO_ROOT}"
        env["YOGABOOK_E2E_USB_LOG"] = trace_path
        if fail_opcode is not None:
            env["YOGABOOK_E2E_FAIL_OPCODE"] = str(fail_opcode)
        else:
            env.pop("YOGABOOK_E2E_FAIL_OPCODE", None)
        env.pop("YOGABOOK_E2E_GUI_AUTOQUIT_MS", None)
        if extra_env:
            env.update(extra_env)

        proc = subprocess.run(
            [sys.executable, CLI_PATH, *args],
            env=env,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        ops: list[dict[str, Any]] = []
        dpy_ops: list[dict[str, Any]] = []
        ld_ops: list[dict[str, Any]] = []
        wf_ops: list[dict[str, Any]] = []
        b1_ops: list[dict[str, Any]] = []
        fb = bytearray([0xFF] * (PANEL_WIDTH * PANEL_HEIGHT))

        if os.path.exists(trace_path):
            with open(trace_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    ops.append(rec)
                    op = rec["op"]
                    if op == OP_DPY_AREA:
                        dpy_ops.append(rec["dpy"])
                    elif op == OP_LD_IMG_AREA2:
                        raw = base64.b64decode(rec["payload_b64"])
                        off = rec["buf_offset"]
                        width = rec.get("arg3", 0)
                        height = rec.get("arg4", 0)
                        if width > 0 and height > 0 and width * height == len(raw):
                            last_end = off + (height - 1) * PANEL_WIDTH + width
                            if last_end > len(fb):
                                raise AssertionError(
                                    f"0xA8 out of T-CON RAM bounds: offset={off}, w={width}, h={height}, end={last_end}"
                                )
                            for dy in range(height):
                                row_off = off + dy * PANEL_WIDTH
                                fb[row_off : row_off + width] = raw[
                                    dy * width : (dy + 1) * width
                                ]
                        else:
                            if off + len(raw) > len(fb):
                                raise AssertionError(
                                    f"0xA8 out of T-CON RAM bounds: offset={off}, len={len(raw)}"
                                )
                            fb[off : off + len(raw)] = raw
                        rec["payload"] = raw
                        ld_ops.append(rec)
                    elif op == OP_SET_WAVEFORM:
                        wf_ops.append(rec)
                    elif op == OP_GET_DPY_STATUS:
                        b1_ops.append(rec)

        return CliRunResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            ops=ops,
            dpy_area_ops=dpy_ops,
            ld_img_ops=ld_ops,
            waveform_set_ops=wf_ops,
            b1_ops=b1_ops,
            framebuffer=bytes(fb),
        )


def assert_cli_r5_invariants(test: unittest.TestCase, res: CliRunResult) -> None:
    """Verify R5 invariants on a CLI subprocess wire trace."""
    test.assertGreater(len(res.dpy_area_ops), 0)
    for dpy in res.dpy_area_ops:
        test.assertEqual(dpy["mem_addr"], IMAGE_BUF_BASE)
    last_op_was_b1 = False
    in_blit_window = False
    a9_since_b1_or_a8 = False
    pending_post_b1 = False
    dpy_idx = 0
    for rec in res.ops:
        op = rec["op"]
        if op == OP_GET_DPY_STATUS:
            last_op_was_b1 = True
            if pending_post_b1:
                pending_post_b1 = False
                in_blit_window = False
                a9_since_b1_or_a8 = False
            elif not in_blit_window:
                a9_since_b1_or_a8 = False
        elif op == OP_LD_IMG_AREA2:
            test.assertFalse(
                pending_post_b1,
                f"CLI Blit #{dpy_idx - 1} missing post-blit 0xB1 wait_display_ready()",
            )
            in_blit_window = True
            a9_since_b1_or_a8 = False
            last_op_was_b1 = False
        elif op == OP_SET_WAVEFORM:
            if in_blit_window or pending_post_b1:
                blit_num = dpy_idx - 1 if pending_post_b1 else dpy_idx
                test.fail(f"CLI Blit #{blit_num} had forbidden 0xA9 inside blit()")
            a9_since_b1_or_a8 = True
            last_op_was_b1 = False
        elif op == OP_DPY_AREA:
            test.assertFalse(
                pending_post_b1,
                f"CLI Blit #{dpy_idx - 1} missing post-blit 0xB1 wait_display_ready()",
            )
            test.assertTrue(
                last_op_was_b1,
                f"CLI Blit #{dpy_idx} missing pre-blit 0xB1 wait_display_ready()",
            )
            test.assertFalse(
                a9_since_b1_or_a8,
                f"CLI Blit #{dpy_idx} had forbidden 0xA9 inside blit()",
            )
            in_blit_window = True
            pending_post_b1 = True
            last_op_was_b1 = False
            dpy_idx += 1
        else:
            last_op_was_b1 = False
    test.assertFalse(
        pending_post_b1,
        f"CLI Blit #{dpy_idx - 1} missing post-blit 0xB1 wait_display_ready()",
    )


def start_ownership_server(srv: Any, timeout: float = 2.0) -> Any:
    """Start an OwnershipServer and wait until its Unix socket is bound."""
    from yogabook_eink.ownership import get_socket_path

    sock_path = getattr(srv, "sock_path", None) or get_socket_path()
    srv.start()
    if not wait_until(lambda: os.path.exists(sock_path), timeout=timeout, step=0.005):
        srv.stop()
        raise AssertionError(f"OwnershipServer socket failed to bind within {timeout}s: {sock_path}")
    return srv


# ---------------------------------------------------------------------------
# Headless GTK3 Display Harness & Dynamic Widget Discovery
# ---------------------------------------------------------------------------


class GtkDisplaySession:
    """Ensures GTK3 has a valid display (native session or ephemeral broadwayd)."""

    _broadway_proc: Optional[subprocess.Popen] = None
    _broadway_dir: Optional[str] = None
    _ready: bool = False

    @classmethod
    def _probe_display_in_subprocess(cls, env: Optional[dict[str, str]] = None) -> bool:
        probe_env = env if env is not None else os.environ
        has_display = bool(
            probe_env.get("DISPLAY")
            or probe_env.get("WAYLAND_DISPLAY")
            or (probe_env.get("GDK_BACKEND") == "broadway" and probe_env.get("BROADWAY_DISPLAY"))
        )
        if not has_display:
            return False
        code = (
            "import gi; gi.require_version('Gtk', '3.0'); "
            "from gi.repository import Gtk; "
            "ok, _ = Gtk.init_check([]); "
            "raise SystemExit(0 if ok else 1)"
        )
        try:
            res = subprocess.run(
                [sys.executable, "-c", code],
                env=dict(probe_env),
                capture_output=True,
                timeout=5.0,
            )
            return res.returncode == 0
        except Exception:
            return False

    @classmethod
    def ensure_ready(cls) -> dict[str, str]:
        if cls._ready:
            return dict(os.environ)

        # Probe any session display in a child process BEFORE importing gi.repository.Gtk
        # so PyGObject never caches initialized=False in the current process.
        if cls._probe_display_in_subprocess():
            import gi

            gi.require_version("Gtk", "3.0")
            from gi.repository import Gtk

            ok, _ = Gtk.init_check([])
            if ok:
                cls._ready = True
                return dict(os.environ)

        # Fallback: launch ephemeral broadwayd BEFORE first gi.repository.Gtk import
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        disp = f":{(port % 80) + 10}"
        bdir = tempfile.mkdtemp(prefix="yb_e2e_broadway_")
        cls._broadway_dir = bdir
        b_env = dict(os.environ)
        b_env["XDG_RUNTIME_DIR"] = bdir
        b_env.pop("DISPLAY", None)
        b_env.pop("WAYLAND_DISPLAY", None)
        proc = subprocess.Popen(
            ["broadwayd", disp, "-p", str(port)],
            env=b_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cls._broadway_proc = proc
        atexit.register(cls.shutdown)
        os.environ.pop("DISPLAY", None)
        os.environ.pop("WAYLAND_DISPLAY", None)
        os.environ["GDK_BACKEND"] = "broadway"
        os.environ["BROADWAY_DISPLAY"] = disp
        prev_xdg = os.environ.get("XDG_RUNTIME_DIR")
        os.environ["XDG_RUNTIME_DIR"] = bdir
        try:
            for _ in range(30):
                if proc.poll() is not None:
                    break
                if cls._probe_display_in_subprocess():
                    break
                time.sleep(0.05)
            import gi

            gi.require_version("Gtk", "3.0")
            from gi.repository import Gtk

            ok, _ = Gtk.init_check([])
            if ok:
                cls._ready = True
                return dict(os.environ)
        finally:
            if prev_xdg is None:
                os.environ.pop("XDG_RUNTIME_DIR", None)
            else:
                os.environ["XDG_RUNTIME_DIR"] = prev_xdg

        cls.shutdown()
        raise RuntimeError("Failed to initialize GTK3 via session display or broadwayd")

    @classmethod
    def shutdown(cls) -> None:
        if cls._broadway_proc is not None:
            try:
                cls._broadway_proc.terminate()
                cls._broadway_proc.wait(timeout=2)
            except Exception:
                try:
                    cls._broadway_proc.kill()
                    cls._broadway_proc.wait(timeout=1)
                except Exception:
                    pass
            cls._broadway_proc = None
        if cls._broadway_dir and os.path.isdir(cls._broadway_dir):
            shutil.rmtree(cls._broadway_dir, ignore_errors=True)
            cls._broadway_dir = None
        cls._ready = False


def pump_gtk_events(max_iters: int = 250) -> None:
    from gi.repository import GLib

    ctx = GLib.MainContext.default()
    for _ in range(max_iters):
        if not ctx.pending():
            break
        ctx.iteration(False)


def wait_until(predicate: Callable[[], bool], timeout: float = 3.0, step: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pump_gtk_events()
        if predicate():
            return True
        time.sleep(step)
    pump_gtk_events()
    return predicate()


def open_isolated_window(env: IsolatedEinkEnvironment) -> tuple[Any, Callable[[], None]]:
    """Instantiate YogaBookEinkWindow safely under GTK3 and return (win, close_fn)."""
    GtkDisplaySession.ensure_ready()
    from gi.repository import Gtk
    from yogabook_eink.gui import YogaBookEinkWindow

    orig_main_quit = Gtk.main_quit

    def _safe_main_quit(*_args) -> None:
        if Gtk.main_level() > 0:
            orig_main_quit()

    Gtk.main_quit = _safe_main_quit  # type: ignore[assignment]
    win = YogaBookEinkWindow()
    pump_gtk_events()
    # Wait for initial _refresh_status_async thread to finish cleanly
    wait_until(lambda: not win._busy and win.lbl_conn.get_text() != "Checking...", timeout=2.0)
    pump_gtk_events()

    def _close() -> None:
        try:
            win._stop_active_app()
        except Exception:
            pass
        try:
            win.destroy()
        except Exception:
            pass
        pump_gtk_events()
        Gtk.main_quit = orig_main_quit  # type: ignore[assignment]

    return win, _close


def iter_gtk_tree(root: Any) -> list[Any]:
    """Traverse all widgets in a GTK3 container tree including Notebook pages."""
    from gi.repository import Gtk

    found: list[Any] = []
    stack = [root]
    seen: set[int] = set()
    while stack:
        w = stack.pop(0)
        if id(w) in seen:
            continue
        seen.add(id(w))
        found.append(w)
        if isinstance(w, Gtk.Notebook):
            for i in range(w.get_n_pages()):
                page = w.get_nth_page(i)
                if page is not None:
                    stack.append(page)
        if isinstance(w, Gtk.Container):
            for child in w.get_children():
                stack.append(child)
    return found


def find_gtk_widgets(root: Any, predicate: Callable[[Any], bool]) -> list[Any]:
    return [w for w in iter_gtk_tree(root) if predicate(w)]


def find_gtk_button(root: Any, label_substr: str) -> Any:
    from gi.repository import Gtk

    needle = label_substr.lower()
    for w in iter_gtk_tree(root):
        if isinstance(w, Gtk.Button):
            lbl = (w.get_label() or "").lower()
            if needle in lbl:
                return w
    raise AssertionError(f"Could not find Gtk.Button matching {label_substr!r}")


def find_gtk_scale(root: Any, keyword: str, min_val: float, max_val: float) -> Any:
    """Dynamically locate a Gtk.Scale / Gtk.SpinButton by nearby label or adjustment bounds."""
    from gi.repository import Gtk

    needle = keyword.lower()
    all_widgets = iter_gtk_tree(root)

    # First try matching via sibling Gtk.Label inside same parent container
    for w in all_widgets:
        if isinstance(w, Gtk.Label) and needle in (w.get_text() or "").lower():
            parent = w.get_parent()
            if parent is not None:
                for sib in iter_gtk_tree(parent):
                    if isinstance(sib, (Gtk.Scale, Gtk.SpinButton)):
                        adj = sib.get_adjustment()
                        if abs(adj.get_lower() - min_val) < 1e-6 and abs(adj.get_upper() - max_val) < 1e-6:
                            return sib

    # Fallback: match exact adjustment bounds across all Scales/SpinButtons
    for w in all_widgets:
        if isinstance(w, (Gtk.Scale, Gtk.SpinButton)):
            adj = w.get_adjustment()
            if abs(adj.get_lower() - min_val) < 1e-6 and abs(adj.get_upper() - max_val) < 1e-6:
                return w

    raise AssertionError(
        f"Missing GTK slider/spin control for {keyword!r} with range [{min_val}, {max_val}]"
    )


def get_combo_items(combo: Any) -> list[tuple[Optional[str], str]]:
    model = combo.get_model()
    if model is None:
        return []
    items: list[tuple[Optional[str], str]] = []
    n_cols = model.get_n_columns()
    id_col = combo.get_id_column()
    for row in model:
        text = str(row[0]) if n_cols > 0 else ""
        row_id = str(row[id_col]) if 0 <= id_col < n_cols and row[id_col] is not None else None
        items.append((row_id, text))
    return items


def find_dither_combo(root: Any) -> Any:
    """Locate the 5-mode dithering ComboBox in YogaBookEinkWindow."""
    from gi.repository import Gtk

    for w in iter_gtk_tree(root):
        if isinstance(w, Gtk.ComboBox):
            items = get_combo_items(w)
            joined = " ".join(f"{rid or ''} {txt}" for rid, txt in items).lower()
            if "atkinson" in joined and "bayer" in joined:
                return w
    raise AssertionError("Missing 5-mode Dithering ComboBox in YogaBookEinkWindow")


def select_dither_mode_in_combo(combo: Any, canonical_mode: str) -> None:
    items = get_combo_items(combo)
    needle = canonical_mode.lower()
    for idx, (rid, txt) in enumerate(items):
        if (rid and rid.lower() == needle) or txt.lower() == needle or needle in txt.lower():
            if rid is not None:
                combo.set_active_id(rid)
            else:
                combo.set_active(idx)
            pump_gtk_events()
            return
    raise AssertionError(f"Mode {canonical_mode!r} not found in dither combo items: {items}")


def find_idle_settle_toggle(root: Any) -> Any:
    """Locate the idle settle toggle (CheckButton/ToggleButton/Switch) in YogaBookEinkWindow."""
    from gi.repository import Gtk

    for w in iter_gtk_tree(root):
        if isinstance(w, Gtk.CheckButton):
            lbl = (w.get_label() or "").lower()
            if "settle" in lbl or "anti-ghost" in lbl or ("idle" in lbl and "gc16" in lbl):
                return w
        if isinstance(w, Gtk.Switch):
            parent = w.get_parent()
            if parent is not None:
                for sib in iter_gtk_tree(parent):
                    if isinstance(sib, Gtk.Label):
                        lbl = (sib.get_text() or "").lower()
                        if "settle" in lbl or "anti-ghost" in lbl or ("idle" in lbl and "gc16" in lbl):
                            return w
    raise AssertionError("Missing idle anti-ghosting settle toggle in YogaBookEinkWindow")


# ---------------------------------------------------------------------------
# Synthetic Image Builders & Assertions
# ---------------------------------------------------------------------------


def make_gradient_image(w: int = 256, h: int = 16) -> Image.Image:
    img = Image.new("L", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = int(round(x * 255.0 / max(1, w - 1)))
    return img


def make_gradient_png(path: str, w: int = 128, h: int = 64) -> str:
    make_gradient_image(w, h).save(path, format="PNG")
    return path


def make_asymmetric_png(path: str, w: int = 80, h: int = 60) -> str:
    """Top half bright white/light gray, bottom half dark, left corner black marker."""
    img = Image.new("L", (w, h), color=220)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, h // 2), (w - 1, h - 1)], fill=30)
    draw.rectangle([(0, 0), (w // 4, h // 4)], fill=0)
    img.save(path, format="PNG")
    return path


def make_photo_sample_png(path: str, w: int = 96, h: int = 72) -> str:
    """RGB photo-like test image with smooth gradients and geometric details."""
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            r = int(255 * x / max(1, w - 1))
            g = int(255 * y / max(1, h - 1))
            b = int(255 * ((x + y) % 64) / 63.0)
            px[x, y] = (r, g, b)
    img.save(path, format="PNG")
    return path


def assert_strict_1bit_image(test: unittest.TestCase, img: Image.Image, msg: str = "") -> None:
    test.assertEqual(img.mode, "L", msg)
    vals = set(img.getdata())
    test.assertTrue(vals.issubset({0, 255}), f"{msg}: non-1bit pixels found: {sorted(vals - {0, 255})[:10]}")


def assert_16_gray_image(test: unittest.TestCase, img: Image.Image, msg: str = "") -> None:
    test.assertEqual(img.mode, "L", msg)
    allowed = {k * 17 for k in range(16)}
    vals = set(img.getdata())
    test.assertTrue(vals.issubset(allowed), f"{msg}: invalid 16-level pixels: {sorted(vals - allowed)[:10]}")


def assert_8px_aligned(test: unittest.TestCase, x: int, y: int, w: int, h: int, msg: str = "") -> None:
    test.assertEqual(x % 8, 0, f"{msg}: x={x} not 8px aligned")
    test.assertEqual(y % 8, 0, f"{msg}: y={y} not 8px aligned")
    test.assertEqual(w % 8, 0, f"{msg}: w={w} not 8px aligned")
    test.assertEqual(h % 8, 0, f"{msg}: h={h} not 8px aligned")
    test.assertGreater(w, 1, f"{msg}: w={w} <= 1")
    test.assertGreater(h, 1, f"{msg}: h={h} <= 1")
    test.assertGreaterEqual(x, 0, msg)
    test.assertGreaterEqual(y, 0, msg)
    test.assertLessEqual(x + w, PANEL_WIDTH, msg)
    test.assertLessEqual(y + h, PANEL_HEIGHT, msg)
