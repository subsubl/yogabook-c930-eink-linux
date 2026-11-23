"""Mocked USB BOT protocol and R5 invariant tests for yogabook_eink.protocol."""

from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from yogabook_eink.protocol import (  # noqa: E402
    IMAGE_BUF_BASE,
    OP_DPY_AREA,
    OP_GET_DPY_STATUS,
    OP_LD_IMG_AREA2,
    OP_SET_WAVEFORM,
    PANEL_HEIGHT,
    PANEL_WIDTH,
    USBC_HEADER,
    USBS_MAGIC,
    WAVEFORM_CURRENT,
    WAVEFORM_DU2,
    WAVEFORM_GC16,
    WAVEFORM_GL16,
    WAVEFORM_INIT,
    Ite8951Controller,
    build_ctrl_packet,
)
from yogabook_eink.usb_lib import EP_IN, EP_OUT, UsbError  # noqa: E402


def _make_recorded_controller(busy_polls: int = 0) -> tuple[Ite8951Controller, list[tuple]]:
    ctrl = Ite8951Controller(usb_dev=MagicMock())
    calls: list[tuple] = []
    poll_state = {"remaining_busy": busy_polls}

    def fake_exchange_out(
        payload: bytes,
        opcode: int,
        addr: int = 0,
        arg1: int = 0,
        arg2: int = 0,
        arg3: int = 0,
        arg4: int = 0,
        timeout_ms: int = 5000,
    ) -> bytes:
        calls.append(("OUT", opcode, payload, addr, arg1, arg2, arg3, arg4))
        return USBS_MAGIC + b"\x00" * 9

    def fake_exchange_in(
        resp_len: int,
        opcode: int,
        addr: int = 0,
        arg1: int = 0,
        arg2: int = 0,
        arg3: int = 0,
        arg4: int = 0,
        timeout_ms: int = 5000,
    ) -> bytes:
        calls.append(("IN", opcode, resp_len, addr, arg1, arg2, arg3, arg4))
        if opcode == OP_GET_DPY_STATUS:
            busy = 0
            if poll_state["remaining_busy"] > 0:
                poll_state["remaining_busy"] -= 1
                busy = 1
            buf = bytearray(136)
            struct.pack_into("<H", buf, 4, busy)
            return bytes(buf)
        return b"\x00" * resp_len

    ctrl._exchange_out = fake_exchange_out  # type: ignore[method-assign]
    ctrl._exchange_in = fake_exchange_in  # type: ignore[method-assign]
    return ctrl, calls


class TestProtocolBlitR5(unittest.TestCase):
    def test_blit_never_calls_0xA9_and_packs_0x94_payload(self) -> None:
        for wf in (WAVEFORM_INIT, WAVEFORM_DU2, WAVEFORM_GC16, WAVEFORM_GL16, WAVEFORM_CURRENT):
            ctrl, calls = _make_recorded_controller()
            x, y, w, h = 80, 120, 256, 128
            buf_offset = y * PANEL_WIDTH + x
            ctrl.blit(buf_offset, x, y, w, h, waveform=wf)

            opcodes = [c[1] for c in calls]
            self.assertNotIn(OP_SET_WAVEFORM, opcodes)
            self.assertEqual(opcodes, [OP_GET_DPY_STATUS, OP_DPY_AREA, OP_GET_DPY_STATUS])

            dpy_call = calls[1]
            self.assertEqual(dpy_call[0], "OUT")
            self.assertEqual(len(dpy_call[2]), 24)
            mem_addr, packed_wf, px, py, pw, ph = struct.unpack(">6I", dpy_call[2])
            self.assertEqual(mem_addr, IMAGE_BUF_BASE)
            self.assertEqual(mem_addr, 0x00382F30)
            self.assertEqual(packed_wf, wf)
            self.assertEqual((px, py, pw, ph), (x, y, w, h))

    def test_wait_display_ready_polls_0xB1_before_and_after_0x94(self) -> None:
        ctrl, calls = _make_recorded_controller(busy_polls=2)
        ctrl.wait_display_ready(timeout_ms=1000, poll_ms=1)
        self.assertEqual([c[1] for c in calls], [OP_GET_DPY_STATUS] * 3)

        # Now verify busy poll inside blit() pre and post 0x94
        ctrl2, calls2 = _make_recorded_controller(busy_polls=2)
        orig_wait = ctrl2.wait_display_ready
        ctrl2.wait_display_ready = lambda timeout_ms=4000, poll_ms=1: orig_wait(timeout_ms=timeout_ms, poll_ms=1)  # type: ignore[method-assign]
        ctrl2.blit(0, 0, 0, 16, 16, waveform=WAVEFORM_DU2)
        opcodes = [c[1] for c in calls2]
        self.assertEqual(opcodes, [OP_GET_DPY_STATUS, OP_GET_DPY_STATUS, OP_GET_DPY_STATUS, OP_DPY_AREA, OP_GET_DPY_STATUS])
        self.assertNotIn(OP_SET_WAVEFORM, opcodes)


class TestProtocolDisplayRegionAndImage(unittest.TestCase):
    def test_display_region_uses_y_dy_stride_and_mem_addr_base(self) -> None:
        for wf in (WAVEFORM_DU2, WAVEFORM_GL16):
            ctrl, calls = _make_recorded_controller()
            x, y, w, h = 80, 120, 64, 32
            pixels = bytes((i & 0xF0) | 0x0F for i in range(w * h))
            ctrl.display_region(x, y, w, h, pixels, waveform=wf)

            opcodes = [c[1] for c in calls]
            self.assertNotIn(OP_SET_WAVEFORM, opcodes)
            self.assertEqual(opcodes, [OP_GET_DPY_STATUS, OP_LD_IMG_AREA2, OP_GET_DPY_STATUS, OP_DPY_AREA, OP_GET_DPY_STATUS])

            ld_call = calls[1]
            self.assertEqual(ld_call[3], IMAGE_BUF_BASE + y * PANEL_WIDTH + x)
            self.assertNotEqual(ld_call[3], IMAGE_BUF_BASE)
            self.assertEqual((ld_call[6], ld_call[7]), (w, h))
            self.assertEqual(ld_call[2], pixels)

            dpy_call = calls[3]
            mem_addr, packed_wf, px, py, pw, ph = struct.unpack(">6I", dpy_call[2])
            self.assertEqual(mem_addr, 0x00382F30)
            self.assertEqual(packed_wf, wf)
            self.assertEqual((px, py, pw, ph), (x, y, w, h))

    def test_display_region_multichunk_uploads(self) -> None:
        ctrl, calls = _make_recorded_controller()
        x, y, w, h = 16, 24, 1904, 70
        pixels = bytes(((row & 0x0F) << 4) | 0x0F for row in range(h) for _ in range(w))
        ctrl.display_region(x, y, w, h, pixels, waveform=WAVEFORM_DU2)

        ld_calls = [c for c in calls if c[1] == OP_LD_IMG_AREA2]
        # rows_per_chunk = 61440 // 1904 = 32 -> 3 chunks: 32, 32, 6 rows
        self.assertEqual(len(ld_calls), 3)
        self.assertEqual([c[7] for c in ld_calls], [32, 32, 6])
        self.assertEqual(
            [c[3] for c in ld_calls],
            [
                IMAGE_BUF_BASE + (24 + 0) * PANEL_WIDTH + 16,
                IMAGE_BUF_BASE + (24 + 32) * PANEL_WIDTH + 16,
                IMAGE_BUF_BASE + (24 + 64) * PANEL_WIDTH + 16,
            ],
        )
        self.assertEqual(b"".join(c[2] for c in ld_calls), pixels)

        dpy_calls = [c for c in calls if c[1] == OP_DPY_AREA]
        self.assertEqual(len(dpy_calls), 1)
        mem_addr, wf, px, py, pw, ph = struct.unpack(">6I", dpy_calls[0][2])
        self.assertEqual(mem_addr, 0x00382F30)
        self.assertEqual((wf, px, py, pw, ph), (WAVEFORM_DU2, x, y, w, h))

    def test_display_image_bytes_full_panel_chunks(self) -> None:
        ctrl, calls = _make_recorded_controller()
        pixels = b"\xFF" * (PANEL_WIDTH * PANEL_HEIGHT)
        progress: list[float] = []
        ctrl.display_image_bytes(pixels, PANEL_WIDTH, PANEL_HEIGHT, waveform=WAVEFORM_GC16, progress_cb=progress.append)

        opcodes = [c[1] for c in calls]
        self.assertNotIn(OP_SET_WAVEFORM, opcodes)
        ld_calls = [c for c in calls if c[1] == OP_LD_IMG_AREA2]
        self.assertEqual(len(ld_calls), 34)
        self.assertAlmostEqual(progress[-1], 1.0)
        dpy_calls = [c for c in calls if c[1] == OP_DPY_AREA]
        self.assertEqual(len(dpy_calls), 1)
        mem_addr, wf, px, py, pw, ph = struct.unpack(">6I", dpy_calls[0][2])
        self.assertEqual(mem_addr, 0x00382F30)
        self.assertEqual((wf, px, py, pw, ph), (WAVEFORM_GC16, 0, 0, PANEL_WIDTH, PANEL_HEIGHT))


class TestProtocolDefensiveGuards(unittest.TestCase):
    def test_blit_rejects_buf_offset_zero_shortcut_and_invalid_inputs(self) -> None:
        ctrl, calls = _make_recorded_controller()
        # Reject buf_offset=0 for non-zero origin
        with self.assertRaises(ValueError):
            ctrl.blit(0, 8, 8, 16, 16, waveform=WAVEFORM_DU2)
        # Reject wrong-stride buf_offset
        with self.assertRaises(ValueError):
            ctrl.blit(10 * 64 + 8, 8, 10, 64, 32, waveform=WAVEFORM_DU2)
        # Reject zero/negative dims
        for w, h in ((0, 8), (8, 0), (-8, 8), (8, -8)):
            with self.assertRaises(ValueError):
                ctrl.blit(0, 0, 0, w, h, waveform=WAVEFORM_DU2)
        # Reject out of bounds
        for x, y, w, h in ((-8, 0, 8, 8), (0, -8, 8, 8), (1912, 0, 16, 8), (0, 1072, 8, 16)):
            with self.assertRaises(ValueError):
                ctrl.blit(max(0, y) * PANEL_WIDTH + max(0, x), x, y, w, h, waveform=WAVEFORM_DU2)
        # Reject invalid waveforms
        for bad_wf in (-1, 256, True):
            with self.assertRaises(ValueError):
                ctrl.blit(0, 0, 0, 8, 8, waveform=bad_wf)  # type: ignore[arg-type]
        self.assertEqual(calls, [])

    def test_display_region_and_load_pixels_and_image_bytes_guards(self) -> None:
        ctrl, calls = _make_recorded_controller()
        with self.assertRaises(ValueError):
            ctrl.display_region(0, 0, 0, 16, b"", waveform=WAVEFORM_DU2)
        with self.assertRaises(ValueError):
            ctrl.display_region(0, 0, 16, 0, b"", waveform=WAVEFORM_DU2)
        with self.assertRaises(ValueError):
            ctrl.display_region(-1, 0, 8, 8, b"\xFF" * 64, waveform=WAVEFORM_DU2)
        with self.assertRaises(ValueError):
            ctrl.display_region(1916, 0, 8, 8, b"\xFF" * 64, waveform=WAVEFORM_DU2)
        with self.assertRaises(ValueError):
            ctrl.display_region(0, 0, 8, 8, b"\xFF" * 32, waveform=WAVEFORM_DU2)

        with self.assertRaises(ValueError):
            ctrl.load_pixels(0, 0, 8, b"")
        with self.assertRaises(ValueError):
            ctrl.load_pixels(-1, 8, 8, b"\xFF" * 64)
        with self.assertRaises(ValueError):
            ctrl.load_pixels(0, 8, 8, b"\xFF" * 10)

        with self.assertRaises(ValueError):
            ctrl.display_image_bytes(b"", 0, 10)
        with self.assertRaises(ValueError):
            ctrl.display_image_bytes(b"\xFF" * (1928 * 8), 1928, 8)
        with self.assertRaises(ValueError):
            ctrl.display_image_bytes(b"\xFF" * 10, 8, 8)
        self.assertEqual(calls, [])


class _FakeRawUsbDev:
    def __init__(self, read_queue: list[bytes]) -> None:
        self.writes: list[tuple[bytes, int]] = []
        self.read_queue = list(read_queue)
        self.is_open = True

    def bulk_write(self, data: bytes, endpoint: int = EP_OUT, timeout_ms: int = 5000) -> int:
        self.writes.append((data, endpoint))
        return len(data)

    def bulk_read(self, length: int, endpoint: int = EP_IN, timeout_ms: int = 5000) -> bytes:
        if not self.read_queue:
            raise UsbError("no queued read")
        return self.read_queue.pop(0)


class TestUsbcPacketFraming(unittest.TestCase):
    def test_build_ctrl_packet_layout(self) -> None:
        pkt = build_ctrl_packet(136, True, OP_GET_DPY_STATUS)
        self.assertEqual(len(pkt), 31)
        self.assertEqual(pkt[:8], USBC_HEADER)
        self.assertEqual(
            pkt.hex(),
            "555342436189518988000000800010fe0000000000b1000000000000000000",
        )

        dpy_pkt = build_ctrl_packet(24, False, OP_DPY_AREA)
        self.assertEqual(len(dpy_pkt), 31)
        self.assertEqual(
            dpy_pkt.hex(),
            "555342436189518918000000000010fe000000000094000000000000000000",
        )

    def test_exchange_out_and_in_usb_bot_framing(self) -> None:
        csw = USBS_MAGIC + b"\x00" * 9
        dev = _FakeRawUsbDev(
            read_queue=[
                csw,  # _exchange_out drain
                b"\xAA" * 136 + csw,  # coalesced _exchange_in
                csw,  # immediate empty _exchange_in
            ]
        )
        ctrl = Ite8951Controller(usb_dev=dev)  # type: ignore[arg-type]

        status = ctrl._exchange_out(b"\x01\x02\x03\x04", OP_DPY_AREA, addr=0, arg1=1)
        self.assertEqual(status, csw)
        self.assertEqual(len(dev.writes[0][0]), 31)
        self.assertEqual(dev.writes[0][1], EP_OUT)
        self.assertEqual(dev.writes[1][0], b"\x01\x02\x03\x04")

        resp = ctrl._exchange_in(136, OP_GET_DPY_STATUS)
        self.assertEqual(resp, b"\xAA" * 136)

        empty_resp = ctrl._exchange_in(136, OP_GET_DPY_STATUS)
        self.assertEqual(empty_resp, b"")


if __name__ == "__main__":
    unittest.main()
