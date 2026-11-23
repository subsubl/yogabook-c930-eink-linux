"""ITE8951 USB Bulk-Only Transport (BOT) protocol for Lenovo Yoga Book C930.

Wire protocol verified against:
  - Lenovo Yoga Book C930 open-source headers (EinkIteAPI.h / EinkInternal.h)
  - aleksb/yogabook-c930-linux-eink-driver (2019 Linux driver)
  - fivehalos/yogabook-c930-linux-eink-driver (2026 Windows USB captures & DRM driver)
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .usb_lib import EP_IN, EP_OUT, LibUsbDevice, UsbError

# USB BOT framing constants
USBC_HEADER = b"USBC\x61\x89\x51\x89"
USBS_MAGIC = b"USBS"
USBS_LEN = 13

# ITE8951 opcodes
OP_GET_SYS = 0x80
OP_READ_MEM = 0x81
OP_WRITE_MEM = 0x82
OP_READ_REG = 0x83
OP_WRITE_REG = 0x84
OP_DPY_AREA = 0x94
OP_SCENARIO = 0xA6
OP_LD_IMG_AREA2 = 0xA8
OP_SET_WAVEFORM = 0xA9
OP_UPDATE = 0xAA
OP_SET_HANDWR_REGION = 0xAC
OP_MODE_HELPER_AE = 0xAE
OP_SET_TP_AREA = 0xAF
OP_GET_DPY_STATUS = 0xB1
OP_DYNAMIC_SETTING = 0xB3

# Known registers
REG_DISPLAY_CFG = 0x18001138
REG_PANEL_MODE = 0x18001224
REG_PANEL_MODE_READY = 0x80000000

# Display config bits
DISPLAY_CFG_DRAW = 0x002E0000
DISPLAY_CFG_MT_FINGER_BIT = 0x00080000

# Scenario IDs
SCENARIO_NORMAL = 0       # Non-KB / Owner-draw
SCENARIO_KEYBOARD = 1     # Firmware keyboard
SCENARIO_PEN_MOUSE = 3    # Pen-mouse / Multitouch latch

SCENARIO_NAMES = {
    0: "Owner Draw / Non-Keyboard (0)",
    1: "Firmware Keyboard (1)",
    3: "Pen-Mouse / Multitouch Latch (3)",
}

# Waveforms
WAVEFORM_INIT = 0   # Full flash clear (removes all ghosting)
WAVEFORM_DU2 = 1    # Fast 1-bit monochrome update
WAVEFORM_GC16 = 2   # 16-level grayscale (photos/images)
WAVEFORM_GL16 = 3   # 16-level smooth text/UI
WAVEFORM_CURRENT = 0xFF  # Use currently active waveform

WAVEFORM_NAMES = {
    WAVEFORM_INIT: "INIT (Full Anti-Ghosting Clear)",
    WAVEFORM_DU2: "DU2 (Fast Monochrome)",
    WAVEFORM_GC16: "GC16 (16-Level Grayscale)",
    WAVEFORM_GL16: "GL16 (Smooth Text / UI)",
    WAVEFORM_CURRENT: "Current / Default (0xFF)",
}

# TP Area flags (from EinkIteAPI.h)
TP_AREA_NO_REPORT = 0x00
TP_AREA_TOUCH_ONLY = 0x01
TP_AREA_PEN_ONLY = 0x02
TP_AREA_TOUCH_PEN = 0x03

# Display geometry & RAM base
PANEL_WIDTH = 1920
PANEL_HEIGHT = 1080
IMAGE_BUF_BASE = 0x00382F30
MAX_XFER_ROWS = 32  # 1920 * 32 = 61440 bytes (proven stable max transfer)


@dataclass
class SystemInfo:
    standard_cmd_no: int
    extend_cmd_no: int
    signature: int
    version: int
    width: int
    height: int
    update_buf_base: int
    image_buf_base: int
    temperature_no: int
    mode_no: int
    tp_x_max: int
    tp_y_max: int
    tp_version: tuple[int, int, int, int]
    epd_type: int

    @property
    def signature_str(self) -> str:
        try:
            return struct.pack(">I", self.signature).decode("ascii")
        except Exception:
            return f"0x{self.signature:08x}"

    @property
    def tp_version_str(self) -> str:
        return ".".join(str(b) for b in self.tp_version)


def build_ctrl_packet(
    payload_sz: int,
    dir_in: bool,
    opcode: int,
    addr: int = 0,
    arg1: int = 0,
    arg2: int = 0,
    arg3: int = 0,
    arg4: int = 0,
) -> bytes:
    """Construct the 31-byte USBC control packet."""
    return struct.pack(
        "<8sIBBBBB",
        USBC_HEADER,
        payload_sz,
        0x80 if dir_in else 0x00,
        0x00,
        0x10,  # CDB length
        0xFE,  # SCSI vendor command
        0x00,  # LUN
    ) + struct.pack(
        ">IBHHHHB",
        addr & 0xFFFFFFFF,
        opcode & 0xFF,
        arg1 & 0xFFFF,
        arg2 & 0xFFFF,
        arg3 & 0xFFFF,
        arg4 & 0xFFFF,
        0x00,
    )


class Ite8951Controller:
    """High-level controller for the Yoga Book C930 ITE8951 E-Ink panel."""

    def __init__(self, usb_dev: Optional[LibUsbDevice] = None) -> None:
        self.usb = usb_dev or LibUsbDevice()
        self._owns_usb = usb_dev is None

    def open(self) -> None:
        self.usb.open()

    def close(self) -> None:
        if self._owns_usb:
            self.usb.close()

    @property
    def is_open(self) -> bool:
        return self.usb.is_open

    def __enter__(self) -> "Ite8951Controller":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _drain_usbs(self, timeout_ms: int = 3000) -> bytes:
        """Read and verify the trailing 13-byte USBS status packet."""
        status = self.usb.bulk_read(512, EP_IN, timeout_ms=timeout_ms)
        if len(status) >= 4 and status[:4] == USBS_MAGIC:
            return status
        # Sometimes a short status arrives on the next read
        if len(status) < USBS_LEN:
            status2 = self.usb.bulk_read(512, EP_IN, timeout_ms=timeout_ms)
            if len(status2) >= 4 and status2[:4] == USBS_MAGIC:
                return status2
        return status

    def _exchange_in(
        self,
        resp_len: int,
        opcode: int,
        addr: int = 0,
        arg1: int = 0,
        arg2: int = 0,
        arg3: int = 0,
        arg4: int = 0,
        timeout_ms: int = 5000,
    ) -> bytes:
        """Send a DIR_IN command, read response payload, and drain USBS."""
        ctrl = build_ctrl_packet(resp_len, True, opcode, addr, arg1, arg2, arg3, arg4)
        self.usb.bulk_write(ctrl, EP_OUT, timeout_ms=timeout_ms)

        data = b""
        if resp_len > 0:
            read_sz = max(resp_len, 512)
            chunk = self.usb.bulk_read(read_sz, EP_IN, timeout_ms=timeout_ms)
            # Check if USBS was coalesced at the end of the payload
            if len(chunk) == resp_len + USBS_LEN and chunk[resp_len : resp_len + 4] == USBS_MAGIC:
                return chunk[:resp_len]
            if len(chunk) == USBS_LEN and chunk[:4] == USBS_MAGIC and resp_len != USBS_LEN:
                # Device returned USBS immediately with no data payload
                return b""
            data = chunk[:resp_len]

        self._drain_usbs(timeout_ms=timeout_ms)
        return data

    def _exchange_out(
        self,
        payload: bytes,
        opcode: int,
        addr: int = 0,
        arg1: int = 0,
        arg2: int = 0,
        arg3: int = 0,
        arg4: int = 0,
        timeout_ms: int = 5000,
    ) -> bytes:
        """Send a DIR_OUT command, optional data payload, and drain USBS."""
        ctrl = build_ctrl_packet(len(payload), False, opcode, addr, arg1, arg2, arg3, arg4)
        self.usb.bulk_write(ctrl, EP_OUT, timeout_ms=timeout_ms)
        if payload:
            self.usb.bulk_write(payload, EP_OUT, timeout_ms=timeout_ms)
        return self._drain_usbs(timeout_ms=timeout_ms)

    def drain_stale_in(self) -> int:
        """Best-effort drain of any stale IN packets if the panel was left mid-exchange."""
        drained = 0
        for _ in range(4):
            try:
                buf = self.usb.bulk_read(512, EP_IN, timeout_ms=150)
                drained += len(buf)
                if not buf:
                    break
            except UsbError:
                break
        return drained

    # ------------------------------------------------------------------
    # Basic info & register access
    # ------------------------------------------------------------------

    def get_system_info(self) -> SystemInfo:
        """Send GET_SYS (0x80) and parse the 112-byte TRSP_SYSTEM_INFO_DATA."""
        raw = self._exchange_in(112, OP_GET_SYS, 0x38393531, 1, 2, 0, 0)
        if len(raw) < 100:
            raise UsbError(f"GET_SYS returned short response ({len(raw)} bytes)")
        # TRSP_SYSTEM_INFO_DATA uint32 fields are big-endian on the wire (0x00000780=1920, 0x00000438=1080)
        vals = struct.unpack_from(">10I", raw, 0)
        # Skip uiFrameCount[8] (32 bytes) + 4 uints (16 bytes) -> offset 40 + 32 + 16 = 88
        tp_x_max, tp_y_max = struct.unpack_from(">2I", raw, 88)
        tp_ver = struct.unpack_from("<4B", raw, 96)
        epd_type = raw[100] if len(raw) > 100 else 0
        return SystemInfo(
            standard_cmd_no=vals[0],
            extend_cmd_no=vals[1],
            signature=vals[2],
            version=vals[3],
            width=vals[4] or PANEL_WIDTH,
            height=vals[5] or PANEL_HEIGHT,
            update_buf_base=vals[6],
            image_buf_base=vals[7],
            temperature_no=vals[8],
            mode_no=vals[9],
            tp_x_max=tp_x_max,
            tp_y_max=tp_y_max,
            tp_version=tp_ver,
            epd_type=epd_type,
        )

    def init_panel(self) -> SystemInfo:
        """Run the 4x GET_SYS doorknock initialization sequence."""
        info: Optional[SystemInfo] = None
        for _ in range(4):
            info = self.get_system_info()
        assert info is not None
        return info

    def read_reg(self, reg_addr: int) -> int:
        """Read a 32-bit big-endian T-CON register (opcode 0x83)."""
        raw = self._exchange_in(4, OP_READ_REG, reg_addr, 4, 0, 0, 0)
        if len(raw) < 4:
            raise UsbError(f"READ_REG(0x{reg_addr:08x}) short read ({len(raw)} B)")
        return struct.unpack(">I", raw[:4])[0]

    def write_reg(self, reg_addr: int, value: int) -> None:
        """Write a 32-bit big-endian T-CON register (opcode 0x84)."""
        payload = struct.pack(">I", value & 0xFFFFFFFF)
        self._exchange_out(payload, OP_WRITE_REG, reg_addr, 4, 0, 0, 0)

    def get_scenario(self) -> int:
        """Query current coordinator scenario (0=draw/non-KB, 1=KB, 3=pen-mouse/MT)."""
        raw = self._exchange_in(4, OP_SCENARIO, 0x00000000, 0, 0, 0, 0)
        if len(raw) >= 2:
            return raw[1]
        return -1

    def set_waveform(self, waveform: int) -> None:
        """Set panel waveform mode (0=INIT, 1=DU2, 2=GC16, 3=GL16) via 0xA9."""
        arg = waveform if waveform > 0xFF else ((waveform & 0xFF) << 8)
        self._exchange_in(1, OP_SET_WAVEFORM, 0, arg & 0xFFFF, 0, 0, 0)

    def query_display_status(self) -> int:
        """Query 0xB1 GET_DPY_STATUS (136 bytes) and return engine_busy (le16 at offset 4)."""
        raw = self._exchange_in(136, OP_GET_DPY_STATUS, 0, 0, 0, 0, 0)
        if len(raw) < 6:
            return 0
        return struct.unpack_from("<H", raw, 4)[0]

    def wait_display_ready(self, timeout_ms: int = 4000, poll_ms: int = 35) -> bool:
        """Poll 0xB1 GET_DPY_STATUS until engine_busy == 0 (prevents T-CON glitches!)."""
        deadline = time.monotonic() + (timeout_ms / 1000.0)
        while time.monotonic() < deadline:
            try:
                if self.query_display_status() == 0:
                    return True
            except UsbError:
                return False
            time.sleep(poll_ms / 1000.0)
        return False

    def dynamic_cdb(
        self, arg1: int, arg2: int, arg3: int, arg4: int = 0, expect_in: bool = True
    ) -> bytes:
        """Send 0xB3 dynamic setting using Windows CDB-inline form."""
        if expect_in:
            return self._exchange_in(6, OP_DYNAMIC_SETTING, 0, arg1, arg2, arg3, arg4)
        return self._exchange_out(b"", OP_DYNAMIC_SETTING, 0, arg1, arg2, arg3, arg4)

    def clear_handwriting_region(self) -> None:
        """Clear handwriting region (opcode 0xAC)."""
        try:
            self._exchange_in(8, OP_SET_HANDWR_REGION, 0, 0, 0, 0, 0)
        except UsbError:
            pass

    def set_tp_area(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        index: int = 0,
        flag: int = TP_AREA_TOUCH_PEN,
    ) -> None:
        """Configure a touch/pen reporting region (opcode 0xAF)."""
        payload = struct.pack("<4IBB", x, y, w, h, index & 0xFF, flag & 0xFF)
        self._exchange_out(payload, OP_SET_TP_AREA, 0, 0, 0, 0, 0)

    # ------------------------------------------------------------------
    # Mode transitions (Keyboard, Draw, Leave-KB, Multitouch Touchpad)
    # ------------------------------------------------------------------

    def enable_keyboard(self) -> int:
        """Enable / restore the Yoga Book C930 firmware E-Ink keyboard."""
        self.init_panel()
        self._exchange_in(1, OP_SET_WAVEFORM, 0, 0x0200, 0, 0, 0)
        self._exchange_in(4, OP_SCENARIO, 0x00040000, 0, 0, 0, 0)

        # Poll panel-ready register (0x18001224 == 0x80000000)
        for _ in range(20):
            try:
                val = self.read_reg(REG_PANEL_MODE)
                if val == REG_PANEL_MODE_READY:
                    break
            except UsbError:
                break
            time.sleep(0.1)

        self._exchange_in(4, OP_SCENARIO, 0x00040000, 0, 0, 0, 0)
        self._exchange_in(4, OP_SCENARIO, 0x01010000, 0, 0, 0, 0)
        scen = self.get_scenario()
        if scen != SCENARIO_KEYBOARD:
            # Also try Windows high-byte scenario SET (1 << 24)
            self._exchange_in(4, OP_SCENARIO, SCENARIO_KEYBOARD << 24, 0, 0, 0, 0)
            scen = self.get_scenario()
        return scen

    def enable_draw_mode(self) -> int:
        """Leave firmware keyboard mode completely and enter Owner-Draw / Touch mode.

        Uses the Windows-validated leave-keyboard sequence (0xB3 + 0xA9 + 0xA6 0x03000000)
        so the firmware keyboard stops intercepting taps and stops redrawing keys over the UI,
        then configures display_cfg (0x18001138 = 0x002e0000) for owner drawing + 0x90 touch.
        """
        # 1. Leave firmware keyboard mode (scenario 1 -> 0) so taps don't type letters
        scen = self.leave_keyboard()
        if scen == SCENARIO_KEYBOARD:
            # Retry once with full doorknock + leave if panel was sleepy
            self.init_panel()
            scen = self.leave_keyboard()

        # 2. Configure owner-draw display register
        try:
            self.read_reg(REG_PANEL_MODE)
            self.read_reg(REG_DISPLAY_CFG)
        except UsbError:
            pass
        self.write_reg(REG_DISPLAY_CFG, DISPLAY_CFG_DRAW)
        self.clear_handwriting_region()
        return self.get_scenario()

    def leave_keyboard(self) -> int:
        """Minimal validated Windows leave-keyboard sequence (stops firmware typing)."""
        try:
            self.dynamic_cdb(0x0100, 0x0103, 0x0301, 0, expect_in=False)
        except UsbError:
            pass
        try:
            self._exchange_in(1, OP_SET_WAVEFORM, 0, 0x0200, 0, 0, 0)
        except UsbError:
            pass
        self._exchange_in(4, OP_SCENARIO, SCENARIO_PEN_MOUSE << 24, 0, 0, 0, 0)
        return self.get_scenario()

    def enable_multitouch_mode(self) -> tuple[int, int]:
        """Windows-validated multitouch / pen-mouse latch (GET=3 + finger MT bit).

        Returns (scenario, display_cfg_reg).
        """
        try:
            self._exchange_in(16, OP_READ_MEM, 0x00000080, 0x0010, 0, 0, 0)
        except UsbError:
            pass
        try:
            self.dynamic_cdb(0x0101, 0x0003, 0x0301, 0, expect_in=True)
        except UsbError:
            pass
        self._exchange_in(4, OP_SCENARIO, 0x01030100, 0, 0, 0, 0)
        try:
            self._exchange_in(1, OP_MODE_HELPER_AE, 0, 0x0100, 0, 0, 0)
        except UsbError:
            pass
        self.clear_handwriting_region()
        try:
            self.dynamic_cdb(0x0100, 0x0003, 0x0301, 0, expect_in=True)
        except UsbError:
            pass
        self._exchange_in(4, OP_SCENARIO, 0x01030100, 0, 0, 0, 0)

        # Enable finger multitouch bit (0x00080000) in display_cfg (0x18001138)
        cfg = self.read_reg(REG_DISPLAY_CFG)
        new_cfg = cfg | DISPLAY_CFG_MT_FINGER_BIT
        self.write_reg(REG_DISPLAY_CFG, new_cfg)

        return self.get_scenario(), self.read_reg(REG_DISPLAY_CFG)

    def enter_app_mode(self, clean_color: int = 0xFF) -> int:
        """Standard glitch-free app entry: leave KB, arm full-screen touch, clean screen (INIT)."""
        self.drain_stale_in()
        scen = self.enable_draw_mode()
        self.set_tp_area(0, 0, PANEL_WIDTH, PANEL_HEIGHT, index=0, flag=TP_AREA_TOUCH_PEN)
        self.clear_screen(fill_byte=clean_color, waveform=WAVEFORM_INIT)
        return scen

    def set_keyboard_style_blob(self, blob_data: bytes, timeout_s: float = 25.0) -> bool:
        """Official Lenovo TconSetKeyBoardStyle (itetcon.dll) hardware implementation.

        Uploads a Lenovo KeyboardBlob (.bin.cap) into T-CON SDRAM in 32 KiB chunks
        via OP_WRITE_MEM (0x82), validates via OP_UPDATE (0xAA, cmd=0x01), and polls
        OP_UPDATE (0xAA, cmd=0x02) until the T-CON commits the native keyboard style.
        """
        if not blob_data:
            return False

        sys_info = self.get_system_info()
        dest_base = (sys_info.image_buf_base + 4 + sys_info.width * sys_info.height) & 0xFFFFFFFF

        # Enter non-KB mode (Scenario 0) and wait 1.0s before uploading blob
        if self.get_scenario() == SCENARIO_KEYBOARD:
            self.leave_keyboard()
        time.sleep(1.0)

        # Upload blob in 32 KiB (0x8000) chunks using OP_WRITE_MEM (0x82)
        total_len = len(blob_data)
        chunk_max = 0x8000
        offset = 0
        while offset < total_len:
            chunk = blob_data[offset : offset + chunk_max]
            chunk_len = len(chunk)
            addr = (dest_base + offset) & 0xFFFFFFFF
            self._exchange_out(chunk, OP_WRITE_MEM, addr, chunk_len, 0, 0, 0, timeout_ms=8000)
            offset += chunk_len

        # Trigger kb_blob_cmd_valid (cmd=0x01) via OP_UPDATE (0xAA)
        addr_valid = (0x01 << 24) | (total_len & 0x00FFFFFF)
        resp = self._exchange_in(4, OP_UPDATE, addr_valid, 0, 0, 0, 0, timeout_ms=8000)
        if len(resp) >= 4:
            val_rc = struct.unpack(">I", resp[:4])[0]
            if val_rc != 0:
                return False

        # Poll kb_blob_cmd_get_status (cmd=0x02) every 500ms while status == 1 (busy)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            time.sleep(0.5)
            st_resp = self._exchange_in(4, OP_UPDATE, 0x02000000, 0, 0, 0, 0, timeout_ms=8000)
            if len(st_resp) >= 4:
                status_byte = st_resp[3]
                if status_byte == 1:
                    continue
                return status_byte == 0
        return False

    def apply_keyboard_style(self, style: str, lang_id: int = 1033) -> bool:
        """Load the official Lenovo `.bin.cap` for `style` and apply it via TconSetKeyBoardStyle."""
        from .keyboard_skin import load_keyboard_blob, normalize_keyboard_style

        st = normalize_keyboard_style(style)
        blob = load_keyboard_blob(st, lang_id=lang_id)
        if not blob:
            return False
        ok = self.set_keyboard_style_blob(blob)
        if ok:
            # Give the T-CON internal SPI flash controller a moment to settle after 0xAA status=0
            time.sleep(1.0)
        return ok

    def exit_to_keyboard(self, style: Optional[str] = None) -> int:
        """Cleanly exit any E-Ink app and force a genuine 0 -> 1 firmware keyboard redraw.

        If `style` (or `EinkSettings.keyboard_style`) differs from `applied_keyboard_style`,
        flashes the selected Lenovo `.bin.cap` keyboard blob first via TconSetKeyBoardStyle.
        """
        if style is not None:
            from .config import EinkSettings
            from .keyboard_skin import normalize_keyboard_style

            target_style = normalize_keyboard_style(style)
            settings = EinkSettings.load()
            if settings.applied_keyboard_style != target_style:
                if self.apply_keyboard_style(target_style):
                    settings.keyboard_style = target_style
                    settings.applied_keyboard_style = target_style
                    settings.save()

        self.wait_display_ready(timeout_ms=3000)

        # Ensure we are in non-KB state (0) before setting TP_AREA and calling enable_keyboard()
        # so 0xAF takes effect in draw mode and firmware sees a real 0 -> 1 transition!
        if self.get_scenario() == SCENARIO_KEYBOARD:
            self.leave_keyboard()
            time.sleep(0.2)

        try:
            self.clear_handwriting_region()
            self.set_tp_area(1700, 0, 220, 140, index=0, flag=TP_AREA_TOUCH_PEN)
        except UsbError:
            pass

        scen = self.enable_keyboard()
        self.wait_display_ready(timeout_ms=3000)
        return scen

    # ------------------------------------------------------------------
    # Display pixel upload & blit
    # ------------------------------------------------------------------

    def load_pixels(self, buf_offset: int, width: int, height: int, pixels: bytes) -> None:
        """Upload raw 1-byte-per-pixel grayscale data to T-CON RAM (opcode 0xA8)."""
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid pixel upload dimensions ({width}x{height})")
        if buf_offset < 0:
            raise ValueError(f"buf_offset must be non-negative, got {buf_offset}")
        expected = width * height
        if len(pixels) != expected:
            raise ValueError(f"Expected {expected} bytes ({width}x{height}), got {len(pixels)}")
        self._exchange_out(
            pixels,
            OP_LD_IMG_AREA2,
            IMAGE_BUF_BASE + buf_offset,
            0,
            0,
            width,
            height,
        )

    def blit(
        self,
        buf_offset: int,
        x: int,
        y: int,
        width: int,
        height: int,
        waveform: int = WAVEFORM_CURRENT,
    ) -> None:
        """Trigger E-Ink display refresh from T-CON RAM (opcode 0x94) with engine-busy waits.

        Passes `waveform` (0=INIT, 1=DU2, 2=GC16, 3=GL16, 0xFF=CURRENT) directly in the
        0x94 payload without calling 0xA9 (which can silence subsequent blits).
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid blit dimensions ({width}x{height})")
        if x < 0 or y < 0 or x + width > PANEL_WIDTH or y + height > PANEL_HEIGHT:
            raise ValueError(f"Blit region ({x},{y},{width},{height}) out of panel bounds")
        if not isinstance(waveform, int) or isinstance(waveform, bool) or not (0 <= waveform <= 0xFF):
            raise ValueError(f"Invalid waveform: {waveform!r}")
        expected_offset = y * PANEL_WIDTH + x
        if buf_offset != expected_offset:
            raise ValueError(
                f"Invalid blit buf_offset={buf_offset}; expected {expected_offset} "
                f"so mem_addr == 0x{IMAGE_BUF_BASE:08x}"
            )

        self.wait_display_ready(timeout_ms=4000)

        mem_addr = (IMAGE_BUF_BASE + buf_offset - x - PANEL_WIDTH * y) & 0xFFFFFFFF
        payload = struct.pack(
            ">6I",
            mem_addr,
            waveform & 0xFFFFFFFF,
            x & 0xFFFFFFFF,
            y & 0xFFFFFFFF,
            width & 0xFFFFFFFF,
            height & 0xFFFFFFFF,
        )
        self._exchange_out(payload, OP_DPY_AREA, 0, 0, 0, 0, 0)
        self.wait_display_ready(timeout_ms=4000)

    def display_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        pixels: bytes,
        waveform: int = WAVEFORM_DU2,
    ) -> None:
        """Fast, glitch-free partial-region update aligned to the 1920-wide framebuffer.

        Always writes into device RAM at (y + dy) * 1920 + x so blit mem_addr is always
        the exact framebuffer base (0x00382f30) with zero (0,0) clobbering or underflow.
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Region width/height must be positive, got ({width}, {height})")
        if x < 0 or y < 0 or x + width > PANEL_WIDTH or y + height > PANEL_HEIGHT:
            raise ValueError(f"Region ({x},{y},{width},{height}) out of panel bounds")
        if len(pixels) != width * height:
            raise ValueError(f"Expected {width * height} bytes, got {len(pixels)}")

        self.wait_display_ready(timeout_ms=4000)

        rows_per_chunk = max(1, 61440 // width)
        dy = 0
        while dy < height:
            chunk_h = min(rows_per_chunk, height - dy)
            buf_offset = (y + dy) * PANEL_WIDTH + x
            start = dy * width
            chunk = pixels[start : start + chunk_h * width]
            self.load_pixels(buf_offset, width, chunk_h, chunk)
            dy += chunk_h

        self.blit(y * PANEL_WIDTH + x, x, y, width, height, waveform=waveform)

    def display_image_bytes(
        self,
        pixels: bytes,
        width: int = PANEL_WIDTH,
        height: int = PANEL_HEIGHT,
        waveform: int = WAVEFORM_GC16,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Upload a full 1920x1080 packed grayscale buffer in 32-row chunks and blit."""
        if width <= 0 or height <= 0 or width > PANEL_WIDTH or height > PANEL_HEIGHT:
            raise ValueError(f"Invalid image dimensions ({width}x{height})")
        if len(pixels) != width * height:
            raise ValueError(f"Expected {width * height} bytes, got {len(pixels)}")

        self.wait_display_ready(timeout_ms=4000)
        rows_per_chunk = max(1, min(MAX_XFER_ROWS, 61440 // width))
        y = 0
        while y < height:
            chunk_h = min(rows_per_chunk, height - y)
            offset = y * width
            chunk = pixels[offset : offset + chunk_h * width]
            self.load_pixels(offset, width, chunk_h, chunk)
            y += chunk_h
            if progress_cb is not None:
                progress_cb(y / height)

        self.blit(0, 0, 0, width, height, waveform=waveform)

    def clear_screen(
        self,
        fill_byte: int = 0xFF,
        waveform: int = WAVEFORM_INIT,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Fill the E-Ink screen with a solid shade (0xFF=white, 0x0F=black) and refresh."""
        # Low nibble is always 0x0F in the panel's pixel packing
        packed = (fill_byte & 0xF0) | 0x0F
        buf = bytes([packed]) * (PANEL_WIDTH * PANEL_HEIGHT)
        self.display_image_bytes(buf, PANEL_WIDTH, PANEL_HEIGHT, waveform=waveform, progress_cb=progress_cb)
