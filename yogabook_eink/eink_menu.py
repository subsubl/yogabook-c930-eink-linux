"""Multi-page 1-Bit E-Ink Dashboard, Calendar, Battery Telemetry & Settings UI (1920x1080).

Top Bar (interactive on every page):
  - Left:   Live CPU %, RAM used/total, Disk used/total
  - Right:  [ Clock & Date ] (tap -> Calendar)   |   [ BAT % ] (tap -> Battery Details)

Pages:
  - dashboard: 2x2 category grid (GAMES, APPS, TOOLS, SETTINGS) + bottom EXIT TO KEYBOARD
  - games:     PLAY DOOM
  - apps:      MATRIX RAIN (DARK / LIGHT)
  - tools:     RULED NOTES, SKETCH GRID, TOUCHPAD (MT), CLEAN SCREEN
  - settings:  All toggles (Key Sound, Sound Style, Tone LUT, Theme, Refresh, KB Style, etc.)
  - calendar:  Monthly calendar grid with today highlighted + Prev/Next month navigation
  - battery:   Detailed battery telemetry (status, %, voltage, watts/amps, time left/to full, health)
"""

from __future__ import annotations

import calendar
import datetime
import glob
import shutil
import time
from dataclasses import dataclass
from typing import Literal, Optional

from PIL import Image, ImageDraw, ImageFont

from .config import EinkSettings
from .image_render import pack_eink_grayscale, pack_eink_region
from .keyboard_skin import KEYBOARD_STYLE_LABELS, normalize_keyboard_style
from .protocol import (
    PANEL_HEIGHT,
    PANEL_WIDTH,
    WAVEFORM_DU2,
    WAVEFORM_GC16,
    WAVEFORM_GL16,
)

WF_LABELS = {
    WAVEFORM_DU2: "DU2 Fast",
    WAVEFORM_GL16: "GL16 Smooth",
    WAVEFORM_GC16: "GC16 Sharp",
}

MenuPage = Literal["dashboard", "games", "apps", "tools", "settings", "calendar", "battery"]

# Previous /proc/stat sample for non-blocking CPU% delta
_PREV_CPU_SAMPLE: Optional[tuple[int, int]] = None


@dataclass
class SettingsWidget:
    action_id: str
    title: str
    value_label: str
    subtitle: str
    x: int
    y: int
    w: int
    h: int
    is_toggle: bool = False
    is_on: bool = False
    is_header_btn: bool = False
    is_card: bool = False
    is_topbar_hit: bool = False
    icon: str = ""

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px <= (self.x + self.w) and self.y <= py <= (self.y + self.h)


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        ]
        if bold
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        ]
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _read_sysfs_int(path: str) -> Optional[int]:
    try:
        return int(open(path, "r", encoding="utf-8").read().strip())
    except (OSError, ValueError):
        return None


def _read_sysfs_str(path: str) -> str:
    try:
        return open(path, "r", encoding="utf-8").read().strip()
    except OSError:
        return ""


def _read_cpu_percent() -> int:
    global _PREV_CPU_SAMPLE
    try:
        line = open("/proc/stat", "r", encoding="utf-8").readline()
        parts = [int(x) for x in line.split()[1:9]]
        idle = parts[3] + parts[4]
        total = sum(parts)
        if _PREV_CPU_SAMPLE is None:
            _PREV_CPU_SAMPLE = (idle, total)
            # Fallback to 1-min load average scaled by CPU count on first sample
            load1 = float(open("/proc/loadavg", "r", encoding="utf-8").read().split()[0])
            import os
            ncpu = os.cpu_count() or 1
            return max(0, min(100, int(round((load1 / ncpu) * 100))))
        prev_idle, prev_total = _PREV_CPU_SAMPLE
        _PREV_CPU_SAMPLE = (idle, total)
        dt = total - prev_total
        di = idle - prev_idle
        if dt <= 0:
            return 0
        return max(0, min(100, int(round(100.0 * (dt - di) / dt))))
    except Exception:
        return 0


def _read_sys_stats_str() -> str:
    """Return compact 'CPU 12%  •  RAM 3.2/8.0G  •  DISK 37%' for the left top bar."""
    cpu = _read_cpu_percent()
    ram_str = "RAM —"
    try:
        mem: dict[str, int] = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                k, v = line.split(":", 1)
                mem[k] = int(v.strip().split()[0])
                if "MemTotal" in mem and "MemAvailable" in mem:
                    break
        tot_gb = mem["MemTotal"] / (1024 * 1024)
        used_gb = (mem["MemTotal"] - mem["MemAvailable"]) / (1024 * 1024)
        ram_str = f"RAM {used_gb:.1f}/{tot_gb:.1f}G"
    except Exception:
        pass

    disk_str = "DISK —"
    try:
        du = shutil.disk_usage("/")
        pct = int(round(100.0 * du.used / max(1, du.total)))
        disk_str = f"DISK {pct}%"
    except Exception:
        pass

    return f"CPU {cpu}%   •   {ram_str}   •   {disk_str}"


def read_battery_details() -> dict[str, str]:
    """Read detailed live battery telemetry from /sys/class/power_supply/BAT*."""
    info: dict[str, str] = {
        "short": "AC Power",
        "status": "AC Power",
        "capacity": "—",
        "voltage": "—",
        "power": "—",
        "current": "—",
        "energy": "—",
        "time_est": "—",
        "health": "—",
        "stylus": "",
    }
    bats = sorted(glob.glob("/sys/class/power_supply/BAT*"))
    if bats:
        bat = bats[0]
        status = _read_sysfs_str(f"{bat}/status") or "Unknown"
        cap = _read_sysfs_int(f"{bat}/capacity")
        v_uv = _read_sysfs_int(f"{bat}/voltage_now")
        p_uw = _read_sysfs_int(f"{bat}/power_now")
        c_ua = _read_sysfs_int(f"{bat}/current_now")
        e_now = _read_sysfs_int(f"{bat}/energy_now")
        e_full = _read_sysfs_int(f"{bat}/energy_full")
        e_des = _read_sysfs_int(f"{bat}/energy_full_design")

        suffix = " ⚡" if status == "Charging" else ""
        info["short"] = f"BAT {cap}%{suffix}" if cap is not None else status
        info["status"] = status
        info["capacity"] = f"{cap}%" if cap is not None else "—"

        v_volts = (v_uv / 1_000_000.0) if v_uv else 0.0
        if v_volts > 0:
            info["voltage"] = f"{v_volts:.2f} V"

        if p_uw is not None and p_uw > 0:
            watts = p_uw / 1_000_000.0
            info["power"] = f"{watts:.2f} W"
            if v_volts > 0:
                info["current"] = f"{watts / v_volts:.2f} A"
        elif c_ua is not None and c_ua > 0:
            amps = c_ua / 1_000_000.0
            info["current"] = f"{amps:.2f} A"
            if v_volts > 0:
                info["power"] = f"{amps * v_volts:.2f} W"

        if e_now is not None and e_full is not None and e_full > 0:
            info["energy"] = f"{e_now / 1_000_000.0:.2f} / {e_full / 1_000_000.0:.2f} Wh"
            if p_uw and p_uw > 50_000:
                if status == "Charging":
                    rem_h = max(0.0, (e_full - e_now) / float(p_uw))
                    h = int(rem_h)
                    m = int(round((rem_h - h) * 60))
                    info["time_est"] = f"{h}h {m:02d}m to full"
                elif status == "Discharging":
                    rem_h = max(0.0, e_now / float(p_uw))
                    h = int(rem_h)
                    m = int(round((rem_h - h) * 60))
                    info["time_est"] = f"{h}h {m:02d}m left"
                else:
                    info["time_est"] = "Fully charged"
            elif status == "Full":
                info["time_est"] = "Fully charged"

        if e_full and e_des and e_des > 0:
            health_pct = 100.0 * e_full / e_des
            info["health"] = f"{health_pct:.1f}% ({e_full / 1_000_000.0:.1f} / {e_des / 1_000_000.0:.1f} Wh)"

    # Optional stylus/HID battery if present
    for hbat in sorted(glob.glob("/sys/class/power_supply/hid-*-battery-*")):
        hcap = _read_sysfs_int(f"{hbat}/capacity")
        if hcap is not None and hcap > 0:
            info["stylus"] = f"Stylus / HID Battery: {hcap}%"
            break

    return info


def _draw_card_icon(draw: ImageDraw.ImageDraw, kind: str, cx: int, cy: int, fg: int, bg: int) -> None:
    """Draw crisp 1-bit geometric icons centered at (cx, cy) inside dashboard cards."""
    if kind == "doom":
        draw.ellipse([(cx - 42, cy - 42), (cx + 42, cy + 42)], outline=fg, width=6)
        draw.ellipse([(cx - 18, cy - 18), (cx + 18, cy + 18)], outline=fg, width=4)
        draw.line([(cx - 56, cy), (cx + 56, cy)], fill=fg, width=5)
        draw.line([(cx, cy - 56), (cx, cy + 56)], fill=fg, width=5)
    elif kind == "matrix":
        for dx, h1, h2 in [(-36, -42, 34), (-12, -26, 46), (12, -46, 18), (36, -18, 42)]:
            draw.line([(cx + dx, cy + h1), (cx + dx, cy + h2)], fill=fg, width=6)
            draw.rectangle([(cx + dx - 6, cy + h2 - 6), (cx + dx + 6, cy + h2 + 6)], fill=fg)
    elif kind == "ruled":
        draw.rectangle([(cx - 38, cy - 44), (cx + 38, cy + 44)], outline=fg, width=5)
        for dy in (-22, -4, 14, 30):
            draw.line([(cx - 24, cy + dy), (cx + 24, cy + dy)], fill=fg, width=4)
    elif kind == "dotgrid":
        for gx in (-30, -10, 10, 30):
            for gy in (-30, -10, 10, 30):
                draw.ellipse([(cx + gx - 4, cy + gy - 4), (cx + gx + 4, cy + gy + 4)], fill=fg)
    elif kind == "touchpad":
        draw.rounded_rectangle([(cx - 46, cy - 34), (cx + 46, cy + 34)], radius=10, outline=fg, width=5)
        draw.line([(cx - 46, cy + 12), (cx + 46, cy + 12)], fill=fg, width=4)
        draw.line([(cx, cy + 12), (cx, cy + 34)], fill=fg, width=4)
    elif kind == "settings":
        for dy, knob_x in [(-26, -14), (0, 18), (26, -6)]:
            draw.line([(cx - 42, cy + dy), (cx + 42, cy + dy)], fill=fg, width=5)
            draw.ellipse(
                [(cx + knob_x - 11, cy + dy - 11), (cx + knob_x + 11, cy + dy + 11)],
                fill=bg,
                outline=fg,
                width=5,
            )
    elif kind == "tools":
        draw.rectangle([(cx - 42, cy - 38), (cx + 22, cy + 38)], outline=fg, width=5)
        for dy in (-18, 0, 18):
            draw.line([(cx - 28, cy + dy), (cx + 8, cy + dy)], fill=fg, width=4)
        draw.line([(cx + 12, cy + 32), (cx + 46, cy - 36)], fill=fg, width=7)
    elif kind == "clean":
        draw.rectangle([(cx - 44, cy - 32), (cx + 44, cy + 32)], outline=fg, width=5)
        draw.line([(cx - 18, cy), (cx + 18, cy)], fill=fg, width=5)
        draw.line([(cx, cy - 18), (cx, cy + 18)], fill=fg, width=5)


class EinkSettingsScreen:
    """Multi-page 1-bit E-Ink UI with live system top bar, Calendar, Battery & Settings."""

    def __init__(self, settings: Optional[EinkSettings] = None) -> None:
        self.settings = settings or EinkSettings.load()
        self.page: MenuPage = "dashboard"
        today = datetime.date.today()
        self.cal_year: int = today.year
        self.cal_month: int = today.month
        self.widgets: list[SettingsWidget] = []
        self._last_img: Optional[Image.Image] = None
        self._build_layout()

    def shift_calendar_month(self, delta: int) -> None:
        m = self.cal_month + delta
        y = self.cal_year
        while m < 1:
            m += 12
            y -= 1
        while m > 12:
            m -= 12
            y += 1
        self.cal_year, self.cal_month = y, m

    def _add_topbar_hit_targets(self) -> None:
        # Clock/Date zone (top-right-center) -> opens Calendar
        self.widgets.append(
            SettingsWidget(
                "page_calendar", "", "", "", 1080, 0, 500, 136, is_header_btn=True, is_topbar_hit=True
            )
        )
        # Battery zone (far top-right) -> opens Battery Details
        self.widgets.append(
            SettingsWidget(
                "page_battery", "", "", "", 1584, 0, 336, 136, is_header_btn=True, is_topbar_hit=True
            )
        )

    def _add_bottom_split_bar(self) -> None:
        self.widgets.append(
            SettingsWidget("page_dashboard", "←  DASHBOARD", "", "", 64, 920, 864, 112, is_header_btn=True)
        )
        self.widgets.append(
            SettingsWidget("kb", "EXIT TO KEYBOARD  →", "", "", 992, 920, 864, 112, is_header_btn=True)
        )

    def _build_layout(self) -> None:
        self.widgets = []
        self._add_topbar_hit_targets()
        if self.page == "dashboard":
            self._build_dashboard_layout()
        elif self.page == "games":
            self._build_games_layout()
        elif self.page == "apps":
            self._build_apps_layout()
        elif self.page == "tools":
            self._build_tools_layout()
        elif self.page == "calendar":
            self._build_calendar_layout()
        elif self.page == "battery":
            self._build_battery_layout()
        else:
            self._build_settings_layout()

    def _build_dashboard_layout(self) -> None:
        left_x = 64
        right_x = 992
        top_y = 176
        gap_y = 32
        card_w = 864
        card_h = 336

        cards = [
            ("page_games", "GAMES", "3D E-Ink DOOM & Demos", "doom"),
            ("page_apps", "APPS", "Matrix Rain Screensavers", "matrix"),
            ("page_tools", "TOOLS", "Notes, Sketch Grid & Touchpad", "tools"),
            ("page_settings", "SETTINGS", "Sound, Tone LUT, Theme & System", "settings"),
        ]

        for idx, (aid, title, sub, icon) in enumerate(cards):
            r = idx // 2
            c = idx % 2
            x = left_x if c == 0 else right_x
            y = top_y + r * (card_h + gap_y)
            self.widgets.append(
                SettingsWidget(aid, title, "", sub, x, y, card_w, card_h, is_card=True, icon=icon)
            )

        self.widgets.append(
            SettingsWidget("kb", "←  EXIT TO KEYBOARD", "", "", 64, 920, 1792, 112, is_header_btn=True)
        )

    def _build_games_layout(self) -> None:
        self.widgets.append(
            SettingsWidget(
                "launch_doom",
                "PLAY DOOM",
                "",
                "Playable 3D E-Ink Raycaster with On-Screen Touch Controls",
                64,
                224,
                1792,
                592,
                is_card=True,
                icon="doom",
            )
        )
        self._add_bottom_split_bar()

    def _build_apps_layout(self) -> None:
        cards = [
            ("launch_matrix", "MATRIX RAIN (DARK)", "Classic Dark Katakana Rain (Touch Anywhere to Exit)", "matrix"),
            ("launch_matrix_light", "MATRIX RAIN (LIGHT)", "Paper-White Katakana Rain (Touch Anywhere to Exit)", "matrix"),
        ]
        for idx, (aid, title, sub, icon) in enumerate(cards):
            x = 64 if idx == 0 else 992
            self.widgets.append(
                SettingsWidget(aid, title, "", sub, x, 224, 864, 592, is_card=True, icon=icon)
            )
        self._add_bottom_split_bar()

    def _build_tools_layout(self) -> None:
        left_x = 64
        right_x = 992
        top_y = 176
        gap_y = 32
        card_w = 864
        card_h = 336

        cards = [
            ("ruled", "RULED NOTES", "Stylus Lined Notebook Paper", "ruled"),
            ("dotgrid", "SKETCH GRID", "Stylus 40px Dot Grid Pad", "dotgrid"),
            ("mt", "TOUCHPAD (MT)", "Full-Panel Multitouch Trackpad", "touchpad"),
            ("clear_init", "CLEAN SCREEN", "Full INIT Anti-Ghosting Flash", "clean"),
        ]
        for idx, (aid, title, sub, icon) in enumerate(cards):
            r = idx // 2
            c = idx % 2
            x = left_x if c == 0 else right_x
            y = top_y + r * (card_h + gap_y)
            self.widgets.append(
                SettingsWidget(aid, title, "", sub, x, y, card_w, card_h, is_card=True, icon=icon)
            )
        self._add_bottom_split_bar()

    def _build_calendar_layout(self) -> None:
        self.widgets.append(
            SettingsWidget("cal_prev", "◀  PREV", "", "", 64, 176, 280, 88, is_header_btn=True)
        )
        self.widgets.append(
            SettingsWidget("cal_today", "TODAY", "", "", 1264, 176, 264, 88, is_header_btn=True)
        )
        self.widgets.append(
            SettingsWidget("cal_next", "NEXT  ▶", "", "", 1560, 176, 296, 88, is_header_btn=True)
        )
        self._add_bottom_split_bar()

    def _build_battery_layout(self) -> None:
        self.widgets.append(
            SettingsWidget("page_battery", "REFRESH STATS", "", "", 1480, 176, 376, 88, is_header_btn=True)
        )
        self._add_bottom_split_bar()

    def _build_settings_layout(self) -> None:
        s = self.settings
        col_w = 864
        row_h = 128
        gap_y = 16
        start_y = 184
        left_x = 64
        right_x = 992

        left_rows = [
            (
                "toggle_key_down",
                "Keyboard Key-Down Sound",
                "ON" if s.key_down_sound else "OFF",
                "Play tactile click when pressing E-Ink keys",
                True,
                s.key_down_sound,
            ),
            (
                "toggle_key_up",
                "Keyboard Key-Up Sound",
                "ON" if s.key_up_sound else "OFF",
                "Play soft release click when lifting finger",
                True,
                s.key_up_sound,
            ),
            (
                "cycle_sound",
                "Key Click Sound Style",
                s.sound_profile.title(),
                "Tap to cycle: Soft / Crisp / Typewriter",
                False,
                False,
            ),
            (
                "toggle_idle_settle",
                "Idle Anti-Ghosting (GC16)",
                "ON" if s.idle_settle_enabled else "OFF",
                "Auto-clean residual shadowing after 2s idle",
                True,
                s.idle_settle_enabled,
            ),
            (
                "open_gui",
                "Pop!_OS Control Window",
                "Open →",
                "Launch full desktop GUI on main display",
                False,
                False,
            ),
        ]

        kb_st = normalize_keyboard_style(s.keyboard_style)
        right_rows = [
            (
                "toggle_theme",
                "E-Ink UI Theme",
                "Dark" if s.dark_theme else "Light",
                "Switch between Light and Dark palette",
                True,
                s.dark_theme,
            ),
            (
                "cycle_waveform",
                "E-Ink Refresh Quality",
                WF_LABELS.get(s.waveform, "GC16 Sharp"),
                "Cycle: DU2 Fast / GL16 Smooth / GC16 Sharp",
                False,
                False,
            ),
            (
                "cycle_lightness",
                "Tone Lightness",
                f"{s.lightness:+d}" if s.lightness else "0",
                "Tap to cycle: -3 .. +3 (photos/DOOM)",
                False,
                False,
            ),
            (
                "cycle_contrast",
                "Tone Contrast",
                f"{s.contrast:+d}" if s.contrast else "0",
                "Tap to cycle: -1 .. +6 (photos/DOOM)",
                False,
                False,
            ),
            (
                "cycle_kb_style",
                "Keyboard & Touchpad Style",
                KEYBOARD_STYLE_LABELS.get(kb_st, "Classic Light"),
                "Classic Dark / Classic Light / Modern Dark / Modern Light",
                False,
                False,
            ),
        ]

        for idx, (aid, title, val, sub, is_tog, is_on) in enumerate(left_rows):
            y = start_y + idx * (row_h + gap_y)
            self.widgets.append(
                SettingsWidget(aid, title, val, sub, left_x, y, col_w, row_h, is_tog, is_on)
            )

        for idx, (aid, title, val, sub, is_tog, is_on) in enumerate(right_rows):
            y = start_y + idx * (row_h + gap_y)
            self.widgets.append(
                SettingsWidget(aid, title, val, sub, right_x, y, col_w, row_h, is_tog, is_on)
            )

        self._add_bottom_split_bar()

    def _render_calendar_body(self, draw: ImageDraw.ImageDraw, fg: int, bg: int) -> None:
        f_month = _load_font(44, bold=True)
        f_dow = _load_font(28, bold=True)
        f_day = _load_font(34, bold=True)

        month_title = f"{calendar.month_name[self.cal_month]}  {self.cal_year}"
        mb = draw.textbbox((0, 0), month_title, font=f_month)
        draw.text((380 + (840 - (mb[2] - mb[0])) // 2, 194), month_title, fill=fg, font=f_month)

        grid_x = 64
        grid_y = 296
        grid_w = 1792
        col_w = grid_w // 7
        dow_h = 56
        weeks = calendar.monthcalendar(self.cal_year, self.cal_month)
        row_h = min(92, (888 - (grid_y + dow_h)) // max(1, len(weeks)))

        # Day-of-week header
        days = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
        for c, dname in enumerate(days):
            cx = grid_x + c * col_w
            db = draw.textbbox((0, 0), dname, font=f_dow)
            draw.text((cx + (col_w - (db[2] - db[0])) // 2, grid_y + 10), dname, fill=fg, font=f_dow)
        draw.line([(grid_x, grid_y + dow_h), (grid_x + grid_w, grid_y + dow_h)], fill=fg, width=4)

        today = datetime.date.today()
        for r, week in enumerate(weeks):
            ry = grid_y + dow_h + r * row_h
            for c, day in enumerate(week):
                if day == 0:
                    continue
                cx = grid_x + c * col_w
                is_today = (
                    self.cal_year == today.year
                    and self.cal_month == today.month
                    and day == today.day
                )
                txt = str(day)
                tb = draw.textbbox((0, 0), txt, font=f_day)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
                if is_today:
                    draw.rounded_rectangle(
                        [(cx + 36, ry + 8), (cx + col_w - 36, ry + row_h - 8)],
                        radius=20,
                        fill=fg,
                    )
                    draw.text((cx + (col_w - tw) // 2, ry + (row_h - th) // 2 - 4), txt, fill=bg, font=f_day)
                else:
                    draw.text((cx + (col_w - tw) // 2, ry + (row_h - th) // 2 - 4), txt, fill=fg, font=f_day)

    def _render_battery_body(self, draw: ImageDraw.ImageDraw, fg: int, bg: int, bat: dict[str, str]) -> None:
        f_big = _load_font(52, bold=True)
        f_lbl = _load_font(28, bold=False)
        f_val = _load_font(40, bold=True)

        draw.text((64, 192), f"BATTERY  •  {bat['status'].upper()}  ({bat['capacity']})", fill=fg, font=f_big)

        # Big 1-bit battery charge bar
        bar_x, bar_y, bar_w, bar_h = 64, 292, 1792, 72
        draw.rounded_rectangle([(bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h)], radius=16, outline=fg, width=5)
        try:
            pct = int(bat["capacity"].replace("%", ""))
        except ValueError:
            pct = 0
        fill_w = max(0, int((bar_w - 20) * min(100, max(0, pct)) / 100.0))
        if fill_w > 0:
            draw.rounded_rectangle(
                [(bar_x + 10, bar_y + 10), (bar_x + 10 + fill_w, bar_y + bar_h - 10)],
                radius=10,
                fill=fg,
            )

        # 3x2 Telemetry Cards
        cards = [
            ("BATTERY VOLTAGE", bat["voltage"]),
            ("POWER / RATE", f"{bat['power']}  ({bat['current']})"),
            ("TIME ESTIMATE", bat["time_est"]),
            ("ENERGY STORED", bat["energy"]),
            ("BATTERY HEALTH", bat["health"]),
            ("ACCESSORIES", bat["stylus"] or "No HID stylus battery detected"),
        ]
        card_w = 576
        card_h = 200
        gap_x = 32
        gap_y = 28
        start_y = 408
        for idx, (label, val) in enumerate(cards):
            r = idx // 3
            c = idx % 3
            x = 64 + c * (card_w + gap_x)
            y = start_y + r * (card_h + gap_y)
            draw.rounded_rectangle([(x, y), (x + card_w, y + card_h)], radius=20, outline=fg, width=4)
            draw.text((x + 28, y + 28), label, fill=fg, font=f_lbl)
            draw.text((x + 28, y + 92), val, fill=fg, font=f_val if len(val) < 18 else _load_font(28, bold=True))

    def render(self, status_line: str = "") -> tuple[Image.Image, bytes]:
        """Render the current page in strict 1-bit monochrome (no footer help text)."""
        self._build_layout()
        dark = self.settings.dark_theme

        bg = 0 if dark else 255
        hdr_bg = 0
        card_bg = 0 if dark else 255
        fg = 255 if dark else 0
        border = 255 if dark else 0

        img = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=bg)
        draw = ImageDraw.Draw(img)
        draw.fontmode = "1"

        f_sys = _load_font(32, bold=True)
        f_clock = _load_font(34, bold=True)
        f_btn = _load_font(34, bold=True)
        f_title = _load_font(34, bold=True)
        f_sub = _load_font(23, bold=False)
        f_val = _load_font(28, bold=True)

        bat = read_battery_details()

        # Top Bar: Left = CPU / RAM / Disk | Right = [Clock & Date] | [Battery]
        draw.rectangle([(0, 0), (PANEL_WIDTH, 136)], fill=hdr_bg)
        if dark:
            draw.line([(0, 136), (PANEL_WIDTH, 136)], fill=255, width=4)

        sys_str = _read_sys_stats_str()
        draw.text((56, 48), sys_str, fill=255, font=f_sys)

        # Interactive Clock/Date pill on top bar
        clock_str = time.strftime("%H:%M  •  %a %d %b")
        clk_box = (1092, 24, 1568, 112)
        clk_active = self.page == "calendar"
        draw.rounded_rectangle(
            clk_box, radius=16, fill=255 if clk_active else 0, outline=255, width=3
        )
        cb = draw.textbbox((0, 0), clock_str, font=f_clock)
        draw.text(
            (clk_box[0] + ((clk_box[2] - clk_box[0]) - (cb[2] - cb[0])) // 2, 46),
            clock_str,
            fill=0 if clk_active else 255,
            font=f_clock,
        )

        # Interactive Battery pill on top bar
        bat_box = (1592, 24, 1864, 112)
        bat_active = self.page == "battery"
        draw.rounded_rectangle(
            bat_box, radius=16, fill=255 if bat_active else 0, outline=255, width=3
        )
        bb_bat = draw.textbbox((0, 0), bat["short"], font=f_clock)
        draw.text(
            (bat_box[0] + ((bat_box[2] - bat_box[0]) - (bb_bat[2] - bb_bat[0])) // 2, 46),
            bat["short"],
            fill=0 if bat_active else 255,
            font=f_clock,
        )

        # Page-specific bodies
        if self.page == "calendar":
            self._render_calendar_body(draw, fg, bg)
        elif self.page == "battery":
            self._render_battery_body(draw, fg, bg, bat)

        # Render interactive widgets
        for w in self.widgets:
            if w.is_topbar_hit:
                continue
            if w.is_card:
                draw.rounded_rectangle(
                    [(w.x, w.y), (w.x + w.w, w.y + w.h)],
                    radius=26,
                    fill=card_bg,
                    outline=border,
                    width=5,
                )
                _draw_card_icon(draw, w.icon, w.x + w.w // 2, w.y + w.h // 2 - 48, fg, card_bg)
                tb = draw.textbbox((0, 0), w.title, font=f_title)
                draw.text((w.x + (w.w - (tb[2] - tb[0])) // 2, w.y + w.h // 2 + 32), w.title, fill=fg, font=f_title)
                sb = draw.textbbox((0, 0), w.subtitle, font=f_sub)
                draw.text((w.x + (w.w - (sb[2] - sb[0])) // 2, w.y + w.h // 2 + 86), w.subtitle, fill=fg, font=f_sub)
            elif w.is_header_btn:
                is_primary = w.action_id == "kb"
                fill = (255 if dark else 0) if is_primary else card_bg
                txt_col = (0 if dark else 255) if is_primary else fg
                draw.rounded_rectangle(
                    [(w.x, w.y), (w.x + w.w, w.y + w.h)],
                    radius=22,
                    fill=fill,
                    outline=border,
                    width=5,
                )
                bb = draw.textbbox((0, 0), w.title, font=f_btn)
                draw.text(
                    (w.x + (w.w - (bb[2] - bb[0])) // 2, w.y + (w.h - (bb[3] - bb[1])) // 2 - 4),
                    w.title,
                    fill=txt_col,
                    font=f_btn,
                )
            else:
                draw.rounded_rectangle(
                    [(w.x, w.y), (w.x + w.w, w.y + w.h)],
                    radius=20,
                    fill=card_bg,
                    outline=border,
                    width=4,
                )
                title_y = w.y + (22 if w.h <= 130 else 32)
                sub_y = w.y + (72 if w.h <= 130 else 88)
                draw.text((w.x + 32, title_y), w.title, fill=fg, font=f_title)
                draw.text((w.x + 32, sub_y), w.subtitle, fill=fg, font=f_sub)

                pill_w, pill_h = 200, 68
                px = w.x + w.w - pill_w - 28
                py = w.y + (w.h - pill_h) // 2
                if w.is_toggle and w.is_on:
                    pill_fill = 255 if dark else 0
                    pill_txt = 0 if dark else 255
                else:
                    pill_fill = card_bg
                    pill_txt = fg
                draw.rounded_rectangle(
                    [(px, py), (px + pill_w, py + pill_h)],
                    radius=34,
                    fill=pill_fill,
                    outline=border,
                    width=4,
                )
                vb = draw.textbbox((0, 0), w.value_label, font=f_val)
                draw.text(
                    (px + (pill_w - (vb[2] - vb[0])) // 2, py + (pill_h - (vb[3] - vb[1])) // 2 - 4),
                    w.value_label,
                    fill=pill_txt,
                    font=f_val,
                )

        self._last_img = img
        return img, pack_eink_grayscale(img)

    def render_widget_patches(
        self,
        action_id: str,
        status_line: str = "",
    ) -> list[tuple[int, int, int, int, bytes]]:
        """Return ONLY the changed 8px-aligned full-width row band for fast DU2 toggle refresh."""
        img, _ = self.render(status_line=status_line)
        patches: list[tuple[int, int, int, int, bytes]] = []
        for w in self.widgets:
            if w.action_id == action_id and not w.is_header_btn and not w.is_card and not w.is_topbar_hit:
                y0 = w.y & ~7
                y1 = min(PANEL_HEIGHT, (w.y + w.h + 7) & ~7)
                h = y1 - y0
                patches.append((0, y0, PANEL_WIDTH, h, pack_eink_region(img, 0, y0, PANEL_WIDTH, h)))
                break
        return patches

    def hit_test(self, px: int, py: int) -> Optional[SettingsWidget]:
        for w in self.widgets:
            if w.contains(px, py):
                return w
        return None


# Backwards-compatible alias
EinkOnScreenMenu = EinkSettingsScreen
