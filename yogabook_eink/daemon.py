"""Lightweight Linux replacement for Lenovo EinkSvr (zero-lag background daemon).

Listens ONLY to /dev/hidraw0 (vendor E-Ink touch/button interface) — zero mouse/keyboard flood.
When you tap the top-right Settings gear on the E-Ink keyboard (~1866, 18), it renders
the Lenovo-style E-Ink Settings screen directly on the E-Ink panel and handles every tap.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Optional

from . import config as _config_mod
from .config import EinkSettings
from .eink_menu import EinkSettingsScreen
from .hid_listener import EinkHidListener, EinkTouchEvent, EinkVendorEvent
from .image_render import generate_template
from .key_sound import KeyboardSoundDaemon, SoundProfile
from .keyboard_skin import KEYBOARD_STYLE_LABELS, KEYBOARD_STYLES, normalize_keyboard_style
from .ownership import OwnershipServer
from .protocol import (
    PANEL_HEIGHT,
    PANEL_WIDTH,
    TP_AREA_TOUCH_PEN,
    WAVEFORM_DU2,
    WAVEFORM_GC16,
    WAVEFORM_GL16,
    WAVEFORM_INIT,
    Ite8951Controller,
)
from .tone_lut import (
    CONTRAST_MAX,
    CONTRAST_MIN,
    LIGHTNESS_MAX,
    LIGHTNESS_MIN,
    clamp_contrast,
    clamp_lightness,
)
from .usb_lib import UsbError


class EinkSettingsDaemon:
    """Zero-lag background daemon replicating Windows EinkSvr Settings behavior."""

    def __init__(self) -> None:
        self.settings = EinkSettings.load()
        self.screen = EinkSettingsScreen(self.settings)
        self.sound = KeyboardSoundDaemon()
        self._apply_sound_settings()

        self._usb_lock = threading.Lock()
        self._menu_active = False
        self._pad_active = False
        self._busy = False
        self._last_tap_ts = 0.0

        self.listener = EinkHidListener(
            on_touch=self._on_touch,
            on_vendor=self._on_vendor,
        )
        self.listener.swap_axes = self.settings.swap_axes
        self.listener.flip_x = self.settings.flip_x
        self.listener.flip_y = self.settings.flip_y

        self.ownership = OwnershipServer(
            on_owner_acquired=self._on_external_owner_acquired,
            on_owner_released=self._on_external_owner_released,
        )

    def _on_external_owner_acquired(self, app_name: str) -> None:
        self._menu_active = False
        self._pad_active = False

    def _on_external_owner_released(self, app_name: str) -> None:
        self._menu_active = False
        self._pad_active = False
        # Wait for any in-flight daemon USB operation (e.g. ~10s KeyboardBlob flash) to finish,
        # then ensure the panel is back in Scenario 1 (visible hardware keyboard).
        def _ensure_kb_after_release() -> None:
            with self._usb_lock:
                try:
                    with Ite8951Controller() as ctrl:
                        if ctrl.get_scenario() != 1:
                            ctrl.exit_to_keyboard()
                except UsbError:
                    pass

        threading.Thread(target=_ensure_kb_after_release, daemon=True).start()

    def _apply_sound_settings(self) -> None:
        self.sound.key_down_enabled = self.settings.key_down_sound
        self.sound.key_up_enabled = self.settings.key_up_sound
        if self.settings.sound_profile in ("soft", "crisp", "typewriter"):
            self.sound.profile = self.settings.sound_profile  # type: ignore[assignment]
        if self.sound.key_down_enabled or self.sound.key_up_enabled:
            self.sound.start()
        else:
            self.sound.stop()

    def _reload_settings(self) -> None:
        if os.path.exists(_config_mod.CONFIG_PATH):
            self.settings = EinkSettings.load()
        self.screen.settings = self.settings
        self._apply_sound_settings()
        self.listener.swap_axes = self.settings.swap_axes
        self.listener.flip_x = self.settings.flip_x
        self.listener.flip_y = self.settings.flip_y

    def _partial_refresh_widget(self, action_id: str, status_line: str) -> None:
        """Fast partial-region update: redraw ONLY the changed setting card + footer."""
        self.screen.settings = self.settings
        patches = self.screen.render_widget_patches(action_id, status_line=status_line)
        with self._usb_lock:
            with Ite8951Controller() as ctrl:
                for x, y, w, h, pixels in patches:
                    ctrl.display_region(x, y, w, h, pixels, waveform=WAVEFORM_DU2)

    def _show_settings_screen(self, status_line: str = "", fast: bool = False) -> None:
        self._busy = True
        try:
            self._reload_settings()
            msg = status_line or "Tap any setting to change  •  Tap ← Keyboard to return"
            _, packed = self.screen.render(status_line=msg)
            wf = WAVEFORM_DU2 if fast else self.settings.waveform
            with self._usb_lock:
                with Ite8951Controller() as ctrl:
                    if not self._menu_active:
                        ctrl.enter_app_mode(clean_color=0x0F if self.settings.dark_theme else 0xFF)
                    ctrl.display_image_bytes(packed, PANEL_WIDTH, PANEL_HEIGHT, waveform=wf)
            self._menu_active = True
        except UsbError as exc:
            print(f"[yogabook-eink-daemon] USB error: {exc}", flush=True)
        finally:
            self._busy = False

    def _on_vendor(self, ev: EinkVendorEvent) -> None:
        # Report 0x10/0x20 is emitted on normal keyboard keystrokes too — do NOT
        # hijack typing to open the Dashboard; only _on_touch (0x90) at the
        # top-right Settings icon (x >= 1700, y <= 140) should open it.
        return

    def _on_touch(self, ev: EinkTouchEvent) -> None:
        if self.ownership.active_owner is not None:
            return
        if not ev.is_down or self._busy:
            return
        now = time.monotonic()
        if now - self._last_tap_ts < 0.45:
            return

        # When in Ruled Notepad / Dot Grid pad mode, tapping bottom-center "← EXIT TO KEYBOARD" exits!
        if self._pad_active:
            if 620 <= ev.panel_x <= 1300 and ev.panel_y >= 920:
                self._last_tap_ts = now
                print(f"[yogabook-eink-daemon] Pad exit tapped at ({ev.panel_x},{ev.panel_y}) -> Keyboard", flush=True)
                threading.Thread(target=self._handle_action, args=("kb",), daemon=True).start()
                return
            if ev.panel_x >= 1700 and ev.panel_y <= 140:
                self._last_tap_ts = now
                self._pad_active = False
                threading.Thread(target=self._show_settings_screen, args=("", False), daemon=True).start()
            return

        # When in Keyboard mode, tapping the top-right Settings gear opens the E-Ink Dashboard!
        if not self._menu_active:
            if ev.panel_x >= 1700 and ev.panel_y <= 140:
                self._last_tap_ts = now
                self.screen.page = "dashboard"
                print(
                    f"[yogabook-eink-daemon] Dashboard opened from ({ev.panel_x},{ev.panel_y})",
                    flush=True,
                )
                threading.Thread(
                    target=self._show_settings_screen,
                    args=("", False),
                    daemon=True,
                ).start()
            return

        widget = self.screen.hit_test(ev.panel_x, ev.panel_y)
        if widget is None:
            print(f"[yogabook-eink-daemon] Menu tap missed at ({ev.panel_x},{ev.panel_y})", flush=True)
            return
        self._last_tap_ts = now
        print(
            f"[yogabook-eink-daemon] Tap ({ev.panel_x},{ev.panel_y}) -> {widget.action_id} ({widget.title})",
            flush=True,
        )
        threading.Thread(target=self._handle_action, args=(widget.action_id,), daemon=True).start()

    def _handle_action(self, action_id: str) -> None:
        self._busy = True
        try:
            self._reload_settings()
            if action_id == "kb":
                if self.settings.keyboard_style != self.settings.applied_keyboard_style:
                    lbl = KEYBOARD_STYLE_LABELS.get(self.settings.keyboard_style, self.settings.keyboard_style)
                    print(
                        f"[yogabook-eink-daemon] Applying native Lenovo KeyboardBlob: {lbl} ({self.settings.keyboard_style})...",
                        flush=True,
                    )
                    self._partial_refresh_widget(
                        "cycle_kb_style",
                        f"Applying {lbl} (~10s) — please wait...",
                    )
                with self._usb_lock:
                    with Ite8951Controller() as ctrl:
                        ctrl.exit_to_keyboard(style=self.settings.keyboard_style)
                self._reload_settings()
                self._menu_active = False
                self._pad_active = False
            elif action_id == "page_settings":
                self.screen.page = "settings"
                self._busy = False
                self._show_settings_screen("", fast=True)
                return
            elif action_id == "page_games":
                self.screen.page = "games"
                self._busy = False
                self._show_settings_screen("", fast=True)
                return
            elif action_id == "page_apps":
                self.screen.page = "apps"
                self._busy = False
                self._show_settings_screen("", fast=True)
                return
            elif action_id == "page_tools":
                self.screen.page = "tools"
                self._busy = False
                self._show_settings_screen("", fast=True)
                return
            elif action_id == "page_calendar":
                self.screen.page = "calendar"
                self._busy = False
                self._show_settings_screen("", fast=True)
                return
            elif action_id == "cal_prev":
                self.screen.shift_calendar_month(-1)
                self._busy = False
                self._show_settings_screen("", fast=True)
                return
            elif action_id == "cal_next":
                self.screen.shift_calendar_month(1)
                self._busy = False
                self._show_settings_screen("", fast=True)
                return
            elif action_id == "cal_today":
                import datetime
                today = datetime.date.today()
                self.screen.cal_year, self.screen.cal_month = today.year, today.month
                self._busy = False
                self._show_settings_screen("", fast=True)
                return
            elif action_id == "page_battery":
                self.screen.page = "battery"
                self._busy = False
                self._show_settings_screen("", fast=True)
                return
            elif action_id == "page_dashboard":
                self.screen.page = "dashboard"
                self._busy = False
                self._show_settings_screen("", fast=True)
                return
            elif action_id == "launch_doom":
                self._menu_active = False
                self._pad_active = False
                try:
                    subprocess.Popen(
                        ["/usr/local/bin/yogabook-eink-cli", "doom"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except OSError:
                    pass
            elif action_id in ("launch_matrix", "launch_matrix_light"):
                self._menu_active = False
                self._pad_active = False
                try:
                    cmd = ["/usr/local/bin/yogabook-eink-cli", "matrix"]
                    if action_id == "launch_matrix_light":
                        cmd.append("--light")
                    subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except OSError:
                    pass
            elif action_id == "mt":
                with self._usb_lock:
                    with Ite8951Controller() as ctrl:
                        ctrl.enable_multitouch_mode()
                self._menu_active = False
                self._pad_active = False
            elif action_id in ("ruled", "dotgrid"):
                _, packed = generate_template(action_id)  # type: ignore[arg-type]
                with self._usb_lock:
                    with Ite8951Controller() as ctrl:
                        ctrl.display_image_bytes(
                            packed, PANEL_WIDTH, PANEL_HEIGHT, waveform=self.settings.waveform
                        )
                self._menu_active = False
                self._pad_active = True
            elif action_id == "toggle_key_down":
                self.settings.key_down_sound = not self.settings.key_down_sound
                self.settings.save()
                self._apply_sound_settings()
                self._partial_refresh_widget(
                    action_id,
                    f"Key-Down Sound: {'ON' if self.settings.key_down_sound else 'OFF'}",
                )
                return
            elif action_id == "toggle_key_up":
                self.settings.key_up_sound = not self.settings.key_up_sound
                self.settings.save()
                self._apply_sound_settings()
                self._partial_refresh_widget(
                    action_id,
                    f"Key-Up Sound: {'ON' if self.settings.key_up_sound else 'OFF'}",
                )
                return
            elif action_id == "cycle_sound":
                order: list[SoundProfile] = ["soft", "crisp", "typewriter"]
                cur = self.settings.sound_profile if self.settings.sound_profile in order else "soft"
                self.settings.sound_profile = order[(order.index(cur) + 1) % len(order)]
                self.settings.save()
                self._apply_sound_settings()
                self._partial_refresh_widget(
                    action_id,
                    f"Click Sound Style: {self.settings.sound_profile.title()}",
                )
                return
            elif action_id == "toggle_theme":
                self.settings.dark_theme = not self.settings.dark_theme
                self.settings.save()
                self._busy = False
                self._show_settings_screen(
                    f"Theme: {'Dark' if self.settings.dark_theme else 'Light'}",
                    fast=False,
                )
                return
            elif action_id == "cycle_waveform":
                wfs = [WAVEFORM_DU2, WAVEFORM_GL16, WAVEFORM_GC16]
                cur = self.settings.waveform if self.settings.waveform in wfs else WAVEFORM_GC16
                self.settings.waveform = wfs[(wfs.index(cur) + 1) % len(wfs)]
                self.settings.save()
                self._partial_refresh_widget(action_id, "Updated E-Ink refresh mode")
                return
            elif action_id == "cycle_lightness":
                cur = clamp_lightness(self.settings.lightness)
                self.settings.lightness = LIGHTNESS_MIN if cur >= LIGHTNESS_MAX else cur + 1
                self.settings.save()
                lbl = f"{self.settings.lightness:+d}" if self.settings.lightness else "0"
                self._partial_refresh_widget(action_id, f"Tone Lightness: {lbl}")
                return
            elif action_id == "cycle_contrast":
                cur = clamp_contrast(self.settings.contrast)
                self.settings.contrast = CONTRAST_MIN if cur >= CONTRAST_MAX else cur + 1
                self.settings.save()
                lbl = f"{self.settings.contrast:+d}" if self.settings.contrast else "0"
                self._partial_refresh_widget(action_id, f"Tone Contrast: {lbl}")
                return
            elif action_id == "toggle_idle_settle":
                self.settings.idle_settle_enabled = not self.settings.idle_settle_enabled
                self.settings.save()
                self._partial_refresh_widget(
                    action_id,
                    f"Idle Anti-Ghosting: {'ON' if self.settings.idle_settle_enabled else 'OFF'}",
                )
                return
            elif action_id == "cycle_kb_style":
                cur_st = normalize_keyboard_style(self.settings.keyboard_style)
                idx = KEYBOARD_STYLES.index(cur_st)
                nxt_st = KEYBOARD_STYLES[(idx + 1) % len(KEYBOARD_STYLES)]
                self.settings.keyboard_style = nxt_st
                self.settings.save()
                lbl = KEYBOARD_STYLE_LABELS.get(nxt_st, nxt_st)
                self._partial_refresh_widget(action_id, f"Keyboard Style: {lbl} (applies on Exit to KB)")
                return
            elif action_id == "clear_init":
                with self._usb_lock:
                    with Ite8951Controller() as ctrl:
                        ctrl.clear_screen(fill_byte=0xFF, waveform=WAVEFORM_INIT)
                self._busy = False
                self._show_settings_screen("Screen cleaned (INIT anti-ghosting refresh)", fast=False)
                return
            elif action_id == "open_gui":
                try:
                    subprocess.Popen(
                        ["/usr/local/bin/yogabook-eink-gui"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except OSError:
                    pass
        except UsbError as exc:
            print(f"[yogabook-eink-daemon] USB error in '{action_id}': {exc}", flush=True)
        finally:
            self._busy = False

    def run_forever(self) -> int:
        self.ownership.start()
        ok, msg = self.listener.start()
        print(f"[yogabook-eink-daemon] {msg}", flush=True)
        if not ok:
            self.ownership.stop()
            return 1
        print(
            "[yogabook-eink-daemon] Ready — ownership socket active; tap top-right Settings icon anytime.",
            flush=True,
        )
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            self.ownership.stop()
            self.listener.stop()
            self.sound.stop()
        return 0
