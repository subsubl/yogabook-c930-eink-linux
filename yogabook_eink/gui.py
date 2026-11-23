"""Native GTK3 Control Center GUI for the Lenovo Yoga Book C930 on Pop!_OS."""

from __future__ import annotations

import io
import os
import threading
import time
from typing import Callable, Optional

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import GdkPixbuf, GLib, Gtk  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from .config import EinkSettings
from .doom_eink import run_eink_doom
from .eink_menu import EinkOnScreenMenu
from .hid_listener import (
    EinkHidListener,
    EinkTouchEvent,
    EinkVendorEvent,
    find_yogabook_hidraw_devices,
)
from .image_render import (
    DITHER_MODES,
    dither_image,
    generate_template,
    normalize_dither_mode,
    prepare_image_file,
)
from .key_sound import KeyboardSoundDaemon, SoundProfile, find_ite_keyboard_event_path
from .keyboard_skin import KEYBOARD_STYLE_LABELS, KEYBOARD_STYLES, normalize_keyboard_style
from .matrix_saver import run_matrix_screensaver
from .protocol import (
    PANEL_HEIGHT,
    PANEL_WIDTH,
    REG_DISPLAY_CFG,
    REG_PANEL_MODE,
    SCENARIO_NAMES,
    TP_AREA_NO_REPORT,
    TP_AREA_PEN_ONLY,
    TP_AREA_TOUCH_ONLY,
    TP_AREA_TOUCH_PEN,
    WAVEFORM_CURRENT,
    WAVEFORM_DU2,
    WAVEFORM_GC16,
    WAVEFORM_GL16,
    WAVEFORM_INIT,
    WAVEFORM_NAMES,
    Ite8951Controller,
)
from .tone_lut import (
    CONTRAST_MAX,
    CONTRAST_MIN,
    LIGHTNESS_MAX,
    LIGHTNESS_MIN,
    apply_tone_lut,
    clamp_contrast,
    clamp_lightness,
)
from .usb_lib import UsbError


def _prepare_preview_pil(
    path: str,
    fit_mode: str = "contain",
    invert: bool = False,
    rotate_180: bool = False,
    dither_mode: str = "Grayscale 16-level",
    lightness: int = 0,
    contrast: int = 0,
    max_w: int = 440,
    max_h: int = 248,
) -> Image.Image:
    """Render a fast thumbnail preview matching prepare_image_file()'s pipeline."""
    with Image.open(path) as raw:
        img = ImageOps.exif_transpose(raw)
        if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
            rgba = img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, rgba).convert("L")
        else:
            img = img.convert("L")

    if img.width <= 0 or img.height <= 0:
        img = Image.new("L", (max_w, max_h), color=255)
    elif fit_mode == "cover":
        img = ImageOps.fit(img, (max_w, max_h), method=Image.Resampling.LANCZOS)
    else:
        im_ratio = img.width / img.height
        dest_ratio = max_w / max_h
        if im_ratio == dest_ratio:
            img = img.resize((max_w, max_h), Image.Resampling.LANCZOS)
        elif im_ratio > dest_ratio:
            new_h = max(1, round(img.height / img.width * max_w))
            if new_h == max_h:
                img = img.resize((max_w, max_h), Image.Resampling.LANCZOS)
            else:
                resized = img.resize((max_w, new_h), Image.Resampling.LANCZOS)
                canvas = Image.new("L", (max_w, max_h), color=255)
                canvas.paste(resized, (0, round((max_h - new_h) * 0.5)))
                img = canvas
        else:
            new_w = max(1, round(img.width / img.height * max_h))
            if new_w == max_w:
                img = img.resize((max_w, max_h), Image.Resampling.LANCZOS)
            else:
                resized = img.resize((new_w, max_h), Image.Resampling.LANCZOS)
                canvas = Image.new("L", (max_w, max_h), color=255)
                canvas.paste(resized, (round((max_w - new_w) * 0.5), 0))
                img = canvas

    if invert:
        img = ImageOps.invert(img)

    img = apply_tone_lut(img, lightness=lightness, contrast=contrast)
    img = dither_image(img, mode=normalize_dither_mode(dither_mode))

    if rotate_180:
        img = img.rotate(180)
    return img


def _pil_to_pixbuf(img: Image.Image, max_w: int = 440, max_h: int = 250) -> GdkPixbuf.Pixbuf:
    thumb = img.copy()
    thumb.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    thumb.convert("RGB").save(buf, format="PNG")
    loader = GdkPixbuf.PixbufLoader.new_with_type("png")
    loader.write(buf.getvalue())
    loader.close()
    return loader.get_pixbuf()


class YogaBookEinkWindow(Gtk.Window):
    """Main Pop!_OS GTK3 window for Yoga Book C930 E-Ink & Keyboard control."""

    def __init__(self) -> None:
        super().__init__(title="Yoga Book C930 — E-Ink & Keyboard Control")
        self.set_default_size(760, 640)
        self.set_border_width(12)

        self._initializing = True
        self._settings = EinkSettings.load()
        self._usb_lock = threading.Lock()
        self._busy = False
        self._selected_image_path: Optional[str] = None
        self._sound_daemon = KeyboardSoundDaemon()
        self._eink_menu = EinkOnScreenMenu(self._settings)
        self._menu_active = False
        self._last_menu_tap_ts = 0.0
        self._activity_stop: Optional[threading.Event] = None
        self._hid_listener = EinkHidListener(
            on_touch=self._on_hid_touch,
            on_vendor=self._on_hid_vendor,
            on_raw=self._on_hid_raw,
        )

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.add(root)

        # Top status bar
        root.pack_start(self._build_status_banner(), False, False, 0)

        # Main tabbed notebook
        notebook = Gtk.Notebook()
        notebook.append_page(self._build_modes_tab(), Gtk.Label(label="Modes & Keyboard"))
        notebook.append_page(self._build_eink_touch_tab(), Gtk.Label(label="E-Ink Settings & Touch"))
        notebook.append_page(self._build_display_tab(), Gtk.Label(label="E-Ink Display"))
        notebook.append_page(self._build_advanced_tab(), Gtk.Label(label="Advanced & Logs"))
        root.pack_start(notebook, True, True, 0)

        # Bottom progress + status message
        self.progress = Gtk.ProgressBar()
        self.progress.set_show_text(True)
        self.progress.set_text("Ready")
        root.pack_start(self.progress, False, False, 0)

        self._hid_listener.swap_axes = self._settings.swap_axes
        self._hid_listener.flip_x = self._settings.flip_x
        self._hid_listener.flip_y = self._settings.flip_y
        self._sound_daemon.key_down_enabled = self._settings.key_down_sound
        self._sound_daemon.key_up_enabled = self._settings.key_up_sound
        if self._settings.sound_profile in ("soft", "crisp", "typewriter"):
            self._sound_daemon.profile = self._settings.sound_profile  # type: ignore[assignment]
        if self._sound_daemon.key_down_enabled or self._sound_daemon.key_up_enabled:
            _ok, msg = self._sound_daemon.start()
            self.lbl_sound_status.set_text(msg)
        self._initializing = False

        self.connect("destroy", self._on_destroy)
        GLib.idle_add(self._refresh_status_async)

    # ------------------------------------------------------------------
    # UI Builders
    # ------------------------------------------------------------------

    def _build_status_banner(self) -> Gtk.Frame:
        frame = Gtk.Frame(label="Hardware Status (USB 048d:8951 — ITE8951 T-CON)")
        grid = Gtk.Grid(column_spacing=16, row_spacing=4)
        grid.set_border_width(8)
        frame.add(grid)

        self.lbl_conn = Gtk.Label(label="Checking...", xalign=0)
        self.lbl_scenario = Gtk.Label(label="—", xalign=0)
        self.lbl_panel = Gtk.Label(label="—", xalign=0)
        self.lbl_regs = Gtk.Label(label="—", xalign=0)

        grid.attach(Gtk.Label(label="Connection:", xalign=0), 0, 0, 1, 1)
        grid.attach(self.lbl_conn, 1, 0, 1, 1)
        grid.attach(Gtk.Label(label="Active Mode:", xalign=0), 2, 0, 1, 1)
        grid.attach(self.lbl_scenario, 3, 0, 1, 1)

        grid.attach(Gtk.Label(label="Panel / TP FW:", xalign=0), 0, 1, 1, 1)
        grid.attach(self.lbl_panel, 1, 1, 1, 1)
        grid.attach(Gtk.Label(label="Registers:", xalign=0), 2, 1, 1, 1)
        grid.attach(self.lbl_regs, 3, 1, 1, 1)

        btn_refresh = Gtk.Button(label="Refresh Status")
        btn_refresh.connect("clicked", lambda *_: self._refresh_status_async())
        grid.attach(btn_refresh, 4, 0, 1, 2)

        return frame

    def _build_modes_tab(self) -> Gtk.Box:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_border_width(10)

        # Panel mode controls
        mode_frame = Gtk.Frame(label="E-Ink Panel & Input Mode")
        mode_grid = Gtk.Grid(column_spacing=10, row_spacing=10)
        mode_grid.set_border_width(10)
        mode_frame.add(mode_grid)

        btn_kb = Gtk.Button(label="Enable / Restore E-Ink Keyboard")
        btn_kb.get_style_context().add_class("suggested-action")
        btn_kb.set_tooltip_text(
            "Initializes the ITE8951 T-CON and restores firmware keyboard mode "
            "(also fixes keyboard loss after suspend/resume)."
        )
        btn_kb.connect("clicked", lambda *_: self._run_usb_task("Enabling Keyboard...", self._task_enable_kb))

        btn_draw = Gtk.Button(label="Owner-Draw / E-Ink Display Mode")
        btn_draw.set_tooltip_text(
            "Leaves keyboard mode and enables owner-draw (display_cfg = 0x002e0000) "
            "for pushing custom images, notes, or wallpapers."
        )
        btn_draw.connect("clicked", lambda *_: self._run_usb_task("Entering Draw Mode...", self._task_enable_draw))

        btn_mt = Gtk.Button(label="Multitouch / Pen-Mouse Mode (mt-arm)")
        btn_mt.set_tooltip_text(
            "Windows-validated multitouch latch (scenario=3 + finger MT bit 0x00080000) "
            "for using the E-Ink panel as a multitouch touchpad / pen surface."
        )
        btn_mt.connect("clicked", lambda *_: self._run_usb_task("Arming Multitouch Mode...", self._task_enable_mt))

        btn_leave = Gtk.Button(label="Leave Keyboard (Minimal)")
        btn_leave.set_tooltip_text("Stops firmware typing without arming multitouch.")
        btn_leave.connect("clicked", lambda *_: self._run_usb_task("Leaving Keyboard...", self._task_leave_kb))

        btn_init = Gtk.Button(label="Init / Doorknock (4x GET_SYS)")
        btn_init.connect("clicked", lambda *_: self._run_usb_task("Initializing Panel...", self._task_init))

        btn_drain = Gtk.Button(label="Drain Stale USB Pipe")
        btn_drain.set_tooltip_text("Use if a previous transfer timed out or got interrupted.")
        btn_drain.connect("clicked", lambda *_: self._run_usb_task("Draining USB IN...", self._task_drain))

        mode_grid.attach(btn_kb, 0, 0, 2, 1)
        mode_grid.attach(btn_draw, 0, 1, 1, 1)
        mode_grid.attach(btn_mt, 1, 1, 1, 1)
        mode_grid.attach(btn_leave, 0, 2, 1, 1)
        mode_grid.attach(btn_init, 1, 2, 1, 1)
        mode_grid.attach(btn_drain, 0, 3, 2, 1)
        mode_grid.set_column_homogeneous(True)
        box.pack_start(mode_frame, False, False, 0)

        # Windows-style Keyboard Feedback settings
        fb_frame = Gtk.Frame(label="Keyboard Feedback Settings (Windows EinkSvr Equivalents)")
        fb_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        fb_box.set_border_width(10)
        fb_frame.add(fb_box)

        ev_path = find_ite_keyboard_event_path() or "Not detected"
        self.lbl_sound_status = Gtk.Label(
            label=f"ITE T-CON Keyboard evdev: {ev_path}", xalign=0
        )
        fb_box.pack_start(self.lbl_sound_status, False, False, 0)

        row_switches = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=18)
        self.chk_key_down = Gtk.CheckButton(label="Key-Down Click Sound")
        self.chk_key_down.set_active(self._settings.key_down_sound)
        self.chk_key_down.connect("toggled", self._on_sound_toggled)
        self.chk_key_up = Gtk.CheckButton(label="Key-Up Release Sound")
        self.chk_key_up.set_active(self._settings.key_up_sound)
        self.chk_key_up.connect("toggled", self._on_sound_toggled)

        self.cmb_sound_profile = Gtk.ComboBoxText()
        for key, label in [("soft", "Soft Tactile"), ("crisp", "Crisp Click"), ("typewriter", "Typewriter")]:
            self.cmb_sound_profile.append(key, label)
        if not self.cmb_sound_profile.set_active_id(self._settings.sound_profile):
            self.cmb_sound_profile.set_active_id("soft")
        self.cmb_sound_profile.connect("changed", self._on_sound_profile_changed)

        row_switches.pack_start(self.chk_key_down, False, False, 0)
        row_switches.pack_start(self.chk_key_up, False, False, 0)
        row_switches.pack_start(Gtk.Label(label="Sound:"), False, False, 0)
        row_switches.pack_start(self.cmb_sound_profile, False, False, 0)
        fb_box.pack_start(row_switches, False, False, 0)

        row_kb_style = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.cmb_kb_style = Gtk.ComboBoxText()
        for st_id in KEYBOARD_STYLES:
            self.cmb_kb_style.append(st_id, KEYBOARD_STYLE_LABELS[st_id])
        if not self.cmb_kb_style.set_active_id(self._settings.keyboard_style):
            self.cmb_kb_style.set_active_id("classic_white")
        self.cmb_kb_style.connect("changed", self._on_kb_style_changed)
        row_kb_style.pack_start(Gtk.Label(label="Keyboard & Touchpad Style:"), False, False, 0)
        row_kb_style.pack_start(self.cmb_kb_style, False, False, 0)
        fb_box.pack_start(row_kb_style, False, False, 0)

        box.pack_start(fb_frame, False, False, 0)

        # Quick Waveform setter
        wf_frame = Gtk.Frame(label="Active Panel Waveform (Opcode 0xA9)")
        wf_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        wf_box.set_border_width(10)
        wf_frame.add(wf_box)

        for wf_id, short_label in [
            (WAVEFORM_INIT, "INIT (Clear)"),
            (WAVEFORM_DU2, "DU2 (Fast)"),
            (WAVEFORM_GC16, "GC16 (Grayscale)"),
            (WAVEFORM_GL16, "GL16 (Smooth)"),
        ]:
            b = Gtk.Button(label=short_label)
            b.connect(
                "clicked",
                lambda _btn, w=wf_id, name=short_label: self._run_usb_task(
                    f"Setting waveform {name}...", lambda c: self._task_set_waveform(c, w)
                ),
            )
            wf_box.pack_start(b, True, True, 0)

        box.pack_start(wf_frame, False, False, 0)
        return box

    def _build_eink_touch_tab(self) -> Gtk.Box:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_border_width(10)

        # On-screen E-Ink Settings Menu launcher
        menu_frame = Gtk.Frame(label="Interactive On-Screen E-Ink Settings Menu")
        menu_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        menu_vbox.set_border_width(10)
        menu_frame.add(menu_vbox)

        desc = Gtk.Label(
            label=(
                "Draws a touchable 8-tile Settings & Mode Dashboard directly on your E-Ink screen "
                "(Keyboard, Touchpad/MT, Ruled Notes, Sketch Grid, Key Sound, Clean Screen) and "
                "enables finger/stylus taps via hidraw0 (0x90) + panel button events (0x10/0x20)."
            ),
            xalign=0,
        )
        desc.set_line_wrap(True)
        menu_vbox.pack_start(desc, False, False, 0)

        btn_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        btn_show_menu = Gtk.Button(label="Show Interactive Settings Menu on E-Ink")
        btn_show_menu.get_style_context().add_class("suggested-action")
        btn_show_menu.connect("clicked", lambda *_: self._show_eink_menu_async())

        self.btn_hid_listen = Gtk.ToggleButton(label="Listen for E-Ink Taps & Settings Button")
        self.btn_hid_listen.connect("toggled", self._on_hid_listen_toggled)

        btn_row.pack_start(btn_show_menu, True, True, 0)
        btn_row.pack_start(self.btn_hid_listen, True, True, 0)
        menu_vbox.pack_start(btn_row, False, False, 0)

        # Behavior + coordinate calibration row
        opt_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=14)
        self.chk_vendor_opens_menu = Gtk.CheckButton(
            label="Top-right E-Ink Settings Icon opens Menu"
        )
        self.chk_vendor_opens_menu.set_active(True)

        self.chk_swap_axes = Gtk.CheckButton(label="Swap X/Y")
        self.chk_swap_axes.set_active(self._settings.swap_axes)
        self.chk_swap_axes.connect("toggled", self._on_touch_cal_changed)

        self.chk_flip_x = Gtk.CheckButton(label="Flip X")
        self.chk_flip_x.set_active(self._settings.flip_x)
        self.chk_flip_x.connect("toggled", self._on_touch_cal_changed)

        self.chk_flip_y = Gtk.CheckButton(label="Flip Y")
        self.chk_flip_y.set_active(self._settings.flip_y)
        self.chk_flip_y.connect("toggled", self._on_touch_cal_changed)

        opt_row.pack_start(self.chk_vendor_opens_menu, False, False, 0)
        opt_row.pack_start(self.chk_swap_axes, False, False, 0)
        opt_row.pack_start(self.chk_flip_x, False, False, 0)
        opt_row.pack_start(self.chk_flip_y, False, False, 0)
        menu_vbox.pack_start(opt_row, False, False, 0)

        devs = find_yogabook_hidraw_devices()
        dev_str = ", ".join(f"{os.path.basename(p)}" for p, _ in devs) if devs else "None"
        self.lbl_hid_status = Gtk.Label(
            label=f"Detected E-Ink hidraw devices: {dev_str} (listener stopped)", xalign=0
        )
        menu_vbox.pack_start(self.lbl_hid_status, False, False, 0)
        box.pack_start(menu_frame, False, False, 0)

        # Live HID / Tap Monitor
        mon_frame = Gtk.Frame(label="Live E-Ink Touch & Button Monitor (raw hidraw reports)")
        mon_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        mon_vbox.set_border_width(8)
        mon_frame.add(mon_vbox)

        self.chk_log_raw_hid = Gtk.CheckButton(label="Show raw hex packets (debug only)")
        self.chk_log_raw_hid.set_active(False)
        mon_vbox.pack_start(self.chk_log_raw_hid, False, False, 0)

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.hid_log_view = Gtk.TextView(editable=False, monospace=True)
        self.hid_log_buf = self.hid_log_view.get_buffer()
        scrolled.add(self.hid_log_view)
        mon_vbox.pack_start(scrolled, True, True, 0)

        box.pack_start(mon_frame, True, True, 0)
        return box

    def _build_display_tab(self) -> Gtk.Box:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_border_width(10)

        # Quick refresh / templates
        tpl_frame = Gtk.Frame(label="Quick Clear & E-Ink Templates (1920x1080)")
        tpl_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        tpl_box.set_border_width(8)
        tpl_frame.add(tpl_box)

        for label, kind, wf in [
            ("Clear White (INIT)", "white", WAVEFORM_INIT),
            ("Clear Black (INIT)", "black", WAVEFORM_INIT),
            ("Ruled Notepad", "ruled", WAVEFORM_GC16),
            ("Dot Grid (Stylus)", "dotgrid", WAVEFORM_GC16),
            ("Pop!_OS Banner", "banner", WAVEFORM_GC16),
        ]:
            btn = Gtk.Button(label=label)
            btn.connect(
                "clicked",
                lambda _b, k=kind, w=wf, lbl=label: self._send_template_async(k, w, lbl),
            )
            tpl_box.pack_start(btn, True, True, 0)
        box.pack_start(tpl_frame, False, False, 0)

        # Matrix Screensaver & E-Ink DOOM
        fun_frame = Gtk.Frame(label="E-Ink Apps & Screensavers (Live DU2 Animation)")
        fun_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        fun_box.set_border_width(8)
        fun_frame.add(fun_box)

        self.btn_matrix = Gtk.ToggleButton(label="Matrix Rain Screensaver")
        self.btn_matrix.connect("toggled", self._on_toggle_matrix)

        self.btn_doom = Gtk.ToggleButton(label="Play E-Ink DOOM (Touch Controls)")
        self.btn_doom.connect("toggled", self._on_toggle_doom)

        self.chk_matrix_light = Gtk.CheckButton(label="Light Paper Matrix")

        self.chk_idle_settle = Gtk.CheckButton(label="Idle GC16 Settle (Anti-Ghosting)")
        self.chk_idle_settle.set_active(self._settings.idle_settle_enabled)
        self.chk_idle_settle.connect("toggled", self._on_idle_settle_toggled)

        fun_box.pack_start(self.btn_matrix, True, True, 0)
        fun_box.pack_start(self.btn_doom, True, True, 0)
        fun_box.pack_start(self.chk_matrix_light, False, False, 0)
        fun_box.pack_start(self.chk_idle_settle, False, False, 0)
        box.pack_start(fun_frame, False, False, 0)

        # Custom image sender
        img_frame = Gtk.Frame(label="Send Custom Image / Wallpaper to E-Ink Panel")
        img_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        img_vbox.set_border_width(10)
        img_frame.add(img_vbox)

        row_file = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.lbl_file = Gtk.Label(label="No image selected", xalign=0)
        self.lbl_file.set_ellipsize(3)
        btn_pick = Gtk.Button(label="Choose Image...")
        btn_pick.connect("clicked", self._on_choose_image)
        row_file.pack_start(btn_pick, False, False, 0)
        row_file.pack_start(self.lbl_file, True, True, 0)
        img_vbox.pack_start(row_file, False, False, 0)

        row_opts = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=14)
        self.cmb_fit = Gtk.ComboBoxText()
        self.cmb_fit.append("contain", "Fit (Pad White)")
        self.cmb_fit.append("cover", "Fill (Crop)")
        self.cmb_fit.set_active_id("contain")
        self.cmb_fit.connect("changed", lambda *_: self._update_image_preview())

        self.cmb_wf = Gtk.ComboBoxText()
        for wf_id in (WAVEFORM_GC16, WAVEFORM_GL16, WAVEFORM_DU2, WAVEFORM_INIT, WAVEFORM_CURRENT):
            self.cmb_wf.append(str(wf_id), WAVEFORM_NAMES[wf_id])
        if not self.cmb_wf.set_active_id(str(self._settings.waveform)):
            self.cmb_wf.set_active_id(str(WAVEFORM_GC16))
        self.cmb_wf.connect("changed", self._on_waveform_changed)

        self.cmb_dither = Gtk.ComboBoxText()
        for mode in DITHER_MODES:
            self.cmb_dither.append(mode, mode)
        if not self.cmb_dither.set_active_id(self._settings.dither_mode):
            self.cmb_dither.set_active_id(DITHER_MODES[0])
        self.cmb_dither.connect("changed", self._on_dither_changed)

        self.chk_invert = Gtk.CheckButton(label="Invert")
        self.chk_invert.connect("toggled", lambda *_: self._update_image_preview())
        self.chk_rot180 = Gtk.CheckButton(label="Rotate 180°")
        self.chk_rot180.set_active(self._settings.rotate_180)
        self.chk_rot180.connect("toggled", self._on_rot180_toggled)

        row_opts.pack_start(self.cmb_fit, False, False, 0)
        row_opts.pack_start(self.cmb_wf, False, False, 0)
        row_opts.pack_start(self.cmb_dither, False, False, 0)
        row_opts.pack_start(self.chk_invert, False, False, 0)
        row_opts.pack_start(self.chk_rot180, False, False, 0)
        img_vbox.pack_start(row_opts, False, False, 0)

        row_tone = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=14)
        row_tone.set_homogeneous(True)

        row_tone_l = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        row_tone_l.pack_start(Gtk.Label(label="Lightness:", xalign=0), False, False, 0)
        self.scl_lightness = Gtk.Scale.new_with_range(
            Gtk.Orientation.HORIZONTAL, LIGHTNESS_MIN, LIGHTNESS_MAX, 1
        )
        self.scl_lightness.set_digits(0)
        self.scl_lightness.set_round_digits(0)
        self.scl_lightness.set_value_pos(Gtk.PositionType.RIGHT)
        self.scl_lightness.set_hexpand(True)
        self.scl_lightness.set_value(self._settings.lightness)
        self.scl_lightness.connect("value-changed", self._on_tone_changed)
        row_tone_l.pack_start(self.scl_lightness, True, True, 0)

        row_tone_c = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        row_tone_c.pack_start(Gtk.Label(label="Contrast:", xalign=0), False, False, 0)
        self.scl_contrast = Gtk.Scale.new_with_range(
            Gtk.Orientation.HORIZONTAL, CONTRAST_MIN, CONTRAST_MAX, 1
        )
        self.scl_contrast.set_digits(0)
        self.scl_contrast.set_round_digits(0)
        self.scl_contrast.set_value_pos(Gtk.PositionType.RIGHT)
        self.scl_contrast.set_hexpand(True)
        self.scl_contrast.set_value(self._settings.contrast)
        self.scl_contrast.connect("value-changed", self._on_tone_changed)
        row_tone_c.pack_start(self.scl_contrast, True, True, 0)

        row_tone.pack_start(row_tone_l, True, True, 0)
        row_tone.pack_start(row_tone_c, True, True, 0)
        img_vbox.pack_start(row_tone, False, False, 0)

        self.img_preview = Gtk.Image()
        self.img_preview.set_size_request(440, 220)
        img_vbox.pack_start(self.img_preview, True, True, 0)

        btn_send = Gtk.Button(label="Send Image to E-Ink Display")
        btn_send.get_style_context().add_class("suggested-action")
        btn_send.connect("clicked", self._on_send_image)
        img_vbox.pack_start(btn_send, False, False, 0)

        box.pack_start(img_frame, True, True, 0)
        return box

    def _build_advanced_tab(self) -> Gtk.Box:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_border_width(10)

        # Register / TP area controls
        reg_frame = Gtk.Frame(label="T-CON Registers & Touch Area Routing")
        reg_grid = Gtk.Grid(column_spacing=8, row_spacing=8)
        reg_grid.set_border_width(8)
        reg_frame.add(reg_grid)

        self.ent_reg_addr = Gtk.Entry(text="0x18001138")
        self.ent_reg_val = Gtk.Entry(text="0x002e0000")
        btn_read_reg = Gtk.Button(label="Read Reg (0x83)")
        btn_read_reg.connect("clicked", self._on_read_reg)
        btn_write_reg = Gtk.Button(label="Write Reg (0x84)")
        btn_write_reg.connect("clicked", self._on_write_reg)

        reg_grid.attach(Gtk.Label(label="Register Addr:", xalign=0), 0, 0, 1, 1)
        reg_grid.attach(self.ent_reg_addr, 1, 0, 1, 1)
        reg_grid.attach(btn_read_reg, 2, 0, 1, 1)
        reg_grid.attach(Gtk.Label(label="Value (hex):", xalign=0), 0, 1, 1, 1)
        reg_grid.attach(self.ent_reg_val, 1, 1, 1, 1)
        reg_grid.attach(btn_write_reg, 2, 1, 1, 1)

        tp_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        for lbl, flag in [
            ("TP Full: Touch+Pen", TP_AREA_TOUCH_PEN),
            ("TP Full: Touch Only", TP_AREA_TOUCH_ONLY),
            ("TP Full: Pen Only", TP_AREA_PEN_ONLY),
            ("TP Clear (No Report)", TP_AREA_NO_REPORT),
        ]:
            b = Gtk.Button(label=lbl)
            b.connect(
                "clicked",
                lambda _btn, fl=flag, name=lbl: self._run_usb_task(
                    f"Applying {name}...",
                    lambda c: self._task_set_tp_area(c, fl),
                ),
            )
            tp_box.pack_start(b, True, True, 0)
        reg_grid.attach(tp_box, 0, 2, 3, 1)
        box.pack_start(reg_frame, False, False, 0)

        # Activity log
        log_frame = Gtk.Frame(label="USB Protocol Activity Log")
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.log_view = Gtk.TextView(editable=False, monospace=True)
        self.log_buf = self.log_view.get_buffer()
        scrolled.add(self.log_view)
        log_frame.add(scrolled)
        box.pack_start(log_frame, True, True, 0)

        return box

    # ------------------------------------------------------------------
    # Logging & Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"

        def _append() -> bool:
            end_iter = self.log_buf.get_end_iter()
            self.log_buf.insert(end_iter, line)
            if self.log_view.get_realized():
                self.log_view.scroll_to_iter(self.log_buf.get_end_iter(), 0.0, False, 0.0, 0.0)
            return False

        GLib.idle_add(_append)

    def _set_progress(self, fraction: float, text: str) -> None:
        GLib.idle_add(self.progress.set_fraction, max(0.0, min(1.0, fraction)))
        GLib.idle_add(self.progress.set_text, text)

    def _run_usb_task(
        self,
        status_text: str,
        fn: Callable[[Ite8951Controller], str],
    ) -> None:
        if self._busy:
            self._log("Busy — wait for the current USB operation to finish.")
            return
        self._busy = True
        self._set_progress(0.15, status_text)

        def _worker() -> None:
            try:
                with self._usb_lock:
                    with Ite8951Controller() as ctrl:
                        result = fn(ctrl)
                self._log(result)
                self._set_progress(1.0, result)
            except (UsbError, OSError, ValueError) as exc:
                err_msg = f"Error: {exc}"
                self._log(err_msg)
                self._set_progress(0.0, err_msg)
            finally:
                self._busy = False
                GLib.idle_add(self._refresh_status_async)

        threading.Thread(target=_worker, daemon=True).start()

    def _refresh_status_async(self) -> None:
        if self._busy:
            return

        def _worker() -> None:
            try:
                with self._usb_lock:
                    with Ite8951Controller() as ctrl:
                        info = ctrl.get_system_info()
                        scenario = ctrl.get_scenario()
                        disp_cfg = ctrl.read_reg(REG_DISPLAY_CFG)
                        panel_mode = ctrl.read_reg(REG_PANEL_MODE)
                conn_txt = f"Connected (sig: {info.signature_str})"
                scen_txt = SCENARIO_NAMES.get(scenario, f"Unknown ({scenario})")
                panel_txt = f"{info.width}×{info.height} | TP v{info.tp_version_str} | EPD={info.epd_type}"
                reg_txt = f"cfg=0x{disp_cfg:08x} | mode=0x{panel_mode:08x}"
            except UsbError as exc:
                conn_txt = "Permission / Access Error" if exc.code == -3 else "Disconnected"
                scen_txt = str(exc)
                panel_txt = "Run ./install.sh (udev rule) or sudo"
                reg_txt = "—"

            GLib.idle_add(self.lbl_conn.set_text, conn_txt)
            GLib.idle_add(self.lbl_scenario.set_text, scen_txt)
            GLib.idle_add(self.lbl_panel.set_text, panel_txt)
            GLib.idle_add(self.lbl_regs.set_text, reg_txt)

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # USB Tasks
    # ------------------------------------------------------------------

    def _task_enable_kb(self, ctrl: Ite8951Controller) -> str:
        scen = ctrl.exit_to_keyboard(style=self._settings.keyboard_style)
        self._settings = EinkSettings.load()
        return f"Keyboard restored and refreshed (scenario={scen}, style={self._settings.keyboard_style})."

    def _task_enable_draw(self, ctrl: Ite8951Controller) -> str:
        scen = ctrl.enable_draw_mode()
        return f"Owner-draw / display mode enabled (scenario={scen}, cfg=0x002e0000)."

    def _task_enable_mt(self, ctrl: Ite8951Controller) -> str:
        scen, cfg = ctrl.enable_multitouch_mode()
        return f"Multitouch mode armed (scenario={scen}, display_cfg=0x{cfg:08x})."

    def _task_leave_kb(self, ctrl: Ite8951Controller) -> str:
        scen = ctrl.leave_keyboard()
        return f"Left firmware keyboard (scenario={scen})."

    def _task_init(self, ctrl: Ite8951Controller) -> str:
        info = ctrl.init_panel()
        return f"Initialized panel: {info.width}x{info.height}, TP v{info.tp_version_str}."

    def _task_drain(self, ctrl: Ite8951Controller) -> str:
        n = ctrl.drain_stale_in()
        return f"Drained {n} stale bytes from bulk IN."

    def _task_set_waveform(self, ctrl: Ite8951Controller, wf: int) -> str:
        ctrl.set_waveform(wf)
        return f"Waveform set to {WAVEFORM_NAMES.get(wf, wf)}."

    def _task_set_tp_area(self, ctrl: Ite8951Controller, flag: int) -> str:
        ctrl.set_tp_area(0, 0, PANEL_WIDTH, PANEL_HEIGHT, index=0, flag=flag)
        return f"Set TP area 0 (full panel) to flag=0x{flag:02x}."

    # ------------------------------------------------------------------
    # Sound Feedback Callbacks
    # ------------------------------------------------------------------

    def _on_sound_toggled(self, _btn: Gtk.CheckButton) -> None:
        if self._initializing:
            return
        self._sound_daemon.key_down_enabled = self.chk_key_down.get_active()
        self._sound_daemon.key_up_enabled = self.chk_key_up.get_active()
        self._settings.key_down_sound = self._sound_daemon.key_down_enabled
        self._settings.key_up_sound = self._sound_daemon.key_up_enabled
        self._settings.save()
        if self._sound_daemon.key_down_enabled or self._sound_daemon.key_up_enabled:
            _ok, msg = self._sound_daemon.start()
            self.lbl_sound_status.set_text(msg)
            self._log(f"Key sounds: {msg}")
        else:
            self._sound_daemon.stop()
            self._log("Key sounds stopped.")

    def _on_sound_profile_changed(self, combo: Gtk.ComboBoxText) -> None:
        if self._initializing:
            return
        prof = combo.get_active_id()
        if prof in ("soft", "crisp", "typewriter"):
            self._sound_daemon.profile = prof  # type: ignore[assignment]
            self._settings.sound_profile = prof
            self._settings.save()
            self._log(f"Key sound profile: {prof}")

    def _on_kb_style_changed(self, combo: Gtk.ComboBoxText) -> None:
        if self._initializing:
            return
        st = normalize_keyboard_style(combo.get_active_id() or "classic_white")
        self._settings.keyboard_style = st
        self._eink_menu.settings.keyboard_style = st
        self._settings.save()
        self._log(f"Keyboard & Touchpad Style: {KEYBOARD_STYLE_LABELS.get(st, st)}")

    # ------------------------------------------------------------------
    # Display / Image Callbacks
    # ------------------------------------------------------------------

    def _on_choose_image(self, _btn: Gtk.Button) -> None:
        dialog = Gtk.FileChooserDialog(
            title="Select Image for E-Ink Panel",
            parent=self,
            action=Gtk.FileChooserAction.OPEN,
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN, Gtk.ResponseType.OK,
        )
        flt = Gtk.FileFilter()
        flt.set_name("Images (PNG, JPG, BMP, WebP)")
        flt.add_pixbuf_formats()
        dialog.add_filter(flt)

        if dialog.run() == Gtk.ResponseType.OK:
            self._selected_image_path = dialog.get_filename()
            self.lbl_file.set_text(os.path.basename(self._selected_image_path or ""))
            self._update_image_preview()
        dialog.destroy()

    def _on_tone_changed(self, _scale: Gtk.Range) -> None:
        if self._initializing:
            return
        self._settings.lightness = clamp_lightness(int(round(self.scl_lightness.get_value())))
        self._settings.contrast = clamp_contrast(int(round(self.scl_contrast.get_value())))
        self._settings.save()
        self._update_image_preview()

    def _on_dither_changed(self, combo: Gtk.ComboBoxText) -> None:
        if self._initializing:
            return
        raw_mode = combo.get_active_id() or combo.get_active_text() or self._settings.dither_mode
        self._settings.dither_mode = normalize_dither_mode(raw_mode)
        self._settings.save()
        self._update_image_preview()

    def _on_waveform_changed(self, combo: Gtk.ComboBoxText) -> None:
        if self._initializing:
            return
        self._settings.waveform = int(combo.get_active_id() or WAVEFORM_GC16)
        self._settings.save()

    def _on_rot180_toggled(self, btn: Gtk.CheckButton) -> None:
        if self._initializing:
            return
        self._settings.rotate_180 = btn.get_active()
        self._settings.save()
        self._update_image_preview()

    def _on_idle_settle_toggled(self, btn: Gtk.CheckButton) -> None:
        if self._initializing:
            return
        self._settings.idle_settle_enabled = btn.get_active()
        self._settings.save()

    def _update_image_preview(self) -> None:
        if not self._selected_image_path:
            return
        try:
            fit = "cover" if self.cmb_fit.get_active_id() == "cover" else "contain"
            dither_mode = normalize_dither_mode(
                self.cmb_dither.get_active_id() or self.cmb_dither.get_active_text() or self._settings.dither_mode
            )
            lightness = clamp_lightness(int(round(self.scl_lightness.get_value())))
            contrast = clamp_contrast(int(round(self.scl_contrast.get_value())))
            img = _prepare_preview_pil(
                self._selected_image_path,
                fit_mode=fit,
                invert=self.chk_invert.get_active(),
                rotate_180=self.chk_rot180.get_active(),
                dither_mode=dither_mode,
                lightness=lightness,
                contrast=contrast,
            )
            self.img_preview.set_from_pixbuf(_pil_to_pixbuf(img))
        except Exception as exc:
            self._log(f"Preview error: {exc}")

    def _on_send_image(self, _btn: Gtk.Button) -> None:
        if not self._selected_image_path:
            self._log("Select an image first.")
            return
        path = self._selected_image_path
        fit = "cover" if self.cmb_fit.get_active_id() == "cover" else "contain"
        invert = self.chk_invert.get_active()
        rot180 = self.chk_rot180.get_active()
        dither_mode = normalize_dither_mode(
            self.cmb_dither.get_active_id() or self.cmb_dither.get_active_text() or self._settings.dither_mode
        )
        lightness = clamp_lightness(int(round(self.scl_lightness.get_value())))
        contrast = clamp_contrast(int(round(self.scl_contrast.get_value())))
        wf = int(self.cmb_wf.get_active_id() or WAVEFORM_GC16)

        def _task(ctrl: Ite8951Controller) -> str:
            _, packed = prepare_image_file(
                path,
                fit_mode=fit,
                invert=invert,
                rotate_180=rot180,
                dither_mode=dither_mode,
                lightness=lightness,
                contrast=contrast,
            )
            ctrl.enable_draw_mode()
            ctrl.display_image_bytes(
                packed,
                PANEL_WIDTH,
                PANEL_HEIGHT,
                waveform=wf,
                progress_cb=lambda frac: self._set_progress(frac, f"Uploading image... {int(frac * 100)}%"),
            )
            return f"Displayed {os.path.basename(path)} (waveform={wf})."

        self._run_usb_task("Preparing & uploading image...", _task)

    def _send_template_async(self, kind: str, waveform: int, label: str) -> None:
        invert = self.chk_invert.get_active()
        rot180 = self.chk_rot180.get_active()

        def _task(ctrl: Ite8951Controller) -> str:
            img, packed = generate_template(kind, invert=invert, rotate_180=rot180)  # type: ignore[arg-type]
            GLib.idle_add(self.img_preview.set_from_pixbuf, _pil_to_pixbuf(img))
            ctrl.enable_draw_mode()
            ctrl.display_image_bytes(
                packed,
                PANEL_WIDTH,
                PANEL_HEIGHT,
                waveform=waveform,
                progress_cb=lambda frac: self._set_progress(frac, f"Sending {label}... {int(frac * 100)}%"),
            )
            return f"Displayed template: {label}."

        self._run_usb_task(f"Sending {label}...", _task)

    # ------------------------------------------------------------------
    # Advanced Register Callbacks
    # ------------------------------------------------------------------

    def _on_read_reg(self, _btn: Gtk.Button) -> None:
        try:
            addr = int(self.ent_reg_addr.get_text().strip(), 0)
        except ValueError:
            self._log("Invalid register address.")
            return

        def _task(ctrl: Ite8951Controller) -> str:
            val = ctrl.read_reg(addr)
            GLib.idle_add(self.ent_reg_val.set_text, f"0x{val:08x}")
            return f"READ_REG(0x{addr:08x}) = 0x{val:08x}"

        self._run_usb_task(f"Reading 0x{addr:08x}...", _task)

    def _on_write_reg(self, _btn: Gtk.Button) -> None:
        try:
            addr = int(self.ent_reg_addr.get_text().strip(), 0)
            val = int(self.ent_reg_val.get_text().strip(), 0)
        except ValueError:
            self._log("Invalid register address or value.")
            return

        def _task(ctrl: Ite8951Controller) -> str:
            ctrl.write_reg(addr, val)
            rb = ctrl.read_reg(addr)
            return f"WRITE_REG(0x{addr:08x}, 0x{val:08x}) -> readback 0x{rb:08x}"

        self._run_usb_task(f"Writing 0x{addr:08x}...", _task)

    def _on_destroy(self, *_args) -> None:
        if self._activity_stop:
            self._activity_stop.set()
        self._hid_listener.stop()
        self._sound_daemon.stop()
        Gtk.main_quit()

    # ------------------------------------------------------------------
    # Matrix Screensaver & E-Ink DOOM Callbacks
    # ------------------------------------------------------------------

    def _stop_active_app(self) -> None:
        if self._activity_stop:
            self._activity_stop.set()
            self._activity_stop = None

    def _on_toggle_matrix(self, btn: Gtk.ToggleButton) -> None:
        if not btn.get_active():
            self._stop_active_app()
            return
        if self.btn_doom.get_active():
            self.btn_doom.set_active(False)
        self._stop_active_app()
        stop_ev = threading.Event()
        self._activity_stop = stop_ev
        dark_bg = not self.chk_matrix_light.get_active()
        self._set_progress(0.5, "Running Matrix Rain on E-Ink...")

        def _worker() -> None:
            try:
                run_matrix_screensaver(
                    stop_ev,
                    dark_bg=dark_bg,
                    on_frame=lambda n: self._set_progress((n % 20) / 20.0, f"Matrix Rain — frame {n}"),
                )
            except Exception as exc:
                self._log(f"Matrix error: {exc}")
            finally:
                GLib.idle_add(self.btn_matrix.set_active, False)
                self._set_progress(0.0, "Matrix screensaver stopped.")

        threading.Thread(target=_worker, daemon=True, name="eink-matrix").start()

    def _on_toggle_doom(self, btn: Gtk.ToggleButton) -> None:
        if not btn.get_active():
            self._stop_active_app()
            return
        if self.btn_matrix.get_active():
            self.btn_matrix.set_active(False)
        self._stop_active_app()
        stop_ev = threading.Event()
        self._activity_stop = stop_ev
        self._set_progress(0.5, "Running E-Ink DOOM — use touch buttons on E-Ink screen!")

        def _worker() -> None:
            try:
                run_eink_doom(
                    stop_ev,
                    restore_kb_on_exit=True,
                    on_status=lambda m: self._log(m),
                )
            except Exception as exc:
                self._log(f"DOOM error: {exc}")
            finally:
                GLib.idle_add(self.btn_doom.set_active, False)
                self._set_progress(0.0, "Exited E-Ink DOOM.")

        threading.Thread(target=_worker, daemon=True, name="eink-doom").start()

    # ------------------------------------------------------------------
    # E-Ink On-Screen Settings Menu & HID Touch/Button Callbacks
    # ------------------------------------------------------------------

    def _hid_log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"

        def _append() -> bool:
            if self.hid_log_buf.get_line_count() > 200:
                start = self.hid_log_buf.get_start_iter()
                cut = self.hid_log_buf.get_iter_at_line(50)
                self.hid_log_buf.delete(start, cut)
            end_iter = self.hid_log_buf.get_end_iter()
            self.hid_log_buf.insert(end_iter, line)
            if self.hid_log_view.get_realized():
                self.hid_log_view.scroll_to_iter(self.hid_log_buf.get_end_iter(), 0.0, False, 0.0, 0.0)
            return False

        GLib.idle_add(_append)

    def _on_touch_cal_changed(self, _btn: Gtk.CheckButton) -> None:
        if self._initializing:
            return
        self._hid_listener.swap_axes = self.chk_swap_axes.get_active()
        self._hid_listener.flip_x = self.chk_flip_x.get_active()
        self._hid_listener.flip_y = self.chk_flip_y.get_active()
        self._settings.swap_axes = self._hid_listener.swap_axes
        self._settings.flip_x = self._hid_listener.flip_x
        self._settings.flip_y = self._hid_listener.flip_y
        self._settings.save()

    def _on_hid_listen_toggled(self, btn: Gtk.ToggleButton) -> None:
        if btn.get_active():
            ok, msg = self._hid_listener.start()
            self.lbl_hid_status.set_text(msg)
            self._hid_log(msg)
            if not ok:
                btn.set_active(False)
        else:
            self._hid_listener.stop()
            self._menu_active = False
            self.lbl_hid_status.set_text("HID listener stopped.")
            self._hid_log("HID listener stopped.")

    def _show_eink_menu_async(self, status_line: str = "Tap any setting to change  •  Tap ← Keyboard to return") -> None:
        self._eink_menu.settings = self._settings
        self._settings.key_down_sound = self._sound_daemon.key_down_enabled
        self._settings.key_up_sound = self._sound_daemon.key_up_enabled
        self._settings.sound_profile = self._sound_daemon.profile

        # Ensure HID listener is running so taps on the E-Ink screen work immediately
        if not self._hid_listener.is_running:
            GLib.idle_add(self.btn_hid_listen.set_active, True)

        def _task(ctrl: Ite8951Controller) -> str:
            img, packed = self._eink_menu.render(status_line=status_line)
            GLib.idle_add(self.img_preview.set_from_pixbuf, _pil_to_pixbuf(img))
            ctrl.enter_app_mode(clean_color=0x0F if self._eink_menu.settings.dark_theme else 0xFF)
            ctrl.display_image_bytes(
                packed,
                PANEL_WIDTH,
                PANEL_HEIGHT,
                waveform=self._eink_menu.settings.waveform,
                progress_cb=lambda f: self._set_progress(f, f"Drawing E-Ink Settings... {int(f*100)}%"),
            )
            self._menu_active = True
            return "E-Ink Settings active — tap any setting on the E-Ink screen."

        self._run_usb_task("Drawing E-Ink Settings...", _task)

    def _on_hid_raw(self, dev_name: str, data: bytes) -> None:
        if self.chk_log_raw_hid.get_active():
            self._hid_log(f"{dev_name} [{len(data)}B] id=0x{data[0]:02x} hex={data.hex(' ')}")

    def _on_hid_vendor(self, ev: EinkVendorEvent) -> None:
        self._hid_log(
            f"HID VENDOR {ev.device} report=0x{ev.report_id:02x} data={ev.payload.hex(' ')}"
        )

    def _on_hid_touch(self, ev: EinkTouchEvent) -> None:
        if not ev.is_down:
            return
        self._hid_log(
            f"TOUCH DOWN panel=({ev.panel_x}, {ev.panel_y}) raw=({ev.raw_x}, {ev.raw_y}) z={ev.pressure}"
        )
        if self._busy:
            return

        now = time.monotonic()
        if now - self._last_menu_tap_ts < 0.7:
            return

        # When NOT in menu mode (e.g. using the E-Ink keyboard), tapping the top-right
        # Settings / Homebar icon (verified live at panel ~ (1866, 18)) opens the Settings Menu!
        if not self._menu_active:
            if (
                self.chk_vendor_opens_menu.get_active()
                and ev.panel_x >= 1700
                and ev.panel_y <= 140
            ):
                self._last_menu_tap_ts = now
                self._hid_log("SETTINGS ICON TAPPED on E-Ink -> Opening Interactive Settings Menu!")
                GLib.idle_add(self._show_eink_menu_async, "Opened from E-Ink Settings button")
            return

        tile = self._eink_menu.hit_test(ev.panel_x, ev.panel_y)
        if tile is None:
            return
        self._last_menu_tap_ts = now
        self._hid_log(f"MENU TAP -> {tile.action_id} ({tile.title})")
        GLib.idle_add(self._dispatch_menu_action, tile.action_id)

    def _partial_refresh_eink_widget(self, action_id: str, status_line: str) -> None:
        self._eink_menu.settings.key_down_sound = self._sound_daemon.key_down_enabled
        self._eink_menu.settings.key_up_sound = self._sound_daemon.key_up_enabled
        self._eink_menu.settings.sound_profile = self._sound_daemon.profile

        def _task(ctrl: Ite8951Controller) -> str:
            patches = self._eink_menu.render_widget_patches(action_id, status_line=status_line)
            for x, y, w, h, pixels in patches:
                ctrl.display_region(x, y, w, h, pixels, waveform=WAVEFORM_DU2)
            return f"Partial refresh: {status_line}"

        self._run_usb_task(status_line, _task)

    def _dispatch_menu_action(self, action_id: str) -> None:
        if action_id == "kb":
            self._menu_active = False
            self._run_usb_task("Switching to E-Ink Keyboard...", self._task_enable_kb)
        elif action_id == "page_settings":
            self._eink_menu.page = "settings"
            self._show_eink_menu_async("")
        elif action_id == "page_games":
            self._eink_menu.page = "games"
            self._show_eink_menu_async("")
        elif action_id == "page_apps":
            self._eink_menu.page = "apps"
            self._show_eink_menu_async("")
        elif action_id == "page_tools":
            self._eink_menu.page = "tools"
            self._show_eink_menu_async("")
        elif action_id == "page_calendar":
            self._eink_menu.page = "calendar"
            self._show_eink_menu_async("")
        elif action_id == "cal_prev":
            self._eink_menu.shift_calendar_month(-1)
            self._show_eink_menu_async("")
        elif action_id == "cal_next":
            self._eink_menu.shift_calendar_month(1)
            self._show_eink_menu_async("")
        elif action_id == "cal_today":
            import datetime
            today = datetime.date.today()
            self._eink_menu.cal_year, self._eink_menu.cal_month = today.year, today.month
            self._show_eink_menu_async("")
        elif action_id == "page_battery":
            self._eink_menu.page = "battery"
            self._show_eink_menu_async("")
        elif action_id == "page_dashboard":
            self._eink_menu.page = "dashboard"
            self._show_eink_menu_async("")
        elif action_id == "launch_doom":
            self._menu_active = False
            self.btn_doom.set_active(True)
        elif action_id in ("launch_matrix", "launch_matrix_light"):
            self._menu_active = False
            self.chk_matrix_light.set_active(action_id == "launch_matrix_light")
            self.btn_matrix.set_active(True)
        elif action_id == "mt":
            self._menu_active = False
            self._run_usb_task("Arming Multitouch Touchpad...", self._task_enable_mt)
        elif action_id == "ruled":
            self._menu_active = False
            self._send_template_async("ruled", self._eink_menu.settings.waveform, "Ruled Notepad")
        elif action_id == "dotgrid":
            self._menu_active = False
            self._send_template_async("dotgrid", self._eink_menu.settings.waveform, "Dot Grid")
        elif action_id == "toggle_key_down":
            new_state = not self.chk_key_down.get_active()
            self.chk_key_down.set_active(new_state)
            self._eink_menu.settings.key_down_sound = new_state
            self._eink_menu.settings.save()
            self._partial_refresh_eink_widget(
                action_id, f"Key-Down Sound: {'ON' if new_state else 'OFF'}"
            )
        elif action_id == "toggle_key_up":
            new_state = not self.chk_key_up.get_active()
            self.chk_key_up.set_active(new_state)
            self._eink_menu.settings.key_up_sound = new_state
            self._eink_menu.settings.save()
            self._partial_refresh_eink_widget(
                action_id, f"Key-Up Sound: {'ON' if new_state else 'OFF'}"
            )
        elif action_id == "cycle_sound":
            order: list[SoundProfile] = ["soft", "crisp", "typewriter"]
            cur = self._sound_daemon.profile
            nxt = order[(order.index(cur) + 1) % len(order)]
            self.cmb_sound_profile.set_active_id(nxt)
            self._eink_menu.settings.sound_profile = nxt
            self._eink_menu.settings.save()
            self._partial_refresh_eink_widget(action_id, f"Sound Style: {nxt.title()}")
        elif action_id == "toggle_theme":
            self._eink_menu.settings.dark_theme = not self._eink_menu.settings.dark_theme
            self._eink_menu.settings.save()
            self._show_eink_menu_async(
                f"Theme: {'Dark' if self._eink_menu.settings.dark_theme else 'Light'}"
            )
        elif action_id == "cycle_waveform":
            wfs = [WAVEFORM_DU2, WAVEFORM_GL16, WAVEFORM_GC16]
            cur = self._settings.waveform
            nxt_wf = wfs[(wfs.index(cur) + 1) % len(wfs)] if cur in wfs else WAVEFORM_GC16
            self._settings.waveform = nxt_wf
            self._settings.save()
            self.cmb_wf.set_active_id(str(nxt_wf))
            self._partial_refresh_eink_widget(action_id, "Updated E-Ink refresh mode")
        elif action_id == "cycle_lightness":
            cur = clamp_lightness(self._settings.lightness)
            nxt_l = LIGHTNESS_MIN if cur >= LIGHTNESS_MAX else cur + 1
            self._settings.lightness = nxt_l
            self._settings.save()
            self.scl_lightness.set_value(nxt_l)
            lbl = f"{nxt_l:+d}" if nxt_l else "0"
            self._partial_refresh_eink_widget(action_id, f"Tone Lightness: {lbl}")
        elif action_id == "cycle_contrast":
            cur = clamp_contrast(self._settings.contrast)
            nxt_c = CONTRAST_MIN if cur >= CONTRAST_MAX else cur + 1
            self._settings.contrast = nxt_c
            self._settings.save()
            self.scl_contrast.set_value(nxt_c)
            lbl = f"{nxt_c:+d}" if nxt_c else "0"
            self._partial_refresh_eink_widget(action_id, f"Tone Contrast: {lbl}")
        elif action_id == "toggle_idle_settle":
            new_state = not self.chk_idle_settle.get_active()
            self.chk_idle_settle.set_active(new_state)
            self._settings.idle_settle_enabled = new_state
            self._settings.save()
            self._partial_refresh_eink_widget(
                action_id, f"Idle Anti-Ghosting: {'ON' if new_state else 'OFF'}"
            )
        elif action_id == "cycle_kb_style":
            cur_st = normalize_keyboard_style(self._settings.keyboard_style)
            nxt_st = KEYBOARD_STYLES[(KEYBOARD_STYLES.index(cur_st) + 1) % len(KEYBOARD_STYLES)]
            self._settings.keyboard_style = nxt_st
            self._settings.save()
            self.cmb_kb_style.set_active_id(nxt_st)
            lbl = KEYBOARD_STYLE_LABELS.get(nxt_st, nxt_st)
            self._partial_refresh_eink_widget(action_id, f"Keyboard Style: {lbl}")
        elif action_id == "clear_init":
            def _clean_and_redraw(ctrl: Ite8951Controller) -> str:
                ctrl.clear_screen(fill_byte=0xFF, waveform=WAVEFORM_INIT)
                img, packed = self._eink_menu.render(status_line="Ghosting cleared (INIT refresh)")
                GLib.idle_add(self.img_preview.set_from_pixbuf, _pil_to_pixbuf(img))
                ctrl.display_image_bytes(
                    packed, PANEL_WIDTH, PANEL_HEIGHT, waveform=self._eink_menu.settings.waveform
                )
                self._menu_active = True
                return "Cleared ghosting and redrew E-Ink Settings."

            self._run_usb_task("Clearing ghosting (INIT)...", _clean_and_redraw)
        elif action_id == "open_gui":
            self.present()
            self._hid_log("Raised Pop!_OS Control Window on main LCD.")


def main() -> None:
    win = YogaBookEinkWindow()
    win.show_all()
    Gtk.main()
