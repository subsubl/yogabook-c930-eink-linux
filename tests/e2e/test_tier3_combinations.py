"""Tier 3 E2E Cross-Feature Pairwise Combination Tests (14 tests: T3-01..T3-14)."""

from __future__ import annotations

import inspect
import os
import socket
import sys
import tempfile
import threading
import time
import unittest

from PIL import Image, ImageDraw

E2E_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(E2E_DIR, "..", ".."))
for _p in (E2E_DIR, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import helpers  # noqa: E402
from yogabook_eink.protocol import (  # noqa: E402
    IMAGE_BUF_BASE,
    PANEL_HEIGHT,
    PANEL_WIDTH,
    WAVEFORM_DU2,
    WAVEFORM_GC16,
    WAVEFORM_GL16,
    WAVEFORM_INIT,
    Ite8951Controller,
)


class TestT3CrossFeaturePairwise(unittest.TestCase):
    def test_t3_01_tone_lut_x_5_mode_dithering(self) -> None:
        from yogabook_eink.image_render import DITHER_MODES, prepare_image_file

        with tempfile.TemporaryDirectory(prefix="yb_t3_01_") as tmp:
            png = helpers.make_gradient_png(os.path.join(tmp, "grad.png"), 64, 36)
            base_g16, _ = prepare_image_file(png, dither_mode="Grayscale 16-level", lightness=0, contrast=0)
            base_mean = sum(base_g16.getdata()) / (PANEL_WIDTH * PANEL_HEIGHT)

            for mode in DITHER_MODES:
                img, packed = prepare_image_file(png, dither_mode=mode, lightness=2, contrast=3)
                self.assertEqual(img.mode, "L")
                self.assertEqual(img.size, (PANEL_WIDTH, PANEL_HEIGHT))
                self.assertEqual(len(packed), PANEL_WIDTH * PANEL_HEIGHT)
                self.assertTrue(all((b & 0x0F) == 0x0F for b in packed[::997]))
                if mode == "Grayscale 16-level":
                    helpers.assert_16_gray_image(self, img, mode)
                    mean_bright = sum(img.getdata()) / (PANEL_WIDTH * PANEL_HEIGHT)
                    self.assertGreater(mean_bright, base_mean)
                else:
                    helpers.assert_strict_1bit_image(self, img, mode)

            # Endpoint compression under C=-1, L=-2
            comp_img, _ = prepare_image_file(png, dither_mode="Grayscale 16-level", lightness=-2, contrast=-1)
            helpers.assert_16_gray_image(self, comp_img)
            self.assertGreater(min(comp_img.getdata()), 0)
            self.assertLess(max(comp_img.getdata()), 255)

    def test_t3_02_tone_lut_x_invert_x_rotate_180(self) -> None:
        from yogabook_eink.image_render import pack_eink_grayscale, prepare_image_file

        with tempfile.TemporaryDirectory(prefix="yb_t3_02_") as tmp:
            png = helpers.make_asymmetric_png(os.path.join(tmp, "asym.png"), 80, 60)
            img_plain, _ = prepare_image_file(
                png, fit_mode="cover", invert=True, rotate_180=True, dither_mode="4x4 Bayer", lightness=0, contrast=0
            )
            img_toned, packed_toned = prepare_image_file(
                png, fit_mode="cover", invert=True, rotate_180=True, dither_mode="4x4 Bayer", lightness=2, contrast=1
            )
            helpers.assert_strict_1bit_image(self, img_toned)
            self.assertNotEqual(img_toned.tobytes(), img_plain.tobytes())
            self.assertEqual(packed_toned, pack_eink_grayscale(img_toned))

            # Verify 180-degree rotation flipped dark/bright halves vertically
            img_unrot, _ = prepare_image_file(
                png, fit_mode="cover", invert=True, rotate_180=False, dither_mode="4x4 Bayer", lightness=2, contrast=1
            )
            top_mean_unrot = sum(img_unrot.crop((0, 0, PANEL_WIDTH, PANEL_HEIGHT // 2)).getdata())
            bot_mean_toned = sum(img_toned.crop((0, PANEL_HEIGHT // 2, PANEL_WIDTH, PANEL_HEIGHT)).getdata())
            self.assertAlmostEqual(top_mean_unrot, bot_mean_toned, delta=50000)

    def test_t3_03_config_x_cli_image_options(self) -> None:
        from yogabook_eink.config import EinkSettings
        from yogabook_eink.image_render import prepare_image_file

        with tempfile.TemporaryDirectory(prefix="yb_t3_03_") as tmp:
            cfg_path = os.path.join(tmp, ".config", "yogabook-eink", "settings.json")
            EinkSettings(lightness=-2, contrast=4, dither_mode="Atkinson").save(path=cfg_path)
            png = helpers.make_gradient_png(os.path.join(tmp, "g.png"), 48, 32)

            res_a = helpers.run_cli_isolated(
                ["image", png, "--dither-mode", "floyd-steinberg", "--lightness", "3", "--contrast", "2"],
                home_dir=tmp,
            )
            self.assertEqual(res_a.returncode, 0, res_a.stderr)
            _, exp_a = prepare_image_file(png, dither_mode="Floyd-Steinberg", lightness=3, contrast=2)
            self.assertEqual(res_a.framebuffer, exp_a)
            self.assertEqual(res_a.dpy_area_ops[-1]["waveform"], WAVEFORM_GC16)

            res_b = helpers.run_cli_isolated(["image", png, "--dither"], home_dir=tmp)
            self.assertEqual(res_b.returncode, 0, res_b.stderr)
            _, exp_b = prepare_image_file(png, dither_mode="Floyd-Steinberg")
            self.assertEqual(res_b.framebuffer, exp_b)

            res_c = helpers.run_cli_isolated(["image", png, "--dither-mode", "bogus"], home_dir=tmp)
            self.assertNotEqual(res_c.returncode, 0)
            self.assertEqual(len(res_c.dpy_area_ops), 0)

    def test_t3_04_config_clamping_x_tone_lut_x_compositor_settle(self) -> None:
        import json
        from yogabook_eink.compositor import EinkCompositor
        from yogabook_eink.config import EinkSettings
        from yogabook_eink.image_render import DITHER_MODES
        from yogabook_eink.tone_lut import build_tone_lut

        with tempfile.TemporaryDirectory(prefix="yb_t3_04_") as tmp:
            p = os.path.join(tmp, "settings.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "lightness": 99,
                        "contrast": -99,
                        "dither_mode": "invalid",
                        "idle_settle_sec": -2.0,
                    },
                    f,
                )
            s = EinkSettings.load(path=p)
            self.assertEqual(s.lightness, 3)
            self.assertEqual(s.contrast, -1)
            self.assertIn(s.dither_mode, DITHER_MODES)
            self.assertGreaterEqual(s.idle_settle_sec, 0.0)

            oracle = helpers.get_glider_c_oracle()
            self.assertEqual(build_tone_lut(s.lightness, s.contrast), oracle(3, -1))

            rec = helpers.UsbBotRecorder()
            ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
            ctrl.open()
            comp = EinkCompositor(
                ctrl,
                max_fps=1000.0,
                idle_settle_enabled=s.idle_settle_enabled,
                idle_settle_sec=max(0.05, s.idle_settle_sec),
            )
            try:
                comp.reset_framebuffer(255)
                f = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
                ImageDraw.Draw(f).rectangle([(16, 16), (40, 40)], fill=0)
                comp.present(f, policy="fast")
                self.assertEqual(rec.dpy_area_ops[0].waveform, WAVEFORM_DU2)
            finally:
                comp.close()
                ctrl.close()

    def test_t3_05_compositor_8px_alignment_x_usb_protocol(self) -> None:
        from yogabook_eink.compositor import EinkCompositor

        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        comp = EinkCompositor(ctrl, max_fps=1000.0, idle_settle_enabled=False)
        try:
            comp.reset_framebuffer(255)
            f1 = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
            ImageDraw.Draw(f1).rectangle([(13, 19), (17, 25)], fill=0)
            comp.present(f1, policy="ui")

            f2 = f1.copy()
            ImageDraw.Draw(f2).rectangle([(1915, 1075), (1919, 1079)], fill=0)
            comp.present(f2, policy="ui")

            f3 = f2.copy()
            d3 = ImageDraw.Draw(f3)
            d3.rectangle([(3, 5), (12, 14)], fill=0)
            d3.rectangle([(1501, 899), (1509, 907)], fill=0)
            comp.present(f3, policy="ui")

            helpers.assert_r5_protocol_invariants(self, rec, allow_standalone_0xa9=False)
            for dpy in rec.dpy_area_ops:
                helpers.assert_8px_aligned(self, dpy.x, dpy.y, dpy.width, dpy.height)
            self.assertEqual(len(rec.dpy_area_ops), 4)
            # Every 0xA8 offset matches (y + dy) * 1920 + x
            for ld, dpy in zip(rec.ld_img_ops, rec.dpy_area_ops):
                self.assertEqual(ld.buf_offset, dpy.y * PANEL_WIDTH + dpy.x)
        finally:
            comp.close()
            ctrl.close()

    def test_t3_06_semantic_policies_fast_ui_partial_framing(self) -> None:
        from yogabook_eink.compositor import EinkCompositor

        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        comp = EinkCompositor(ctrl, max_fps=1000.0, idle_settle_enabled=False)
        try:
            base = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
            comp.present(base, policy="partial")  # initial GC16
            self.assertEqual(rec.dpy_area_ops[-1].waveform, WAVEFORM_GC16)

            f_gray = base.copy()
            ImageDraw.Draw(f_gray).rectangle([(32, 32), (80, 80)], fill=102)
            comp.present(f_gray, policy="partial")
            self.assertEqual(rec.dpy_area_ops[-1].waveform, WAVEFORM_GL16)
            self.assertIn(0x6F, rec.ld_img_ops[-1].payload)

            f_ui = f_gray.copy()
            ImageDraw.Draw(f_ui).rectangle([(200, 200), (240, 240)], fill=100)
            comp.present(f_ui, policy="ui")
            self.assertEqual(rec.dpy_area_ops[-1].waveform, WAVEFORM_DU2)
            self.assertTrue(set(rec.ld_img_ops[-1].payload).issubset({0x0F, 0xFF}))

            f_fast = f_ui.copy()
            ImageDraw.Draw(f_fast).rectangle([(400, 400), (440, 440)], fill=100)
            comp.present(f_fast, policy="fast")
            self.assertEqual(rec.dpy_area_ops[-1].waveform, WAVEFORM_DU2)
            self.assertTrue(set(rec.ld_img_ops[-1].payload).issubset({0x0F, 0xFF}))
        finally:
            comp.close()
            ctrl.close()

    def test_t3_07_full_and_clean_cancel_pending_idle_settle(self) -> None:
        from yogabook_eink.compositor import EinkCompositor

        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        comp = EinkCompositor(ctrl, max_fps=1000.0, idle_settle_enabled=True, idle_settle_sec=0.12)
        try:
            comp.reset_framebuffer(255)
            f = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
            ImageDraw.Draw(f).rectangle([(16, 16), (48, 48)], fill=0)
            comp.present(f, policy="fast")
            time.sleep(0.04)
            comp.present(f, policy="full")
            self.assertEqual(rec.dpy_area_ops[-1].waveform, WAVEFORM_GC16)
            time.sleep(0.25)
            self.assertEqual(len(rec.dpy_area_ops), 2)

            ImageDraw.Draw(f).rectangle([(64, 64), (96, 96)], fill=0)
            comp.present(f, policy="fast")
            time.sleep(0.04)
            comp.present(f, policy="clean")
            self.assertEqual(rec.dpy_area_ops[-1].waveform, WAVEFORM_INIT)
            time.sleep(0.25)
            self.assertEqual(len(rec.dpy_area_ops), 4)
        finally:
            comp.close()
            ctrl.close()

    def test_t3_08_idle_settle_x_gc16_budget_x_usb_lock_serialization(self) -> None:
        from yogabook_eink.compositor import EinkCompositor

        usb_lock = threading.RLock()
        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        comp = EinkCompositor(
            ctrl, gc16_budget=5, max_fps=1000.0, idle_settle_enabled=True, idle_settle_sec=0.10, usb_lock=usb_lock
        )
        try:
            comp.reset_framebuffer(255)
            base = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
            # Case A: 2 fast frames + wait -> 1 settle GC16 resets burst counter -> next 3 stay DU2
            for i in range(2):
                f = base.copy()
                ImageDraw.Draw(f).rectangle([(16 + i * 24, 16), (31 + i * 24, 31)], fill=0)
                comp.present(f, policy="fast")
            self.assertTrue(
                helpers.wait_until(
                    lambda: len([d for d in rec.dpy_area_ops if d.waveform == WAVEFORM_GC16]) == 1,
                    timeout=1.0,
                )
            )
            self.assertEqual(comp.partial_count, 0)
            before_next3 = len(rec.dpy_area_ops)
            for i in range(3):
                f = base.copy()
                ImageDraw.Draw(f).rectangle([(100 + i * 24, 16), (115 + i * 24, 31)], fill=0)
                comp.present(f, policy="fast")
            self.assertTrue(all(d.waveform == WAVEFORM_DU2 for d in rec.dpy_area_ops[before_next3:]))

            # Case B: hitting gc16_budget (5) fires GC16 and cancels duplicate settle
            for i in range(2):
                f_b = base.copy()
                ImageDraw.Draw(f_b).rectangle([(200 + i * 24, 300), (215 + i * 24, 315)], fill=0)
                comp.present(f_b, policy="fast")
            f5 = base.copy()
            ImageDraw.Draw(f5).rectangle([(340, 300), (360, 320)], fill=0)
            comp.present(f5, policy="fast")  # budget exhaustion -> GC16
            self.assertEqual(rec.dpy_area_ops[-1].waveform, WAVEFORM_GC16)
            count_at_budget = len(rec.dpy_area_ops)
            time.sleep(0.22)
            self.assertEqual(len(rec.dpy_area_ops), count_at_budget)
            self.assertFalse(rec.interleaved_transfer_detected)
        finally:
            comp.close()
            ctrl.close()

    def test_t3_09_doom_tone_lut_x_compositor_fast_x_timer_cleanup(self) -> None:
        from yogabook_eink.config import EinkSettings
        from yogabook_eink.doom_eink import HUD_TOP, EinkDoomGame, run_eink_doom

        with helpers.IsolatedEinkEnvironment() as env:
            EinkSettings(lightness=2, contrast=3, idle_settle_enabled=True, idle_settle_sec=0.10).save()
            init_sig = inspect.signature(EinkDoomGame.__init__)
            g_toned = EinkDoomGame(
                **{k: v for k, v in (("lightness", 2), ("contrast", 3)) if k in init_sig.parameters}
            )
            rf_sig = inspect.signature(g_toned.render_frame)
            f_toned, _ = g_toned.render_frame(
                **{k: v for k, v in (("lightness", 2), ("contrast", 3)) if k in rf_sig.parameters}
            )

            EinkSettings(lightness=0, contrast=0).save()
            g_neutral = EinkDoomGame(
                **{k: v for k, v in (("lightness", 0), ("contrast", 0)) if k in init_sig.parameters}
            )
            f_neutral, _ = g_neutral.render_frame(
                **{k: v for k, v in (("lightness", 0), ("contrast", 0)) if k in rf_sig.parameters}
            )

            self.assertTrue(
                f_toned.crop((0, 0, PANEL_WIDTH, HUD_TOP)).tobytes()
                != f_neutral.crop((0, 0, PANEL_WIDTH, HUD_TOP)).tobytes(),
                "DOOM 3D viewport unchanged under Tone LUT",
            )
            self.assertEqual(
                f_toned.crop((0, HUD_TOP, PANEL_WIDTH, PANEL_HEIGHT)).tobytes(),
                f_neutral.crop((0, HUD_TOP, PANEL_WIDTH, PANEL_HEIGHT)).tobytes(),
            )

            EinkSettings(lightness=2, contrast=3, idle_settle_enabled=True, idle_settle_sec=0.10).save()
            stop_ev = threading.Event()
            t = threading.Thread(target=run_eink_doom, args=(stop_ev,), daemon=True)
            t.start()
            try:
                self.assertTrue(helpers.wait_until(lambda: len(env.usb_recorder.dpy_area_ops) >= 2, timeout=3.0))
                env.inject_touch(275, 890, is_down=True)
                time.sleep(0.10)
                env.inject_touch(1500, 950, is_down=True)
                time.sleep(0.05)
                env.inject_touch(1785, 950, is_down=True)
                t.join(timeout=3.0)
                self.assertFalse(t.is_alive())
                blits_at_exit = len(env.usb_recorder.dpy_area_ops)
                time.sleep(0.18)
                self.assertEqual(len(env.usb_recorder.dpy_area_ops), blits_at_exit)
                for dpy in env.usb_recorder.dpy_area_ops[2:]:
                    helpers.assert_8px_aligned(self, dpy.x, dpy.y, dpy.width, dpy.height)
            finally:
                stop_ev.set()
                t.join(timeout=2.0)

    def test_t3_10_eink_menu_1bit_x_config_x_daemon_partial_refresh(self) -> None:
        from yogabook_eink.config import EinkSettings
        from yogabook_eink.daemon import EinkSettingsDaemon

        with helpers.IsolatedEinkEnvironment() as env:
            EinkSettings(lightness=-3, contrast=6).save()
            daemon = EinkSettingsDaemon()
            daemon.listener.start()
            try:
                daemon.screen.page = "settings"
                img, _ = daemon.screen.render()
                helpers.assert_strict_1bit_image(self, img)

                l_widgets = [
                    w for w in daemon.screen.widgets
                    if not w.is_card and not w.is_header_btn and "lightness" in (w.action_id + w.title).lower()
                ]
                c_widgets = [
                    w for w in daemon.screen.widgets
                    if not w.is_card and not w.is_header_btn and "contrast" in (w.action_id + w.title).lower()
                ]
                self.assertGreater(len(l_widgets), 0)
                self.assertGreater(len(c_widgets), 0)
                lw = next((w for w in l_widgets if any(t in (w.action_id + w.title).lower() for t in ("inc", "plus", "up", "+"))), l_widgets[0])
                cw = next((w for w in c_widgets if any(t in (w.action_id + w.title).lower() for t in ("dec", "minus", "down", "-"))), c_widgets[0])

                daemon._menu_active = True
                daemon._handle_action(lw.action_id)
                daemon._handle_action(cw.action_id)
                daemon._handle_action("toggle_key_down")
                saved = env.read_settings_json()
                self.assertIs(saved["key_down_sound"], True)
                self.assertTrue(any(d.waveform == WAVEFORM_DU2 for d in env.usb_recorder.dpy_area_ops))
                helpers.assert_r5_protocol_invariants(self, env.usb_recorder, allow_standalone_0xa9=True)

                # Pad mode bottom-center exit
                daemon._handle_action("ruled")
                daemon._last_tap_ts = 0.0
                before_kb = env.usb_recorder.exit_kb_count
                env.inject_touch(960, 1000, is_down=True)
                self.assertTrue(helpers.wait_until(lambda: not daemon._pad_active, timeout=2.0))
                self.assertEqual(env.usb_recorder.exit_kb_count, before_kb + 1)
            finally:
                daemon.sound.stop()
                daemon.listener.stop()
                daemon.ownership.stop()

    def test_t3_11_gui_sliders_dither_x_config_x_preview_send(self) -> None:
        from yogabook_eink.config import EinkSettings
        from yogabook_eink.image_render import prepare_image_file

        with helpers.IsolatedEinkEnvironment() as env:
            EinkSettings(lightness=-1, contrast=2, dither_mode="8x8 Bayer", idle_settle_enabled=False).save()
            png = helpers.make_photo_sample_png(os.path.join(env.tmpdir, "photo.png"), 64, 48)
            win, close_fn = helpers.open_isolated_window(env)
            try:
                scl_l = helpers.find_gtk_scale(win, "lightness", -3, 3)
                scl_c = helpers.find_gtk_scale(win, "contrast", -1, 6)
                cmb_d = helpers.find_dither_combo(win)
                tgl_s = helpers.find_idle_settle_toggle(win)

                self.assertEqual(int(round(scl_l.get_value())), -1)
                self.assertEqual(int(round(scl_c.get_value())), 2)
                self.assertIn("8x8", (cmb_d.get_active_text() or cmb_d.get_active_id() or ""))
                self.assertIs(tgl_s.get_active(), False)

                win._selected_image_path = png
                scl_l.set_value(2)
                scl_c.set_value(4)
                helpers.select_dither_mode_in_combo(cmb_d, "Atkinson")
                helpers.pump_gtk_events()

                saved = env.read_settings_json()
                self.assertEqual((saved["lightness"], saved["contrast"], saved["dither_mode"]), (2, 4, "Atkinson"))

                before = len(env.usb_recorder.dpy_area_ops)
                helpers.find_gtk_button(win, "Send Image to E-Ink Display").clicked()
                self.assertTrue(
                    helpers.wait_until(lambda: len(env.usb_recorder.dpy_area_ops) > before and not win._busy, timeout=45.0)
                )
                _, exp = prepare_image_file(png, dither_mode="Atkinson", lightness=2, contrast=4)
                self.assertEqual(bytes(env.usb_recorder.framebuffer), exp)
            finally:
                close_fn()

    def test_t3_12_matrix_1bit_untoned_x_fast_policy_x_exit(self) -> None:
        from yogabook_eink.config import EinkSettings
        from yogabook_eink.matrix_saver import run_matrix_screensaver

        with helpers.IsolatedEinkEnvironment() as env:
            EinkSettings(lightness=3, contrast=5).save()
            for dark in (True, False):
                before_dpy = len(env.usb_recorder.dpy_area_ops)
                stop_ev = threading.Event()
                t = threading.Thread(
                    target=run_matrix_screensaver,
                    kwargs={"stop_event": stop_ev, "dark_bg": dark},
                    daemon=True,
                )
                t.start()
                try:
                    self.assertTrue(
                        helpers.wait_until(
                            lambda b=before_dpy: len(env.usb_recorder.dpy_area_ops) >= b + 2
                            and any(l.is_running for l in env.listeners),
                            timeout=3.0,
                        )
                    )
                    env.inject_touch(960, 1010, is_down=True)
                    t.join(timeout=3.0)
                    self.assertFalse(t.is_alive())
                    for ld in env.usb_recorder.ld_img_ops:
                        self.assertTrue(set(ld.payload).issubset({0x0F, 0xFF}))
                finally:
                    stop_ev.set()
                    t.join(timeout=2.0)

    def test_t3_13_daemon_syncs_external_settings_update(self) -> None:
        from yogabook_eink.config import EinkSettings
        from yogabook_eink.daemon import EinkSettingsDaemon

        with helpers.IsolatedEinkEnvironment() as env:
            EinkSettings(dark_theme=False, lightness=0, contrast=0, waveform=WAVEFORM_GC16).save()
            daemon = EinkSettingsDaemon()
            daemon.listener.start()
            try:
                EinkSettings(dark_theme=True, lightness=2, contrast=3, waveform=WAVEFORM_GL16).save()
                env.inject_touch(1866, 18, is_down=True)
                self.assertTrue(helpers.wait_until(lambda: daemon._menu_active and not daemon._busy, timeout=2.0))
                self.assertIs(daemon.settings.dark_theme, True)
                self.assertEqual(daemon.settings.lightness, 2)
                self.assertEqual(daemon.settings.contrast, 3)
                self.assertEqual(env.usb_recorder.dpy_area_ops[-1].waveform, WAVEFORM_GL16)
            finally:
                daemon.sound.stop()
                daemon.listener.stop()
                daemon.ownership.stop()

    def test_t3_14_ownership_lease_clean_vs_unclean_recovery(self) -> None:
        from yogabook_eink.ownership import OwnershipServer, PanelOwnershipLease, get_socket_path

        with helpers.IsolatedEinkEnvironment() as env:
            srv = OwnershipServer()
            helpers.start_ownership_server(srv)
            try:
                # (a) Clean release: no extra exit_to_keyboard from OwnershipServer
                with PanelOwnershipLease("doom") as lease:
                    self.assertTrue(helpers.wait_until(lambda: srv.active_owner == "doom", timeout=2.0))
                    lease.mark_clean_exit()
                self.assertTrue(helpers.wait_until(lambda: srv.active_owner is None, timeout=2.0))
                self.assertEqual(env.usb_recorder.exit_kb_count, 0)

                # (b) Unclean EOF drop -> OwnershipServer invokes exit_to_keyboard()
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(get_socket_path())
                sock.sendall(b"ACQUIRE doom\n")
                self.assertEqual(sock.recv(64).decode().strip(), "OK")
                self.assertTrue(helpers.wait_until(lambda: srv.active_owner == "doom", timeout=2.0))
                sock.close()
                self.assertTrue(
                    helpers.wait_until(
                        lambda: srv.active_owner is None and env.usb_recorder.exit_kb_count >= 1,
                        timeout=2.0,
                    )
                )
            finally:
                srv.stop()


if __name__ == "__main__":
    unittest.main()
