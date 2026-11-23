"""Tier 4 E2E Real-World Application Scenarios (7 tests: T4-01..T4-07)."""

from __future__ import annotations

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
    PANEL_HEIGHT,
    PANEL_WIDTH,
    WAVEFORM_DU2,
    WAVEFORM_GC16,
    WAVEFORM_GL16,
    WAVEFORM_INIT,
    Ite8951Controller,
)


class TestT4RealWorldScenarios(unittest.TestCase):
    def test_t4_01_photo_workflow_gui_send_cli_override_gui_reopen(self) -> None:
        from yogabook_eink.image_render import prepare_image_file

        with helpers.IsolatedEinkEnvironment() as env:
            png = helpers.make_photo_sample_png(os.path.join(env.tmpdir, "photo.png"), 64, 48)
            win, close_fn = helpers.open_isolated_window(env)
            try:
                win._selected_image_path = png
                scl_l = helpers.find_gtk_scale(win, "lightness", -3, 3)
                scl_c = helpers.find_gtk_scale(win, "contrast", -1, 6)
                cmb_d = helpers.find_dither_combo(win)

                scl_l.set_value(2)
                scl_c.set_value(2)
                helpers.select_dither_mode_in_combo(cmb_d, "8x8 Bayer")
                win.cmb_fit.set_active_id("cover")
                win.cmb_wf.set_active_id(str(WAVEFORM_GC16))
                helpers.pump_gtk_events()

                before = len(env.usb_recorder.dpy_area_ops)
                helpers.find_gtk_button(win, "Send Image to E-Ink Display").clicked()
                self.assertTrue(
                    helpers.wait_until(lambda: len(env.usb_recorder.dpy_area_ops) > before and not win._busy, timeout=3.0)
                )
                helpers.assert_r5_protocol_invariants(self, env.usb_recorder, allow_standalone_0xa9=True)
                self.assertEqual(env.usb_recorder.dpy_area_ops[-1].waveform, WAVEFORM_GC16)
                saved = env.read_settings_json()
                self.assertEqual((saved["lightness"], saved["contrast"], saved["dither_mode"]), (2, 2, "8x8 Bayer"))
            finally:
                close_fn()

            # Step 2: CLI overrides with Atkinson + (-1, 3) + GL16 without mutating GUI persisted defaults
            res = helpers.run_cli_isolated(
                [
                    "image",
                    png,
                    "--dither-mode",
                    "atkinson",
                    "--lightness",
                    "-1",
                    "--contrast",
                    "3",
                    "--waveform",
                    "gl16",
                ],
                home_dir=env.home_dir,
                runtime_dir=env.runtime_dir,
            )
            self.assertEqual(res.returncode, 0, res.stderr)
            helpers.assert_cli_r5_invariants(self, res)
            _, exp_cli = prepare_image_file(png, dither_mode="Atkinson", lightness=-1, contrast=3)
            self.assertEqual(res.framebuffer, exp_cli)
            self.assertEqual(res.dpy_area_ops[-1]["waveform"], WAVEFORM_GL16)

            # Step 3: Re-open GUI -> restores persisted (+2, +2, "8x8 Bayer")
            win2, close_fn2 = helpers.open_isolated_window(env)
            try:
                scl_l2 = helpers.find_gtk_scale(win2, "lightness", -3, 3)
                scl_c2 = helpers.find_gtk_scale(win2, "contrast", -1, 6)
                cmb_d2 = helpers.find_dither_combo(win2)
                self.assertEqual(int(round(scl_l2.get_value())), 2)
                self.assertEqual(int(round(scl_c2.get_value())), 2)
                self.assertIn("8x8", (cmb_d2.get_active_text() or cmb_d2.get_active_id() or ""))
            finally:
                close_fn2()

    def test_t4_02_on_panel_dashboard_settings_notepad_bottom_exit(self) -> None:
        from yogabook_eink.daemon import EinkSettingsDaemon

        with helpers.IsolatedEinkEnvironment() as env:
            daemon = EinkSettingsDaemon()
            daemon.listener.start()
            try:
                # 1. Tap top-right gear -> Dashboard
                before_dpy = len(env.usb_recorder.dpy_area_ops)
                env.inject_touch(1866, 18, is_down=True)
                self.assertTrue(
                    helpers.wait_until(
                        lambda b=before_dpy: len(env.usb_recorder.dpy_area_ops) > b
                        and daemon._menu_active
                        and daemon.screen.page == "dashboard"
                        and any(w.action_id == "page_settings" for w in daemon.screen.widgets)
                        and not daemon._busy,
                        timeout=2.0,
                    )
                )
                self.assertEqual(daemon.screen.page, "dashboard")
                helpers.assert_strict_1bit_image(self, daemon.screen._last_img)

                # 2. Tap SETTINGS card
                settings_card = [w for w in daemon.screen.widgets if w.action_id == "page_settings"][0]
                daemon._last_tap_ts = 0.0
                before_dpy = len(env.usb_recorder.dpy_area_ops)
                env.inject_touch(settings_card.x + 50, settings_card.y + 50, is_down=True)
                self.assertTrue(
                    helpers.wait_until(
                        lambda b=before_dpy: len(env.usb_recorder.dpy_area_ops) > b
                        and daemon.screen.page == "settings"
                        and any(w.action_id == "toggle_key_down" for w in daemon.screen.widgets)
                        and not daemon._busy,
                        timeout=2.0,
                    )
                )

                # 3. Tap Lightness, Contrast, and Key-Down Sound controls
                l_w = [
                    w for w in daemon.screen.widgets
                    if not w.is_card and not w.is_header_btn and "lightness" in (w.action_id + w.title).lower()
                ]
                c_w = [
                    w for w in daemon.screen.widgets
                    if not w.is_card and not w.is_header_btn and "contrast" in (w.action_id + w.title).lower()
                ]
                self.assertGreater(len(l_w), 0, "Missing Lightness control")
                self.assertGreater(len(c_w), 0, "Missing Contrast control")
                lw = next((w for w in l_w if any(t in (w.action_id + w.title).lower() for t in ("inc", "plus", "up", "+"))), l_w[0])
                cw = next((w for w in c_w if any(t in (w.action_id + w.title).lower() for t in ("inc", "plus", "up", "+"))), c_w[0])
                kd = [w for w in daemon.screen.widgets if w.action_id == "toggle_key_down"][0]

                for target_w in (lw, cw, kd):
                    daemon._last_tap_ts = 0.0
                    before_dpy = len(env.usb_recorder.dpy_area_ops)
                    env.inject_touch(target_w.x + target_w.w // 2, target_w.y + target_w.h // 2, is_down=True)
                    self.assertTrue(
                        helpers.wait_until(
                            lambda b=before_dpy: len(env.usb_recorder.dpy_area_ops) > b and not daemon._busy,
                            timeout=2.0,
                        )
                    )
                    self.assertEqual(env.usb_recorder.dpy_area_ops[-1].waveform, WAVEFORM_DU2)

                saved = env.read_settings_json()
                self.assertIs(saved["key_down_sound"], True)

                # 4. Back to Dashboard -> Ruled Notes -> Bottom-center exit to KB
                dash_w = [w for w in daemon.screen.widgets if w.action_id == "page_dashboard"][0]
                daemon._last_tap_ts = 0.0
                before_dpy = len(env.usb_recorder.dpy_area_ops)
                env.inject_touch(dash_w.x + 50, dash_w.y + 50, is_down=True)
                self.assertTrue(
                    helpers.wait_until(
                        lambda b=before_dpy: len(env.usb_recorder.dpy_area_ops) > b
                        and daemon.screen.page == "dashboard"
                        and any(w.action_id == "ruled" for w in daemon.screen.widgets)
                        and not daemon._busy,
                        timeout=2.0,
                    )
                )

                ruled_w = [w for w in daemon.screen.widgets if w.action_id == "ruled"][0]
                daemon._last_tap_ts = 0.0
                env.inject_touch(ruled_w.x + 50, ruled_w.y + 50, is_down=True)
                self.assertTrue(helpers.wait_until(lambda: daemon._pad_active and not daemon._busy, timeout=2.0))

                before_kb = env.usb_recorder.exit_kb_count
                daemon._last_tap_ts = 0.0
                env.inject_touch(960, 990, is_down=True)
                self.assertTrue(helpers.wait_until(lambda: not daemon._pad_active and not daemon._busy, timeout=2.0))
                self.assertEqual(env.usb_recorder.exit_kb_count, before_kb + 1)
            finally:
                daemon.sound.stop()
                daemon.listener.stop()
                daemon.ownership.stop()

    def test_t4_03_full_doom_gameplay_tone_lease_settle_exit(self) -> None:
        from yogabook_eink.config import EinkSettings
        from yogabook_eink.doom_eink import run_eink_doom
        from yogabook_eink.ownership import OwnershipServer

        with helpers.IsolatedEinkEnvironment() as env:
            EinkSettings(lightness=1, contrast=2, idle_settle_enabled=True, idle_settle_sec=0.14).save()
            srv = OwnershipServer()
            helpers.start_ownership_server(srv)
            stop_ev = threading.Event()
            t = threading.Thread(target=run_eink_doom, args=(stop_ev,), daemon=True)
            t.start()
            try:
                self.assertTrue(helpers.wait_until(lambda: len(env.usb_recorder.dpy_area_ops) >= 2, timeout=3.0))
                self.assertEqual(srv.active_owner, "doom")

                env.inject_touch(275, 890, is_down=True)  # fwd
                time.sleep(0.12)
                env.inject_touch(445, 955, is_down=True)  # right
                time.sleep(0.12)
                env.inject_touch(1500, 950, is_down=True)  # fire

                # Wait for idle GC16 settle pass
                self.assertTrue(
                    helpers.wait_until(
                        lambda: any(d.waveform == WAVEFORM_GC16 for d in env.usb_recorder.dpy_area_ops),
                        timeout=2.0,
                    )
                )

                # Tap EXIT button after idle GC16 settle -> exits DOOM and restores KB cleanly
                env.inject_touch(1785, 950, is_down=True)
                t.join(timeout=3.0)
                self.assertFalse(t.is_alive())
                self.assertGreaterEqual(env.usb_recorder.exit_kb_count, 1)
                self.assertTrue(helpers.wait_until(lambda: srv.active_owner is None, timeout=2.0))
                helpers.assert_r5_protocol_invariants(self, env.usb_recorder, allow_standalone_0xa9=True)
            finally:
                stop_ev.set()
                t.join(timeout=2.0)
                srv.stop()

    def test_t4_04_matrix_burst_settle_cancel_exit_and_eof_recovery(self) -> None:
        from yogabook_eink.config import EinkSettings
        from yogabook_eink.matrix_saver import run_matrix_screensaver
        from yogabook_eink.ownership import OwnershipServer, get_socket_path

        with helpers.IsolatedEinkEnvironment() as env:
            EinkSettings(lightness=3, contrast=6, idle_settle_enabled=True, idle_settle_sec=0.14).save()
            srv = OwnershipServer()
            helpers.start_ownership_server(srv)
            stop_ev = threading.Event()
            frames = []
            t = threading.Thread(
                target=run_matrix_screensaver,
                kwargs={"stop_event": stop_ev, "dark_bg": True, "on_frame": lambda n: frames.append(n)},
                daemon=True,
            )
            t.start()
            try:
                self.assertTrue(helpers.wait_until(lambda: len(frames) >= 4, timeout=3.0))
                env.inject_touch(960, 1010, is_down=True)
                t.join(timeout=3.0)
                self.assertFalse(t.is_alive())
                for ld in env.usb_recorder.ld_img_ops:
                    self.assertTrue(set(ld.payload).issubset({0x0F, 0xFF}))
                self.assertTrue(helpers.wait_until(lambda: srv.active_owner is None, timeout=2.0))

                # Abrupt socket EOF auto-restores keyboard
                before_kb = env.usb_recorder.exit_kb_count
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(get_socket_path())
                sock.sendall(b"ACQUIRE matrix\n")
                self.assertEqual(sock.recv(64).decode().strip(), "OK")
                sock.close()
                self.assertTrue(
                    helpers.wait_until(
                        lambda: srv.active_owner is None and env.usb_recorder.exit_kb_count > before_kb,
                        timeout=2.0,
                    )
                )
            finally:
                stop_ev.set()
                t.join(timeout=2.0)
                srv.stop()

    def test_t4_05_gui_launch_and_stop_doom_and_matrix_no_deadlock_or_leaks(self) -> None:
        from yogabook_eink.ownership import OwnershipServer

        def _worker_or_timer_threads() -> list[tuple[int, str]]:
            out = []
            for th in threading.enumerate():
                if not th.is_alive():
                    continue
                name = th.name or ""
                if isinstance(th, threading.Timer) or name.startswith("eink-") or "Timer" in type(th).__name__:
                    out.append((th.ident or 0, f"{type(th).__name__}:{name}"))
            return out

        with helpers.IsolatedEinkEnvironment() as env:
            before_threads = {t[0] for t in _worker_or_timer_threads()}
            srv = OwnershipServer()
            close_fn = None
            try:
                helpers.start_ownership_server(srv)
                win, close_fn = helpers.open_isolated_window(env)
                # 1. Start & stop DOOM via GUI toggle
                win.btn_doom.set_active(True)
                self.assertTrue(helpers.wait_until(lambda: srv.active_owner == "doom", timeout=3.0))
                self.assertTrue(helpers.wait_until(lambda: len(env.usb_recorder.dpy_area_ops) >= 2, timeout=3.0))
                win.btn_doom.set_active(False)
                self.assertTrue(helpers.wait_until(lambda: srv.active_owner is None, timeout=3.0))

                # 2. Start & stop Matrix via GUI toggle
                before = len(env.usb_recorder.dpy_area_ops)
                win.btn_matrix.set_active(True)
                self.assertTrue(helpers.wait_until(lambda: srv.active_owner == "matrix", timeout=3.0))
                self.assertTrue(helpers.wait_until(lambda: len(env.usb_recorder.dpy_area_ops) > before + 1, timeout=3.0))
                win.btn_matrix.set_active(False)
                self.assertTrue(helpers.wait_until(lambda: srv.active_owner is None, timeout=3.0))
                helpers.assert_r5_protocol_invariants(self, env.usb_recorder, allow_standalone_0xa9=True)
            finally:
                if close_fn is not None:
                    close_fn()
                srv.stop()

            self.assertTrue(
                helpers.wait_until(
                    lambda: all(t[0] in before_threads for t in _worker_or_timer_threads()),
                    timeout=0.5,
                ),
                f"Leaked Timer/eink worker threads: {[t for t in _worker_or_timer_threads() if t[0] not in before_threads]}",
            )

    def test_t4_06_multi_phase_semantic_refresh_pipeline(self) -> None:
        from yogabook_eink.compositor import EinkCompositor

        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        comp = EinkCompositor(ctrl, max_fps=1000.0, idle_settle_enabled=True, idle_settle_sec=0.10)
        try:
            # Phase 1: UI 1-bit partial update
            comp.reset_framebuffer(255)
            base = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
            ImageDraw.Draw(base).rectangle([(16, 16), (64, 48)], fill=0)
            comp.present(base, policy="ui")
            self.assertEqual(rec.dpy_area_ops[-1].waveform, WAVEFORM_DU2)

            # Phase 2: Partial 16-level grayscale at unaligned (17, 23, 101, 45) -> 8px aligned GL16
            f_gray = base.copy()
            ImageDraw.Draw(f_gray).rectangle([(17, 23), (117, 67)], fill=128)
            comp.present(f_gray, policy="partial")
            d_gray = rec.dpy_area_ops[-1]
            self.assertEqual(d_gray.waveform, WAVEFORM_GL16)
            helpers.assert_8px_aligned(self, d_gray.x, d_gray.y, d_gray.width, d_gray.height)
            self.assertIn(0x8F, rec.ld_img_ops[-1].payload)

            # Phase 3: 4 rapid fast updates -> DU2, then idle -> single GC16 settle
            f_fast = f_gray.copy()
            for i in range(4):
                ImageDraw.Draw(f_fast).rectangle([(300 + i * 24, 300), (315 + i * 24, 315)], fill=0)
                comp.present(f_fast, policy="fast")
            self.assertTrue(
                helpers.wait_until(
                    lambda: any(d.waveform == WAVEFORM_GC16 for d in rec.dpy_area_ops),
                    timeout=1.0,
                )
            )

            # Phase 4: Clean -> full INIT
            comp.present(base, policy="clean")
            self.assertEqual(rec.dpy_area_ops[-1].waveform, WAVEFORM_INIT)
            helpers.assert_r5_protocol_invariants(self, rec, allow_standalone_0xa9=False)
        finally:
            comp.close()
            ctrl.close()

    def test_t4_07_usb_timeout_fault_drain_clear_image_recovery(self) -> None:
        from yogabook_eink.image_render import prepare_image_file

        with tempfile.TemporaryDirectory(prefix="yb_t4_07_") as tmp:
            png = helpers.make_gradient_png(os.path.join(tmp, "g.png"), 48, 32)
            # 1. Fault on 0x94 -> non-zero clean exit
            res_err = helpers.run_cli_isolated(["image", png], home_dir=tmp, fail_opcode=0x94)
            self.assertEqual(res_err.returncode, 1)
            self.assertIn("USB Error:", res_err.stderr)

            # 2. Drain -> Clear -> Image recovery
            res_drain = helpers.run_cli_isolated(["drain"], home_dir=tmp)
            self.assertEqual(res_drain.returncode, 0, res_drain.stderr)

            res_clear = helpers.run_cli_isolated(["clear"], home_dir=tmp)
            self.assertEqual(res_clear.returncode, 0, res_clear.stderr)
            self.assertEqual(res_clear.dpy_area_ops[-1]["waveform"], WAVEFORM_INIT)

            res_img = helpers.run_cli_isolated(
                ["image", png, "--dither-mode", "floyd-steinberg"],
                home_dir=tmp,
            )
            self.assertEqual(res_img.returncode, 0, res_img.stderr)
            helpers.assert_cli_r5_invariants(self, res_img)
            _, exp = prepare_image_file(png, dither_mode="Floyd-Steinberg")
            self.assertEqual(res_img.framebuffer, exp)


if __name__ == "__main__":
    unittest.main()
