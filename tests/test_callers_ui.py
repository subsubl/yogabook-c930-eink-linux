"""Concise M3 caller & UI integration tests (isolated, no hardware USB or real config touched)."""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image

from yogabook_eink import config as _config_mod
from yogabook_eink.config import EinkSettings
from yogabook_eink.daemon import EinkSettingsDaemon
from yogabook_eink.doom_eink import HUD_TOP, EinkDoomGame
from yogabook_eink.eink_menu import EinkSettingsScreen
from yogabook_eink.gui import _prepare_preview_pil
from yogabook_eink.matrix_saver import MatrixRainRenderer
from yogabook_eink.protocol import PANEL_HEIGHT, PANEL_WIDTH


class TestM3CallersConcise(unittest.TestCase):
    def test_doom_tone_lut_and_strict_1bit_hud(self) -> None:
        g0 = EinkDoomGame(lightness=0, contrast=0)
        g_hi = EinkDoomGame(lightness=3, contrast=4)
        g_oob = EinkDoomGame(lightness=99, contrast=99)
        g_max = EinkDoomGame(lightness=3, contrast=6)

        f0, _ = g0.render_frame()
        fhi, _ = g_hi.render_frame()
        foob, _ = g_oob.render_frame()
        fmax, _ = g_max.render_frame()

        self.assertEqual(f0.size, (PANEL_WIDTH, PANEL_HEIGHT))
        self.assertEqual(set(fhi.getdata()), {0, 255})
        self.assertNotEqual(
            f0.crop((0, 0, PANEL_WIDTH, HUD_TOP)).tobytes(),
            fhi.crop((0, 0, PANEL_WIDTH, HUD_TOP)).tobytes(),
        )
        self.assertEqual(
            f0.crop((0, HUD_TOP, PANEL_WIDTH, PANEL_HEIGHT)).tobytes(),
            fhi.crop((0, HUD_TOP, PANEL_WIDTH, PANEL_HEIGHT)).tobytes(),
        )
        self.assertEqual(foob.tobytes(), fmax.tobytes())

    def test_matrix_strict_1bit_and_exit_badge(self) -> None:
        r = MatrixRainRenderer(dark_bg=True)
        for _ in range(5):
            frame = r.step_frame(cols_per_tick=4)
            self.assertEqual(set(frame.getdata()), {0, 255})
        self.assertIn(255, set(r.canvas.getdata()))

    def test_eink_menu_8px_patches_and_lightness_contrast_widgets(self) -> None:
        s = EinkSettings(lightness=1, contrast=2)
        screen = EinkSettingsScreen(s)
        screen.page = "settings"
        img, _ = screen.render("Test")
        self.assertEqual(set(img.getdata()), {0, 255})

        aids = {w.action_id for w in screen.widgets}
        self.assertIn("cycle_lightness", aids)
        self.assertIn("cycle_contrast", aids)

        for w in screen.widgets:
            if not w.is_card and not w.is_header_btn:
                patches = screen.render_widget_patches(w.action_id, "OK")
                self.assertTrue(patches)
                for px, py, pw, ph, data in patches:
                    self.assertEqual((px % 8, py % 8, pw % 8, ph % 8), (0, 0, 0, 0))
                    self.assertEqual(len(data), pw * ph)

    def test_daemon_reload_settings_and_tone_cycle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="yb_m3_unit_") as tmp:
            cfg_path = os.path.join(tmp, "settings.json")
            EinkSettings(lightness=0, contrast=0, dither_mode="Atkinson").save(cfg_path)
            with (
                patch.object(_config_mod, "CONFIG_DIR", tmp),
                patch.object(_config_mod, "CONFIG_PATH", cfg_path),
                patch("yogabook_eink.daemon.Ite8951Controller"),
                patch("yogabook_eink.daemon.EinkHidListener"),
                patch("yogabook_eink.daemon.OwnershipServer"),
            ):
                daemon = EinkSettingsDaemon()
                EinkSettings(lightness=2, contrast=4, dither_mode="Atkinson", idle_settle_sec=0.5).save(cfg_path)
                daemon._handle_action("cycle_lightness")
                reloaded = EinkSettings.load(cfg_path)
                self.assertEqual(reloaded.lightness, 3)
                self.assertEqual(reloaded.contrast, 4)
                self.assertEqual(reloaded.dither_mode, "Atkinson")
                self.assertAlmostEqual(reloaded.idle_settle_sec, 0.5)
                daemon.sound.stop()

    def test_gui_thumbnail_preview_pipeline(self) -> None:
        with tempfile.TemporaryDirectory(prefix="yb_m3_prev_") as tmp:
            png = os.path.join(tmp, "g.png")
            im = Image.new("L", (64, 36))
            im.putdata([int(x * 255 / 63) for _ in range(36) for x in range(64)])
            im.save(png)
            p0 = _prepare_preview_pil(png, dither_mode="Atkinson", lightness=0, contrast=0)
            p1 = _prepare_preview_pil(png, dither_mode="Atkinson", lightness=-2, contrast=3)
            self.assertEqual(p0.size, (440, 248))
            self.assertEqual(set(p0.getdata()), {0, 255})
            self.assertNotEqual(p0.tobytes(), p1.tobytes())


if __name__ == "__main__":
    unittest.main()
