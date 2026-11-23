"""Tier 1 E2E Feature Tests (63 tests across Features #1-#12)."""

from __future__ import annotations

import inspect
import os
import socket
import subprocess
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
    OP_DPY_AREA,
    OP_GET_DPY_STATUS,
    OP_LD_IMG_AREA2,
    OP_SET_WAVEFORM,
    PANEL_HEIGHT,
    PANEL_WIDTH,
    WAVEFORM_DU2,
    WAVEFORM_GC16,
    WAVEFORM_GL16,
    WAVEFORM_INIT,
    Ite8951Controller,
)


def _find_menu_widgets(screen, keyword: str):
    needle = keyword.lower()
    matches = [
        w
        for w in screen.widgets
        if not w.is_card
        and not w.is_header_btn
        and (needle in w.action_id.lower() or needle in w.title.lower())
    ]
    return matches


def _pick_step_widget(matches):
    for w in matches:
        aid = w.action_id.lower()
        ttl = w.title.lower()
        if any(tok in aid or tok in ttl for tok in ("inc", "plus", "up", "+")):
            return w
    return matches[0]


# ===========================================================================
# Feature #1: Glider 256-byte Parabolic Tone LUT (6 tests)
# ===========================================================================


class TestT1Feature01ToneLut(unittest.TestCase):
    def test_t1_f1_01_identity_0_0(self) -> None:
        from yogabook_eink.tone_lut import build_tone_lut

        lut = build_tone_lut(0, 0)
        self.assertIsInstance(lut, (bytes, bytearray))
        self.assertEqual(len(lut), 256)
        self.assertEqual(bytes(lut), bytes(range(256)))

    def test_t1_f1_02_full_56_pair_c_oracle_parity(self) -> None:
        from yogabook_eink.tone_lut import build_tone_lut

        c_oracle = helpers.get_glider_c_oracle()
        for lightness in range(-3, 4):
            for contrast in range(-1, 7):
                expected = c_oracle(lightness, contrast)
                actual = bytes(build_tone_lut(lightness, contrast))
                self.assertEqual(
                    actual,
                    expected,
                    f"Tone LUT mismatch at lightness={lightness}, contrast={contrast}",
                )

    def test_t1_f1_03_lightness_curve_monotonic_shape(self) -> None:
        from yogabook_eink.tone_lut import build_tone_lut

        mids = []
        for lightness in range(-3, 4):
            lut = build_tone_lut(lightness, 0)
            self.assertEqual(lut[0], 0)
            self.assertEqual(lut[255], 255)
            mids.append(lut[128])
            if lightness < 0:
                self.assertLess(lut[128], 128)
            elif lightness > 0:
                self.assertGreater(lut[128], 128)
        self.assertEqual(mids, sorted(mids))
        self.assertTrue(all(a < b for a, b in zip(mids, mids[1:])))

    def test_t1_f1_04_contrast_expansion_s_curve(self) -> None:
        from yogabook_eink.tone_lut import build_tone_lut

        lut = build_tone_lut(0, 3)
        self.assertEqual(lut[0], 0)
        self.assertEqual(lut[255], 255)
        self.assertLess(lut[96], 96)
        self.assertGreater(lut[160], 160)

    def test_t1_f1_05_apply_tone_lut_on_L_ramp(self) -> None:
        from yogabook_eink.tone_lut import apply_tone_lut, build_tone_lut

        ramp = helpers.make_gradient_image(256, 4)
        expected_lut = build_tone_lut(2, 3)
        toned = apply_tone_lut(ramp, lightness=2, contrast=3)
        self.assertEqual(toned.mode, "L")
        self.assertEqual(toned.size, (256, 4))
        row = toned.tobytes()[:256]
        self.assertEqual(row, expected_lut)

        ident = apply_tone_lut(ramp, lightness=0, contrast=0)
        self.assertEqual(ident.tobytes(), ramp.tobytes())

    def test_t1_f1_06_apply_tone_lut_converts_non_L_modes(self) -> None:
        from yogabook_eink.tone_lut import apply_tone_lut, build_tone_lut

        rgb = Image.new("RGB", (32, 16), color=(128, 128, 128))
        toned_rgb = apply_tone_lut(rgb, lightness=2, contrast=1)
        self.assertEqual(toned_rgb.mode, "L")
        self.assertEqual(toned_rgb.size, (32, 16))
        expected_val = build_tone_lut(2, 1)[128]
        self.assertEqual(set(toned_rgb.getdata()), {expected_val})

        bw = Image.new("1", (16, 8), color=1)
        toned_bw = apply_tone_lut(bw, lightness=-1, contrast=0)
        self.assertEqual(toned_bw.mode, "L")
        self.assertEqual(toned_bw.size, (16, 8))
        self.assertEqual(set(toned_bw.getdata()), {255})


# ===========================================================================
# Feature #2: Settings Persistence (5 tests)
# ===========================================================================


class TestT1Feature02ConfigPersistence(unittest.TestCase):
    def test_t1_f2_01_default_settings_fields(self) -> None:
        from yogabook_eink.config import EinkSettings

        s = EinkSettings()
        self.assertEqual(s.lightness, 0)
        self.assertEqual(s.contrast, 0)
        self.assertEqual(s.dither_mode, "Grayscale 16-level")
        self.assertIs(s.idle_settle_enabled, True)
        self.assertAlmostEqual(s.idle_settle_sec, 2.0, places=3)
        self.assertIs(s.key_down_sound, False)
        self.assertIs(s.key_up_sound, False)
        self.assertEqual(s.sound_profile, "soft")
        self.assertIs(s.dark_theme, False)
        self.assertEqual(s.waveform, WAVEFORM_GC16)
        self.assertIs(s.rotate_180, False)
        self.assertIs(s.swap_axes, False)
        self.assertIs(s.flip_x, False)
        self.assertIs(s.flip_y, False)

    def test_t1_f2_02_save_and_load_roundtrip(self) -> None:
        from yogabook_eink.config import EinkSettings

        with tempfile.TemporaryDirectory(prefix="yb_t1_f2_") as tmp:
            path = os.path.join(tmp, "settings.json")
            orig = EinkSettings(
                key_down_sound=True,
                key_up_sound=True,
                sound_profile="typewriter",
                dark_theme=True,
                waveform=WAVEFORM_GL16,
                rotate_180=True,
                swap_axes=True,
                flip_x=True,
                flip_y=True,
                lightness=2,
                contrast=4,
                dither_mode="Atkinson",
                idle_settle_enabled=False,
                idle_settle_sec=1.25,
            )
            orig.save(path=path)
            loaded = EinkSettings.load(path=path)
            self.assertEqual(loaded, orig)

    def test_t1_f2_03_atomic_save_creates_parents_no_tmp_left(self) -> None:
        from yogabook_eink.config import EinkSettings

        with tempfile.TemporaryDirectory(prefix="yb_t1_f2_atom_") as tmp:
            nested = os.path.join(tmp, "nested", "dir", "settings.json")
            s = EinkSettings(lightness=-2, contrast=3)
            s.save(path=nested)
            self.assertTrue(os.path.isfile(nested))
            self.assertFalse(os.path.exists(nested + ".tmp"))
            self.assertEqual(EinkSettings.load(path=nested).lightness, -2)

    def test_t1_f2_04_subprocess_home_isolation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="yb_t1_f2_sub_") as tmp:
            env = dict(os.environ)
            env["HOME"] = tmp
            env["XDG_CONFIG_HOME"] = os.path.join(tmp, ".config")
            env["PYTHONPATH"] = REPO_ROOT
            code = (
                "from yogabook_eink import config\n"
                "s = config.EinkSettings(lightness=3, contrast=5)\n"
                "s.save()\n"
                "print(config.CONFIG_PATH)\n"
            )
            res = subprocess.run(
                [sys.executable, "-c", code],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            expected = os.path.join(tmp, ".config", "yogabook-eink", "settings.json")
            self.assertEqual(res.stdout.strip(), expected)
            self.assertTrue(os.path.isfile(expected))

    def test_t1_f2_05_legacy_9_field_json_populates_defaults(self) -> None:
        import json
        from yogabook_eink.config import EinkSettings

        with tempfile.TemporaryDirectory(prefix="yb_t1_f2_leg_") as tmp:
            path = os.path.join(tmp, "settings.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "key_down_sound": True,
                        "key_up_sound": False,
                        "sound_profile": "crisp",
                        "dark_theme": True,
                        "waveform": WAVEFORM_DU2,
                        "rotate_180": False,
                        "swap_axes": True,
                        "flip_x": False,
                        "flip_y": True,
                    },
                    f,
                )
            loaded = EinkSettings.load(path=path)
            self.assertIs(loaded.key_down_sound, True)
            self.assertEqual(loaded.sound_profile, "crisp")
            self.assertEqual(loaded.lightness, 0)
            self.assertEqual(loaded.contrast, 0)
            self.assertEqual(loaded.dither_mode, "Grayscale 16-level")
            self.assertIs(loaded.idle_settle_enabled, True)
            self.assertAlmostEqual(loaded.idle_settle_sec, 2.0, places=3)


# ===========================================================================
# Feature #3: epdoptimize 5-mode Dithering (6 tests)
# ===========================================================================


class TestT1Feature03Dithering(unittest.TestCase):
    def test_t1_f3_01_dither_modes_and_aliases(self) -> None:
        from yogabook_eink.image_render import (
            DITHER_MODE_ALIASES,
            DITHER_MODES,
            normalize_dither_mode,
        )

        expected_modes = (
            "Grayscale 16-level",
            "4x4 Bayer",
            "8x8 Bayer",
            "Floyd-Steinberg",
            "Atkinson",
        )
        self.assertEqual(tuple(DITHER_MODES), expected_modes)
        for m in expected_modes:
            self.assertEqual(normalize_dither_mode(m), m)
            self.assertEqual(normalize_dither_mode(m.lower()), m)

        for slug in (
            "grayscale-16",
            "16-level",
            "grayscale",
            "bayer-4x4",
            "4x4",
            "bayer-8x8",
            "8x8",
            "floyd-steinberg",
            "fs",
            "atkinson",
        ):
            self.assertIn(slug, DITHER_MODE_ALIASES)
            self.assertIn(normalize_dither_mode(slug), expected_modes)

    def test_t1_f3_02_grayscale_16_level_ramp(self) -> None:
        from yogabook_eink.image_render import dither_image

        ramp = helpers.make_gradient_image(256, 8)
        out = dither_image(ramp, mode="Grayscale 16-level")
        self.assertEqual(out.mode, "L")
        self.assertEqual(out.size, (256, 8))
        helpers.assert_16_gray_image(self, out)
        self.assertEqual(set(out.getdata()), {k * 17 for k in range(16)})

    def test_t1_f3_03_bayer_4x4_and_8x8_periodic_and_levels(self) -> None:
        from yogabook_eink.image_render import dither_image

        mid = Image.new("L", (32, 32), color=128)
        for mode, period in (("4x4 Bayer", 4), ("8x8 Bayer", 8)):
            out2 = dither_image(mid, mode=mode)
            helpers.assert_strict_1bit_image(self, out2, mode)
            self.assertEqual(set(out2.getdata()), {0, 255})
            px = out2.load()
            for y in range(32):
                for x in range(32):
                    self.assertEqual(px[x, y], px[x % period, y % period])

            out16 = dither_image(helpers.make_gradient_image(64, 16), mode=mode, levels=16)
            helpers.assert_16_gray_image(self, out16, f"{mode} levels=16")

    def test_t1_f3_04_error_diffusion_fs_and_atkinson(self) -> None:
        from yogabook_eink.image_render import dither_image

        grad = helpers.make_gradient_image(64, 32)
        src_mean = sum(grad.getdata()) / (64 * 32)
        for mode in ("Floyd-Steinberg", "Atkinson"):
            out2 = dither_image(grad, mode=mode)
            helpers.assert_strict_1bit_image(self, out2, mode)
            mean2 = sum(out2.getdata()) / (64 * 32)
            self.assertAlmostEqual(mean2, src_mean, delta=25.0)

            out_serp = dither_image(grad, mode=mode, serpentine=True)
            helpers.assert_strict_1bit_image(self, out_serp, f"{mode} serpentine")

            out16 = dither_image(grad, mode=mode, levels=16)
            helpers.assert_16_gray_image(self, out16, f"{mode} levels=16")

    def test_t1_f3_05_prepare_image_file_all_5_modes(self) -> None:
        from yogabook_eink.image_render import DITHER_MODES, prepare_image_file

        with tempfile.TemporaryDirectory(prefix="yb_t1_f3_") as tmp:
            png = helpers.make_gradient_png(os.path.join(tmp, "grad.png"), 64, 36)
            for mode in DITHER_MODES:
                img, packed = prepare_image_file(png, dither_mode=mode)
                self.assertEqual(img.mode, "L")
                self.assertEqual(img.size, (PANEL_WIDTH, PANEL_HEIGHT))
                self.assertEqual(len(packed), PANEL_WIDTH * PANEL_HEIGHT)
                raw = img.tobytes()
                self.assertEqual(packed[0], (raw[0] & 0xF0) | 0x0F)
                self.assertEqual(packed[-1], (raw[-1] & 0xF0) | 0x0F)
                if mode == "Grayscale 16-level":
                    helpers.assert_16_gray_image(self, img, mode)
                else:
                    helpers.assert_strict_1bit_image(self, img, mode)

    def test_t1_f3_06_prepare_image_file_tone_lut_before_dither(self) -> None:
        from yogabook_eink.image_render import prepare_image_file

        with tempfile.TemporaryDirectory(prefix="yb_t1_f3_lut_") as tmp:
            png = helpers.make_gradient_png(os.path.join(tmp, "grad.png"), 64, 36)
            base_img, _ = prepare_image_file(png, dither_mode="Grayscale 16-level", lightness=0, contrast=0)
            bright_img, _ = prepare_image_file(png, dither_mode="Grayscale 16-level", lightness=3, contrast=0)
            high_c_img, _ = prepare_image_file(png, dither_mode="Grayscale 16-level", lightness=0, contrast=6)

            base_mean = sum(base_img.getdata()) / (PANEL_WIDTH * PANEL_HEIGHT)
            bright_mean = sum(bright_img.getdata()) / (PANEL_WIDTH * PANEL_HEIGHT)
            self.assertGreater(bright_mean, base_mean)
            self.assertTrue(set(high_c_img.getdata()).issubset({0, 255}))


# ===========================================================================
# Feature #4: CLI Dithering & Tone Options (5 tests)
# ===========================================================================


class TestT1Feature04CliOptions(unittest.TestCase):
    def test_t1_f4_01_cli_image_all_5_dither_modes(self) -> None:
        from yogabook_eink.image_render import DITHER_MODES

        with tempfile.TemporaryDirectory(prefix="yb_t1_f4_") as tmp:
            png = helpers.make_gradient_png(os.path.join(tmp, "sample.png"), 48, 32)
            for mode in DITHER_MODES:
                res = helpers.run_cli_isolated(
                    ["image", png, "--dither-mode", mode],
                    home_dir=tmp,
                )
                self.assertEqual(res.returncode, 0, f"mode={mode} stderr={res.stderr}")
                helpers.assert_cli_r5_invariants(self, res)

    def test_t1_f4_02_cli_wire_payload_and_waveforms(self) -> None:
        from yogabook_eink.image_render import prepare_image_file

        with tempfile.TemporaryDirectory(prefix="yb_t1_f4_wf_") as tmp:
            png = helpers.make_gradient_png(os.path.join(tmp, "sample.png"), 48, 32)
            for wf_flag, wf_id in (("gc16", WAVEFORM_GC16), ("du2", WAVEFORM_DU2), ("gl16", WAVEFORM_GL16)):
                res = helpers.run_cli_isolated(
                    ["image", png, "--dither-mode", "bayer-8x8", "--waveform", wf_flag],
                    home_dir=tmp,
                )
                self.assertEqual(res.returncode, 0, res.stderr)
                _, expected_packed = prepare_image_file(png, dither_mode="8x8 Bayer")
                self.assertEqual(res.framebuffer, expected_packed)
                self.assertEqual(res.dpy_area_ops[-1]["waveform"], wf_id)

    def test_t1_f4_03_cli_lightness_and_contrast_wire_parity(self) -> None:
        from yogabook_eink.image_render import prepare_image_file

        with tempfile.TemporaryDirectory(prefix="yb_t1_f4_lc_") as tmp:
            png = helpers.make_gradient_png(os.path.join(tmp, "sample.png"), 48, 32)
            res = helpers.run_cli_isolated(
                ["image", png, "--lightness", "2", "--contrast", "3", "--dither-mode", "atkinson"],
                home_dir=tmp,
            )
            self.assertEqual(res.returncode, 0, res.stderr)
            _, expected = prepare_image_file(png, dither_mode="Atkinson", lightness=2, contrast=3)
            self.assertEqual(res.framebuffer, expected)

    def test_t1_f4_04_legacy_dither_flag_matches_floyd_steinberg(self) -> None:
        with tempfile.TemporaryDirectory(prefix="yb_t1_f4_leg_") as tmp:
            png = helpers.make_gradient_png(os.path.join(tmp, "sample.png"), 48, 32)
            res_old = helpers.run_cli_isolated(["image", png, "--dither"], home_dir=tmp)
            res_new = helpers.run_cli_isolated(["image", png, "--dither-mode", "floyd-steinberg"], home_dir=tmp)
            self.assertEqual(res_old.returncode, 0, res_old.stderr)
            self.assertEqual(res_new.returncode, 0, res_new.stderr)
            self.assertEqual(res_old.framebuffer, res_new.framebuffer)

    def test_t1_f4_05_cli_help_and_home_isolation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="yb_t1_f4_help_") as tmp:
            res = helpers.run_cli_isolated(["image", "--help"], home_dir=tmp)
            self.assertEqual(res.returncode, 0)
            for opt in ("--dither-mode", "--lightness", "--contrast", "--dither"):
                self.assertIn(opt, res.stdout)


# ===========================================================================
# Feature #5: KOReader 8px Aligned Partial Rects (5 tests)
# ===========================================================================


class TestT1Feature05AlignedRects(unittest.TestCase):
    def test_t1_f5_01_dither_alignment_constant_and_align_rect(self) -> None:
        from yogabook_eink.compositor import DITHER_ALIGNMENT, align_rect

        self.assertEqual(DITHER_ALIGNMENT, 8)
        self.assertEqual(align_rect(10, 12, 20, 14), (8, 8, 24, 24))

    def test_t1_f5_02_already_aligned_rect_unchanged(self) -> None:
        from yogabook_eink.compositor import align_rect

        self.assertEqual(align_rect(8, 16, 64, 32), (8, 16, 64, 32))

    def test_t1_f5_03_compute_dirty_rects_small_unaligned_diff(self) -> None:
        from yogabook_eink.compositor import compute_dirty_rects

        prev = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
        curr = prev.copy()
        ImageDraw.Draw(curr).rectangle([(101, 203), (113, 217)], fill=0)

        rects = compute_dirty_rects(prev, curr)
        self.assertEqual(len(rects), 1)
        r = rects[0]
        helpers.assert_8px_aligned(self, r.x, r.y, r.w, r.h)
        self.assertLessEqual(r.x, 101)
        self.assertLessEqual(r.y, 203)
        self.assertGreaterEqual(r.x + r.w, 114)
        self.assertGreaterEqual(r.y + r.h, 218)

    def test_t1_f5_04_compute_dirty_rects_two_distant_rects_stay_separate(self) -> None:
        from yogabook_eink.compositor import compute_dirty_rects

        prev = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
        curr = prev.copy()
        d = ImageDraw.Draw(curr)
        d.rectangle([(10, 10), (30, 30)], fill=0)
        d.rectangle([(1800, 1000), (1825, 1025)], fill=0)

        rects = compute_dirty_rects(prev, curr)
        self.assertEqual(len(rects), 2)
        for r in rects:
            helpers.assert_8px_aligned(self, r.x, r.y, r.w, r.h)

    def test_t1_f5_05_compute_dirty_rects_collapse_over_max_rects(self) -> None:
        from yogabook_eink.compositor import compute_dirty_rects

        prev = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
        curr = prev.copy()
        d = ImageDraw.Draw(curr)
        # Create > max_rects non-adjacent islands
        for gx in range(0, 1600, 200):
            for gy in range(0, 900, 200):
                d.rectangle([(gx + 3, gy + 5), (gx + 13, gy + 15)], fill=0)

        rects = compute_dirty_rects(prev, curr, max_rects=4)
        self.assertEqual(len(rects), 1)
        r = rects[0]
        helpers.assert_8px_aligned(self, r.x, r.y, r.w, r.h)


# ===========================================================================
# Feature #6: Semantic Refresh Policy (5 tests)
# ===========================================================================


class TestT1Feature06SemanticPolicy(unittest.TestCase):
    def test_t1_f6_01_policy_constants_and_all_5_present_succeed(self) -> None:
        from yogabook_eink.compositor import SEMANTIC_REFRESH_POLICIES, EinkCompositor

        self.assertEqual(
            tuple(SEMANTIC_REFRESH_POLICIES),
            ("fast", "ui", "partial", "full", "clean"),
        )
        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        comp = EinkCompositor(ctrl, max_fps=1000.0, idle_settle_enabled=False)
        try:
            comp.reset_framebuffer(255)
            f = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
            for idx, pol in enumerate(SEMANTIC_REFRESH_POLICIES):
                f2 = f.copy()
                ImageDraw.Draw(f2).rectangle([(16 + idx * 16, 16), (31 + idx * 16, 31)], fill=0)
                stats = comp.present(f2, policy=pol)
                self.assertIsNotNone(stats)
            helpers.assert_r5_protocol_invariants(self, rec, allow_standalone_0xa9=False)
        finally:
            comp.close()
            ctrl.close()

    def test_t1_f6_02_policy_fast_1bit_du2_and_arms_settle(self) -> None:
        from yogabook_eink.compositor import EinkCompositor

        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        comp = EinkCompositor(ctrl, max_fps=1000.0, idle_settle_enabled=True, idle_settle_sec=0.10)
        try:
            # Grayscale 128 should be thresholded to strict 1-bit under fast policy
            f = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=128)
            comp.present(f, policy="fast")
            self.assertEqual(rec.dpy_area_ops[-1].waveform, WAVEFORM_DU2)
            self.assertTrue(set(rec.framebuffer).issubset({0x0F, 0xFF}))
            # Wait for idle settle to confirm fast armed it
            deadline = time.monotonic() + 0.5
            while len(rec.dpy_area_ops) < 2 and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(len(rec.dpy_area_ops), 2)
            self.assertEqual(rec.dpy_area_ops[-1].waveform, WAVEFORM_GC16)
        finally:
            comp.close()
            ctrl.close()

    def test_t1_f6_03_policy_ui_1bit_aligned_du2(self) -> None:
        from yogabook_eink.compositor import EinkCompositor

        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        comp = EinkCompositor(ctrl, max_fps=1000.0, idle_settle_enabled=False)
        try:
            comp.reset_framebuffer(255)
            f = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
            ImageDraw.Draw(f).rectangle([(19, 21), (45, 51)], fill=100)  # will threshold to 0
            comp.present(f, policy="ui")
            dpy = rec.dpy_area_ops[-1]
            self.assertEqual(dpy.waveform, WAVEFORM_DU2)
            helpers.assert_8px_aligned(self, dpy.x, dpy.y, dpy.width, dpy.height)
            self.assertTrue(set(rec.ld_img_ops[-1].payload).issubset({0x0F, 0xFF}))
        finally:
            comp.close()
            ctrl.close()

    def test_t1_f6_04_policy_partial_preserves_grayscale_and_gl16(self) -> None:
        from yogabook_eink.compositor import EinkCompositor

        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        comp = EinkCompositor(ctrl, max_fps=1000.0, idle_settle_enabled=False)
        try:
            f0 = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
            comp.present(f0, policy="partial")
            self.assertEqual(rec.dpy_area_ops[-1].waveform, WAVEFORM_GC16)

            f1 = f0.copy()
            ImageDraw.Draw(f1).rectangle([(32, 32), (95, 95)], fill=128)
            comp.present(f1, policy="partial")
            dpy = rec.dpy_area_ops[-1]
            self.assertEqual(dpy.waveform, WAVEFORM_GL16)
            helpers.assert_8px_aligned(self, dpy.x, dpy.y, dpy.width, dpy.height)
            self.assertIn(0x8F, rec.ld_img_ops[-1].payload)
        finally:
            comp.close()
            ctrl.close()

    def test_t1_f6_05_policy_full_and_clean_waveforms_and_cancel_settle(self) -> None:
        from yogabook_eink.compositor import EinkCompositor

        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        comp = EinkCompositor(ctrl, max_fps=1000.0, idle_settle_enabled=True, idle_settle_sec=0.10)
        try:
            comp.reset_framebuffer(255)
            f = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
            ImageDraw.Draw(f).rectangle([(16, 16), (48, 48)], fill=0)
            comp.present(f, policy="fast")
            comp.present(f, policy="full")
            self.assertEqual(rec.dpy_area_ops[-1].waveform, WAVEFORM_GC16)
            self.assertEqual((rec.dpy_area_ops[-1].width, rec.dpy_area_ops[-1].height), (PANEL_WIDTH, PANEL_HEIGHT))

            time.sleep(0.20)
            count_after_full = len(rec.dpy_area_ops)
            self.assertEqual(count_after_full, 2)

            ImageDraw.Draw(f).rectangle([(80, 80), (112, 112)], fill=0)
            comp.present(f, policy="fast")
            comp.present(f, policy="clean")
            self.assertEqual(rec.dpy_area_ops[-1].waveform, WAVEFORM_INIT)
            self.assertEqual((rec.dpy_area_ops[-1].width, rec.dpy_area_ops[-1].height), (PANEL_WIDTH, PANEL_HEIGHT))
            time.sleep(0.20)
            self.assertEqual(len(rec.dpy_area_ops), 4)
        finally:
            comp.close()
            ctrl.close()


# ===========================================================================
# Feature #7: Idle Anti-Ghosting Settle Refresh (5 tests)
# ===========================================================================


class TestT1Feature07IdleSettle(unittest.TestCase):
    def test_t1_f7_01_idle_gc16_settle_fires_after_du2_burst(self) -> None:
        from yogabook_eink.compositor import EinkCompositor

        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        comp = EinkCompositor(ctrl, max_fps=1000.0, idle_settle_enabled=True, idle_settle_sec=0.12)
        try:
            comp.reset_framebuffer(255)
            f = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
            ImageDraw.Draw(f).rectangle([(24, 24), (64, 64)], fill=0)
            comp.present(f, policy="fast")
            self.assertEqual(len(rec.dpy_area_ops), 1)
            self.assertEqual(rec.dpy_area_ops[0].waveform, WAVEFORM_DU2)

            deadline = time.monotonic() + 0.6
            while len(rec.dpy_area_ops) < 2 and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(len(rec.dpy_area_ops), 2)
            settle = rec.dpy_area_ops[1]
            self.assertEqual(settle.waveform, WAVEFORM_GC16)
            self.assertEqual((settle.x, settle.y, settle.width, settle.height), (0, 0, PANEL_WIDTH, PANEL_HEIGHT))
        finally:
            comp.close()
            ctrl.close()

    def test_t1_f7_02_single_settle_only_over_3x_interval(self) -> None:
        from yogabook_eink.compositor import EinkCompositor

        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        comp = EinkCompositor(ctrl, max_fps=1000.0, idle_settle_enabled=True, idle_settle_sec=0.10)
        try:
            comp.reset_framebuffer(255)
            f = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
            ImageDraw.Draw(f).rectangle([(16, 16), (48, 48)], fill=0)
            comp.present(f, policy="fast")
            time.sleep(0.35)
            gc16_ops = [d for d in rec.dpy_area_ops if d.waveform == WAVEFORM_GC16]
            self.assertEqual(len(gc16_ops), 1)
        finally:
            comp.close()
            ctrl.close()

    def test_t1_f7_03_incoming_frames_reset_settle_timer(self) -> None:
        from yogabook_eink.compositor import EinkCompositor

        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        comp = EinkCompositor(ctrl, max_fps=1000.0, idle_settle_enabled=True, idle_settle_sec=0.15)
        try:
            comp.reset_framebuffer(255)
            base = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
            for i in range(4):
                f = base.copy()
                ImageDraw.Draw(f).rectangle([(16 + i * 32, 16), (31 + i * 32, 31)], fill=0)
                comp.present(f, policy="fast")
                # No GC16 should have fired yet between rapid frames
                gc16_ops = [
                    (d.waveform, d.x, d.y, d.width, d.height)
                    for d in rec.dpy_area_ops
                    if d.waveform == WAVEFORM_GC16
                ]
                self.assertEqual(len(gc16_ops), 0, f"Unexpected GC16 blits: {gc16_ops}")
                time.sleep(0.06)

            self.assertTrue(
                helpers.wait_until(
                    lambda: len([d for d in rec.dpy_area_ops if d.waveform == WAVEFORM_GC16]) == 1,
                    timeout=1.5,
                ),
                "Expected single idle GC16 settle after rapid frame burst quiesced",
            )
            gc16_ops = [d for d in rec.dpy_area_ops if d.waveform == WAVEFORM_GC16]
            self.assertEqual(len(gc16_ops), 1)
        finally:
            comp.close()
            ctrl.close()

    def test_t1_f7_04_full_or_clean_cancels_armed_settle(self) -> None:
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
            before = len(rec.dpy_area_ops)
            time.sleep(0.22)
            self.assertEqual(len(rec.dpy_area_ops), before)
        finally:
            comp.close()
            ctrl.close()

    def test_t1_f7_05_cancel_settle_and_close_prevent_blits(self) -> None:
        from yogabook_eink.compositor import EinkCompositor

        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        comp = EinkCompositor(ctrl, max_fps=1000.0, idle_settle_enabled=True, idle_settle_sec=0.10)
        try:
            comp.reset_framebuffer(255)
            f = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
            ImageDraw.Draw(f).rectangle([(16, 16), (48, 48)], fill=0)
            comp.present(f, policy="fast")
            comp.cancel_settle()
            time.sleep(0.20)
            self.assertEqual(len(rec.dpy_area_ops), 1)

            ImageDraw.Draw(f).rectangle([(64, 64), (96, 96)], fill=0)
            comp.present(f, policy="fast")
            comp.close()
            time.sleep(0.20)
            self.assertEqual(len(rec.dpy_area_ops), 2)
        finally:
            comp.close()
            ctrl.close()


# ===========================================================================
# Feature #8: USB Protocol & R5 Invariants (5 tests)
# ===========================================================================


class TestT1Feature08ProtocolR5(unittest.TestCase):
    def test_t1_f8_01_blit_uses_pre_post_b1_and_no_a9(self) -> None:
        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        try:
            ctrl.blit(0, 0, 0, PANEL_WIDTH, PANEL_HEIGHT, waveform=WAVEFORM_DU2)
            helpers.assert_r5_protocol_invariants(self, rec, allow_standalone_0xa9=False)
            self.assertEqual(rec.dpy_area_ops[0].waveform, WAVEFORM_DU2)
            opcodes = [c.opcode for c in rec.commands]
            self.assertEqual(opcodes, [OP_GET_DPY_STATUS, OP_DPY_AREA, OP_GET_DPY_STATUS])
        finally:
            ctrl.close()

    def test_t1_f8_02_display_region_ram_offset_and_mem_addr(self) -> None:
        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        try:
            patch = bytes([0x0F]) * (64 * 32)
            ctrl.display_region(80, 120, 64, 32, patch, waveform=WAVEFORM_DU2)
            helpers.assert_r5_protocol_invariants(self, rec, allow_standalone_0xa9=False)
            self.assertEqual(len(rec.ld_img_ops), 1)
            ld = rec.ld_img_ops[0]
            self.assertEqual(ld.addr, IMAGE_BUF_BASE + 120 * PANEL_WIDTH + 80)
            self.assertEqual(ld.payload, patch)
            dpy = rec.dpy_area_ops[0]
            self.assertEqual((dpy.mem_addr, dpy.waveform, dpy.x, dpy.y, dpy.width, dpy.height),
                             (IMAGE_BUF_BASE, WAVEFORM_DU2, 80, 120, 64, 32))
        finally:
            ctrl.close()

    def test_t1_f8_03_display_region_multi_chunk_offsets(self) -> None:
        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        try:
            # 640 * 200: rows_per_chunk = 61440 // 640 = 96 -> chunks at dy = 0, 96, 192
            patch = bytes((i % 256) for i in range(640 * 200))
            ctrl.display_region(160, 240, 640, 200, patch, waveform=WAVEFORM_GL16)
            helpers.assert_r5_protocol_invariants(self, rec, allow_standalone_0xa9=False)
            self.assertEqual(len(rec.ld_img_ops), 3)
            self.assertEqual([op.buf_offset for op in rec.ld_img_ops], [
                (240 + 0) * PANEL_WIDTH + 160,
                (240 + 96) * PANEL_WIDTH + 160,
                (240 + 192) * PANEL_WIDTH + 160,
            ])
            self.assertEqual(b"".join(op.payload for op in rec.ld_img_ops), patch)
            self.assertEqual(rec.dpy_area_ops[0].waveform, WAVEFORM_GL16)
        finally:
            ctrl.close()

    def test_t1_f8_04_display_image_bytes_34_chunks_gc16(self) -> None:
        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        try:
            buf = bytes([0x8F]) * (PANEL_WIDTH * PANEL_HEIGHT)
            ctrl.display_image_bytes(buf, PANEL_WIDTH, PANEL_HEIGHT, waveform=WAVEFORM_GC16)
            helpers.assert_r5_protocol_invariants(self, rec, allow_standalone_0xa9=False)
            self.assertEqual(len(rec.ld_img_ops), 34)
            self.assertEqual(rec.dpy_area_ops[0].waveform, WAVEFORM_GC16)
        finally:
            ctrl.close()

    def test_t1_f8_05_full_stack_compositor_all_policies_r5(self) -> None:
        from yogabook_eink.compositor import SEMANTIC_REFRESH_POLICIES, EinkCompositor

        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        comp = EinkCompositor(ctrl, max_fps=1000.0, idle_settle_enabled=True, idle_settle_sec=0.10)
        try:
            comp.reset_framebuffer(255)
            base = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
            for i, pol in enumerate(SEMANTIC_REFRESH_POLICIES):
                f = base.copy()
                ImageDraw.Draw(f).rectangle([(20 + i * 24, 20), (40 + i * 24, 44)], fill=64)
                comp.present(f, policy=pol)
            comp.present(base, policy="fast")
            time.sleep(0.18)
            helpers.assert_r5_protocol_invariants(self, rec, allow_standalone_0xa9=False)
        finally:
            comp.close()
            ctrl.close()


# ===========================================================================
# Feature #9: DOOM Tone LUT & Compositor Wiring (6 tests)
# ===========================================================================


class TestT1Feature09DoomWiring(unittest.TestCase):
    @staticmethod
    def _render_doom_with_tone(lightness: int, contrast: int):
        from yogabook_eink.config import EinkSettings
        from yogabook_eink.doom_eink import EinkDoomGame

        EinkSettings(lightness=lightness, contrast=contrast).save()
        init_sig = inspect.signature(EinkDoomGame.__init__)
        init_kwargs = {}
        if "lightness" in init_sig.parameters:
            init_kwargs["lightness"] = lightness
        if "contrast" in init_sig.parameters:
            init_kwargs["contrast"] = contrast
        game = EinkDoomGame(**init_kwargs)

        rf_sig = inspect.signature(game.render_frame)
        rf_kwargs = {}
        if "lightness" in rf_sig.parameters:
            rf_kwargs["lightness"] = lightness
        if "contrast" in rf_sig.parameters:
            rf_kwargs["contrast"] = contrast
        return game, game.render_frame(**rf_kwargs)

    def test_t1_f9_01_neutral_render_1bit_and_packed(self) -> None:
        with helpers.IsolatedEinkEnvironment():
            _game, (frame, packed) = self._render_doom_with_tone(0, 0)
            self.assertEqual(frame.size, (PANEL_WIDTH, PANEL_HEIGHT))
            helpers.assert_strict_1bit_image(self, frame)
            self.assertEqual(set(frame.getdata()), {0, 255})
            self.assertEqual(len(packed), PANEL_WIDTH * PANEL_HEIGHT)
            self.assertTrue(set(packed).issubset({0x0F, 0xFF}))

    def test_t1_f9_02_lightness_affects_viewport_not_hud(self) -> None:
        from yogabook_eink.doom_eink import HUD_TOP

        with helpers.IsolatedEinkEnvironment():
            _, (f_dark, _) = self._render_doom_with_tone(-3, 0)
            _, (f_mid, _) = self._render_doom_with_tone(0, 0)
            _, (f_bright, _) = self._render_doom_with_tone(3, 0)

            whites = []
            huds = []
            for f in (f_dark, f_mid, f_bright):
                vp = f.crop((0, 0, PANEL_WIDTH, HUD_TOP))
                hud = f.crop((0, HUD_TOP, PANEL_WIDTH, PANEL_HEIGHT))
                helpers.assert_strict_1bit_image(self, vp)
                helpers.assert_strict_1bit_image(self, hud)
                whites.append(sum(1 for p in vp.getdata() if p == 255))
                huds.append(hud.tobytes())

            self.assertLess(whites[0], whites[1])
            self.assertLess(whites[1], whites[2])
            self.assertEqual(huds[0], huds[1])
            self.assertEqual(huds[1], huds[2])

    def test_t1_f9_03_contrast_affects_viewport_not_hud(self) -> None:
        from yogabook_eink.doom_eink import HUD_TOP

        with helpers.IsolatedEinkEnvironment():
            _, (f_0, _) = self._render_doom_with_tone(0, 0)
            _, (f_3, _) = self._render_doom_with_tone(0, 3)
            _, (f_6, _) = self._render_doom_with_tone(0, 6)

            vp0 = f_0.crop((0, 0, PANEL_WIDTH, HUD_TOP)).tobytes()
            vp3 = f_3.crop((0, 0, PANEL_WIDTH, HUD_TOP)).tobytes()
            vp6 = f_6.crop((0, 0, PANEL_WIDTH, HUD_TOP)).tobytes()
            self.assertTrue(vp0 != vp3, "Viewport unchanged for contrast=+3")
            self.assertTrue(vp0 != vp6, "Viewport unchanged for contrast=+6")

            hud0 = f_0.crop((0, HUD_TOP, PANEL_WIDTH, PANEL_HEIGHT)).tobytes()
            hud3 = f_3.crop((0, HUD_TOP, PANEL_WIDTH, PANEL_HEIGHT)).tobytes()
            hud6 = f_6.crop((0, HUD_TOP, PANEL_WIDTH, PANEL_HEIGHT)).tobytes()
            self.assertEqual(hud0, hud3)
            self.assertEqual(hud0, hud6)

    def test_t1_f9_04_run_eink_doom_movement_fire_aligned_du2(self) -> None:
        from yogabook_eink.doom_eink import run_eink_doom
        from yogabook_eink.ownership import OwnershipServer

        with helpers.IsolatedEinkEnvironment() as env:
            srv = OwnershipServer()
            helpers.start_ownership_server(srv)
            stop_ev = threading.Event()
            t = threading.Thread(target=run_eink_doom, args=(stop_ev,), daemon=True)
            t.start()
            try:
                self.assertTrue(
                    helpers.wait_until(lambda: len(env.usb_recorder.dpy_area_ops) >= 2, timeout=3.0)
                )
                self.assertEqual(srv.active_owner, "doom")
                before = len(env.usb_recorder.dpy_area_ops)
                env.inject_touch(275, 890, is_down=True)  # fwd
                time.sleep(0.15)
                env.inject_touch(445, 955, is_down=True)  # right
                time.sleep(0.15)
                env.inject_touch(1500, 950, is_down=True)  # fire (muzzle flash redraw)
                self.assertTrue(
                    helpers.wait_until(lambda: len(env.usb_recorder.dpy_area_ops) > before + 1, timeout=3.0)
                )
                for dpy in env.usb_recorder.dpy_area_ops[2:]:
                    helpers.assert_8px_aligned(self, dpy.x, dpy.y, dpy.width, dpy.height)
                helpers.assert_r5_protocol_invariants(self, env.usb_recorder, allow_standalone_0xa9=True)
            finally:
                stop_ev.set()
                t.join(timeout=3.0)
                srv.stop()

    def test_t1_f9_05_doom_exit_button_clean_teardown(self) -> None:
        from yogabook_eink.doom_eink import run_eink_doom
        from yogabook_eink.ownership import OwnershipServer

        with helpers.IsolatedEinkEnvironment() as env:
            srv = OwnershipServer()
            helpers.start_ownership_server(srv)
            stop_ev = threading.Event()
            t = threading.Thread(target=run_eink_doom, args=(stop_ev,), daemon=True)
            t.start()
            try:
                self.assertTrue(
                    helpers.wait_until(lambda: len(env.usb_recorder.dpy_area_ops) >= 2, timeout=3.0)
                )
                env.inject_touch(1785, 950, is_down=True)  # EXIT
                t.join(timeout=3.0)
                self.assertFalse(t.is_alive())
                self.assertTrue(stop_ev.is_set())
                self.assertGreaterEqual(env.usb_recorder.exit_kb_count, 1)
                self.assertTrue(helpers.wait_until(lambda: srv.active_owner is None, timeout=2.0))
                self.assertTrue(all(not l.is_running for l in env.listeners))
            finally:
                stop_ev.set()
                t.join(timeout=2.0)
                srv.stop()

    def test_t1_f9_06_doom_gc16_budget_or_idle_settle(self) -> None:
        from yogabook_eink.config import EinkSettings
        from yogabook_eink.doom_eink import run_eink_doom

        with helpers.IsolatedEinkEnvironment() as env:
            EinkSettings(idle_settle_enabled=True, idle_settle_sec=0.15).save()
            stop_ev = threading.Event()
            t = threading.Thread(target=run_eink_doom, kwargs={"stop_event": stop_ev, "gc16_budget": 5}, daemon=True)
            t.start()
            try:
                self.assertTrue(
                    helpers.wait_until(lambda: len(env.usb_recorder.dpy_area_ops) >= 2, timeout=3.0)
                )
                for _ in range(6):
                    env.inject_touch(445, 955, is_down=True)
                    time.sleep(0.10)
                self.assertTrue(
                    helpers.wait_until(
                        lambda: any(d.waveform == WAVEFORM_GC16 for d in env.usb_recorder.dpy_area_ops),
                        timeout=3.0,
                    )
                )
                helpers.assert_r5_protocol_invariants(self, env.usb_recorder, allow_standalone_0xa9=True)
            finally:
                stop_ev.set()
                t.join(timeout=3.0)


# ===========================================================================
# Feature #10: E-Ink Settings Menu Controls & Daemon Sync (5 tests)
# ===========================================================================


class TestT1Feature10EinkMenuDaemon(unittest.TestCase):
    def test_t1_f10_01_dashboard_and_settings_strict_1bit_untoned(self) -> None:
        from yogabook_eink.config import EinkSettings
        from yogabook_eink.eink_menu import EinkSettingsScreen

        with helpers.IsolatedEinkEnvironment():
            for dark in (False, True):
                s = EinkSettings(lightness=3, contrast=6, dark_theme=dark)
                screen = EinkSettingsScreen(s)
                for page in ("dashboard", "settings"):
                    screen.page = page
                    img, packed = screen.render()
                    self.assertEqual(img.size, (PANEL_WIDTH, PANEL_HEIGHT))
                    helpers.assert_strict_1bit_image(self, img, f"{page} dark={dark}")
                    self.assertEqual(set(img.getdata()), {0, 255})
                    self.assertTrue(set(packed).issubset({0x0F, 0xFF}))

    def test_t1_f10_02_settings_page_has_lightness_contrast_controls_and_persists(self) -> None:
        from yogabook_eink.daemon import EinkSettingsDaemon
        from yogabook_eink.eink_menu import EinkSettingsScreen

        with helpers.IsolatedEinkEnvironment() as env:
            daemon = EinkSettingsDaemon()
            try:
                daemon.screen.page = "settings"
                daemon.screen.render()
                l_widgets = _find_menu_widgets(daemon.screen, "lightness")
                c_widgets = _find_menu_widgets(daemon.screen, "contrast")
                self.assertGreater(len(l_widgets), 0, "Missing Lightness control on E-Ink Settings page")
                self.assertGreater(len(c_widgets), 0, "Missing Contrast control on E-Ink Settings page")

                lw = _pick_step_widget(l_widgets)
                cw = _pick_step_widget(c_widgets)
                self.assertIsNotNone(daemon.screen.hit_test(lw.x + lw.w // 2, lw.y + lw.h // 2))
                self.assertIsNotNone(daemon.screen.hit_test(cw.x + cw.w // 2, cw.y + cw.h // 2))

                daemon._menu_active = True
                daemon._handle_action(lw.action_id)
                daemon._handle_action(cw.action_id)
                saved = env.read_settings_json()
                self.assertNotEqual(
                    (saved.get("lightness", 0), saved.get("contrast", 0)),
                    (0, 0),
                    "Lightness/Contrast action did not update settings.json",
                )
                patches = daemon.screen.render_widget_patches(lw.action_id)
                self.assertGreater(len(patches), 0)
            finally:
                daemon.sound.stop()
                daemon.listener.stop()
                daemon.ownership.stop()

    def test_t1_f10_03_daemon_hid_flow_gear_settings_toggle_kb(self) -> None:
        from yogabook_eink.daemon import EinkSettingsDaemon

        with helpers.IsolatedEinkEnvironment() as env:
            daemon = EinkSettingsDaemon()
            ok, _ = daemon.listener.start()
            self.assertTrue(ok)
            try:
                # 1. Tap top-right gear (1866, 18) -> Dashboard
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

                # 3. Tap toggle_key_down -> DU2 partial refresh + save
                kd_w = [w for w in daemon.screen.widgets if w.action_id == "toggle_key_down"][0]
                daemon._last_tap_ts = 0.0
                before_dpy = len(env.usb_recorder.dpy_area_ops)
                env.inject_touch(kd_w.x + 50, kd_w.y + 50, is_down=True)
                self.assertTrue(
                    helpers.wait_until(
                        lambda: len(env.usb_recorder.dpy_area_ops) > before_dpy and not daemon._busy,
                        timeout=2.0,
                    )
                )
                self.assertIs(env.read_settings_json()["key_down_sound"], True)
                self.assertEqual(env.usb_recorder.dpy_area_ops[-1].waveform, WAVEFORM_DU2)

                # 4. Tap bottom-right kb button -> exit_to_keyboard
                kb_w = [w for w in daemon.screen.widgets if w.action_id == "kb"][0]
                daemon._last_tap_ts = 0.0
                env.inject_touch(kb_w.x + 50, kb_w.y + 50, is_down=True)
                self.assertTrue(
                    helpers.wait_until(lambda: not daemon._menu_active and not daemon._busy, timeout=2.0)
                )
                self.assertGreaterEqual(env.usb_recorder.exit_kb_count, 1)
            finally:
                daemon.sound.stop()
                daemon.listener.stop()
                daemon.ownership.stop()

    def test_t1_f10_04_daemon_ruled_pad_bottom_center_exit(self) -> None:
        from yogabook_eink.daemon import EinkSettingsDaemon

        with helpers.IsolatedEinkEnvironment() as env:
            daemon = EinkSettingsDaemon()
            daemon.listener.start()
            try:
                daemon._menu_active = True
                daemon._handle_action("ruled")
                self.assertTrue(daemon._pad_active)
                daemon._last_tap_ts = 0.0
                before_kb = env.usb_recorder.exit_kb_count
                env.inject_touch(960, 1000, is_down=True)
                self.assertTrue(
                    helpers.wait_until(lambda: not daemon._pad_active and not daemon._busy, timeout=2.0)
                )
                self.assertEqual(env.usb_recorder.exit_kb_count, before_kb + 1)
            finally:
                daemon.sound.stop()
                daemon.listener.stop()
                daemon.ownership.stop()

    def test_t1_f10_05_daemon_syncs_external_settings_updates(self) -> None:
        from yogabook_eink.config import EinkSettings
        from yogabook_eink.daemon import EinkSettingsDaemon

        with helpers.IsolatedEinkEnvironment() as env:
            EinkSettings(lightness=0, contrast=0, dark_theme=False).save()
            daemon = EinkSettingsDaemon()
            try:
                # External writer updates settings.json
                EinkSettings(lightness=-2, contrast=3, dark_theme=True).save()
                daemon.screen.page = "settings"
                daemon._show_settings_screen("", fast=False)
                self.assertEqual(daemon.settings.lightness, -2)
                self.assertEqual(daemon.settings.contrast, 3)
                self.assertIs(daemon.settings.dark_theme, True)
            finally:
                daemon.sound.stop()
                daemon.listener.stop()
                daemon.ownership.stop()


# ===========================================================================
# Feature #11: GTK3 GUI Sliders, Dithering Selector & Settings Binding (5 tests)
# ===========================================================================


class TestT1Feature11GtkGui(unittest.TestCase):
    def test_t1_f11_01_gui_initializes_widgets_from_settings(self) -> None:
        from yogabook_eink.config import EinkSettings

        with helpers.IsolatedEinkEnvironment() as env:
            EinkSettings(
                lightness=2,
                contrast=4,
                dither_mode="Atkinson",
                key_down_sound=True,
                sound_profile="crisp",
                swap_axes=True,
                idle_settle_enabled=False,
            ).save()
            win, close_fn = helpers.open_isolated_window(env)
            try:
                scl_l = helpers.find_gtk_scale(win, "lightness", -3, 3)
                scl_c = helpers.find_gtk_scale(win, "contrast", -1, 6)
                cmb_d = helpers.find_dither_combo(win)
                tgl_settle = helpers.find_idle_settle_toggle(win)

                self.assertEqual(int(round(scl_l.get_value())), 2)
                self.assertEqual(int(round(scl_c.get_value())), 4)
                active_txt = (cmb_d.get_active_text() or cmb_d.get_active_id() or "").lower()
                self.assertIn("atkinson", active_txt)
                self.assertIs(win.chk_key_down.get_active(), True)
                self.assertEqual(win.cmb_sound_profile.get_active_id(), "crisp")
                self.assertIs(win.chk_swap_axes.get_active(), True)
                self.assertIs(tgl_settle.get_active(), False)
            finally:
                close_fn()

    def test_t1_f11_02_gui_sliders_persist_and_update_preview(self) -> None:
        from yogabook_eink.image_render import prepare_image_file

        with helpers.IsolatedEinkEnvironment() as env:
            png = helpers.make_gradient_png(os.path.join(env.tmpdir, "grad.png"), 64, 36)
            win, close_fn = helpers.open_isolated_window(env)
            try:
                win._selected_image_path = png
                win._update_image_preview()
                pix_before = win.img_preview.get_pixbuf().get_pixels()

                scl_l = helpers.find_gtk_scale(win, "lightness", -3, 3)
                scl_c = helpers.find_gtk_scale(win, "contrast", -1, 6)
                scl_l.set_value(-2)
                scl_c.set_value(3)
                helpers.pump_gtk_events()

                saved = env.read_settings_json()
                self.assertEqual(saved["lightness"], -2)
                self.assertEqual(saved["contrast"], 3)
                pix_after = win.img_preview.get_pixbuf().get_pixels()
                self.assertNotEqual(pix_before, pix_after)
                # Ensure prepare_image_file agrees with saved state
                img_exp, _ = prepare_image_file(png, lightness=-2, contrast=3)
                self.assertEqual(img_exp.size, (PANEL_WIDTH, PANEL_HEIGHT))
            finally:
                close_fn()

    def test_t1_f11_03_gui_dither_selector_and_send_image(self) -> None:
        from yogabook_eink.image_render import DITHER_MODES, prepare_image_file

        with helpers.IsolatedEinkEnvironment() as env:
            png = helpers.make_gradient_png(os.path.join(env.tmpdir, "grad.png"), 64, 36)
            win, close_fn = helpers.open_isolated_window(env)
            try:
                win._selected_image_path = png
                cmb_d = helpers.find_dither_combo(win)
                for mode in list(DITHER_MODES[1:]) + [DITHER_MODES[0]]:
                    helpers.select_dither_mode_in_combo(cmb_d, mode)
                    self.assertEqual(env.read_settings_json()["dither_mode"], mode)

                helpers.select_dither_mode_in_combo(cmb_d, "8x8 Bayer")
                scl_l = helpers.find_gtk_scale(win, "lightness", -3, 3)
                scl_c = helpers.find_gtk_scale(win, "contrast", -1, 6)
                scl_l.set_value(1)
                scl_c.set_value(2)
                helpers.pump_gtk_events()

                before = len(env.usb_recorder.dpy_area_ops)
                btn_send = helpers.find_gtk_button(win, "Send Image to E-Ink Display")
                btn_send.clicked()
                self.assertTrue(
                    helpers.wait_until(lambda: len(env.usb_recorder.dpy_area_ops) > before and not win._busy, timeout=3.0)
                )
                _, expected_packed = prepare_image_file(
                    png, dither_mode="8x8 Bayer", lightness=1, contrast=2
                )
                self.assertEqual(bytes(env.usb_recorder.framebuffer), expected_packed)
                helpers.assert_r5_protocol_invariants(self, env.usb_recorder, allow_standalone_0xa9=True)
            finally:
                close_fn()

    def test_t1_f11_04_gui_control_toggles_persist_and_propagate(self) -> None:
        with helpers.IsolatedEinkEnvironment() as env:
            win, close_fn = helpers.open_isolated_window(env)
            try:
                win.chk_key_up.set_active(True)
                win.chk_flip_x.set_active(True)
                win.chk_rot180.set_active(True)
                win.cmb_wf.set_active_id(str(WAVEFORM_GL16))
                tgl_settle = helpers.find_idle_settle_toggle(win)
                tgl_settle.set_active(False)
                helpers.pump_gtk_events()

                saved = env.read_settings_json()
                self.assertIs(saved["key_up_sound"], True)
                self.assertIs(saved["flip_x"], True)
                self.assertIs(saved["rotate_180"], True)
                self.assertEqual(saved["waveform"], WAVEFORM_GL16)
                self.assertIs(saved["idle_settle_enabled"], False)
                self.assertIs(win._hid_listener.flip_x, True)
                self.assertIs(win._sound_daemon.key_up_enabled, True)
                self.assertEqual(win._eink_menu.settings.waveform, WAVEFORM_GL16)
            finally:
                close_fn()

    def test_t1_f11_05_gui_launcher_subprocess_clean_exit(self) -> None:
        helpers.GtkDisplaySession.ensure_ready()
        with tempfile.TemporaryDirectory(prefix="yb_t1_f11_sub_") as tmp:
            with tempfile.TemporaryDirectory(prefix="yb_t1_f11_shim_") as shim_dir:
                with open(os.path.join(shim_dir, "sitecustomize.py"), "w", encoding="utf-8") as f:
                    f.write(helpers._SITECUSTOMIZE_CODE)
                trace_path = os.path.join(shim_dir, "usb.jsonl")
                env = dict(os.environ)
                env["HOME"] = tmp
                env["XDG_CONFIG_HOME"] = os.path.join(tmp, ".config")
                env["XDG_RUNTIME_DIR"] = os.path.join(tmp, "run")
                os.makedirs(env["XDG_RUNTIME_DIR"], exist_ok=True)
                env["PYTHONPATH"] = f"{shim_dir}{os.pathsep}{REPO_ROOT}"
                env["YOGABOOK_E2E_USB_LOG"] = trace_path
                env["YOGABOOK_E2E_GUI_AUTOQUIT_MS"] = "350"
                broadway_proc = None
                try:
                    if env.get("GDK_BACKEND") == "broadway":
                        with socket.socket() as s:
                            s.bind(("127.0.0.1", 0))
                            b_port = s.getsockname()[1]
                        b_disp = f":{(b_port % 80) + 10}"
                        env["BROADWAY_DISPLAY"] = b_disp
                        broadway_proc = subprocess.Popen(
                            ["broadwayd", b_disp, "-p", str(b_port)],
                            env=env,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        for _ in range(30):
                            if broadway_proc.poll() is not None:
                                break
                            if helpers.GtkDisplaySession._probe_display_in_subprocess(env):
                                break
                            time.sleep(0.05)
                    proc = subprocess.run(
                        [sys.executable, helpers.GUI_PATH],
                        env=env,
                        cwd=REPO_ROOT,
                        capture_output=True,
                        text=True,
                        timeout=15.0,
                    )
                    self.assertEqual(proc.returncode, 0, f"stderr={proc.stderr}")
                finally:
                    if broadway_proc is not None:
                        try:
                            broadway_proc.terminate()
                            broadway_proc.wait(timeout=2)
                        except Exception:
                            try:
                                broadway_proc.kill()
                                broadway_proc.wait(timeout=1)
                            except Exception:
                                pass


# ===========================================================================
# Feature #12: Matrix Saver 1-bit & Bottom-Center Exit (5 tests)
# ===========================================================================


class TestT1Feature12MatrixSaver(unittest.TestCase):
    def test_t1_f12_01_matrix_renderer_strict_1bit_dark_and_light(self) -> None:
        from yogabook_eink.matrix_saver import MatrixRainRenderer

        for dark in (True, False):
            r = MatrixRainRenderer(dark_bg=dark)
            for _ in range(15):
                frame = r.step_frame()
                self.assertEqual(frame.size, (PANEL_WIDTH, PANEL_HEIGHT))
                helpers.assert_strict_1bit_image(self, frame)
                self.assertEqual(set(frame.getdata()), {0, 255})

    def test_t1_f12_02_matrix_ignores_extreme_tone_lut_settings(self) -> None:
        from yogabook_eink.config import EinkSettings
        from yogabook_eink.matrix_saver import run_matrix_screensaver

        with helpers.IsolatedEinkEnvironment() as env:
            EinkSettings(lightness=3, contrast=6).save()
            stop_ev = threading.Event()
            frames = []

            def _on_frame(n: int) -> None:
                frames.append(n)
                if n >= 5:
                    stop_ev.set()

            run_matrix_screensaver(stop_ev, dark_bg=True, on_frame=_on_frame)
            self.assertGreaterEqual(len(frames), 5)
            for op in env.usb_recorder.ld_img_ops:
                self.assertTrue(set(op.payload).issubset({0x0F, 0xFF}))

    def test_t1_f12_03_matrix_exit_badge_survives_rain_overlap(self) -> None:
        from yogabook_eink.matrix_saver import MatrixRainRenderer

        r = MatrixRainRenderer(dark_bg=True)
        for _ in range(30):
            frame = r.step_frame(cols_per_tick=8)
        badge = frame.crop((720, 976, 1200, 1052))
        helpers.assert_strict_1bit_image(self, badge)
        self.assertEqual(set(badge.getdata()), {0, 255})

    def test_t1_f12_04_matrix_uses_compositor_8px_du2_and_gc16_budget(self) -> None:
        from yogabook_eink.matrix_saver import run_matrix_screensaver
        from yogabook_eink.ownership import OwnershipServer

        with helpers.IsolatedEinkEnvironment() as env:
            srv = OwnershipServer()
            helpers.start_ownership_server(srv)
            stop_ev = threading.Event()

            def _on_frame(n: int) -> None:
                if (n >= 6 and any(d.waveform == WAVEFORM_GC16 for d in env.usb_recorder.dpy_area_ops)) or n >= 20:
                    stop_ev.set()

            try:
                run_matrix_screensaver(stop_ev, dark_bg=True, gc16_budget=4, on_frame=_on_frame)
                self.assertTrue(helpers.wait_until(lambda: srv.active_owner is None, timeout=2.0))
                wfs = [d.waveform for d in env.usb_recorder.dpy_area_ops]
                self.assertIn(WAVEFORM_DU2, wfs)
                self.assertIn(WAVEFORM_GC16, wfs)
                for dpy in env.usb_recorder.dpy_area_ops:
                    if not (dpy.x == 0 and dpy.y == 0 and dpy.width == PANEL_WIDTH and dpy.height == PANEL_HEIGHT):
                        helpers.assert_8px_aligned(self, dpy.x, dpy.y, dpy.width, dpy.height)
                helpers.assert_r5_protocol_invariants(self, env.usb_recorder, allow_standalone_0xa9=True)
            finally:
                srv.stop()

    def test_t1_f12_05_matrix_bottom_center_touch_clean_exit(self) -> None:
        from yogabook_eink.matrix_saver import run_matrix_screensaver
        from yogabook_eink.ownership import OwnershipServer

        with helpers.IsolatedEinkEnvironment() as env:
            srv = OwnershipServer()
            helpers.start_ownership_server(srv)
            stop_ev = threading.Event()
            t = threading.Thread(
                target=run_matrix_screensaver,
                kwargs={"stop_event": stop_ev, "dark_bg": True},
                daemon=True,
            )
            t.start()
            try:
                self.assertTrue(
                    helpers.wait_until(lambda: len(env.usb_recorder.dpy_area_ops) >= 2, timeout=3.0)
                )
                env.inject_touch(960, 1014, is_down=True)
                t.join(timeout=3.0)
                self.assertFalse(t.is_alive())
                self.assertTrue(stop_ev.is_set())
                self.assertGreaterEqual(env.usb_recorder.exit_kb_count, 1)
                self.assertTrue(helpers.wait_until(lambda: srv.active_owner is None, timeout=2.0))
                self.assertTrue(all(not l.is_running for l in env.listeners))
            finally:
                stop_ev.set()
                t.join(timeout=2.0)
                srv.stop()


if __name__ == "__main__":
    unittest.main()
