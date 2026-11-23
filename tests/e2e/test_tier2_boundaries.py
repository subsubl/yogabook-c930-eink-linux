"""Tier 2 E2E Boundary, Corner, and Error Tests (65 tests across Features #1-#12)."""

from __future__ import annotations

import inspect
import os
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
    OP_GET_DPY_STATUS,
    PANEL_HEIGHT,
    PANEL_WIDTH,
    USBC_HEADER,
    WAVEFORM_DU2,
    WAVEFORM_GC16,
    WAVEFORM_GL16,
    WAVEFORM_INIT,
    Ite8951Controller,
)
from yogabook_eink.usb_lib import LIBUSB_ERROR_TIMEOUT, UsbError  # noqa: E402


# ===========================================================================
# Feature #1: Tone LUT Boundaries (6 tests)
# ===========================================================================


class TestT2Feature01ToneLutBoundaries(unittest.TestCase):
    def test_t2_f1_01_constants_and_clamp_helpers(self) -> None:
        from yogabook_eink.tone_lut import (
            CONTRAST_MAX,
            CONTRAST_MIN,
            LIGHTNESS_MAX,
            LIGHTNESS_MIN,
            clamp_contrast,
            clamp_lightness,
        )

        self.assertEqual(LIGHTNESS_MIN, -3)
        self.assertEqual(LIGHTNESS_MAX, 3)
        self.assertEqual(CONTRAST_MIN, -1)
        self.assertEqual(CONTRAST_MAX, 6)
        self.assertEqual(clamp_lightness(-99), -3)
        self.assertEqual(clamp_lightness(99), 3)
        self.assertEqual(clamp_lightness(0), 0)
        self.assertEqual(clamp_contrast(-99), -1)
        self.assertEqual(clamp_contrast(99), 6)
        self.assertEqual(clamp_contrast(2), 2)

    def test_t2_f1_02_out_of_range_build_tone_lut_clamps_to_endpoints(self) -> None:
        from yogabook_eink.tone_lut import build_tone_lut

        self.assertEqual(build_tone_lut(-100, -100), build_tone_lut(-3, -1))
        self.assertEqual(build_tone_lut(100, 100), build_tone_lut(3, 6))

    def test_t2_f1_03_contrast_minus_1_endpoint_compression(self) -> None:
        from yogabook_eink.tone_lut import build_tone_lut

        lut = build_tone_lut(0, -1)
        self.assertEqual(lut[0], 25)
        self.assertEqual(lut[255], 230)

    def test_t2_f1_04_contrast_plus_6_step_threshold(self) -> None:
        from yogabook_eink.tone_lut import build_tone_lut

        lut = build_tone_lut(0, 6)
        self.assertEqual(lut[:128], b"\x00" * 128)
        self.assertEqual(lut[128:], b"\xff" * 128)

    def test_t2_f1_05_float32_vs_float64_precision_guards(self) -> None:
        from yogabook_eink.tone_lut import build_tone_lut

        self.assertEqual(build_tone_lut(0, -1)[45], 61)
        self.assertEqual(build_tone_lut(0, 5)[117], 2)
        self.assertEqual(build_tone_lut(0, 5)[128], 134)

    def test_t2_f1_06_edge_sizes_and_input_immutability(self) -> None:
        from yogabook_eink.tone_lut import apply_tone_lut, build_tone_lut

        for w, h in ((1, 1), (17, 1)):
            src = Image.new("L", (w, h), color=120)
            raw_before = src.tobytes()
            out = apply_tone_lut(src, lightness=2, contrast=2)
            self.assertEqual(src.tobytes(), raw_before)
            self.assertEqual(out.size, (w, h))
            self.assertEqual(set(out.getdata()), {build_tone_lut(2, 2)[120]})


# ===========================================================================
# Feature #2: Settings Persistence Boundaries (6 tests)
# ===========================================================================


class TestT2Feature02ConfigBoundaries(unittest.TestCase):
    def test_t2_f2_01_out_of_range_lightness_contrast_clamped(self) -> None:
        import json
        from yogabook_eink.config import EinkSettings

        with tempfile.TemporaryDirectory(prefix="yb_t2_f2_") as tmp:
            p = os.path.join(tmp, "settings.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"lightness": -50, "contrast": 50}, f)
            s = EinkSettings.load(path=p)
            self.assertEqual(s.lightness, -3)
            self.assertEqual(s.contrast, 6)

            with open(p, "w", encoding="utf-8") as f:
                json.dump({"lightness": 50, "contrast": -50}, f)
            s2 = EinkSettings.load(path=p)
            self.assertEqual(s2.lightness, 3)
            self.assertEqual(s2.contrast, -1)

    def test_t2_f2_02_invalid_dither_mode_falls_back(self) -> None:
        import json
        from yogabook_eink.config import EinkSettings
        from yogabook_eink.image_render import DITHER_MODES

        with tempfile.TemporaryDirectory(prefix="yb_t2_f2_dm_") as tmp:
            p = os.path.join(tmp, "settings.json")
            for bad in ("unknown-dither", 123, None, ""):
                with open(p, "w", encoding="utf-8") as f:
                    json.dump({"dither_mode": bad}, f)
                s = EinkSettings.load(path=p)
                self.assertIn(s.dither_mode, DITHER_MODES)
                self.assertEqual(s.dither_mode, "Grayscale 16-level")

    def test_t2_f2_03_corrupt_or_non_dict_json_returns_defaults(self) -> None:
        from yogabook_eink.config import EinkSettings

        with tempfile.TemporaryDirectory(prefix="yb_t2_f2_bad_") as tmp:
            p = os.path.join(tmp, "settings.json")
            for payload in ("{not valid", "", "[]", '"hello"', "42"):
                with open(p, "w", encoding="utf-8") as f:
                    f.write(payload)
                s = EinkSettings.load(path=p)
                self.assertEqual(s, EinkSettings())

    def test_t2_f2_04_missing_file_returns_defaults_without_creating(self) -> None:
        from yogabook_eink.config import EinkSettings

        with tempfile.TemporaryDirectory(prefix="yb_t2_f2_miss_") as tmp:
            p = os.path.join(tmp, "nonexistent", "settings.json")
            s = EinkSettings.load(path=p)
            self.assertEqual(s, EinkSettings())
            self.assertFalse(os.path.exists(p))

    def test_t2_f2_05_non_numeric_fields_and_bad_idle_settle_sec(self) -> None:
        import json
        from yogabook_eink.config import EinkSettings

        with tempfile.TemporaryDirectory(prefix="yb_t2_f2_num_") as tmp:
            p = os.path.join(tmp, "settings.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "lightness": "bad",
                        "contrast": None,
                        "idle_settle_sec": -3.0,
                    },
                    f,
                )
            s = EinkSettings.load(path=p)
            self.assertEqual(s.lightness, 0)
            self.assertEqual(s.contrast, 0)
            self.assertGreaterEqual(s.idle_settle_sec, 0.0)

            with open(p, "w", encoding="utf-8") as f:
                json.dump({"idle_settle_sec": "nan"}, f)
            s2 = EinkSettings.load(path=p)
            self.assertGreater(s2.idle_settle_sec, 0.0)

    def test_t2_f2_06_unknown_extra_keys_ignored(self) -> None:
        import json
        from yogabook_eink.config import EinkSettings

        with tempfile.TemporaryDirectory(prefix="yb_t2_f2_extra_") as tmp:
            p = os.path.join(tmp, "settings.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"lightness": 2, "future_key": "xyz", "another": 999}, f)
            s = EinkSettings.load(path=p)
            self.assertEqual(s.lightness, 2)
            self.assertFalse(hasattr(s, "future_key"))


# ===========================================================================
# Feature #3: Dithering Boundaries (6 tests)
# ===========================================================================


class TestT2Feature03DitheringBoundaries(unittest.TestCase):
    def test_t2_f3_01_unknown_dither_mode_raises_value_error(self) -> None:
        from yogabook_eink.image_render import dither_image, normalize_dither_mode

        with self.assertRaises(ValueError):
            normalize_dither_mode("nonexistent-mode")
        with self.assertRaises(ValueError):
            dither_image(Image.new("L", (8, 8), color=128), mode="nonexistent-mode")

    def test_t2_f3_02_pure_black_and_white_remain_uniform(self) -> None:
        from yogabook_eink.image_render import DITHER_MODES, dither_image

        for mode in DITHER_MODES:
            for val in (0, 255):
                img = Image.new("L", (16, 16), color=val)
                out2 = dither_image(img, mode=mode)
                self.assertEqual(set(out2.getdata()), {val}, f"{mode} val={val}")
                out16 = dither_image(img, mode=mode, levels=16)
                self.assertEqual(set(out16.getdata()), {val}, f"{mode} L16 val={val}")

    def test_t2_f3_03_tiny_and_odd_dimensions_no_index_error(self) -> None:
        from yogabook_eink.image_render import DITHER_MODES, dither_image

        for mode in DITHER_MODES:
            for size in ((1, 1), (1, 10), (10, 1), (2, 2), (3, 3)):
                src = helpers.make_gradient_image(*size) if size[0] > 1 else Image.new("L", size, color=128)
                out = dither_image(src, mode=mode)
                self.assertEqual(out.size, size, f"{mode} {size}")
                if mode == "Grayscale 16-level":
                    helpers.assert_16_gray_image(self, out)
                else:
                    helpers.assert_strict_1bit_image(self, out)

    def test_t2_f3_04_invalid_levels_and_non_L_conversions(self) -> None:
        from yogabook_eink.image_render import dither_image

        src = Image.new("L", (16, 16), color=128)
        for bad_lvl in (0, 1, -2, 257, True):
            with self.assertRaises(ValueError):
                dither_image(src, mode="4x4 Bayer", levels=bad_lvl)

        for img_mode, col in (("RGB", (100, 150, 200)), ("RGBA", (100, 150, 200, 255)), ("1", 1)):
            im = Image.new(img_mode, (16, 16), color=col)
            out = dither_image(im, mode="Atkinson")
            self.assertEqual(out.mode, "L")
            helpers.assert_strict_1bit_image(self, out)

    def test_t2_f3_05_prepare_image_file_dither_fallback_precedence(self) -> None:
        from yogabook_eink.image_render import prepare_image_file

        with tempfile.TemporaryDirectory(prefix="yb_t2_f3_fb_") as tmp:
            png = helpers.make_gradient_png(os.path.join(tmp, "g.png"), 64, 36)
            img_fs, _ = prepare_image_file(png, dither_mode="Floyd-Steinberg")
            img_def1, _ = prepare_image_file(png, dither_1bit=True, dither_mode=None)
            self.assertEqual(img_def1.tobytes(), img_fs.tobytes())

            img_g16, _ = prepare_image_file(png, dither_mode="Grayscale 16-level")
            img_def0, _ = prepare_image_file(png, dither_1bit=False, dither_mode=None)
            self.assertEqual(img_def0.tobytes(), img_g16.tobytes())

            img_atk, _ = prepare_image_file(png, dither_mode="Atkinson")
            img_ovr, _ = prepare_image_file(png, dither_1bit=True, dither_mode="Atkinson")
            self.assertEqual(img_ovr.tobytes(), img_atk.tobytes())

    def test_t2_f3_06_cover_invert_rotate_and_missing_file(self) -> None:
        from yogabook_eink.image_render import prepare_image_file

        with tempfile.TemporaryDirectory(prefix="yb_t2_f3_cov_") as tmp:
            png = helpers.make_asymmetric_png(os.path.join(tmp, "asym.png"), 40, 80)
            img_con, _ = prepare_image_file(png, fit_mode="contain", dither_mode="4x4 Bayer")
            img_cov, _ = prepare_image_file(png, fit_mode="cover", dither_mode="4x4 Bayer")
            self.assertNotEqual(img_con.tobytes(), img_cov.tobytes())

            img_inv, _ = prepare_image_file(png, fit_mode="cover", invert=True, dither_mode="4x4 Bayer")
            img_rot, _ = prepare_image_file(png, fit_mode="cover", invert=True, rotate_180=True, dither_mode="4x4 Bayer")
            self.assertEqual(img_rot.tobytes(), img_inv.rotate(180).tobytes())

            with self.assertRaises(FileNotFoundError):
                prepare_image_file(os.path.join(tmp, "missing.png"))


# ===========================================================================
# Feature #4: CLI Boundaries & Errors (5 tests)
# ===========================================================================


class TestT2Feature04CliBoundaries(unittest.TestCase):
    def test_t2_f4_01_invalid_dither_mode_fails_with_zero_blits(self) -> None:
        with tempfile.TemporaryDirectory(prefix="yb_t2_f4_bad_") as tmp:
            png = helpers.make_gradient_png(os.path.join(tmp, "g.png"), 32, 24)
            res = helpers.run_cli_isolated(["image", png, "--dither-mode", "nope"], home_dir=tmp)
            self.assertNotEqual(res.returncode, 0)
            self.assertEqual(len(res.dpy_area_ops), 0)

    def test_t2_f4_02_non_int_lightness_and_missing_file_fail_cleanly(self) -> None:
        with tempfile.TemporaryDirectory(prefix="yb_t2_f4_err_") as tmp:
            png = helpers.make_gradient_png(os.path.join(tmp, "g.png"), 32, 24)
            res_arg = helpers.run_cli_isolated(["image", png, "--lightness", "abc"], home_dir=tmp)
            self.assertNotEqual(res_arg.returncode, 0)
            self.assertEqual(len(res_arg.dpy_area_ops), 0)

            res_miss = helpers.run_cli_isolated(["image", os.path.join(tmp, "nope.png")], home_dir=tmp)
            self.assertNotEqual(res_miss.returncode, 0)
            self.assertEqual(len(res_miss.dpy_area_ops), 0)

    def test_t2_f4_03_boundary_endpoints_and_out_of_range_lightness_contrast(self) -> None:
        from yogabook_eink.image_render import prepare_image_file

        with tempfile.TemporaryDirectory(prefix="yb_t2_f4_bnd_") as tmp:
            png = helpers.make_gradient_png(os.path.join(tmp, "g.png"), 48, 32)
            for l_val, c_val in ((-3, -1), (3, 6)):
                res = helpers.run_cli_isolated(
                    ["image", png, "--lightness", str(l_val), "--contrast", str(c_val)],
                    home_dir=tmp,
                )
                self.assertEqual(res.returncode, 0, res.stderr)
                _, exp = prepare_image_file(png, lightness=l_val, contrast=c_val)
                self.assertEqual(res.framebuffer, exp)

            res_oob = helpers.run_cli_isolated(
                ["image", png, "--lightness", "99", "--contrast", "99"],
                home_dir=tmp,
            )
            self.assertNotIn("Traceback", res_oob.stderr)
            if res_oob.returncode == 0:
                _, exp_clamped = prepare_image_file(png, lightness=3, contrast=6)
                self.assertEqual(res_oob.framebuffer, exp_clamped)

    def test_t2_f4_04_explicit_dither_mode_overrides_legacy_dither(self) -> None:
        from yogabook_eink.image_render import prepare_image_file

        with tempfile.TemporaryDirectory(prefix="yb_t2_f4_ovr_") as tmp:
            png = helpers.make_gradient_png(os.path.join(tmp, "g.png"), 48, 32)
            res = helpers.run_cli_isolated(
                ["image", png, "--dither", "--dither-mode", "atkinson"],
                home_dir=tmp,
            )
            self.assertEqual(res.returncode, 0, res.stderr)
            _, exp = prepare_image_file(png, dither_mode="Atkinson")
            self.assertEqual(res.framebuffer, exp)

    def test_t2_f4_05_usb_error_exits_1_with_stderr_message(self) -> None:
        with tempfile.TemporaryDirectory(prefix="yb_t2_f4_usb_") as tmp:
            res = helpers.run_cli_isolated(["status"], home_dir=tmp, fail_opcode=0x80)
            self.assertEqual(res.returncode, 1)
            self.assertIn("USB Error:", res.stderr)


# ===========================================================================
# Feature #5: 8px Alignment Boundaries (5 tests)
# ===========================================================================


class TestT2Feature05AlignmentBoundaries(unittest.TestCase):
    def test_t2_f5_01_negative_origin_clips_then_aligns(self) -> None:
        from yogabook_eink.compositor import align_rect

        self.assertEqual(align_rect(-10, -5, 20, 12), (0, 0, 16, 8))

    def test_t2_f5_02_bottom_right_boundary_clamps(self) -> None:
        from yogabook_eink.compositor import align_rect

        self.assertEqual(align_rect(1915, 1075, 10, 10), (1912, 1072, 8, 8))

    def test_t2_f5_03_out_of_bounds_origins_return_none(self) -> None:
        from yogabook_eink.compositor import align_rect

        self.assertIsNone(align_rect(1920, 0, 10, 10))
        self.assertIsNone(align_rect(0, 1080, 10, 10))
        self.assertIsNone(align_rect(-20, 0, 5, 10))

    def test_t2_f5_04_zero_negative_and_1px_post_clip_return_none(self) -> None:
        from yogabook_eink.compositor import align_rect

        self.assertIsNone(align_rect(10, 10, 0, 10))
        self.assertIsNone(align_rect(10, 10, 10, 0))
        self.assertIsNone(align_rect(10, 10, -5, 10))
        self.assertIsNone(align_rect(1912, 0, 1, 16, max_w=1913))

    def test_t2_f5_05_identical_frames_empty_and_1x1_corner_pixel(self) -> None:
        from yogabook_eink.compositor import compute_dirty_rects

        f0 = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
        self.assertEqual(compute_dirty_rects(f0, f0.copy()), [])

        f1 = f0.copy()
        f1.putpixel((1919, 1079), 0)
        rects = compute_dirty_rects(f0, f1)
        self.assertEqual(len(rects), 1)
        self.assertEqual((rects[0].x, rects[0].y, rects[0].w, rects[0].h), (1912, 1072, 8, 8))


# ===========================================================================
# Feature #6: Semantic Refresh Policy Boundaries (5 tests)
# ===========================================================================


class TestT2Feature06PolicyBoundaries(unittest.TestCase):
    def test_t2_f6_01_invalid_policy_raises_value_error_no_usb(self) -> None:
        from yogabook_eink.compositor import EinkCompositor

        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        comp = EinkCompositor(ctrl, max_fps=1000.0, idle_settle_enabled=False)
        try:
            f = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
            for bad in ("invalid", ""):
                with self.assertRaises(ValueError):
                    comp.present(f, policy=bad)
            self.assertEqual(len(rec.dpy_area_ops), 0)
        finally:
            comp.close()
            ctrl.close()

    def test_t2_f6_02_force_full_promotes_and_resets_partial_count(self) -> None:
        from yogabook_eink.compositor import EinkCompositor

        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        comp = EinkCompositor(ctrl, max_fps=1000.0, idle_settle_enabled=False)
        try:
            comp.reset_framebuffer(255)
            f = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
            ImageDraw.Draw(f).rectangle([(16, 16), (32, 32)], fill=0)
            comp.present(f, policy="ui")
            self.assertEqual(comp.partial_count, 1)

            ImageDraw.Draw(f).rectangle([(48, 48), (64, 64)], fill=0)
            comp.present(f, policy="ui", force_full=True, clean_waveform=WAVEFORM_GC16)
            self.assertEqual(comp.partial_count, 0)
            dpy = rec.dpy_area_ops[-1]
            self.assertEqual(dpy.waveform, WAVEFORM_GC16)
            self.assertEqual((dpy.width, dpy.height), (PANEL_WIDTH, PANEL_HEIGHT))
        finally:
            comp.close()
            ctrl.close()

    def test_t2_f6_03_gc16_budget_triggers_full_and_resets_counter(self) -> None:
        from yogabook_eink.compositor import EinkCompositor

        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        comp = EinkCompositor(ctrl, gc16_budget=5, max_fps=1000.0, idle_settle_enabled=False)
        try:
            comp.reset_framebuffer(255)
            base = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
            for i in range(5):
                f = base.copy()
                ImageDraw.Draw(f).rectangle([(16 + i * 24, 16), (31 + i * 24, 31)], fill=0)
                comp.present(f, policy="fast")
            self.assertEqual(comp.partial_count, 5)
            f_next = base.copy()
            ImageDraw.Draw(f_next).rectangle([(300, 300), (320, 320)], fill=0)
            comp.present(f_next, policy="fast")
            self.assertEqual(comp.partial_count, 0)
            dpy = rec.dpy_area_ops[-1]
            self.assertEqual(dpy.waveform, WAVEFORM_GC16)
            self.assertEqual((dpy.width, dpy.height), (PANEL_WIDTH, PANEL_HEIGHT))
        finally:
            comp.close()
            ctrl.close()

    def test_t2_f6_04_over_65_percent_dirty_collapses_to_full_blit(self) -> None:
        from yogabook_eink.compositor import EinkCompositor

        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        comp = EinkCompositor(ctrl, max_fps=1000.0, idle_settle_enabled=False)
        try:
            comp.reset_framebuffer(255)
            f = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
            # Change > 75% of screen
            ImageDraw.Draw(f).rectangle([(0, 0), (1919, 850)], fill=0)
            comp.present(f, policy="fast")
            self.assertEqual(len(rec.dpy_area_ops), 1)
            dpy = rec.dpy_area_ops[0]
            self.assertEqual((dpy.x, dpy.y, dpy.width, dpy.height), (0, 0, PANEL_WIDTH, PANEL_HEIGHT))
            self.assertEqual(dpy.waveform, WAVEFORM_DU2)

            f2 = f.copy()
            ImageDraw.Draw(f2).rectangle([(0, 0), (1919, 850)], fill=128)
            comp.present(f2, policy="partial")
            dpy2 = rec.dpy_area_ops[-1]
            self.assertEqual((dpy2.x, dpy2.y, dpy2.width, dpy2.height), (0, 0, PANEL_WIDTH, PANEL_HEIGHT))
            self.assertIn(dpy2.waveform, (WAVEFORM_GL16, WAVEFORM_GC16))
        finally:
            comp.close()
            ctrl.close()

    def test_t2_f6_05_unchanged_frame_zero_blits_and_resizes_input(self) -> None:
        from yogabook_eink.compositor import EinkCompositor

        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        comp = EinkCompositor(ctrl, max_fps=1000.0, idle_settle_enabled=False)
        try:
            comp.reset_framebuffer(255)
            small_white = Image.new("L", (320, 240), color=255)
            comp.present(small_white, policy="fast")
            self.assertEqual(len(rec.dpy_area_ops), 0)

            small_black = Image.new("L", (320, 240), color=0)
            comp.present(small_black, policy="partial")
            self.assertEqual(len(rec.dpy_area_ops), 1)
            self.assertEqual((rec.dpy_area_ops[0].width, rec.dpy_area_ops[0].height), (PANEL_WIDTH, PANEL_HEIGHT))
        finally:
            comp.close()
            ctrl.close()


# ===========================================================================
# Feature #7: Idle Settle Boundaries (5 tests)
# ===========================================================================


class TestT2Feature07IdleSettleBoundaries(unittest.TestCase):
    def test_t2_f7_01_idle_settle_disabled_never_fires(self) -> None:
        from yogabook_eink.compositor import EinkCompositor

        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        comp = EinkCompositor(ctrl, max_fps=1000.0, idle_settle_enabled=False, idle_settle_sec=0.08)
        try:
            comp.reset_framebuffer(255)
            f = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
            ImageDraw.Draw(f).rectangle([(16, 16), (48, 48)], fill=0)
            comp.present(f, policy="fast")
            time.sleep(0.20)
            gc16_ops = [
                (d.waveform, d.x, d.y, d.width, d.height)
                for d in rec.dpy_area_ops
                if d.waveform == WAVEFORM_GC16
            ]
            self.assertEqual(len(gc16_ops), 0, f"Unexpected GC16 blits: {gc16_ops}")
        finally:
            comp.close()
            ctrl.close()

    def test_t2_f7_02_unchanged_frame_after_settle_does_not_retrigger(self) -> None:
        from yogabook_eink.compositor import EinkCompositor

        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        comp = EinkCompositor(ctrl, max_fps=1000.0, idle_settle_enabled=True, idle_settle_sec=0.08)
        try:
            comp.reset_framebuffer(255)
            f = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
            ImageDraw.Draw(f).rectangle([(16, 16), (48, 48)], fill=0)
            comp.present(f, policy="fast")
            self.assertTrue(
                helpers.wait_until(
                    lambda: len([d for d in rec.dpy_area_ops if d.waveform == WAVEFORM_GC16]) == 1,
                    timeout=1.5,
                ),
                "Expected initial idle GC16 settle to fire",
            )

            comp.present(f, policy="fast")
            time.sleep(0.16)
            self.assertEqual(len([d for d in rec.dpy_area_ops if d.waveform == WAVEFORM_GC16]), 1)
        finally:
            comp.close()
            ctrl.close()

    def test_t2_f7_03_usb_lock_blocks_settle_until_released(self) -> None:
        from yogabook_eink.compositor import EinkCompositor

        usb_lock = threading.RLock()
        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        comp = EinkCompositor(
            ctrl, max_fps=1000.0, idle_settle_enabled=True, idle_settle_sec=0.08, usb_lock=usb_lock
        )
        try:
            comp.reset_framebuffer(255)
            f = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
            ImageDraw.Draw(f).rectangle([(16, 16), (48, 48)], fill=0)
            comp.present(f, policy="fast")

            with usb_lock:
                time.sleep(0.18)
                gc16_ops = [
                    (d.waveform, d.x, d.y, d.width, d.height)
                    for d in rec.dpy_area_ops
                    if d.waveform == WAVEFORM_GC16
                ]
                self.assertEqual(len(gc16_ops), 0, f"Unexpected GC16 blits: {gc16_ops}")

            self.assertTrue(
                helpers.wait_until(
                    lambda: len([d for d in rec.dpy_area_ops if d.waveform == WAVEFORM_GC16]) == 1,
                    timeout=1.0,
                )
            )
            self.assertFalse(rec.interleaved_transfer_detected)
        finally:
            comp.close()
            ctrl.close()

    def test_t2_f7_04_repeated_and_concurrent_close_safe(self) -> None:
        from yogabook_eink.compositor import EinkCompositor

        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        comp = EinkCompositor(ctrl, max_fps=1000.0, idle_settle_enabled=True, idle_settle_sec=0.05)
        try:
            comp.reset_framebuffer(255)
            f = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
            ImageDraw.Draw(f).rectangle([(16, 16), (48, 48)], fill=0)
            comp.present(f, policy="fast")
            time.sleep(0.04)
            comp.close()
            comp.close()
            count_at_close = len(rec.dpy_area_ops)
            time.sleep(0.12)
            self.assertEqual(len(rec.dpy_area_ops), count_at_close)
        finally:
            comp.close()
            ctrl.close()

    def test_t2_f7_05_burst_cancel_du2_full_resets_partial_count(self) -> None:
        from yogabook_eink.compositor import EinkCompositor

        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        comp = EinkCompositor(ctrl, max_fps=1000.0, idle_settle_enabled=True, idle_settle_sec=0.08)
        try:
            comp.reset_framebuffer(255)
            f = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
            ImageDraw.Draw(f).rectangle([(16, 16), (48, 48)], fill=0)
            comp.present(f, policy="fast")
            comp.cancel_settle()
            ImageDraw.Draw(f).rectangle([(64, 64), (96, 96)], fill=0)
            comp.present(f, policy="fast")
            comp.present(f, policy="full")
            self.assertEqual(comp.partial_count, 0)
            before = len(rec.dpy_area_ops)
            time.sleep(0.18)
            self.assertEqual(len(rec.dpy_area_ops), before)
        finally:
            comp.close()
            ctrl.close()


# ===========================================================================
# Feature #8: USB Protocol Boundaries (5 tests)
# ===========================================================================


class TestT2Feature08ProtocolBoundaries(unittest.TestCase):
    def test_t2_f8_01_wait_display_ready_polls_timeout_and_usb_error(self) -> None:
        rec = helpers.UsbBotRecorder(busy_polls_per_wait=2)
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        try:
            self.assertTrue(ctrl.wait_display_ready(timeout_ms=500, poll_ms=1))
            self.assertEqual(rec.b1_count, 3)

            rec.busy_polls_per_wait = 100
            self.assertFalse(ctrl.wait_display_ready(timeout_ms=25, poll_ms=5))

            rec.busy_polls_per_wait = 0
            rec.inject_error_on_opcode(OP_GET_DPY_STATUS, count=1)
            self.assertFalse(ctrl.wait_display_ready(timeout_ms=200, poll_ms=5))
        finally:
            ctrl.close()

    def test_t2_f8_02_display_region_validation_and_zero_dim_reject(self) -> None:
        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        try:
            for args in ((0, 0, 0, 10, b""), (0, 0, 10, 0, b""), (0, 0, -4, 8, b"")):
                with self.assertRaises(ValueError):
                    ctrl.display_region(*args)
            for args in ((-1, 0, 8, 8), (0, -1, 8, 8), (1916, 0, 8, 8), (0, 1076, 8, 8)):
                with self.assertRaises(ValueError):
                    ctrl.display_region(*args, b"\xff" * 64)
            with self.assertRaises(ValueError):
                ctrl.display_region(0, 0, 8, 8, b"\xff" * 63)
            self.assertEqual(len(rec.commands), 0)
        finally:
            ctrl.close()

    def test_t2_f8_03_load_pixels_and_display_image_bytes_length_mismatch(self) -> None:
        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        try:
            with self.assertRaises(ValueError):
                ctrl.load_pixels(0, 16, 16, b"\xff" * 10)
            with self.assertRaises(ValueError):
                ctrl.display_image_bytes(b"\xff" * 10, 16, 16)
            self.assertEqual(len(rec.commands), 0)
        finally:
            ctrl.close()

    def test_t2_f8_04_usbc_framing_and_coalesced_vs_split_usbs(self) -> None:
        for mode in ("coalesced", "split"):
            rec = helpers.UsbBotRecorder(usbs_mode=mode)
            ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
            ctrl.open()
            try:
                info = ctrl.get_system_info()
                self.assertEqual((info.width, info.height), (PANEL_WIDTH, PANEL_HEIGHT))
                self.assertEqual(info.image_buf_base, IMAGE_BUF_BASE)
                cbw = rec.commands[0].raw_cbw
                self.assertEqual(len(cbw), 31)
                self.assertEqual(cbw[:8], USBC_HEADER)
                ctrl.write_reg(0x18001138, 0x002E0000)
                self.assertEqual(ctrl.read_reg(0x18001138), 0x002E0000)
            finally:
                ctrl.close()

    def test_t2_f8_05_bottom_right_corner_and_bottom_strip_mem_addr(self) -> None:
        rec = helpers.UsbBotRecorder()
        ctrl = Ite8951Controller(usb_dev=helpers.MockLibUsbDevice(rec))
        ctrl.open()
        try:
            ctrl.display_region(1912, 1072, 8, 8, bytes([0x0F]) * 64, waveform=WAVEFORM_DU2)
            ctrl.display_region(0, 1072, 1920, 8, bytes([0xFF]) * (1920 * 8), waveform=WAVEFORM_DU2)
            helpers.assert_r5_protocol_invariants(self, rec, allow_standalone_0xa9=False)
            self.assertEqual(rec.ld_img_ops[0].buf_offset, 1072 * PANEL_WIDTH + 1912)
            self.assertEqual(rec.ld_img_ops[1].buf_offset, 1072 * PANEL_WIDTH)
        finally:
            ctrl.close()


# ===========================================================================
# Feature #9: DOOM Boundaries & Crash Recovery (6 tests)
# ===========================================================================


class TestT2Feature09DoomBoundaries(unittest.TestCase):
    def test_t2_f9_01_out_of_bounds_tone_settings_clamped(self) -> None:
        from yogabook_eink.config import EinkSettings
        from yogabook_eink.doom_eink import EinkDoomGame

        def _render(l_val, c_val):
            EinkSettings(lightness=l_val, contrast=c_val).save()
            init_sig = inspect.signature(EinkDoomGame.__init__)
            kw = {}
            if "lightness" in init_sig.parameters:
                kw["lightness"] = l_val
            if "contrast" in init_sig.parameters:
                kw["contrast"] = c_val
            g = EinkDoomGame(**kw)
            rf_sig = inspect.signature(g.render_frame)
            rkw = {}
            if "lightness" in rf_sig.parameters:
                rkw["lightness"] = l_val
            if "contrast" in rf_sig.parameters:
                rkw["contrast"] = c_val
            return g.render_frame(**rkw)[0]

        with helpers.IsolatedEinkEnvironment():
            f_oob_lo = _render(-99, -99)
            f_min = _render(-3, -1)
            f_oob_hi = _render(99, 99)
            f_max = _render(3, 6)
            self.assertEqual(f_oob_lo.tobytes(), f_min.tobytes())
            self.assertEqual(f_oob_hi.tobytes(), f_max.tobytes())

    def test_t2_f9_02_hinge_deadzone_viewport_and_is_up_ignored(self) -> None:
        from yogabook_eink.doom_eink import run_eink_doom

        with helpers.IsolatedEinkEnvironment() as env:
            stop_ev = threading.Event()
            t = threading.Thread(target=run_eink_doom, args=(stop_ev,), daemon=True)
            t.start()
            try:
                self.assertTrue(
                    helpers.wait_until(lambda: len(env.usb_recorder.dpy_area_ops) >= 2, timeout=3.0)
                )
                env.inject_touch(10, 10, is_down=True)
                env.inject_touch(960, 400, is_down=True)
                env.inject_touch(1785, 950, is_down=False)  # release on exit
                time.sleep(0.15)
                self.assertFalse(stop_ev.is_set())
            finally:
                stop_ev.set()
                t.join(timeout=2.0)

    def test_t2_f9_03_touch_button_exact_perimeter_pixels(self) -> None:
        from yogabook_eink.doom_eink import EinkDoomGame

        game = EinkDoomGame()
        by_name = {b.name: b for b in game.buttons}
        exit_b = by_name["exit"]
        fire_b = by_name["fire"]

        self.assertTrue(exit_b.contains(exit_b.x, exit_b.y))
        self.assertTrue(exit_b.contains(exit_b.x + exit_b.w, exit_b.y + exit_b.h))
        self.assertFalse(exit_b.contains(exit_b.x - 1, exit_b.y))
        self.assertFalse(exit_b.contains(exit_b.x + exit_b.w + 1, exit_b.y + exit_b.h))

        self.assertTrue(fire_b.contains(fire_b.x, fire_b.y))
        self.assertTrue(fire_b.contains(fire_b.x + fire_b.w, fire_b.y + fire_b.h))
        self.assertFalse(fire_b.contains(fire_b.x - 1, fire_b.y))
        self.assertFalse(fire_b.contains(fire_b.x + fire_b.w + 1, fire_b.y))

    def test_t2_f9_04_shoot_beyond_ammo_never_negative(self) -> None:
        from yogabook_eink.doom_eink import EinkDoomGame

        game = EinkDoomGame()
        for _ in range(55):
            game.shoot()
        self.assertEqual(game.ammo, 0)
        game.flash_frames = 0
        game.shoot()
        self.assertEqual(game.ammo, 0)
        self.assertEqual(game.flash_frames, 0)
        frame, _ = game.render_frame()
        helpers.assert_strict_1bit_image(self, frame)

    def test_t2_f9_05_doom_sigkill_triggers_ownership_crash_recovery(self) -> None:
        from yogabook_eink.ownership import OwnershipServer

        with helpers.IsolatedEinkEnvironment() as env:
            srv = OwnershipServer()
            helpers.start_ownership_server(srv)
            proc = None
            try:
                child_env = dict(os.environ)
                child_env["HOME"] = env.home_dir
                child_env["XDG_RUNTIME_DIR"] = env.runtime_dir
                child_env["PYTHONPATH"] = REPO_ROOT
                code = (
                    "import time\n"
                    "from yogabook_eink.ownership import PanelOwnershipLease\n"
                    "with PanelOwnershipLease('doom'):\n"
                    "    print('ACQUIRED', flush=True)\n"
                    "    time.sleep(30)\n"
                )
                proc = subprocess.Popen(
                    [sys.executable, "-c", code],
                    env=child_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                assert proc.stdout is not None
                line = proc.stdout.readline().strip()
                self.assertEqual(line, "ACQUIRED")
                self.assertEqual(srv.active_owner, "doom")
                before_kb = env.usb_recorder.exit_kb_count
                proc.kill()
                proc.wait(timeout=3.0)
                self.assertTrue(
                    helpers.wait_until(
                        lambda: srv.active_owner is None and env.usb_recorder.exit_kb_count > before_kb,
                        timeout=3.0,
                    )
                )
            finally:
                if proc is not None:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait(timeout=2.0)
                    if proc.stdout is not None:
                        proc.stdout.close()
                    if proc.stderr is not None:
                        proc.stderr.close()
                srv.stop()

    def test_t2_f9_06_immediate_stop_cancels_pending_idle_settle(self) -> None:
        from yogabook_eink.config import EinkSettings
        from yogabook_eink.doom_eink import run_eink_doom

        with helpers.IsolatedEinkEnvironment() as env:
            EinkSettings(idle_settle_enabled=True, idle_settle_sec=0.10).save()
            stop_ev = threading.Event()
            t = threading.Thread(target=run_eink_doom, args=(stop_ev,), daemon=True)
            t.start()
            try:
                self.assertTrue(
                    helpers.wait_until(lambda: len(env.usb_recorder.dpy_area_ops) >= 2, timeout=3.0)
                )
                env.inject_touch(275, 890, is_down=True)
                time.sleep(0.03)
                stop_ev.set()
                t.join(timeout=3.0)
                blits_at_exit = len(env.usb_recorder.dpy_area_ops)
                time.sleep(0.18)
                self.assertEqual(len(env.usb_recorder.dpy_area_ops), blits_at_exit)
            finally:
                stop_ev.set()
                t.join(timeout=2.0)


# ===========================================================================
# Feature #10: E-Ink Menu & Daemon Boundaries (6 tests)
# ===========================================================================


class TestT2Feature10EinkMenuBoundaries(unittest.TestCase):
    def test_t2_f10_01_hinge_deadzone_ignored_across_modes(self) -> None:
        from yogabook_eink.daemon import EinkSettingsDaemon

        with helpers.IsolatedEinkEnvironment() as env:
            daemon = EinkSettingsDaemon()
            daemon.listener.start()
            try:
                for px, py in ((10, 10), (50, 50)):
                    env.inject_touch(px, py, is_down=True)
                    time.sleep(0.05)
                self.assertFalse(daemon._menu_active)
                self.assertEqual(len(env.usb_recorder.dpy_area_ops), 0)

                daemon._pad_active = True
                env.inject_touch(10, 10, is_down=True)
                self.assertTrue(daemon._pad_active)

                daemon._pad_active = False
                daemon._menu_active = True
                daemon.screen.page = "settings"
                daemon.screen.render()
                env.inject_touch(10, 10, is_down=True)
                self.assertEqual(len(env.usb_recorder.dpy_area_ops), 0)
            finally:
                daemon.sound.stop()
                daemon.listener.stop()
                daemon.ownership.stop()

    def test_t2_f10_02_stepping_lightness_contrast_stays_in_bounds(self) -> None:
        from yogabook_eink.daemon import EinkSettingsDaemon

        with helpers.IsolatedEinkEnvironment() as env:
            daemon = EinkSettingsDaemon()
            try:
                daemon.screen.page = "settings"
                daemon.screen.render()
                l_widgets = [
                    w
                    for w in daemon.screen.widgets
                    if not w.is_card and not w.is_header_btn and "lightness" in (w.action_id + w.title).lower()
                ]
                c_widgets = [
                    w
                    for w in daemon.screen.widgets
                    if not w.is_card and not w.is_header_btn and "contrast" in (w.action_id + w.title).lower()
                ]
                self.assertGreater(len(l_widgets), 0, "Missing Lightness control")
                self.assertGreater(len(c_widgets), 0, "Missing Contrast control")
                for w in l_widgets:
                    for _ in range(10):
                        daemon._handle_action(w.action_id)
                        self.assertGreaterEqual(daemon.settings.lightness, -3)
                        self.assertLessEqual(daemon.settings.lightness, 3)
                for w in c_widgets:
                    for _ in range(12):
                        daemon._handle_action(w.action_id)
                        self.assertGreaterEqual(daemon.settings.contrast, -1)
                        self.assertLessEqual(daemon.settings.contrast, 6)
                saved = env.read_settings_json()
                self.assertGreaterEqual(saved["lightness"], -3)
                self.assertLessEqual(saved["lightness"], 3)
                self.assertGreaterEqual(saved["contrast"], -1)
                self.assertLessEqual(saved["contrast"], 6)
            finally:
                daemon.sound.stop()
                daemon.listener.stop()
                daemon.ownership.stop()

    def test_t2_f10_03_tap_debounce_and_busy_guard(self) -> None:
        from yogabook_eink.daemon import EinkSettingsDaemon

        with helpers.IsolatedEinkEnvironment() as env:
            daemon = EinkSettingsDaemon()
            daemon.listener.start()
            try:
                daemon._menu_active = True
                daemon.screen.page = "settings"
                daemon.screen.render()
                tw = [w for w in daemon.screen.widgets if w.action_id == "toggle_theme"][0]
                daemon._last_tap_ts = 0.0
                env.inject_touch(tw.x + 50, tw.y + 50, is_down=True)
                # Second tap within 0.45s debounce window
                env.inject_touch(tw.x + 50, tw.y + 50, is_down=True)
                self.assertTrue(helpers.wait_until(lambda: not daemon._busy, timeout=2.0))
                self.assertIs(daemon.settings.dark_theme, True)

                # Busy guard ignores tap
                daemon._busy = True
                daemon._last_tap_ts = 0.0
                env.inject_touch(tw.x + 50, tw.y + 50, is_down=True)
                self.assertIs(daemon.settings.dark_theme, True)
            finally:
                daemon._busy = False
                daemon.sound.stop()
                daemon.listener.stop()
                daemon.ownership.stop()

    def test_t2_f10_04_ownership_suppresses_touch_and_vendor_until_released(self) -> None:
        from yogabook_eink.daemon import EinkSettingsDaemon
        from yogabook_eink.ownership import PanelOwnershipLease

        with helpers.IsolatedEinkEnvironment() as env:
            daemon = EinkSettingsDaemon()
            helpers.start_ownership_server(daemon.ownership)
            daemon.listener.start()
            try:
                with PanelOwnershipLease("doom") as lease:
                    self.assertTrue(helpers.wait_until(lambda: daemon.ownership.active_owner == "doom", timeout=2.0))
                    env.inject_touch(1866, 18, is_down=True)
                    env.inject_vendor(0x10)
                    time.sleep(0.10)
                    self.assertFalse(daemon._menu_active)
                    self.assertEqual(len(env.usb_recorder.dpy_area_ops), 0)
                    lease.mark_clean_exit()

                self.assertTrue(helpers.wait_until(lambda: daemon.ownership.active_owner is None, timeout=2.0))
                daemon._last_tap_ts = 0.0
                env.inject_touch(1866, 18, is_down=True)
                self.assertTrue(helpers.wait_until(lambda: daemon._menu_active, timeout=2.0))
            finally:
                daemon.sound.stop()
                daemon.listener.stop()
                daemon.ownership.stop()

    def test_t2_f10_05_corrupt_json_fallback_and_usb_error_resets_busy(self) -> None:
        from yogabook_eink.daemon import EinkSettingsDaemon

        with helpers.IsolatedEinkEnvironment() as env:
            env.write_settings_json("{bad json")
            daemon = EinkSettingsDaemon()
            try:
                self.assertEqual(daemon.settings.lightness, 0)
                env.usb_recorder.inject_error_on_opcode(0x80, count=1)
                daemon._show_settings_screen("", fast=False)
                self.assertFalse(daemon._busy)
            finally:
                daemon.sound.stop()
                daemon.listener.stop()
                daemon.ownership.stop()

    def test_t2_f10_06_render_widget_patches_bounds_4px_and_1bit(self) -> None:
        from yogabook_eink.eink_menu import EinkSettingsScreen

        with helpers.IsolatedEinkEnvironment():
            screen = EinkSettingsScreen()
            screen.page = "settings"
            screen.render()
            interactive = [w for w in screen.widgets if not w.is_card and not w.is_header_btn]
            self.assertGreater(len(interactive), 0)
            for w in interactive:
                patches = screen.render_widget_patches(w.action_id, status_line="Test")
                self.assertGreater(len(patches), 0)
                for x, y, pw, ph, pixels in patches:
                    self.assertGreaterEqual(x, 0)
                    self.assertGreaterEqual(y, 0)
                    self.assertLessEqual(x + pw, PANEL_WIDTH)
                    self.assertLessEqual(y + ph, PANEL_HEIGHT)
                    self.assertEqual(pw % 4, 0)
                    self.assertEqual(len(pixels), pw * ph)
                    self.assertTrue(set(pixels).issubset({0x0F, 0xFF}))


# ===========================================================================
# Feature #11: GTK3 GUI Boundaries (5 tests)
# ===========================================================================


class TestT2Feature11GtkGuiBoundaries(unittest.TestCase):
    def test_t2_f11_01_corrupt_or_oob_settings_initializes_cleanly(self) -> None:
        with helpers.IsolatedEinkEnvironment() as env:
            env.write_settings_json({"lightness": 99, "contrast": -99, "dither_mode": "Bogus"})
            win, close_fn = helpers.open_isolated_window(env)
            try:
                scl_l = helpers.find_gtk_scale(win, "lightness", -3, 3)
                scl_c = helpers.find_gtk_scale(win, "contrast", -1, 6)
                self.assertLessEqual(scl_l.get_value(), 3)
                self.assertGreaterEqual(scl_c.get_value(), -1)
            finally:
                close_fn()

    def test_t2_f11_02_sliders_clamp_at_extremes(self) -> None:
        with helpers.IsolatedEinkEnvironment() as env:
            win, close_fn = helpers.open_isolated_window(env)
            try:
                scl_l = helpers.find_gtk_scale(win, "lightness", -3, 3)
                scl_c = helpers.find_gtk_scale(win, "contrast", -1, 6)

                scl_l.set_value(-10)
                helpers.pump_gtk_events()
                self.assertEqual(int(round(scl_l.get_value())), -3)
                scl_l.set_value(10)
                helpers.pump_gtk_events()
                self.assertEqual(int(round(scl_l.get_value())), 3)

                scl_c.set_value(-10)
                helpers.pump_gtk_events()
                self.assertEqual(int(round(scl_c.get_value())), -1)
                scl_c.set_value(20)
                helpers.pump_gtk_events()
                self.assertEqual(int(round(scl_c.get_value())), 6)

                saved = env.read_settings_json()
                self.assertEqual(saved["lightness"], 3)
                self.assertEqual(saved["contrast"], 6)
            finally:
                close_fn()

    def test_t2_f11_03_send_image_none_or_corrupt_logs_error_no_usb(self) -> None:
        with helpers.IsolatedEinkEnvironment() as env:
            win, close_fn = helpers.open_isolated_window(env)
            try:
                before = len(env.usb_recorder.dpy_area_ops)
                btn_send = helpers.find_gtk_button(win, "Send Image to E-Ink Display")
                btn_send.clicked()
                helpers.pump_gtk_events()
                self.assertEqual(len(env.usb_recorder.dpy_area_ops), before)
                self.assertFalse(win._busy)

                bad_png = os.path.join(env.tmpdir, "corrupt.png")
                with open(bad_png, "wb") as f:
                    f.write(b"not a png")
                win._selected_image_path = bad_png
                btn_send.clicked()
                self.assertTrue(helpers.wait_until(lambda: not win._busy, timeout=2.0))
                self.assertEqual(len(env.usb_recorder.dpy_area_ops), before)
                txt = win.log_buf.get_text(win.log_buf.get_start_iter(), win.log_buf.get_end_iter(), True)
                self.assertIn("Error", txt)
            finally:
                close_fn()

    def test_t2_f11_04_busy_guard_blocks_concurrent_usb_worker(self) -> None:
        with helpers.IsolatedEinkEnvironment() as env:
            win, close_fn = helpers.open_isolated_window(env)
            try:
                before = len(env.usb_recorder.commands)
                win._busy = True
                win._show_eink_menu_async("ignored")
                helpers.pump_gtk_events()
                self.assertEqual(len(env.usb_recorder.commands), before)
                txt = win.log_buf.get_text(win.log_buf.get_start_iter(), win.log_buf.get_end_iter(), True)
                self.assertIn("Busy", txt)
            finally:
                win._busy = False
                close_fn()

    def test_t2_f11_05_show_menu_preserves_tone_dither_and_destroy_cleans_up(self) -> None:
        from yogabook_eink.config import EinkSettings

        with helpers.IsolatedEinkEnvironment() as env:
            EinkSettings(lightness=-1, contrast=5, dither_mode="Floyd-Steinberg").save()
            win, close_fn = helpers.open_isolated_window(env)
            try:
                win._show_eink_menu_async("Test")
                self.assertTrue(helpers.wait_until(lambda: win._menu_active and not win._busy, timeout=3.0))
                saved = env.read_settings_json()
                self.assertEqual(saved["lightness"], -1)
                self.assertEqual(saved["contrast"], 5)
                self.assertEqual(saved["dither_mode"], "Floyd-Steinberg")

                stop_ev = threading.Event()
                win._activity_stop = stop_ev
                close_fn()
                self.assertTrue(stop_ev.is_set())
                self.assertFalse(win._hid_listener.is_running)
            finally:
                close_fn()


# ===========================================================================
# Feature #12: Matrix Saver Boundaries (5 tests)
# ===========================================================================


class TestT2Feature12MatrixBoundaries(unittest.TestCase):
    def test_t2_f12_01_preset_stop_event_exits_immediately_and_cleanly(self) -> None:
        from yogabook_eink.matrix_saver import run_matrix_screensaver
        from yogabook_eink.ownership import OwnershipServer

        with helpers.IsolatedEinkEnvironment() as env:
            srv = OwnershipServer()
            helpers.start_ownership_server(srv)
            try:
                stop_ev = threading.Event()
                stop_ev.set()
                calls = []
                run_matrix_screensaver(stop_ev, on_frame=lambda n: calls.append(n))
                self.assertEqual(calls, [])
                self.assertGreaterEqual(env.usb_recorder.exit_kb_count, 1)
                self.assertTrue(helpers.wait_until(lambda: srv.active_owner is None, timeout=2.0))
                self.assertTrue(all(not l.is_running for l in env.listeners))
            finally:
                srv.stop()

    def test_t2_f12_02_restore_kb_on_exit_false_skips_exit_to_keyboard(self) -> None:
        from yogabook_eink.matrix_saver import run_matrix_screensaver

        with helpers.IsolatedEinkEnvironment() as env:
            stop_ev = threading.Event()
            run_matrix_screensaver(
                stop_ev,
                restore_kb_on_exit=False,
                on_frame=lambda n: stop_ev.set() if n >= 2 else None,
            )
            self.assertEqual(env.usb_recorder.exit_kb_count, 0)
            self.assertTrue(all(not l.is_running for l in env.listeners))

    def test_t2_f12_03_touch_up_ignored_touch_down_stops(self) -> None:
        from yogabook_eink.matrix_saver import run_matrix_screensaver

        with helpers.IsolatedEinkEnvironment() as env:
            stop_ev = threading.Event()
            frames = []
            t = threading.Thread(
                target=run_matrix_screensaver,
                kwargs={"stop_event": stop_ev, "on_frame": lambda n: frames.append(n)},
                daemon=True,
            )
            t.start()
            try:
                self.assertTrue(helpers.wait_until(lambda: len(frames) >= 2, timeout=3.0))
                env.inject_touch(960, 1014, is_down=False)  # release ignored
                n_before = len(frames)
                self.assertTrue(helpers.wait_until(lambda: len(frames) > n_before, timeout=2.0))
                self.assertFalse(stop_ev.is_set())

                env.inject_touch(960, 1014, is_down=True)
                t.join(timeout=3.0)
                self.assertFalse(t.is_alive())
            finally:
                stop_ev.set()
                t.join(timeout=2.0)

    def test_t2_f12_04_all_48_columns_50_frames_strict_1bit(self) -> None:
        from yogabook_eink.matrix_saver import MatrixRainRenderer

        r = MatrixRainRenderer(dark_bg=True, cols=48, rows=24)
        for _ in range(50):
            frame = r.step_frame(cols_per_tick=48)
            helpers.assert_strict_1bit_image(self, frame)

    def test_t2_f12_05_matrix_sigkill_triggers_ownership_crash_recovery(self) -> None:
        from yogabook_eink.ownership import OwnershipServer

        with helpers.IsolatedEinkEnvironment() as env:
            srv = OwnershipServer()
            helpers.start_ownership_server(srv)
            proc = None
            try:
                child_env = dict(os.environ)
                child_env["HOME"] = env.home_dir
                child_env["XDG_RUNTIME_DIR"] = env.runtime_dir
                child_env["PYTHONPATH"] = REPO_ROOT
                code = (
                    "import time\n"
                    "from yogabook_eink.ownership import PanelOwnershipLease\n"
                    "with PanelOwnershipLease('matrix'):\n"
                    "    print('ACQUIRED', flush=True)\n"
                    "    time.sleep(30)\n"
                )
                proc = subprocess.Popen(
                    [sys.executable, "-c", code],
                    env=child_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                assert proc.stdout is not None
                self.assertEqual(proc.stdout.readline().strip(), "ACQUIRED")
                self.assertEqual(srv.active_owner, "matrix")
                before_kb = env.usb_recorder.exit_kb_count
                proc.kill()
                proc.wait(timeout=3.0)
                self.assertTrue(
                    helpers.wait_until(
                        lambda: srv.active_owner is None and env.usb_recorder.exit_kb_count > before_kb,
                        timeout=3.0,
                    )
                )
            finally:
                if proc is not None:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait(timeout=2.0)
                    if proc.stdout is not None:
                        proc.stdout.close()
                    if proc.stderr is not None:
                        proc.stderr.close()
                srv.stop()


if __name__ == "__main__":
    unittest.main()
