"""Unit tests for Glider 256-byte Parabolic Tone LUT and EinkSettings persistence."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from PIL import Image

from yogabook_eink import config
from yogabook_eink.config import EinkSettings
from yogabook_eink.protocol import WAVEFORM_DU2, WAVEFORM_GC16, WAVEFORM_GL16
from yogabook_eink.tone_lut import (
    CONTRAST_LUT,
    CONTRAST_MAX,
    CONTRAST_MIN,
    LIGHTNESS_LUT,
    LIGHTNESS_MAX,
    LIGHTNESS_MIN,
    TONE_LUT_SIZE,
    apply_tone_lut,
    build_tone_lut,
    clamp_contrast,
    clamp_lightness,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
GLIDER_USER_DIR = os.path.join(REPO_ROOT, "references", "Glider", "fw", "User")
GLIDER_TONE_C = os.path.join(GLIDER_USER_DIR, "tone_lut.c")
GLIDER_TONE_TEST_C = os.path.join(GLIDER_USER_DIR, "tests", "test_tone_lut.c")


def _find_cc() -> str | None:
    return shutil.which("gcc") or shutil.which("cc")


class TestToneLut(unittest.TestCase):
    def test_constants(self) -> None:
        self.assertEqual(TONE_LUT_SIZE, 256)
        self.assertEqual((LIGHTNESS_MIN, LIGHTNESS_MAX), (-3, 3))
        self.assertEqual((CONTRAST_MIN, CONTRAST_MAX), (-1, 6))
        self.assertEqual(len(LIGHTNESS_LUT), 7)
        self.assertEqual(len(CONTRAST_LUT), 8)

    def test_clamping_and_type_validation(self) -> None:
        self.assertEqual(clamp_lightness(-100), -3)
        self.assertEqual(clamp_lightness(-3), -3)
        self.assertEqual(clamp_lightness(0), 0)
        self.assertEqual(clamp_lightness(3), 3)
        self.assertEqual(clamp_lightness(99), 3)

        self.assertEqual(clamp_contrast(-50), -1)
        self.assertEqual(clamp_contrast(-1), -1)
        self.assertEqual(clamp_contrast(0), 0)
        self.assertEqual(clamp_contrast(6), 6)
        self.assertEqual(clamp_contrast(50), 6)

        for bad in (True, False, 1.5, "0", None):
            with self.assertRaises(TypeError):
                clamp_lightness(bad)  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                clamp_contrast(bad)  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                build_tone_lut(bad, 0)  # type: ignore[arg-type]

        self.assertEqual(build_tone_lut(-10, -5), build_tone_lut(-3, -1))
        self.assertEqual(build_tone_lut(10, 99), build_tone_lut(3, 6))

    def test_identity_endpoints_and_monotonicity(self) -> None:
        ident = build_tone_lut(0, 0)
        self.assertIsInstance(ident, bytes)
        self.assertEqual(len(ident), 256)
        self.assertEqual(ident, bytes(range(256)))

        for l in range(LIGHTNESS_MIN, LIGHTNESS_MAX + 1):
            for c in range(CONTRAST_MIN, CONTRAST_MAX + 1):
                lut = build_tone_lut(l, c)
                self.assertIsInstance(lut, bytes)
                self.assertEqual(len(lut), 256)
                for i in range(255):
                    self.assertLessEqual(lut[i], lut[i + 1], f"Non-monotonic at l={l}, c={c}, i={i}")
                if c == 0:
                    self.assertEqual(lut[0], 0)
                    self.assertEqual(lut[255], 255)

    def test_float32_vs_float64_regression_points(self) -> None:
        lut_c_neg1 = build_tone_lut(0, -1)
        self.assertEqual(lut_c_neg1[45], 61)
        self.assertEqual(lut_c_neg1[50], 65)
        self.assertEqual(lut_c_neg1[155], 150)

        lut_c_5 = build_tone_lut(0, 5)
        self.assertEqual(lut_c_5[117], 2)
        self.assertEqual(lut_c_5[120], 38)
        self.assertEqual(lut_c_5[121], 50)
        self.assertEqual(lut_c_5[128], 134)

    def test_glider_c_reference_parity_all_entries(self) -> None:
        cc = _find_cc()
        if cc is None:
            self.skipTest("No C compiler (gcc/cc) available")

        with tempfile.TemporaryDirectory() as td:
            # 1. Run Glider's own C unit test executable.
            test_exe = os.path.join(td, "test_tone_lut")
            subprocess.run(
                [cc, "-O2", "-Wall", "-Wextra", f"-I{GLIDER_USER_DIR}", GLIDER_TONE_C, GLIDER_TONE_TEST_C, "-o", test_exe],
                check=True,
            )
            subprocess.run([test_exe], check=True)

            # 2. Compile shared library and compare all 56 * 256 entries + OOB inputs byte-for-byte.
            so_path = os.path.join(td, "libtone_lut.so")
            subprocess.run(
                [cc, "-O2", "-shared", "-fPIC", f"-I{GLIDER_USER_DIR}", GLIDER_TONE_C, "-o", so_path],
                check=True,
            )
            lib = ctypes.CDLL(so_path)
            lib.tone_lut_build_y.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint8)]
            lib.tone_lut_build_y.restype = None

            buf = (ctypes.c_uint8 * 256)()
            for l in range(-5, 6):
                for c in range(-3, 9):
                    lib.tone_lut_build_y(l, c, buf)
                    c_bytes = bytes(buf)
                    py_bytes = build_tone_lut(l, c)
                    self.assertEqual(py_bytes, c_bytes, f"Mismatch at lightness={l}, contrast={c}")

    def test_apply_tone_lut(self) -> None:
        ramp = Image.new("L", (256, 4))
        ramp.putdata(list(range(256)) * 4)
        orig_bytes = ramp.tobytes()

        # Identity preserves pixels and returns a separate L image
        ident_out = apply_tone_lut(ramp, 0, 0)
        self.assertEqual(ident_out.mode, "L")
        self.assertEqual(ident_out.size, (256, 4))
        self.assertEqual(ident_out.tobytes(), orig_bytes)
        self.assertIsNot(ident_out, ramp)

        # RGB input converts cleanly to L
        rgb_img = Image.new("RGB", (8, 8), (120, 120, 120))
        rgb_out = apply_tone_lut(rgb_img, 0, 0)
        self.assertEqual(rgb_out.mode, "L")
        self.assertEqual(rgb_out.size, (8, 8))
        self.assertEqual(rgb_out.getpixel((0, 0)), 120)

        # Non-zero parameters match build_tone_lut byte-for-byte without mutating input
        lut = build_tone_lut(2, 4)
        toned = apply_tone_lut(ramp, 2, 4)
        self.assertEqual(toned.mode, "L")
        self.assertEqual(toned.size, (256, 4))
        self.assertEqual(toned.tobytes()[:256], lut)
        self.assertEqual(ramp.tobytes(), orig_bytes)


class TestEinkSettingsConfig(unittest.TestCase):
    def test_roundtrip_atomic_save_and_missing_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target = os.path.join(td, "nested", "dir", "settings.json")
            settings = EinkSettings(
                key_down_sound=True,
                key_up_sound=True,
                sound_profile="typewriter",
                dark_theme=True,
                waveform=WAVEFORM_GL16,
                rotate_180=True,
                swap_axes=True,
                flip_x=False,
                flip_y=True,
                lightness=2,
                contrast=4,
                dither_mode="Atkinson",
                idle_settle_enabled=False,
                idle_settle_sec=3.5,
            )
            settings.save(target)
            self.assertTrue(os.path.exists(target))
            # No leftover .tmp files
            leftovers = [n for n in os.listdir(os.path.dirname(target)) if n.endswith(".tmp")]
            self.assertEqual(leftovers, [])

            loaded = EinkSettings.load(target)
            self.assertEqual(loaded, settings)

    def test_zero_arg_load_and_save_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target = os.path.join(td, "settings.json")
            with mock.patch.object(config, "CONFIG_PATH", target):
                s = EinkSettings(lightness=-2, contrast=3, dither_mode="fs")
                s.save()
                loaded = EinkSettings.load()
                self.assertEqual(loaded.lightness, -2)
                self.assertEqual(loaded.contrast, 3)
                self.assertEqual(loaded.dither_mode, "Floyd-Steinberg")

    def test_legacy_nine_key_json_compatibility_and_unknown_keys(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target = os.path.join(td, "settings.json")
            legacy = {
                "key_down_sound": False,
                "key_up_sound": True,
                "sound_profile": "crisp",
                "dark_theme": False,
                "waveform": WAVEFORM_DU2,
                "rotate_180": True,
                "swap_axes": False,
                "flip_x": True,
                "flip_y": False,
                "unexpected_future_key": 42,
            }
            with open(target, "w", encoding="utf-8") as f:
                json.dump(legacy, f)

            loaded = EinkSettings.load(target)
            self.assertFalse(loaded.key_down_sound)
            self.assertTrue(loaded.key_up_sound)
            self.assertEqual(loaded.sound_profile, "crisp")
            self.assertEqual(loaded.waveform, WAVEFORM_DU2)
            self.assertTrue(loaded.rotate_180)
            self.assertTrue(loaded.flip_x)
            self.assertEqual(loaded.lightness, 0)
            self.assertEqual(loaded.contrast, 0)
            self.assertEqual(loaded.dither_mode, "Grayscale 16-level")
            self.assertTrue(loaded.idle_settle_enabled)
            self.assertEqual(loaded.idle_settle_sec, 2.0)

    def test_corrupt_and_non_dict_json_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target = os.path.join(td, "settings.json")
            self.assertEqual(EinkSettings.load(target), EinkSettings())

            for payload in ("{not valid json", "[1, 2, 3]", "123", "null", '"hello"'):
                with open(target, "w", encoding="utf-8") as f:
                    f.write(payload)
                self.assertEqual(EinkSettings.load(target), EinkSettings())

    def test_clamping_and_per_field_invalid_type_fallbacks(self) -> None:
        # Constructor clamping
        s = EinkSettings(lightness=-99, contrast=99, idle_settle_sec=999.0, dither_mode="bayer-4x4")
        self.assertEqual(s.lightness, -3)
        self.assertEqual(s.contrast, 6)
        self.assertEqual(s.idle_settle_sec, 60.0)
        self.assertEqual(s.dither_mode, "4x4 Bayer")

        s2 = EinkSettings(lightness=99, contrast=-5, idle_settle_sec=-5.0)
        self.assertEqual(s2.lightness, 3)
        self.assertEqual(s2.contrast, -1)
        self.assertEqual(s2.idle_settle_sec, 0.05)

        # Save normalizes mutated attributes
        with tempfile.TemporaryDirectory() as td:
            target = os.path.join(td, "settings.json")
            s.lightness = 42  # mutated after init
            s.idle_settle_enabled = "false"  # type: ignore[assignment]
            s.save(target)
            with open(target, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["lightness"], 3)
            self.assertIs(saved["idle_settle_enabled"], True)

            # Per-field invalid types preserve other valid fields
            mixed = {
                "key_down_sound": "true",
                "key_up_sound": 1,
                "sound_profile": "invalid-profile",
                "dark_theme": True,
                "waveform": True,
                "lightness": True,
                "contrast": "high",
                "dither_mode": "unknown-mode",
                "idle_settle_enabled": "false",
                "idle_settle_sec": True,
            }
            with open(target, "w", encoding="utf-8") as f:
                json.dump(mixed, f)
            loaded = EinkSettings.load(target)
            self.assertIs(loaded.key_down_sound, False)
            self.assertIs(loaded.key_up_sound, False)
            self.assertEqual(loaded.sound_profile, "soft")
            self.assertIs(loaded.dark_theme, True)  # valid field preserved
            self.assertEqual(loaded.waveform, WAVEFORM_GC16)
            self.assertEqual(loaded.lightness, 0)
            self.assertEqual(loaded.contrast, 0)
            self.assertEqual(loaded.dither_mode, "Grayscale 16-level")
            self.assertIs(loaded.idle_settle_enabled, True)
            self.assertEqual(loaded.idle_settle_sec, 2.0)

            # NaN/Inf check via normalize()
            for bad_sec in (float("nan"), float("inf"), float("-inf"), "abc", None):
                inst = EinkSettings(idle_settle_sec=bad_sec)  # type: ignore[arg-type]
                self.assertEqual(inst.idle_settle_sec, 2.0)

    def test_huge_json_integer_idle_settle_and_atomic_replace_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target = os.path.join(td, "settings.json")
            # Huge integer (10**400) that overflows float() must fall back to default 2.0
            with open(target, "w", encoding="utf-8") as f:
                f.write('{"idle_settle_sec": ' + ("1" + "0" * 400) + ', "lightness": 2}')
            loaded = EinkSettings.load(target)
            self.assertEqual(loaded.idle_settle_sec, 2.0)
            self.assertEqual(loaded.lightness, 2)

            # Atomic rollback on os.replace failure preserves previous file and leaves no .tmp files
            initial = EinkSettings(lightness=1, contrast=3, dither_mode="4x4 Bayer")
            initial.save(target)
            with open(target, "rb") as f:
                original_bytes = f.read()

            updated = EinkSettings(lightness=-3, contrast=6, dither_mode="Atkinson")
            with mock.patch("yogabook_eink.config.os.replace", side_effect=OSError("simulated replace error")):
                with self.assertRaises(OSError):
                    updated.save(target)

            with open(target, "rb") as f:
                self.assertEqual(f.read(), original_bytes)
            leftovers = [n for n in os.listdir(td) if n.endswith(".tmp")]
            self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
