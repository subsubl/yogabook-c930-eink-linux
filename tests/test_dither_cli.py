"""Unit and CLI tests for epdoptimize-style dithering, image_render pipeline, and yogabook-eink-cli."""

from __future__ import annotations

from importlib.machinery import SourceFileLoader
import importlib.util
import math
import os
import subprocess
import sys
import tempfile
import time
import unittest

from PIL import Image

from yogabook_eink.image_render import (
    DITHER_MODE_ALIASES,
    DITHER_MODES,
    _generate_bayer_index,
    dither_image,
    generate_template,
    normalize_dither_mode,
    pack_eink_grayscale,
    pack_eink_region,
    prepare_image_file,
)
from yogabook_eink.protocol import PANEL_HEIGHT, PANEL_WIDTH, WAVEFORM_GC16, WAVEFORM_GL16

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CLI_PATH = os.path.join(REPO_ROOT, "yogabook-eink-cli")

LEVELS_16_SET = frozenset(k * 17 for k in range(16))
LEVELS_2_SET = frozenset({0, 255})
PACKED_16_SET = frozenset(((k * 17) & 0xF0) | 0x0F for k in range(16))
PACKED_2_SET = frozenset({0x0F, 0xFF})


def _load_cli_module():
    loader = SourceFileLoader("yogabook_eink_cli_mod", CLI_PATH)
    spec = importlib.util.spec_from_loader("yogabook_eink_cli_mod", loader)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


class FakeController:
    instances: list["FakeController"] = []
    init_count: int = 0

    def __init__(self) -> None:
        type(self).init_count += 1
        type(self).instances.append(self)
        self.calls: list[tuple] = []
        self.last_packed: bytes | None = None

    def __enter__(self) -> "FakeController":
        self.calls.append(("enter",))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.calls.append(("exit",))

    def enable_draw_mode(self) -> int:
        self.calls.append(("enable_draw_mode",))
        return 0

    def display_image_bytes(self, packed: bytes, waveform: int = WAVEFORM_GC16) -> None:
        self.last_packed = packed
        self.calls.append(("display_image_bytes", len(packed), frozenset(packed), waveform))


class TestDitherModesAndImageRender(unittest.TestCase):
    def test_mode_registry_and_normalization(self) -> None:
        self.assertEqual(
            DITHER_MODES,
            ("Grayscale 16-level", "4x4 Bayer", "8x8 Bayer", "Floyd-Steinberg", "Atkinson"),
        )
        for alias, canonical in DITHER_MODE_ALIASES.items():
            self.assertIn(canonical, DITHER_MODES)
            self.assertEqual(normalize_dither_mode(alias), canonical)
            self.assertEqual(normalize_dither_mode(alias.upper()), canonical)
            self.assertEqual(normalize_dither_mode(f"  {alias.replace('-', '_')}  "), canonical)

        for bad in ("", "   ", "ordered", "bayer-16x16", None, 123):
            with self.assertRaises(ValueError):
                normalize_dither_mode(bad)  # type: ignore[arg-type]

    def test_invalid_levels_and_zero_dimensions(self) -> None:
        img = Image.new("L", (4, 4), 128)
        for bad_levels in (0, 1, -2, 257, True, False, "2"):
            with self.assertRaises(ValueError):
                dither_image(img, mode="4x4 Bayer", levels=bad_levels)  # type: ignore[arg-type]

        for w, h in ((0, 10), (10, 0), (0, 0)):
            empty = Image.new("L", (w, h))
            for mode in DITHER_MODES:
                out = dither_image(empty, mode=mode)
                self.assertEqual(out.size, (w, h))
                self.assertEqual(out.mode, "L")

    def test_grayscale_16_level_exact_mapping(self) -> None:
        ramp = Image.new("L", (256, 1))
        ramp.putdata(list(range(256)))
        out = dither_image(ramp, mode="Grayscale 16-level")
        self.assertEqual(out.mode, "L")
        self.assertEqual(out.size, (256, 1))
        pixels = list(out.getdata())
        expected = [((v + 8) // 17) * 17 for v in range(256)]
        self.assertEqual(pixels, expected)
        self.assertTrue(set(pixels).issubset(LEVELS_16_SET))

        out2 = dither_image(ramp, mode="Grayscale 16-level", levels=2)
        self.assertEqual(list(out2.getdata()), [255 if v >= 128 else 0 for v in range(256)])

    def test_bayer_4x4_and_8x8_exact_formulas(self) -> None:
        ranks4 = _generate_bayer_index(4)
        self.assertEqual(
            ranks4,
            (
                (0, 8, 2, 10),
                (12, 4, 14, 6),
                (3, 11, 1, 9),
                (15, 7, 13, 5),
            ),
        )
        ranks8 = _generate_bayer_index(8)
        self.assertEqual(len(ranks8), 8)
        self.assertEqual(sorted(v for row in ranks8 for v in row), list(range(64)))

        # Test all 256 gray levels tiled across 8x8 for both 4x4 and 8x8 Bayer at levels=2 and levels=16
        for mode, size, ranks in (("4x4 Bayer", 4, ranks4), ("8x8 Bayer", 8, ranks8)):
            n = size * size
            for v in (0, 1, 8, 17, 64, 127, 128, 200, 254, 255):
                src = Image.new("L", (size, size), v)
                out2 = dither_image(src, mode=mode, levels=2)
                self.assertEqual(out2.mode, "L")
                self.assertTrue(set(out2.getdata()).issubset(LEVELS_2_SET))
                for y in range(size):
                    for x in range(size):
                        thresh = math.floor((ranks[y][x] + 0.5) / n * 256)
                        self.assertEqual(out2.getpixel((x, y)), 255 if v >= thresh else 0)

                out16 = dither_image(src, mode=mode, levels=16)
                self.assertEqual(out16.mode, "L")
                self.assertTrue(set(out16.getdata()).issubset(LEVELS_16_SET))
                k = v // 17
                rem = v - k * 17
                for y in range(size):
                    for x in range(size):
                        if k >= 15:
                            expected_px = 255
                        else:
                            inc = 1 if rem * (2 * n) >= (2 * ranks[y][x] + 1) * 17 else 0
                            expected_px = (k + inc) * 17
                        self.assertEqual(out16.getpixel((x, y)), expected_px)

            # Exact multiples of 17 must have zero spatial jitter in levels=16
            for k in range(16):
                exact = Image.new("L", (size * 2, size * 2), k * 17)
                out_exact = dither_image(exact, mode=mode, levels=16)
                self.assertEqual(set(out_exact.getdata()), {k * 17})

        # 50% gray tile: both 4x4 and 8x8 stay near 50% brightness; non-50% gray (e.g. 100) distinguishes 4x4 vs 8x8
        gray128 = Image.new("L", (16, 16), 128)
        b4 = dither_image(gray128, mode="4x4 Bayer", levels=2)
        b8 = dither_image(gray128, mode="8x8 Bayer", levels=2)
        self.assertAlmostEqual(sum(b4.getdata()) / 256, 127.5, delta=16.0)
        self.assertAlmostEqual(sum(b8.getdata()) / 256, 127.5, delta=16.0)

        gray100 = Image.new("L", (16, 16), 100)
        b4_100 = dither_image(gray100, mode="4x4 Bayer", levels=2)
        b8_100 = dither_image(gray100, mode="8x8 Bayer", levels=2)
        self.assertNotEqual(b4_100.tobytes(), b8_100.tobytes())

    def test_all_five_modes_levels_edge_sizes_and_rgb_input(self) -> None:
        edge_sizes = ((1, 1), (2, 2), (1, 5), (5, 1), (3, 7), (32, 24))
        for mode in DITHER_MODES:
            for w, h in edge_sizes:
                grad = Image.new("L", (w, h))
                grad.putdata([((x * 37 + y * 53) % 256) for y in range(h) for x in range(w)])
                rgb = grad.convert("RGB")

                # Default levels
                out_def = dither_image(rgb, mode=mode)
                self.assertEqual(out_def.mode, "L")
                self.assertEqual(out_def.size, (w, h))
                allowed_def = LEVELS_16_SET if mode == "Grayscale 16-level" else LEVELS_2_SET
                self.assertTrue(set(out_def.getdata()).issubset(allowed_def))

                # Explicit levels=2 and levels=16
                out2 = dither_image(grad, mode=mode, levels=2)
                self.assertTrue(set(out2.getdata()).issubset(LEVELS_2_SET))
                out16 = dither_image(grad, mode=mode, levels=16)
                self.assertTrue(set(out16.getdata()).issubset(LEVELS_16_SET))

                # Arbitrary levels=5
                pal5 = {round(k * 255 / 4) for k in range(5)}
                out5 = dither_image(grad, mode=mode, levels=5)
                self.assertTrue(set(out5.getdata()).issubset(pal5))

                # Solid 0 and solid 255 stay pure
                self.assertEqual(set(dither_image(Image.new("L", (w, h), 0), mode=mode).getdata()), {0})
                self.assertEqual(set(dither_image(Image.new("L", (w, h), 255), mode=mode).getdata()), {255})

    def test_error_diffusion_serpentine_and_atkinson_vs_floyd_steinberg(self) -> None:
        grad = Image.new("L", (64, 32))
        grad.putdata([int(x * 255 / 63) for _ in range(32) for x in range(64)])

        for mode in ("Floyd-Steinberg", "Atkinson"):
            for levels, allowed in ((2, LEVELS_2_SET), (16, LEVELS_16_SET)):
                out_std = dither_image(grad, mode=mode, levels=levels, serpentine=False)
                out_serp = dither_image(grad, mode=mode, levels=levels, serpentine=True)
                self.assertEqual(out_std.size, (64, 32))
                self.assertEqual(out_serp.size, (64, 32))
                self.assertTrue(set(out_std.getdata()).issubset(allowed))
                self.assertTrue(set(out_serp.getdata()).issubset(allowed))
                self.assertNotEqual(out_std.tobytes(), out_serp.tobytes())

        # Atkinson diffuses only 6/8 of residual error, pushing highlights brighter than Floyd-Steinberg
        bright = Image.new("L", (40, 40), 220)
        fs_mean = sum(dither_image(bright, mode="Floyd-Steinberg", levels=2).getdata()) / 1600
        atk_mean = sum(dither_image(bright, mode="Atkinson", levels=2).getdata()) / 1600
        self.assertGreater(atk_mean, fs_mean)

    def test_pillow_accelerated_modes_performance_1080p(self) -> None:
        full_img = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 128)
        for mode in ("Grayscale 16-level", "4x4 Bayer", "8x8 Bayer", "Floyd-Steinberg"):
            t0 = time.monotonic()
            out = dither_image(full_img, mode=mode)
            elapsed = time.monotonic() - t0
            self.assertEqual(out.size, (PANEL_WIDTH, PANEL_HEIGHT))
            self.assertLess(elapsed, 2.0, f"{mode} took too long at 1080p: {elapsed:.3f}s")

    def test_prepare_image_file_pipeline_and_backward_compat(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            # 1. Grayscale gradient PNG
            grad_path = os.path.join(td, "grad.png")
            grad = Image.new("L", (256, 128))
            grad.putdata([x for _ in range(128) for x in range(256)])
            grad.save(grad_path)

            # Default: Grayscale 16-level
            img_def, packed_def = prepare_image_file(grad_path)
            self.assertEqual(img_def.size, (PANEL_WIDTH, PANEL_HEIGHT))
            self.assertEqual(img_def.mode, "L")
            self.assertEqual(len(packed_def), PANEL_WIDTH * PANEL_HEIGHT)
            self.assertTrue(set(img_def.getdata()).issubset(LEVELS_16_SET))
            self.assertTrue(set(packed_def).issubset(PACKED_16_SET))

            # Legacy dither_1bit=True -> Floyd-Steinberg
            img_1bit, packed_1bit = prepare_image_file(grad_path, dither_1bit=True)
            self.assertTrue(set(img_1bit.getdata()).issubset(LEVELS_2_SET))
            self.assertTrue(set(packed_1bit).issubset(PACKED_2_SET))

            # Explicit dither_mode overrides dither_1bit=True
            img_override, packed_override = prepare_image_file(
                grad_path, dither_1bit=True, dither_mode="grayscale-16"
            )
            self.assertTrue(set(img_override.getdata()).issubset(LEVELS_16_SET))
            self.assertEqual(packed_override, packed_def)

            # All 5 modes through prepare_image_file
            for mode in DITHER_MODES:
                img_m, packed_m = prepare_image_file(grad_path, fit_mode="cover", dither_mode=mode)
                self.assertEqual(img_m.size, (PANEL_WIDTH, PANEL_HEIGHT))
                self.assertEqual(len(packed_m), PANEL_WIDTH * PANEL_HEIGHT)

            # Lightness shifts brightness before dithering
            img_dark, _ = prepare_image_file(grad_path, fit_mode="cover", lightness=-3)
            img_bright, _ = prepare_image_file(grad_path, fit_mode="cover", lightness=3)
            self.assertGreater(sum(img_bright.getdata()), sum(img_dark.getdata()))

            # Contrast=+6 thresholds midtones sharply
            img_hc, _ = prepare_image_file(grad_path, fit_mode="cover", contrast=6)
            self.assertTrue(set(img_hc.getdata()).issubset(LEVELS_2_SET))

            # Invert + rotate_180
            img_inv_rot, _ = prepare_image_file(grad_path, fit_mode="cover", invert=True, rotate_180=True)
            self.assertEqual(img_inv_rot.size, (PANEL_WIDTH, PANEL_HEIGHT))

            # Transparent RGBA PNG composites onto paper-white (255), not black (0)
            rgba_path = os.path.join(td, "alpha.png")
            rgba = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
            rgba.save(rgba_path)
            img_alpha, _ = prepare_image_file(rgba_path)
            self.assertEqual(set(img_alpha.getdata()), {255})

        # Backward compatibility of pack_eink_grayscale, pack_eink_region, generate_template
        tpl_img, tpl_packed = generate_template("ruled")
        self.assertEqual(tpl_img.size, (PANEL_WIDTH, PANEL_HEIGHT))
        self.assertEqual(len(tpl_packed), PANEL_WIDTH * PANEL_HEIGHT)
        self.assertEqual(len(pack_eink_grayscale(Image.new("L", (100, 100), 170))), PANEL_WIDTH * PANEL_HEIGHT)
        self.assertEqual(pack_eink_region(tpl_img, 0, 0, 16, 8), b"".join([bytes([((p & 0xF0) | 0x0F)]) for p in tpl_img.crop((0, 0, 16, 8)).getdata()]))

    def test_prepare_image_file_trns_colorkey_rgb_and_l(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            # Pure transparent RGB PNG with tRNS (0, 0, 0) composites onto paper-white (255)
            rgb_pure = os.path.join(td, "rgb_trns_pure.png")
            Image.new("RGB", (64, 64), (0, 0, 0)).save(rgb_pure, transparency=(0, 0, 0))
            img_rgb_pure, _ = prepare_image_file(rgb_pure, fit_mode="cover")
            self.assertEqual(set(img_rgb_pure.getdata()), {255})

            # Mixed RGB PNG with tRNS (0, 0, 0): left half transparent -> 255, right half opaque (64, 64, 64) -> dark
            rgb_mixed = os.path.join(td, "rgb_trns_mixed.png")
            rgb_src = Image.new("RGB", (64, 64), (0, 0, 0))
            rgb_src.paste((64, 64, 64), (32, 0, 64, 64))
            rgb_src.save(rgb_mixed, transparency=(0, 0, 0))
            img_rgb_mixed, _ = prepare_image_file(rgb_mixed, fit_mode="cover")
            self.assertEqual(img_rgb_mixed.getpixel((0, 0)), 255)
            self.assertLess(img_rgb_mixed.getpixel((PANEL_WIDTH - 1, 0)), 128)

            # Pure transparent L PNG with tRNS 0 composites onto paper-white (255)
            l_pure = os.path.join(td, "l_trns_pure.png")
            Image.new("L", (64, 64), 0).save(l_pure, transparency=0)
            img_l_pure, _ = prepare_image_file(l_pure, fit_mode="cover")
            self.assertEqual(set(img_l_pure.getdata()), {255})

            # Mixed L PNG with tRNS 0: left half transparent -> 255, right half opaque 64 -> dark
            l_mixed = os.path.join(td, "l_trns_mixed.png")
            l_src = Image.new("L", (64, 64), 0)
            l_src.paste(64, (32, 0, 64, 64))
            l_src.save(l_mixed, transparency=0)
            img_l_mixed, _ = prepare_image_file(l_mixed, fit_mode="cover")
            self.assertEqual(img_l_mixed.getpixel((0, 0)), 255)
            self.assertLess(img_l_mixed.getpixel((PANEL_WIDTH - 1, 0)), 128)

    def test_prepare_image_file_extreme_aspect_ratios_contain(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            wide_path = os.path.join(td, "wide.png")
            Image.new("L", (10000, 1), 0).save(wide_path)
            img_wide, packed_wide = prepare_image_file(wide_path, fit_mode="contain")
            self.assertEqual(img_wide.mode, "L")
            self.assertEqual(img_wide.size, (PANEL_WIDTH, PANEL_HEIGHT))
            self.assertEqual(len(packed_wide), PANEL_WIDTH * PANEL_HEIGHT)
            self.assertEqual(img_wide.getpixel((0, 0)), 255)
            self.assertEqual(img_wide.getpixel((0, PANEL_HEIGHT - 1)), 255)
            self.assertEqual(img_wide.getpixel((PANEL_WIDTH // 2, PANEL_HEIGHT // 2)), 0)

            tall_path = os.path.join(td, "tall.png")
            Image.new("L", (1, 10000), 0).save(tall_path)
            img_tall, packed_tall = prepare_image_file(tall_path, fit_mode="contain")
            self.assertEqual(img_tall.mode, "L")
            self.assertEqual(img_tall.size, (PANEL_WIDTH, PANEL_HEIGHT))
            self.assertEqual(len(packed_tall), PANEL_WIDTH * PANEL_HEIGHT)
            self.assertEqual(img_tall.getpixel((0, 0)), 255)
            self.assertEqual(img_tall.getpixel((PANEL_WIDTH - 1, 0)), 255)
            self.assertEqual(img_tall.getpixel((PANEL_WIDTH // 2, PANEL_HEIGHT // 2)), 0)


class TestCliDitherAndTone(unittest.TestCase):
    def test_cli_help_and_invalid_arguments_subprocess(self) -> None:
        res_help = subprocess.run(
            [sys.executable, CLI_PATH, "image", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res_help.returncode, 0)
        for opt in ("--dither-mode", "--dither", "--lightness", "--contrast"):
            self.assertIn(opt, res_help.stdout)

        with tempfile.TemporaryDirectory() as td:
            dummy = os.path.join(td, "dummy.png")
            Image.new("L", (8, 8), 128).save(dummy)

            bad_cases = [
                ["image", dummy, "--dither-mode", "invalid-mode"],
                ["image", dummy, "--lightness", "99"],
                ["image", dummy, "--lightness", "-5"],
                ["image", dummy, "--contrast", "9"],
                ["image", dummy, "--contrast", "-3"],
                ["image", dummy, "--lightness", "abc"],
            ]
            for argv in bad_cases:
                res = subprocess.run(
                    [sys.executable, CLI_PATH, *argv],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(res.returncode, 2, f"Expected exit 2 for {argv}, got {res.returncode}: {res.stderr}")

            # Missing file exits 1 cleanly without opening USB
            missing = os.path.join(td, "does_not_exist.png")
            res_missing = subprocess.run(
                [sys.executable, CLI_PATH, "image", missing, "--dither-mode", "atkinson"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res_missing.returncode, 1)
            self.assertIn("Image Error", res_missing.stderr)
            self.assertNotIn("USB Error", res_missing.stderr)

    def test_cli_inprocess_mocked_controller(self) -> None:
        cli = _load_cli_module()
        parser = cli.build_parser()

        with tempfile.TemporaryDirectory() as td:
            img_path = os.path.join(td, "sample.png")
            grad = Image.new("L", (64, 32))
            grad.putdata([int(x * 255 / 63) for _ in range(32) for x in range(64)])
            grad.save(img_path)

            # Missing file must fail BEFORE instantiating FakeController
            FakeController.instances.clear()
            FakeController.init_count = 0
            args_missing = parser.parse_args(["image", os.path.join(td, "missing.png")])
            rc_missing = cli.cmd_image(args_missing, controller_cls=FakeController)
            self.assertEqual(rc_missing, 1)
            self.assertEqual(FakeController.init_count, 0)

            # Atkinson + negative/positive tone + gl16 + cover + invert + rotate-180
            FakeController.instances.clear()
            FakeController.init_count = 0
            args = parser.parse_args(
                [
                    "image",
                    img_path,
                    "--dither-mode",
                    "atkinson",
                    "--lightness",
                    "-3",
                    "--contrast",
                    "-1",
                    "--waveform",
                    "gl16",
                    "--fit",
                    "cover",
                    "--invert",
                    "--rotate-180",
                ]
            )
            self.assertEqual(args.dither_mode, "Atkinson")
            self.assertEqual(args.lightness, -3)
            self.assertEqual(args.contrast, -1)
            rc = cli.cmd_image(args, controller_cls=FakeController)
            self.assertEqual(rc, 0)
            self.assertEqual(FakeController.init_count, 1)
            ctrl = FakeController.instances[0]
            self.assertEqual(ctrl.calls[1], ("enable_draw_mode",))
            dpy_call = ctrl.calls[2]
            self.assertEqual(dpy_call[0], "display_image_bytes")
            self.assertEqual(dpy_call[1], PANEL_WIDTH * PANEL_HEIGHT)
            self.assertTrue(dpy_call[2].issubset(PACKED_2_SET))
            self.assertEqual(dpy_call[3], WAVEFORM_GL16)

            # Backward-compatible --dither without --dither-mode
            FakeController.instances.clear()
            args_legacy = parser.parse_args(["image", img_path, "--dither"])
            self.assertEqual(cli.cmd_image(args_legacy, controller_cls=FakeController), 0)
            self.assertTrue(FakeController.instances[0].calls[2][2].issubset(PACKED_2_SET))

            # Explicit --dither-mode grayscale-16 overrides --dither
            FakeController.instances.clear()
            args_override = parser.parse_args(["image", img_path, "--dither", "--dither-mode", "grayscale-16"])
            self.assertEqual(cli.cmd_image(args_override, controller_cls=FakeController), 0)
            packed_bytes = FakeController.instances[0].calls[2][2]
            self.assertTrue(packed_bytes.issubset(PACKED_16_SET))
            self.assertGreater(len(packed_bytes), 2)

    def test_cli_lightness_neg3_vs_pos3_distinct_packed_output(self) -> None:
        cli = _load_cli_module()
        parser = cli.build_parser()

        with tempfile.TemporaryDirectory() as td:
            img_path = os.path.join(td, "grad.png")
            grad = Image.new("L", (64, 32))
            grad.putdata([int(x * 255 / 63) for _ in range(32) for x in range(64)])
            grad.save(img_path)

            FakeController.instances.clear()
            args_dark = parser.parse_args(["image", img_path, "--fit", "cover", "--lightness", "-3"])
            self.assertEqual(cli.cmd_image(args_dark, controller_cls=FakeController), 0)
            packed_dark = FakeController.instances[0].last_packed
            assert packed_dark is not None

            FakeController.instances.clear()
            args_bright = parser.parse_args(["image", img_path, "--fit", "cover", "--lightness", "3"])
            self.assertEqual(cli.cmd_image(args_bright, controller_cls=FakeController), 0)
            packed_bright = FakeController.instances[0].last_packed
            assert packed_bright is not None

            self.assertEqual(len(packed_dark), PANEL_WIDTH * PANEL_HEIGHT)
            self.assertEqual(len(packed_bright), PANEL_WIDTH * PANEL_HEIGHT)
            self.assertNotEqual(packed_dark, packed_bright)
            self.assertGreater(sum(packed_bright), sum(packed_dark))


if __name__ == "__main__":
    unittest.main()
