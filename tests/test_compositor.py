"""Unit, policy, concurrency, and timing tests for yogabook_eink.compositor."""

from __future__ import annotations

import random
import sys
import threading
import time
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from yogabook_eink.compositor import (  # noqa: E402
    DITHER_ALIGNMENT,
    SEMANTIC_POLICIES,
    SEMANTIC_REFRESH_POLICIES,
    DirtyRect,
    EinkCompositor,
    PresentStats,
    align_rect,
    bayer_dither_1bit,
    compute_dirty_rects,
    dither_alignment_constraint,
    threshold_1bit,
)
from yogabook_eink.protocol import (  # noqa: E402
    PANEL_HEIGHT,
    PANEL_WIDTH,
    WAVEFORM_DU2,
    WAVEFORM_GC16,
    WAVEFORM_GL16,
    WAVEFORM_INIT,
)
from yogabook_eink.usb_lib import UsbError  # noqa: E402


class FakeController:
    """Thread-safe mock Ite8951Controller recording all display blits and concurrency."""

    def __init__(self, usb_delay_sec: float = 0.0) -> None:
        self.usb_delay_sec = usb_delay_sec
        self._guard = threading.Lock()
        self._in_flight = 0
        self.max_in_flight = 0
        self.calls: list[dict] = []
        self.fail_next_full_error: Optional[Exception] = None
        self.full_settle_event = threading.Event()
        self.full_enter_event: Optional[threading.Event] = None
        self.full_release_event: Optional[threading.Event] = None
        self.region_enter_event: Optional[threading.Event] = None
        self.region_release_event: Optional[threading.Event] = None
        self.exit_called = False

    def _enter_usb(self) -> None:
        with self._guard:
            self._in_flight += 1
            if self._in_flight > self.max_in_flight:
                self.max_in_flight = self._in_flight

    def _exit_usb(self) -> None:
        with self._guard:
            self._in_flight -= 1

    def exit_to_keyboard(self) -> None:
        self._enter_usb()
        try:
            if self.usb_delay_sec > 0:
                time.sleep(self.usb_delay_sec)
            with self._guard:
                self.exit_called = True
                self.calls.append(
                    {
                        "method": "exit_to_keyboard",
                        "timestamp": time.monotonic(),
                        "thread_id": threading.get_ident(),
                    }
                )
        finally:
            self._exit_usb()

    def display_image_bytes(
        self,
        pixels: bytes,
        width: int = PANEL_WIDTH,
        height: int = PANEL_HEIGHT,
        waveform: int = WAVEFORM_GC16,
        progress_cb=None,
    ) -> None:
        self._enter_usb()
        try:
            if self.full_enter_event is not None:
                self.full_enter_event.set()
            if self.full_release_event is not None:
                self.full_release_event.wait(timeout=5.0)
            if self.usb_delay_sec > 0:
                time.sleep(self.usb_delay_sec)
            with self._guard:
                if self.fail_next_full_error is not None:
                    err = self.fail_next_full_error
                    self.fail_next_full_error = None
                    raise err
                self.calls.append(
                    {
                        "method": "display_image_bytes",
                        "x": 0,
                        "y": 0,
                        "w": width,
                        "h": height,
                        "waveform": waveform,
                        "pixels": pixels,
                        "timestamp": time.monotonic(),
                        "thread_id": threading.get_ident(),
                    }
                )
            if waveform == WAVEFORM_GC16:
                self.full_settle_event.set()
        finally:
            self._exit_usb()

    def display_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        pixels: bytes,
        waveform: int = WAVEFORM_DU2,
    ) -> None:
        self._enter_usb()
        try:
            if self.region_enter_event is not None:
                self.region_enter_event.set()
            if self.region_release_event is not None:
                self.region_release_event.wait(timeout=5.0)
            if self.usb_delay_sec > 0:
                time.sleep(self.usb_delay_sec)
            with self._guard:
                self.calls.append(
                    {
                        "method": "display_region",
                        "x": x,
                        "y": y,
                        "w": width,
                        "h": height,
                        "waveform": waveform,
                        "pixels": pixels,
                        "timestamp": time.monotonic(),
                        "thread_id": threading.get_ident(),
                    }
                )
        finally:
            self._exit_usb()


class TestAlignRect(unittest.TestCase):
    def test_exports(self) -> None:
        self.assertEqual(DITHER_ALIGNMENT, 8)
        self.assertEqual(dither_alignment_constraint, 8)

    def test_already_aligned(self) -> None:
        self.assertEqual(align_rect(8, 16, 64, 32), (8, 16, 64, 32))
        self.assertEqual(align_rect(0, 0, 1920, 1080), (0, 0, 1920, 1080))

    def test_unaligned_interior_expands(self) -> None:
        self.assertEqual(align_rect(3, 5, 10, 12), (0, 0, 16, 24))
        self.assertEqual(align_rect(1, 1, 1, 1), (0, 0, 8, 8))

    def test_bottom_right_corner_and_clamping(self) -> None:
        self.assertEqual(align_rect(1919, 1079, 1, 1), (1912, 1072, 8, 8))
        self.assertEqual(align_rect(1915, 1075, 10, 10), (1912, 1072, 8, 8))

    def test_negative_origin_clipping(self) -> None:
        self.assertEqual(align_rect(-10, 5, 20, 10), (0, 0, 16, 16))
        self.assertEqual(align_rect(-10, 5, 11, 10), (0, 0, 8, 16))
        self.assertIsNone(align_rect(-10, 5, 10, 10))
        self.assertIsNone(align_rect(-20, 5, 10, 10))
        self.assertIsNone(align_rect(5, -15, 10, 15))
        self.assertIsNone(align_rect(5, -20, 10, 5))

    def test_oob_origins_and_degenerate_inputs(self) -> None:
        self.assertIsNone(align_rect(1920, 0, 10, 10))
        self.assertIsNone(align_rect(0, 1080, 10, 10))
        self.assertIsNone(align_rect(2500, 1500, 20, 20))
        self.assertIsNone(align_rect(1, 0, 0, 10))
        self.assertIsNone(align_rect(10, 0, -1, 10))
        self.assertIsNone(align_rect(10, 10, 10, 0))

    def test_post_clamp_degenerate_discard_and_alignment_1(self) -> None:
        self.assertIsNone(align_rect(10, 10, 1, 10, alignment=1))
        self.assertIsNone(align_rect(10, 10, 1, 10, alignment=None))
        self.assertEqual(align_rect(100, 10, 1, 10, max_w=101, max_h=100), (96, 8, 5, 16))
        self.assertIsNone(align_rect(100, 10, 1, 10, max_w=100, max_h=100))
        self.assertIsNone(align_rect(16, 10, 1, 10, max_w=17, max_h=100))

    def test_float_snap_and_invalid_alignment(self) -> None:
        self.assertEqual(align_rect(1.2, 2.8, 3.1, 4.1), (0, 0, 8, 8))
        self.assertEqual(align_rect(7.8, 7.8, 0.4, 0.4), (0, 0, 16, 16))
        self.assertEqual(align_rect(7.9, 15.9, 1.0, 1.0), (0, 8, 16, 16))
        self.assertEqual(align_rect(-0.8, 7.7, 9.0, 0.5), (0, 0, 16, 16))
        for bad in (0, -8, 3, 6, True, "8"):
            with self.assertRaises(ValueError):
                align_rect(0, 0, 16, 16, alignment=bad)  # type: ignore[arg-type]

    def test_randomized_float_enclosure(self) -> None:
        rng = random.Random(42)
        for _ in range(500):
            alignment = rng.choice([None, 1, 2, 4, 8, 16])
            a = 1 if alignment is None else alignment
            max_w, max_h = rng.choice([(1920, 1080), (645, 483), (101, 77), (17, 17)])
            x = rng.uniform(-50.0, max_w + 20.0)
            y = rng.uniform(-50.0, max_h + 20.0)
            w = rng.uniform(-10.0, 400.0)
            h = rng.uniform(-10.0, 300.0)
            res = align_rect(x, y, w, h, max_w=max_w, max_h=max_h, alignment=alignment)
            if res is None:
                continue
            rx, ry, rw, rh = res
            self.assertEqual(rx % a, 0)
            self.assertEqual(ry % a, 0)
            self.assertGreaterEqual(rx, 0)
            self.assertGreaterEqual(ry, 0)
            self.assertLessEqual(rx + rw, max_w)
            self.assertLessEqual(ry + rh, max_h)
            self.assertGreater(rw, 1)
            self.assertGreater(rh, 1)
            cx0 = max(0.0, x)
            cy0 = max(0.0, y)
            cx1 = min(float(max_w), x + w)
            cy1 = min(float(max_h), y + h)
            self.assertLessEqual(rx, cx0)
            self.assertLessEqual(ry, cy0)
            self.assertGreaterEqual(rx + rw, cx1)
            self.assertGreaterEqual(ry + rh, cy1)

    def test_non_finite_and_bool_inputs_raise_value_error(self) -> None:
        bad_vals = (float("nan"), float("inf"), float("-inf"), True, False)
        for bad in bad_vals:
            for kwargs in (
                {"x": bad, "y": 0, "w": 10, "h": 10},
                {"x": 0, "y": bad, "w": 10, "h": 10},
                {"x": 0, "y": 0, "w": bad, "h": 10},
                {"x": 0, "y": 0, "w": 10, "h": bad},
                {"x": 0, "y": 0, "w": 10, "h": 10, "max_w": bad},
                {"x": 0, "y": 0, "w": 10, "h": 10, "max_h": bad},
            ):
                with self.assertRaises(ValueError):
                    align_rect(**kwargs)  # type: ignore[arg-type]

        for args in (
            (float("-inf"), 0, float("inf"), 10),
            (0, 0, 10, 10, float("inf"), float("inf")),
            (0, 0, 10, 10, 1920.5, 1080),
            (0, 0, 10, 10, 1920, 1080.5),
            (0, 0, "10", 10),
        ):
            with self.assertRaises(ValueError):
                align_rect(*args)  # type: ignore[arg-type]


class TestComputeDirtyRects(unittest.TestCase):
    def _assert_rects_valid(
        self,
        rects: list[DirtyRect],
        max_w: int = PANEL_WIDTH,
        max_h: int = PANEL_HEIGHT,
        alignment: int = DITHER_ALIGNMENT,
    ) -> None:
        for r in rects:
            self.assertGreater(r.w, 1)
            self.assertGreater(r.h, 1)
            self.assertGreaterEqual(r.x, 0)
            self.assertGreaterEqual(r.y, 0)
            self.assertLessEqual(r.x + r.w, max_w)
            self.assertLessEqual(r.y + r.h, max_h)
            self.assertEqual(r.x % alignment, 0)
            self.assertEqual(r.y % alignment, 0)
            self.assertEqual(r.w % alignment, 0)
            self.assertEqual(r.h % alignment, 0)
            self.assertEqual(r.area, r.w * r.h)
        # Pairwise strictly non-overlapping
        for i, a in enumerate(rects):
            for b in rects[i + 1 :]:
                overlaps = (
                    a.x < b.x + b.w
                    and b.x < a.x + a.w
                    and a.y < b.y + b.h
                    and b.y < a.y + a.h
                )
                self.assertFalse(overlaps, f"Overlapping dirty rects: {a} and {b}")

    def test_identical_and_mismatch_validation(self) -> None:
        a = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
        b = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
        self.assertEqual(compute_dirty_rects(a, b), [])

        with self.assertRaises(ValueError):
            compute_dirty_rects(a, Image.new("L", (640, 480), 255))

        # Mode mismatch handled cleanly
        rgb = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), (255, 255, 255))
        self.assertEqual(compute_dirty_rects(a, rgb), [])

    def test_small_bbox_fast_path_is_8px_aligned(self) -> None:
        prev = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
        curr = prev.copy()
        curr.putpixel((1919, 1079), 0)
        rects = compute_dirty_rects(prev, curr)
        self.assertEqual(rects, [DirtyRect(1912, 1072, 8, 8)])
        self._assert_rects_valid(rects)

        curr2 = prev.copy()
        ImageDraw.Draw(curr2).rectangle([101, 203, 117, 215], fill=0)
        rects2 = compute_dirty_rects(prev, curr2)
        self.assertEqual(rects2, [DirtyRect(96, 200, 24, 16)])
        self._assert_rects_valid(rects2)

    def test_tile_path_odd_row_y_alignment(self) -> None:
        prev = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
        curr = prev.copy()
        d = ImageDraw.Draw(curr)
        # Span > 61440 overall bbox across two distant regions, including y=61 (tile row 1, 60%8==4)
        d.rectangle([13, 61, 22, 70], fill=0)
        d.rectangle([1803, 997, 1817, 1007], fill=0)
        rects = compute_dirty_rects(prev, curr)
        self.assertEqual(len(rects), 2)
        self._assert_rects_valid(rects)

    def test_overlap_merge_l_shape_enclosing_isolated_tile(self) -> None:
        prev = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
        curr = prev.copy()
        d = ImageDraw.Draw(curr)
        # L-shaped connected component across tiles (0,0)-(2,0)-(2,2) plus disconnected tile (0,2)
        # Tile (0,0)=[0..80, 0..60], (1,0)=[80..160, 0..60], (2,0)=[160..240, 0..60], (2,1)=[160..240, 60..120],
        # (2,2)=[160..240, 120..180], (1,2)=[80..160, 120..180], (0,2)=[0..80, 120..180]
        # Wait: (0,2) touches (1,2) diagonally/horizontally; put disconnected tile at (0,3)? No, put L around (0,2)
        # Put L on cols 0..4, rows 0..4 and isolated tile inside its bbox:
        # Left arm (0,0)->(0,4), bottom arm (0,4)->(4,4) -> bbox covers tiles (0..4, 0..4)
        # Isolated tile at (2,1) (not 8-connected to col 0 or row 4!) lies strictly inside L-bbox.
        for c, r in [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (2, 1)]:
            x = c * 80 + 10
            y = r * 60 + 10
            d.rectangle([x, y, x + 5, y + 5], fill=0)
        rects = compute_dirty_rects(prev, curr)
        self.assertEqual(len(rects), 1)
        self._assert_rects_valid(rects)

    def test_fragment_collapse_exceeding_max_rects_is_8px_aligned(self) -> None:
        prev = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
        curr = prev.copy()
        d = ImageDraw.Draw(curr)
        # 15 isolated tiny dots spaced >= 160px apart (> 1 tile gap)
        positions = [
            (13 + (i % 5) * 320, 11 + (i // 5) * 300)
            for i in range(15)
        ]
        for x, y in positions:
            d.rectangle([x, y, x + 2, y + 2], fill=0)
        rects = compute_dirty_rects(prev, curr, max_rects=12)
        self.assertEqual(len(rects), 1)
        self._assert_rects_valid(rects)

    def test_trailing_edge_pixel_coverage_on_dim_mod_alignment_1(self) -> None:
        def assert_covers_all_diffs(
            prev_img: Image.Image,
            curr_img: Image.Image,
            rects: list[DirtyRect],
            alignment: int = 8,
        ) -> None:
            w, h = curr_img.size
            for r in rects:
                self.assertGreater(r.w, 1)
                self.assertGreater(r.h, 1)
                self.assertGreaterEqual(r.x, 0)
                self.assertGreaterEqual(r.y, 0)
                self.assertLessEqual(r.x + r.w, w)
                self.assertLessEqual(r.y + r.h, h)
                self.assertEqual(r.x % alignment, 0)
                self.assertEqual(r.y % alignment, 0)
            for py in range(h):
                for px in range(w):
                    if prev_img.getpixel((px, py)) != curr_img.getpixel((px, py)):
                        covered = any(
                            r.x <= px < r.x + r.w and r.y <= py < r.y + r.h for r in rects
                        )
                        self.assertTrue(covered, f"Uncovered diff pixel ({px}, {py}) in {rects}")

        # (17, 17) at (16, 16)
        p17 = Image.new("L", (17, 17), 255)
        c17 = p17.copy()
        c17.putpixel((16, 16), 0)
        r17 = compute_dirty_rects(p17, c17)
        self.assertEqual(r17, [DirtyRect(8, 8, 9, 9)])
        assert_covers_all_diffs(p17, c17, r17)

        # (17, 17) at (16, 16) with alignment=1
        r17_a1 = compute_dirty_rects(p17, c17, alignment=1)
        self.assertEqual(len(r17_a1), 1)
        assert_covers_all_diffs(p17, c17, r17_a1, alignment=1)

        # (1920, 1081) at (1919, 1080) fast path
        p1081 = Image.new("L", (1920, 1081), 255)
        c1081 = p1081.copy()
        c1081.putpixel((1919, 1080), 0)
        r1081 = compute_dirty_rects(p1081, c1081)
        self.assertEqual(r1081, [DirtyRect(1912, 1072, 8, 9)])
        self.assertGreater(len(r1081), 0)
        self.assertTrue(
            any(r.x <= 1919 < r.x + r.w and r.y <= 1080 < r.y + r.h for r in r1081)
        )

        # (1920, 1081) spanning (10, 10) and (1919, 1080) -> tile component path
        c1081_span = p1081.copy()
        c1081_span.putpixel((10, 10), 0)
        c1081_span.putpixel((1919, 1080), 0)
        r_span = compute_dirty_rects(p1081, c1081_span)
        self.assertEqual(len(r_span), 2)
        for px, py in ((10, 10), (1919, 1080)):
            self.assertTrue(
                any(r.x <= px < r.x + r.w and r.y <= py < r.y + r.h for r in r_span),
                f"Uncovered pixel ({px}, {py}) in {r_span}",
            )
            for r in r_span:
                self.assertEqual(r.x % 8, 0)
                self.assertEqual(r.y % 8, 0)
                self.assertGreater(r.w, 1)
                self.assertGreater(r.h, 1)

        # (1920, 1081) > max_rects collapse path including (1919, 1080)
        c1081_many = p1081.copy()
        for i in range(14):
            c1081_many.putpixel((10 + (i % 5) * 320, 10 + (i // 5) * 300), 0)
        c1081_many.putpixel((1919, 1080), 0)
        r_many = compute_dirty_rects(p1081, c1081_many, max_rects=12)
        self.assertEqual(len(r_many), 1)
        self.assertTrue(
            r_many[0].x <= 1919 < r_many[0].x + r_many[0].w
            and r_many[0].y <= 1080 < r_many[0].y + r_many[0].h
        )
        self.assertEqual(r_many[0].x % 8, 0)
        self.assertEqual(r_many[0].y % 8, 0)
        self.assertGreater(r_many[0].w, 1)
        self.assertGreater(r_many[0].h, 1)

    def test_same_mode_alpha_and_palette_images(self) -> None:
        # Identical RGBA/LA/PA/P return []
        rgba_prev = Image.new("RGBA", (PANEL_WIDTH, PANEL_HEIGHT), (255, 255, 255, 255))
        self.assertEqual(compute_dirty_rects(rgba_prev, rgba_prev.copy()), [])

        # Same-mode RGBA with matching alpha=255 and differing RGB (both small and tile paths)
        rgba_curr_small = rgba_prev.copy()
        rgba_curr_small.putpixel((101, 203), (10, 20, 30, 255))
        rects_rgba_small = compute_dirty_rects(rgba_prev, rgba_curr_small)
        self.assertEqual(rects_rgba_small, [DirtyRect(96, 200, 8, 8)])
        self._assert_rects_valid(rects_rgba_small)

        rgba_curr_tiles = rgba_prev.copy()
        rgba_curr_tiles.putpixel((13, 19), (0, 0, 0, 255))
        rgba_curr_tiles.putpixel((1803, 997), (40, 50, 60, 255))
        rects_rgba_tiles = compute_dirty_rects(rgba_prev, rgba_curr_tiles)
        self.assertEqual(len(rects_rgba_tiles), 2)
        self._assert_rects_valid(rects_rgba_tiles)
        self.assertTrue(any(r.x <= 13 < r.x + r.w and r.y <= 19 < r.y + r.h for r in rects_rgba_tiles))
        self.assertTrue(any(r.x <= 1803 < r.x + r.w and r.y <= 997 < r.y + r.h for r in rects_rgba_tiles))

        # Same-mode LA with matching alpha=255 and differing luminance
        la_prev = Image.new("LA", (PANEL_WIDTH, PANEL_HEIGHT), (255, 255))
        self.assertEqual(compute_dirty_rects(la_prev, la_prev.copy()), [])
        la_curr = la_prev.copy()
        la_curr.putpixel((25, 33), (0, 255))
        rects_la = compute_dirty_rects(la_prev, la_curr)
        self.assertEqual(rects_la, [DirtyRect(24, 32, 8, 8)])
        self._assert_rects_valid(rects_la)

        # Same-mode PA with matching alpha=255 and differing color/palette index
        pa_prev = Image.new("PA", (PANEL_WIDTH, PANEL_HEIGHT), (0, 255))
        pa_prev.putpalette([255, 255, 255, 0, 0, 0])
        self.assertEqual(compute_dirty_rects(pa_prev, pa_prev.copy()), [])
        pa_curr = pa_prev.copy()
        pa_curr.putpixel((41, 57), (1, 255))
        rects_pa = compute_dirty_rects(pa_prev, pa_curr)
        self.assertEqual(rects_pa, [DirtyRect(40, 56, 8, 8)])
        self._assert_rects_valid(rects_pa)

        # Same-mode P with identical pixel indices but differing palettes
        p_prev = Image.new("P", (64, 64), 0)
        p_prev.putpalette([255, 255, 255, 255, 255, 255])
        p_prev.putpixel((13, 19), 1)
        p_curr_local = p_prev.copy()
        p_curr_local.putpalette([255, 255, 255, 0, 0, 0])
        rects_p_local = compute_dirty_rects(p_prev, p_curr_local)
        self.assertEqual(rects_p_local, [DirtyRect(8, 16, 8, 8)])
        self._assert_rects_valid(rects_p_local, max_w=64, max_h=64)

        p_curr_full = Image.new("P", (64, 64), 0)
        p_curr_full.putpalette([0, 0, 0, 0, 0, 0])
        rects_p_full = compute_dirty_rects(p_prev, p_curr_full)
        self.assertEqual(rects_p_full, [DirtyRect(0, 0, 64, 64)])
        self._assert_rects_valid(rects_p_full, max_w=64, max_h=64)


class TestSemanticRefreshPolicy(unittest.TestCase):
    def test_constants_and_invalid_policy(self) -> None:
        self.assertEqual(SEMANTIC_POLICIES, ("fast", "ui", "partial", "full", "clean"))
        self.assertEqual(SEMANTIC_REFRESH_POLICIES, SEMANTIC_POLICIES)
        ctrl = FakeController()
        with EinkCompositor(ctrl, idle_settle_enabled=False, max_fps=100.0) as comp:
            frame = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
            with self.assertRaises(ValueError):
                comp.present(frame, policy="turbo")

    def test_present_stats_int_and_bool(self) -> None:
        s0 = PresentStats(blits=0, policy="fast", waveform=None, full_refresh=False)
        s2 = PresentStats(blits=2, policy="fast", waveform=WAVEFORM_DU2, full_refresh=False)
        self.assertEqual(int(s0), 0)
        self.assertFalse(bool(s0))
        self.assertEqual(int(s2), 2)
        self.assertTrue(bool(s2))

    def test_fast_policy_1bit_budget_and_coverage(self) -> None:
        ctrl = FakeController()
        with EinkCompositor(ctrl, gc16_budget=5, max_fps=100.0, idle_settle_enabled=False) as comp:
            # First frame -> full DU2 with strict 1-bit binarization (mid-gray 100 -> 0 -> 0x0F)
            f0 = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 200)
            ImageDraw.Draw(f0).rectangle([10, 10, 40, 40], fill=100)
            st0 = comp.present(f0, policy="fast")
            self.assertEqual(st0.blits, 1)
            self.assertEqual(st0.waveform, WAVEFORM_DU2)
            self.assertTrue(st0.full_refresh)
            self.assertEqual(set(ctrl.calls[-1]["pixels"]), {0x0F, 0xFF})

            # Unchanged frame -> no-op
            st_same = comp.present(f0, policy="fast")
            self.assertEqual(st_same.blits, 0)
            self.assertIsNone(st_same.waveform)

            # 5 partial updates -> DU2 aligned partial, then 6th triggers GC16 budget
            for i in range(5):
                fi = f0.copy()
                ImageDraw.Draw(fi).rectangle([101 + i * 20, 203, 111 + i * 20, 213], fill=50)
                st = comp.present(fi, policy="fast")
                self.assertEqual(st.waveform, WAVEFORM_DU2)
                self.assertFalse(st.full_refresh)
                for r in st.rects:
                    self.assertEqual((r.x % 8, r.y % 8, r.w % 8, r.h % 8), (0, 0, 0, 0))
            self.assertEqual(comp.partial_count, 5)

            f_budget = f0.copy()
            ImageDraw.Draw(f_budget).rectangle([500, 500, 520, 520], fill=0)
            st_b = comp.present(f_budget, policy="fast")
            self.assertEqual(st_b.waveform, WAVEFORM_GC16)
            self.assertTrue(st_b.full_refresh)
            self.assertEqual(comp.partial_count, 0)

            # >65% coverage -> full DU2 in fast mode
            f_big = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 0)
            st_big = comp.present(f_big, policy="fast")
            self.assertEqual(st_big.blits, 1)
            self.assertEqual(st_big.waveform, WAVEFORM_DU2)
            self.assertTrue(st_big.full_refresh)

    def test_ui_policy(self) -> None:
        ctrl = FakeController()
        with EinkCompositor(ctrl, gc16_budget=5, max_fps=100.0, idle_settle_enabled=False) as comp:
            f0 = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
            st0 = comp.present(f0, policy="ui")
            self.assertEqual(st0.waveform, WAVEFORM_GC16)
            self.assertTrue(st0.full_refresh)

            f1 = f0.copy()
            ImageDraw.Draw(f1).rectangle([51, 53, 81, 83], fill=90)
            st1 = comp.present(f1, policy="ui")
            self.assertEqual(st1.waveform, WAVEFORM_DU2)
            self.assertFalse(st1.full_refresh)
            self.assertEqual(set(ctrl.calls[-1]["pixels"]), {0x0F, 0xFF})

            # >65% coverage -> full GC16 in ui mode
            f2 = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 0)
            st2 = comp.present(f2, policy="ui")
            self.assertEqual(st2.waveform, WAVEFORM_GC16)
            self.assertTrue(st2.full_refresh)
            self.assertEqual(comp.partial_count, 0)

    def test_partial_policy_preserves_grayscale_and_uses_gl16(self) -> None:
        ctrl = FakeController()
        with EinkCompositor(ctrl, gc16_budget=5, max_fps=100.0, strict_1bit=True, idle_settle_enabled=True) as comp:
            f0 = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
            st0 = comp.present(f0, policy="partial")
            self.assertEqual(st0.waveform, WAVEFORM_GC16)
            self.assertFalse(st0.settle_armed)

            f1 = f0.copy()
            ImageDraw.Draw(f1).rectangle([101, 103, 141, 143], fill=64)
            ImageDraw.Draw(f1).rectangle([110, 110, 130, 130], fill=128)
            st1 = comp.present(f1, policy="partial")
            self.assertEqual(st1.waveform, WAVEFORM_GL16)
            self.assertFalse(st1.full_refresh)
            self.assertFalse(st1.settle_armed)
            packed_vals = set(ctrl.calls[-1]["pixels"])
            self.assertIn(0x4F, packed_vals)
            self.assertIn(0x8F, packed_vals)
            self.assertIn(0xFF, packed_vals)

    def test_ui_and_partial_gc16_budget_rollover(self) -> None:
        for pol, partial_wf in (("ui", WAVEFORM_DU2), ("partial", WAVEFORM_GL16)):
            ctrl = FakeController()
            with EinkCompositor(ctrl, gc16_budget=5, max_fps=100.0, idle_settle_enabled=False) as comp:
                f0 = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
                st0 = comp.present(f0, policy=pol)
                self.assertEqual(st0.waveform, WAVEFORM_GC16)
                self.assertEqual(comp.partial_count, 0)

                for i in range(5):
                    fi = f0.copy()
                    ImageDraw.Draw(fi).rectangle([101 + i * 24, 203, 113 + i * 24, 215], fill=40)
                    st = comp.present(fi, policy=pol)
                    self.assertEqual(st.waveform, partial_wf)
                    self.assertFalse(st.full_refresh)
                self.assertEqual(comp.partial_count, 5)

                f_budget = f0.copy()
                ImageDraw.Draw(f_budget).rectangle([500, 500, 520, 520], fill=0)
                st_b = comp.present(f_budget, policy=pol)
                self.assertEqual(st_b.waveform, WAVEFORM_GC16)
                self.assertTrue(st_b.full_refresh)
                self.assertEqual(comp.partial_count, 0)

    def test_full_clean_and_force_full(self) -> None:
        ctrl = FakeController()
        with EinkCompositor(ctrl, gc16_budget=5, max_fps=100.0, idle_settle_enabled=False) as comp:
            gray = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 128)
            st_full = comp.present(gray, policy="full")
            self.assertEqual(st_full.waveform, WAVEFORM_GC16)
            self.assertTrue(st_full.full_refresh)
            self.assertEqual(set(ctrl.calls[-1]["pixels"]), {0x8F})

            # Unconditional blit even on identical frame
            st_full2 = comp.present(gray, policy="full")
            self.assertEqual(st_full2.blits, 1)
            self.assertEqual(st_full2.waveform, WAVEFORM_GC16)

            st_clean = comp.present(gray, policy="clean")
            self.assertEqual(st_clean.waveform, WAVEFORM_INIT)
            self.assertTrue(st_clean.full_refresh)
            self.assertEqual(set(ctrl.calls[-1]["pixels"]), {0x8F})

            # force_full promotes fast/ui/partial to full clean_waveform
            st_ff = comp.present(gray, policy="partial", force_full=True, clean_waveform=WAVEFORM_GC16)
            self.assertEqual(st_ff.policy, "full")
            self.assertEqual(st_ff.waveform, WAVEFORM_GC16)
            self.assertTrue(st_ff.full_refresh)

            # force_full on clean keeps WAVEFORM_INIT
            st_fc = comp.present(gray, policy="clean", force_full=True)
            self.assertEqual(st_fc.policy, "clean")
            self.assertEqual(st_fc.waveform, WAVEFORM_INIT)

    def test_bayer_and_threshold_helpers_preserved(self) -> None:
        grad = Image.new("L", (32, 32), 120)
        b = bayer_dither_1bit(grad)
        t = threshold_1bit(grad)
        self.assertEqual(b.mode, "L")
        self.assertEqual(set(b.tobytes()), {0, 255})
        self.assertEqual(set(t.tobytes()), {0})


class TestIdleSettleAndConcurrency(unittest.TestCase):
    def test_fast_idle_settle_fires_once_and_resets_count(self) -> None:
        ctrl = FakeController()
        with EinkCompositor(ctrl, gc16_budget=10, max_fps=100.0, idle_settle_sec=0.35) as comp:
            comp.reset_framebuffer(255)
            f1 = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
            ImageDraw.Draw(f1).rectangle([20, 20, 80, 80], fill=0)
            st = comp.present(f1, policy="fast")
            self.assertTrue(st.settle_armed)
            self.assertTrue(comp.settle_armed)
            self.assertEqual(comp.partial_count, 1)

            # Static unchanged frame must NOT cancel or reset armed timer
            st_same = comp.present(f1, policy="fast")
            self.assertEqual(st_same.blits, 0)
            self.assertTrue(st_same.settle_armed)

            self.assertTrue(ctrl.full_settle_event.wait(timeout=1.2))
            time.sleep(0.05)
            gc16_calls = [c for c in ctrl.calls if c["waveform"] == WAVEFORM_GC16]
            self.assertEqual(len(gc16_calls), 1)
            self.assertEqual(gc16_calls[0]["method"], "display_image_bytes")
            self.assertEqual(comp.partial_count, 0)
            self.assertEqual(comp.settle_count, 1)
            self.assertFalse(comp.settle_armed)

    def test_idle_settle_disabled(self) -> None:
        ctrl = FakeController()
        with EinkCompositor(ctrl, max_fps=100.0, idle_settle_enabled=False, idle_settle_sec=0.04) as comp:
            comp.reset_framebuffer(255)
            f1 = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
            ImageDraw.Draw(f1).rectangle([20, 20, 80, 80], fill=0)
            st = comp.present(f1, policy="fast")
            self.assertFalse(st.settle_armed)
            time.sleep(0.12)
            self.assertEqual(comp.settle_count, 0)

    def test_incoming_frame_cancels_and_resets_timer(self) -> None:
        ctrl = FakeController()
        with EinkCompositor(ctrl, max_fps=100.0, idle_settle_sec=0.35) as comp:
            comp.reset_framebuffer(255)
            f1 = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
            ImageDraw.Draw(f1).rectangle([20, 20, 60, 60], fill=0)
            comp.present(f1, policy="ui")
            time.sleep(0.10)
            self.assertFalse(ctrl.full_settle_event.is_set())

            f2 = f1.copy()
            ImageDraw.Draw(f2).rectangle([100, 100, 140, 140], fill=0)
            t_last = time.monotonic()
            comp.present(f2, policy="ui")

            # Must not fire at original 0.35s mark
            time.sleep(0.12)
            self.assertEqual(comp.settle_count, 0)

            self.assertTrue(ctrl.full_settle_event.wait(timeout=1.5))
            elapsed = ctrl.calls[-1]["timestamp"] - t_last
            self.assertGreaterEqual(elapsed, 0.25)
            self.assertEqual(comp.settle_count, 1)

    def test_cancel_on_full_clean_partial_cancel_reset_close(self) -> None:
        ctrl = FakeController()
        with EinkCompositor(ctrl, max_fps=100.0, idle_settle_sec=0.25) as comp:
            comp.reset_framebuffer(255)
            f1 = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
            ImageDraw.Draw(f1).rectangle([20, 20, 60, 60], fill=0)

            comp.present(f1, policy="fast")
            self.assertTrue(comp.settle_armed)
            comp.cancel_settle()
            self.assertFalse(comp.settle_armed)
            time.sleep(0.32)
            self.assertEqual(comp.settle_count, 0)

            f2 = f1.copy()
            ImageDraw.Draw(f2).rectangle([80, 80, 120, 120], fill=0)
            comp.present(f2, policy="fast")
            self.assertTrue(comp.settle_armed)
            comp.present(f2, policy="clean")
            self.assertFalse(comp.settle_armed)
            time.sleep(0.32)
            self.assertEqual(comp.settle_count, 0)

            f3 = f2.copy()
            ImageDraw.Draw(f3).rectangle([140, 140, 180, 180], fill=0)
            comp.present(f3, policy="fast")
            self.assertTrue(comp.settle_armed)
            comp.present(f3, policy="full")
            self.assertFalse(comp.settle_armed)
            time.sleep(0.32)
            self.assertEqual(comp.settle_count, 0)

            f3_partial = f3.copy()
            ImageDraw.Draw(f3_partial).rectangle([160, 160, 195, 195], fill=0)
            comp.present(f3_partial, policy="fast")
            self.assertTrue(comp.settle_armed)
            f3_partial_gray = f3_partial.copy()
            ImageDraw.Draw(f3_partial_gray).rectangle([168, 168, 188, 188], fill=128)
            st_partial = comp.present(f3_partial_gray, policy="partial")
            self.assertGreaterEqual(st_partial.blits, 1)
            self.assertEqual(st_partial.waveform, WAVEFORM_GL16)
            self.assertFalse(st_partial.settle_armed)
            self.assertFalse(comp.settle_armed)
            time.sleep(0.32)
            self.assertEqual(comp.settle_count, 0)

            f4 = f3.copy()
            ImageDraw.Draw(f4).rectangle([200, 200, 240, 240], fill=0)
            comp.present(f4, policy="fast")
            self.assertTrue(comp.settle_armed)
            comp.reset_framebuffer(255)
            self.assertFalse(comp.settle_armed)
            time.sleep(0.32)
            self.assertEqual(comp.settle_count, 0)

            comp.present(f4, policy="fast")
            self.assertTrue(comp.settle_armed)

        # After __exit__ -> close()
        self.assertTrue(comp.is_closed)
        self.assertFalse(comp.settle_armed)
        time.sleep(0.32)
        self.assertEqual(comp.settle_count, 0)

    def test_idle_settle_handles_usb_error_cleanly(self) -> None:
        ctrl = FakeController()
        with EinkCompositor(ctrl, max_fps=100.0, idle_settle_sec=0.10) as comp:
            comp.reset_framebuffer(255)
            f1 = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
            ImageDraw.Draw(f1).rectangle([20, 20, 60, 60], fill=0)
            ctrl.fail_next_full_error = UsbError("simulated disconnect")
            comp.present(f1, policy="fast")
            time.sleep(0.25)
            self.assertEqual(comp.settle_count, 0)
            self.assertFalse(comp.settle_armed)

    def test_tight_margin_incoming_diff_cancels_while_unchanged_preserves_settle(self) -> None:
        ctrl = FakeController()
        with EinkCompositor(ctrl, max_fps=100.0, idle_settle_sec=0.06) as comp:
            comp.reset_framebuffer(255)
            f1 = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
            ImageDraw.Draw(f1).rectangle([20, 20, 60, 60], fill=0)
            f2 = f1.copy()
            ImageDraw.Draw(f2).rectangle([120, 120, 160, 160], fill=0)

            orig_threshold = threshold_1bit

            def slow_threshold(img: Image.Image, cutoff: int = 128) -> Image.Image:
                time.sleep(0.09)
                return orig_threshold(img, cutoff=cutoff)

            # Part A: unchanged frame with preprocessing exceeding idle_settle_sec preserves armed timer
            comp.present(f1, policy="fast")
            with mock.patch("yogabook_eink.compositor.threshold_1bit", side_effect=slow_threshold):
                st_same = comp.present(f1, policy="fast")
            self.assertEqual(st_same.blits, 0)
            self.assertTrue(st_same.settle_armed)
            self.assertTrue(ctrl.full_settle_event.wait(timeout=1.0))
            self.assertEqual(comp.settle_count, 1)

            # Part B: differing frame whose preprocessing spans past idle_settle_sec cancels old settle cleanly
            ctrl.full_settle_event.clear()
            comp.present(f1, policy="fast")
            with mock.patch("yogabook_eink.compositor.threshold_1bit", side_effect=slow_threshold):
                st_diff = comp.present(f2, policy="ui")
            self.assertEqual(st_diff.waveform, WAVEFORM_DU2)
            self.assertTrue(st_diff.settle_armed)
            self.assertEqual(comp.settle_count, 1)
            self.assertTrue(ctrl.full_settle_event.wait(timeout=1.0))
            self.assertEqual(comp.settle_count, 2)

    def test_real_default_2s_idle_settle_timing(self) -> None:
        ctrl = FakeController()
        with EinkCompositor(ctrl, max_fps=100.0) as comp:
            self.assertEqual(comp.idle_settle_sec, 2.0)
            comp.reset_framebuffer(255)

            base = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
            t_last_burst = time.monotonic()
            for i in range(3):
                frame = base.copy()
                ImageDraw.Draw(frame).rectangle([40 + i * 40, 40, 70 + i * 40, 70], fill=0)
                st = comp.present(frame, policy="fast")
                t_last_burst = time.monotonic()
                self.assertTrue(st.settle_armed)

            # Must not fire prematurely before ~2s
            time.sleep(1.3)
            self.assertFalse(ctrl.full_settle_event.is_set())
            self.assertEqual(comp.settle_count, 0)

            self.assertTrue(ctrl.full_settle_event.wait(timeout=1.5))
            elapsed = ctrl.calls[-1]["timestamp"] - t_last_burst
            self.assertGreaterEqual(elapsed, 1.85)
            self.assertEqual(comp.settle_count, 1)

    def test_multithreaded_usb_lock_serialization_and_close_no_deadlock(self) -> None:
        ctrl = FakeController(usb_delay_sec=0.002)
        ext_lock = threading.RLock()
        errors: list[Exception] = []

        with EinkCompositor(
            ctrl,
            gc16_budget=15,
            max_fps=100.0,
            idle_settle_sec=0.02,
            usb_lock=ext_lock,
        ) as comp:
            comp.reset_framebuffer(255)
            base = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)

            def worker(tid: int) -> None:
                try:
                    for step in range(12):
                        frame = base.copy()
                        ImageDraw.Draw(frame).rectangle(
                            [20 + tid * 100, 20 + step * 30, 50 + tid * 100, 45 + step * 30],
                            fill=(step * 40) % 256,
                        )
                        pol = ("fast", "ui", "partial", "full")[(tid + step) % 4]
                        comp.present(frame, policy=pol)
                        if step % 5 == 0:
                            comp.cancel_settle()
                except Exception as exc:  # pragma: no cover
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10.0)

        self.assertEqual(errors, [])
        self.assertEqual(ctrl.max_in_flight, 1)

    def test_close_does_not_deadlock_while_external_lock_held(self) -> None:
        ctrl = FakeController()
        ext_lock = threading.Lock()
        comp = EinkCompositor(ctrl, max_fps=100.0, idle_settle_sec=0.02, usb_lock=ext_lock)
        comp.reset_framebuffer(255)
        f1 = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
        ImageDraw.Draw(f1).rectangle([20, 20, 60, 60], fill=0)
        comp.present(f1, policy="fast")

        ext_lock.acquire()
        try:
            # Wait briefly so timer enters _run_idle_settle waiting on ext_lock
            time.sleep(0.06)
            t0 = time.monotonic()
            comp.close()
            self.assertLess(time.monotonic() - t0, 0.3)
            self.assertTrue(comp.is_closed)
            self.assertEqual(comp.settle_count, 0)
        finally:
            ext_lock.release()

    def test_close_blocks_until_inflight_settle_usb_transfer_finishes(self) -> None:
        ctrl = FakeController()
        ctrl.full_enter_event = threading.Event()
        ctrl.full_release_event = threading.Event()

        comp = EinkCompositor(ctrl, max_fps=100.0, idle_settle_sec=0.02)
        comp.reset_framebuffer(255)
        f1 = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
        ImageDraw.Draw(f1).rectangle([20, 20, 60, 60], fill=0)
        comp.present(f1, policy="fast")

        self.assertTrue(ctrl.full_enter_event.wait(timeout=1.0))

        close_done = threading.Event()

        def do_close() -> None:
            comp.close()
            close_done.set()
            ctrl.exit_to_keyboard()

        closer = threading.Thread(target=do_close)
        closer.start()

        # close() must block while display_image_bytes is still in flight
        time.sleep(0.06)
        self.assertFalse(close_done.is_set())

        ctrl.full_release_event.set()
        self.assertTrue(close_done.wait(timeout=2.0))
        closer.join(timeout=2.0)

        self.assertTrue(comp.is_closed)
        self.assertTrue(ctrl.exit_called)
        self.assertEqual(ctrl.max_in_flight, 1)
        self.assertEqual(
            [c["method"] for c in ctrl.calls],
            ["display_region", "display_image_bytes", "exit_to_keyboard"],
        )
        self.assertEqual(ctrl.calls[1]["waveform"], WAVEFORM_GC16)

    def test_concurrent_present_and_close_no_unstarted_timer_runtime_error(self) -> None:
        base = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
        f1 = base.copy()
        ImageDraw.Draw(f1).rectangle([24, 24, 64, 64], fill=0)
        errors: list[Exception] = []

        for _ in range(30):
            ctrl = FakeController()
            comp = EinkCompositor(ctrl, max_fps=100.0, idle_settle_sec=0.02)
            comp.reset_framebuffer(255)
            barrier = threading.Barrier(2)

            def run_present() -> None:
                try:
                    barrier.wait(timeout=1.0)
                    comp.present(f1, policy="fast")
                except Exception as exc:  # pragma: no cover
                    errors.append(exc)

            def run_close() -> None:
                try:
                    barrier.wait(timeout=1.0)
                    comp.close()
                except Exception as exc:  # pragma: no cover
                    errors.append(exc)

            t1 = threading.Thread(target=run_present)
            t2 = threading.Thread(target=run_close)
            t1.start()
            t2.start()
            t1.join(timeout=2.0)
            t2.join(timeout=2.0)
            self.assertTrue(comp.is_closed)

        self.assertEqual(errors, [])

    def test_reset_framebuffer_concurrent_with_slow_present_not_clobbered(self) -> None:
        ctrl = FakeController()
        comp = EinkCompositor(ctrl, max_fps=100.0, idle_settle_sec=0.05)
        comp.reset_framebuffer(255)

        f1 = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
        ImageDraw.Draw(f1).rectangle([24, 24, 88, 88], fill=0)

        ctrl.region_enter_event = threading.Event()
        ctrl.region_release_event = threading.Event()
        present_result: list[PresentStats] = []

        def run_present() -> None:
            present_result.append(comp.present(f1, policy="fast"))

        t = threading.Thread(target=run_present)
        t.start()
        self.assertTrue(ctrl.region_enter_event.wait(timeout=1.0))

        # Reset framebuffer while present() is blocked inside display_region()
        comp.reset_framebuffer(0)
        ctrl.region_release_event.set()
        t.join(timeout=2.0)

        try:
            self.assertEqual(len(present_result), 1)
            self.assertFalse(present_result[0].settle_armed)
            self.assertFalse(comp.settle_armed)
            self.assertEqual(comp.partial_count, 0)
            self.assertIsNotNone(comp.prev_frame)
            assert comp.prev_frame is not None
            self.assertEqual(comp.prev_frame.getpixel((0, 0)), 0)
            time.sleep(0.1)
            self.assertEqual(comp.settle_count, 0)
        finally:
            comp.close()

    def test_cancel_settle_then_close_blocks_until_inflight_settle_finishes(self) -> None:
        ctrl = FakeController()
        ctrl.full_enter_event = threading.Event()
        ctrl.full_release_event = threading.Event()

        comp = EinkCompositor(ctrl, max_fps=100.0, idle_settle_sec=0.02)
        comp.reset_framebuffer(255)
        f1 = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
        ImageDraw.Draw(f1).rectangle([20, 20, 60, 60], fill=0)
        comp.present(f1, policy="fast")

        self.assertTrue(ctrl.full_enter_event.wait(timeout=1.0))

        comp.cancel_settle()
        close_done = threading.Event()

        def do_close() -> None:
            comp.close()
            close_done.set()
            ctrl.exit_to_keyboard()

        closer = threading.Thread(target=do_close)
        closer.start()

        time.sleep(0.06)
        self.assertFalse(close_done.is_set())

        ctrl.full_release_event.set()
        self.assertTrue(close_done.wait(timeout=2.0))
        closer.join(timeout=2.0)

        self.assertTrue(comp.is_closed)
        self.assertTrue(ctrl.exit_called)
        self.assertEqual(ctrl.max_in_flight, 1)
        self.assertEqual(comp.settle_count, 0)
        self.assertEqual(len(comp._active_settle_threads), 0)
        self.assertEqual(
            [c["method"] for c in ctrl.calls],
            ["display_region", "display_image_bytes", "exit_to_keyboard"],
        )

    def test_reset_framebuffer_then_close_blocks_until_inflight_settle_finishes(self) -> None:
        ctrl = FakeController()
        ctrl.full_enter_event = threading.Event()
        ctrl.full_release_event = threading.Event()

        comp = EinkCompositor(ctrl, max_fps=100.0, idle_settle_sec=0.02)
        comp.reset_framebuffer(255)
        f1 = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
        ImageDraw.Draw(f1).rectangle([20, 20, 60, 60], fill=0)
        comp.present(f1, policy="fast")

        self.assertTrue(ctrl.full_enter_event.wait(timeout=1.0))

        comp.reset_framebuffer(0)
        close_done = threading.Event()

        def do_close() -> None:
            comp.close()
            close_done.set()
            ctrl.exit_to_keyboard()

        closer = threading.Thread(target=do_close)
        closer.start()

        time.sleep(0.06)
        self.assertFalse(close_done.is_set())

        ctrl.full_release_event.set()
        self.assertTrue(close_done.wait(timeout=2.0))
        closer.join(timeout=2.0)

        self.assertTrue(comp.is_closed)
        self.assertTrue(ctrl.exit_called)
        self.assertEqual(ctrl.max_in_flight, 1)
        self.assertEqual(comp.partial_count, 0)
        self.assertEqual(comp.settle_count, 0)
        self.assertEqual(len(comp._active_settle_threads), 0)
        self.assertEqual(
            [c["method"] for c in ctrl.calls],
            ["display_region", "display_image_bytes", "exit_to_keyboard"],
        )

    def test_concurrent_double_close_blocks_both_until_inflight_settle_finishes(self) -> None:
        ctrl = FakeController()
        ctrl.full_enter_event = threading.Event()
        ctrl.full_release_event = threading.Event()

        comp = EinkCompositor(ctrl, max_fps=100.0, idle_settle_sec=0.02)
        comp.reset_framebuffer(255)
        f1 = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
        ImageDraw.Draw(f1).rectangle([20, 20, 60, 60], fill=0)
        comp.present(f1, policy="fast")

        self.assertTrue(ctrl.full_enter_event.wait(timeout=1.0))

        t1_done = threading.Event()
        t2_done = threading.Event()

        def do_close_1() -> None:
            comp.close()
            t1_done.set()

        def do_close_2() -> None:
            comp.close()
            t2_done.set()
            ctrl.exit_to_keyboard()

        t1 = threading.Thread(target=do_close_1)
        t1.start()
        time.sleep(0.03)
        t2 = threading.Thread(target=do_close_2)
        t2.start()

        time.sleep(0.06)
        self.assertFalse(t1_done.is_set())
        self.assertFalse(t2_done.is_set())

        ctrl.full_release_event.set()
        self.assertTrue(t1_done.wait(timeout=2.0))
        self.assertTrue(t2_done.wait(timeout=2.0))
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)

        self.assertTrue(comp.is_closed)
        self.assertTrue(ctrl.exit_called)
        self.assertEqual(ctrl.max_in_flight, 1)
        self.assertEqual(len(comp._active_settle_threads), 0)
        self.assertEqual(
            [c["method"] for c in ctrl.calls],
            ["display_region", "display_image_bytes", "exit_to_keyboard"],
        )

    def test_cancelled_timers_do_not_leak_in_active_settle_threads(self) -> None:
        ctrl = FakeController()
        comp = EinkCompositor(ctrl, max_fps=1000.0, idle_settle_sec=10.0)
        try:
            comp.reset_framebuffer(255)
            f_a = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
            ImageDraw.Draw(f_a).rectangle([20, 20, 60, 60], fill=0)
            f_b = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
            ImageDraw.Draw(f_b).rectangle([120, 120, 160, 160], fill=0)

            for i in range(30):
                comp.present(f_a if i % 2 == 0 else f_b, policy="fast")
                comp.cancel_settle()
            for i in range(20):
                comp.present(f_a if i % 2 == 0 else f_b, policy="fast")

            comp.cancel_settle()
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                with comp._state_lock:
                    remaining = len(comp._active_settle_threads)
                if remaining == 0:
                    break
                time.sleep(0.01)

            with comp._state_lock:
                self.assertEqual(len(comp._active_settle_threads), 0)
            self.assertEqual(comp.settle_count, 0)
        finally:
            comp.close()

        # Stress zero-second timers against cancel_settle() and close()
        for _ in range(50):
            s_ctrl = FakeController()
            s_comp = EinkCompositor(s_ctrl, max_fps=1000.0, idle_settle_sec=0.0)
            s_comp.reset_framebuffer(255)
            f_s = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
            ImageDraw.Draw(f_s).rectangle([32, 32, 72, 72], fill=0)
            s_comp.present(f_s, policy="fast")
            s_comp.cancel_settle()
            s_comp.close()
            s_ctrl.exit_to_keyboard()
            self.assertLessEqual(s_ctrl.max_in_flight, 1)
            with s_comp._state_lock:
                self.assertEqual(len(s_comp._active_settle_threads), 0)

    def test_unchanged_frame_tight_loop_does_not_starve_idle_settle(self) -> None:
        f1 = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
        ImageDraw.Draw(f1).rectangle([20, 20, 80, 80], fill=0)

        for max_fps, settle_sec in ((12.0, 0.20), (1000.0, 0.15)):
            ctrl = FakeController()
            comp = EinkCompositor(ctrl, max_fps=max_fps, idle_settle_sec=settle_sec)
            stop = threading.Event()
            errors: list[Exception] = []
            iters = [0]

            def tight_loop() -> None:
                try:
                    while not stop.is_set():
                        st = comp.present(f1, policy="fast")
                        self.assertEqual(st.blits, 0)
                        iters[0] += 1
                except Exception as exc:  # pragma: no cover
                    errors.append(exc)

            try:
                comp.reset_framebuffer(255)
                st0 = comp.present(f1, policy="fast")
                self.assertTrue(st0.settle_armed)

                worker = threading.Thread(target=tight_loop)
                worker.start()
                try:
                    fired = ctrl.full_settle_event.wait(timeout=1.0)
                finally:
                    stop.set()
                    worker.join(timeout=2.0)

                comp.close()
                self.assertEqual(errors, [])
                self.assertTrue(
                    fired,
                    f"idle settle starved under tight unchanged loop (max_fps={max_fps}, iters={iters[0]})",
                )
                self.assertGreaterEqual(iters[0], 3)
                self.assertEqual(comp.settle_count, 1)
                self.assertEqual(ctrl.max_in_flight, 1)
            finally:
                stop.set()
                comp.close()

        # 2-thread overlapping tight unchanged-frame loop at max_fps=1000.0, idle_settle_sec=0.15
        ctrl2 = FakeController()
        comp2 = EinkCompositor(ctrl2, max_fps=1000.0, idle_settle_sec=0.15)
        stop2 = threading.Event()
        errors2: list[Exception] = []
        counts2 = [0, 0]
        barrier = threading.Barrier(2)

        def overlapping_loop(idx: int) -> None:
            try:
                barrier.wait(timeout=1.0)
                while not stop2.is_set():
                    st = comp2.present(f1, policy="fast")
                    self.assertEqual(st.blits, 0)
                    counts2[idx] += 1
            except Exception as exc:  # pragma: no cover
                errors2.append(exc)

        try:
            comp2.reset_framebuffer(255)
            st0 = comp2.present(f1, policy="fast")
            self.assertTrue(st0.settle_armed)

            t_a = threading.Thread(target=overlapping_loop, args=(0,))
            t_b = threading.Thread(target=overlapping_loop, args=(1,))
            t_a.start()
            t_b.start()
            try:
                fired2 = ctrl2.full_settle_event.wait(timeout=1.0)
            finally:
                stop2.set()
                t_a.join(timeout=2.0)
                t_b.join(timeout=2.0)

            comp2.close()
            self.assertEqual(errors2, [])
            self.assertTrue(
                fired2,
                f"idle settle starved under 2-thread tight unchanged loop (counts={counts2})",
            )
            self.assertGreaterEqual(sum(counts2), 4)
            self.assertEqual(comp2.settle_count, 1)
            self.assertEqual(ctrl2.max_in_flight, 1)
        finally:
            stop2.set()
            comp2.close()

    def test_external_rlock_concurrent_present_and_settle_draining_no_deadlock(self) -> None:
        """Regression for 3-thread external RLock + _present_in_flight + _settle_draining_gen deadlock."""
        f1 = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), 255)
        ImageDraw.Draw(f1).rectangle([20, 20, 80, 80], fill=0)
        f2 = f1.copy()
        ImageDraw.Draw(f2).rectangle([120, 120, 180, 180], fill=0)

        for reentrant_frame, expected_reentrant_blits in ((f1, 0), (f2, 1)):
            ctrl = FakeController()
            ext_rlock = threading.RLock()
            comp = EinkCompositor(
                ctrl,
                max_fps=1000.0,
                idle_settle_sec=0.05,
                usb_lock=ext_rlock,
            )
            errors: list[Exception] = []
            other_done = threading.Event()
            ext_done = threading.Event()
            reentrant_stats: list[PresentStats] = []

            try:
                comp.reset_framebuffer(255)
                st0 = comp.present(f1, policy="fast")
                self.assertTrue(st0.settle_armed)

                def run_other() -> None:
                    try:
                        comp.present(f1, policy="fast")
                    except Exception as exc:  # pragma: no cover
                        errors.append(exc)
                    finally:
                        other_done.set()

                def run_ext() -> None:
                    try:
                        with ext_rlock:
                            t_other = threading.Thread(target=run_other)
                            t_other.start()

                            # Wait for T_other to enter _present_in_flight and block on ext_rlock
                            deadline = time.monotonic() + 1.0
                            while time.monotonic() < deadline:
                                with comp._state_lock:
                                    if comp._present_in_flight >= 1:
                                        break
                                time.sleep(0.002)

                            # Wait for idle_settle_sec=0.05 to fire and enter _run_idle_settle
                            deadline = time.monotonic() + 1.0
                            saw_draining = False
                            while time.monotonic() < deadline:
                                with comp._state_lock:
                                    if comp._settle_draining_gen is not None:
                                        saw_draining = True
                                        break
                                time.sleep(0.002)
                            if not saw_draining:
                                time.sleep(0.06)

                            # Reentrant present() while ext_rlock is held and T_other is blocked on ext_rlock
                            st_re = comp.present(reentrant_frame, policy="fast")
                            reentrant_stats.append(st_re)
                            ext_done.set()

                        t_other.join(timeout=2.0)
                    except Exception as exc:  # pragma: no cover
                        errors.append(exc)
                        ext_done.set()

                t_ext = threading.Thread(target=run_ext)
                t_ext.start()

                self.assertTrue(
                    ext_done.wait(timeout=1.5),
                    "3-thread external RLock deadlock: T_ext hung in reentrant present()",
                )
                self.assertTrue(
                    other_done.wait(timeout=1.5),
                    "T_other did not complete after ext_rlock release",
                )
                t_ext.join(timeout=1.5)
                self.assertFalse(t_ext.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(len(reentrant_stats), 1)
                self.assertEqual(reentrant_stats[0].blits, expected_reentrant_blits)

                # Subsequent settle completes cleanly after both threads finish
                self.assertTrue(ctrl.full_settle_event.wait(timeout=1.0))
                self.assertEqual(comp.settle_count, 1)
                self.assertEqual(ctrl.max_in_flight, 1)
            finally:
                comp.close()


if __name__ == "__main__":
    unittest.main()
