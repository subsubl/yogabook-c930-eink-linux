"""Authentic Matrix Digital Rain screensaver for the Yoga Book C930 (1920x1080 E-Ink).

- Uses real Japanese half-width Katakana + mirrored Katakana/digits (NotoSansCJK-Bold),
  matching the iconic Matrix movie glyphs rather than plain ASCII.
- Pre-renders a 4-level 4x4 Bayer-dithered strict 1-bit {0, 255} sprite atlas so trails
  visibly fade from bright leading head -> dense -> sparse -> black while staying 100%
  in fast DU2 mode.
- Uses full-frame DU2 blits (avoiding narrow x!=0 partial stride glitches) + white INIT
  clean on entry so zero keyboard ghosting bleeds through.
- Crash-safe PanelOwnershipLease + 0.6s launch-tap grace period + clean keyboard exit.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Callable, Optional

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .compositor import bayer_dither_1bit
from .hid_listener import EinkHidListener, EinkTouchEvent
from .image_render import pack_eink_grayscale
from .ownership import PanelOwnershipLease
from .protocol import (
    PANEL_HEIGHT,
    PANEL_WIDTH,
    WAVEFORM_DU2,
    WAVEFORM_GC16,
    WAVEFORM_INIT,
    Ite8951Controller,
)

# Authentic Matrix glyph set: half-width Katakana + digits + Matrix symbols
MATRIX_KATAKANA = "ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ"
MATRIX_ASCII = "012345789Z:=*+-<>¦｜"
MATRIX_CHARS = MATRIX_KATAKANA + MATRIX_ASCII


def _load_matrix_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _load_badge_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


class MatrixRainRenderer:
    """Renders authentic mirrored-Katakana Matrix Digital Rain at 1920x1080 in strict 1-bit."""

    def __init__(self, dark_bg: bool = True, cols: int = 60, rows: int = 27) -> None:
        self.dark_bg = dark_bg
        self.cols = cols
        self.rows = rows
        self.cell_w = PANEL_WIDTH // cols  # 32px
        self.cell_h = PANEL_HEIGHT // rows  # 40px
        self.font = _load_matrix_font(int(self.cell_h * 0.78))
        self.badge_font = _load_badge_font(30)

        # Build pre-rendered 1-bit sprite atlas: (char_idx, level 0..3) -> Image
        # level 0 = bright glowing head (inverted highlight block or solid bold),
        # level 1 = full crisp glyph, level 2 = 65% Bayer dithered, level 3 = 35% Bayer dithered
        self._atlas = self._build_glyph_atlas()
        self._num_glyphs = len(MATRIX_CHARS) * 2  # normal + horizontally mirrored

        # Stream state per column
        self.heads = [random.randint(-rows, rows) for _ in range(cols)]
        self.speeds = [random.choice([1, 1, 1, 2]) for _ in range(cols)]
        self.lengths = [random.randint(6, 18) for _ in range(cols)]
        self.grid = [
            [random.randrange(self._num_glyphs) for _ in range(cols)]
            for _ in range(rows)
        ]

        bg = 0 if self.dark_bg else 255
        self.canvas = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=bg)

    def _build_glyph_atlas(self) -> list[list[Image.Image]]:
        """Pre-render normal + horizontally mirrored glyphs across 4 strict 1-bit intensity tiers."""
        bg = 0 if self.dark_bg else 255
        fg = 255 if self.dark_bg else 0
        # Intermediate gray intensities before 4x4 Bayer dithering for trail fade
        trail_shades = (
            (255, 195, 115) if self.dark_bg else (0, 65, 145)
        )

        atlas: list[list[Image.Image]] = []
        base_chars = list(MATRIX_CHARS)
        # Render each char once, then also create a horizontally mirrored twin (iconic Matrix look)
        for mirror in (False, True):
            for ch in base_chars:
                # Render crisp glyph centered in cell
                base = Image.new("L", (self.cell_w, self.cell_h), color=0)
                d = ImageDraw.Draw(base)
                d.fontmode = "1"
                bb = d.textbbox((0, 0), ch, font=self.font)
                gw = bb[2] - bb[0]
                gh = bb[3] - bb[1]
                gx = (self.cell_w - gw) // 2 - bb[0]
                gy = (self.cell_h - gh) // 2 - bb[1]
                d.text((gx, gy), ch, fill=255, font=self.font)
                if mirror:
                    base = ImageOps.mirror(base)

                mask = base.point(lambda p: 255 if p >= 128 else 0)
                variants: list[Image.Image] = []

                # Tier 0: Bright leading head (high-contrast inverted pill around glyph)
                head_img = Image.new("L", (self.cell_w, self.cell_h), color=fg)
                head_img.paste(bg, (0, 0), mask)
                variants.append(head_img)

                # Tiers 1..3: Fading trail via 4x4 Bayer ordered dithering
                for shade in trail_shades:
                    tier = Image.new("L", (self.cell_w, self.cell_h), color=bg)
                    fill_tile = Image.new("L", (self.cell_w, self.cell_h), color=shade)
                    tier.paste(fill_tile, (0, 0), mask)
                    if shade not in (0, 255):
                        tier = bayer_dither_1bit(tier)
                    variants.append(tier)

                atlas.append(variants)
        return atlas

    def _draw_exit_badge(self, img: Image.Image) -> None:
        draw = ImageDraw.Draw(img)
        draw.fontmode = "1"
        fg = 255 if self.dark_bg else 0
        bg = 0 if self.dark_bg else 255
        draw.rectangle([(720, 980), (1200, 1056)], fill=fg, outline=bg, width=4)
        draw.text((772, 1002), "← TAP TO EXIT", fill=bg, font=self.badge_font)

    def step_frame(self, cols_per_tick: int = 60) -> Image.Image:
        """Advance the Matrix rain streams and render a crisp 1-bit frame."""
        bg = 0 if self.dark_bg else 255
        img = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=bg)

        for c in range(self.cols):
            self.heads[c] += self.speeds[c]
            head = self.heads[c]
            trail = self.lengths[c]

            if head - trail > self.rows:
                self.heads[c] = random.randint(-14, -1)
                self.speeds[c] = random.choice([1, 1, 1, 2])
                self.lengths[c] = random.randint(6, 18)
                continue

            # Mutate leading glyph + occasional random glyphs inside the stream
            if 0 <= head < self.rows:
                self.grid[head][c] = random.randrange(self._num_glyphs)
            if 0 <= head - 1 < self.rows and random.random() < 0.35:
                self.grid[head - 1][c] = random.randrange(self._num_glyphs)
            if trail > 4 and random.random() < 0.25:
                mr = head - random.randint(2, trail - 1)
                if 0 <= mr < self.rows:
                    self.grid[mr][c] = random.randrange(self._num_glyphs)

            x0 = c * self.cell_w
            r_start = max(0, head - trail)
            r_end = min(self.rows - 1, head)
            for r in range(r_start, r_end + 1):
                dist = head - r
                if dist == 0:
                    tier = 0
                elif dist <= trail // 3:
                    tier = 1
                elif dist <= (trail * 2) // 3:
                    tier = 2
                else:
                    tier = 3
                sprite = self._atlas[self.grid[r][c]][tier]
                img.paste(sprite, (x0, r * self.cell_h))

        self.canvas = img
        return img


def run_matrix_screensaver(
    stop_event: threading.Event,
    dark_bg: bool = True,
    gc16_budget: int = 40,
    restore_kb_on_exit: bool = True,
    on_frame: Optional[Callable[[int], None]] = None,
) -> None:
    """Run authentic Matrix Digital Rain with clean entry, full-frame DU2 updates, and clean exit."""
    renderer = MatrixRainRenderer(dark_bg=dark_bg)
    frame_num = 0
    started_at = time.monotonic()

    def _on_touch(ev: EinkTouchEvent) -> None:
        # Ignore the launch tap for 0.6s so tapping MATRIX RAIN on the Dashboard doesn't exit immediately
        if ev.is_down and (time.monotonic() - started_at) > 0.6:
            stop_event.set()

    listener = EinkHidListener(on_touch=_on_touch)
    listener.start()

    try:
        with PanelOwnershipLease("matrix") as lease:
            with Ite8951Controller() as ctrl:
                # Always flash white INIT first so previous keyboard/menu charge is completely wiped
                ctrl.enter_app_mode(clean_color=0xFF)
                if dark_bg:
                    ctrl.clear_screen(fill_byte=0x0F, waveform=WAVEFORM_DU2)

                while not stop_event.is_set():
                    frame = renderer.step_frame()
                    packed = pack_eink_grayscale(frame)
                    frame_num += 1

                    # Full-frame DU2 avoids narrow x!=0 stride glitches; periodic GC16 wipes ghosting
                    wf = WAVEFORM_GC16 if (gc16_budget > 0 and frame_num % gc16_budget == 0) else WAVEFORM_DU2
                    ctrl.display_image_bytes(packed, PANEL_WIDTH, PANEL_HEIGHT, waveform=wf)

                    if on_frame:
                        on_frame(frame_num)

                    # Gentle pace (~6-8 FPS) so DU2 refreshes stay crisp and responsive to taps
                    stop_event.wait(0.04)

                if restore_kb_on_exit:
                    ctrl.exit_to_keyboard()
                    lease.mark_clean_exit()
    finally:
        listener.stop()
