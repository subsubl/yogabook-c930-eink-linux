"""Playable 3D E-Ink DOOM raycaster for the Lenovo Yoga Book C930 (1920x1080).

Features:
  - Real-time 3D first-person raycasting (Corridors, shaded walls, enemies, shotgun HUD)
  - On-screen touch controls directly on the E-Ink panel (via hidraw0 0x90):
      Left side:  [◀] [▲] [▼] [▶]
      Right side: [FIRE!]  and  [EXIT -> KB]
  - Fast DU2 E-Ink refresh (~3-5 FPS) with periodic anti-ghosting clean.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from PIL import Image, ImageDraw, ImageFont

from .compositor import EinkCompositor, bayer_dither_1bit
from .config import EinkSettings
from .hid_listener import EinkHidListener, EinkTouchEvent
from .image_render import pack_eink_grayscale
from .ownership import PanelOwnershipLease
from .protocol import (
    PANEL_HEIGHT,
    PANEL_WIDTH,
    WAVEFORM_GC16,
    Ite8951Controller,
)
from .tone_lut import apply_tone_lut, clamp_contrast, clamp_lightness

# Internal 3D render resolution (upscaled cleanly to 1920x820 viewport above HUD)
VIEW_W = 240
VIEW_H = 102
HUD_TOP = 820

WORLD_MAP = [
    "1111111111111111",
    "1000001000000001",
    "1011001001110001",
    "1010000001000101",
    "1000011000000101",
    "1000010001000001",
    "1011000001110001",
    "1000001000000001",
    "1001001000100101",
    "1000000000000001",
    "1111111111111111",
]
MAP_H = len(WORLD_MAP)
MAP_W = len(WORLD_MAP[0])


@dataclass
class Enemy:
    x: float
    y: float
    hp: int = 2
    alive: bool = True


@dataclass
class TouchButton:
    name: str
    label: str
    x: int
    y: int
    w: int
    h: int

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


class EinkDoomGame:
    """Self-contained playable 3D E-Ink DOOM raycaster with touchable HUD controls."""

    def __init__(
        self,
        lightness: Optional[int] = None,
        contrast: Optional[int] = None,
    ) -> None:
        if lightness is None or contrast is None:
            settings = EinkSettings.load()
            self.lightness = clamp_lightness(settings.lightness if lightness is None else lightness)
            self.contrast = clamp_contrast(settings.contrast if contrast is None else contrast)
        else:
            self.lightness = clamp_lightness(lightness)
            self.contrast = clamp_contrast(contrast)
        self.px = 2.5
        self.py = 1.5
        self.angle = 0.0
        self.fov = math.pi / 3.0
        self.health = 100
        self.ammo = 50
        self.kills = 0
        self.flash_frames = 0
        self.enemies = [
            Enemy(5.5, 3.5),
            Enemy(8.5, 4.5),
            Enemy(11.5, 3.5),
            Enemy(13.5, 7.5),
            Enemy(4.5, 8.5),
            Enemy(10.5, 9.5),
        ]
        self.font_hud = _load_font(34)
        self.font_btn = _load_font(38)

        # On-screen touch controls along the bottom HUD (y: 830..1065)
        self.buttons = [
            TouchButton("left", "◀", 30, 880, 150, 150),
            TouchButton("fwd", "▲", 200, 835, 150, 110),
            TouchButton("back", "▼", 200, 955, 150, 110),
            TouchButton("right", "▶", 370, 880, 150, 150),
            TouchButton("fire", "FIRE!", 1360, 845, 290, 210),
            TouchButton("exit", "EXIT", 1680, 845, 210, 210),
        ]

    def _is_wall(self, x: float, y: float) -> bool:
        mx, my = int(x), int(y)
        if mx < 0 or my < 0 or mx >= MAP_W or my >= MAP_H:
            return True
        return WORLD_MAP[my][mx] == "1"

    def move(self, dist: float) -> None:
        nx = self.px + math.cos(self.angle) * dist
        ny = self.py + math.sin(self.angle) * dist
        if not self._is_wall(nx, self.py):
            self.px = nx
        if not self._is_wall(self.px, ny):
            self.py = ny

    def turn(self, delta: float) -> None:
        self.angle = (self.angle + delta) % (2.0 * math.pi)

    def shoot(self) -> None:
        if self.ammo <= 0:
            return
        self.ammo -= 1
        self.flash_frames = 2
        # Hitscan check against nearest enemy near crosshair
        best: Optional[Enemy] = None
        best_dist = 999.0
        for e in self.enemies:
            if not e.alive:
                continue
            dx, dy = e.x - self.px, e.y - self.py
            dist = math.hypot(dx, dy)
            ang = math.atan2(dy, dx)
            diff = (ang - self.angle + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) < 0.22 and dist < best_dist:
                best = e
                best_dist = dist
        if best is not None:
            best.hp -= 1
            if best.hp <= 0:
                best.alive = False
                self.kills += 1

    def render_frame(
        self,
        lightness: Optional[int] = None,
        contrast: Optional[int] = None,
    ) -> tuple[Image.Image, bytes]:
        eff_lightness = clamp_lightness(self.lightness if lightness is None else lightness)
        eff_contrast = clamp_contrast(self.contrast if contrast is None else contrast)
        # 1. Raycast 3D scene onto compact buffer for speed
        view = Image.new("L", (VIEW_W, VIEW_H), color=240)
        vdraw = ImageDraw.Draw(view)

        # Ceiling (light gray) & Floor (darker dithered shade)
        vdraw.rectangle([(0, 0), (VIEW_W, VIEW_H // 2)], fill=235)
        vdraw.rectangle([(0, VIEW_H // 2), (VIEW_W, VIEW_H)], fill=170)

        zbuf = [99.0] * VIEW_W
        for col in range(VIEW_W):
            ray_angle = self.angle - self.fov / 2.0 + (col / VIEW_W) * self.fov
            sin_a = math.sin(ray_angle)
            cos_a = math.cos(ray_angle)

            dist = 0.1
            hit = False
            while dist < 14.0:
                rx = self.px + cos_a * dist
                ry = self.py + sin_a * dist
                if self._is_wall(rx, ry):
                    hit = True
                    break
                dist += 0.08

            if not hit:
                continue
            # Fish-eye correction
            corr_dist = max(0.2, dist * math.cos(ray_angle - self.angle))
            zbuf[col] = corr_dist

            wall_h = int(VIEW_H / corr_dist)
            top = max(0, (VIEW_H - wall_h) // 2)
            bot = min(VIEW_H - 1, (VIEW_H + wall_h) // 2)

            # High-contrast E-Ink wall shading + brick edge striping
            fx = rx - int(rx)
            fy = ry - int(ry)
            is_edge = (fx < 0.06 or fx > 0.94) and (fy < 0.06 or fy > 0.94)
            shade = 0 if is_edge else min(210, int(20 + corr_dist * 24))
            vdraw.line([(col, top), (col, bot)], fill=shade)

        # 2. Draw enemies (billboard sprites) sorted far-to-near
        sorted_enemies = sorted(
            [e for e in self.enemies if e.alive],
            key=lambda e: math.hypot(e.x - self.px, e.y - self.py),
            reverse=True,
        )
        for e in sorted_enemies:
            dx, dy = e.x - self.px, e.y - self.py
            dist = max(0.3, math.hypot(dx, dy))
            ang = math.atan2(dy, dx)
            diff = (ang - self.angle + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) > self.fov * 0.65:
                continue
            sx = int((0.5 + diff / self.fov) * VIEW_W)
            size = min(VIEW_H, int(VIEW_H * 0.85 / dist))
            if 0 <= sx < VIEW_W and dist < zbuf[sx]:
                top = (VIEW_H - size) // 2 + size // 8
                w = max(4, size // 2)
                # Imp silhouette + glowing white eyes
                vdraw.ellipse([(sx - w // 2, top), (sx + w // 2, top + size // 2)], fill=10, outline=255)
                vdraw.rectangle([(sx - w // 3, top + size // 2), (sx + w // 3, top + size)], fill=20)
                eye_y = top + size // 5
                vdraw.point([(sx - max(1, w // 6), eye_y), (sx + max(1, w // 6), eye_y)], fill=255)

        # 3. Apply Glider Tone LUT then 4x4 Ordered Bayer 1-bit dithering to the 3D scene
        toned_view = apply_tone_lut(view, lightness=eff_lightness, contrast=eff_contrast)
        dithered_view = bayer_dither_1bit(toned_view)
        frame = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=0)
        up = dithered_view.resize((PANEL_WIDTH, HUD_TOP), Image.Resampling.NEAREST)
        frame.paste(up, (0, 0))
        draw = ImageDraw.Draw(frame)
        # Strict 1-bit font rendering (zero anti-aliasing gray pixels)
        draw.fontmode = "1"

        # Crosshair
        cx, cy = PANEL_WIDTH // 2, HUD_TOP // 2
        draw.line([(cx - 24, cy), (cx + 24, cy)], fill=0, width=4)
        draw.line([(cx, cy - 24), (cx, cy + 24)], fill=0, width=4)

        # Classic centered DOOM shotgun barrel + muzzle flash
        if self.flash_frames > 0:
            draw.polygon(
                [(cx, HUD_TOP - 290), (cx - 90, HUD_TOP - 170), (cx + 90, HUD_TOP - 170)],
                fill=255,
                outline=0,
            )
            self.flash_frames -= 1
        draw.polygon(
            [(cx - 50, HUD_TOP), (cx - 24, HUD_TOP - 190), (cx + 24, HUD_TOP - 190), (cx + 50, HUD_TOP)],
            fill=0,
            outline=255,
        )
        draw.rectangle([(cx - 12, HUD_TOP - 190), (cx + 12, HUD_TOP - 140)], fill=255)

        # Bottom DOOM HUD bar (strict 1-bit black/white)
        draw.rectangle([(0, HUD_TOP), (PANEL_WIDTH, PANEL_HEIGHT)], fill=0)
        draw.line([(0, HUD_TOP), (PANEL_WIDTH, HUD_TOP)], fill=255, width=4)

        # Status readout (center of HUD)
        alive_left = sum(1 for e in self.enemies if e.alive)
        hud_text = f"E-INK DOOM   |   HP: {self.health}%   |   AMMO: {self.ammo}   |   KILLS: {self.kills}/{len(self.enemies)}"
        draw.text((560, 865), hud_text, fill=255, font=self.font_hud)
        sub_text = (
            "VICTORY! ALL IMPS CLEARED — TAP EXIT"
            if alive_left == 0
            else "Tap ◀ ▲ ▼ ▶ to move/turn  •  Tap FIRE! to shoot  •  Tap EXIT for Keyboard"
        )
        draw.text((560, 945), sub_text, fill=255, font=_load_font(22))

        # Touch buttons (strict 1-bit)
        for b in self.buttons:
            fill = 255 if b.name == "fire" else 0
            txt_col = 0 if b.name == "fire" else 255
            draw.rounded_rectangle(
                [(b.x, b.y), (b.x + b.w, b.y + b.h)],
                radius=18,
                fill=fill,
                outline=255,
                width=4,
            )
            bb = draw.textbbox((0, 0), b.label, font=self.font_btn)
            draw.text(
                (b.x + (b.w - (bb[2] - bb[0])) // 2, b.y + (b.h - (bb[3] - bb[1])) // 2 - 4),
                b.label,
                fill=txt_col,
                font=self.font_btn,
            )

        return frame, pack_eink_grayscale(frame)


def run_eink_doom(
    stop_event: threading.Event,
    restore_kb_on_exit: bool = True,
    gc16_budget: int = 35,
    on_status: Optional[Callable[[str], None]] = None,
) -> None:
    """Run playable 3D E-Ink DOOM with 4x4 Bayer dithering, dirty-rect compositor, and lease."""
    settings = EinkSettings.load()
    game = EinkDoomGame(lightness=settings.lightness, contrast=settings.contrast)
    dirty = threading.Event()
    dirty.set()

    def _on_touch(ev: EinkTouchEvent) -> None:
        if not ev.is_down:
            return
        for b in game.buttons:
            if b.contains(ev.panel_x, ev.panel_y):
                if b.name == "fwd":
                    game.move(0.45)
                elif b.name == "back":
                    game.move(-0.45)
                elif b.name == "left":
                    game.turn(-0.32)
                elif b.name == "right":
                    game.turn(0.32)
                elif b.name == "fire":
                    game.shoot()
                elif b.name == "exit":
                    stop_event.set()
                dirty.set()
                return

    listener = EinkHidListener(on_touch=_on_touch)
    ok, msg = listener.start()
    if on_status:
        on_status(f"E-Ink DOOM started ({msg})")

    try:
        with PanelOwnershipLease("doom") as lease:
            with Ite8951Controller() as ctrl:
                ctrl.enter_app_mode(clean_color=0xFF)
                comp = EinkCompositor(
                    ctrl,
                    gc16_budget=gc16_budget,
                    max_fps=12.0,
                    strict_1bit=True,
                    idle_settle_enabled=settings.idle_settle_enabled,
                    idle_settle_sec=settings.idle_settle_sec,
                )
                try:
                    comp.reset_framebuffer(fill_color=255)

                    while not stop_event.is_set():
                        if not dirty.wait(timeout=0.15):
                            continue
                        dirty.clear()
                        if stop_event.is_set():
                            break
                        frame_img, _ = game.render_frame()
                        comp.present(frame_img, policy="fast", clean_waveform=WAVEFORM_GC16)
                        if game.flash_frames > 0:
                            dirty.set()
                finally:
                    comp.close()

                if restore_kb_on_exit:
                    ctrl.exit_to_keyboard()
                    lease.mark_clean_exit()
    finally:
        listener.stop()
        if on_status:
            on_status("Exited E-Ink DOOM.")
