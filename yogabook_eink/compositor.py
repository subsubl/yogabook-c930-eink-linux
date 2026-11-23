"""Dual-Framebuffer Dirty-Rectangle Compositor for Yoga Book C930 E-Ink.

Implements strict E-Ink rendering best practices:
  1. Strict 1-bit monochrome fast path (0x0F black / 0xFF white) to stay in DU2/A2 mode.
  2. Real-time 4x4 Ordered Bayer Matrix dithering for shaded 3D/grayscale graphics.
  3. Dual-framebuffer (prev_frame vs curr_frame) tile-grid diffing with KOReader-style
     8-pixel alignment (`dither_alignment_constraint = 8`) across all rect paths.
  4. Semantic Refresh Policy (`fast`, `ui`, `partial`, `full`, `clean`) mapping to
     ITE8951 waveforms (`DU2`, `GL16`, `GC16`, `INIT`) and pixel formats.
  5. Thread-safe USB locking (`_usb_lock -> _state_lock`) and ~2s idle `GC16`
     anti-ghosting settle refresh after DU2 bursts stop.
"""

from __future__ import annotations

import functools
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageChops

from .image_render import pack_eink_grayscale, pack_eink_region
from .protocol import (
    PANEL_HEIGHT,
    PANEL_WIDTH,
    WAVEFORM_DU2,
    WAVEFORM_GC16,
    WAVEFORM_GL16,
    WAVEFORM_INIT,
    Ite8951Controller,
)
from .usb_lib import UsbError

DITHER_ALIGNMENT: int = 8
dither_alignment_constraint: int = 8

SEMANTIC_POLICIES: tuple[str, ...] = ("fast", "ui", "partial", "full", "clean")
SEMANTIC_REFRESH_POLICIES: tuple[str, ...] = SEMANTIC_POLICIES

# 4x4 Bayer ordered dithering threshold matrix (0..255)
_BAYER_4X4 = (
    (15, 143, 47, 175),
    (207, 79, 239, 111),
    (63, 191, 31, 159),
    (255, 127, 223, 95),
)

# Pre-built tiled 4x4 Bayer threshold image (cached lazily per resolution)
_BAYER_CACHE: dict[tuple[int, int], bytes] = {}


def _get_bayer_bytes(w: int, h: int) -> bytes:
    key = (w, h)
    cached = _BAYER_CACHE.get(key)
    if cached is not None:
        return cached
    tile = Image.new("L", (4, 4))
    px = tile.load()
    for y in range(4):
        for x in range(4):
            px[x, y] = _BAYER_4X4[y][x]
    # Tile out to (w, h) using fast PIL paste
    row_img = Image.new("L", (w, 4))
    for x in range(0, w, 4):
        row_img.paste(tile, (x, 0))
    full = Image.new("L", (w, h))
    for y in range(0, h, 4):
        full.paste(row_img, (0, y))
    data = full.tobytes()
    _BAYER_CACHE[key] = data
    return data


def bayer_dither_1bit(img: Image.Image) -> Image.Image:
    """Apply fast 4x4 Ordered Bayer dithering, returning a strict 0/255 'L' image."""
    gray = img.convert("L") if img.mode != "L" else img
    w, h = gray.size
    src = gray.tobytes()
    thresh = _get_bayer_bytes(w, h)
    # Strict 1-bit output: 255 if pixel >= bayer_threshold else 0
    out = bytes(255 if a >= b else 0 for a, b in zip(src, thresh))
    return Image.frombytes("L", (w, h), out)


@functools.lru_cache(maxsize=16)
def _threshold_lut(cutoff: int) -> list[int]:
    return [255 if i >= cutoff else 0 for i in range(256)]


def threshold_1bit(img: Image.Image, cutoff: int = 128) -> Image.Image:
    """Fast strict 1-bit monochrome threshold (0 or 255 only — zero intermediate grays)."""
    gray = img.convert("L") if img.mode != "L" else img
    return gray.point(_threshold_lut(cutoff))


@dataclass(frozen=True)
class DirtyRect:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h


@dataclass(frozen=True)
class PresentStats:
    blits: int
    policy: str
    waveform: Optional[int]
    full_refresh: bool
    rects: tuple[DirtyRect, ...] = ()
    dirty_area: int = 0
    settle_armed: bool = False

    def __int__(self) -> int:
        return self.blits

    def __bool__(self) -> bool:
        return self.blits > 0


def _validate_alignment(alignment: Optional[int]) -> int:
    if alignment is None:
        return 1
    if (
        not isinstance(alignment, int)
        or isinstance(alignment, bool)
        or alignment <= 0
        or (alignment & (alignment - 1)) != 0
    ):
        raise ValueError(f"alignment must be a positive power of two, got {alignment!r}")
    return alignment


def align_rect(
    x: int | float,
    y: int | float,
    w: int | float,
    h: int | float,
    max_w: int = PANEL_WIDTH,
    max_h: int = PANEL_HEIGHT,
    alignment: Optional[int] = DITHER_ALIGNMENT,
) -> Optional[tuple[int, int, int, int]]:
    """Bound and expand (x, y, w, h) to `alignment` (KOReader getBoundedRect + degenerate filter).

    Returns `(x, y, w, h)` clamped within `[0..max_w) x [0..max_h)`, or `None` if empty,
    out-of-bounds, or degenerate (`w <= 1 or h <= 1`) after alignment and clamping.
    """
    a = _validate_alignment(alignment)

    for name, val in (("x", x), ("y", y), ("w", w), ("h", h)):
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(f"{name} must be a finite int or float, got {val!r}")
    for name, val in (("max_w", max_w), ("max_h", max_h)):
        if isinstance(val, bool) or not isinstance(val, int):
            raise ValueError(f"{name} must be a non-bool int, got {val!r}")

    if max_w <= 1 or max_h <= 1 or w <= 0 or h <= 0 or x >= max_w or y >= max_h:
        return None

    if x < 0:
        w += x
        x = 0
    if y < 0:
        h += y
        y = 0
    if w <= 0 or h <= 0:
        return None

    ix = math.floor(x)
    iy = math.floor(y)
    iw = math.ceil(min(max_w, x + w)) - ix
    ih = math.ceil(min(max_h, y + h)) - iy
    if iw <= 0 or ih <= 0:
        return None

    if a > 1:
        mask = a - 1
        ax = ix & ~mask
        iw = (iw + (ix - ax) + mask) & ~mask
        ix = ax
        ay = iy & ~mask
        ih = (ih + (iy - ay) + mask) & ~mask
        iy = ay

    if ix + iw > max_w:
        iw = max_w - ix
    if iy + ih > max_h:
        ih = max_h - iy

    if iw <= 1 or ih <= 1:
        return None

    return (ix, iy, iw, ih)


def _merge_overlapping_rects(rects: list[DirtyRect]) -> list[DirtyRect]:
    """Merge strictly overlapping rects to a fixed point without merging disjoint neighbors."""
    merged = list(rects)
    changed = True
    while changed:
        changed = False
        out: list[DirtyRect] = []
        for r in merged:
            rx0, ry0, rx1, ry1 = r.x, r.y, r.x + r.w, r.y + r.h
            did_merge = False
            for idx, e in enumerate(out):
                ex0, ey0, ex1, ey1 = e.x, e.y, e.x + e.w, e.y + e.h
                if rx0 < ex1 and ex0 < rx1 and ry0 < ey1 and ey0 < ry1:
                    nx0 = min(rx0, ex0)
                    ny0 = min(ry0, ey0)
                    nx1 = max(rx1, ex1)
                    ny1 = max(ry1, ey1)
                    out[idx] = DirtyRect(nx0, ny0, nx1 - nx0, ny1 - ny0)
                    did_merge = True
                    changed = True
                    break
            if not did_merge:
                out.append(r)
        merged = out
    return merged


def _align_axis_cover(r0: int, r1: int, dim: int, a: int) -> tuple[int, int]:
    """Align integer span [r0, r1) within [0, dim) ensuring span > 1 when dim >= 2."""
    if a == 1:
        ax = r0
        aw = r1 - r0
        if aw <= 1:
            if ax + 2 <= dim:
                aw = 2
            else:
                ax = dim - 2
                aw = 2
        return ax, aw
    mask = a - 1
    ax = r0 & ~mask
    aw = ((r1 - ax) + mask) & ~mask
    if ax + aw > dim:
        aw = dim - ax
    if aw <= 1:
        ax = max(0, ax - a)
        aw = min(dim - ax, ((r1 - ax) + mask) & ~mask)
    return ax, aw


def _align_dirty_bbox(
    rx0: int,
    ry0: int,
    rx1: int,
    ry1: int,
    img_w: int,
    img_h: int,
    a: int,
) -> Optional[DirtyRect]:
    """Align a non-empty dirty bbox, falling back on trailing-edge clamping for dim % a == 1."""
    aligned = align_rect(rx0, ry0, rx1 - rx0, ry1 - ry0, max_w=img_w, max_h=img_h, alignment=a)
    if aligned is not None:
        return DirtyRect(*aligned)
    if rx1 > rx0 and ry1 > ry0 and img_w >= 2 and img_h >= 2:
        ax, aw = _align_axis_cover(rx0, rx1, img_w, a)
        ay, ah = _align_axis_cover(ry0, ry1, img_h, a)
        if aw > 1 and ah > 1:
            return DirtyRect(ax, ay, aw, ah)
    return None


def compute_dirty_rects(
    prev_img: Image.Image,
    curr_img: Image.Image,
    tile_w: int = 80,
    tile_h: int = 60,
    max_rects: int = 12,
    alignment: Optional[int] = DITHER_ALIGNMENT,
) -> list[DirtyRect]:
    """Diff prev_img vs curr_img on a coarse tile grid and emit aligned dirty rects.

    Avoids merging distant small updates into one giant full-screen bounding box while
    guaranteeing every emitted DirtyRect is aligned to `alignment` (default 8px) and
    bounded within the image dimensions.
    """
    a = _validate_alignment(alignment)

    if prev_img.size != curr_img.size:
        raise ValueError(
            f"Image size mismatch in compute_dirty_rects: {prev_img.size} != {curr_img.size}"
        )

    prev_l = prev_img if prev_img.mode == "L" else prev_img.convert("L")
    curr_l = curr_img if curr_img.mode == "L" else curr_img.convert("L")

    diff = ImageChops.difference(prev_l, curr_l)
    overall = diff.getbbox()
    if overall is None:
        return []

    img_w, img_h = curr_l.size
    ox0, oy0, ox1, oy1 = overall

    # Fast path: small overall bbox (<= 61440 px) routed through align_rect
    if (ox1 - ox0) * (oy1 - oy0) <= 61440:
        rect = _align_dirty_bbox(ox0, oy0, ox1, oy1, img_w, img_h, a)
        return [rect] if rect is not None else []

    cols = (img_w + tile_w - 1) // tile_w
    rows = (img_h + tile_h - 1) // tile_h

    c0 = max(0, ox0 // tile_w)
    c1 = min(cols - 1, (ox1 - 1) // tile_w)
    r0 = max(0, oy0 // tile_h)
    r1 = min(rows - 1, (oy1 - 1) // tile_h)

    dirty: set[tuple[int, int]] = set()
    for r in range(r0, r1 + 1):
        ty0 = r * tile_h
        ty1 = min(img_h, ty0 + tile_h)
        for c in range(c0, c1 + 1):
            tx0 = c * tile_w
            tx1 = min(img_w, tx0 + tile_w)
            if diff.crop((tx0, ty0, tx1, ty1)).getbbox() is not None:
                dirty.add((c, r))

    if not dirty:
        return []

    # 8-neighbor connected components keep distant dirty regions separate
    visited: set[tuple[int, int]] = set()
    rects: list[DirtyRect] = []
    for cell in dirty:
        if cell in visited:
            continue
        stack = [cell]
        visited.add(cell)
        min_c = max_c = cell[0]
        min_r = max_r = cell[1]
        while stack:
            cc, rr = stack.pop()
            if cc < min_c:
                min_c = cc
            if cc > max_c:
                max_c = cc
            if rr < min_r:
                min_r = rr
            if rr > max_r:
                max_r = rr
            for dc in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    nb = (cc + dc, rr + dr)
                    if nb in dirty and nb not in visited:
                        visited.add(nb)
                        stack.append(nb)

        rx0 = max(ox0, min_c * tile_w)
        ry0 = max(oy0, min_r * tile_h)
        rx1 = min(img_w, min(ox1, (max_c + 1) * tile_w))
        ry1 = min(img_h, min(oy1, (max_r + 1) * tile_h))
        rect = _align_dirty_bbox(rx0, ry0, rx1, ry1, img_w, img_h, a)
        if rect is not None:
            rects.append(rect)

    rects = _merge_overlapping_rects(rects)

    # If too many tiny fragments, collapse to overall aligned bbox
    if len(rects) > max_rects:
        rect = _align_dirty_bbox(ox0, oy0, ox1, oy1, img_w, img_h, a)
        return [rect] if rect is not None else []

    return rects


class _SettleTimer(threading.Timer):
    """Timer subclass that notifies `on_exit` even when cancelled before firing."""

    def __init__(
        self,
        interval: float,
        function,
        args=None,
        kwargs=None,
        on_exit=None,
    ) -> None:
        super().__init__(interval, function, args=args, kwargs=kwargs)
        self._on_exit = on_exit

    def run(self) -> None:
        try:
            super().run()
        finally:
            cb = self._on_exit
            self._on_exit = None
            if cb is not None:
                cb(self)


class EinkCompositor:
    """Dual-framebuffer dirty-rect compositor with Semantic Refresh Policy and idle GC16 settle."""

    def __init__(
        self,
        ctrl: Ite8951Controller,
        gc16_budget: int = 35,
        max_fps: float = 12.0,
        strict_1bit: bool = True,
        idle_settle_enabled: bool = True,
        idle_settle_sec: float = 2.0,
        usb_lock: Optional[threading.RLock | threading.Lock] = None,
    ) -> None:
        self.ctrl = ctrl
        self.gc16_budget = max(5, gc16_budget)
        self.min_frame_interval = 1.0 / max(1.0, max_fps)
        self.strict_1bit = strict_1bit
        self.idle_settle_enabled = bool(idle_settle_enabled)
        self.idle_settle_sec = max(0.0, float(idle_settle_sec))

        # Strict lock hierarchy: _usb_lock -> _state_lock (never inverted).
        self._usb_lock = usb_lock if usb_lock is not None else threading.RLock()
        self._lock = self._usb_lock
        self._state_lock = threading.RLock()
        self._present_cond = threading.Condition(self._state_lock)

        self.prev_frame: Optional[Image.Image] = None
        self.partial_count: int = 0
        self.settle_count: int = 0
        self._last_present_ts: float = 0.0
        self._settle_timer: Optional[threading.Timer] = None
        self._active_settle_threads: set[threading.Thread] = set()
        self._settle_armed: bool = False
        self._settle_generation: int = 0
        self._settle_draining_gen: Optional[int] = None
        self._reset_generation: int = 0
        self._present_in_flight: int = 0
        self._present_exit_seq: int = 0
        self._closed: bool = False

    @property
    def settle_armed(self) -> bool:
        with self._state_lock:
            return self._settle_armed

    @property
    def is_closed(self) -> bool:
        with self._state_lock:
            return self._closed

    def __enter__(self) -> "EinkCompositor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _prune_dead_settle_threads_locked(self) -> None:
        if self._active_settle_threads:
            dead = [t for t in self._active_settle_threads if not t.is_alive()]
            for t in dead:
                self._active_settle_threads.discard(t)

    def _on_settle_thread_exit(self, thread: threading.Thread) -> None:
        with self._state_lock:
            if self._settle_timer is thread:
                self._settle_timer = None
                self._settle_armed = False
            self._active_settle_threads.discard(thread)

    def _cancel_settle_timer(self) -> None:
        with self._state_lock:
            self._settle_generation += 1
            self._settle_draining_gen = None
            timer = self._settle_timer
            self._settle_timer = None
            self._settle_armed = False
            self._prune_dead_settle_threads_locked()
            self._present_cond.notify_all()
        if timer is not None:
            timer.cancel()

    def cancel_settle(self) -> None:
        """Cancel any pending idle GC16 settle refresh cleanly."""
        self._cancel_settle_timer()

    def _arm_settle_timer(self, expected_reset_gen: Optional[int] = None) -> bool:
        """Arm a fresh idle GC16 settle timer if enabled and not closed."""
        with self._state_lock:
            self._prune_dead_settle_threads_locked()
            self._settle_generation += 1
            self._settle_draining_gen = None
            self._present_cond.notify_all()
            gen = self._settle_generation
            old_timer = self._settle_timer
            stale_reset = (
                expected_reset_gen is not None
                and expected_reset_gen != self._reset_generation
            )
            if not self.idle_settle_enabled or self._closed or stale_reset:
                self._settle_timer = None
                self._settle_armed = False
                armed = False
            else:
                new_timer = _SettleTimer(
                    self.idle_settle_sec,
                    self._run_idle_settle,
                    args=(gen,),
                    on_exit=self._on_settle_thread_exit,
                )
                new_timer.daemon = True
                self._settle_timer = None
                self._settle_armed = False
                self._active_settle_threads.add(new_timer)
                try:
                    new_timer.start()
                except Exception:
                    self._active_settle_threads.discard(new_timer)
                    if old_timer is not None:
                        old_timer.cancel()
                    raise
                self._settle_timer = new_timer
                self._settle_armed = True
                armed = True

        if old_timer is not None:
            old_timer.cancel()
        return armed

    def _run_idle_settle(self, expected_gen: int) -> None:
        cur = threading.current_thread()
        try:
            with self._state_lock:
                if self._closed or expected_gen != self._settle_generation or self.prev_frame is None:
                    return

            acquired = False
            while not acquired:
                with self._state_lock:
                    if self._closed or expected_gen != self._settle_generation or self.prev_frame is None:
                        return
                    self._settle_draining_gen = expected_gen
                    while (
                        not self._closed
                        and expected_gen == self._settle_generation
                        and self.prev_frame is not None
                        and self._present_in_flight > 0
                    ):
                        exit_seq_before = self._present_exit_seq
                        self._present_cond.wait(timeout=0.05)
                        if self._present_in_flight > 0 and self._present_exit_seq == exit_seq_before:
                            break
                    if self._closed or expected_gen != self._settle_generation or self.prev_frame is None:
                        return
                    in_flight = self._present_in_flight

                if in_flight > 0:
                    # Probe _usb_lock outside _state_lock (strict _usb_lock -> _state_lock).
                    # If free, in-flight callers are preprocessing and will drain cleanly;
                    # keep _settle_draining_gen set so tight loops cannot starve settle.
                    if self._usb_lock.acquire(blocking=False):
                        self._usb_lock.release()
                        continue
                    need_yield = True
                else:
                    # _present_in_flight == 0 and _settle_draining_gen == expected_gen blocks
                    # new present() calls from racing ahead before _usb_lock is acquired.
                    acquired = self._usb_lock.acquire(blocking=False)
                    need_yield = not acquired

                if need_yield:
                    # _usb_lock is held (e.g., by an external RLock holder). Clear the
                    # draining gate so a reentrant present() call cannot deadlock, and wait
                    # for _usb_lock without re-arming the gate until _usb_lock makes progress.
                    with self._state_lock:
                        if self._settle_draining_gen == expected_gen:
                            self._settle_draining_gen = None
                            self._present_cond.notify_all()
                        yield_exit_seq = self._present_exit_seq

                    while not acquired:
                        if self._usb_lock.acquire(timeout=0.05):
                            aborted = False
                            with self._state_lock:
                                if self._closed or expected_gen != self._settle_generation or self.prev_frame is None:
                                    aborted = True
                                elif self._present_in_flight == 0:
                                    acquired = True
                                else:
                                    self._settle_draining_gen = expected_gen
                            if aborted:
                                self._usb_lock.release()
                                return
                            if not acquired:
                                self._usb_lock.release()
                                break
                        else:
                            with self._state_lock:
                                if self._closed or expected_gen != self._settle_generation or self.prev_frame is None:
                                    return
                                if self._present_in_flight == 0 or self._present_exit_seq != yield_exit_seq:
                                    break

            try:
                with self._state_lock:
                    if self._settle_draining_gen == expected_gen:
                        self._settle_draining_gen = None
                        self._present_cond.notify_all()
                    if self._closed or expected_gen != self._settle_generation or self.prev_frame is None:
                        return
                    self._settle_armed = False
                    frame_copy = self.prev_frame.copy()

                packed = pack_eink_grayscale(frame_copy)
                try:
                    self.ctrl.display_image_bytes(
                        packed,
                        PANEL_WIDTH,
                        PANEL_HEIGHT,
                        waveform=WAVEFORM_GC16,
                    )
                except (UsbError, OSError):
                    return

                with self._state_lock:
                    if not self._closed and expected_gen == self._settle_generation:
                        self.partial_count = 0
                        self.settle_count += 1
                        self._last_present_ts = time.monotonic()
            finally:
                self._usb_lock.release()
        finally:
            with self._state_lock:
                if self._settle_draining_gen == expected_gen:
                    self._settle_draining_gen = None
                    self._present_cond.notify_all()
                if self._settle_timer is cur:
                    self._settle_timer = None
                    self._settle_armed = False
                self._active_settle_threads.discard(cur)

    def close(self) -> None:
        """Mark closed, cancel pending idle settle timers, and join all active settle threads."""
        with self._state_lock:
            self._closed = True
            self._settle_generation += 1
            self._settle_draining_gen = None
            timer = self._settle_timer
            self._settle_timer = None
            self._settle_armed = False
            self._prune_dead_settle_threads_locked()
            threads = list(self._active_settle_threads)
            self._present_cond.notify_all()
        if timer is not None:
            timer.cancel()
        cur = threading.current_thread()
        for t in threads:
            if t is not cur and isinstance(t, threading.Timer):
                t.cancel()
        for t in threads:
            if t is not cur and t.is_alive():
                try:
                    t.join(timeout=5.0)
                except RuntimeError:
                    pass
        with self._state_lock:
            self._prune_dead_settle_threads_locked()

    def reset_framebuffer(self, fill_color: int = 255) -> None:
        """Reset compositor state after a full INIT clear."""
        with self._state_lock:
            self._reset_generation += 1
            self._settle_generation += 1
            self._settle_draining_gen = None
            timer = self._settle_timer
            self._settle_timer = None
            self._settle_armed = False
            self._prune_dead_settle_threads_locked()
            self.prev_frame = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=fill_color)
            self.partial_count = 0
            self._present_cond.notify_all()
        if timer is not None:
            timer.cancel()

    def present(
        self,
        frame: Image.Image,
        policy: str = "fast",
        force_full: bool = False,
        clean_waveform: int = WAVEFORM_GC16,
    ) -> PresentStats:
        """Diff `frame` against `prev_frame` and present according to `policy`.

        Policies:
          - `fast`: strict 1-bit `{0, 255}`, full DU2 on first frame or >65% coverage,
            8px-aligned partial DU2, `clean_waveform` (GC16) on budget; arms idle GC16 settle.
          - `ui`: strict 1-bit `{0, 255}`, `clean_waveform` (GC16) on first frame/budget/>65%,
            8px-aligned partial DU2; arms idle GC16 settle iff DU2 blitted.
          - `partial`: grayscale `"L"` preserved, `clean_waveform` (GC16) on first frame/budget/>65%,
            8px-aligned partial GL16; cancels pending settle, never arms DU2 settle.
          - `full`: grayscale `"L"`, unconditional full `clean_waveform` (GC16), resets burst state.
          - `clean`: grayscale `"L"`, unconditional full `WAVEFORM_INIT`, resets burst state.
        """
        if policy not in SEMANTIC_POLICIES:
            raise ValueError(f"Unknown refresh policy {policy!r}; expected one of {SEMANTIC_POLICIES}")

        effective_policy = "full" if (force_full and policy in ("fast", "ui", "partial")) else policy

        with self._state_lock:
            if self._closed:
                return PresentStats(
                    blits=0,
                    policy=effective_policy,
                    waveform=None,
                    full_refresh=False,
                    settle_armed=False,
                )
            last_ts = self._last_present_ts

        # Enforce tick cap outside all locks so close() and settle cancellation never stall
        if effective_policy in ("fast", "ui", "partial") and last_ts > 0.0:
            wait = self.min_frame_interval - (time.monotonic() - last_ts)
            if wait > 0:
                time.sleep(wait)

        with self._state_lock:
            while not self._closed and self._settle_draining_gen is not None:
                self._present_cond.wait(timeout=0.05)
            if self._closed:
                return PresentStats(
                    blits=0,
                    policy=effective_policy,
                    waveform=None,
                    full_refresh=False,
                    settle_armed=False,
                )
            self._present_in_flight += 1

        try:
            # Prepare frame outside locks
            curr = frame
            if effective_policy in ("fast", "ui"):
                if curr.size != (PANEL_WIDTH, PANEL_HEIGHT):
                    curr = curr.resize((PANEL_WIDTH, PANEL_HEIGHT), Image.Resampling.NEAREST)
                curr = threshold_1bit(curr)
            else:
                if curr.size != (PANEL_WIDTH, PANEL_HEIGHT):
                    curr = curr.resize((PANEL_WIDTH, PANEL_HEIGHT), Image.Resampling.LANCZOS)
                if curr.mode != "L":
                    curr = curr.convert("L")

            with self._usb_lock:
                with self._state_lock:
                    if self._closed:
                        return PresentStats(
                            blits=0,
                            policy=effective_policy,
                            waveform=None,
                            full_refresh=False,
                            settle_armed=False,
                        )
                    prev_frame = self.prev_frame
                    partial_count = self.partial_count
                    reset_gen = self._reset_generation

                if effective_policy in ("full", "clean"):
                    wf = WAVEFORM_INIT if effective_policy == "clean" else clean_waveform
                    self._cancel_settle_timer()
                    packed = pack_eink_grayscale(curr)
                    self.ctrl.display_image_bytes(packed, PANEL_WIDTH, PANEL_HEIGHT, waveform=wf)
                    with self._state_lock:
                        if not self._closed and self._reset_generation == reset_gen:
                            self.prev_frame = curr.copy()
                            self.partial_count = 0
                            self._last_present_ts = time.monotonic()
                    return PresentStats(
                        blits=1,
                        policy=effective_policy,
                        waveform=wf,
                        full_refresh=True,
                        rects=(),
                        dirty_area=PANEL_WIDTH * PANEL_HEIGHT,
                        settle_armed=False,
                    )

                # First frame or anti-ghosting GC16 budget reached
                if prev_frame is None or partial_count >= self.gc16_budget:
                    wf = WAVEFORM_DU2 if (effective_policy == "fast" and prev_frame is None) else clean_waveform
                    self._cancel_settle_timer()
                    packed = pack_eink_grayscale(curr)
                    self.ctrl.display_image_bytes(packed, PANEL_WIDTH, PANEL_HEIGHT, waveform=wf)
                    with self._state_lock:
                        if not self._closed and self._reset_generation == reset_gen:
                            self.prev_frame = curr.copy()
                            self.partial_count = 0
                            self._last_present_ts = time.monotonic()
                            should_arm = wf == WAVEFORM_DU2
                        else:
                            should_arm = False
                    armed = self._arm_settle_timer(expected_reset_gen=reset_gen) if should_arm else False
                    return PresentStats(
                        blits=1,
                        policy=effective_policy,
                        waveform=wf,
                        full_refresh=True,
                        rects=(),
                        dirty_area=PANEL_WIDTH * PANEL_HEIGHT,
                        settle_armed=armed,
                    )

                rects = compute_dirty_rects(prev_frame, curr)
                if not rects:
                    with self._state_lock:
                        armed = self._settle_armed
                    return PresentStats(
                        blits=0,
                        policy=effective_policy,
                        waveform=None,
                        full_refresh=False,
                        rects=(),
                        dirty_area=0,
                        settle_armed=armed,
                    )

                dirty_area = sum(r.area for r in rects)
                if dirty_area > int(PANEL_WIDTH * PANEL_HEIGHT * 0.65):
                    self._cancel_settle_timer()
                    if effective_policy == "fast":
                        wf = WAVEFORM_DU2
                        next_partial_count = partial_count + 1
                    else:
                        wf = clean_waveform
                        next_partial_count = 0
                    packed = pack_eink_grayscale(curr)
                    self.ctrl.display_image_bytes(packed, PANEL_WIDTH, PANEL_HEIGHT, waveform=wf)
                    with self._state_lock:
                        if not self._closed and self._reset_generation == reset_gen:
                            self.prev_frame = curr.copy()
                            self.partial_count = next_partial_count
                            self._last_present_ts = time.monotonic()
                            should_arm = wf == WAVEFORM_DU2
                        else:
                            should_arm = False
                    armed = self._arm_settle_timer(expected_reset_gen=reset_gen) if should_arm else False
                    return PresentStats(
                        blits=1,
                        policy=effective_policy,
                        waveform=wf,
                        full_refresh=True,
                        rects=(),
                        dirty_area=dirty_area,
                        settle_armed=armed,
                    )

                # Aligned partial blits
                self._cancel_settle_timer()
                wf = WAVEFORM_GL16 if effective_policy == "partial" else WAVEFORM_DU2
                for r in rects:
                    patch = pack_eink_region(curr, r.x, r.y, r.w, r.h)
                    self.ctrl.display_region(r.x, r.y, r.w, r.h, patch, waveform=wf)

                with self._state_lock:
                    if not self._closed and self._reset_generation == reset_gen:
                        self.prev_frame = curr.copy()
                        self.partial_count = partial_count + 1
                        self._last_present_ts = time.monotonic()
                        should_arm = wf == WAVEFORM_DU2
                    else:
                        should_arm = False

                armed = self._arm_settle_timer(expected_reset_gen=reset_gen) if should_arm else False
                return PresentStats(
                    blits=len(rects),
                    policy=effective_policy,
                    waveform=wf,
                    full_refresh=False,
                    rects=tuple(rects),
                    dirty_area=dirty_area,
                    settle_armed=armed,
                )
        finally:
            with self._state_lock:
                self._present_in_flight -= 1
                self._present_exit_seq += 1
                if self._present_in_flight == 0:
                    self._present_cond.notify_all()
