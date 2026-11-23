"""Persistent settings store for Yoga Book C930 E-Ink & Keyboard Control."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass

from .image_render import normalize_dither_mode
from .keyboard_skin import normalize_keyboard_style
from .protocol import (
    WAVEFORM_CURRENT,
    WAVEFORM_DU2,
    WAVEFORM_GC16,
    WAVEFORM_GL16,
    WAVEFORM_INIT,
)
from .tone_lut import clamp_contrast, clamp_lightness

CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".config", "yogabook-eink")
CONFIG_PATH = os.path.join(CONFIG_DIR, "settings.json")

VALID_SOUND_PROFILES: tuple[str, ...] = ("soft", "crisp", "typewriter")
VALID_WAVEFORMS: frozenset[int] = frozenset(
    {WAVEFORM_INIT, WAVEFORM_DU2, WAVEFORM_GC16, WAVEFORM_GL16, WAVEFORM_CURRENT}
)
IDLE_SETTLE_MIN_SEC = 0.05
IDLE_SETTLE_MAX_SEC = 60.0


def _coerce_bool(value: object, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _coerce_int(value: object, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _coerce_idle_settle_sec(value: object, default: float = 2.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    try:
        val = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(val):
        return default
    if val < IDLE_SETTLE_MIN_SEC:
        return IDLE_SETTLE_MIN_SEC
    if val > IDLE_SETTLE_MAX_SEC:
        return IDLE_SETTLE_MAX_SEC
    return val


@dataclass
class EinkSettings:
    key_down_sound: bool = False
    key_up_sound: bool = False
    sound_profile: str = "soft"  # soft | crisp | typewriter
    dark_theme: bool = False     # Light (white bg) vs Dark (black bg) E-Ink Settings UI
    waveform: int = WAVEFORM_GC16
    rotate_180: bool = False
    swap_axes: bool = False
    flip_x: bool = False
    flip_y: bool = False
    lightness: int = 0
    contrast: int = 0
    dither_mode: str = "Grayscale 16-level"
    idle_settle_enabled: bool = True
    idle_settle_sec: float = 2.0
    keyboard_style: str = "classic_white"
    applied_keyboard_style: str = "classic_white"

    def __post_init__(self) -> None:
        self.normalize()

    def normalize(self) -> "EinkSettings":
        """Validate, coerce, and clamp all fields in-place."""
        self.key_down_sound = _coerce_bool(self.key_down_sound, False)
        self.key_up_sound = _coerce_bool(self.key_up_sound, False)
        self.dark_theme = _coerce_bool(self.dark_theme, False)
        self.rotate_180 = _coerce_bool(self.rotate_180, False)
        self.swap_axes = _coerce_bool(self.swap_axes, False)
        self.flip_x = _coerce_bool(self.flip_x, False)
        self.flip_y = _coerce_bool(self.flip_y, False)
        self.idle_settle_enabled = _coerce_bool(self.idle_settle_enabled, True)

        self.sound_profile = (
            self.sound_profile
            if isinstance(self.sound_profile, str) and self.sound_profile in VALID_SOUND_PROFILES
            else "soft"
        )

        wf = _coerce_int(self.waveform, WAVEFORM_GC16)
        self.waveform = wf if wf in VALID_WAVEFORMS else WAVEFORM_GC16

        self.lightness = clamp_lightness(_coerce_int(self.lightness, 0))
        self.contrast = clamp_contrast(_coerce_int(self.contrast, 0))

        try:
            self.dither_mode = normalize_dither_mode(self.dither_mode)
        except (ValueError, TypeError):
            self.dither_mode = "Grayscale 16-level"

        self.idle_settle_sec = _coerce_idle_settle_sec(self.idle_settle_sec, 2.0)
        self.keyboard_style = normalize_keyboard_style(self.keyboard_style)
        self.applied_keyboard_style = normalize_keyboard_style(self.applied_keyboard_style)
        return self

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "EinkSettings":
        target = os.fspath(path) if path is not None else CONFIG_PATH
        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return cls()

        if not isinstance(data, dict):
            return cls()

        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def save(self, path: str | os.PathLike[str] | None = None) -> None:
        self.normalize()
        target = os.path.abspath(os.fspath(path) if path is not None else CONFIG_PATH)
        parent = os.path.dirname(target)
        os.makedirs(parent, exist_ok=True)
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=parent,
                prefix=".settings.",
                suffix=".tmp",
                delete=False,
            ) as f:
                tmp_path = f.name
                json.dump(asdict(self), f, indent=2)
                f.flush()
            os.replace(tmp_path, target)
            tmp_path = None
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
