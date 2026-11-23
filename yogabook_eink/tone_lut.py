"""Glider 256-byte Parabolic Tone LUT for Yoga Book C930 E-Ink rendering.

Exact IEEE 754 float32 port of ``references/Glider/fw/User/tone_lut.c``.
"""

from __future__ import annotations

import functools
import struct

from PIL import Image

TONE_LUT_SIZE = 256
LIGHTNESS_MIN = -3
LIGHTNESS_MAX = 3
CONTRAST_MIN = -1
CONTRAST_MAX = 6

LIGHTNESS_LUT: tuple[float, ...] = (0.0, 0.3, 0.6, 1.0, 1.3, 1.6, 2.0)
CONTRAST_LUT: tuple[float, ...] = (0.8, 1.0, 1.35, 1.8, 2.7, 5.0, 12.0, 255.0)

_F32_STRUCT = struct.Struct("f")


def _f32(value: float) -> float:
    """Round a Python float to single-precision IEEE 754 float32."""
    return _F32_STRUCT.unpack(_F32_STRUCT.pack(value))[0]


def clamp_lightness(lightness: int) -> int:
    """Clamp lightness parameter to [-3, 3]."""
    if isinstance(lightness, bool) or not isinstance(lightness, int):
        raise TypeError(f"lightness must be an int, got {type(lightness).__name__}")
    if lightness < LIGHTNESS_MIN:
        return LIGHTNESS_MIN
    if lightness > LIGHTNESS_MAX:
        return LIGHTNESS_MAX
    return lightness


def clamp_contrast(contrast: int) -> int:
    """Clamp contrast parameter to [-1, 6]."""
    if isinstance(contrast, bool) or not isinstance(contrast, int):
        raise TypeError(f"contrast must be an int, got {type(contrast).__name__}")
    if contrast < CONTRAST_MIN:
        return CONTRAST_MIN
    if contrast > CONTRAST_MAX:
        return CONTRAST_MAX
    return contrast


@functools.lru_cache(maxsize=64)
def _build_tone_lut_clamped(lightness: int, contrast: int) -> bytes:
    f32_0 = _f32(0.0)
    f32_0_5 = _f32(0.5)
    f32_1 = _f32(1.0)
    f32_255 = _f32(255.0)

    lightness_factor = _f32(LIGHTNESS_LUT[lightness - LIGHTNESS_MIN])
    contrast_factor = _f32(CONTRAST_LUT[contrast - CONTRAST_MIN])
    bias = _f32(lightness_factor - f32_1)

    lut = bytearray(TONE_LUT_SIZE)
    for i in range(TONE_LUT_SIZE):
        norm = _f32(_f32(i) / f32_255)
        c_raw = _f32(f32_0_5 + _f32(_f32(norm - f32_0_5) * contrast_factor))
        if c_raw < f32_0:
            contrasted = f32_0
        elif c_raw > f32_1:
            contrasted = f32_1
        else:
            contrasted = c_raw

        c_sq = _f32(contrasted * contrasted)
        shaped = _f32(contrasted + _f32(bias * _f32(contrasted - c_sq)))
        if shaped < f32_0:
            clamped = f32_0
        elif shaped > f32_1:
            clamped = f32_1
        else:
            clamped = shaped

        lut[i] = int(_f32(_f32(clamped * f32_255) + f32_0_5))
    return bytes(lut)


def build_tone_lut(lightness: int = 0, contrast: int = 0) -> bytes:
    """Build a 256-byte parabolic tone LUT matching Glider's ``tone_lut.c``."""
    return _build_tone_lut_clamped(clamp_lightness(lightness), clamp_contrast(contrast))


def apply_tone_lut(
    img: Image.Image,
    lightness: int = 0,
    contrast: int = 0,
) -> Image.Image:
    """Apply the Glider 256-byte tone LUT to an image, returning a new 'L' Image."""
    l_clamped = clamp_lightness(lightness)
    c_clamped = clamp_contrast(contrast)
    gray = img if img.mode == "L" else img.convert("L")
    if l_clamped == 0 and c_clamped == 0:
        return gray.copy() if img.mode == "L" else gray
    lut = build_tone_lut(l_clamped, c_clamped)
    return gray.point(lut)
