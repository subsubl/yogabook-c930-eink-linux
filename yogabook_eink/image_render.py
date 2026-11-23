"""Image conversion, dithering, and template generation for the 1920x1080 ITE8951 E-Ink panel."""

from __future__ import annotations

import functools
from typing import Literal

from PIL import Image, ImageChops, ImageDraw, ImageOps

from .protocol import PANEL_HEIGHT, PANEL_WIDTH
from .tone_lut import apply_tone_lut

_EINK_LUT = bytes((i & 0xF0) | 0x0F for i in range(256))
_GRAY16_LUT = bytes(((i + 8) // 17) * 17 for i in range(256))
_GRAY2_LUT = bytes(255 if i >= 128 else 0 for i in range(256))
_ZERO_TO_255_LUT = bytes([255] + [0] * 255)
_FLOOR17_LUT = bytes(min(15, i // 17) * 17 for i in range(256))

DITHER_MODES: tuple[str, ...] = (
    "Grayscale 16-level",
    "4x4 Bayer",
    "8x8 Bayer",
    "Floyd-Steinberg",
    "Atkinson",
)

DITHER_MODE_ALIASES: dict[str, str] = {
    "grayscale 16-level": "Grayscale 16-level",
    "grayscale-16": "Grayscale 16-level",
    "16-level": "Grayscale 16-level",
    "16level": "Grayscale 16-level",
    "grayscale": "Grayscale 16-level",
    "gray16": "Grayscale 16-level",
    "4x4 bayer": "4x4 Bayer",
    "bayer-4x4": "4x4 Bayer",
    "4x4-bayer": "4x4 Bayer",
    "4x4": "4x4 Bayer",
    "8x8 bayer": "8x8 Bayer",
    "bayer-8x8": "8x8 Bayer",
    "8x8-bayer": "8x8 Bayer",
    "8x8": "8x8 Bayer",
    "floyd-steinberg": "Floyd-Steinberg",
    "floyd steinberg": "Floyd-Steinberg",
    "floydsteinberg": "Floyd-Steinberg",
    "fs": "Floyd-Steinberg",
    "atkinson": "Atkinson",
}


def normalize_dither_mode(mode: str) -> str:
    """Normalize a canonical dither mode name or CLI alias; raise ValueError if invalid."""
    if not isinstance(mode, str):
        raise ValueError(f"Unsupported dither mode: {mode!r}")
    key = " ".join(mode.strip().lower().replace("_", "-").split())
    canonical = DITHER_MODE_ALIASES.get(key)
    if canonical is None:
        canonical = DITHER_MODE_ALIASES.get(key.replace(" ", "-"))
    if canonical is None:
        raise ValueError(
            f"Unknown dither mode {mode!r}; expected one of: {', '.join(DITHER_MODES)}"
        )
    return canonical


@functools.lru_cache(maxsize=4)
def _generate_bayer_index(size: int) -> tuple[tuple[int, ...], ...]:
    """Generate an epdoptimize-compatible recursive Bayer rank matrix of size 2^k."""
    if size == 2:
        return ((0, 2), (3, 1))
    half = size // 2
    sub = _generate_bayer_index(half)
    rows: list[tuple[int, ...]] = []
    for y in range(size):
        sy = y % half
        q_y = 0 if y < half else 1
        row = []
        for x in range(size):
            sx = x % half
            q_x = 0 if x < half else 1
            if q_y == 0 and q_x == 0:
                offset = 0
            elif q_y == 0 and q_x == 1:
                offset = 2
            elif q_y == 1 and q_x == 0:
                offset = 3
            else:
                offset = 1
            row.append(4 * sub[sy][sx] + offset)
        rows.append(tuple(row))
    return tuple(rows)


@functools.lru_cache(maxsize=16)
def _tile_bayer_matrix(size: int, levels: int, w: int, h: int) -> Image.Image:
    ranks = _generate_bayer_index(size)
    n = size * size
    if levels == 2:
        mat = [[((2 * r + 1) * 128) // n for r in row] for row in ranks]
    elif levels == 16:
        mat = [[17 - ((2 * r + 1) * 17 + 2 * n - 1) // (2 * n) for r in row] for row in ranks]
    else:
        raise ValueError(f"Unsupported cached Bayer levels: {levels}")
    block = b"".join((bytes(row) * ((w + size - 1) // size))[:w] for row in mat)
    tiled = (block * ((h + size - 1) // size))[: w * h]
    return Image.frombytes("L", (w, h), tiled)


@functools.lru_cache(maxsize=1)
def _get_palette_16_img() -> Image.Image:
    pal_img = Image.new("P", (1, 1))
    palette: list[int] = []
    for k in range(16):
        v = k * 17
        palette.extend((v, v, v))
    palette.extend([0] * (768 - len(palette)))
    pal_img.putpalette(palette)
    return pal_img


def _ordered_bayer_general(gray: Image.Image, size: int, levels: int) -> Image.Image:
    w, h = gray.size
    ranks = _generate_bayer_index(size)
    n = size * size
    palette = [round(k * 255 / (levels - 1)) for k in range(levels)]
    step = 255.0 / (levels - 1)
    deltas = [[((r + 0.5) / n - 0.5) * step for r in row] for row in ranks]
    src = gray.tobytes()
    out = bytearray(w * h)
    idx = 0
    for y in range(h):
        drow = deltas[y % size]
        for x in range(w):
            v = src[idx] + drow[x % size]
            out[idx] = min(palette, key=lambda p: abs(p - v))
            idx += 1
    return Image.frombytes("L", (w, h), bytes(out))


def _diffuse_floyd_steinberg(
    gray: Image.Image,
    levels: int,
    serpentine: bool,
) -> Image.Image:
    w, h = gray.size
    src = gray.tobytes()
    out = bytearray(w * h)
    pad_len = w + 3
    row0 = [0.0] * pad_len
    row1 = [0.0] * pad_len
    palette = None if levels in (2, 16) else [round(k * 255 / (levels - 1)) for k in range(levels)]

    w7 = 7.0 / 16.0
    w3 = 3.0 / 16.0
    w5 = 5.0 / 16.0
    w1 = 1.0 / 16.0

    for y in range(h):
        row_off = y * w
        reverse = serpentine and (y & 1) == 1
        if not reverse:
            for x in range(w):
                old = src[row_off + x] + row0[x]
                if levels == 2:
                    new = 255 if old > 127.5 else 0
                elif levels == 16:
                    if old <= 0.0:
                        new = 0
                    elif old >= 255.0:
                        new = 255
                    else:
                        q = int(old // 17)
                        if old - q * 17 > 8.5:
                            q += 1
                        new = q * 17
                else:
                    new = min(palette, key=lambda p: abs(p - old))
                out[row_off + x] = new
                res = old - new
                abs_res = -res if res < 0.0 else res
                if abs_res > 255.0:
                    res *= 255.0 / abs_res
                row0[x + 1] += res * w7
                row1[x - 1] += res * w3
                row1[x] += res * w5
                row1[x + 1] += res * w1
        else:
            for x in range(w - 1, -1, -1):
                old = src[row_off + x] + row0[x]
                if levels == 2:
                    new = 255 if old > 127.5 else 0
                elif levels == 16:
                    if old <= 0.0:
                        new = 0
                    elif old >= 255.0:
                        new = 255
                    else:
                        q = int(old // 17)
                        if old - q * 17 > 8.5:
                            q += 1
                        new = q * 17
                else:
                    new = min(palette, key=lambda p: abs(p - old))
                out[row_off + x] = new
                res = old - new
                abs_res = -res if res < 0.0 else res
                if abs_res > 255.0:
                    res *= 255.0 / abs_res
                row0[x - 1] += res * w7
                row1[x + 1] += res * w3
                row1[x] += res * w5
                row1[x - 1] += res * w1

        row0 = row1
        row1 = [0.0] * pad_len

    return Image.frombytes("L", (w, h), bytes(out))


def _diffuse_atkinson(
    gray: Image.Image,
    levels: int,
    serpentine: bool,
) -> Image.Image:
    w, h = gray.size
    src = gray.tobytes()
    out = bytearray(w * h)
    pad_len = w + 3
    row0 = [0.0] * pad_len
    row1 = [0.0] * pad_len
    row2 = [0.0] * pad_len
    palette = None if levels in (2, 16) else [round(k * 255 / (levels - 1)) for k in range(levels)]

    for y in range(h):
        row_off = y * w
        reverse = serpentine and (y & 1) == 1
        if not reverse:
            for x in range(w):
                old = src[row_off + x] + row0[x]
                if levels == 2:
                    new = 255 if old > 127.5 else 0
                elif levels == 16:
                    if old <= 0.0:
                        new = 0
                    elif old >= 255.0:
                        new = 255
                    else:
                        q = int(old // 17)
                        if old - q * 17 > 8.5:
                            q += 1
                        new = q * 17
                else:
                    new = min(palette, key=lambda p: abs(p - old))
                out[row_off + x] = new
                res = old - new
                abs_res = -res if res < 0.0 else res
                if abs_res > 255.0:
                    res *= 255.0 / abs_res
                err = res * 0.125
                row0[x + 1] += err
                row0[x + 2] += err
                row1[x - 1] += err
                row1[x] += err
                row1[x + 1] += err
                row2[x] += err
        else:
            for x in range(w - 1, -1, -1):
                old = src[row_off + x] + row0[x]
                if levels == 2:
                    new = 255 if old > 127.5 else 0
                elif levels == 16:
                    if old <= 0.0:
                        new = 0
                    elif old >= 255.0:
                        new = 255
                    else:
                        q = int(old // 17)
                        if old - q * 17 > 8.5:
                            q += 1
                        new = q * 17
                else:
                    new = min(palette, key=lambda p: abs(p - old))
                out[row_off + x] = new
                res = old - new
                abs_res = -res if res < 0.0 else res
                if abs_res > 255.0:
                    res *= 255.0 / abs_res
                err = res * 0.125
                row0[x - 1] += err
                row0[x - 2] += err
                row1[x + 1] += err
                row1[x] += err
                row1[x - 1] += err
                row2[x] += err

        row0 = row1
        row1 = row2
        row2 = [0.0] * pad_len

    return Image.frombytes("L", (w, h), bytes(out))


def dither_image(
    img: Image.Image,
    mode: str = "Grayscale 16-level",
    levels: int | None = None,
    serpentine: bool = False,
) -> Image.Image:
    """Dither an image to 'L' mode using one of the 5 epdoptimize-style modes."""
    canonical = normalize_dither_mode(mode)
    if levels is None:
        levels = 16 if canonical == "Grayscale 16-level" else 2
    elif isinstance(levels, bool) or not isinstance(levels, int) or levels < 2 or levels > 256:
        raise ValueError(f"levels must be an integer in [2, 256], got {levels!r}")

    gray = img.convert("L") if img.mode != "L" else img.copy()
    w, h = gray.size
    if w == 0 or h == 0:
        return gray

    if canonical == "Grayscale 16-level":
        if levels == 16:
            return gray.point(_GRAY16_LUT)
        if levels == 2:
            return gray.point(_GRAY2_LUT)
        palette = [round(k * 255 / (levels - 1)) for k in range(levels)]
        lut = bytes(min(palette, key=lambda p: abs(p - i)) for i in range(256))
        return gray.point(lut)

    if canonical in ("4x4 Bayer", "8x8 Bayer"):
        size = 4 if canonical == "4x4 Bayer" else 8
        if levels == 2:
            thresh_img = _tile_bayer_matrix(size, 2, w, h)
            return ImageChops.subtract(thresh_img, gray).point(_ZERO_TO_255_LUT)
        if levels == 16:
            offset_img = _tile_bayer_matrix(size, 16, w, h)
            return ImageChops.add(gray, offset_img).point(_FLOOR17_LUT)
        return _ordered_bayer_general(gray, size, levels)

    if canonical == "Floyd-Steinberg":
        if not serpentine and levels == 2:
            return gray.convert("1", dither=Image.Dither.FLOYDSTEINBERG).convert("L")
        if not serpentine and levels == 16:
            return (
                gray.convert("RGB")
                .quantize(palette=_get_palette_16_img(), dither=Image.Dither.FLOYDSTEINBERG)
                .convert("L")
            )
        return _diffuse_floyd_steinberg(gray, levels, serpentine)

    # canonical == "Atkinson"
    return _diffuse_atkinson(gray, levels, serpentine)


def pack_eink_grayscale(img: Image.Image) -> bytes:
    """Pack a 1920x1080 'L' mode PIL image into ITE8951 format: (gray & 0xF0) | 0x0F."""
    if img.size != (PANEL_WIDTH, PANEL_HEIGHT):
        img = img.resize((PANEL_WIDTH, PANEL_HEIGHT), Image.Resampling.LANCZOS)
    if img.mode != "L":
        img = img.convert("L")
    return img.tobytes().translate(_EINK_LUT)


def pack_eink_region(img: Image.Image, x: int, y: int, w: int, h: int) -> bytes:
    """Crop (x, y, w, h) from a full 'L' image and pack into ITE8951 format."""
    if img.mode != "L":
        img = img.convert("L")
    crop = img.crop((x, y, x + w, y + h))
    return crop.tobytes().translate(_EINK_LUT)


def prepare_image_file(
    path: str,
    fit_mode: Literal["contain", "cover"] = "contain",
    invert: bool = False,
    rotate_180: bool = False,
    dither_1bit: bool = False,
    dither_mode: str | None = None,
    lightness: int = 0,
    contrast: int = 0,
) -> tuple[Image.Image, bytes]:
    """Load an image, fit to 1920x1080 grayscale, tone-adjust, dither, and return (preview_pil, packed_bytes)."""
    with Image.open(path) as raw:
        img = ImageOps.exif_transpose(raw)
        if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
            rgba = img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, rgba).convert("L")
        else:
            img = img.convert("L")

    if img.width <= 0 or img.height <= 0:
        img = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
    elif fit_mode == "cover":
        img = ImageOps.fit(img, (PANEL_WIDTH, PANEL_HEIGHT), method=Image.Resampling.LANCZOS)
    else:
        im_ratio = img.width / img.height
        dest_ratio = PANEL_WIDTH / PANEL_HEIGHT
        if im_ratio == dest_ratio:
            img = img.resize((PANEL_WIDTH, PANEL_HEIGHT), Image.Resampling.LANCZOS)
        elif im_ratio > dest_ratio:
            new_h = max(1, round(img.height / img.width * PANEL_WIDTH))
            if new_h == PANEL_HEIGHT:
                img = img.resize((PANEL_WIDTH, PANEL_HEIGHT), Image.Resampling.LANCZOS)
            else:
                resized = img.resize((PANEL_WIDTH, new_h), Image.Resampling.LANCZOS)
                canvas = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
                y_off = round((PANEL_HEIGHT - new_h) * 0.5)
                canvas.paste(resized, (0, y_off))
                img = canvas
        else:
            new_w = max(1, round(img.width / img.height * PANEL_HEIGHT))
            if new_w == PANEL_WIDTH:
                img = img.resize((PANEL_WIDTH, PANEL_HEIGHT), Image.Resampling.LANCZOS)
            else:
                resized = img.resize((new_w, PANEL_HEIGHT), Image.Resampling.LANCZOS)
                canvas = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=255)
                x_off = round((PANEL_WIDTH - new_w) * 0.5)
                canvas.paste(resized, (x_off, 0))
                img = canvas

    if invert:
        img = ImageOps.invert(img)

    img = apply_tone_lut(img, lightness=lightness, contrast=contrast)

    resolved_mode = (
        normalize_dither_mode(dither_mode)
        if dither_mode is not None
        else ("Floyd-Steinberg" if dither_1bit else "Grayscale 16-level")
    )
    img = dither_image(img, mode=resolved_mode)

    if rotate_180:
        img = img.rotate(180)

    return img, pack_eink_grayscale(img)


def generate_template(
    kind: Literal["white", "black", "ruled", "dotgrid", "banner"],
    banner_text: str = "Yoga Book C930 — Pop!_OS E-Ink",
    invert: bool = False,
    rotate_180: bool = False,
) -> tuple[Image.Image, bytes]:
    """Generate built-in E-Ink templates (ruled paper, dot grid, blank, or banner)."""
    bg = 0 if kind == "black" else 255
    fg = 255 if kind == "black" else 40
    img = Image.new("L", (PANEL_WIDTH, PANEL_HEIGHT), color=bg)
    draw = ImageDraw.Draw(img)

    if kind == "ruled":
        # Subtle ruled notebook lines + bottom-center Exit to Keyboard button
        for y in range(100, PANEL_HEIGHT - 140, 56):
            draw.line([(80, y), (PANEL_WIDTH - 80, y)], fill=160, width=2)
        draw.line([(160, 60), (160, PANEL_HEIGHT - 140)], fill=120, width=2)
        draw.fontmode = "1"
        draw.rounded_rectangle([(660, 960), (1260, 1052)], radius=18, fill=0)
        draw.text((820, 994), "← EXIT TO KEYBOARD", fill=255)
    elif kind == "dotgrid":
        # 40px dot grid ideal for stylus sketching / notes + bottom-center Exit button
        for y in range(60, PANEL_HEIGHT - 140, 40):
            for x in range(60, PANEL_WIDTH - 40, 40):
                draw.ellipse([(x - 2, y - 2), (x + 2, y + 2)], fill=130)
        draw.fontmode = "1"
        draw.rounded_rectangle([(660, 960), (1260, 1052)], radius=18, fill=0)
        draw.text((820, 994), "← EXIT TO KEYBOARD", fill=255)
    elif kind == "banner":
        draw.rectangle([(48, 48), (PANEL_WIDTH - 48, PANEL_HEIGHT - 48)], outline=fg, width=6)
        draw.rectangle([(64, 64), (PANEL_WIDTH - 64, PANEL_HEIGHT - 64)], outline=140, width=2)
        # Render centered text using Pillow's default scalable font if available
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)
        except Exception:
            font = None
        bbox = draw.textbbox((0, 0), banner_text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(
            ((PANEL_WIDTH - tw) // 2, (PANEL_HEIGHT - th) // 2),
            banner_text,
            fill=fg,
            font=font,
        )

    if invert:
        img = ImageOps.invert(img)
    if rotate_180:
        img = img.rotate(180)

    return img, pack_eink_grayscale(img)
