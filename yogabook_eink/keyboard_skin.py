"""Lenovo Yoga Book C930 Native Hardware Keyboard & Touchpad Style Manager.

Manages the 4 official Lenovo E-Ink KeyboardBlob (.bin.cap) styles from EinkIteAPI.h:
  - classic_black (KEYBOARD_STYLE_CLASSIC_BLACK = 1 -> 1033_CLS_V_3D.bin.cap)
  - classic_white (KEYBOARD_STYLE_CLASSIC_WHITE = 2 -> 1033_CLS_C_IN.bin.cap)
  - modern_black  (KEYBOARD_STYLE_MODERN_BLACK  = 3 -> 1033_MDRN_V_3D.bin.cap)
  - modern_white  (KEYBOARD_STYLE_MODERN_WHITE  = 4 -> 1033_MDRN_C_IN.bin.cap)

Applies the selected `.bin.cap` directly to the ITE8951 T-CON hardware via
`Ite8951Controller.set_keyboard_style_blob()` (official `TconSetKeyBoardStyle` protocol:
OP_WRITE_MEM 0x82 + OP_UPDATE 0xAA `kb_blob_cmd_valid` / `kb_blob_cmd_get_status`),
so the panel runs in genuine Scenario 1 (native hardware keyboard + touchpad with
native key highlights and zero glitches).
"""

from __future__ import annotations

import os
from typing import Optional

KEYBOARD_STYLES: tuple[str, ...] = (
    "classic_black",
    "classic_white",
    "modern_black",
    "modern_white",
)

KEYBOARD_STYLE_LABELS: dict[str, str] = {
    "classic_black": "Classic Dark",
    "classic_white": "Classic Light",
    "modern_black": "Modern Dark",
    "modern_white": "Modern Light",
}

KEYBOARD_STYLE_BLOBS: dict[str, str] = {
    "classic_black": "1033_CLS_V_3D.bin.cap",
    "classic_white": "1033_CLS_C_IN.bin.cap",
    "modern_black": "1033_MDRN_V_3D.bin.cap",
    "modern_white": "1033_MDRN_C_IN.bin.cap",
}

BLOBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blobs")


def normalize_keyboard_style(style: object) -> str:
    """Coerce a style identifier to one of the 4 official Lenovo keyboard styles."""
    if isinstance(style, str):
        s = style.strip().lower()
        if s in KEYBOARD_STYLES:
            return s
    return "classic_white"


def get_keyboard_blob_path(style: str, lang_id: int = 1033) -> Optional[str]:
    """Return the absolute path to the official Lenovo `.bin.cap` blob for `style`."""
    st = normalize_keyboard_style(style)
    suffix = KEYBOARD_STYLE_BLOBS[st].split("_", 1)[1]
    candidate = os.path.join(BLOBS_DIR, f"{lang_id}_{suffix}")
    if os.path.isfile(candidate):
        return candidate
    fallback = os.path.join(BLOBS_DIR, KEYBOARD_STYLE_BLOBS[st])
    if os.path.isfile(fallback):
        return fallback
    return None


def load_keyboard_blob(style: str, lang_id: int = 1033) -> Optional[bytes]:
    """Read the official Lenovo `.bin.cap` binary payload for `style`."""
    path = get_keyboard_blob_path(style, lang_id=lang_id)
    if path is None:
        return None
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None
