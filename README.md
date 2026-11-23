# Lenovo Yoga Book C930 — Linux E-Ink & Keyboard Control Suite (`yogabook-eink-gui`)

Complete userspace Linux E-Ink display, on-screen dashboard daemon (`EinkSvr` replacement), GTK3 desktop control app, and CLI for the **Lenovo Yoga Book C930** (`YB-J912`, USB `048d:8951` ITE8951 1920×1080 Mobius E-Ink panel) — **no custom kernel module compilation required**.

Tested on **Pop!_OS 24.04 LTS (COSMIC / Wayland, Linux 7.2+)** and compatible with Ubuntu, Debian, Fedora, and Arch Linux.

---

## ✨ Key Features

### 1. Native On-Screen E-Ink Dashboard Daemon (`yogabook-eink-daemon.service`)
Replaces Windows `EinkSvr.exe` with a zero-lag background daemon listening exclusively to the vendor E-Ink touch interface (`input1`, HID Report `0x90`). Tapping the **top-right Settings icon** (`x >= 1700, y <= 140`) on the E-Ink keyboard opens a strict 1-bit monochrome (`{0, 255}`) on-screen dashboard directly on the E-Ink panel:

- **Interactive Live Top Bar:**
  - **Left:** Real-time system telemetry (`CPU % • RAM used/total GB • DISK %`).
  - **Right (`[HH:MM • Day DD Mon]`):** Tap to open the **Monthly Calendar** view (`◀ PREV`, `TODAY`, `NEXT ▶` navigation with today highlighted).
  - **Right (`[BAT %]`):** Tap to open **Detailed Battery Telemetry** (charge progress bar, charging/discharging status, battery voltage `V`, power `W` & current `A`, estimated time to full / remaining, energy `Wh`, and battery health `%`).
- **2×2 Categorized Dashboard:**
  - **GAMES → `PLAY DOOM`:** Playable 3D E-Ink raycaster (`EinkDoomGame`) rendered directly on the E-Ink panel with Glider parabolic tone mapping, 4×4 Bayer dithering, and bottom on-screen touch controls (`◀`, `▲`, `▼`, `▶`, `FIRE!`, `EXIT`).
  - **APPS → `MATRIX RAIN (DARK / LIGHT)`:** Authentic edge-to-edge Matrix Digital Rain screensaver rendered with half-width Japanese Katakana, mirrored Katakana, and ASCII glyphs in a 4-tier 1-bit sprite atlas (touch anywhere on the panel to exit back to the keyboard).
  - **TOOLS:**
    - **`RULED NOTES`:** 1920×1080 lined notebook paper for Wacom stylus note-taking.
    - **`SKETCH GRID`:** 1920×1080 40px dot-grid pad for sketching and diagrams.
    - **`TOUCHPAD (MT)`:** Arms the Windows-validated full-panel multitouch / pen-mouse latch (`Scenario 3`, `display_cfg |= 0x00080000`).
    - **`CLEAN SCREEN`:** Full `INIT` anti-ghosting flash clear.
  - **SETTINGS:**
    - **Native Hardware Keyboard & Touchpad Styles (All 4 Original Lenovo Modes):**
      - **`Classic Dark`** (`classic_black` → `1033_CLS_V_3D.bin.cap`)
      - **`Classic Light`** (`classic_white` → `1033_CLS_C_IN.bin.cap`)
      - **`Modern Dark`** (`modern_black` → `1033_MDRN_V_3D.bin.cap`)
      - **`Modern Light`** (`modern_white` → `1033_MDRN_C_IN.bin.cap`)
      Flashes the official Lenovo `KeyboardBlob` (`.bin.cap`) directly to the ITE8951 T-CON hardware via `OP_WRITE_MEM` (`0x82`) + `OP_UPDATE` (`0xAA`) so the panel runs in genuine hardware **`Scenario 1`** with native capacitive typing, classic/modern touchpad, and hardware key-press highlights.
    - **Keyboard Key-Down & Key-Up Click Sounds:** Low-latency evdev sound feedback (`Soft Tactile`, `Crisp Click`, `Typewriter`).
    - **E-Ink UI Theme:** Light (paper white) or Dark (obsidian black) on-screen UI palette.
    - **E-Ink Refresh Quality:** Cycle `DU2 Fast`, `GL16 Smooth`, and `GC16 Sharp`.
    - **Glider Parabolic Tone LUT:** Adjust **Tone Lightness** (`-3 .. +3`) and **Tone Contrast** (`-1 .. +6`) for images and 3D DOOM.
    - **Idle Anti-Ghosting (`GC16` Settle):** Automatically performs a crisp `GC16` settle refresh after fast `DU2` interactions.

### 2. E-Ink Rendering & Compositor Pipeline
- **Glider 256-Byte Parabolic Tone LUT (`yogabook_eink/tone_lut.py`):** Pre-dither shadow/highlight and S-curve contrast shaping.
- **5-Mode Dithering Engine (`yogabook_eink/image_render.py`):**
  - `Grayscale 16-level` (4-bit native E-Ink packing `(val & 0xF0) | 0x0F`)
  - `4x4 Bayer` (ordered 1-bit matrix)
  - `8x8 Bayer` (64-threshold ordered 1-bit matrix)
  - `Floyd-Steinberg` (error-diffusion 1-bit)
  - `Atkinson` (high-contrast 6-neighbor `1/8` error-diffusion 1-bit)
- **KOReader 8-Pixel Aligned Partial Compositor (`yogabook_eink/compositor.py`):** Dirty-region bounding, 8px coordinate alignment, semantic refresh policies (`fast`, `ui`, `partial`, `full`, `clean`), flash budget promotion, and automatic idle `GC16` settle timer.
- **Crash-Safe Display Ownership Arbitrator (`yogabook_eink/ownership.py`):** Unix domain socket lease server (`/run/user/$UID/yogabook-eink.sock`) using `SCM_CREDENTIALS` so full-screen E-Ink apps (`matrix`, `doom`, custom scripts) pause the background daemon while active and automatically restore the hardware keyboard if an app exits or crashes.

### 3. Hardware-Safe ITE8951 USB BOT Driver (`yogabook_eink/protocol.py` & `usb_lib.py`)
- Pure-Python `ctypes` wrapper around `libusb-1.0` talking to `048d:8951` Interface 0 (`EP_OUT = 0x02`, `EP_IN = 0x81`).
- 31-byte `USBC` command block + payload + mandatory 13-byte `USBS` status drain.
- Pre- and post-blit `0xB1` (`GET_DPY_STATUS`) `engine_busy` polling (`wait_display_ready()`) to prevent T-CON pipeline collisions.
- Direct `0x94` (`DPY_AREA`) waveform passing (`0=INIT, 1=DU2, 2=GC16, 3=GL16, 0xFF=CURRENT`) with exact `IMAGE_BUF_BASE` (`0x00382f30`) row-aligned region writes.
- Reliable repeated entry/exit between `Scenario 0` (Owner-Draw apps) and `Scenario 1` (Hardware Firmware Keyboard) by arming `0xAF` (`SET_TP_AREA`) while in `Scenario 0` prior to `enable_keyboard()`.

---

## 🚀 Quick Install

```bash
git clone https://github.com/subsubl/yogabook-c930-eink-linux.git
cd yogabook-c930-eink-linux
./install.sh
```

What `./install.sh` sets up:
1. **udev Rule (`/etc/udev/rules.d/99-yogabook-eink.rules`):** Grants non-root USB and `hidraw` access to `048d:8951`.
2. **CLI & GUI Symlinks:** `/usr/local/bin/yogabook-eink-cli` and `/usr/local/bin/yogabook-eink-gui`.
3. **Desktop Launcher:** `~/.local/share/applications/yogabook-eink-gui.desktop`.
4. **Suspend/Resume Hook (`/lib/systemd/system-sleep/yogabook-eink-resume`):** Automatically restores the E-Ink keyboard on wake from sleep.
5. **User Systemd Daemon (`yogabook-eink-daemon.service`):** Runs the background E-Ink Settings & Touch listener automatically on login.

---

## 💻 CLI Reference (`yogabook-eink-cli`)

```bash
# Query live hardware info, active scenario, and T-CON registers
yogabook-eink-cli status

# Restore E-Ink keyboard (or switch between the 4 native Lenovo hardware styles)
yogabook-eink-cli kb
yogabook-eink-cli kb --style classic_black
yogabook-eink-cli kb --style classic_white
yogabook-eink-cli kb --style modern_black
yogabook-eink-cli kb --style modern_white

# Launch E-Ink apps directly on the panel
yogabook-eink-cli matrix              # Matrix Digital Rain (Dark mode; touch anywhere to exit)
yogabook-eink-cli matrix --light      # Matrix Digital Rain (Paper-White mode)
yogabook-eink-cli doom                # Playable 3D E-Ink DOOM with touch controls

# Display custom images / wallpapers with Glider Tone LUT & 5-mode dithering
yogabook-eink-cli image ~/Pictures/wallpaper.png --fit cover --dither-mode atkinson --lightness 1 --contrast 2

# Display built-in 1920x1080 stylus templates
yogabook-eink-cli template ruled
yogabook-eink-cli template dotgrid
yogabook-eink-cli clear               # Full white INIT anti-ghosting clear
yogabook-eink-cli clear --black       # Full black INIT clear

# Mode switching & debugging
yogabook-eink-cli draw                # Enter Scenario 0 (Owner-Draw mode)
yogabook-eink-cli mt                  # Enter Scenario 3 (Multitouch / Pen-Mouse trackpad)
yogabook-eink-cli monitor             # Live-print raw HID touch & button reports (0x90 / 0x10 / 0x20)
```

---

## 🧪 Running the Test Suite

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

---

## 🙏 Credits & References

- **Lenovo Open-Source Compliance Package (`YOGA.BOOK.2.Eink.Reader_v1.0.0.5`)** — Official `itetcon.h`, `tconcmd.h`, `EinkIteAPI.h`, and `USBHIDAPI.cpp` reference headers.
- **[fivehalos/yogabook-c930-linux-eink-driver](https://github.com/fivehalos/yogabook-c930-linux-eink-driver)** & **[aleksb/yogabook-c930-linux-eink-driver](https://github.com/aleksb/yogabook-c930-linux-eink-driver)** — USB BOT protocol & Windows capture documentation.
- **Glider**, **epdoptimize**, and **KOReader** — Tone LUT, dithering, and partial-refresh E-Ink scheduling concepts.
