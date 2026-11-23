"""Windows-style E-Ink keyboard click sound feedback daemon for Linux.

Replicates Lenovo's EiSetKeyboardDownSounds / EiSetKeyboardUpSounds by monitoring
the ITE T-CON Keyboard evdev node and playing low-latency synthesized click sounds.
"""

from __future__ import annotations

import glob
import math
import os
import shutil
import struct
import subprocess
import tempfile
import threading
import wave
from typing import Literal, Optional

# Linux struct input_event on 64-bit: timeval (16B) + type (H) + code (H) + value (i) = 24B
INPUT_EVENT_FMT = "llHHi"
INPUT_EVENT_SIZE = struct.calcsize(INPUT_EVENT_FMT)
EV_KEY = 0x01

SoundProfile = Literal["soft", "crisp", "typewriter"]


def find_ite_keyboard_event_path() -> Optional[str]:
    """Find the /dev/input/eventX device for 'ITE Tech. Inc. ITE T-CON Keyboard'."""
    for name_path in sorted(glob.glob("/sys/class/input/event*/device/name")):
        try:
            with open(name_path, "r", encoding="utf-8", errors="ignore") as f:
                name = f.read().strip()
            if "ITE T-CON Keyboard" in name:
                event_name = name_path.split("/")[4]
                return f"/dev/input/{event_name}"
        except OSError:
            continue
    return None


def _synthesize_click_wav(
    path: str,
    freq_hz: float,
    duration_ms: float,
    amplitude: float = 0.35,
    decay: float = 18.0,
) -> None:
    sample_rate = 44100
    n_samples = int(sample_rate * duration_ms / 1000.0)
    frames = bytearray()
    for i in range(n_samples):
        t = i / sample_rate
        env = math.exp(-decay * (i / max(1, n_samples)))
        # Short damped sine + subtle harmonic for tactile click feel
        val = (
            math.sin(2.0 * math.pi * freq_hz * t)
            + 0.35 * math.sin(2.0 * math.pi * (freq_hz * 2.3) * t)
        ) * env * amplitude
        sample = max(-32767, min(32767, int(val * 32767)))
        frames += struct.pack("<h", sample)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(frames))


def _find_audio_player() -> Optional[list[str]]:
    if shutil.which("pw-play"):
        return ["pw-play"]
    if shutil.which("paplay"):
        return ["paplay"]
    if shutil.which("aplay"):
        return ["aplay", "-q"]
    return None


class KeyboardSoundDaemon:
    """Background thread that plays key-down / key-up clicks for the E-Ink keyboard."""

    def __init__(self) -> None:
        self.key_down_enabled: bool = False
        self.key_up_enabled: bool = False
        self.profile: SoundProfile = "soft"
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tmpdir = tempfile.mkdtemp(prefix="yogabook-clicks-")
        self._player = _find_audio_player()
        self._wav_files: dict[tuple[SoundProfile, str], str] = {}
        self._build_sounds()

    def _build_sounds(self) -> None:
        specs: dict[SoundProfile, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
            "soft": ((680.0, 14.0, 0.25), (920.0, 10.0, 0.14)),
            "crisp": ((1250.0, 12.0, 0.35), (1650.0, 9.0, 0.18)),
            "typewriter": ((420.0, 22.0, 0.45), (780.0, 14.0, 0.22)),
        }
        for prof, (down_spec, up_spec) in specs.items():
            down_path = os.path.join(self._tmpdir, f"{prof}_down.wav")
            up_path = os.path.join(self._tmpdir, f"{prof}_up.wav")
            _synthesize_click_wav(down_path, *down_spec)
            _synthesize_click_wav(up_path, *up_spec)
            self._wav_files[(prof, "down")] = down_path
            self._wav_files[(prof, "up")] = up_path

    def _play(self, direction: Literal["down", "up"]) -> None:
        if not self._player:
            return
        wav = self._wav_files.get((self.profile, direction))
        if not wav:
            return
        try:
            subprocess.Popen(
                [*self._player, wav],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            pass

    def start(self) -> tuple[bool, str]:
        if self._thread and self._thread.is_alive():
            return True, "Key sound listener already running."
        if not self._player:
            return False, "No audio player found (pw-play / paplay / aplay)."
        ev_path = find_ite_keyboard_event_path()
        if not ev_path:
            return False, "ITE T-CON Keyboard evdev device not found."
        if not os.access(ev_path, os.R_OK):
            return (
                False,
                f"Cannot read {ev_path} (permission denied). Add your user to the "
                "'input' group (sudo usermod -aG input $USER) and re-login.",
            )

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, args=(ev_path,), daemon=True, name="eink-key-sound"
        )
        self._thread.start()
        return True, f"Listening on {ev_path}"

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._thread = None

    def _run_loop(self, ev_path: str) -> None:
        try:
            fd = os.open(ev_path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            return
        try:
            import select

            while not self._stop_event.is_set():
                r, _, _ = select.select([fd], [], [], 0.2)
                if not r:
                    continue
                try:
                    data = os.read(fd, INPUT_EVENT_SIZE * 16)
                except BlockingIOError:
                    continue
                except OSError:
                    break
                for offset in range(0, len(data) - INPUT_EVENT_SIZE + 1, INPUT_EVENT_SIZE):
                    _, _, ev_type, _code, value = struct.unpack_from(INPUT_EVENT_FMT, data, offset)
                    if ev_type != EV_KEY:
                        continue
                    if value == 1 and self.key_down_enabled:
                        self._play("down")
                    elif value == 0 and self.key_up_enabled:
                        self._play("up")
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
