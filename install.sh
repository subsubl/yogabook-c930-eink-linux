#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Installing Yoga Book C930 E-Ink & Keyboard Control for Pop!_OS"

chmod +x "$ROOT/yogabook-eink-gui" "$ROOT/yogabook-eink-cli" "$ROOT/systemd/yogabook-eink-resume"

echo "==> Installing udev rule (allows non-root USB & hidraw access to 048d:8951)..."
sudo cp "$ROOT/systemd/99-yogabook-eink.rules" /etc/udev/rules.d/99-yogabook-eink.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=usb --attr-match=idVendor=048d --attr-match=idProduct=8951
sudo udevadm trigger --subsystem-match=hidraw

echo "==> Installing CLI & GUI symlinks to /usr/local/bin..."
sudo ln -sf "$ROOT/yogabook-eink-gui" /usr/local/bin/yogabook-eink-gui
sudo ln -sf "$ROOT/yogabook-eink-cli" /usr/local/bin/yogabook-eink-cli

echo "==> Installing Pop!_OS / COSMIC desktop launcher..."
mkdir -p "$HOME/.local/share/applications"
cp "$ROOT/systemd/yogabook-eink-gui.desktop" "$HOME/.local/share/applications/yogabook-eink-gui.desktop"

echo "==> Installing systemd sleep/resume keyboard auto-restore hook..."
sudo cp "$ROOT/systemd/yogabook-eink-resume" /lib/systemd/system-sleep/yogabook-eink-resume
sudo chmod +x /lib/systemd/system-sleep/yogabook-eink-resume

echo "==> Installing user systemd background service (E-Ink Settings button daemon)..."
mkdir -p "$HOME/.config/systemd/user"
cp "$ROOT/systemd/yogabook-eink-daemon.service" "$HOME/.config/systemd/user/yogabook-eink-daemon.service"
systemctl --user daemon-reload || true
systemctl --user enable --now yogabook-eink-daemon.service || true

echo "==> Adding $USER to 'input' group (for optional Windows-style keyboard click sounds)..."
sudo usermod -aG input "$USER" || true

echo ""
echo "Done! You can now:"
echo "  1. Run 'yogabook-eink-cli status' to verify live hardware communication."
echo "  2. Launch 'Yoga Book C930 E-Ink Control' from your Pop!_OS app menu (or run 'yogabook-eink-gui')."
echo "  3. Log out and back in once if you want the Key-Down/Key-Up click sounds to read /dev/input/event*."
