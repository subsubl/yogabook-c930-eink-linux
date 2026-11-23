"""Crash-safe E-Ink Display Ownership Arbitrator (Unix socket lease).

When an E-Ink app acquires a PanelOwnershipLease:
  - It connects to /run/user/<uid>/yogabook-eink.sock (served by yogabook-eink-daemon).
  - The daemon suspends its own E-Ink handling while the app holds the lease.
  - If the app exits normally OR crashes / is SIGKILL'd, the kernel closes the socket
    immediately -> the daemon detects EOF and automatically runs a full INIT clear
    + restores & refreshes the native E-Ink keyboard!
"""

from __future__ import annotations

import os
import select
import socket
import threading
import time
from typing import Callable, Optional

from .protocol import Ite8951Controller
from .usb_lib import UsbError


def get_socket_path() -> str:
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    return os.path.join(runtime_dir, "yogabook-eink.sock")


class OwnershipServer:
    """Background Unix-socket arbitrator run inside yogabook-eink-daemon."""

    def __init__(
        self,
        on_owner_acquired: Optional[Callable[[str], None]] = None,
        on_owner_released: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.sock_path = get_socket_path()
        self.on_owner_acquired = on_owner_acquired
        self.on_owner_released = on_owner_released
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._active_owner: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def active_owner(self) -> Optional[str]:
        with self._lock:
            return self._active_owner

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._serve, daemon=True, name="eink-ownership-srv"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.8)
        self._thread = None
        try:
            if os.path.exists(self.sock_path):
                os.unlink(self.sock_path)
        except OSError:
            pass

    def _serve(self) -> None:
        try:
            if os.path.exists(self.sock_path):
                os.unlink(self.sock_path)
        except OSError:
            pass

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            srv.bind(self.sock_path)
            os.chmod(self.sock_path, 0o600)
            srv.listen(4)
            srv.setblocking(False)
        except OSError as exc:
            print(f"[eink-ownership] Could not bind {self.sock_path}: {exc}", flush=True)
            srv.close()
            return

        client_sock: Optional[socket.socket] = None
        client_name = ""

        try:
            while not self._stop.is_set():
                watch = [srv] + ([client_sock] if client_sock else [])
                try:
                    ready, _, _ = select.select(watch, [], [], 0.25)
                except (OSError, ValueError):
                    break

                if srv in ready:
                    try:
                        conn, _ = srv.accept()
                    except OSError:
                        continue
                    conn.settimeout(1.0)
                    try:
                        req = conn.recv(256).decode("utf-8", errors="ignore").strip()
                    except OSError:
                        conn.close()
                        continue

                    if client_sock is not None:
                        try:
                            conn.sendall(f"BUSY {client_name}\n".encode())
                        except OSError:
                            pass
                        conn.close()
                    elif req.startswith("ACQUIRE"):
                        client_name = req.split(" ", 1)[1] if " " in req else "app"
                        client_sock = conn
                        client_sock.setblocking(False)
                        with self._lock:
                            self._active_owner = client_name
                        try:
                            client_sock.sendall(b"OK\n")
                        except OSError:
                            pass
                        print(f"[eink-ownership] Lease ACQUIRED by '{client_name}'", flush=True)
                        if self.on_owner_acquired:
                            self.on_owner_acquired(client_name)
                    else:
                        conn.close()

                if client_sock and client_sock in ready:
                    try:
                        data = client_sock.recv(128)
                    except OSError:
                        data = b""
                    if not data or b"RELEASE" in data:
                        released_name = client_name
                        clean_exit = b"CLEAN" in data
                        try:
                            client_sock.close()
                        except OSError:
                            pass
                        client_sock = None
                        client_name = ""
                        with self._lock:
                            self._active_owner = None
                        print(
                            f"[eink-ownership] Lease RELEASED by '{released_name}' "
                            f"(clean={clean_exit})",
                            flush=True,
                        )
                        if not clean_exit:
                            # Crash / SIGKILL teardown: enforce INIT clear + keyboard restore!
                            time.sleep(0.15)
                            try:
                                with Ite8951Controller() as ctrl:
                                    ctrl.exit_to_keyboard()
                                print(
                                    f"[eink-ownership] Auto-restored keyboard after '{released_name}' crash/exit.",
                                    flush=True,
                                )
                            except UsbError as exc:
                                print(f"[eink-ownership] Auto-restore failed: {exc}", flush=True)
                        if self.on_owner_released:
                            self.on_owner_released(released_name)
        finally:
            if client_sock:
                try:
                    client_sock.close()
                except OSError:
                    pass
            srv.close()


class PanelOwnershipLease:
    """Context manager used by E-Ink apps to acquire exclusive panel ownership from the daemon.

    If the daemon isn't running, gracefully degrades to a no-op so standalone CLI usage works.
    If the app crashes (SIGKILL/SIGTERM), the daemon automatically resets the panel to Keyboard.
    """

    def __init__(self, app_name: str) -> None:
        self.app_name = app_name
        self._sock: Optional[socket.socket] = None
        self.clean_teardown = False

    def __enter__(self) -> "PanelOwnershipLease":
        path = get_socket_path()
        if not os.path.exists(path):
            return self
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(1.5)
            s.connect(path)
            s.sendall(f"ACQUIRE {self.app_name}\n".encode())
            resp = s.recv(128).decode("utf-8", errors="ignore").strip()
            if resp.startswith("OK"):
                self._sock = s
            else:
                s.close()
        except OSError:
            pass
        return self

    def mark_clean_exit(self) -> None:
        self.clean_teardown = True

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._sock is not None:
            try:
                msg = b"RELEASE CLEAN\n" if self.clean_teardown else b"RELEASE\n"
                self._sock.sendall(msg)
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
