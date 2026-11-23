"""Pure-ctypes wrapper around libusb-1.0 for USB bulk communication with the ITE8951 T-CON."""

from __future__ import annotations

import ctypes
import ctypes.util
from typing import Optional

VENDOR_ID = 0x048D
PRODUCT_ID = 0x8951
INTERFACE_NUM = 0
EP_IN = 0x81
EP_OUT = 0x02

# libusb error codes
LIBUSB_SUCCESS = 0
LIBUSB_ERROR_IO = -1
LIBUSB_ERROR_INVALID_PARAM = -2
LIBUSB_ERROR_ACCESS = -3
LIBUSB_ERROR_NO_DEVICE = -4
LIBUSB_ERROR_NOT_FOUND = -5
LIBUSB_ERROR_BUSY = -6
LIBUSB_ERROR_TIMEOUT = -7
LIBUSB_ERROR_OVERFLOW = -8
LIBUSB_ERROR_PIPE = -9
LIBUSB_ERROR_INTERRUPTED = -10
LIBUSB_ERROR_NO_MEM = -11
LIBUSB_ERROR_NOT_SUPPORTED = -12
LIBUSB_ERROR_OTHER = -99

_ERR_NAMES = {
    0: "SUCCESS",
    -1: "ERROR_IO",
    -2: "ERROR_INVALID_PARAM",
    -3: "ERROR_ACCESS (permission denied — install udev rule or run with sudo)",
    -4: "ERROR_NO_DEVICE",
    -5: "ERROR_NOT_FOUND",
    -6: "ERROR_BUSY (interface claimed by another driver/process)",
    -7: "ERROR_TIMEOUT",
    -8: "ERROR_OVERFLOW",
    -9: "ERROR_PIPE (endpoint stalled)",
    -10: "ERROR_INTERRUPTED",
    -11: "ERROR_NO_MEM",
    -12: "ERROR_NOT_SUPPORTED",
    -99: "ERROR_OTHER",
}


class UsbError(RuntimeError):
    """Raised when a libusb operation fails."""

    def __init__(self, message: str, code: int = 0) -> None:
        self.code = code
        super().__init__(message)


def _load_libusb() -> ctypes.CDLL:
    lib_path = ctypes.util.find_library("usb-1.0") or "libusb-1.0.so.0"
    lib = ctypes.CDLL(lib_path)

    lib.libusb_init.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.libusb_init.restype = ctypes.c_int

    lib.libusb_exit.argtypes = [ctypes.c_void_p]
    lib.libusb_exit.restype = None

    lib.libusb_open_device_with_vid_pid.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint16,
        ctypes.c_uint16,
    ]
    lib.libusb_open_device_with_vid_pid.restype = ctypes.c_void_p

    lib.libusb_close.argtypes = [ctypes.c_void_p]
    lib.libusb_close.restype = None

    lib.libusb_claim_interface.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.libusb_claim_interface.restype = ctypes.c_int

    lib.libusb_release_interface.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.libusb_release_interface.restype = ctypes.c_int

    lib.libusb_kernel_driver_active.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.libusb_kernel_driver_active.restype = ctypes.c_int

    lib.libusb_detach_kernel_driver.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.libusb_detach_kernel_driver.restype = ctypes.c_int

    lib.libusb_bulk_transfer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ubyte,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_uint,
    ]
    lib.libusb_bulk_transfer.restype = ctypes.c_int

    lib.libusb_clear_halt.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
    lib.libusb_clear_halt.restype = ctypes.c_int

    lib.libusb_reset_device.argtypes = [ctypes.c_void_p]
    lib.libusb_reset_device.restype = ctypes.c_int

    return lib


class LibUsbDevice:
    """Context-managed bulk USB connection to the Yoga Book C930 ITE8951 T-CON."""

    def __init__(
        self,
        vid: int = VENDOR_ID,
        pid: int = PRODUCT_ID,
        interface: int = INTERFACE_NUM,
    ) -> None:
        self.vid = vid
        self.pid = pid
        self.interface = interface
        self._lib = _load_libusb()
        self._ctx: Optional[ctypes.c_void_p] = None
        self._handle: Optional[ctypes.c_void_p] = None
        self._claimed = False

    def open(self) -> None:
        if self._handle is not None:
            return
        ctx = ctypes.c_void_p()
        rc = self._lib.libusb_init(ctypes.byref(ctx))
        if rc != 0:
            raise UsbError(f"libusb_init failed: {_ERR_NAMES.get(rc, rc)}", rc)
        self._ctx = ctx

        handle = self._lib.libusb_open_device_with_vid_pid(
            self._ctx, self.vid, self.pid
        )
        if not handle:
            self.close()
            raise UsbError(
                f"Could not open USB device {self.vid:04x}:{self.pid:04x}. "
                "Check that the device is present and you have permission "
                "(run ./install.sh to install the udev rule, or run with sudo).",
                LIBUSB_ERROR_ACCESS,
            )
        self._handle = handle

        # Detach kernel driver on interface 0 only if one is attached (e.g. legacy eink.ko)
        active = self._lib.libusb_kernel_driver_active(self._handle, self.interface)
        if active == 1:
            rc = self._lib.libusb_detach_kernel_driver(self._handle, self.interface)
            if rc != 0:
                self.close()
                raise UsbError(
                    f"Failed to detach kernel driver on interface {self.interface}: "
                    f"{_ERR_NAMES.get(rc, rc)}",
                    rc,
                )

        rc = self._lib.libusb_claim_interface(self._handle, self.interface)
        if rc != 0:
            self.close()
            raise UsbError(
                f"libusb_claim_interface({self.interface}) failed: "
                f"{_ERR_NAMES.get(rc, rc)}",
                rc,
            )
        self._claimed = True

    def close(self) -> None:
        if self._handle is not None:
            if self._claimed:
                try:
                    self._lib.libusb_release_interface(self._handle, self.interface)
                except Exception:
                    pass
                self._claimed = False
            try:
                self._lib.libusb_close(self._handle)
            except Exception:
                pass
            self._handle = None
        if self._ctx is not None:
            try:
                self._lib.libusb_exit(self._ctx)
            except Exception:
                pass
            self._ctx = None

    @property
    def is_open(self) -> bool:
        return self._handle is not None and self._claimed

    def bulk_write(
        self, data: bytes, endpoint: int = EP_OUT, timeout_ms: int = 5000
    ) -> int:
        if not self.is_open:
            raise UsbError("USB device is not open")
        length = len(data)
        buf = (ctypes.c_ubyte * length).from_buffer_copy(data)
        transferred = ctypes.c_int(0)
        rc = self._lib.libusb_bulk_transfer(
            self._handle,
            endpoint,
            buf,
            length,
            ctypes.byref(transferred),
            timeout_ms,
        )
        if rc != 0:
            raise UsbError(
                f"bulk_write(ep=0x{endpoint:02x}, len={length}) failed: "
                f"{_ERR_NAMES.get(rc, rc)}",
                rc,
            )
        return transferred.value

    def bulk_read(
        self, length: int, endpoint: int = EP_IN, timeout_ms: int = 5000
    ) -> bytes:
        if not self.is_open:
            raise UsbError("USB device is not open")
        buf = (ctypes.c_ubyte * length)()
        transferred = ctypes.c_int(0)
        rc = self._lib.libusb_bulk_transfer(
            self._handle,
            endpoint,
            buf,
            length,
            ctypes.byref(transferred),
            timeout_ms,
        )
        if rc != 0:
            raise UsbError(
                f"bulk_read(ep=0x{endpoint:02x}, len={length}) failed: "
                f"{_ERR_NAMES.get(rc, rc)}",
                rc,
            )
        return bytes(buf[: transferred.value])

    def clear_halt(self, endpoint: int) -> None:
        if self.is_open:
            self._lib.libusb_clear_halt(self._handle, endpoint)

    def reset_device(self) -> None:
        if self.is_open:
            self._lib.libusb_reset_device(self._handle)

    def __enter__(self) -> "LibUsbDevice":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
