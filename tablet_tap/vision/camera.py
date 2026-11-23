"""The cameras that can look at the tablet: one port (``Camera``), three adapters, and the factory ``open_camera``."""

from __future__ import annotations

import pathlib
from typing import Protocol

import cv2
import numpy as np

from tablet_tap.core.pinhole import CameraModel


class Camera(Protocol):
    def read(self) -> np.ndarray:
        """The newest BGR frame."""
        ...

    def close(self) -> None: ...


class RealSenseColor:
    """The colour stream of one RealSense camera, chosen by serial number (BGR frames).

    Not tested on hardware. One device can be opened by one process only: a camera in use by a robot run cannot be
    opened here.
    """

    TIMEOUT_MS = 5000

    def __init__(self, serial: str, width: int = 1280, height: int = 720, fps: int = 30, warmup_frames: int = 30):
        import pyrealsense2 as rs  # noqa: PLC0415 - optional hardware dependency

        self._serial = serial
        self._shape = (height, width, 3)
        context = rs.context()
        present = sorted(device.get_info(rs.camera_info.serial_number) for device in context.query_devices())
        if serial not in present:
            raise RuntimeError(f"USB 上没有序列号为 {serial} 的 RealSense（现有：{present or '无'}）")
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._pipeline = rs.pipeline(context)
        self._pipeline.start(config)
        self._open = True
        try:
            for _ in range(warmup_frames):  # auto exposure and white balance settle
                self._wait()
        except BaseException:
            self.close()
            raise

    def _wait(self) -> np.ndarray:
        try:
            frames = self._pipeline.wait_for_frames(self.TIMEOUT_MS)
        except RuntimeError as exc:
            raise RuntimeError(f"RealSense {self._serial} 在 {self.TIMEOUT_MS / 1000:.0f} 秒内没有出图：{exc}") from exc
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError(f"RealSense {self._serial} 这一帧里没有彩色图")
        image = np.asanyarray(color.get_data())
        if image.shape != self._shape:
            raise RuntimeError(f"RealSense {self._serial} 给出的图是 {image.shape}，要的是 {self._shape}")
        return image.copy()

    def model(self) -> CameraModel:
        """The factory calibration of the colour stream at this resolution."""
        import pyrealsense2 as rs  # noqa: PLC0415 - optional hardware dependency

        stream = self._pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile()
        k = stream.get_intrinsics()
        return CameraModel(k.width, k.height, k.fx, k.fy, k.ppx, k.ppy, tuple(float(c) for c in k.coeffs))

    def read(self) -> np.ndarray:
        """A frame that arrived after this call: frames already waiting are dropped, then the next one is taken
        (so a picture taken right after a tap is not an old one; costs up to one frame period)."""
        while self._pipeline.poll_for_frames():
            pass
        return self._wait()

    def close(self) -> None:
        if self._open:
            self._open = False
            try:
                self._pipeline.stop()
            except RuntimeError:  # the device went away
                pass


class OpenCVCamera:
    """cv2.VideoCapture of a device index or a URL."""

    def __init__(self, source: int | str, width: int | None = None, height: int | None = None) -> None:
        self._source = source
        self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():
            raise RuntimeError(f"打不开相机 {source!r}")
        if width and height:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> np.ndarray:
        """The newest frame: the driver's buffered frames are skipped first."""
        for _ in range(4):
            self._capture.grab()
        ok, image = self._capture.read()
        if not ok or image is None:
            raise RuntimeError(f"相机 {self._source!r} 没有出图")
        return image

    def close(self) -> None:
        self._capture.release()


class FileCamera:
    """A single image file, or a directory whose images are returned in name order (the last one repeats)."""

    SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")

    def __init__(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path).expanduser()
        if path.is_dir():
            self._paths = sorted(p for p in path.iterdir() if p.suffix.lower() in self.SUFFIXES)
        else:
            self._paths = [path]
        if not self._paths or not all(p.is_file() for p in self._paths):
            raise RuntimeError(f"{path} 不是图片，也不是放着图片的目录")
        self._next = 0

    def read(self) -> np.ndarray:
        path = self._paths[min(self._next, len(self._paths) - 1)]
        self._next += 1
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"读不了图片 {path}")
        return image

    def close(self) -> None:
        pass


def open_camera(spec: str) -> Camera:
    """`realsense:<serial>[:WxH]` | `cv:<index-or-url>` | `file:<path>`."""
    kind, _, rest = spec.partition(":")
    if kind == "realsense" and rest:
        serial, _, size = rest.partition(":")
        if not size:
            return RealSenseColor(serial)
        try:
            width, height = (int(n) for n in size.lower().split("x"))
        except ValueError:
            raise ValueError(f"相机 {spec!r} 的分辨率应写成 宽x高，例如 realsense:{serial}:1280x720") from None
        return RealSenseColor(serial, width=width, height=height)
    if kind == "cv" and rest:
        return OpenCVCamera(int(rest) if rest.isdigit() else rest)
    if kind == "file" and rest:
        return FileCamera(rest)
    raise ValueError(f"不认识的相机 {spec!r}：应为 realsense:<序列号>[:宽x高]、cv:<编号或URL> 或 file:<路径>")


# --------------------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------------------


def camera_model(camera: Camera) -> CameraModel | None:
    """The camera's own calibration, when it has one (a RealSense's factory values); None for a webcam or a file."""
    model = getattr(camera, "model", None)
    return model() if callable(model) else None
