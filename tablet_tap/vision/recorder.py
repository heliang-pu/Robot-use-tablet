"""Videos of a camera, at the speed of real time, encoded by an ffmpeg process on the CPU.

``VideoWriter`` turns "the newest picture" into a video of fixed frame rate: a camera that stalls gives a still, not a
video that runs too fast, so several cameras' videos stay together. ``SharedCamera`` lets one more reader see every
frame of a camera that the console is using. Run as a program, the module records a camera nobody else has open, in a
process of its own, until its standard input is closed.
"""

from __future__ import annotations

import argparse
import contextlib
import pathlib
import subprocess
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np

from tablet_tap.vision.camera import open_camera

FPS = 30
NICE = 10  # encoding gives way to what moves the arm


class VideoWriter:
    """Pictures handed to ``put`` from any thread, written as a video; ``put`` never waits for the encoder."""

    def __init__(self, path: pathlib.Path, size: tuple[int, int], fps: int = FPS) -> None:
        self.path, self.size, self.fps = pathlib.Path(path), size, fps
        self.started: float | None = None  # time.time() of the first picture
        self.frames = 0  # written
        self.seen = 0  # handed in
        self._newest: np.ndarray | None = None
        self._stop, self._lock = threading.Event(), threading.Lock()
        self._ffmpeg = subprocess.Popen(
            ["nice", "-n", str(NICE), "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-threads", "3", "-pix_fmt", "yuv420p", str(self.path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True,
        )  # fmt: skip
        self._thread = threading.Thread(target=self._write, daemon=True, name=f"video-{self.path.stem}")
        self._thread.start()

    def put(self, image_bgr: np.ndarray) -> None:
        with self._lock:
            self._newest = image_bgr
            self.seen += 1
            if self.started is None:
                self.started = time.time()

    def _write(self) -> None:
        while not self._stop.wait(0.3 / self.fps):
            if self.started is None:
                continue
            due = int((time.time() - self.started) * self.fps) + 1 - self.frames
            if due <= 0:
                continue
            with self._lock:
                image = self._newest
            if image.shape[1::-1] != self.size:
                image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
            data = np.ascontiguousarray(image).data
            try:
                for _ in range(due):
                    self._ffmpeg.stdin.write(data)
                    self.frames += 1
            except (BrokenPipeError, OSError, ValueError):
                return

    def close(self) -> str:
        """Finish the file; what the encoder complained about ("" for nothing)."""
        self._stop.set()
        self._thread.join(timeout=5.0)
        with contextlib.suppress(OSError, ValueError):
            self._ffmpeg.stdin.close()
        try:
            self._ffmpeg.wait(timeout=20.0)
        except subprocess.TimeoutExpired:
            self._ffmpeg.kill()
            return "编码没有按时结束"
        return self._ffmpeg.stderr.read().decode(errors="replace").strip()


class SharedCamera:
    """A camera read by its own thread at its full rate; ``sink`` gets every frame, ``read`` a frame that came after the
    call. ``release`` gives the camera back as it was, also to whoever is inside ``read``."""

    TIMEOUT_S = 5.0

    def __init__(self, camera: Any, sink) -> None:  # noqa: ANN001
        self.camera, self._sink = camera, sink
        self._fresh = threading.Condition()
        self._count, self._image, self._error = 0, None, None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="shared-camera")
        self._thread.start()

    def _run(self) -> None:
        take = getattr(self.camera, "_wait", self.camera.read)  # the next frame, without dropping the ones waiting
        while not self._stop.is_set():
            try:
                image, error = take(), None
            except Exception as exc:  # noqa: BLE001 - handed to the reader, as if it had read itself
                image, error = None, exc
            with self._fresh:
                self._count, self._image, self._error = self._count + 1, image, error
                self._fresh.notify_all()
            if image is None:
                self._stop.wait(0.2)
            elif not self._stop.is_set():
                self._sink(image)

    def read(self) -> np.ndarray:
        with self._fresh:
            before = self._count
            self._fresh.wait_for(lambda: self._count > before or self._stop.is_set(), self.TIMEOUT_S)
            if self._count > before:
                if self._error is not None:
                    raise self._error
                return self._image.copy()
        if self._stop.is_set():
            return self.camera.read()
        raise RuntimeError(f"相机在 {self.TIMEOUT_S:.0f} 秒内没有出图")

    def release(self) -> Any:
        self._stop.set()
        with self._fresh:
            self._fresh.notify_all()
        self._thread.join(timeout=6.0)
        return self.camera

    def close(self) -> None:
        self.release().close()

    def __getattr__(self, name: str) -> Any:  # model() and the like
        return getattr(self.camera, name)


def main(argv: list[str] | None = None) -> int:
    """Record until standard input ends (the console closes it, or is gone)."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    stop = threading.Event()
    threading.Thread(target=lambda: (sys.stdin.read(), stop.set()), daemon=True).start()
    try:
        camera = open_camera(args.camera)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"ERROR {exc}", flush=True)
        return 1
    take = getattr(camera, "_wait", camera.read)
    writer = None
    try:
        while not stop.is_set():
            image = take()
            if writer is None:
                writer = VideoWriter(pathlib.Path(args.out), image.shape[1::-1])
                writer.put(image)
                print(f"READY {writer.started}", flush=True)
            else:
                writer.put(image)
    except RuntimeError as exc:
        print(f"ERROR {exc}", flush=True)
    finally:
        seen = 0 if writer is None else writer.seen
        complaint = "" if writer is None else writer.close()
        camera.close()
    print(f"DONE {seen} {complaint}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
