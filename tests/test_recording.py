"""Videos of the cameras while the arm does something (web.recording), with pictures made here and real ffmpeg."""

import json
import pathlib
import shutil
import subprocess
import threading
import time

import numpy as np
import pytest

from tablet_tap import config
from tablet_tap.calibration import page_server
from tablet_tap.vision import recorder
from tablet_tap.web import console as web
from tablet_tap.web import recording

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="no ffmpeg")


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height,codec_name:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout  # fmt: skip
    data = json.loads(out)
    return data["streams"][0]["width"], data["streams"][0]["height"], float(data["format"]["duration"])


def wait_for(done, timeout=20.0):
    end = time.monotonic() + timeout
    while not done():
        assert time.monotonic() < end
        time.sleep(0.05)


class Camera:
    def __init__(self, hz=30.0):
        self.hz, self.n, self.closed, self.fails = hz, 0, False, False

    def read(self):
        time.sleep(1 / self.hz)
        if self.fails:
            raise RuntimeError("no frame")
        self.n += 1
        return np.full((90, 160, 3), self.n % 256, np.uint8)

    def model(self):
        return "model"

    def close(self):
        self.closed = True


def test_a_slow_camera_makes_a_video_at_the_speed_of_real_time(tmp_path):
    writer = recorder.VideoWriter(tmp_path / "a.mkv", (160, 90))
    began = time.monotonic()
    while time.monotonic() - began < 1.5:
        writer.put(np.zeros((90, 160, 3), np.uint8))
        time.sleep(0.2)  # 5 pictures a second
    assert writer.close() == ""
    assert probe(tmp_path / "a.mkv")[2] == pytest.approx(1.5, abs=0.35)


def test_a_shared_camera_is_read_by_the_console_as_before_and_given_back():
    camera, got = Camera(), []
    shared = recorder.SharedCamera(camera, got.append)
    first = shared.read()
    second = shared.read()
    assert second[0, 0, 0] > first[0, 0, 0] and shared.model() == "model"
    wait_for(lambda: len(got) > 5)
    waiting = threading.Thread(target=lambda: got.append(shared.read()))
    waiting.start()
    assert shared.release() is camera and not camera.closed
    waiting.join(5.0)
    assert not waiting.is_alive()  # whoever was reading got a frame
    assert shared.read().shape == (90, 160, 3)  # from the camera itself


def test_a_failing_shared_camera_tells_the_reader():
    camera = Camera()
    camera.fails = True
    shared = recorder.SharedCamera(camera, lambda image: None)
    with pytest.raises(RuntimeError, match="no frame"):
        shared.read()
    camera.fails = False
    wait_for(lambda: shared._error is None and shared._image is not None)
    assert shared.read() is not None
    shared.close()
    assert camera.closed


@pytest.fixture
def bench(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(recording, "LINGER_S", 0.4)
    con = web.Console(config.SetupStore(tmp_path / "setup.json"), tmp_path / "view.json", page_server.CalibState(), fake=True, realtime=False)
    client = web.create_app(con).test_client()
    con.camera = Camera()
    con.recorder.children = {"全景": "stub", "左腕": "stub"}
    yield con, client, tmp_path
    con.close()


class StubChild:
    """A camera's own process, without the process."""

    def __init__(self, name, camera, path):
        self.name, self.path, self.error, self.seen = name, path, "", 0
        self.ready, self.ended = threading.Event(), threading.Event()
        self.started = None
        self._stop = threading.Event()
        if name == "左腕" and StubChild.broken:
            self.error = "USB 上没有"
            self.ready.set()
            self.ended.set()
            return
        self.started = time.time()
        self.ready.set()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        writer = recorder.VideoWriter(self.path, (160, 90))
        while not self._stop.wait(0.05):
            writer.put(np.full((90, 160, 3), 90, np.uint8))
        writer.close()
        self.ended.set()

    def close(self):
        self._stop.set()


def run_job(con, name, seconds=1.0):
    assert con.submit(name, lambda: time.sleep(seconds))
    wait_for(lambda: con.busy is None)


@pytest.mark.parametrize("broken", [False, True])
def test_ticked_every_operation_is_recorded_and_put_upright(bench, monkeypatch, broken):
    con, client, _ = bench
    StubChild.broken = broken
    monkeypatch.setattr(recording, "Child", StubChild)
    camera = con.camera
    run_job(con, "执行指令", 0.1)
    assert client.get("/api/rec/state").get_json()["recording"] is False  # not ticked
    assert client.post("/api/rec/on", json={"on": "yes"}).status_code == 400
    assert client.post("/api/rec/on", json={"on": True}).get_json()["on"]
    run_job(con, "连接", 0.1)
    assert not con.recorder.recording  # not an operation
    con.job_title = "打开音乐，播放音乐"
    run_job(con, "执行指令")
    assert con.recorder.recording  # for a little longer: the next one goes into the same video
    run_job(con, "点击位置")
    wait_for(lambda: not con.recorder.recording and not con.recorder.state()["making"], 40)
    assert con.camera is camera and not camera.closed
    (row,) = con.recorder.listing()
    files = {"douyin.mp4", "tablet.mp4", "overview.mp4"} | (set() if broken else {"wrist.mp4"})
    assert row["title"] == "打开音乐，播放音乐" and set(row["files"]) == files
    folder = con.recorder.root() / row["id"]
    width, height, seconds = probe(folder / "douyin.mp4")
    assert (width, height) == (1080, 1920) and seconds > 1.7
    assert not list(folder.glob("*.mkv"))
    page = client.get("/rec").get_data(as_text=True)
    assert "打开音乐，播放音乐" in page and "抖音竖屏" in page
    assert client.get(f"/api/rec/{row['id']}/douyin.mp4").status_code == 200
    assert client.get(f"/api/rec/{row['id']}/recording.json").status_code == 404
    assert client.get("/api/rec/..%2F..%2Fsetup/douyin.mp4").status_code == 404
    assert json.loads((pathlib.Path(con.store.path).with_name("recording.json")).read_text()) == {"on": True}  # kept


def test_a_recording_that_cannot_start_does_not_keep_the_arm(bench):
    con, client, _ = bench
    con.camera = None
    con.recorder.on = True
    ran = []
    con.submit("执行指令", lambda: ran.append(1))
    wait_for(lambda: con.busy is None)
    log = "\n".join(line["text"] for line in client.get("/api/state").get_json()["log"])
    assert ran and "没有录像" in log


def test_a_recording_cut_short_by_a_restart_is_put_together_by_the_next_console(bench, monkeypatch):
    con, _, tmp = bench
    folder = con.recorder.root() / "20261124_220000"
    folder.mkdir(parents=True)
    (folder / "title").write_text("打开音乐", encoding="utf-8")
    for name, seconds in (("tablet", 1.6), ("overview", 1.0)):
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc=s=160x90:r=30:d={seconds}", "-c:v", "libx264", str(folder / f"{name}.mkv")], check=True)
    (folder / "wrist.mkv").write_bytes(b"")
    real = time.sleep
    monkeypatch.setattr(recording.time, "sleep", lambda s: real(min(s, 0.05)))
    con.recorder._rescue()
    (row,) = con.recorder.listing()
    assert row["title"] == "打开音乐" and set(row["files"]) == {"douyin.mp4", "tablet.mp4", "overview.mp4"}
    assert probe(folder / "douyin.mp4")[2] == pytest.approx(1.0, abs=0.2)  # they ended together
    assert not list(folder.glob("*.mkv"))
