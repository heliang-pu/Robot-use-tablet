"""Three cameras recorded while the arm does something, when the box on the page is ticked.

The tablet's top camera is the console's own: its frames are shared. The other two are recorded by a process each, so
that nothing they do can hold up the arm or the top camera. Jobs that follow one another closely go into one recording.
Afterwards the three are also put one above the other into an upright 1080 x 1920 video.
"""

from __future__ import annotations

import datetime
import json
import logging
import pathlib
import re
import subprocess
import sys
import threading
import time
from typing import Any

import flask

from tablet_tap import config
from tablet_tap.vision import recorder

logger = logging.getLogger("tablet_tap.recording")

TOP = "平板"
OTHERS = {"全景": "realsense:262422077281:1280x720", "左腕": "realsense:262322071383:1280x720"}
FILES = {TOP: "tablet", "全景": "overview", "左腕": "wrist"}
JOBS = {"执行指令", "一键校准", "点击位置", "点按", "滑动", "回桌面", "返回"}
READY_S = 6.0  # an arm's job waits for the cameras no longer than this
LINGER_S = 5.0  # after a job: the next one within it goes into the same recording
MAX_S = 1800.0
PANEL = (1080, 608)
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
NAME = re.compile(r"\d{8}_\d{6}(_\d+)?")
FILE = re.compile(r"(douyin|tablet|overview|wrist)\.mp4")


class Child:
    """One of the other cameras, recorded by a process of its own."""

    def __init__(self, name: str, camera: str, path: pathlib.Path) -> None:
        self.name, self.path = name, path
        self.ready, self.ended = threading.Event(), threading.Event()
        self.started: float | None = None
        self.error, self.seen = "", 0
        self._process = subprocess.Popen(
            [sys.executable, "-m", "tablet_tap.vision.recorder", "--camera", camera, "--out", str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, start_new_session=True,
            cwd=pathlib.Path(__file__).resolve().parents[2],
        )  # fmt: skip
        threading.Thread(target=self._listen, daemon=True, name=f"record-{name}").start()

    def _listen(self) -> None:
        for line in self._process.stdout:
            word, _, rest = line.strip().partition(" ")
            if word == "READY":
                self.started = float(rest)
                self.ready.set()
            elif word == "DONE":
                self.seen = int(rest.split()[0])
            elif word == "ERROR":
                self.error = rest
        self.error = self.error or ("" if self.started is not None else "没有出图")
        self.ended.set()
        self.ready.set()

    def close(self) -> None:
        try:
            self._process.stdin.close()
        except OSError:
            pass
        if not self.ended.wait(30.0):
            self._process.kill()
            self.error = self.error or "没有按时结束"


class Recorder:
    def __init__(self, console: Any) -> None:
        self.console = console
        self.store = console.store.path.with_name("recording.json")
        self.on = False
        try:
            self.on = bool(json.loads(self.store.read_text())["on"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        self.children = {} if console.fake else dict(OTHERS)  # the software bench opens no real camera
        self._lock = threading.RLock()
        self._folder: pathlib.Path | None = None
        self._shared: recorder.SharedCamera | None = None
        self._top: recorder.VideoWriter | None = None
        self._others: list[Child] = []
        self._title, self._began, self._timer = "", 0.0, None
        self._making = 0  # videos being put together
        self.said = ""  # for the page
        if self.root().is_dir():
            threading.Thread(target=self._rescue, daemon=True, name="record-rescue").start()

    @staticmethod
    def root() -> pathlib.Path:
        return config.runs_dir() / "videos"

    def _rescue(self) -> None:
        """Recordings a restart cut short: their videos are there, only not put together. They ended together."""
        time.sleep(3.0)  # the encoders of the console before this one finish their files
        for folder in sorted(self.root().glob("*")):
            paths = sorted(folder.glob("*.mkv"))
            if not NAME.fullmatch(folder.name) or not paths or folder == self._folder:
                continue
            with self._lock:
                self._making += 1
            try:
                names = {file: name for name, file in FILES.items()}
                parts = []
                for path in paths:
                    final = path.with_suffix(".mp4")
                    try:
                        if path.stat().st_size == 0:
                            raise RuntimeError("空文件")
                        _ffmpeg("-i", str(path), "-c", "copy", "-movflags", "+faststart", str(final))
                        parts.append((names.get(path.stem, path.stem), final, -_seconds(final)))
                    except RuntimeError as exc:
                        logger.warning("%s：%s", path, exc)
                        if path.stat().st_size:
                            continue  # kept: it may still be mended by hand
                    path.unlink()
                title = (folder / "title").read_text(encoding="utf-8") if (folder / "title").exists() else ""
                if parts:
                    (folder / "recording.json").write_text(json.dumps({"title": title, "notes": ["重启时中断，后来补合成"]}, ensure_ascii=False))
                    upright(folder, title, parts)
                    self._say(f"■ 重启前那段录像补合成好了：{folder}")
            except Exception:  # noqa: BLE001
                logger.exception("recording %s", folder)
            finally:
                with self._lock:
                    self._making -= 1

    @property
    def recording(self) -> bool:
        return self._folder is not None

    def set_on(self, on: bool) -> None:
        self.on = on
        self.store.parent.mkdir(parents=True, exist_ok=True)
        self.store.write_text(json.dumps({"on": on}))
        if not on:
            threading.Thread(target=self.stop, daemon=True, name="record-stop").start()

    # ---- around the arm's jobs -----------------------------------------------------------------------------------------
    def before(self, job: str, title: str = "") -> None:
        """Called by the console's worker before a job: returns when the cameras record, or cannot."""
        if not self.on or job not in JOBS:
            return
        try:
            self.start(title or job)
        except Exception as exc:  # noqa: BLE001 - a video must never keep the arm from its job
            logger.exception("recording")
            self._say(f"⚠ 没有录像：{exc}")

    def after(self, job: str) -> None:
        with self._lock:
            if self._folder is None or job not in JOBS:
                return
            self._cancel()
            self._timer = threading.Timer(LINGER_S, self.stop)
            self._timer.daemon = True
            self._timer.start()

    def _cancel(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _say(self, text: str) -> None:
        self.said = text
        self.console.say(text)

    def start(self, title: str) -> None:
        with self._lock:
            self._cancel()
            if self._folder is not None:
                if time.monotonic() - self._began < MAX_S:
                    return
                self.stop()
            if self.console.camera is None:
                raise RuntimeError("顶部相机没有打开")
            if not self.console.fake:
                self.console._hardware_free("录像")  # noqa: SLF001 - a robot run may have the other cameras
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            folder = self.root() / stamp
            for i in range(2, 100):
                if not folder.exists():
                    break
                folder = self.root() / f"{stamp}_{i}"
            folder.mkdir(parents=True)
            (folder / "title").write_text(title, encoding="utf-8")  # should a restart cut it short
            self._folder, self._title, self._began = folder, title, time.monotonic()
            camera = self.console.camera
            wrist = getattr(self.console, "wrist", None)
            taken = {wrist.wrist.camera.split(":")[1]} if wrist is not None and wrist.use and wrist.wrist.camera.count(":") else set()
            self._others = [Child(name, spec, folder / f"{FILES[name]}.mkv") for name, spec in self.children.items()
                            if not (spec.count(":") and spec.split(":")[1] in taken)]  # fmt: skip
            self._shared = recorder.SharedCamera(camera, self._frame)
            self.console.camera = self._shared
        deadline = time.monotonic() + READY_S
        for child in self._others:
            child.ready.wait(max(0.0, deadline - time.monotonic()))
        lost = [f"{c.name}（{c.error or '还没出图'}）" for c in self._others if c.started is None]
        self._say("● 开始录像" + (f"；这几路没有：{'、'.join(lost)}" if lost else f"（{1 + len(self._others)} 路）"))

    def _frame(self, image: Any) -> None:
        top = self._top
        if top is None:
            folder = self._folder
            if folder is None:
                return
            top = self._top = recorder.VideoWriter(folder / f"{FILES[TOP]}.mkv", image.shape[1::-1])
        top.put(image)

    def stop(self) -> None:
        with self._lock:
            if self._folder is None:
                return
            self._cancel()
            folder, title, shared, others = self._folder, self._title, self._shared, self._others
            if shared is not None:
                camera = shared.release()
                if self.console.camera is shared:  # not closed or changed meanwhile
                    self.console.camera = camera
            top = self._top
            for child in others:  # all at once: they end together
                threading.Thread(target=child.close, daemon=True).start()
            self._folder, self._shared, self._top, self._others = None, None, None, []
            self._making += 1
        threading.Thread(target=self._finish, args=(folder, title, top, others), daemon=True, name="record-finish").start()

    def _finish(self, folder: pathlib.Path, title: str, top: recorder.VideoWriter | None, others: list[Child]) -> None:
        try:
            parts: list[tuple[str, pathlib.Path, float]] = []
            notes = []
            if top is not None:
                complaint = top.close()
                notes += [f"{TOP}：{complaint}"] if complaint else []
                if top.started is not None:
                    parts.append((TOP, top.path, top.started))
            for child in others:
                child.ended.wait(35.0)
                if child.started is not None:
                    parts.append((child.name, child.path, child.started))
                if child.error:
                    notes.append(f"{child.name}：{child.error}")
            meta = {"title": title, "notes": notes, "fps_seen": {}, "start": {name: start for name, _, start in parts}}
            if top is not None and top.started is not None:
                meta["fps_seen"][TOP] = round(top.seen / max(0.1, top.frames / top.fps), 1)
            (folder / "recording.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
            for _, path, _ in parts:
                final = path.with_suffix(".mp4")
                _ffmpeg("-i", str(path), "-c", "copy", "-movflags", "+faststart", str(final))
                path.unlink()
            (folder / "title").unlink(missing_ok=True)
            if parts:
                upright(folder, title, [(name, path.with_suffix(".mp4"), start) for name, path, start in parts])
            self._say(f"■ 录像好了：{folder}（{len(parts)} 路 + 抖音竖屏）" + (f"；{'；'.join(notes)}" if notes else ""))
        except Exception as exc:  # noqa: BLE001
            logger.exception("recording")
            self._say(f"⚠ 录像没有合成好：{exc}（原始视频在 {folder}）")
        finally:
            with self._lock:
                self._making -= 1

    # ---- the page --------------------------------------------------------------------------------------------------
    def listing(self) -> list[dict[str, Any]]:
        rows = []
        for folder in sorted(self.root().glob("*"), reverse=True):
            files = {p.name: p.stat().st_size for p in folder.glob("*.mp4") if FILE.fullmatch(p.name)}
            if not NAME.fullmatch(folder.name) or not files:
                continue
            try:
                title = json.loads((folder / "recording.json").read_text())["title"]
            except (OSError, ValueError, KeyError):
                title = ""
            rows.append({"id": folder.name, "title": title, "files": files})
        return rows

    def state(self) -> dict[str, Any]:
        return {"on": self.on, "recording": self.recording, "making": self._making > 0, "said": self.said,
                "seconds": round(time.monotonic() - self._began) if self.recording else 0}  # fmt: skip

    def close(self) -> None:
        self.stop()


def _ffmpeg(*args: str) -> None:
    done = subprocess.run(["nice", "-n", "15", "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
                          capture_output=True, text=True, stdin=subprocess.DEVNULL, check=False)  # fmt: skip
    if done.returncode:
        raise RuntimeError(done.stderr.strip()[-300:] or f"ffmpeg 退出码 {done.returncode}")


def _seconds(path: pathlib.Path) -> float:
    done = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                          capture_output=True, text=True, check=False)  # fmt: skip
    try:
        return float(done.stdout)
    except ValueError:
        raise RuntimeError(f"{path.name} 没有时长") from None


def upright(folder: pathlib.Path, title: str, parts: list[tuple[str, pathlib.Path, float]]) -> pathlib.Path:
    """The videos one above the other, beginning together, with the title in the strip above: 1080 x 1920."""
    order = {"全景": 0, TOP: 1, "左腕": 2}
    parts = sorted(parts, key=lambda p: order.get(p[0], 9))
    latest = max(start for _, _, start in parts)
    inputs, chains = [], []
    for i, (_, path, start) in enumerate(parts):
        inputs += ["-ss", f"{latest - start:.3f}", "-i", str(path)]
        chains.append(f"[{i}:v]scale={PANEL[0]}:{PANEL[1]}:force_original_aspect_ratio=decrease,pad={PANEL[0]}:{PANEL[1]}:-1:-1,setsar=1[v{i}]")
    stacked = "".join(f"[v{i}]" for i in range(len(parts)))
    stack = f"{stacked}vstack=inputs={len(parts)}:shortest=1" if len(parts) > 1 else "[v0]null"
    strip = 96
    tail = f"pad=1080:1920:0:{strip}+(1920-{strip}-ih)/2"
    if pathlib.Path(FONT).exists() and title:
        words = folder / "title.txt"
        words.write_text(title[:40], encoding="utf-8")
        size = 44 if len(title) <= 22 else 26
        tail += f",drawtext=fontfile={FONT}:textfile={words}:expansion=none:fontsize={size}:fontcolor=white:x=(w-tw)/2:y=(96-th)/2"
    out = folder / "douyin.mp4"
    _ffmpeg(*inputs, "-filter_complex", ";".join(chains) + f";{stack}[s];[s]{tail}[o]", "-map", "[o]", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "19", "-threads", "6", "-pix_fmt", "yuv420p", "-r", str(recorder.FPS), "-movflags", "+faststart", str(out))  # fmt: skip
    (folder / "title.txt").unlink(missing_ok=True)
    return out


PAGE = """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>录像</title>
<style>body{font:15px/1.6 system-ui,sans-serif;max-width:860px;margin:24px auto;padding:0 16px}h1{font-size:20px}li{margin:10px 0}a{margin-right:12px}</style>
<h1>录像</h1><p>“抖音竖屏”是三路合成的 1080×1920；下载：右键“链接存储为”。没有声音。发之前看一遍屏幕上有没有手机号、账号。</p><ul>{% for r in rows %}
<li>{{ r.id[4:6] }}-{{ r.id[6:8] }} {{ r.id[9:11] }}:{{ r.id[11:13] }}:{{ r.id[13:15] }}　{{ r.title }}<br>{% for f, n in r.files.items() %}<a href="/api/rec/{{ r.id }}/{{ f }}{{ q }}" download="{{ r.id }}_{{ f }}">{{ names[f] }}（{{ '%.1f'|format(n / 1e6) }} MB）</a>{% endfor %}</li>
{% else %}<li>还没有录像</li>{% endfor %}</ul></html>"""


def register(app: flask.Flask, console: Any, token: str = "") -> Recorder:
    recorder_ = console.recorder = Recorder(console)

    def refused() -> Any:
        if token and flask.request.args.get("token", "") != token and flask.request.headers.get("X-Token", "") != token:
            return flask.jsonify({"error": "需要口令（网址后加 ?token=…）"}), 401
        return None

    @app.get("/api/rec/state")
    def rec_state() -> Any:
        return refused() or flask.jsonify(recorder_.state())

    @app.post("/api/rec/on")
    def rec_on() -> Any:
        denied = refused()
        if denied is not None:
            return denied
        data = flask.request.get_json(silent=True) if flask.request.is_json else None
        if not isinstance(data, dict) or not isinstance(data.get("on"), bool):
            return flask.jsonify({"error": "要给出 on"}), 400
        recorder_.set_on(data["on"])
        return flask.jsonify(recorder_.state())

    @app.get("/rec")
    def rec_page() -> Any:
        names = {"douyin.mp4": "抖音竖屏", "tablet.mp4": "平板", "overview.mp4": "全景", "wrist.mp4": "左腕"}
        return refused() or flask.render_template_string(PAGE, rows=recorder_.listing(), names=names, q=f"?token={token}" if token else "")

    @app.get("/api/rec/<name>/<file>")
    def rec_file(name: str, file: str) -> Any:
        denied = refused()
        if denied is not None:
            return denied
        if not NAME.fullmatch(name) or not FILE.fullmatch(file):
            return flask.jsonify({"error": "没有这个视频"}), 404
        return flask.send_from_directory(recorder_.root() / name, file, conditional=True)

    return recorder_
