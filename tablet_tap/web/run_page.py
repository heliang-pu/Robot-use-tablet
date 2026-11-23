"""The run page (/run): an instruction, typed or spoken, is carried out on the tablet by the GUI agent, step by step.

It lives in the calibration console's process and uses that console's one worker, its stop and its arm, so nothing else
can move the arm while an instruction runs. The page shows every step: what the camera saw, what the model said, what
the arm did. Every run is also on disk (runs/<id>/), which is where the page reads finished runs from.
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import datetime
import json
import pathlib
import re
import threading
import time
from typing import Any

import cv2
import flask
import numpy as np
import requests

from tablet_tap import config
from tablet_tap.agent import gui_agent
from tablet_tap.agent.robot_device import RobotTabletDevice
from tablet_tap.core.keyboard import ANCHORS
from tablet_tap.core.keyboard import Keyboard
from tablet_tap.errors import SetupError
from tablet_tap.errors import Stopped
from tablet_tap.robot.tablet_robot import ArmPlace
from tablet_tap.speech import voice

STATIC_DIR = pathlib.Path(__file__).resolve().parent / "static"
JOB_NAME = "执行指令"
DEFAULT_PRESETS = ["打开音乐", "打开音乐，播放音乐", "暂停播放", "播放下一首", "把音量调大一点", "回到桌面"]
MAX_INSTRUCTION = 1000  # characters
MAX_ADD_INFO = 1000
MAX_PRESETS = 30
MAX_WAV_BYTES = 8 * 1024 * 1024  # about four minutes of 16 kHz mono
RUN_ID = re.compile(r"[0-9A-Za-z_-]{1,40}")
HISTORY = 30


@dataclasses.dataclass(frozen=True)
class RunOptions:
    vlm_url: str = "http://127.0.0.1:8802/v1"
    asr_url: str = "http://127.0.0.1:8803"
    model: str = "gui-owl"


# ---- are the services up ------------------------------------------------------------------------------------------
class ServiceProbe:
    """Whether the model server and the speech service answer. Asked in the background: the page is polled often."""

    def __init__(self, options: RunOptions, period_s: float = 3.0, timeout_s: float = 1.5) -> None:
        self.options, self.period_s, self.timeout_s = options, period_s, timeout_s
        self._lock = threading.Lock()
        self._asked = -1e9
        self._running = False
        unknown = {"ok": False, "detail": "还没有检查"}
        self._status: dict[str, dict[str, Any]] = {"vlm": dict(unknown), "asr": dict(unknown)}

    def status(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            if not self._running and time.monotonic() - self._asked > self.period_s:
                self._running, self._asked = True, time.monotonic()
                threading.Thread(target=self.check, daemon=True, name="service-probe").start()
            return {name: dict(value) for name, value in self._status.items()}

    def check(self) -> dict[str, dict[str, Any]]:
        status = {"vlm": self._vlm(), "asr": self._asr()}
        with self._lock:
            self._status, self._running = status, False
        return status

    def _get(self, url: str) -> requests.Response:
        session = requests.Session()
        session.trust_env = False  # local services, never through the machine's proxy
        return session.get(url, timeout=self.timeout_s)

    def _vlm(self) -> dict[str, Any]:
        try:
            answer = self._get(f"{self.options.vlm_url.rstrip('/')}/models")
            names = [str(m.get("id")) for m in answer.json().get("data", [])]
        except (requests.RequestException, ValueError, AttributeError):
            return {"ok": False, "detail": "没有启动（bash scripts/serve_vlm.sh）"}
        if self.options.model not in names:
            return {"ok": False, "detail": f"服务在，但没有叫 {self.options.model} 的模型（有：{names}）"}
        return {"ok": True, "detail": self.options.model}

    def _asr(self) -> dict[str, Any]:
        try:
            answer = self._get(f"{self.options.asr_url.rstrip('/')}/health")
        except requests.RequestException:
            return {"ok": False, "detail": "没有启动（bash scripts/serve_asr.sh）"}
        return (
            {"ok": True, "detail": "SenseVoice"}
            if answer.status_code == 200
            else {"ok": False, "detail": "正在加载模型"}
        )


class ConsoleCamera:
    """The console's camera as a ``vision.camera.Camera``: every read is a frame taken after the call. The console's own
    thread is the only reader of the device; a second one would take every other frame from it."""

    def __init__(self, console: Any) -> None:
        self.console = console

    def read(self) -> np.ndarray:
        return self.console.fresh_frame()

    def close(self) -> None:
        pass  # the console owns the device


# ---- runs ---------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Run:
    id: str
    instruction: str
    started: float
    max_steps: int
    status: str = "running"
    answer: str = ""
    finished: float | None = None
    steps: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    def to_json(self, since: int = 0) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["n_steps"] = len(self.steps)
        data["steps"] = self.steps[since:]
        data["elapsed_s"] = round((self.finished or time.time()) - self.started, 1)
        return data


def _tidy(text: object) -> str:
    """The text without stray blanks; its line breaks stay."""
    return "\n".join(line for line in (" ".join(raw.split()) for raw in str(text).splitlines()) if line)


def _did(action: dict[str, Any], record: dict[str, Any]) -> str:
    """What the step did, in a few words."""
    if not action:
        return record.get("note", "")
    if record.get("status"):  # the step that ended the run: how it ended, rather than the action's name
        text = STATUS_TEXT.get(record["status"], record["status"])
        return f"结束：{text}" + (f"｜{record['answer']}" if record.get("answer") else "")
    return gui_agent.describe(action, record)


def step_summary(record: dict[str, Any]) -> dict[str, Any]:
    """What the page needs of one step of the agent's log."""
    action = record.get("action") or {}
    return {
        "step": record["step"],
        "thought": re.sub(r"^Action:\s*", "", gui_agent.thought(record.get("output", ""))),
        "did": _did(action, record),
        "action": action.get("action", ""),
        "coords": record.get("coords", {}),
        "annotated": bool(record.get("annotated")),
        "llm_s": record.get("llm_s", 0.0),
        "device_s": record.get("device_s", 0.0),
        "status": record.get("status", ""),
        "note": record.get("note", ""),
        "output": record.get("output", ""),
    }


class RunSession:
    """The instructions given on the page: one at a time, in the console's worker."""

    def __init__(
        self,
        console: Any,
        options: RunOptions | None = None,
        llm_factory: Callable[[], Any] | None = None,
        probe: ServiceProbe | None = None,
    ) -> None:
        self.console, self.options, self.llm_factory = console, options or RunOptions(), llm_factory
        self.probe = probe or ServiceProbe(self.options)
        self.current: Run | None = None
        self._lock = threading.Lock()
        self._mic: tuple[threading.Thread, threading.Event, dict[str, Any]] | None = None

    # ---- starting ----------------------------------------------------------------------------------------------------
    def missing(self, services: dict[str, dict[str, Any]] | None = None) -> list[str]:
        """What stands between the operator and a run, in words; empty when an instruction can be carried out."""
        con, services = self.console, services or self.probe.status()
        wants = []
        if con.arm is None:
            wants.append("机械臂没有连接（在标定控制台里以“可运动”连接）")
        elif not con.moves:
            wants.append("机械臂是只读连接，不能运动（重新以“可运动”连接）")
        if con.camera is None:
            wants.append("相机没有打开")
        if con.rectifier is None:
            wants.append("相机还没有对屏幕标定（标定控制台里的“相机标定”）")
        if con.setup.screen_map is None or con.setup.park_q is None:
            wants.append("屏幕平面或停靠位姿还没有标定")
        elif con.from_camera:
            wants.append("屏幕还没有校准（在“点按”页点“一键校准精度”）")
        if self.llm_factory is None and not services["vlm"]["ok"]:
            wants.append(f"GUI-Owl 模型服务：{services['vlm']['detail']}")
        return wants

    def start(
        self, instruction: str, max_steps: int = 15, settle_s: float = 0.8, add_info: str | None = None
    ) -> Run:
        """``add_info``: the note appended to the instruction; None is the usual one, "" none at all."""
        instruction = _tidy(instruction)
        if add_info is not None:
            add_info = _tidy(add_info)
            if len(add_info) > MAX_ADD_INFO:
                raise ValueError(f"附加说明太长（{len(add_info)} 字，最多 {MAX_ADD_INFO}）")
        if not instruction:
            raise ValueError("指令是空的")
        if len(instruction) > MAX_INSTRUCTION:
            raise ValueError(f"指令太长（{len(instruction)} 字，最多 {MAX_INSTRUCTION}）")
        if not 1 <= int(max_steps) <= 40:
            raise ValueError("最多步数应在 1–40 之间")
        if not 0.0 <= float(settle_s) <= 10.0:
            raise ValueError("每步等待应在 0–10 秒之间")
        wants = self.missing(None if self.llm_factory else self.probe.check())
        if wants:
            raise SetupError("还不能执行：" + "；".join(wants))
        run = Run(self._new_id(), instruction, time.time(), int(max_steps))
        if not self.console.submit(JOB_NAME, lambda: self._job(run, float(settle_s), add_info)):
            raise BusyError(f"正在执行“{self.console.busy}”，这条指令没有执行（不排队）")
        with self._lock:
            self.current = run
        self.console.say(f"▶ {JOB_NAME}：{instruction}")
        return run

    @staticmethod
    def _new_id() -> str:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        for suffix in ("", *(f"_{i}" for i in range(2, 100))):
            if not (config.runs_dir() / (stamp + suffix)).exists():
                return stamp + suffix
        raise RuntimeError("no free run directory")

    def _job(self, run: Run, settle_s: float, note: str | None = None) -> None:
        con, robot = self.console, None
        try:
            robot = con.touching_robot()
            keyboard = con.setup.on_screen_keyboard()
            device = RobotTabletDevice(robot, ConsoleCamera(con), con.rectifier, nav=con.setup.nav, keyboard=keyboard)
            agent = gui_agent.GuiAgent(
                device,
                base_url=self.options.vlm_url,
                model=self.options.model,
                max_steps=run.max_steps,
                settle_s=settle_s,
                add_info=gui_agent.add_info_for(typing=keyboard is not None) if note is None else note,
                should_stop=con.stop.is_set,
                run_dir=config.runs_dir() / run.id,
                llm=self.llm_factory() if self.llm_factory else None,
                llm_max_retry=2,  # a server that went away must not hold the arm's worker for minutes
                llm_retry_wait_s=1.0,
                on_step=lambda record: self._on_step(run, record),
            )
            result = agent.run(run.instruction)
            self._finish(run, result.status, result.answer)
        except Stopped:
            self._finish(run, "stopped", "已停止，机械臂保持在当前位置。")
            raise
        except Exception as exc:
            self._finish(run, "error", f"{type(exc).__name__}: {exc}" if not str(exc) else str(exc))
            raise
        finally:
            if robot is not None and robot.where is ArmPlace.HOVER and not con.stop.is_set():
                robot.to_park()
        con.say(
            f"■ {JOB_NAME}结束：{STATUS_TEXT.get(run.status, run.status)}" + (f"，{run.answer}" if run.answer else "")
        )

    def _on_step(self, run: Run, record: dict[str, Any]) -> None:
        with self._lock:
            run.steps.append(step_summary(record))

    def _finish(self, run: Run, status: str, answer: str) -> None:
        with self._lock:
            run.status, run.answer, run.finished = status, answer, time.time()

    # ---- what the page shows -----------------------------------------------------------------------------------------
    def state(self, run_id: str = "", since: int = 0) -> dict[str, Any]:
        services = self.probe.status()
        with self._lock:
            run = self.current
            shown = None if run is None else run.to_json(since if run_id == run.id else 0)
        return {
            "run": shown,
            "busy": self.console.busy,
            "stopping": self.console.stop.is_set() and self.console.busy is not None,
            "fake": self.console.fake,
            "fault": self.console.fault,
            "missing": self.missing(services),
            "services": services,
            "recording": self._mic is not None,
            "camera_open": self.console.camera is not None,
            "keyboard": self.console.setup.keyboard is not None,
            "add_info": gui_agent.add_info_for(typing=self.console.setup.keyboard is not None),
        }

    def history(self) -> list[dict[str, Any]]:
        """Finished runs on disk, newest first."""
        rows = []
        for path in sorted(config.runs_dir().glob("*/result.json"), reverse=True):
            if len(rows) >= HISTORY or not RUN_ID.fullmatch(path.parent.name):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if "instruction" in data:
                rows.append({key: data.get(key) for key in ("instruction", "status", "steps", "answer", "started")}
                            | {"id": path.parent.name})  # fmt: skip
        return rows

    def load(self, run_id: str) -> dict[str, Any] | None:
        """One run: the one in memory, or a finished one from its records on disk."""
        with self._lock:
            if self.current is not None and self.current.id == run_id:
                return self.current.to_json()
        folder = run_folder(run_id)
        if folder is None or not (folder / "result.json").exists():
            return None
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        log = folder / "actions.jsonl"
        lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
        steps = [step_summary(json.loads(line)) for line in lines if line.strip()]
        return {"id": run_id, "instruction": result.get("instruction", ""), "status": result.get("status", ""),
                "answer": result.get("answer", ""), "n_steps": len(steps), "steps": steps}  # fmt: skip

    # ---- the on-screen keyboard ------------------------------------------------------------------------------------
    def set_keyboard(self, anchors: Any) -> dict[str, list[float]] | None:
        """Remember where the keyboard's keys are (``anchors``: key -> [u, v]; None forgets them)."""
        if anchors is None:
            self.console.setup.keyboard = None
        else:
            if not isinstance(anchors, dict):
                raise ValueError("键盘标定应是 键名 → [u, v] 的对照表")
            keys = {str(k): (float(v[0]), float(v[1])) for k, v in anchors.items()}
            Keyboard(keys)  # checks the keys and their arrangement; SetupError says which key to point at again
            self.console.setup.keyboard = {k: list(v) for k, v in keys.items()}
        self.console.store.save(self.console.setup)
        return self.console.setup.keyboard

    # ---- speech ------------------------------------------------------------------------------------------------------
    def transcribe(self, wav: bytes) -> str:
        return voice.transcribe(wav, url=self.options.asr_url)

    def mic_start(self, device: str | None = None) -> None:
        """Record from this computer's microphone until ``mic_stop``."""
        with self._lock:
            if self._mic is not None:
                raise BusyError("正在录音")
            stop, box = threading.Event(), {}

            def record() -> None:
                try:
                    box["wav"] = voice.record_wav(device=device, stop=stop)
                except Exception as exc:  # noqa: BLE001 - reported by mic_stop
                    box["error"] = exc

            thread = threading.Thread(target=record, daemon=True, name="run-mic")
            self._mic = (thread, stop, box)
        thread.start()

    def mic_stop(self) -> str:
        with self._lock:
            mic, self._mic = self._mic, None
        if mic is None:
            raise ValueError("没有在录音")
        thread, stop, box = mic
        stop.set()
        thread.join(timeout=10.0)
        if "error" in box:
            raise box["error"]
        if "wav" not in box:
            raise voice.VoiceError("录音没有正常结束")
        voice.require_signal(box["wav"])
        return self.transcribe(box["wav"])


class BusyError(RuntimeError):
    """Something else is under way; nothing was started."""


STATUS_TEXT = {
    "running": "正在执行",
    "success": "完成",
    "answer": "完成（有回答）",
    "failure": "没有完成",
    "need_user": "需要人处理",
    "unsupported": "需要打字，机械臂还不会",
    "max_steps": "步数用完",
    "parse_error": "模型输出看不懂",
    "llm_error": "模型服务出错",
    "stopped": "已停止",
    "error": "出错",
    "exception": "出错",
}


def run_folder(run_id: str) -> pathlib.Path | None:
    if not RUN_ID.fullmatch(run_id):
        return None
    folder = config.runs_dir() / run_id
    return folder if folder.is_dir() else None


def step_image(run_id: str, step: int, *, annotated: bool, width: int = 900) -> bytes | None:
    """A step's screenshot (with the model's click drawn on it, when there is one) as a JPEG of at most ``width``."""
    folder = run_folder(run_id)
    if folder is None:
        return None
    path = folder / f"step_{step:02d}_anno.png"
    if not annotated or not path.exists():
        path = folder / f"step_{step:02d}.png"
    image = cv2.imread(str(path)) if path.exists() else None
    if image is None:
        return None
    if image.shape[1] > width:
        image = cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])), interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return data.tobytes() if ok else None


# ---- the instructions kept on the page --------------------------------------------------------------------------------
def presets_path() -> pathlib.Path:
    return config.PROJECT_ROOT / "calib" / "run_presets.json"


def load_presets() -> list[str]:
    try:
        saved = json.loads(presets_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return list(DEFAULT_PRESETS)
    return [str(text) for text in saved] if isinstance(saved, list) else list(DEFAULT_PRESETS)


def save_presets(presets: Any) -> list[str]:
    if not isinstance(presets, list) or not all(isinstance(text, str) for text in presets):
        raise ValueError("常用指令应是一组文字")
    cleaned = list(dict.fromkeys(" ".join(text.split()) for text in presets if text.strip()))
    if len(cleaned) > MAX_PRESETS or any(len(text) > MAX_INSTRUCTION for text in cleaned):
        raise ValueError(f"常用指令最多 {MAX_PRESETS} 条、每条最多 {MAX_INSTRUCTION} 字")
    presets_path().parent.mkdir(parents=True, exist_ok=True)
    presets_path().write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    return cleaned


# ---- HTTP ----------------------------------------------------------------------------------------------------------
def register(app: flask.Flask, console: Any, token: str = "", session: RunSession | None = None) -> RunSession:
    """Add the run page's routes to the console's app; returns the session they work on."""
    run = session or RunSession(console)

    def guarded(view: Callable[..., Any]) -> Callable[..., Any]:
        def inner(**kwargs: Any) -> Any:
            given = flask.request.args.get("token", "") or flask.request.headers.get("X-Token", "")
            if token and given != token:
                return flask.jsonify({"error": "需要口令（网址后加 ?token=…）"}), 401
            try:
                return view(**kwargs)
            except BusyError as exc:
                return flask.jsonify({"error": str(exc)}), 409
            except (SetupError, ValueError, TypeError) as exc:
                return flask.jsonify({"error": str(exc)}), 400
            except voice.VoiceError as exc:
                return flask.jsonify({"error": str(exc)}), 502

        inner.__name__ = view.__name__
        return inner

    def body() -> dict[str, Any]:
        if not flask.request.is_json:  # a form on some other web page cannot send JSON without asking first
            raise ValueError("请求必须是 JSON")
        data = flask.request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ValueError("请求内容不是 JSON 对象")
        return data

    @app.get("/run")
    @guarded
    def run_page() -> Any:
        return flask.send_from_directory(STATIC_DIR, "run.html")

    @app.get("/api/run/state")
    @guarded
    def run_state() -> Any:
        args = flask.request.args
        return flask.jsonify(run.state(args.get("id", ""), max(int(args.get("since", 0)), 0)))

    @app.post("/api/run/start")
    @guarded
    def run_start() -> Any:
        data = body()
        note = data.get("add_info")
        started = run.start(data.get("instruction", ""), data.get("max_steps", 15), data.get("settle_s", 0.8),
                            None if note is None else str(note))  # fmt: skip
        return flask.jsonify(started.to_json()), 202

    @app.post("/api/run/transcribe")
    @guarded
    def run_transcribe() -> Any:
        if flask.request.mimetype not in ("audio/wav", "audio/x-wav", "audio/wave"):
            raise ValueError("请求内容应是 WAV 音频")
        if (flask.request.content_length or 0) > MAX_WAV_BYTES:
            raise ValueError("录音太长")
        return flask.jsonify({"text": run.transcribe(flask.request.get_data(cache=False))})

    @app.post("/api/run/mic")
    @guarded
    def run_mic() -> Any:
        data = body()
        if data.get("action") == "start":
            run.mic_start(data.get("device") or None)
            return flask.jsonify({"ok": True})
        if data.get("action") == "stop":
            return flask.jsonify({"text": run.mic_stop()})
        raise ValueError("action 应是 start 或 stop")

    @app.get("/api/run/history")
    @guarded
    def run_history() -> Any:
        return flask.jsonify({"runs": run.history()})

    @app.route("/api/run/presets", methods=["GET", "POST"])
    @guarded
    def run_presets() -> Any:
        presets = save_presets(body().get("presets")) if flask.request.method == "POST" else load_presets()
        return flask.jsonify({"presets": presets})

    @app.route("/api/run/keyboard", methods=["GET", "POST"])
    @guarded
    def run_keyboard() -> Any:
        if flask.request.method == "POST":
            if run.console.busy is not None:
                raise BusyError("正在执行别的任务，稍后再改键盘标定")
            run.set_keyboard(body().get("anchors"))
        return flask.jsonify({"anchors": run.console.setup.keyboard, "order": list(ANCHORS)})

    @app.get("/api/run/<run_id>")
    @guarded
    def run_one(run_id: str) -> Any:
        found = run.load(run_id)
        return flask.jsonify(found) if found is not None else (flask.jsonify({"error": "没有这次运行"}), 404)

    @app.get("/api/run/<run_id>/step_<int:step>.jpg")
    @guarded
    def run_step_image(run_id: str, step: int) -> Any:
        data = step_image(run_id, step, annotated=flask.request.args.get("anno") == "1")
        if data is None:
            return flask.jsonify({"error": "没有这张图"}), 404
        return flask.Response(data, mimetype="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    return run
