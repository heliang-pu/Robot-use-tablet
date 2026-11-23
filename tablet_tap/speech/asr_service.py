"""Local speech-recognition service of tablet_tap: SenseVoiceSmall (FunASR) on the CPU behind a small Flask app.

Runs in venvs/asr (start it with scripts/serve_asr.sh); the client is `tablet_tap/speech/voice.py`.

    POST /asr[?lang=zh|auto|en|yue|ja|ko]   body: WAV bytes, or multipart field `file`
        -> {"text": "打开音乐播放音乐。", "raw": "<|zh|><|NEUTRAL|>...", "seconds_audio": 2.3, "seconds_infer": 0.09}
    GET /health -> {"ok": true, "model": "SenseVoiceSmall", "device": "cpu", ...}   (503 while the model loads)

The model is given local directories only and FunASR's update check is off, so nothing is fetched at start-up.
"""

from __future__ import annotations

import os

# Before torch is imported: this service never touches the GPU (it is shared with the robot's policy), and the
# hub libraries FunASR pulls in must not look anything up on the network.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Small idle pools instead of one thread per core (28 here); torch's own count is set from --threads at load.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import argparse  # noqa: E402
import io  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

from flask import Flask  # noqa: E402
from flask import jsonify  # noqa: E402
from flask import request  # noqa: E402
import numpy as np  # noqa: E402

LOG = logging.getLogger("asr_service")

SAMPLE_RATE = 16000
LANGS = ("zh", "auto", "en", "yue", "ja", "ko")
MIN_SECONDS = 0.1  # shorter than this is a key bounce, not speech
MAX_SECONDS = 120.0
MAX_BODY_BYTES = 64 << 20
DEFAULT_THREADS = 4
_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = _ROOT / "weights" / "SenseVoiceSmall"
DEFAULT_VAD_DIR = _ROOT / "weights" / "fsmn-vad"

# Pictographs, dingbats, variation selector and joiner: what FunASR's rich post-processing adds for emotions/events.
_EMOJI = re.compile("[\U0001f000-\U0001faff\u2600-\u27bf\ufe0f\u200d]")


class AudioError(ValueError):
    """The request's audio cannot be used; the message goes to the client (Chinese)."""


def decode_audio(data: bytes) -> tuple[np.ndarray, float]:
    """Decode WAV bytes (any rate, any channel count) to 16 kHz mono float32 in [-1, 1]; also returns the seconds."""
    import soundfile

    if not data:
        raise AudioError("请求里没有音频数据")
    try:
        samples, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as e:  # noqa: BLE001 - libsndfile raises several types
        raise AudioError(f"音频解码失败（需要 WAV）：{e}") from e
    mono = samples.mean(axis=1) if samples.shape[1] > 1 else samples[:, 0]
    seconds = len(mono) / float(rate)
    if seconds > MAX_SECONDS:
        raise AudioError(f"音频太长：{seconds:.0f} 秒，最多 {MAX_SECONDS:.0f} 秒")
    if rate != SAMPLE_RATE and len(mono):
        import torch
        import torchaudio

        mono = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(mono)), rate, SAMPLE_RATE).numpy()
    return np.ascontiguousarray(mono, dtype=np.float32), seconds


def clean_text(raw: str) -> str:
    """SenseVoice output -> plain text: tags removed, no emoji, trimmed; punctuation kept."""
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    if not raw:
        return ""
    text = _EMOJI.sub("", rich_transcription_postprocess(raw))
    # A segment the model found nothing in comes back as a lone "。": dropped at the start, and when it is all there is.
    text = re.sub(r"\s+", " ", text).strip().lstrip("。，、？！.,?! ")
    return text if any(ch.isalnum() for ch in text) else ""


def _release_freed_memory() -> None:
    """Hand freed heap back to the system: loading leaves a second copy of the weights (about 1 GB) in the heap."""
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass  # not glibc


class Recognizer:
    """SenseVoiceSmall (optionally behind the FSMN VAD) loaded once; `transcribe` is serialised by a lock."""

    def __init__(self, model_dir: Path, vad_dir: Path | None, threads: int) -> None:
        from funasr import AutoModel
        import torch

        for d in (model_dir, vad_dir):
            if d is not None and not (d / "model.pt").is_file():
                raise FileNotFoundError(f"模型目录不完整，缺少 {d / 'model.pt'}")
        torch.set_num_threads(threads)
        self._torch = torch
        self.name = model_dir.name
        self.vad = vad_dir is not None
        self.threads = threads
        kwargs: dict = {}
        if vad_dir is not None:
            kwargs = {"vad_model": str(vad_dir), "vad_kwargs": {"max_single_segment_time": 30000}}
        self._model = AutoModel(
            model=str(model_dir),
            device="cpu",
            ncpu=threads,
            disable_update=True,
            disable_pbar=True,
            log_level="WARNING",
            **kwargs,
        )
        self._lock = threading.Lock()
        _release_freed_memory()

    def transcribe(self, audio: np.ndarray, lang: str = "zh") -> tuple[str, str, float]:
        """Recognise 16 kHz mono float32 audio; returns (text, raw model output, seconds of inference)."""
        extra = {"merge_vad": True, "merge_length_s": 15} if self.vad else {}
        with self._lock:
            # The front end dithers the samples with random noise; from a fixed seed the same audio always gives
            # the same text (unseeded, a borderline syllable came out differently in 3 of 40 runs of one sample).
            self._torch.manual_seed(0)
            t0 = time.perf_counter()
            res = self._model.generate(input=audio, cache={}, language=lang, use_itn=True, batch_size_s=60, **extra)
            dt = time.perf_counter() - t0
        raw = str(res[0].get("text", "")) if res else ""
        return clean_text(raw), raw, dt

    def warm_up(self, sample: Path | None = None) -> tuple[float, str]:
        """Two inferences, so the first real request does not pay for lazy initialisation; returns (seconds, text).

        `sample` is a speech file (the one shipped with the model): real speech passes the VAD, so the recogniser
        itself runs, and its text in the log shows the model works. Without it a synthetic buzz is used.
        """
        audio = None
        if sample is not None and sample.is_file():
            try:
                audio, _ = decode_audio(sample.read_bytes())
            except AudioError as e:
                LOG.warning("预热用的样例 %s 读不了（%s），改用合成声音", sample, e)
        if audio is None:
            t = np.arange(3 * SAMPLE_RATE) / SAMPLE_RATE
            buzz = 0.2 * np.sign(np.sin(2 * np.pi * 140 * t)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
            audio = (buzz + 0.02 * np.random.default_rng(0).standard_normal(len(t))).astype(np.float32)
        t0 = time.perf_counter()
        text = ""
        for _ in range(2):
            text = self.transcribe(audio)[0]
        return time.perf_counter() - t0, text


def create_app(get_recognizer) -> Flask:
    """The Flask app; `get_recognizer()` returns the loaded Recognizer, or None while it is still loading."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES
    app.json.ensure_ascii = False

    def fail(code: int, message: str):
        return jsonify({"error": message}), code

    @app.get("/health")
    def health():
        rec = get_recognizer()
        if rec is None:
            return jsonify({"ok": False, "loading": True, "error": "模型加载中"}), 503
        return jsonify({"ok": True, "model": rec.name, "device": "cpu", "vad": rec.vad, "threads": rec.threads})

    @app.post("/asr")
    def asr():
        rec = get_recognizer()
        if rec is None:
            return jsonify({"loading": True, "error": "模型加载中，请稍后再试"}), 503
        lang = request.args.get("lang", "zh")
        if lang not in LANGS:
            return fail(400, f"不支持的 lang={lang!r}，可用：{' / '.join(LANGS)}")
        if request.mimetype == "multipart/form-data":
            upload = request.files.get("file")
            if upload is None:
                return fail(400, "multipart 请求里没有 file 字段")
            data = upload.read()
        else:  # the body is the audio, whatever Content-Type says (curl --data-binary sends a form type)
            data = request.get_data()
        try:
            audio, seconds = decode_audio(data)
        except AudioError as e:
            return fail(400, str(e))
        if seconds < MIN_SECONDS:
            text, raw, dt = "", "", 0.0
        else:
            try:
                text, raw, dt = rec.transcribe(audio, lang)
            except Exception as e:  # noqa: BLE001 - a model failure must not take the service down
                LOG.exception("识别出错")
                return fail(500, f"识别出错：{type(e).__name__}: {e}")
        LOG.info("识别 %.2f 秒音频，用时 %.0f 毫秒：%s", seconds, dt * 1000, text or "（没有听到说话）")
        return jsonify({"text": text, "raw": raw, "seconds_audio": round(seconds, 3), "seconds_infer": round(dt, 3)})

    @app.errorhandler(413)
    def too_large(_e):
        return fail(413, f"请求太大，最多 {MAX_BODY_BYTES >> 20} MB")

    return app


def main(argv: list[str] | None = None) -> int:
    """Serve on 127.0.0.1; the port answers at once (503) and turns ready when the model is loaded and warm."""
    p = argparse.ArgumentParser(
        prog="tablet_tap.speech.asr_service", description="本机语音识别服务（SenseVoiceSmall，只用 CPU）"
    )
    p.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1，只供本机使用）")
    p.add_argument("--port", type=int, default=8803, help="端口（默认 8803）")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="SenseVoiceSmall 模型目录")
    p.add_argument("--vad-dir", type=Path, default=DEFAULT_VAD_DIR, help="fsmn-vad 模型目录")
    p.add_argument("--no-vad", action="store_true", help="不用 VAD（静音和噪声会被识别成“我。”之类的字，仅用于对比）")
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS, help=f"CPU 线程数（默认 {DEFAULT_THREADS}）")
    args = p.parse_args(argv)

    # FunASR logs through the root logger (a line per model file, per utterance); only its warnings are shown.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    LOG.setLevel(logging.INFO)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)  # no line per request; /asr logs its own
    for handler in logging.getLogger().handlers:
        handler.addFilter(lambda record: "trust_remote_code" not in record.getMessage())  # FunASR's start-up noise
    from werkzeug.serving import make_server

    state: dict[str, Recognizer] = {}
    app = create_app(lambda: state.get("rec"))
    try:  # werkzeug answers a busy port with its own English text and SystemExit; ask first
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((args.host, args.port))
    except OSError as e:
        LOG.error("端口 %s:%d 用不了：%s（是不是已经有一个语音识别服务在运行？）", args.host, args.port, e)
        return 1
    server = make_server(args.host, args.port, app, threaded=True)

    def load() -> None:
        t0 = time.perf_counter()
        try:
            LOG.info("加载模型 %s（CPU，%d 线程）……", args.model_dir, args.threads)
            rec = Recognizer(args.model_dir, None if args.no_vad else args.vad_dir, args.threads)
            t1 = time.perf_counter()
            warm, heard = rec.warm_up(args.model_dir / "example" / "zh.mp3")
        except Exception:  # noqa: BLE001 - nothing to serve without the model
            LOG.exception("模型加载失败，服务退出")
            os._exit(1)
        state["rec"] = rec
        LOG.info(
            "服务就绪 http://%s:%d （加载 %.1f 秒，预热 %.1f 秒，VAD %s；预热听到：%s）",
            args.host,
            args.port,
            t1 - t0,
            warm,
            "开" if rec.vad else "关",
            heard or "（无）",
        )

    threading.Thread(target=load, daemon=True, name="load-model").start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("收到 Ctrl-C，服务停止")
    return 0


if __name__ == "__main__":
    sys.exit(main())
