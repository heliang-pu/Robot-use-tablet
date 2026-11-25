#!/usr/bin/env python3
"""GUI-Owl-1.5 on the CPU behind the OpenAI interface the run page talks to: a stand-in for ``scripts/serve_vlm.sh``
when the GPU must be left alone (rehearsals, offline checks). It never touches the GPU: ``CUDA_VISIBLE_DEVICES`` is
emptied before torch is imported.

    venvs/vlm/bin/python scripts/serve_vlm_cpu.py                       # the 2B on 127.0.0.1:8822, served as gui-owl
    venvs/vlm/bin/python scripts/serve_vlm_cpu.py --port 8822 --threads 12 --model weights/GUI-Owl-1.5-2B-Instruct
    bin/py -m tablet_tap.agent.evaluate --base_url http://127.0.0.1:8822/v1      # the project's client against it

The model, the processor (the pixel budget of ``serve_vlm.sh``) and the generation are those of
``scripts/smoke_vlm_cpu.py`` (``CpuAgentModel``): chat template -> HF processor -> greedy decoding.

What differs from the vLLM server, and what a rehearsal with it therefore does not show:

- **Greedy decoding, always.** The project's client sends no sampling parameters, so a vLLM server samples with the
  model's ``generation_config.json`` (temperature 0.6, top_k 40, top_p 0.95) unless it was started with
  ``--override-generation-config '{"temperature": 0}'``. Here the same picture always gets the same answer.
- **Slow.** About 25 s for one 1280x800 screenshot on 12 threads, and every earlier step still in the prompt (the
  client keeps four) adds its picture. The client gives up on an HTTP request after 30 s and sends it again
  (the OpenAI library twice, then upstream's own retry loop), so a request that is already being answered, or was
  answered in the last ten minutes, is not started again: the repeat waits for, and gets, the first one's answer.
  With vLLM a step takes about a second and no request is ever repeated.
- **Only the endpoints and fields the client uses**: ``GET /v1/models``, ``POST /v1/chat/completions`` with
  ``model`` and ``messages`` (text and ``image_url`` parts) and optionally ``max_tokens``; no streaming, no tools,
  one generation at a time. The default model here is the 2B; the GPU server's default is the 8B.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # the GPU is shared with live robot runs: never use it from here

import argparse  # noqa: E402
import base64  # noqa: E402
import collections  # noqa: E402
import hashlib  # noqa: E402
import http.server  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402
import uuid  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KEEP_S = 600.0  # an answer is kept this long for a repeat of the same request
KEEP_ANSWERS = 64
MAX_BODY_BYTES = 256 * 1024 * 1024

logger = logging.getLogger("serve_vlm_cpu")


class BadRequest(ValueError):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def png_data_url(source: str) -> str:
    """The picture of an ``image_url`` as the PNG data URL the model code takes: a data URL of any image type, or a
    local file (``file://`` or a path, as with vLLM's --allowed-local-media-path /). Nothing is fetched over a network."""
    from PIL import Image  # noqa: PLC0415 - after the thread settings

    if source.startswith("data:image/png;base64,"):
        return source
    if source.startswith("data:"):
        head, _, body = source.partition(",")
        if not head.endswith(";base64") or not body:
            raise BadRequest("image_url: 只认 base64 的 data URL")
        raw = base64.b64decode(body)
    elif source.startswith(("http://", "https://")):
        raise BadRequest("image_url: 这个服务不从网络取图，给 data URL 或本机路径")
    else:
        path = Path(source.removeprefix("file://"))
        if not path.is_file():
            raise BadRequest(f"image_url: 没有这个文件 {path}")
        raw = path.read_bytes()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except OSError as exc:
        raise BadRequest(f"image_url: 打不开这张图（{exc}）") from None
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def tidy_messages(messages: Any) -> list[dict[str, Any]]:
    """The request's messages as ``smoke_vlm_cpu.to_hf_messages`` takes them: every content a list of text and
    image_url parts, every picture a PNG data URL."""
    if not isinstance(messages, list) or not messages:
        raise BadRequest("messages 应该是非空列表")
    tidy = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant"):
            raise BadRequest("每条 message 要有 role（system / user / assistant）")
        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list):
            raise BadRequest("message 的 content 应该是字符串或列表")
        parts: list[dict[str, Any]] = []
        for part in content:
            kind = part.get("type") if isinstance(part, dict) else None
            if kind == "text" and isinstance(part.get("text"), str):
                parts.append({"type": "text", "text": part["text"]})
            elif kind == "image_url":
                url = part.get("image_url")
                url = url.get("url") if isinstance(url, dict) else url
                if not isinstance(url, str):
                    raise BadRequest("image_url 里没有 url")
                parts.append({"type": "image_url", "image_url": {"url": png_data_url(url)}})
            else:
                raise BadRequest(f"不认识的 content 项：{kind!r}（只有 text 和 image_url）")
        tidy.append({"role": message["role"], "content": parts})
    return tidy


class Job:
    def __init__(self) -> None:
        self.done = threading.Event()
        self.result: dict[str, Any] | None = None
        self.error: BaseException | None = None
        self.finished = 0.0


class Service:
    """The model and the requests in flight. One generation at a time; a repeat of a request joins the first."""

    def __init__(self, model: Any, name: str, max_new_tokens: int) -> None:
        self.model, self.name, self.max_new_tokens = model, name, max_new_tokens
        self._generate = threading.Lock()
        self._lock = threading.Lock()
        self._jobs: collections.OrderedDict[str, Job] = collections.OrderedDict()
        self.created = int(time.time())

    def models(self) -> dict[str, Any]:
        card = {"id": self.name, "object": "model", "created": self.created, "owned_by": "tablet_tap",
                "root": str(self.model.model_dir), "parent": None}  # fmt: skip
        return {"object": "list", "data": [card]}

    def complete(self, body: dict[str, Any]) -> dict[str, Any]:
        if body.get("model") != self.name:
            raise BadRequest(f"The model `{body.get('model')}` does not exist（这里的模型叫 {self.name}）", 404)
        if body.get("stream"):
            raise BadRequest("这个服务不支持 stream")
        if int(body.get("n") or 1) != 1:
            raise BadRequest("这个服务一次只给一个回答（n=1）")
        limit = body.get("max_completion_tokens") or body.get("max_tokens") or self.max_new_tokens
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise BadRequest("max_tokens 应该是正整数")
        if body.get("temperature") not in (None, 0, 0.0):
            logger.info("请求要 temperature=%s：这里一律贪心解码", body["temperature"])
        messages = tidy_messages(body.get("messages"))
        key = hashlib.sha256(json.dumps([messages, limit], sort_keys=True).encode()).hexdigest()
        with self._lock:
            now = time.time()
            for old in [k for k, j in self._jobs.items() if j.done.is_set() and now - j.finished > KEEP_S]:
                del self._jobs[old]
            job = self._jobs.get(key)
            mine = job is None
            if job is None:
                job = self._jobs[key] = Job()
                while len(self._jobs) > KEEP_ANSWERS:
                    oldest = next(iter(self._jobs))
                    if not self._jobs[oldest].done.is_set():
                        break
                    del self._jobs[oldest]
        if mine:
            self._run(key, job, messages, limit)
        else:
            waited = time.time()
            logger.info("%s 和正在答或刚答过的请求相同，等它的回答", key[:8])
            job.done.wait()
            logger.info("%s 重复的请求拿到了回答（等了 %.1f s）", key[:8], time.time() - waited)
        if job.error is not None:
            raise job.error
        assert job.result is not None
        step = job.result
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": step["raw_output"]},
                    "logprobs": None,
                    "finish_reason": "length" if step["output_tokens"] >= limit else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": step["prompt_tokens"],
                "completion_tokens": step["output_tokens"],
                "total_tokens": step["prompt_tokens"] + step["output_tokens"],
            },
        }

    def _run(self, key: str, job: Job, messages: list[dict[str, Any]], limit: int) -> None:
        """Answer in this thread to the end, whether or not the client is still waiting: its repeat collects it."""
        queued = time.time()
        try:
            with self._generate:
                waited = time.time() - queued
                step = self.model.step(messages, limit)
            job.result = step
            logger.info(
                "%s %.1f s（排队 %.1f s）：提示 %d token（图 %d 张 %d token），回答 %d token：%s",
                key[:8], step["seconds"], waited, step["prompt_tokens"], len(step["image_sizes_sent"]),
                step["image_tokens"], step["output_tokens"], step["raw_output"].replace("\n", " ⏎ ")[:300],
            )  # fmt: skip
        except BaseException as exc:  # a failed generation must not leave its repeats waiting for ever
            job.error = exc if isinstance(exc, Exception) else RuntimeError(f"生成被打断：{type(exc).__name__}")
            logger.exception("%s 生成失败", key[:8])
            with self._lock:
                self._jobs.pop(key, None)  # a repeat starts again
            if not isinstance(exc, Exception):
                job.finished = time.time()
                job.done.set()
                raise
        job.finished = time.time()
        job.done.set()


def handler_for(service: Service) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "serve_vlm_cpu"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - the base class's name
            pass  # requests are logged by the service, with their timings

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                logger.info("客户端没等到回答就断开了（它会重发，重发的请求直接拿这次的回答）")

        def _fail(self, status: int, message: str) -> None:
            kind = "NotFoundError" if status == 404 else "BadRequestError" if status < 500 else "InternalServerError"
            self._send(status, {"error": {"message": message, "type": kind, "param": None, "code": status}})

        def do_GET(self) -> None:  # noqa: N802 - the base class's name
            path = self.path.split("?", 1)[0].rstrip("/")
            if path == "/v1/models":
                self._send(200, service.models())
            elif path in ("/health", "/v1/health"):
                self._send(200, {"status": "ok"})
            else:
                self._fail(404, f"没有 {self.path}")

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0].rstrip("/") != "/v1/chat/completions":
                self._fail(404, f"没有 {self.path}")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if not 0 < length <= MAX_BODY_BYTES:
                    raise BadRequest("请求体是空的或太大")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise BadRequest("请求体应该是 JSON 对象")
                answer = service.complete(body)
            except BadRequest as exc:
                self._fail(exc.status, str(exc))
            except (ValueError, AssertionError) as exc:
                self._fail(400, f"{type(exc).__name__}: {exc}")
            except Exception as exc:  # noqa: BLE001 - the client must get an answer, the log has the stack
                self._fail(500, f"{type(exc).__name__}: {exc}")
            else:
                self._send(200, answer)

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="在 CPU 上起 GUI-Owl-1.5 的 OpenAI 兼容服务（不使用 GPU；排练用，慢）")
    parser.add_argument("--model", type=Path, default=Path("weights/GUI-Owl-1.5-2B-Instruct"), help="模型目录（CPU 上只建议 2B）")
    parser.add_argument("--port", type=int, default=8822, help="监听端口（只监听 127.0.0.1）；8802 留给 GPU 上的 vLLM")
    parser.add_argument("--name", default="gui-owl", help="客户端请求里的 model 字段")
    parser.add_argument("--threads", type=int, default=12, help="推理用的 CPU 线程数")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="请求没给 max_tokens 时每次最多生成多少 token（一步的回答约 50 个）")
    parser.add_argument("--pidfile", type=Path, default=None, help="模型加载好、端口开始监听后把进程号写到这里")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    if args.threads < 1 or args.max_new_tokens < 1:
        parser.error("--threads 和 --max-new-tokens 应该是正整数")

    os.environ["SMOKE_VLM_THREADS"] = str(args.threads)  # read by smoke_vlm_cpu when it is imported, before torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import smoke_vlm_cpu  # noqa: PLC0415 - sets the thread counts and hides the GPU, then the model code

    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    model_dir = args.model if args.model.is_absolute() else (args.model if args.model.is_dir() else ROOT / args.model)
    model_dir = model_dir.resolve()
    smoke_vlm_cpu.check_weights_complete(model_dir)
    logger.info("加载 %s（CPU，%d 线程）……", model_dir, args.threads)
    model = smoke_vlm_cpu.CpuAgentModel(model_dir, args.threads)
    logger.info("加载完成：%.1f s，%.2f B 参数，CUDA 可见=%s", model.load_seconds, model.n_params / 1e9, model.torch.cuda.is_available())

    service = Service(model, args.name, args.max_new_tokens)
    http.server.ThreadingHTTPServer.daemon_threads = True
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(service))

    def leave(signum: int, frame: Any) -> None:
        logger.info("收到信号 %s，退出", signal.Signals(signum).name)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, leave)
    if args.pidfile is not None:
        args.pidfile.parent.mkdir(parents=True, exist_ok=True)
        args.pidfile.write_text(f"{os.getpid()}\n")
    logger.info("就绪：http://127.0.0.1:%d/v1 ，模型名 %s（贪心解码，一次答一个请求）", args.port, args.name)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if args.pidfile is not None and args.pidfile.exists() and args.pidfile.read_text().strip() == str(os.getpid()):
            args.pidfile.unlink()
        logger.info("已退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
