"""Speech input for tablet_tap: record from the PC microphone with `arecord` and ask the local ASR service for text.

Runs under bin/py (stdlib + requests only); the model itself lives in `tablet_tap/speech/asr_service.py` (venvs/asr).

    bin/py -m tablet_tap.speech.voice            # press Enter, speak, press Enter -> prints the text
    bin/py -m tablet_tap.speech.voice --seconds 4
    bin/py -m tablet_tap.speech.voice --list
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import struct
import subprocess
import sys
import threading
import time

import requests

DEFAULT_URL = "http://127.0.0.1:8803"
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # S16_LE
CHANNELS = 1
MAX_RECORD_S = 120.0  # an open-ended recording (stop event) is cut here
_START_TIMEOUT_S = 5.0  # arecord must deliver its first audio within this time
_FRAME_BYTES = SAMPLE_WIDTH * CHANNELS

_CARD_LINE = re.compile(r"^card (\d+): (\S+) \[(.*?)\], device (\d+): (.*?) \[(.*?)\]\s*$")


class VoiceError(RuntimeError):
    """A recording or recognition failure with a message for the operator (Chinese)."""


@dataclasses.dataclass(frozen=True)
class CaptureDevice:
    """One capture device of `arecord -l`."""

    card: int
    card_id: str
    card_name: str
    device: int
    device_name: str

    @property
    def alsa_name(self) -> str:
        """The name for `arecord -D` (card id rather than index: indices move between boots)."""
        return f"plughw:CARD={self.card_id},DEV={self.device}"

    @property
    def label(self) -> str:
        """A line for the operator."""
        return f"{self.alsa_name}  ({self.card_name} / {self.device_name})"


def parse_arecord_list(text: str) -> list[CaptureDevice]:
    """Parse the output of `LC_ALL=C arecord -l` into its capture devices."""
    devices = []
    for line in text.splitlines():
        m = _CARD_LINE.match(line.strip())
        if m:
            devices.append(CaptureDevice(int(m[1]), m[2], m[3], int(m[4]), m[5]))
    return devices


def _capture_devices() -> list[CaptureDevice]:
    """Run `arecord -l` (C locale, so the lines are not translated) and parse it."""
    try:
        out = subprocess.run(
            ["arecord", "-l"],
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "LC_ALL": "C"},
            check=False,
        )
    except FileNotFoundError as e:
        raise VoiceError("找不到 arecord：请先安装 alsa-utils（sudo apt install alsa-utils）") from e
    except subprocess.TimeoutExpired as e:
        raise VoiceError("arecord -l 超时没有返回，声卡驱动可能卡住了") from e
    return parse_arecord_list(out.stdout)


def list_input_devices() -> list[str]:
    """ALSA names of the capture devices (`plughw:CARD=<id>,DEV=<n>`), usable as `device` of `record_wav`."""
    return [d.alsa_name for d in _capture_devices()]


def wav_header(n_data_bytes: int, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS) -> bytes:
    """A canonical 44-byte PCM S16_LE WAV header for `n_data_bytes` of samples."""
    block = channels * SAMPLE_WIDTH
    return (
        b"RIFF"
        + struct.pack("<I", 36 + n_data_bytes)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, sample_rate * block, block, SAMPLE_WIDTH * 8)
        + b"data"
        + struct.pack("<I", n_data_bytes)
    )


def fix_wav_header(data: bytes, max_data_bytes: int | None = None) -> bytes:
    """Make a WAV that was streamed to a pipe valid: the RIFF and data sizes are set to what is really there.

    `arecord` cannot seek back in a pipe, so an open-ended recording carries a placeholder size, and a recording cut
    off by a signal may end in half a frame. Chunks before `data` are kept, the samples are trimmed to whole frames
    (and to `max_data_bytes` when given).
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise VoiceError("录音数据不是 WAV（没有 RIFF/WAVE 头），arecord 可能没有正常启动")
    pos = 12
    block = _FRAME_BYTES
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        (size,) = struct.unpack("<I", data[pos + 4 : pos + 8])
        if chunk_id == b"fmt " and pos + 8 + 16 <= len(data):
            block = struct.unpack("<H", data[pos + 20 : pos + 22])[0] or block
        if chunk_id == b"data":
            body = data[pos + 8 :]
            if max_data_bytes is not None:
                body = body[: max(0, max_data_bytes)]
            body = body[: len(body) - len(body) % block]
            head = bytearray(data[: pos + 8])
            head[4:8] = struct.pack("<I", len(head) - 8 + len(body))
            head[pos + 4 : pos + 8] = struct.pack("<I", len(body))
            return bytes(head) + body
        pos += 8 + size + (size & 1)
    raise VoiceError("录音数据里没有 data 块，arecord 可能还没录到声音就退出了")


def _data_offset(data: bytes) -> int | None:
    """Offset of the first sample in a (possibly still growing) WAV stream, or None while the header is incomplete."""
    pos = 12
    while pos + 8 <= len(data):
        if data[pos : pos + 4] == b"data":
            return pos + 8
        (size,) = struct.unpack("<I", data[pos + 4 : pos + 8])
        pos += 8 + size + (size & 1)
    return None


def _pump(stream, sink: bytearray) -> None:
    """Copy a pipe into `sink` until it closes."""
    while True:
        chunk = stream.read1(4096)
        if not chunk:
            return
        sink.extend(chunk)


def _arecord_failure(stderr: str, device: str | None) -> VoiceError:
    """The error for an arecord that produced no audio."""
    try:
        devices = _capture_devices()
    except VoiceError:
        devices = []
    if not devices:
        return VoiceError("没有检测到录音设备（arecord -l 为空）：请插上麦克风后重试")
    known = "；".join(d.label for d in devices)
    detail = stderr.strip().splitlines()[-1] if stderr.strip() else "没有输出"
    which = device or "系统默认设备"
    return VoiceError(f"用 {which} 录音失败：{detail}。可用的录音设备：{known}（用 --device 指定）")


def record_wav(
    seconds: float | None = None,
    device: str | None = None,
    stop: threading.Event | None = None,
) -> bytes:
    """Record 16 kHz mono S16_LE audio with `arecord` and return it as a valid WAV.

    `seconds` gives the length; with `seconds=None` the recording runs until `stop` is set (at most MAX_RECORD_S).
    With both, whichever comes first ends it. `device` is an ALSA name (`list_input_devices()`); None is the system
    default input (the one chosen in the desktop sound settings).
    """
    if seconds is None and stop is None:
        raise ValueError("record_wav needs seconds or a stop event")
    if seconds is not None and seconds <= 0:
        raise ValueError("seconds must be positive")
    limit_s = min(seconds, MAX_RECORD_S) if seconds is not None else MAX_RECORD_S
    want_bytes = int(round(limit_s * SAMPLE_RATE)) * _FRAME_BYTES

    cmd = ["arecord", "-q", "-t", "wav", "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", str(CHANNELS)]
    if device:
        cmd += ["-D", device]
    cmd.append("-")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
    except FileNotFoundError as e:
        raise VoiceError("找不到 arecord：请先安装 alsa-utils（sudo apt install alsa-utils）") from e

    out, err = bytearray(), bytearray()
    pumps = [
        threading.Thread(target=_pump, args=(proc.stdout, out), daemon=True),
        threading.Thread(target=_pump, args=(proc.stderr, err), daemon=True),
    ]
    for t in pumps:
        t.start()
    started = time.monotonic()
    stopped_early = False
    try:
        while proc.poll() is None:
            offset = _data_offset(bytes(out[:4096]))
            got = len(out) - offset if offset is not None else 0
            if got >= want_bytes:
                break
            if stop is not None and stop.is_set():
                stopped_early = True
                break
            if got == 0 and time.monotonic() - started > _START_TIMEOUT_S:
                break  # the device opened but delivers nothing
            time.sleep(0.01)
    finally:
        if proc.poll() is None:
            proc.terminate()  # arecord ends the capture and exits on SIGTERM
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for t in pumps:
            t.join(timeout=2.0)

    stderr = err.decode("utf-8", "replace")
    offset = _data_offset(bytes(out[:4096]))
    if offset is None or len(out) - offset < _FRAME_BYTES:
        if stopped_early and not stderr.strip():
            raise VoiceError("录音太短，还没有录到声音就结束了：请按回车后稍等一下再说话")
        raise _arecord_failure(stderr, device)
    return fix_wav_header(bytes(out), max_data_bytes=want_bytes)


def wav_seconds(wav: bytes) -> float:
    """Length of a 16 kHz mono S16_LE WAV made by `record_wav`."""
    offset = _data_offset(wav)
    return 0.0 if offset is None else (len(wav) - offset) / (SAMPLE_RATE * _FRAME_BYTES)


def is_digital_silence(wav: bytes) -> bool:
    """True when every sample is exactly 0: no microphone gives that, a muted input or an output's monitor does."""
    offset = _data_offset(wav)
    return offset is not None and len(wav) > offset and not wav[offset:].strip(b"\x00")


def require_signal(wav: bytes, device: str | None = None) -> None:
    """Raise VoiceError for a recording of exact zeros, which means the input is not a working microphone."""
    if is_digital_silence(wav):
        which = device or "系统默认输入"
        raise VoiceError(
            f"录到的全是 0：“{which}”不是麦克风，或者被静音了。请插上麦克风，在系统声音设置里把它选为输入设备，"
            "或用 --device 指定（--list 查看设备）"
        )


def _session() -> requests.Session:
    """A session that ignores HTTP(S)_PROXY / ALL_PROXY: the service is on this machine."""
    s = requests.Session()
    s.trust_env = False
    s.proxies = {"http": None, "https": None}
    return s


def _service_down(url: str) -> VoiceError:
    return VoiceError(f"语音识别服务没有响应（{url}）：请先在另一个终端运行 scripts/serve_asr.sh，等它打印“服务就绪”")


def check_service(url: str = DEFAULT_URL, timeout_s: float = 3.0) -> dict:
    """GET /health of the ASR service; raises VoiceError when it is down or not ready."""
    try:
        r = _session().get(f"{url.rstrip('/')}/health", timeout=timeout_s)
    except requests.exceptions.RequestException as e:
        raise _service_down(url) from e
    try:
        info = r.json()
    except ValueError:
        info = None
    if r.status_code != 200 or not isinstance(info, dict) or not info.get("ok"):
        raise VoiceError(f"{url} 上的服务不是就绪的语音识别服务（HTTP {r.status_code}）：端口可能被别的程序占用")
    return info


def transcribe_detail(wav: bytes, url: str = DEFAULT_URL, lang: str = "zh", timeout_s: float = 30.0) -> dict:
    """POST a WAV to the service's /asr and return its whole answer (text, raw, seconds_audio, seconds_infer)."""
    if not wav:
        raise VoiceError("没有录到音频（0 字节）")
    try:
        r = _session().post(
            f"{url.rstrip('/')}/asr",
            params={"lang": lang},
            data=wav,
            headers={"Content-Type": "audio/wav"},
            timeout=timeout_s,
        )
    except requests.exceptions.Timeout as e:
        raise VoiceError(f"语音识别超时（{timeout_s:.0f} 秒没有结果，{url}）") from e
    except requests.exceptions.RequestException as e:
        raise _service_down(url) from e
    try:
        body = r.json()
    except ValueError:
        body = None
    if r.status_code != 200:
        detail = body.get("error") if isinstance(body, dict) else r.text[:200]
        raise VoiceError(f"语音识别服务报错（HTTP {r.status_code}）：{detail}")
    if not isinstance(body, dict) or not isinstance(body.get("text"), str):
        raise VoiceError(f"语音识别服务的返回里没有 text 字段：{r.text[:200]}")
    return body


def transcribe(wav: bytes, url: str = DEFAULT_URL, lang: str = "zh", timeout_s: float = 30.0) -> str:
    """Recognise a WAV with the local ASR service and return the text (empty when nothing was said)."""
    return transcribe_detail(wav, url=url, lang=lang, timeout_s=timeout_s)["text"]


def listen_once(url: str = DEFAULT_URL, device: str | None = None, lang: str = "zh") -> str:
    """Interactive push-to-talk: Enter starts the recording, Enter ends it; returns the recognised text."""
    check_service(url)  # fail before the operator speaks, not after
    input("按回车开始说话，再按回车结束 > ")
    stop = threading.Event()
    result: dict[str, object] = {}

    def work() -> None:
        try:
            result["wav"] = record_wav(device=device, stop=stop)
        except Exception as e:  # noqa: BLE001 - handed to the caller's thread below
            result["error"] = e

    t = threading.Thread(target=work, daemon=True)
    t.start()
    print("录音中……说完按回车", flush=True)
    try:
        input()
    finally:
        stop.set()
        t.join(timeout=10.0)
    if "error" in result:
        raise result["error"]  # type: ignore[misc]
    wav = result.get("wav")
    if not isinstance(wav, bytes):
        raise VoiceError("录音线程没有按时结束")
    require_signal(wav, device)
    print(f"录到 {wav_seconds(wav):.1f} 秒，识别中……", flush=True)
    return transcribe(wav, url=url, lang=lang)


def main(argv: list[str] | None = None) -> int:
    """Command line: record (push-to-talk or a fixed length) and print the recognised text."""
    p = argparse.ArgumentParser(
        prog="tablet_tap.speech.voice", description="用电脑麦克风录一句话，交给本机语音识别服务，打印文字"
    )
    p.add_argument("--device", help="录音设备（--list 里的名字，如 plughw:CARD=PCH,DEV=0）；不给则用系统默认输入")
    p.add_argument("--seconds", type=float, help="固定录 N 秒；不给则按回车开始、再按回车结束")
    p.add_argument("--list", action="store_true", help="列出录音设备后退出")
    p.add_argument("--url", default=DEFAULT_URL, help=f"语音识别服务地址（默认 {DEFAULT_URL}）")
    p.add_argument("--lang", default="zh", help="语言：zh（默认）/ auto / en / yue / ja / ko")
    p.add_argument("--wav", help="不录音，直接识别这个 WAV 文件")
    args = p.parse_args(argv)
    try:
        if args.list:
            devices = _capture_devices()
            if not devices:
                print("没有检测到录音设备：请插上麦克风", file=sys.stderr)
                return 1
            for d in devices:
                print(d.label)
            return 0
        if args.wav:
            with open(args.wav, "rb") as f:
                text = transcribe(f.read(), url=args.url, lang=args.lang)
        elif args.seconds is not None:
            check_service(args.url)
            print(f"录音 {args.seconds:g} 秒，请说话……", file=sys.stderr, flush=True)
            wav = record_wav(seconds=args.seconds, device=args.device)
            require_signal(wav, args.device)
            text = transcribe(wav, url=args.url, lang=args.lang)
        else:
            text = listen_once(url=args.url, device=args.device, lang=args.lang)
    except VoiceError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"错误：读不了文件：{e}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("已取消", file=sys.stderr)
        return 130
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
