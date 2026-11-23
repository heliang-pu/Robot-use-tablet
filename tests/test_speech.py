"""Tests of tablet_tap.speech.voice without a microphone or the model: a fake `arecord` on PATH and a stub HTTP service."""

from __future__ import annotations

import http.server
import io
import json
import os
from pathlib import Path
import socket
import stat
import struct
import sys
import threading
import time
import urllib.parse
import wave

import pytest

from tablet_tap.speech import voice

ARECORD_L_EN = """\
**** List of CAPTURE Hardware Devices ****
card 0: PCH [HDA Intel PCH], device 0: ALC897 Analog [ALC897 Analog]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 0: PCH [HDA Intel PCH], device 2: ALC897 Alt Analog [ALC897 Alt Analog]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 2: Device [USB PnP Sound Device], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
"""

ARECORD_L_ZH = """\
**** CAPTURE 硬體裝置清單 ****
card 0: PCH [HDA Intel PCH], device 0: ALC897 Analog [ALC897 Analog]
  子设备: 1/1
  子设备 #0: subdevice #0
"""

ARECORD_L_NONE = "**** List of CAPTURE Hardware Devices ****\n"

# A stand-in for arecord. `-l` prints list.txt; a capture follows mode.txt:
#   stream: a WAV header with arecord's placeholder sizes, then 16 kHz mono samples in real time until killed
#   fail:   "audio open error" on stderr, exit 1
#   mute:   the header only, then nothing
#   zeros:  like stream, but every sample is 0 (the monitor of an idle output, a muted input)
# The arguments of the last call are written to args.txt.
FAKE_ARECORD = """\
#!{python}
import pathlib, struct, sys, time
here = pathlib.Path(__file__).resolve().parent
args = sys.argv[1:]
(here / "args.txt").write_text(" ".join(args))
if "-l" in args:
    sys.stdout.write((here / "list.txt").read_text())
    sys.exit(0)
mode = (here / "mode.txt").read_text().strip()
if mode == "fail":
    sys.stderr.write("arecord: main:831: audio open error: No such file or directory\\n")
    sys.exit(1)
out = sys.stdout.buffer
out.write(b"RIFF" + struct.pack("<I", 0x80000024) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16))
out.write(b"data" + struct.pack("<I", 0x80000000))
out.flush()
if mode == "mute":
    time.sleep(30)
    sys.exit(0)
n = 0
while True:
    if mode == "zeros":
        out.write(bytes(640))
    else:
        out.write(b"".join(struct.pack("<h", (n + i) % 1000) for i in range(320)) + (b"\\x7f" if n == 0 else b""))
    out.flush()
    n += 320
    time.sleep(0.02)
"""


@pytest.fixture
def fake_arecord(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A directory with a fake `arecord`, put first on PATH; returns the directory (list.txt, mode.txt, args.txt)."""
    exe = tmp_path / "arecord"
    exe.write_text(FAKE_ARECORD.format(python=sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    (tmp_path / "list.txt").write_text(ARECORD_L_EN)
    (tmp_path / "mode.txt").write_text("stream")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    return tmp_path


class _Stub(http.server.BaseHTTPRequestHandler):
    """A stand-in for asr_service; behaviour comes from the server's `mode`, requests are kept in `seen`."""

    def log_message(self, *args: object) -> None:
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        mode = self.server.mode  # type: ignore[attr-defined]
        if self.path == "/health" and mode != "not_asr":
            self._send(200, json.dumps({"ok": True, "model": "stub", "device": "cpu"}).encode())
        else:
            self._send(404, b"<html>not found</html>", "text/html")

    def do_POST(self) -> None:  # noqa: N802
        server = self.server
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        url = urllib.parse.urlsplit(self.path)
        server.seen.append({"path": url.path, "query": urllib.parse.parse_qs(url.query), "body": body})  # type: ignore[attr-defined]
        mode = server.mode  # type: ignore[attr-defined]
        if mode == "ok":
            answer = {
                "text": "打开音乐播放音乐。",
                "raw": "<|zh|>打开音乐播放音乐。",
                "seconds_audio": 1.5,
                "seconds_infer": 0.1,
            }
            self._send(200, json.dumps(answer).encode())
        elif mode == "error":
            self._send(400, json.dumps({"error": "音频解码失败"}, ensure_ascii=False).encode())
        elif mode == "html":
            self._send(200, b"<html>hello</html>", "text/html")
        elif mode == "no_text":
            self._send(200, json.dumps({"result": "x"}).encode())
        elif mode == "slow":
            time.sleep(1.0)
            self._send(200, json.dumps({"text": "late"}).encode())


@pytest.fixture
def stub():
    """A stub ASR service on a free local port; yields the server (`.url`, `.mode`, `.seen`)."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    server.mode = "ok"  # type: ignore[attr-defined]
    server.seen = []  # type: ignore[attr-defined]
    server.url = f"http://127.0.0.1:{server.server_address[1]}"  # type: ignore[attr-defined]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield server
    server.shutdown()
    server.server_close()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _read_wav(data: bytes) -> tuple[int, int, int, bytes]:
    """(rate, channels, sample width, samples) as the stdlib reads the file; it rejects an invalid one."""
    with wave.open(io.BytesIO(data)) as w:
        return w.getframerate(), w.getnchannels(), w.getsampwidth(), w.readframes(w.getnframes())


# ---------------------------------------------------------------- WAV header


def test_wav_header_is_read_by_the_stdlib() -> None:
    pcm = struct.pack("<4h", 1, -2, 3, -4)
    assert _read_wav(voice.wav_header(len(pcm)) + pcm) == (16000, 1, 2, pcm)


def test_fix_wav_header_replaces_the_placeholder_sizes() -> None:
    pcm = struct.pack("<5h", 10, 20, 30, 40, 50)
    streamed = bytearray(voice.wav_header(0) + pcm)
    streamed[4:8] = struct.pack("<I", 0x80000024)  # what arecord writes to a pipe
    streamed[40:44] = struct.pack("<I", 0x80000000)
    fixed = voice.fix_wav_header(bytes(streamed))
    assert struct.unpack("<I", fixed[4:8])[0] == len(fixed) - 8
    assert struct.unpack("<I", fixed[40:44])[0] == len(pcm)
    assert _read_wav(fixed) == (16000, 1, 2, pcm)


def test_fix_wav_header_drops_half_a_frame_and_honours_the_limit() -> None:
    pcm = struct.pack("<5h", 10, 20, 30, 40, 50)
    cut = voice.wav_header(1 << 20) + pcm + b"\x01"  # killed in the middle of a sample
    assert _read_wav(voice.fix_wav_header(cut))[3] == pcm
    assert _read_wav(voice.fix_wav_header(cut, max_data_bytes=6))[3] == pcm[:6]
    assert _read_wav(voice.fix_wav_header(cut, max_data_bytes=5))[3] == pcm[:4]


def test_fix_wav_header_keeps_chunks_before_data() -> None:
    pcm = struct.pack("<3h", 7, 8, 9)
    head = voice.wav_header(0)
    extra = b"LIST" + struct.pack("<I", 5) + b"abcde" + b"\x00"  # odd size, padded
    streamed = head[:36] + extra + b"data" + struct.pack("<I", 0xFFFFFFFF) + pcm
    fixed = voice.fix_wav_header(streamed)
    assert extra in fixed
    assert struct.unpack("<I", fixed[4:8])[0] == len(fixed) - 8
    assert _read_wav(fixed) == (16000, 1, 2, pcm)


def test_fix_wav_header_of_an_empty_recording_is_a_valid_empty_wav() -> None:
    assert _read_wav(voice.fix_wav_header(voice.wav_header(0x80000000)))[3] == b""


@pytest.mark.parametrize("junk", [b"", b"RIFF", b"not a wav file at all", voice.wav_header(0)[:36]])
def test_fix_wav_header_rejects_what_is_not_a_wav(junk: bytes) -> None:
    with pytest.raises(voice.VoiceError):
        voice.fix_wav_header(junk)


def test_wav_seconds() -> None:
    assert voice.wav_seconds(voice.wav_header(32000) + bytes(32000)) == pytest.approx(1.0)


# ---------------------------------------------------------------- arecord -l


def test_parse_arecord_list() -> None:
    devices = voice.parse_arecord_list(ARECORD_L_EN)
    assert [d.alsa_name for d in devices] == [
        "plughw:CARD=PCH,DEV=0",
        "plughw:CARD=PCH,DEV=2",
        "plughw:CARD=Device,DEV=0",
    ]
    assert (devices[2].card, devices[2].card_name, devices[2].device_name) == (2, "USB PnP Sound Device", "USB Audio")
    assert "USB PnP Sound Device" in devices[2].label


def test_parse_arecord_list_translated_and_empty() -> None:
    assert [d.alsa_name for d in voice.parse_arecord_list(ARECORD_L_ZH)] == ["plughw:CARD=PCH,DEV=0"]
    assert voice.parse_arecord_list(ARECORD_L_NONE) == []
    assert voice.parse_arecord_list("") == []


def test_list_input_devices_runs_arecord(fake_arecord: Path) -> None:
    assert voice.list_input_devices() == ["plughw:CARD=PCH,DEV=0", "plughw:CARD=PCH,DEV=2", "plughw:CARD=Device,DEV=0"]
    assert (fake_arecord / "args.txt").read_text() == "-l"


def test_arecord_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))  # an empty directory: no arecord anywhere
    with pytest.raises(voice.VoiceError, match="找不到 arecord"):
        voice.list_input_devices()
    with pytest.raises(voice.VoiceError, match="找不到 arecord"):
        voice.record_wav(seconds=0.1)


# ---------------------------------------------------------------- record_wav


def test_record_wav_fixed_length(fake_arecord: Path) -> None:
    wav = voice.record_wav(seconds=0.2, device="plughw:CARD=PCH,DEV=0")
    rate, channels, width, pcm = _read_wav(wav)
    assert (rate, channels, width) == (16000, 1, 2)
    assert len(pcm) == 3200 * 2  # exactly 0.2 s, although the fake recorder never stops by itself
    assert struct.unpack("<I", wav[4:8])[0] == len(wav) - 8
    args = (fake_arecord / "args.txt").read_text()
    assert "-f S16_LE -r 16000 -c 1" in args and "-D plughw:CARD=PCH,DEV=0" in args and args.endswith(" -")


def test_record_wav_default_device_passes_no_D(fake_arecord: Path) -> None:
    voice.record_wav(seconds=0.05)
    assert "-D" not in (fake_arecord / "args.txt").read_text().split()


def test_record_wav_until_stop(fake_arecord: Path) -> None:
    stop = threading.Event()
    threading.Timer(0.4, stop.set).start()
    t0 = time.monotonic()
    wav = voice.record_wav(stop=stop)
    assert time.monotonic() - t0 < 3.0
    rate, channels, width, pcm = _read_wav(wav)  # sizes fixed up, whole frames only
    assert (rate, channels, width) == (16000, 1, 2)
    assert len(pcm) % 2 == 0
    assert 0.1 < voice.wav_seconds(wav) < 1.0


def test_record_wav_stop_wins_over_seconds(fake_arecord: Path) -> None:
    stop = threading.Event()
    threading.Timer(0.3, stop.set).start()
    assert voice.wav_seconds(voice.record_wav(seconds=20.0, stop=stop)) < 2.0


def test_record_wav_needs_an_end() -> None:
    with pytest.raises(ValueError):
        voice.record_wav()
    with pytest.raises(ValueError):
        voice.record_wav(seconds=0)


def test_record_wav_no_capture_device(fake_arecord: Path) -> None:
    (fake_arecord / "mode.txt").write_text("fail")
    (fake_arecord / "list.txt").write_text(ARECORD_L_NONE)
    with pytest.raises(voice.VoiceError, match="没有检测到录音设备"):
        voice.record_wav(seconds=0.2)


def test_record_wav_device_fails_but_others_exist(fake_arecord: Path) -> None:
    (fake_arecord / "mode.txt").write_text("fail")
    with pytest.raises(voice.VoiceError) as e:
        voice.record_wav(seconds=0.2, device="plughw:CARD=Nope,DEV=0")
    assert "plughw:CARD=Nope,DEV=0" in str(e.value) and "audio open error" in str(e.value)
    assert "plughw:CARD=PCH,DEV=0" in str(e.value)  # the devices that do exist are offered


def test_record_wav_stopped_before_any_audio(fake_arecord: Path) -> None:
    (fake_arecord / "mode.txt").write_text("mute")
    stop = threading.Event()
    threading.Timer(0.3, stop.set).start()
    with pytest.raises(voice.VoiceError, match="录音太短"):
        voice.record_wav(stop=stop)


def test_record_wav_device_delivers_nothing(fake_arecord: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (fake_arecord / "mode.txt").write_text("mute")
    monkeypatch.setattr(voice, "_START_TIMEOUT_S", 0.3)
    with pytest.raises(voice.VoiceError, match="录音失败"):
        voice.record_wav(seconds=1.0)


# ---------------------------------------------------------------- transcribe


def test_transcribe_posts_the_wav(stub) -> None:
    wav = voice.wav_header(4) + b"\x01\x00\x02\x00"
    assert voice.transcribe(wav, url=stub.url) == "打开音乐播放音乐。"
    assert stub.seen == [{"path": "/asr", "query": {"lang": ["zh"]}, "body": wav}]
    assert voice.transcribe(wav, url=stub.url + "/", lang="auto") == "打开音乐播放音乐。"
    assert stub.seen[1]["path"] == "/asr" and stub.seen[1]["query"] == {"lang": ["auto"]}
    assert voice.transcribe_detail(wav, url=stub.url)["seconds_infer"] == 0.1


def test_transcribe_bypasses_the_proxy(stub, monkeypatch: pytest.MonkeyPatch) -> None:
    dead = f"http://127.0.0.1:{_free_port()}"
    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.setenv(name, dead)
    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:1/")
    monkeypatch.setenv("all_proxy", "socks://127.0.0.1:1/")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    assert voice.transcribe(voice.wav_header(0), url=stub.url) == "打开音乐播放音乐。"
    assert voice.check_service(stub.url)["ok"] is True


def test_transcribe_service_down() -> None:
    url = f"http://127.0.0.1:{_free_port()}"
    with pytest.raises(voice.VoiceError, match="没有响应.*serve_asr.sh"):
        voice.transcribe(voice.wav_header(0), url=url)
    with pytest.raises(voice.VoiceError, match="没有响应"):
        voice.check_service(url)


def test_transcribe_service_error_message_is_passed_on(stub) -> None:
    stub.mode = "error"
    with pytest.raises(voice.VoiceError, match="HTTP 400.*音频解码失败"):
        voice.transcribe(voice.wav_header(0), url=stub.url)


@pytest.mark.parametrize("mode", ["html", "no_text"])
def test_transcribe_rejects_an_answer_without_text(stub, mode: str) -> None:
    stub.mode = mode
    with pytest.raises(voice.VoiceError, match="没有 text 字段"):
        voice.transcribe(voice.wav_header(0), url=stub.url)


def test_transcribe_timeout(stub) -> None:
    stub.mode = "slow"
    with pytest.raises(voice.VoiceError, match="超时"):
        voice.transcribe(voice.wav_header(0), url=stub.url, timeout_s=0.2)


def test_transcribe_empty_audio() -> None:
    with pytest.raises(voice.VoiceError, match="0 字节"):
        voice.transcribe(b"")


def test_check_service_on_a_port_that_is_something_else(stub) -> None:
    stub.mode = "not_asr"
    with pytest.raises(voice.VoiceError, match="不是就绪的语音识别服务"):
        voice.check_service(stub.url)


# ---------------------------------------------------------------- command line and listen_once


def test_main_list(fake_arecord: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert voice.main(["--list"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 3 and out[0].startswith("plughw:CARD=PCH,DEV=0") and "ALC897 Analog" in out[0]
    (fake_arecord / "list.txt").write_text(ARECORD_L_NONE)
    assert voice.main(["--list"]) == 1
    assert "请插上麦克风" in capsys.readouterr().err


def test_digital_silence() -> None:
    assert voice.is_digital_silence(voice.wav_header(8) + bytes(8))
    assert not voice.is_digital_silence(voice.wav_header(8) + bytes(6) + b"\x01\x00")
    assert not voice.is_digital_silence(voice.wav_header(0))  # nothing recorded is another error
    voice.require_signal(voice.wav_header(2) + b"\x05\x00")
    with pytest.raises(voice.VoiceError, match="全是 0.*plughw:CARD=PCH,DEV=2"):
        voice.require_signal(voice.wav_header(8) + bytes(8), "plughw:CARD=PCH,DEV=2")


def test_main_refuses_an_input_that_is_not_a_microphone(
    fake_arecord: Path, stub, capsys: pytest.CaptureFixture[str]
) -> None:
    (fake_arecord / "mode.txt").write_text("zeros")
    assert _read_wav(voice.record_wav(seconds=0.1))[3] == bytes(3200)  # the recording itself is a valid WAV
    assert voice.main(["--seconds", "0.1", "--url", stub.url]) == 1
    assert "全是 0" in capsys.readouterr().err
    assert stub.seen == []  # nothing was sent to the service


def test_main_seconds(fake_arecord: Path, stub, capsys: pytest.CaptureFixture[str]) -> None:
    assert voice.main(["--seconds", "0.1", "--url", stub.url, "--device", "plughw:CARD=PCH,DEV=2"]) == 0
    assert capsys.readouterr().out.strip() == "打开音乐播放音乐。"
    assert len(_read_wav(stub.seen[0]["body"])[3]) == 1600 * 2
    assert "-D plughw:CARD=PCH,DEV=2" in (fake_arecord / "args.txt").read_text()


def test_main_wav_file(tmp_path: Path, stub, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "a.wav"
    f.write_bytes(voice.wav_header(2) + b"\x00\x00")
    assert voice.main(["--wav", str(f), "--url", stub.url, "--lang", "auto"]) == 0
    assert capsys.readouterr().out.strip() == "打开音乐播放音乐。"
    assert stub.seen[0]["query"] == {"lang": ["auto"]}
    assert voice.main(["--wav", str(tmp_path / "missing.wav"), "--url", stub.url]) == 1


def test_main_service_down_is_reported_before_recording(fake_arecord: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert voice.main(["--seconds", "0.1", "--url", f"http://127.0.0.1:{_free_port()}"]) == 1
    assert "serve_asr.sh" in capsys.readouterr().err
    assert not (fake_arecord / "args.txt").exists()  # nothing was recorded


def test_listen_once(fake_arecord: Path, stub, monkeypatch: pytest.MonkeyPatch) -> None:
    presses = iter(["", ""])

    def fake_input(prompt: str = "") -> str:
        time.sleep(0.3)  # the operator takes a moment for each Enter
        return next(presses)

    monkeypatch.setattr("builtins.input", fake_input)
    assert voice.listen_once(url=stub.url) == "打开音乐播放音乐。"
    assert 0.05 < voice.wav_seconds(stub.seen[0]["body"]) < 2.0
