#!/usr/bin/env python3
"""台架状态一览：CAN、相机、控制台、标定、模型和语音、显卡，最后一段说现在还缺什么。

    scripts/status.py            （或 scripts/console.sh status）

只看不碰：CAN 看内核的收包计数，控制台用它自己的状态接口；不发总线报文、不打开相机。查相机序列号（rs-enumerate-devices）
会碰到正在出图的相机，所以只在没有程序占用相机时才查。由 RoboSynChallenge 的 sim2real_4090.sh status 改来。
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
PORT = 8801
ARMS = {"can_l_follower": "左臂", "can_r_follower": "右臂"}
missing: list[str] = []


def run(*command: str, timeout: float = 5.0) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def get(url: str, timeout: float = 3.0):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # local services, never through a proxy
    try:
        with opener.open(url, timeout=timeout) as answer:
            return json.loads(answer.read() or b"null")
    except (OSError, ValueError):
        return None


def load(path: pathlib.Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def mark(ok: bool) -> str:
    return "✓" if ok else "✗"


def can(pen_arm: str) -> None:
    print("== CAN（从臂上电就一直发报文；主臂本来不发，不看）")

    def count(name: str) -> int | None:
        try:
            return int(pathlib.Path(f"/sys/class/net/{name}/statistics/rx_packets").read_text())
        except (OSError, ValueError):
            return None

    before, started = {name: count(name) for name in ARMS}, time.monotonic()
    time.sleep(1.0)
    after, took = {name: count(name) for name in ARMS}, time.monotonic() - started
    for name, label in ARMS.items():
        role = "（拿笔）" if name == pen_arm else ""
        if before[name] is None or after[name] is None:
            print(f"  {label}{role} {name}：没有这个接口 ✗ —— USB-CAN 适配器没插或没配置")
            alive = False
        else:
            rate = (after[name] - before[name]) / took
            alive = rate > 100
            print(f"  {label}{role} {name}：{rate:.0f} 帧/秒 {mark(alive)}"
                  + ("" if alive else " —— 臂没上电 / CAN 线没接 / 急停按下 / 适配器被拔过后总线挂死（给这条臂断电重启）"))  # fmt: skip
        if name == pen_arm and not alive:
            missing.append(f"{label}没有报文")


def cameras(wanted: dict[str, str]) -> None:
    print("== 相机")
    usb = []
    for device in pathlib.Path("/sys/bus/usb/devices").glob("*/idVendor"):
        name = device.parent / "product"
        if device.read_text().strip() == "8086" and name.exists() and "RealSense" in name.read_text():
            usb.append((device.parent.name, int((device.parent / "speed").read_text().strip() or 0)))
    print(f"  USB 上有 {len(usb)} 台 RealSense" + "".join(f"；口 {p} 只有 {v} Mb/s，换 USB3 口 ✗" for p, v in usb if v < 5000))
    users = run("fuser", *[str(p) for p in sorted(pathlib.Path("/dev").glob("video*"))]).split()
    if users:
        names = sorted({run("ps", "-o", "args=", "-p", pid).strip()[:70] for pid in set(users)} - {""})
        print("  正被占用，不枚举序列号：" + "；".join(names))
        return
    listed = run("rs-enumerate-devices", "-s", timeout=15)
    serials = [line.split()[-2] for line in listed.splitlines() if "RealSense" in line and len(line.split()) >= 2]
    for label, spec in wanted.items():
        serial = spec.split(":")[1] if spec.count(":") else spec
        print(f"  {label} {serial} {mark(serial in serials)}")
        if serial not in serials:
            missing.append(f"{label}没有插好")
    others = sorted(set(serials) - {s.split(":")[1] for s in wanted.values() if ":" in s})
    if others:
        print("  另外还有：" + "、".join(others))


def console() -> dict | None:
    print("== 控制台")
    state = get(f"http://127.0.0.1:{PORT}/api/state")
    if state is None:
        print("  没有运行（或没有回答） ✗ —— scripts/console.sh start")
        missing.append("控制台没有运行")
        return None
    arm, camera, tablet = state["arm"], state["camera"], state["tablet"]
    print(f"  机械臂：{'已连接（' + str(arm['mode']) + '）' if arm['connected'] else '未连接'} {mark(bool(arm['connected']))}"
          f"；正在做：{state['busy'] or '空闲'}；位置：{(state['place'] or {}).get('where', '不知道')}")  # fmt: skip
    print(f"  相机：{'已打开' if camera['open'] else '未打开'} {mark(camera['open'] and not camera['error'])}" + (f" —— {camera['error']}" if camera["error"] else ""))
    print(f"  平板：{tablet['state']}，标定码 {tablet['seen']} 个，贴码 {tablet.get('tags', 0)} 个" + (f"；{tablet['reach']}" if tablet["reach"] else ""))
    if state["fault"]:
        print(f"  故障：{state['fault']} ✗")
        missing.append("控制台报故障")
    if state["stopping"]:
        print("  ⚠ 已经叫停但还在“执行”：等几秒还这样就是卡住了，scripts/console.sh stop --force 再 start")
    for address in run("hostname", "-I").split():
        if address.startswith(("192.168.", "10.")):
            print(f"  操作页 http://{address}:{PORT}/console    标定页 http://{address}:{PORT}/calib    麦克风 https://{address}:8443/console")
    return state


def calibration(state: dict | None, setup: dict, wrist: dict) -> None:
    print("== 标定")
    opened, believed = (state or {}).get("setup", {}).get("camera"), setup.get("camera")
    if opened or believed:
        print(f"  顶部相机：{opened or believed}，内参{'有' if setup.get('camera_model') else '没有'}，位置{'有' if setup.get('camera_pose') else '没有'}")
    if state is not None:
        accuracy = state["accuracy"]
        print(f"  点按：{'已校准' if accuracy['affine'] else '没有校准'} {mark(accuracy['affine'])}（触点 {accuracy['contacts']} 个；{accuracy['correction']}）")
    newest = max((ROOT / "runs").glob("tapcal_*.json"), default=None, key=lambda p: p.name)
    summary = load(newest).get("summary") if newest else None
    if summary:
        print(f"  上次一键校准 {newest.stem[7:]}：平均 {summary['mean_mm']:.2f} mm，最大 {summary['max_mm']:.2f}，≤1 mm {summary['within_1mm']}/{summary['n']}")
    hand_eye = wrist.get("records", {}).get("eye_in_hand", {})
    rms = hand_eye.get("held_out_px")
    print(f"  腕部相机：支架 {wrist.get('mount', '没有记录')}，{'已标定' if wrist.get('in_flange') else '没有标定'}"
          + (f"（没参加拟合的位姿 {rms:.1f} px）" if isinstance(rms, (int, float)) else "")
          + f"；看屏位{'有' if wrist.get('look_q') else '没有'}；指令{'用' if wrist.get('records', {}).get('use_for_runs') else '不用'}腕部相机")  # fmt: skip


def services(state: dict | None) -> None:
    print("== 模型和语音")
    models = get("http://127.0.0.1:8802/v1/models")
    print(f"  GUI-Owl 模型：{'在运行 ✓' if models else '没有启动'}" + ("" if models else "（“指令”页点“启动模型”，占显卡约 24 GB）"))
    print(f"  语音识别：{'在运行 ✓' if get('http://127.0.0.1:8803/health') is not None else '没有启动（bash scripts/serve_asr.sh）'}")
    gpu = run("nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits").strip().splitlines()
    if gpu:
        used, total = (int(x) / 1024 for x in gpu[0].split(","))
        print(f"  显卡：已用 {used:.1f} / {total:.0f} GB")
    if run("pgrep", "-f", "run_real\\.py").strip():
        print("  ⚠ run_real 在运行：它占着机械臂和相机")
        missing.append("run_real 在运行")
    if state is not None:
        answer = get(f"http://127.0.0.1:{PORT}/api/run/state") or {}
        missing.extend(answer.get("missing", []))


def main() -> int:
    setup, wrist = load(ROOT / "calib" / "setup.json"), load(ROOT / "calib" / "wrist.json")
    can(setup.get("arm", {}).get("channel") or "can_l_follower")
    cameras({label: spec for label, spec in (("顶部相机", setup.get("camera")), ("腕部相机", wrist.get("camera"))) if spec})
    state = console()
    calibration(state, setup, wrist)
    services(state)
    print("== 现在能不能执行指令")
    print("  可以 ✓" if not missing else "  还不能，缺：\n" + "\n".join(f"    - {item}" for item in dict.fromkeys(missing)))
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
