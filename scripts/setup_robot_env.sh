#!/usr/bin/env bash
# 建机械臂这一侧的独立环境 venvs/robot（bin/py 用的就是它）。可重复运行。
#   bash scripts/setup_robot_env.sh
# 版本和 RoboSynChallenge 的 policy/fmc3_no1/.venv 里在台子上跑过的保持一致（2026-11-23 抄录）：
#   pyAgxArm：松灵官方仓库的同一个提交；pyrealsense2 2.59.1；numpy 1.26.4；opencv 4.11（带 aruco）；
#   pin 2.7.0（Pinocchio：在 assets/CobotMagicArm 的 URDF 上做正逆运动学）。
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset ALL_PROXY all_proxy   # socks:// 的写法 uv / pip 不认；HTTP(S)_PROXY 可以保留
cd "$here"
uv venv venvs/robot --python 3.11 --allow-existing
uv pip install --python venvs/robot/bin/python \
  "pyAgxArm @ git+https://github.com/agilexrobotics/pyAgxArm.git@841a625f5f4920e776f20b934eb13048b747e6d0" \
  "pyrealsense2==2.59.1.11053" \
  "numpy==1.26.4" "scipy==1.15.3" "opencv-python-headless==4.11.0.86" \
  "flask==3.1.1" "werkzeug==3.1.3" "requests==2.32.3" "openai==2.28.0" "pillow==11.2.1" \
  "pin==2.7.0" "pypinyin==0.55.0" "pytest==8.3.5" "ruff==0.11.12"
venvs/robot/bin/python - <<'PY'
import importlib.metadata as m
import cv2, pinocchio, pyAgxArm, pyrealsense2  # noqa: F401
assert hasattr(cv2, "aruco")
print("venvs/robot 就绪：", ", ".join(f"{p} {m.version(p)}" for p in ("pyAgxArm", "python-can", "pyrealsense2", "pin", "numpy", "opencv-python-headless", "openai")))
PY
