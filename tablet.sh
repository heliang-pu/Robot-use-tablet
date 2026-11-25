#!/usr/bin/env bash
# 平板点按的一键入口（转调 scripts/ 里的同名功能），在任意目录都能运行：
#
#   ./tablet.sh status             整个台架的状态：CAN、相机、控制台、标定、模型和语音、显卡，最后说现在还缺什么（只看不碰）
#   ./tablet.sh console [--fake]   启动网页控制台并打印网址；已经在跑就只报告状态（--fake：模拟手臂，不碰硬件）
#   ./tablet.sh stop               停止控制台；手臂连着（可运动）或正在执行时不停，先在网页点“断开”（或加 --force）
#   ./tablet.sh restart            强制重启：手臂连着、正在执行也重启，手臂停在原处不断电；之后刷新网页、重新点“连接”
#   ./tablet.sh log                看控制台日志（Ctrl-C 退出，不影响控制台）
#
# 手臂在示教模式、急停、碰撞：网页“点按”页的“故障恢复”，不用断电。
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
command="${1:-help}"
[[ $# -gt 0 ]] && shift
case "$command" in
  status) exec "$ROOT/scripts/console.sh" status "$@" ;;
  console | start) exec "$ROOT/scripts/console.sh" start "$@" ;;
  stop | restart) exec "$ROOT/scripts/console.sh" "$command" "$@" ;;
  log) exec tail -n 60 -F "$ROOT/runs/console.log" ;;
  help | -h | --help) sed -n '2,10p' "${BASH_SOURCE[0]}" ;;
  *) echo "不认识的命令：$command" >&2; sed -n '2,10p' "${BASH_SOURCE[0]}" >&2; exit 2 ;;
esac
