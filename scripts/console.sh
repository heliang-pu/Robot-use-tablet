#!/usr/bin/env bash
# 网页控制台（平板点按）的启动、停止和状态。
#
#   scripts/console.sh start [--fake]   后台启动；已经在跑就只报告状态
#   scripts/console.sh stop [--force]   停止；机械臂连着或正在执行时拒绝（--force 照停，手臂保持使能、停在原处）
#   scripts/console.sh restart [--fake]  默认强制：连着手臂、正在执行也重启（先叫停，手臂保持使能、停在原处）
#   scripts/console.sh status           整个台架的状态（scripts/status.py）：CAN、相机、控制台、标定、模型，还缺什么
#
# 端口：PORT（默认 8801，平板的标定页和操作页都用它）；HTTPS_PORT（默认 8443，浏览器里用麦克风说指令要走它，设 0 关掉）。
# 日志：runs/console.log。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PORT="${PORT:-8801}"
HTTPS_PORT="${HTTPS_PORT:-8443}"
LOG="$ROOT/runs/console.log"
PIDFILE="$ROOT/runs/console_$PORT.pid"

state() { curl -s -m 3 "http://127.0.0.1:$PORT/api/state" 2>/dev/null || true; }
field() { python3 -c 'import json,sys
try: d=json.load(sys.stdin)
except ValueError: sys.exit(0)
print(eval(sys.argv[1], {"d": d}))' "$1"; }
running_pid() {
  local pid
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && grep -qa "tablet_tap.web.console" "/proc/$pid/cmdline" 2>/dev/null; then echo "$pid"; return; fi
  pgrep -u "$(id -u)" -f "tablet_tap.web.console .*--port $PORT( |\$)" | head -1 || true
}
addresses() { hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^(192\.168|10\.)' || true; }

status() {
  local pid; pid="$(running_pid)"
  if [[ -z "$pid" ]]; then echo "控制台没有在运行。"; return 1; fi
  local s; s="$(state)"
  echo "控制台在运行（进程 $pid）。"
  if [[ -n "$s" ]]; then
    echo "  机械臂：$(echo "$s" | field '"已连接（" + str(d["arm"]["mode"]) + "）" if d["arm"]["connected"] else "未连接"')；正在做：$(echo "$s" | field 'd["busy"] or "空闲"')；相机：$(echo "$s" | field '"已打开" if d["camera"]["open"] else "未打开"')"
  else
    echo "  但端口 $PORT 没有回答。看 $LOG"
  fi
  for a in $(addresses); do
    echo "  操作页 http://$a:$PORT/console    平板标定页 http://$a:$PORT/calib"
    [[ "$HTTPS_PORT" != 0 ]] && echo "  要用麦克风 https://$a:$HTTPS_PORT/console（自签证书，浏览器里点“继续访问”）"
  done
  for name_url in "语音识别|http://127.0.0.1:8803/health" "GUI-Owl 模型|http://127.0.0.1:8802/v1/models"; do
    if curl -s -m 2 -o /dev/null "${name_url#*|}"; then echo "  ${name_url%|*}：在运行"; else echo "  ${name_url%|*}：没有启动"; fi
  done
}

start() {
  if [[ -n "$(running_pid)" ]]; then status; return; fi
  mkdir -p "$ROOT/runs"
  local extra=()
  [[ "$HTTPS_PORT" != 0 ]] && extra+=(--https_port "$HTTPS_PORT")
  if [[ -n "${TABLET_TAP_NO_HARDWARE:-}" && "${1:-}" != "--fake" ]]; then
    echo "（环境变量 TABLET_TAP_NO_HARDWARE 对控制台不生效：它要能连机械臂和相机。只看页面请用 start --fake）"
  fi
  echo "---- $(date '+%F %T') scripts/console.sh start $* ----" >> "$LOG"
  env -u TABLET_TAP_NO_HARDWARE setsid nohup "$ROOT/bin/py" -m tablet_tap.web.console --host 0.0.0.0 --port "$PORT" "${extra[@]}" "$@" >> "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  for _ in $(seq 40); do [[ -n "$(state)" ]] && break; sleep 0.5; done
  if [[ -z "$(state)" ]]; then echo "启动失败，日志最后几行："; tail -15 "$LOG"; return 1; fi
  status
}

stop() {
  local pid; pid="$(running_pid)"
  if [[ -z "$pid" ]]; then echo "控制台没有在运行。"; return 0; fi
  local s; s="$(state)"
  if [[ "${1:-}" != "--force" && -n "$s" ]]; then
    if [[ "$(echo "$s" | field 'bool(d["busy"]) or bool(d["arm"]["moves"])')" == "True" ]]; then
      echo "没有停：机械臂正连着（可运动）或正在执行“$(echo "$s" | field 'd["busy"]')”。先在网页里点“断开”，或者加 --force。"
      return 1
    fi
  fi
  if [[ -n "$s" && "$(echo "$s" | field 'bool(d["busy"])')" == "True" ]]; then
    echo "正在执行“$(echo "$s" | field 'd["busy"]')”：先叫停（手臂停在原处）"
    curl -s --noproxy '*' -m 3 -X POST "http://127.0.0.1:$PORT/api/stop" >/dev/null || true
    for _ in $(seq 20); do [[ "$(state | field 'bool(d["busy"])')" == "False" ]] && break; sleep 0.25; done
  fi
  kill "$pid"
  for _ in $(seq 40); do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done
  if kill -0 "$pid" 2>/dev/null; then echo "进程 $pid 没有退出。"; return 1; fi
  rm -f "$PIDFILE"
  echo "控制台已停止。"
}

case "${1:-status}" in
  start) shift; start "$@" ;;
  stop) shift; stop "$@" ;;
  restart) shift; [[ "${1:-}" == "--force" ]] && shift; stop --force; start "$@" ;;  # always: the arm stays enabled where it is
  status) exec python3 "$ROOT/scripts/status.py" ;;  # 整个台架，不只控制台
  *) sed -n '2,11p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
