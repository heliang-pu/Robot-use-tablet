#!/bin/bash
# =============================================================================
# setup_piper_can.sh —— PiPER 双臂主从 CAN 一键配置
#
# 把 4 个 candleLight USB-CAN 适配器（1d50:606f，gs_usb 驱动）按「序列号」固定成 4 个名字：
# 插上就自动改名 + 拉到 1 Mbps，掉线 / BUS-OFF 自动恢复。名字 ↔ 序列号的对应关系写在下面的「登记表」里。
#
#   用法
#     sudo bash setup_piper_can.sh                          一键配置（= install，幂等，可重复跑）
#     sudo bash setup_piper_can.sh install --dry-run        只打印会改什么，不动系统（不需要 sudo）
#          bash setup_piper_can.sh status                   只读：四个口的状态 / 序列号 / 从臂报文
#     sudo bash setup_piper_can.sh register can_l_leader    换了适配器：把「当前唯一一个未登记」的适配器登记成这个名字
#
#   迁移到别的电脑：只需要这一个文件。
#     适配器随臂一起搬（序列号不变）→ 直接 install；换了新适配器 → 一次只插一个，逐个 register。
#       scp setup_piper_can.sh 新机器:~ && ssh 新机器 'sudo bash setup_piper_can.sh'
#       不落地：ssh 新机器 'sudo bash -s' < setup_piper_can.sh
#       register 同理：ssh 新机器 'sudo bash -s' -- register can_l_leader < setup_piper_can.sh
#       （bash -s 方式下脚本改不了自己，register 会把要放进登记表的行打印出来，手动贴回本文件）
#
#   它会改动（每个被改的文件先备份成 <原名>.bak-<时间>；内容没变就不动）
#     /etc/udev/rules.d/99-piper-bimanual-can.rules   序列号 → 名字；改名后调 can_up.sh 拉起。
#                                                      登记表之外的旧规则（别的适配器）原样保留，不删
#     /usr/local/bin/can_up.sh                        1 Mbps、txqueuelen 1000、up
#     /usr/local/bin/can_watchdog.sh                  看门狗：接口 DOWN / BUS-OFF 时自动重拉
#     /etc/systemd/system/can-watchdog.service        看门狗的服务单元（开机自启）
#     /etc/modules-load.d/piper-can.conf              开机加载 can / can_raw / gs_usb
#     can-utils                                       没有 candump 时 apt 装一个（--no-apt 跳过）
#   不碰：已经在跑的接口。install 只对「名字不对 / 没拉起 / 波特率不对」的口动手，正在控制的臂不会被断一下。
#
#   约定：名字以 follower 结尾的是从臂，status 会听它的反馈报文；以 leader 结尾的是主臂，PiPER 主臂本来就不反馈，不听。
#   撤销：删上面列的文件（或把 .bak-* 改回来），systemctl disable --now can-watchdog，udevadm control --reload。
#   已验证：2026-10-12 在 4090 工作站(fmc3-robotics-01000，Ubuntu 24.04.4，内核 7.0.0-38，candleLight/gs_usb)上实测——
#     install 幂等且不打断在跑的接口；真实热插拔自动改名 + 拉到 1 Mbps；名字不对 / 波特率不对的口会被纠正；register 能改写本脚本的登记表。
#     没有在全新机器上整机跑过：第一次迁移建议先 `install --dry-run` 看一遍它要改什么。
# =============================================================================

# ---- 登记表：接口名 ← 适配器序列号 -------------------------------------------
# 换适配器用 `register`，它会自动改写下面这块（# >>> / # <<< 之间）；也可以手改后再 install。
IFS= read -r -d '' MAP_TEXT <<'MAP_EOF'
# >>> MAP-BEGIN
# 接口名(<=15字符)   适配器序列号                  说明
can_l_leader        002100205547570420303135     左主臂
can_r_leader        0028002C5547571120343930     右主臂
can_l_follower      002200385547570420303135     左从臂
can_r_follower      002E002F5547571120343930     右从臂
# <<< MAP-END
MAP_EOF

BITRATE=1000000

usage() {
  cat <<'USAGE_EOF'
用法:
  sudo bash setup_piper_can.sh                         一键配置(= install, 幂等, 可重复跑)
  sudo bash setup_piper_can.sh install --dry-run       只打印会改什么, 不动系统
       bash setup_piper_can.sh status                  只读: 四个口的状态 / 序列号 / 从臂报文
  sudo bash setup_piper_can.sh register <接口名>        换了适配器: 把「当前唯一一个未登记」的适配器登记成这个名字
选项: --dry-run 只打印不动系统;  --no-apt 不自动装 can-utils
USAGE_EOF
}

# ---------------------------------------------------------------- 基础工具 ----
say()  { printf '%s\n' "$*"; }
warn() { printf '  ⚠ %s\n' "$*"; }

# 登记表 → 三个并行数组 M_NAME / M_SER / M_NOTE
load_map() {
  M_NAME=(); M_SER=(); M_NOTE=()
  local line name ser note
  while IFS= read -r line; do
    case "$line" in ''|'#'*) continue;; esac
    read -r name ser note <<< "$line"
    if ! printf '%s' "$name" | grep -Eq '^[a-z][a-z0-9_]{0,14}$'; then
      say "登记表里的接口名不合法(须小写字母开头、<=15 字符): $name"; return 1
    fi
    if ! printf '%s' "$ser" | grep -Eq '^[0-9A-Za-z]+$'; then
      say "登记表里 $name 的序列号不合法: $ser"; return 1
    fi
    M_NAME+=("$name"); M_SER+=("$ser"); M_NOTE+=("$note")
  done <<< "$MAP_TEXT"
  [ "${#M_NAME[@]}" -gt 0 ] || { say "登记表是空的"; return 1; }
}

map_name_of() {            # 序列号 → 登记表里的名字（没有则无输出）
  local i; for i in "${!M_SER[@]}"; do [ "${M_SER[$i]}" = "$1" ] && { printf '%s\n' "${M_NAME[$i]}"; return 0; }; done; return 1
}
rules_name_of() {          # 序列号 → 现有规则文件里已绑定的名字（没有则无输出）
  [ -f "$RULES" ] || return 0
  grep -v '^[[:space:]]*#' "$RULES" | grep -F "ATTRS{serial}==\"$1\"" | head -1 | sed -n 's/.*NAME="\([^"]*\)".*/\1/p'
}
known_name_of() { local n; n=$(map_name_of "$1"); [ -n "$n" ] || n=$(rules_name_of "$1"); printf '%s\n' "$n"; }

# 在位的 candleLight 适配器 → A_SER / A_PORT / A_IF
scan_adapters() {
  A_SER=(); A_PORT=(); A_IF=()
  local d n s i
  for d in /sys/bus/usb/devices/*; do
    [ -f "$d/idVendor" ] || continue
    [ "$(cat "$d/idVendor" 2>/dev/null)" = 1d50 ] && [ "$(cat "$d/idProduct" 2>/dev/null)" = 606f ] || continue
    s=$(cat "$d/serial" 2>/dev/null)
    i=$(for n in "$d"/*/net/*; do [ -e "$n" ] && basename "$n"; done | head -1)
    A_SER+=("$s"); A_PORT+=("$(basename "$d")"); A_IF+=("$i")
  done
}

link_flags() { ip -o link show "$1" 2>/dev/null | sed -n 's/.*<\(.*\)>.*/\1/p'; }
link_state() {             # 接口名 → "UP|DOWN 波特率 CAN状态"
  local det br cs
  det=$(ip -details link show "$1" 2>/dev/null)
  br=$(printf '%s\n' "$det" | grep -o 'bitrate [0-9]*' | head -1 | awk '{print $2}')
  cs=$(printf '%s\n' "$det" | grep -o 'can state [A-Z-]*' | head -1 | awk '{print $3}')
  case ",$(link_flags "$1")," in *,UP,*) echo "UP ${br:--} ${cs:--}";; *) echo "DOWN ${br:--} ${cs:--}";; esac
}

# 内容规范化（去注释/空行/排序），用来判断 udev 规则语义上有没有变化
norm() { grep -v '^[[:space:]]*#' "$1" | grep -v '^[[:space:]]*$' | sort; }

# 写文件：内容没变 → 不动(返回 0)；变了 → 先备份再写(返回 1)；dry-run 只打印(返回 1)
put_file() {               # 目标 权限 内容文件 [sem]
  local dest="$1" mode="$2" src="$3" sem="$4" bk
  if [ -e "$dest" ]; then
    if [ "$sem" = sem ]; then
      [ "$(norm "$dest")" = "$(norm "$src")" ] && { say "  = $dest 已是最新"; return 0; }
    else
      cmp -s "$dest" "$src" && { say "  = $dest 已是最新"; return 0; }
    fi
  fi
  if [ "$DRY" = 1 ]; then
    say "  + [dry-run] 会写入 $dest"
    [ -e "$dest" ] && diff -u "$dest" "$src" | sed 's/^/      /' | head -24
    return 1
  fi
  mkdir -p "$(dirname "$dest")"
  if [ -e "$dest" ]; then bk="$dest.bak-$(date +%Y%m%d-%H%M%S)"; cp -a "$dest" "$bk" && say "  备份 → $bk"; fi
  cp "$src" "$dest.new" && chmod "$mode" "$dest.new" || return 1
  [ -n "$SANDBOX" ] || chown root:root "$dest.new"
  mv -f "$dest.new" "$dest" && say "  + 已写入 $dest"
  return 1
}

# ------------------------------------------------------------ 要装的文件内容 ----
rule_line() {              # 名字 序列号
  printf 'SUBSYSTEM=="net", ACTION=="add", ATTRS{idVendor}=="1d50", ATTRS{idProduct}=="606f", ATTRS{serial}=="%s", NAME="%s", RUN+="/usr/local/bin/can_up.sh %s"\n' "$2" "$1" "$1"
}

# 现有规则文件里「不在登记表里」的规则行（别的适配器），原样保留
preserved_lines() {
  [ -f "$RULES" ] || return 0
  local line t ser s skip
  while IFS= read -r line; do
    t="${line#"${line%%[![:space:]]*}"}"
    case "$t" in ''|'#'*) continue;; esac
    ser=$(printf '%s' "$line" | sed -n 's/.*ATTRS{serial}=="\([^"]*\)".*/\1/p')
    skip=0
    if [ -n "$ser" ]; then for s in "${M_SER[@]}" $DROP_SERIALS; do [ "$s" = "$ser" ] && skip=1; done; fi
    [ "$skip" = 1 ] || printf '%s\n' "$line"
  done < "$RULES"
}

gen_rules() {
  local i names=""
  for i in "${!M_NAME[@]}"; do names="${names:+$names / }${M_NAME[$i]}"; done
  say "# PiPER 双臂主从 CAN：按适配器序列号永久命名 + 插上自动拉到 1 Mbps (candleLight USB-to-CAN, 1d50:606f)"
  say "# 由 setup_piper_can.sh 生成：登记表在脚本里，手改会在下次 install 时被覆盖；登记表之外的旧规则原样保留在「保留」段"
  say "# 命名(<=15字符): $names"
  local kept; kept=$(preserved_lines)
  if [ -n "$kept" ]; then say ""; say "# ---- 保留：不在登记表里的旧规则（别的适配器），原样 ----"; say "$kept"; fi
  say ""; say "# ---- 登记表 ----"
  for i in "${!M_NAME[@]}"; do rule_line "${M_NAME[$i]}" "${M_SER[$i]}"; done
}

all_names() {              # 登记表里的名字 + 保留的旧规则里的名字（去重），给看门狗用
  local n out=""
  for n in "${M_NAME[@]}" $(preserved_lines | sed -n 's/.*NAME="\([^"]*\)".*/\1/p'); do
    case " $out " in *" $n "*) ;; *) out="$out $n";; esac
  done
  echo $out
}

gen_can_up() {
  cat <<'CANUP_EOF'
#!/bin/bash
# 自动拉起 PiPER CAN 接口(udev 在插入并改名后用真名调用)
IFACE="$1"
BITRATE=1000000
# 等接口出现(改名可能稍滞后),最多约 3s
for _ in $(seq 1 30); do ip link show "$IFACE" >/dev/null 2>&1 && break; sleep 0.1; done
ip link set "$IFACE" down 2>/dev/null || true
ip link set "$IFACE" type can bitrate "$BITRATE"
ip link set "$IFACE" txqueuelen 1000
ip link set "$IFACE" up
CANUP_EOF
}

gen_watchdog() {
  local names; names=$(all_names)
  cat <<'WD_EOF' | sed "s|@IFACES@|$names|"
#!/bin/bash
# 监控 PiPER CAN 接口:管理态变 DOWN 或进入 BUS-OFF 时自动重拉(gs_usb 不支持 restart-ms,用软件看门狗)
# 由 setup_piper_can.sh 安装;监控名单来自它的登记表(+ 规则里保留的旧名字)
IFACES="@IFACES@"
INTERVAL=2
while true; do
  for IFACE in $IFACES; do
    ip link show "$IFACE" >/dev/null 2>&1 || continue          # 没插 → 跳过
    flags=$(ip -o link show "$IFACE" | sed -n 's/.*<\(.*\)>.*/\1/p')
    canstate=$(ip -details link show "$IFACE" 2>/dev/null | grep -o 'can state [A-Z-]*' | awk '{print $3}')
    case ",$flags," in *,UP,*) up=1;; *) up=0;; esac
    if [ "$up" = 0 ] || [ "$canstate" = "BUS-OFF" ]; then
      logger -t can-watchdog "reviving $IFACE (up=$up can=$canstate)"
      /usr/local/bin/can_up.sh "$IFACE"
    fi
  done
  sleep "$INTERVAL"
done
WD_EOF
}

gen_unit() {
  cat <<'UNIT_EOF'
[Unit]
Description=PiPER CAN interface watchdog (auto-recover DOWN/BUS-OFF)
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/local/bin/can_watchdog.sh
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
UNIT_EOF
}

gen_modules() {
  cat <<'MOD_EOF'
# PiPER CAN: 开机就加载 SocketCAN 核心 + candleLight(gs_usb) 驱动，
# 避免适配器刚拉起时到的帧因 can 协议模块还没加载而被丢
can
can_raw
gs_usb
MOD_EOF
}

# ------------------------------------------------------------------ 安装步骤 ----
step_rules() {
  gen_rules > "$TMPD/rules"
  put_file "$RULES" 644 "$TMPD/rules" sem; RULES_CHANGED=$?
}

step_canup() {
  gen_can_up > "$TMPD/can_up.sh"
  put_file "$CAN_UP" 755 "$TMPD/can_up.sh"; CANUP_CHANGED=$?
}

step_watchdog() {
  gen_watchdog > "$TMPD/wd.sh";  put_file "$WD" 755 "$TMPD/wd.sh";    WD_CHANGED=$?
  gen_unit     > "$TMPD/unit";   put_file "$UNIT" 644 "$TMPD/unit";   UNIT_CHANGED=$?
  if [ -n "$SANDBOX" ] || [ "$DRY" = 1 ]; then
    say "  [跳过] systemctl（沙盒 / dry-run）"
  elif command -v systemctl >/dev/null 2>&1 && [ -d /run/systemd/system ]; then
    [ "$UNIT_CHANGED" = 1 ] && systemctl daemon-reload
    systemctl enable can-watchdog.service >/dev/null 2>&1
    if [ "$WD_CHANGED" = 1 ] || [ "$UNIT_CHANGED" = 1 ] || ! systemctl is-active --quiet can-watchdog.service; then
      systemctl restart can-watchdog.service
    fi
    say "  can-watchdog.service: $(systemctl is-active can-watchdog.service)，开机自启: $(systemctl is-enabled can-watchdog.service 2>/dev/null)"
  else
    warn "没有 systemd，看门狗没启用（接口掉线要手动 can_up.sh <名字>）"
  fi
  if [ -z "$SANDBOX" ] && command -v systemctl >/dev/null 2>&1 && systemctl is-enabled piper-can.service >/dev/null 2>&1; then
    warn "发现老方案 piper-can.service（和本脚本无关）：它用 USB 口位置命名，可能与按序列号命名冲突，建议 systemctl disable piper-can.service"
  fi
}

step_modules() {
  gen_modules > "$TMPD/mod"
  put_file "$MODF" 644 "$TMPD/mod"
  if [ -z "$SANDBOX" ] && [ "$DRY" != 1 ]; then modprobe -a can can_raw gs_usb 2>/dev/null || true; fi
}

step_deps() {
  if command -v candump >/dev/null 2>&1; then say "  = can-utils 已装（candump 可用）"; return; fi
  if [ -n "$SANDBOX" ] || [ "$DRY" = 1 ] || [ "$NOAPT" = 1 ]; then say "  [跳过] 没有 candump；status 听不了报文（apt install can-utils 可装）"; return; fi
  if command -v apt-get >/dev/null 2>&1; then
    say "  装 can-utils …"
    DEBIAN_FRONTEND=noninteractive timeout 180 apt-get install -y can-utils >/dev/null 2>&1 \
      && say "  + 已装 can-utils" || warn "can-utils 没装成（不影响 CAN 本身，只是 status 听不了报文）"
  else
    warn "没有 apt-get：请自行安装 can-utils（可选）"
  fi
}

# 让一个在位的适配器进入「名字对 + UP + 1 Mbps」。已经就绪的绝不动它。
apply_one() {              # 当前接口名 目标名
  local cur="$1" want="$2" st b c k
  if [ -z "$cur" ]; then say "  ✗ 适配器在位但没有对应的网络接口（gs_usb 驱动没加载？试试 modprobe gs_usb）"; return 1; fi
  if [ "$cur" = "$want" ]; then
    read -r st b c <<< "$(link_state "$cur")"
    if [ "$st" = UP ] && [ "$b" = "$BITRATE" ]; then say "  ✓ $want 已就绪（$st ${b}bps $c），没动它"; return 0; fi
  fi
  if [ -n "$SANDBOX" ]; then say "  [沙盒] 会对 $cur 触发 udev → $want"; return 0; fi
  if [ "$DRY" = 1 ];   then say "  [dry-run] 会对 $cur 触发 udev add → $want"; return 0; fi
  if [ "$cur" != "$want" ]; then            # 改名要求接口是 down 的
    case ",$(link_flags "$cur")," in *,UP,*) ip link set "$cur" down;; esac
  fi
  udevadm trigger --action=add --subsystem-match=net --sysname-match="$cur"
  udevadm settle --timeout=10
  for k in $(seq 1 50); do ip link show "$want" >/dev/null 2>&1 && break; sleep 0.1; done
  sleep 0.6                                  # 等 can_up.sh 跑完
  if ip link show "$want" >/dev/null 2>&1; then
    read -r st b c <<< "$(link_state "$want")"; say "  ✓ $want  $st ${b}bps $c"
  else
    say "  ✗ 改名失败：没出现 $want（现在还叫 $cur）。看 journalctl -u systemd-udevd -n 20"; return 1
  fi
}

step_apply() {
  scan_adapters
  if [ "${#A_SER[@]}" -eq 0 ]; then say "  还没插适配器；插上后会自动改名并拉起，不用再跑本脚本"; return 0; fi
  local i want
  for i in "${!A_SER[@]}"; do
    want=$(known_name_of "${A_SER[$i]}")
    if [ -z "$want" ]; then
      say "  ? 未登记的适配器 serial=${A_SER[$i]}（USB口 ${A_PORT[$i]}，现名 ${A_IF[$i]}）→ 用 register <名字> 登记"
    else
      apply_one "${A_IF[$i]}" "$want"
    fi
  done
}

# ------------------------------------------------------------------ 子命令 ----
cmd_install() {
  say "== 1/6 udev 规则（序列号 → 名字）";      step_rules
  [ "$RULES_CHANGED" = 1 ] && [ -z "$SANDBOX" ] && [ "$DRY" != 1 ] && udevadm control --reload
  say "== 2/6 拉起脚本 can_up.sh";             step_canup
  say "== 3/6 看门狗（掉线 / BUS-OFF 自动恢复）"; step_watchdog
  say "== 4/6 开机加载内核模块";                step_modules
  say "== 5/6 can-utils（调试 / 验证用）";       step_deps
  say "== 6/6 对已插着的适配器立即生效";         step_apply
  say ""
  cmd_status
  local i l="" r=""
  for i in "${!M_NAME[@]}"; do
    case "${M_NAME[$i]}" in can_l_follower) l=1;; can_r_follower) r=1;; esac
  done
  if [ -n "$l" ] && [ -n "$r" ]; then
    say ""
    say "真机代码（policy/fmc3_no1/real_robot）里从臂的通道参数 —— 它默认找 can_left / can_right，要显式传："
    say "  --left_channel can_l_follower --right_channel can_r_follower"
  fi
}

cmd_status() {
  scan_adapters
  say "== 状态"
  printf '  %-16s %-26s %-11s %-6s %-9s %-13s %s\n' NAME SERIAL USB-PORT LINK BITRATE CAN-STATE NOTE
  local i j cur found st b c n frames
  for i in "${!M_NAME[@]}"; do
    found=""
    for j in "${!A_SER[@]}"; do [ "${A_SER[$j]}" = "${M_SER[$i]}" ] && found=$j; done
    if [ -z "$found" ]; then
      printf '  %-16s %-26s %-11s %s\n' "${M_NAME[$i]}" "${M_SER[$i]}" "-" "未插（${M_NOTE[$i]}）"
      continue
    fi
    cur="${A_IF[$found]}"
    if [ "$cur" != "${M_NAME[$i]}" ]; then
      printf '  %-16s %-26s %-11s %s\n' "${M_NAME[$i]}" "${M_SER[$i]}" "${A_PORT[$found]}" "✗ 在位但还叫 $cur（跑 install；新插的会自动改名）"
      continue
    fi
    read -r st b c <<< "$(link_state "$cur")"
    n="${M_NOTE[$i]}"
    case "$cur" in
      *follower)
        if command -v candump >/dev/null 2>&1 && command -v timeout >/dev/null 2>&1 && [ "$st" = UP ]; then
          frames=$(timeout 2 candump -L "$cur" 2>/dev/null | wc -l | tr -d ' ')
          n="$n；2秒收到 $frames 帧"; [ "$frames" -gt 0 ] || n="$n（从臂没上电 / 线没接好？）"
        fi;;
      *leader) n="$n；主臂不反馈报文，不听";;
    esac
    printf '  %-16s %-26s %-11s %-6s %-9s %-13s %s\n' "$cur" "${M_SER[$i]}" "${A_PORT[$found]}" "$st" "$b" "$c" "$n"
  done
  for j in "${!A_SER[@]}"; do
    [ -n "$(known_name_of "${A_SER[$j]}")" ] || printf '  ? 未登记的适配器 serial=%s（USB口 %s，现名 %s）→ register <名字>\n' "${A_SER[$j]}" "${A_PORT[$j]}" "${A_IF[$j]}"
  done
}

map_block() {
  local i
  printf '# 接口名(<=15字符)   适配器序列号                  说明\n'
  for i in "${!M_NAME[@]}"; do printf '%-19s %-28s %s\n' "${M_NAME[$i]}" "${M_SER[$i]}" "${M_NOTE[$i]}"; done
}

save_map_to_self() {
  if [ -z "$SELF" ] || [ ! -w "$SELF" ]; then
    warn "本脚本不是可写的文件（可能是 bash -s 方式运行），登记表没法自动更新。请手动把下面这块贴回本文件的登记表："
    map_block | sed 's/^/      /'; return 0
  fi
  map_block > "$TMPD/blk"
  awk 'FNR==NR { blk[++n]=$0; next }
       $0 == "# >>> MAP-BEGIN" { print; for (k=1;k<=n;k++) print blk[k]; skip=1; next }
       $0 == "# <<< MAP-END"   { skip=0 }
       !skip { print }' "$TMPD/blk" "$SELF" > "$TMPD/self.new"
  if bash -n "$TMPD/self.new" 2>/dev/null && grep -q '^# >>> MAP-BEGIN$' "$TMPD/self.new" && grep -q '^# <<< MAP-END$' "$TMPD/self.new"; then
    cp -a "$SELF" "$SELF.bak-$(date +%Y%m%d-%H%M%S)" 2>/dev/null
    cat "$TMPD/self.new" > "$SELF" && say "  + 已把新的登记表写回 $SELF（旧版本备份为 $SELF.bak-*）"
  else
    warn "改写本脚本的登记表没通过自检，没动它。请手动贴："; map_block | sed 's/^/      /'
  fi
}

cmd_register() {
  local name="$1" i k st idx
  local -a unreg
  unreg=()
  scan_adapters
  say "== 在位的适配器"
  if [ "${#A_SER[@]}" -eq 0 ]; then say "  没看到任何 candleLight 适配器（1d50:606f），什么都没改。"; return 1; fi
  for i in "${!A_SER[@]}"; do
    k=$(known_name_of "${A_SER[$i]}")
    st="未登记"; [ -n "$k" ] && st="已登记为 $k"
    printf '  USB口=%-12s serial=%s 现名=%-16s %s\n' "${A_PORT[$i]}" "${A_SER[$i]}" "${A_IF[$i]}" "$st"
    [ -z "$k" ] && unreg+=("$i")
  done
  if [ "${#unreg[@]}" -eq 0 ]; then
    for i in "${!A_SER[@]}"; do
      if [ "$(known_name_of "${A_SER[$i]}")" = "$name" ] && [ "${#A_SER[@]}" -eq 1 ]; then
        say "→ 唯一在位的适配器已经登记为 $name，只确认它的状态："; apply_one "${A_IF[$i]}" "$name"; return $?
      fi
    done
    say "→ 在位的适配器都已登记过（见上），没有新的可登记，什么都没改。"; return 1
  fi
  if [ "${#unreg[@]}" -gt 1 ]; then say "→ 有 ${#unreg[@]} 个未登记的适配器，分不清哪个是 $name，什么都没改。请只留一个新的。"; return 1; fi
  idx=${unreg[0]}
  say "→ 要登记的: serial=${A_SER[$idx]}（USB口 ${A_PORT[$idx]}，现名 ${A_IF[$idx]}）→ $name"

  # 更新登记表（同名则替换，旧序列号的规则一并丢掉）
  k=""
  for i in "${!M_NAME[@]}"; do [ "${M_NAME[$i]}" = "$name" ] && k=$i; done
  if [ -n "$k" ]; then
    warn "$name 原来登记的是 ${M_SER[$k]}，将被替换"; DROP_SERIALS="$DROP_SERIALS ${M_SER[$k]}"
    M_SER[$k]="${A_SER[$idx]}"; M_NOTE[$k]="${M_NOTE[$k]}"
  else
    M_NAME+=("$name"); M_SER+=("${A_SER[$idx]}"); M_NOTE+=("register $(date +%F)")
  fi
  say "== udev 规则";  step_rules
  [ "$RULES_CHANGED" = 1 ] && [ -z "$SANDBOX" ] && [ "$DRY" != 1 ] && udevadm control --reload
  say "== can_up.sh / 看门狗"; step_canup; step_watchdog
  say "== 立即生效";  apply_one "${A_IF[$idx]}" "$name"
  [ "$DRY" = 1 ] || { say "== 更新本脚本里的登记表"; save_map_to_self; }
  say ""; cmd_status
}

# ------------------------------------------------------------------- 入口 ----
need_root() {
  [ -n "$SANDBOX" ] && return 0
  [ "$(id -u)" -eq 0 ] && return 0
  if [ -n "$SELF" ] && command -v sudo >/dev/null 2>&1; then exec sudo bash "$SELF" "${ORIG_ARGS[@]}"; fi
  say "需要 root：请用  sudo bash setup_piper_can.sh ...  运行"; exit 1
}

main() {
  PATH="$PATH:/usr/sbin:/sbin:/usr/local/sbin"
  ORIG_ARGS=("$@"); DRY=0; NOAPT=0; CMD=""; ARG=""; DROP_SERIALS=""; FOLLOWERS=""
  SELF=""; [ -f "$0" ] && SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  local a
  for a in "$@"; do
    case "$a" in
      --dry-run|-n) DRY=1;;
      --no-apt)     NOAPT=1;;
      -h|--help)    usage; return 0;;
      install|status|register) if [ -z "$CMD" ]; then CMD="$a"; elif [ "$CMD" = register ] && [ -z "$ARG" ]; then ARG="$a"; else say "多余的参数: $a"; usage; return 2; fi;;
      *) if [ "$CMD" = register ] && [ -z "$ARG" ]; then ARG="$a"; else say "不认识的参数: $a"; usage; return 2; fi;;
    esac
  done
  [ -n "$CMD" ] || CMD=install

  # 沙盒：设了 PIPER_CAN_ROOT 就只往那个目录下写文件、不碰系统（测试用）
  P="${PIPER_CAN_ROOT:-}"; SANDBOX=""; [ -n "$P" ] && SANDBOX=1
  RULES="$P/etc/udev/rules.d/99-piper-bimanual-can.rules"
  CAN_UP="$P/usr/local/bin/can_up.sh"
  WD="$P/usr/local/bin/can_watchdog.sh"
  UNIT="$P/etc/systemd/system/can-watchdog.service"
  MODF="$P/etc/modules-load.d/piper-can.conf"

  if [ "$CMD" = register ]; then
    [ -n "$ARG" ] || { say "register 需要接口名，例如: register can_l_leader"; return 2; }
    printf '%s' "$ARG" | grep -Eq '^[a-z][a-z0-9_]{0,14}$' || { say "接口名不合法(小写字母开头, <=15 字符): $ARG"; return 2; }
  fi
  if [ "$CMD" != status ] && [ "$DRY" != 1 ]; then need_root; fi    # 需要的话在这里用 sudo 重新执行自己

  TMPD=$(mktemp -d 2>/dev/null || mktemp -d -t piper-can)
  trap 'rm -rf "$TMPD"' EXIT
  load_map || return 1

  case "$CMD" in
    install)  [ "$DRY" = 1 ] && say "（dry-run：只打印，不改系统）"; cmd_install;;
    status)   cmd_status;;
    register) [ "$DRY" = 1 ] && say "（dry-run：只打印，不改系统）"; cmd_register "$ARG";;
  esac
}

# 整个脚本都在函数里、最后一行才执行：register 改写本文件的登记表时，正在跑的 bash 不会读到被改过的内容
main "$@"; exit $?
