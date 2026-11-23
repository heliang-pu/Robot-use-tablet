#!/usr/bin/env bash
# 用 vLLM 起 GUI-Owl-1.5 的 OpenAI 兼容服务（上游模型卡推荐的部署方式）。环境见 scripts/setup_vlm_env.sh。
#
# 用法:
#   scripts/serve_vlm.sh [--model weights/GUI-Owl-1.5-8B-Instruct] [--port 8802] [--gpu-frac auto|0.50]
#                        [--name gui-owl] [--max-model-len 32768] [--gpu 0] [--force] [--dry-run] [-- 其它 vllm 参数]
#
#   --model          模型目录（相对路径先按当前目录找，再按项目根目录找）。默认 8B。
#   --port           监听端口，默认 8802；只监听 127.0.0.1。
#   --gpu-frac       本服务最多占整块显卡显存的比例（vLLM 的 --gpu-memory-utilization）。
#                    默认 auto：按下面的算法从模型目录算出来（8B → 0.50，2B → 0.22）。
#   --name           --served-model-name，客户端请求里的 model 字段，默认 gui-owl。
#   --max-model-len  上下文长度，默认 32768（上游的值）。
#   --gpu            用哪块显卡（CUDA_VISIBLE_DEVICES），默认 0。
#   --force          空闲显存不够时也照样启动（默认拒绝）。
#   --dry-run        只做检查并打印最终命令，不启动。
#   -- 之后的参数原样交给 vllm serve。例：让服务器默认用贪心解码（上游运行脚本的请求不带 temperature，
#      服务器就按模型 generation_config.json 里的 temperature 0.6 / top_k 40 / top_p 0.95 采样）：
#        scripts/serve_vlm.sh -- --override-generation-config '{"temperature": 0}'
#
# 上游模型卡（weights/GUI-Owl-1.5-8B-Instruct/README.md）的参数原样保留：
#   --max-model-len 32768
#   --mm-processor-kwargs '{"size": {"longest_edge": 3072000, "shortest_edge": 65536}}'   每张图最多 3072000 像素 = 3000 个 token
#   --limit-mm-per-prompt image=5      （vLLM 0.34 的帮助里只有 JSON 写法，这里写成 '{"image": 5}'，意思相同）
#   --allowed-local-media-path /
# 另外加的：--served-model-name、--host 127.0.0.1、--gpu-memory-utilization，并且清掉代理环境变量。
#
# ---- 显存比例怎么算的 ------------------------------------------------------------------------------------------
# 这块 4090 是 48 GB 版（nvidia-smi 报 49140 MiB = 47.99 GiB），和机器人推理进程共用，所以不能用 vLLM 的默认值
# （0.34 不设就是 1.0：可扩展 KV 缓存会把启动后量到的空闲显存全拿走；关掉可扩展缓存时是 0.92）。
# `vllm serve --help=gpu-memory-utilization` 的说法：这个比例是"本实例"的上限（占整块显卡显存的比例，和别的
# 进程占多少无关），有别的进程共用显卡时就该显式设它，"调低它是从 KV 缓存里扣"。也就是
# 总显存 × 比例 − 权重 − 激活峰值 − 非 torch 开销 = KV 缓存；KV 缓存装不下一条 max-model-len 长的序列就拒绝启动。
# 所以需要的显存 = 权重 + 一条满长序列的 KV + 开销。
#
#   权重（bf16，*.safetensors 的大小）        8B: 17.53 GB = 16.33 GiB        2B: 4.26 GB = 3.96 GiB
#   每个 token 的 KV（bf16）= 2(K,V) × 层数 × KV 头数 × 头维度 × 2 字节
#                                             8B: 2×36×8×128×2 = 147456 B     2B: 2×28×8×128×2 = 114688 B
#   32768 个 token 的 KV                      8B: 4.50 GiB                    2B: 3.50 GiB
#   开销（CUDA 上下文 + 启动时用 5 张最大图探测的激活峰值），按 3 GiB 留
#   合计                                      8B: 23.83 GiB → 0.497 → 0.50    2B: 10.46 GiB → 0.218 → 0.22
#
# 也就是 8B 要占约 24 GiB，2B 约 10.6 GiB。3 GiB 的开销是估计值，没有在显卡上实测过；留多了的部分 vLLM 会
# 拿去当 KV 缓存，留少了 vLLM 会报 KV 缓存不够而拒绝启动（不会挤占别的进程）。
#
# 32768 的上下文放得下吗：单独看放得下（8B 0.50）。但机器人推理（约 15–20 GB）和 RLT 学习进程（约 17 GB）同时
# 在跑时只剩 11 GiB 左右，8B 无论如何放不下（光权重就 16.3 GiB），只能换 2B。显存只是略紧时，先降的是
# --max-model-len（连同 --gpu-frac）：一步请求 = 约 1000 token 的系统提示词 + 每张 1280×800 矫正图 1000 token，
# 带 4 步历史也只有 6000 左右，16384 就够用，8B 可以省 2.25 GiB（16384 → 0.45），2B 省 1.75 GiB（→ 0.19）。
# 其次才是降图像像素上限（longest_edge），因为它直接影响定位精度。默认值仍保持上游的 32768。
# ------------------------------------------------------------------------------------------------------------------
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vllm_bin="${VLLM_BIN:-$here/venvs/vlm/bin/vllm}"   # VLLM_BIN=/bin/echo 可以不碰显卡地测试本脚本
model="weights/GUI-Owl-1.5-8B-Instruct"
port=8802
gpu_frac=auto
name=gui-owl
max_model_len=32768
gpu=0
force=0
dry_run=0
overhead_gib=3
extra=()

need_value() { [[ $# -ge 2 ]] || { echo "参数 $1 后面要跟一个值" >&2; exit 2; }; }
while (($#)); do
    case "$1" in
        --model) need_value "$@"; model="$2"; shift 2 ;;
        --port) need_value "$@"; port="$2"; shift 2 ;;
        --gpu-frac) need_value "$@"; gpu_frac="$2"; shift 2 ;;
        --name) need_value "$@"; name="$2"; shift 2 ;;
        --max-model-len) need_value "$@"; max_model_len="$2"; shift 2 ;;
        --gpu) need_value "$@"; gpu="$2"; shift 2 ;;
        --force) force=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h | --help) sed -n '2,/^# ----/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
        --) shift; extra=("$@"); break ;;
        *) echo "未知参数: $1（-h 看用法）" >&2; exit 2 ;;
    esac
done

# ---- 模型目录 ----
if [[ "$model" != /* ]]; then
    if [[ -d "$model" ]]; then model="$(pwd)/$model"; else model="$here/$model"; fi
fi
problems=()
[[ -f "$model/config.json" ]] || problems+=("缺 config.json")
if [[ -f "$model/model.safetensors.index.json" ]]; then
    while IFS= read -r shard; do
        [[ -f "$model/$shard" ]] || problems+=("缺权重分片 $shard")
    done < <(python3 -c 'import json,sys; print("\n".join(sorted(set(json.load(open(sys.argv[1]))["weight_map"].values()))))' \
        "$model/model.safetensors.index.json")
elif [[ ! -f "$model/model.safetensors" ]]; then
    problems+=("缺 model.safetensors")
fi
if [[ -d "$model/.cache" ]] && [[ -n "$(find "$model/.cache" -name '*.incomplete' -print -quit)" ]]; then
    problems+=("还有 *.incomplete 文件，下载没结束")
fi
if ((${#problems[@]})); then
    echo "模型目录不完整：$model" >&2
    printf '  - %s\n' "${problems[@]}" >&2
    if ((dry_run)); then echo "（--dry-run：继续往下演示；真正启动时会在这里退出）" >&2; else exit 3; fi
fi

# ---- 显存预算 ----
read -r weights_mib kv_mib < <(python3 - "$model" "$max_model_len" <<'PY'
import glob, json, os, sys
model, max_len = sys.argv[1], int(sys.argv[2])
weights = sum(os.path.getsize(p) for p in glob.glob(os.path.join(model, "*.safetensors")))
try:  # shards still downloading: fall back to the size the index declares
    index = json.load(open(os.path.join(model, "model.safetensors.index.json")))
    weights = max(weights, int(index.get("metadata", {}).get("total_size", 0)))
except OSError:
    pass
try:
    text = json.load(open(os.path.join(model, "config.json")))["text_config"]
    per_token = 2 * text["num_hidden_layers"] * text["num_key_value_heads"] * text["head_dim"] * 2  # bf16
except (OSError, KeyError):
    per_token = 0
print(round(weights / 2**20), round(per_token * max_len / 2**20))
PY
)
if ! read -r total_mib used_mib free_mib < <(nvidia-smi -i "$gpu" --query-gpu=memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits | tr -d ','); then
    echo "读不到显卡 $gpu 的显存（nvidia-smi 失败）" >&2
    exit 3
fi
need_mib=$((weights_mib + kv_mib + overhead_gib * 1024))
if [[ "$gpu_frac" == auto ]]; then
    gpu_frac="$(python3 -c 'import math,sys; print(f"{min(0.95, math.ceil(100 * int(sys.argv[1]) / int(sys.argv[2])) / 100):.2f}")' \
        "$need_mib" "$total_mib")"
    frac_note="自动"
else
    frac_note="手动指定"
fi
claim_mib="$(python3 -c 'import sys; print(round(float(sys.argv[1]) * int(sys.argv[2])))' "$gpu_frac" "$total_mib")"

echo "== 显卡 $gpu 现在的显存 =="
nvidia-smi -i "$gpu" --query-gpu=name,memory.total,memory.used,memory.free --format=csv
nvidia-smi -i "$gpu" --query-compute-apps=pid,used_memory,process_name --format=csv || true
echo "== 本次要占用的显存 =="
echo "  模型            $model"
echo "  权重            ${weights_mib} MiB"
echo "  KV 缓存下限     ${kv_mib} MiB（一条 ${max_model_len} token 的序列）"
echo "  开销预留        $((overhead_gib * 1024)) MiB"
echo "  合计需要        ${need_mib} MiB"
echo "  --gpu-frac      ${gpu_frac}（${frac_note}）→ 最多占 ${claim_mib} MiB / 共 ${total_mib} MiB"
echo "  现在空闲        ${free_mib} MiB（已用 ${used_mib} MiB）"
if ((claim_mib < need_mib)); then
    echo "注意：这个比例（${claim_mib} MiB）小于估计的需要量（${need_mib} MiB），vLLM 可能因为 KV 缓存不够而拒绝启动；" \
        "可以调大 --gpu-frac 或调小 --max-model-len。" >&2
fi
if ((free_mib < claim_mib)); then
    echo "空闲显存不够：要占 ${claim_mib} MiB，只剩 ${free_mib} MiB。显卡正被别的进程（多半是机器人推理）占着。" >&2
    echo "可以：换 2B 模型（--model weights/GUI-Owl-1.5-2B-Instruct）、调小 --gpu-frac / --max-model-len，或等别的进程结束。" >&2
    if ((force)); then
        echo "已给 --force，照样启动（可能把别的进程挤爆，后果自负）。" >&2
    elif ((dry_run)); then
        echo "（--dry-run：真正启动时会在这里拒绝，除非加 --force）" >&2
    else
        echo "拒绝启动。确实要启动就加 --force。" >&2
        exit 4
    fi
fi

# ---- 环境变量 ----
# 服务只在本机用，权重也在本地：代理全部清掉（socks:// 的 ALL_PROXY 还会让 httpx 报错），不联网、不上报使用统计。
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
export NO_PROXY="localhost,127.0.0.1" no_proxy="localhost,127.0.0.1"
export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export CUDA_VISIBLE_DEVICES="$gpu"

cmd=(
    "$vllm_bin" serve "$model"
    --served-model-name "$name"
    --host 127.0.0.1
    --port "$port"
    --max-model-len "$max_model_len"
    --mm-processor-kwargs '{"size": {"longest_edge": 3072000, "shortest_edge": 65536}}'
    --limit-mm-per-prompt '{"image": 5}'
    --allowed-local-media-path /
    --gpu-memory-utilization "$gpu_frac"
    "${extra[@]}"
)
echo "== 启动命令 =="
echo "  env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy \\"
echo "      NO_PROXY=$NO_PROXY HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 CUDA_VISIBLE_DEVICES=$gpu \\"
echo "      ${cmd[*]@Q}"
echo "  服务地址: http://127.0.0.1:${port}/v1   模型名: ${name}"
if ((dry_run)); then
    echo "（--dry-run：没有启动）"
    exit 0
fi
[[ -x "$vllm_bin" ]] || { echo "找不到 $vllm_bin，先运行 scripts/setup_vlm_env.sh" >&2; exit 3; }
exec "${cmd[@]}"
