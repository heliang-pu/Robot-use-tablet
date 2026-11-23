#!/usr/bin/env bash
# 建立 / 更新 GUI-Owl-1.5 的推理环境 venvs/vlm（vLLM 服务 + CPU 冒烟测试共用）。可重复执行。
#
# 用法:  bash scripts/setup_vlm_env.sh                  # 建环境、装包、打印并记录解析到的版本
#        bash scripts/setup_vlm_env.sh --check          # 只打印现有环境的版本，不装任何东西
#        bash scripts/setup_vlm_env.sh --with-kernels   # 另外装 FlashInfer 预编译内核（很大，见下）
#
# 版本怎么定的（2026-11-23 查的）：
#   - 模型是 Qwen3VLForConditionalGeneration（model_type qwen3_vl，config.json 写的 transformers 4.57.1）。
#   - vLLM 从 v0.11.0 起支持 "Qwen3-VL series"（v0.11.0 release notes），现在的 supported_models 表里仍有
#     `Qwen3VLForConditionalGeneration`（T + I + V）。PyPI 最新版是 0.34.0（2026-11-16），它固定
#     torch==2.14.0、transformers>=5.16.1,<5.23，Python 3.11–3.14。
#   - Python 取 3.12：vLLM 文档的安装示例用的就是 3.12，本机 /usr/bin/python3.12 带头文件。
#   - `--torch-backend=auto` 让 uv 按本机驱动（nvidia-smi，580.178.04）选 PyTorch 的 CUDA 轮子，本机选到
#     cu132。CUDA 13.2 配套的驱动分支是 R595，本机是 R580；NVIDIA 的说法是 13.x 编出来的程序在 >= 580 的驱动
#     上都能跑（小版本兼容，13.0 之后的新特性不可用）。这一点没在显卡上验证过；万一 torch 起不来，
#     换成和 R580 配套的 CUDA 13.0：  TORCH_BACKEND=cu130 bash scripts/setup_vlm_env.sh
#     （torch 2.14.0 的 cp312 轮子只有 cu130 和 cu132 两种，没有 cu128 / cu129）。
#   - `vllm download-kernels`（vLLM 安装文档里紧跟在 pip install 后面的一步，装 FlashInfer 的预编译内核
#     flashinfer-cubin + flashinfer-jit-cache）默认不跑：flashinfer-cubin 的轮子 1.76 GB，解开超过 7 GB，
#     环境会从 9 GB 涨到 16 GB 以上（2026-11-23 试装到 7.2 GB 时中止了）。不装时 vLLM 会提示
#     "FlashInfer kernels are disabled"（本机没有 nvcc / ninja，也没法现场编译），改用不依赖 FlashInfer 的实现。
#     服务在显卡上到底要不要它，没验证过；需要时用 --with-kernels 补装。
#   - 可用环境变量覆盖：VLLM_VERSION、PY、TORCH_BACKEND、UV。
#
# 这个脚本不碰 GPU：所有 import torch / vllm 的命令都带 CUDA_VISIBLE_DEVICES=（空）。
# 解析到的全部版本写在 venvs/vlm/resolved_versions.txt（uv pip freeze）和本文件末尾的"上次结果"注释里。
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="$here/venvs/vlm"
UV="${UV:-/home/phl/.local/bin/uv}"
VLLM_VERSION="${VLLM_VERSION:-0.34.0}"
PY="${PY:-3.12}"
TORCH_BACKEND="${TORCH_BACKEND:-auto}"

unset ALL_PROXY all_proxy          # socks:// 的值会让 httpx / hf 出错；HTTP(S)_PROXY 保留
export UV_LINK_MODE=copy           # uv 缓存和工作区不在同一块盘，硬链接用不了

report() {
    echo "== 解析到的版本 =="
    CUDA_VISIBLE_DEVICES='' "$venv/bin/python" - <<'PY'
import importlib.metadata as md
import platform

import torch

print(f"{'python':<22} {platform.python_version()}")
for name in ("vllm", "torch", "torchvision", "transformers", "tokenizers", "accelerate", "qwen-vl-utils",
             "openai", "pillow", "opencv-python-headless", "numpy", "flashinfer-python", "flashinfer-cubin",
             "flashinfer-jit-cache"):
    try:
        print(f"{name:<22} {md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name:<22} 未安装")
print(f"torch CUDA 构建  {torch.version.cuda}   (cuda.is_available={torch.cuda.is_available()}，这里故意屏蔽了 GPU)")
PY
    echo "== vllm 命令行 =="
    CUDA_VISIBLE_DEVICES='' "$venv/bin/vllm" --version 2>&1 | grep -v -E '^W[0-9]+ .*deprecated'
    echo "== 驱动 =="
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
    echo "== 环境大小 =="
    du -sh "$venv" | cut -f1
}

with_kernels=0
case "${1:-}" in
    --check)
        [[ -x "$venv/bin/python" ]] || { echo "还没有 $venv，先不带参数运行本脚本" >&2; exit 1; }
        report
        exit 0
        ;;
    --with-kernels) with_kernels=1 ;;
    "") ;;
    *) echo "未知参数: $1（可用：--check、--with-kernels）" >&2; exit 2 ;;
esac

if [[ ! -x "$venv/bin/python" ]]; then
    echo "== 建 venv: $venv (Python $PY) =="
    "$UV" venv --python "$PY" --seed "$venv"
fi

echo "== 装 vllm==$VLLM_VERSION（torch 后端: $TORCH_BACKEND）和冒烟测试要用的包 =="
# vllm 自带 torch / transformers / tokenizers / pillow / opencv-python-headless / openai / numpy；
# 另外装的：qwen-vl-utils（上游 utils.py import 它）、accelerate（transformers 加载模型用）。
"$UV" pip install --python "$venv/bin/python" --torch-backend="$TORCH_BACKEND" \
    "vllm==$VLLM_VERSION" qwen-vl-utils accelerate

if ((with_kernels)); then
    echo "== 装 FlashInfer 预编译内核（vllm download-kernels，下载 1.76 GB，解开 7 GB 以上）=="
    CUDA_VISIBLE_DEVICES='' "$venv/bin/vllm" download-kernels
fi

"$UV" pip freeze --python "$venv/bin/python" > "$venv/resolved_versions.txt"
echo "全部包的版本已写入 $venv/resolved_versions.txt"
report


# ---- 上次结果（2026-11-23，本机：RTX 4090 48 GB，驱动 580.178.04；没带 --with-kernels）----
#   python                 3.12.3
#   vllm                   0.34.0
#   torch                  2.14.0+cu132      （CUDA 13.2 构建；torchvision 0.29.0+cu132）
#   transformers           5.22.2
#   tokenizers             0.23.3
#   accelerate             1.16.0
#   qwen-vl-utils          0.0.14
#   openai                 3.57.0
#   pillow                 13.0.0
#   opencv-python-headless 5.1.0.95
#   numpy                  2.3.5
#   flashinfer-python      0.7.3             （flashinfer-cubin / flashinfer-jit-cache 未安装）
#   环境大小               9.0 GB
