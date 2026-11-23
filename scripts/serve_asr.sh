#!/usr/bin/env bash
# Start tablet_tap's local speech-recognition service (SenseVoiceSmall + fsmn-vad, FunASR, CPU only) on 127.0.0.1.
#
#   scripts/serve_asr.sh [--port 8803] [--threads 4] [--no-vad]
#
# Runs in venvs/asr (built by scripts/setup_asr_env.sh). Start-up needs no network: the models are read from
# weights/, FunASR's update check is off and the hub libraries are put in offline mode. The GPU is never used.
# Client: bin/py -m tablet_tap.speech.voice    Check: curl --noproxy '*' http://127.0.0.1:8803/health
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="$here/venvs/asr/bin/python"
if [ ! -x "$py" ]; then
    echo "没有找到 $py：请先运行 scripts/setup_asr_env.sh 建好语音识别环境" >&2
    exit 1
fi
for d in SenseVoiceSmall fsmn-vad; do
    if [ ! -f "$here/weights/$d/model.pt" ]; then
        echo "没有找到模型文件 $here/weights/$d/model.pt" >&2
        exit 1
    fi
done
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
export CUDA_VISIBLE_DEVICES=
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
cd "$here"
exec "$py" -m tablet_tap.speech.asr_service \
    --model-dir "$here/weights/SenseVoiceSmall" --vad-dir "$here/weights/fsmn-vad" "$@"
