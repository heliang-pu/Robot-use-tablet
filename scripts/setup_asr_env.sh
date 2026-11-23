#!/usr/bin/env bash
# Build venvs/asr: the Python environment of tablet_tap's speech-recognition service (scripts/serve_asr.sh):
# FunASR + CPU-only torch/torchaudio + flask. Nothing is installed anywhere else.
#
#   scripts/setup_asr_env.sh        # needs the network (PyPI and download.pytorch.org, through HTTP(S)_PROXY)
#
# Re-runnable: an existing venv is kept and brought to the pinned versions below. ASR_VENV=<dir> builds elsewhere.
#
# How the environment was first made (2026-11-23, uv 0.12.5, CPython 3.11.14):
#   uv venv venvs/asr --python 3.11
#   uv pip install --python venvs/asr/bin/python --index-url https://download.pytorch.org/whl/cpu \
#       torch==2.7.1 torchaudio==2.7.1
#   uv pip install --python venvs/asr/bin/python --link-mode=copy torch==2.7.1+cpu torchaudio==2.7.1+cpu \
#       funasr flask "numpy<=1.26.4" \
#       --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
# The pins below are what those commands resolved to (uv pip freeze), so a rebuild gives the same environment.
#   torch / torchaudio 2.7.1+cpu: CPU wheels (no CUDA libraries are downloaded); a conservative version, newer ones
#       were not tried.
#   numpy<=1.26.4: from weights/SenseVoiceSmall/requirements.txt.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="${ASR_VENV:-$here/venvs/asr}"
uv="${UV:-$(command -v uv || echo /home/phl/.local/bin/uv)}"
if [ ! -x "$uv" ]; then
    echo "没有找到 uv（试过 $uv）：请先安装 uv，或用 UV=<路径> 指定" >&2
    exit 1
fi
unset ALL_PROXY all_proxy  # the socks:// value there breaks HTTP clients; HTTP(S)_PROXY stays

if [ ! -x "$venv/bin/python" ]; then
    "$uv" venv "$venv" --python 3.11
fi
py="$venv/bin/python"

# 1. torch and torchaudio alone from the PyTorch CPU index (their dependencies come from PyPI in step 2).
"$uv" pip install --python "$py" --link-mode=copy --no-deps \
    --index-url https://download.pytorch.org/whl/cpu "torch==2.7.1+cpu" "torchaudio==2.7.1+cpu"

# 2. Everything else from PyPI, at the versions resolved on 2026-11-23.
pins="$(mktemp)"
trap 'rm -f "$pins"' EXIT
cat > "$pins" <<'PINS'
aliyun-python-sdk-core==2.16.1
aliyun-python-sdk-kms==2.16.5
annotated-doc==0.0.7
annotated-types==0.8.0
antlr4-python3-runtime==4.9.3
anyio==4.16.0
audioread==3.1.0
blinker==1.9.0
certifi==2026.11.20
cffi==2.1.1
charset-normalizer==3.5.3
click==8.5.1
cloudpickle==3.1.3
crcmod==1.7
cryptography==50.0.2
decorator==5.3.2
filelock==4.2.4
flask==3.1.3
fsspec==2026.9.0
funasr==1.4.28
h11==0.16.0
hf-xet==1.8.0
httpcore2==2.16.2
httpx2==2.16.2
huggingface-hub==2.4.0
hydra-core==1.3.8
idna==3.20
itsdangerous==2.2.0
jaconv==0.5.0
jamo==0.4.1
jieba==0.42.1
jinja2==3.1.6
jmespath==1.1.0
joblib==1.6.0
kaldiio==2.18.1
lazy-loader==0.6
librosa==0.11.0
llvmlite==0.50.0
markdown-it-py==4.3.0
markupsafe==3.0.3
mdurl==0.1.2
modelscope==1.40.3
modelscope-hub==0.7.1
mpmath==1.3.0
msgpack==1.2.4
narwhals==2.29.0
networkx==3.6.1
numba==0.68.1
numpy==1.26.4
omegaconf==2.3.1
oss2==2.19.1
packaging==26.3
platformdirs==4.14.8
pooch==1.9.0
protobuf==7.36.2
pycparser==3.12
pycryptodome==4.6.0
pydantic==2.14.4
pydantic-core==2.50.5
pygments==2.21.0
pynndescent==0.7.0
pyyaml==6.0.4
rapidfuzz==3.14.6
regex==2026.11.19
requests==2.34.2
rich==15.0.0
safetensors==0.8.0
scikit-learn==1.9.1
scipy==1.17.1
sentencepiece==0.2.3
setuptools==84.2.1
shellingham==1.5.4
six==1.17.0
soundfile==0.14.0
soxr==1.1.0
sympy==1.14.0
tensorboardx==2.6.5
threadpoolctl==3.8.0
tiktoken==0.14.0
tokenizers==0.23.3
torch-complex==0.4.4
tqdm==4.70.1
transformers==5.24.0
truststore==0.10.5
typer==0.29.2
typing-extensions==4.16.0
typing-inspection==0.4.5
umap-learn==0.5.12
urllib3==2.8.0
websockets==17.4
werkzeug==3.1.9
PINS
"$uv" pip install --python "$py" --link-mode=copy -r "$pins"

# 3. Check: the imports work and torch is the CPU build.
CUDA_VISIBLE_DEVICES= "$py" - <<'PY'
import flask, funasr, numpy, soundfile, torch, torchaudio

assert torch.__version__.endswith("+cpu") and not torch.cuda.is_available(), torch.__version__
print(
    f"语音识别环境就绪：funasr {funasr.__version__}，torch {torch.__version__}，torchaudio {torchaudio.__version__}，"
    f"numpy {numpy.__version__}，flask {flask.__version__}，soundfile {soundfile.__version__}"
    f"（libsndfile {soundfile.__libsndfile_version__}）"
)
PY
echo "下一步：scripts/serve_asr.sh"
