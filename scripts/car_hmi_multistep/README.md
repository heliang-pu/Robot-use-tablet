# 理想 L8 车机纯视觉多步长任务范式与 MAI-Gym-9B 联合微调（v2）

本分支（`maigym-multistep`）在 `car-hmi-mock` 车机仿真器之上，构建了一套面向机械臂电容笔纯视觉（零 ADB、零 DOM 依赖）操作的**原生端到端多步视觉代理范式与全量 BF16 LoRA 联合微调流水线**。

---

## 一、核心范式设计

### 1. 原生端到端多步视觉闭环（`MaiGymMultiStepAgent`）
对应模块：[tablet_tap/agent/maigym_agent.py](../../tablet_tap/agent/maigym_agent.py)（直接实现 `tablet_tap.agent.device.Device` 协议，可在 Playwright 仿真器与 PiPER 真机机械臂间零改动切换）。

- **零外部 Planner / 零硬编码页面树**：由单一 `MAI-Gym-9B` 模型直接观察实时截图，自主完成跨页导航、多步步进调节、循环档位收敛、列表滚动查找与任务终止。
- **滑动窗口双图上下文（Last-2 Images）**：
  - 较早步骤的截图自动折叠为 `[历史截图]`，每次推理最多携带 **2 张真实图像（上一帧 + 当前帧）**，将上下文稳定在 `2700~3000 tokens`，显存与推理延迟可控；
  - 保留上一帧与当前帧对比，使模型原生具备**物理漏点自愈**（点击前后画面一致时自动重试）与**列表滚底拒答**（`drag` 后画面无变化则输出 `terminate(status="failed")`）。
- **极简一行内生状态账本（绕过 Qwen3.5 历史 `<think>` 剥离）**：
  - Qwen3.5 / `MAI-Gym-9B` 的 Jinja `chat_template` 会自动剥离历史轮中的 `<think>...</think>`；
  - 我们保持 `enable_thinking=False`（单步 `~1.5s`），并让模型在每步 `<tool_call>` 之前输出 15~35 字的 `状态：已完成[...]；在XX页，点XX`，该行会被完整保留在多轮上下文中，天然充当跨 3~4 页长任务的进度记忆。
- **官方动作空间子集**：100% 沿用 `Tongyi-MAI/MAI-Gym` 官方 System Prompt，通过机械臂注记 `MULTISTEP_NOTE` 将动作严格约束在 `click`、`drag`、`ask_user`（高危零点击拦截）、`terminate`（`success` / `failed`）。

### 2. 三任务联合无泄露微调数据集 v2（`build_dataset_multistep_v2.py`）
通过多进程 Playwright 在真实 `android/car_hmi` 页面中执行程序化多步 Oracle 轨迹并混入单步强化与视觉读屏（全部 50% Clean + 50% 模拟腕部相机仿射/光照退化）：

1. **`multistep_e2e`（端到端多步轨迹，2,873 步 / 438 条轨迹）**：
   - 覆盖任意随机起始页（桌面、空调、设置深层页、正在导航的 `nav_guide`、音乐、电话、深色模式、来电浮窗遮挡）；
   - 覆盖幂等开关（已开不反手关掉）、温度/风量多步步进、座椅 4 档循环、滑条定位、导航退出重规划、滚动通讯录找人；
   - **主动注入 14% 物理漏点重试样本**（模拟机械臂没点实后看双图无变化自主重试）；
   - 包含不存在联系人滚底拒答（`terminate:failed`）与高危操作零点击拦截（`ask_user`）；
   - **严格留出未见页面**：`安全/哨兵`、`理想同学/音色`、`电话/最近通话` 绝不进入训练轨迹，用于检验 `T16` / `C09` 的真实跨页泛化能力。
2. **`single_grounding`（单步精确定位强化，2,800 条）**：
   - 抽样自单步训练集（同样严格排除 4 个留出测试页），并对 v1 剩余错题（氛围灯色块、前排吹面/吹脚、座椅通风/放倒、前/后除霜、深色音乐入口）执行 **5× 定向过采样**。
3. **`visual_readout`（真值结构化视觉读屏，2,400 条）**：
   - 直接从 `window.hmi.S` 生成页面状态、细线条开关（如前除霜）、温度/风量数值以及区分橙色加热 vs 蓝色通风的座椅档位 JSON 问答，让**单一微调模型**同时胜任端到端 E2E 与反应式闭环（`ReactiveCarAgent`）。

### 3. 全量 BF16 LoRA 多图多轮联合微调（`train_maigym_multistep_lora.py`）
- 基座：`Tongyi-MAI/MAI-Gym-9B`（`Qwen3_5ForConditionalGeneration`，开启 `flash-linear-attention` + `fla-core` Gated DeltaNet 加速）；
- 全精度 BF16 LoRA：`r=32, alpha=64, dropout=0.05`，覆盖语言主干全部 12 类线性层（Gated Attention + Gated DeltaNet + FFN）及 `visual.merger.linear_fc1/2`（87.1M 可训练参数，占 `0.9175%`）；
- 精确计算最终助手回复 token 的交叉熵损失（屏蔽 system / 历史轮 / 图像占位），按 `e2e / ground / readout` 分项监控验证集收敛，结束后在 CPU 端完成 BF16 `merge_and_unload` 以便 vLLM 零开销部署。

---

## 二、目录结构

```text
tablet_tap/agent/
  └── maigym_agent.py                # 生产级端到端多步视觉代理（兼容真机 RobotDevice）
tests/
  └── test_maigym_agent.py           # 无网络/无 GPU 单元测试（滑动窗口、解析、安全护栏）
scripts/car_hmi_multistep/
  ├── build_dataset_multistep_v2.py  # 三任务联合多步数据集 v2 生成器（Playwright 多进程）
  ├── train_maigym_multistep_lora.py # 多图多轮全量 BF16 LoRA 微调 + CPU BF16 Merge
  ├── bench_e2e_multistep.py         # 30 项原生端到端多步闭环基准（T01-T20 + C01-C10）
  ├── car_reactive_agent.py          # 对照方案：反应式目标账本闭环代理（含局部 ROI 防漏点）
  ├── bench_reactive_all.py          # 30 项反应式闭环评测器
  ├── bench_long_tasks.py            # 20 项标准跨页长任务定义与 Playwright Device 封装
  ├── run_bench_maigym.py            # 392 控件单步定位基准（含 4 个留出页，Clean & Camera）
  └── build_dataset_maigym.py        # v1 单步微调数据构建脚本
```

---

## 三、快速复现命令

### 1. 生成多步联合数据集 v2
```bash
python scripts/car_hmi_multistep/build_dataset_multistep_v2.py \
  /mnt/ymtc2t/datasets/carhmi_maigym_v2 \
  --v1-dir /mnt/ymtc2t/datasets/carhmi_maigym_v1 \
  --train-eps 480 --val-eps 36 --workers 4
```

### 2. 单张 RTX 4090（48 GB）全量 BF16 LoRA 多步联合微调
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u scripts/car_hmi_multistep/train_maigym_multistep_lora.py \
  --model /mnt/ymtc2t/models/MAI-Gym-9B \
  --data-dir /mnt/ymtc2t/datasets/carhmi_maigym_v2 \
  --out-dir /mnt/ymtc2t/outputs/maigym9b_carhmi_v2 \
  --merged-dir /mnt/ymtc2t/models/MAI-Gym-9B-carhmi-v2 \
  --epochs 0.6 --grad-accum 16 --lr 1.2e-4 --lora-r 32 --lora-alpha 64 --eval-steps 50 --bf16
```

### 3. 启动 vLLM 推理服务（支持双图滑动窗口）
```bash
vllm serve /mnt/ymtc2t/models/MAI-Gym-9B-carhmi-v2 \
  --served-model-name mai-gym-9b-v2 \
  --host 127.0.0.1 --port 18809 --dtype bfloat16 \
  --gpu-memory-utilization 0.55 --max-model-len 12288 \
  --enforce-eager \
  --mm-processor-kwargs '{"size":{"longest_edge":3072000,"shortest_edge":65536}}' \
  --limit-mm-per-prompt '{"image":2}'
```

### 4. 运行 30 项端到端多步闭环基准（Clean / Camera）
```bash
python scripts/car_hmi_multistep/bench_e2e_multistep.py \
  --base-url http://127.0.0.1:18809/v1 --model mai-gym-9b-v2 \
  --suite all --cond clean --label e2e_maigym9b_v2_clean

python scripts/car_hmi_multistep/bench_e2e_multistep.py \
  --base-url http://127.0.0.1:18809/v1 --model mai-gym-9b-v2 \
  --suite all --cond camera --label e2e_maigym9b_v2_camera
```

---

## 四、阶段性实测数据（持续更新）

### 单步 392 控件基准（24 页，含 4 个留出未见页）
| 模型 | Clean (392) | Camera (392) | 纯图标 (122) | 空调页 (90) | 4 留出页 (31) | 单次耗时 |
|---|---:|---:|---:|---:|---:|---:|
| `GUI-Owl-1.5-8B` | 81% | 80% | 48%–50% | 47%–50% | — | 1.5 s |
| `GUI-Owl-1.5-32B` | 84% | 86% | 61%–62% | 70%–73% | — | 3.8 s |
| `MAI-Gym-9B` 零样本 | 84.7% | 85.2% | 61%–62% | 60%–62% | 93.5% | 1.9 s |
| `MAI-Gym-9B-carhmi-v1`（单步 BF16 LoRA 60 步） | **96.9%** | **97.4%** | **90%–92%** | **91%–93%** | **100.0%** | **1.8 s** |

### 30 项端到端多步基准（20 标准长任务 `T01–T20` + 10 真实边界场景 `C01–C10`）
| 模型 / 模式 | 画质 | 总通过率 (30) | 标准长任务 (20) | 边界场景 (10) |
|---|---|---:|---:|---:|
| `MAI-Gym-9B` 零样本 E2E（`no-think`） | Clean | 15 / 30 (50.0%) | 10 / 20 (50.0%) | 5 / 10 (50.0%) |
| `MAI-Gym-9B` 零样本 E2E（`no-think`） | Camera | 14 / 30 (46.7%) | 10 / 20 (50.0%) | 4 / 10 (40.0%) |
| `MAI-Gym-9B-carhmi-v2`（多步联合 BF16 LoRA） | Clean / Camera | *训练与评测进行中，完成后追加更新* | — | — |
