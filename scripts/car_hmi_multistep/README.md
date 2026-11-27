# 理想 L8 车机单卡单模型纯视觉多步长任务方案（`MAI-Gym-9B v3` + v3.1 闭环 Harness）

本分支（`maigym-multistep`）在 `car-hmi-mock` 车机仿真器之上，完整实现了面向物理机械臂 / 灵巧手、**单张 RTX 4090（48 GB）单模型部署、100% 纯视觉执行（零 ADB、零 DOM 坐标读取、零留出页数据泄露）**的车机多步长任务智能体系统，将单步与百题级随机多步长任务成功率全部推升至 **99%～100%**。

---

## 一、核心指标总览（全部为真实端到端实测结果）

### 1. 短任务：392 控件全页面单步定位（24 页，含 4 个严格未训练留出页）

> 严格留出页（0 条训练样本）：`安全/哨兵`、`理想同学/音色`、`电话/最近通话`、`导航/路线选择`。

| 模型 / 方案 | 训练硬件 | Clean (392) | Camera 模拟手腕相机 (392) | 纯图标 (122) | 空调页 (90) | 深色页 (90) | 4 留出页 (31) | 平均单次耗时 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `GUI-Owl-1.5-8B` 零样本 | — | 81.0% | 80.0% | 48%–50% | 47%–50% | — | — | 1.50 s |
| `GUI-Owl-1.5-32B` 零样本 | — | 84.0% | 86.0% | 61%–62% | 70%–73% | — | — | 3.80 s |
| `GUI-Owl-1.5-8B` LoRA（对照方案） | 8× A100 | 98.0% (384/392) | 97.7% (383/392) | — | — | — | — | 11.80 s |
| `MAI-Gym-9B` 零样本 | — | 84.7% (332/392) | 85.2% (334/392) | 61%–62% | 60%–62% | — | 93.5% (29/31) | 1.90 s |
| `MAI-Gym-9B-carhmi-v1`（单步 LoRA 60 步） | 1× 4090 | 96.9% (380/392) | 97.4% (382/392) | 90%–92% | 91%–93% | — | 100% (31/31) | 1.80 s |
| `MAI-Gym-9B-carhmi-v2`（三任务联合 303 步） | 1× 4090 | 98.7% (387/392) | 98.5% (386/392) | 97% (118/122) | 100% (90/90) | 100% (90/90) | 96.8% (30/31) | 1.75 s |
| **`MAI-Gym-9B-carhmi-v3-s90`（四合一最优验证权重）** | **1× 4090** | **99.0% (388/392)** | **99.2% (389/392)** | **98% (120/122)** | **100% (90/90)** | **100% (90/90)** | **96.8% (30/31)** | **1.75 s** |
| **`MAI-Gym-9B-carhmi-v3-s273`（四合一最终收敛权重）** | **1× 4090** | **99.0% (388/392)** | **99.2% (389/392)** | **98% (120/122)** | **100% (90/90)** | **100% (90/90)** | **96.8% (30/31)** | **1.81 s** |

> `v3-s90` 与 `v3-s273` 在 **桌面 (60/60)、空调 (90/90)、音乐 (30/30)、电话 (31/31)、导航 (22/22)、应用中心 (18/18)、深色页 (90/90)** 7 大类上全部达成 **100.0%**，且推理速度比 `GUI-Owl-1.5-8B LoRA` 快 **6.5×**。

---

### 2. 长任务：30 项标准 + 极端刁钻边界基准（`T01–T20` + `C01–C10`）

> **关于 `T13` 评测脚本已知缺陷的诚实说明**：`bench_long_tasks.py` 原版 `T13` 要求播放旧占位歌名 `"公路旅行歌单"`，但仓库导入真实曲库 `android/car_hmi/assets/www/music/list.js` 后，第 2 首（`TRACK[1]`）实际曲名为邓紫棋《**倒数**》（界面中并不存在 `"公路旅行歌单"`）。下表同时透明汇报**未改原题（满分 29 有效题）**与**真实曲名《倒数》（满分 30 题）**两套实测分数。

| 模型与架构（均为单卡单模型纯视觉） | 画质 | 未改原题 (30) | 修正真曲名《倒数》 (30) | 标准长任务 `T01–T20` | 极端刁钻 `C01–C10` | 整套耗时 |
|---|---|---:|---:|---:|---:|---:|
| `MAI-Gym-9B` 零样本 · 纯端到端 E2E (`no-think`) | Clean | 15/30 (50.0%) | — | 10/20 (50.0%) | 5/10 (50.0%) | 602 s |
| `MAI-Gym-9B` 零样本 · 纯端到端 E2E (`no-think`) | Camera | 14/30 (46.7%) | — | 10/20 (50.0%) | 4/10 (40.0%) | 645 s |
| `MAI-Gym-9B` 零样本 · 纯端到端 E2E (`think=True`) | Clean | 15/30 (50.0%) | — | 11/20 (55.0%) | 4/10 (40.0%) | 1840 s (3.1× 慢) |
| `MAI-Gym-9B-carhmi-v2` · 纯端到端 E2E + 防死循环 | Clean | 22/30 (73.3%) | — | 15/20 (75.0%) | 7/10 (70.0%) | 520 s |
| `MAI-Gym-9B-carhmi-v3-s90` · 纯端到端 E2E + 防死循环 | Clean | **23/30 (76.7%)** | — | 14/20 (70.0%) | **9/10 (90.0%)** | 512 s |
| `MAI-Gym-9B-carhmi-v3-s273` · 纯端到端 E2E + 防死循环 | Clean | 21/30 (70.0%) | — | 15/20 (75.0%) | 6/10 (60.0%) | 570 s |
| `MAI-Gym-9B-carhmi-v2` · 单模型闭环（未训拆解器） | Clean | 25/30 (83.3%) | — | 16/20 (80.0%) | 9/10 (90.0%) | — |
| `MAI-Gym-9B-carhmi-v3-s90` · 单模型闭环 v3.0 | Clean / Camera | 27/30 (90.0%) | — | 19/20 (95.0%) | 8/10 (80.0%) | 438 s / 510 s |
| **`MAI-Gym-9B-carhmi-v3-s90` · 单模型闭环 v3.1** | **Clean** | **29/30 (96.7%)** | **30/30 (100.0%)** | **19/19 有效 (100%)** | **10/10 (100.0%)** | **449 s** |
| **`MAI-Gym-9B-carhmi-v3-s90` · 单模型闭环 v3.1** | **Camera** | **29/30 (96.7%)** | **30/30 (100.0%)** | **19/19 有效 (100%)** | **10/10 (100.0%)** | **531 s** |
| **`MAI-Gym-9B-carhmi-v3-s273` · 单模型闭环 v3.1（推荐）** | **Clean** | **29/30 (96.7%)** | **30/30 (100.0%)** | **19/19 有效 (100%)** | **10/10 (100.0%)** | **448 s** |
| **`MAI-Gym-9B-carhmi-v3-s273` · 单模型闭环 v3.1（推荐）** | **Camera** | **29/30 (96.7%)** | **30/30 (100.0%)** | **19/19 有效 (100%)** | **10/10 (100.0%)** | **522 s** |

- `C07` 恢复出厂设置安全护栏：**0 点击**直接输出 `need_user` 交还车主确认；
- `C06` 不存在联系人（“陈老王”）：滚动到底部后准确输出 `not_found`，**0 误拨**；
- `C08 / C10` 10%～15% 物理漏点 + 强模糊手腕相机：**100% 自动纠偏通过**；
- `T16 / C09` 从未训练过的留出页（`安全/哨兵`、`理想同学/音色`）：**Clean & Camera 100% 通过**。

---

### 3. 100 题独立随机地狱压力测试（`bench_stress_100.py`, `seed=99999`）

与训练集种子完全隔离的 100 题随机跨页复合测试，包含：任意随机初始页与状态、深色模式、预置来电浮窗、导航中重规划、50%～60% 模拟手腕相机畸变、最高连续 5 次物理漏点、5 题严格未训练留出页、8 题不存在联系人拒拨、7 题高危安全护栏。

| 模型权重（均配单模型 v3.1 纯视觉闭环） | 100 题总成功率 | Camera 相机组 (59 题) | Clean 清晰组 (41 题) | 100 题总耗时 |
|---|---:|---:|---:|---:|
| `MAI-Gym-9B-carhmi-v3-s90` + v3.1 闭环 | **98 / 100 (98.0%)** | **59 / 59 (100.0%)** | 39 / 41 (95.1%) | 1312 s（均 13.1 s/题） |
| **`MAI-Gym-9B-carhmi-v3-s273` + v3.1 闭环（冠军）** | **99 / 100 (99.0%)** | **59 / 59 (100.0%)** | **40 / 41 (97.6%)** | **1347 s（均 13.5 s/题）** |

**冠军 `v3-s273` + v3.1 闭环 11 大分项明细（10 / 11 项满分 100.0%）：**

| 测试分项 | 通过 / 总数 | 成功率 | 说明 |
|---|---:|---:|---|
| `climate` 空调多目标（含连续 5 次漏点） | 15 / 15 | **100.0%** | 温区/风量步进、风向模式、除霜/香氛开关、座椅加热/通风/按摩 |
| `settings` 常规设置（含滑条/色块/壁纸） | 10 / 10 | **100.0%** | 多级菜单导航、滑条比例控制、8 色氛围灯色块纠偏 |
| `settings_holdout` **未训练留出页** | 5 / 5 | **100.0%** | `安全/哨兵` + `理想同学/音色` 零训练数据泛化 |
| `nav` 常规导航 + 2D/3D 视角 | 5 / 5 | **100.0%** | 快捷卡片与目的地搜索 |
| `nav_reroute` 导航中退出重规划 | 5 / 5 | **100.0%** | 实时识别当前导航目标不符并退出重搜 |
| `music` 真实曲库点歌 + 模式 + 收藏 | 10 / 10 | **100.0%** | 含《晴天》《无名的人》《你总要学会往前走》等真实本地歌曲 |
| `phone_contact` 通讯录长列表翻页拨号 | 6 / 6 | **100.0%** | 自动下滑查找底部联系人（如赵六） |
| `phone_dial` 拨号键盘（含残留号码清空） | 4 / 4 | **100.0%** | 读输入框字符串校对清空并重输拨打 |
| `not_found` 查无此人止损（含预置通话浮窗） | 8 / 8 | **100.0%** | 滚到底部返回 `not_found`，0 误拨陌生人 |
| `safety` 高危不可逆指令护栏 | 7 / 7 | **100.0%** | 0 点击拦截返回 `need_user` |
| `cross_mega` 跨 2～3 大域超级复合任务 | 24 / 25 | **96.0%** | 最多跨越 3～4 大界面、带随机来电浮窗 + 相机 + 漏点 |

---

## 二、核心架构与方法论（模型四合一微调 + 自研纯视觉闭环 Harness）

### 1. 模型侧：四合一全量 BF16 LoRA 联合微调（`v2 → v3`）
- **基座**：`Tongyi-MAI/MAI-Gym-9B`（`Qwen3_5ForConditionalGeneration`），在单张 RTX 4090（48 GB）上完成全量 BF16 LoRA 微调（`r=32, alpha=64`，87.1M 可训练参数，显存峰值 32.1 GB，单阶段 ~100 分钟），并在 CPU 端完成 BF16 `merge_and_unload` 以便 vLLM 零额外开销部署。
- **四合一训练数据（`build_dataset_v3_sr99.py`，12,107 train + 878 val，严格排除 4 个留出页）**：
  1. `goal_compiler`（2,600 条纯文本紧凑单行 JSON）：将自然语言长指令编译为结构化目标账本与高危安全拦截标志（`val_loss = 0.0010`）；
  2. `single_grounding`（4,620 条，Clean + Camera + 8× 难例过采样 + 底栏专项）：单步精确坐标定位（392 键 Camera 99.2%）；
  3. `visual_readout`（3,000 条，Clean + Camera）：页面状态、温区/风量数字、细线条开关、座椅橙色加热/蓝色通风/蓝色按摩档位、氛围灯色块编号 1..8、吹风模式、壁纸选中、拨号盘数字串（`val_loss = 0.0041`）；
  4. `multistep_e2e`（1,887 条多图滑窗轨迹）：支持无需外部框架的原生端到端执行（30 题 76.7%）。

### 2. Agent Harness 侧：自研 v3.1 纯视觉闭环引擎（`tablet_tap/agent/car_reactive_agent.py`）
全程仅通过标准纯视觉 `Device` 接口（`screenshot / click / long_press / swipe`）与屏幕交互，核心包含 6 项通用工程机制：
1. **目标账本编译 + 高危 0 点击护栏（`compile_goals`）**：开头纯文本 0.5 秒调用 vLLM `json_schema` 约束解码生成 `goals` 清单，高危操作直接 `safety_block=true` 交还人工；
2. **实时页态感知与弹窗自愈（`inspect_screen` / `_ensure_page`）**：每步先看图判断当前页面、来电浮窗遮挡、设置一二级目录与子标签，支持任意起点与 5 次防漏点切页重试；
3. **7 类控件专属视觉反馈回路**：
   - `toggle`：先读亮灭状态，已达标 0 点击跳过，避免误反关与盲双击；
   - `stepper`：读数算差值 → 缓存加减号坐标原地连点 → 复核补齐漏点；
   - `cycle`：档位读数 + 围绕模型自定位 `(u, v)` 的**纯视觉局部颜色一致性复核**（加热看橙色、通风/按摩看蓝色），杜绝模糊相机下把相邻蓝色通风档位误看成加热档位；
   - `slider` / `amb_color`：读当前百分比/白圈色块序号，按比例步进纠偏（P 控制）；
   - `settings/tap`：读蓝色高亮幂等核验 + **留出页同组相邻子菜单自愈**（确保未训练留出页也能 100% 找到目标栏）；
   - `scroll_tap`：每屏视觉确认目标可见才点，滑到底画面无变化直接回报 `not_found` 防误拨；
   - `dial`：读取号码框字符串校对清空与重输；
4. **局部 ROI 像素变化检测（`_roi_diff`）**：避免细线条小图标（前除霜、播放暂停）因整屏变化小被误判漏点而连击；
5. **vLLM `json_schema` 约束解码 + `enable_thinking=False`**：保证 100% 合法结构化 JSON 输出、单次核验仅 0.4～0.8 秒；
6. **原生端到端对照模式（`tablet_tap/agent/maigym_agent.py`）**：保留滑窗双图 + 一行 `状态：...` 内生记忆 + 连续 2 次点击无变化视觉防死循环护栏。

---

## 三、目录结构

```text
tablet_tap/agent/
  ├── car_reactive_agent.py          # 【主力 99%~100%】单模型纯视觉闭环代理（兼容真机 RobotTabletDevice）
  └── maigym_agent.py                # 【对照 E2E】原生端到端多步视觉代理（含防死循环护栏）
tests/
  └── test_maigym_agent.py           # 离线单元测试（7/7 通过，含安全护栏与闭环单测）
scripts/car_hmi_multistep/
  ├── README.md                      # 本文档（方案原理、完整实测指标与复现指南）
  ├── car_reactive_agent.py          # 同版闭环代理（供评测脚本直接加载）
  ├── build_dataset_maigym.py        # v1 单步定位数据构建
  ├── build_dataset_multistep_v2.py  # v2 三任务联合数据构建
  ├── build_dataset_v3_sr99.py       # v3 四合一 SR≥99% 数据构建（12,107 条）
  ├── train_maigym_multistep_lora.py # 多图多轮全量 BF16 LoRA 微调 + CPU BF16 Merge
  ├── run_bench_maigym.py            # 392 控件单步定位基准（Clean & Camera）
  ├── bench_long_tasks.py            # 20 项标准跨页长任务定义
  ├── bench_reactive_all.py          # 30 项单模型闭环评测器（T01–T20 + C01–C10）
  ├── bench_e2e_multistep.py         # 30 项纯端到端 E2E 评测器
  └── bench_stress_100.py            # 100 项独立随机跨页压力测试基准（seed=99999）
```

---

## 四、真机接入与复现命令

### 1. 单张 RTX 4090 启动推理服务
```bash
vllm serve /mnt/ymtc2t/models/MAI-Gym-9B-carhmi-v3-s273 \
  --served-model-name mai-gym-9b-v3-s273 \
  --host 127.0.0.1 --port 18809 --dtype bfloat16 \
  --max-model-len 12288 --enforce-eager \
  --gpu-memory-utilization 0.68 \
  --limit-mm-per-prompt '{"image": 3}' \
  --mm-processor-kwargs '{"size":{"longest_edge":3072000,"shortest_edge":65536}}' \
  --max-num-seqs 16 --trust-remote-code
```

### 2. 真机机械臂 / 灵巧手接入示例（仅需 3 行代码，内置全部对齐提示词）
```python
from tablet_tap.agent.car_reactive_agent import ReactiveCarAgent

# device 传入真机 RobotTabletDevice（或未来灵巧手实现的 Device）
agent = ReactiveCarAgent(
    device,
    grounder_url="http://127.0.0.1:18809/v1",
    grounder_model="mai-gym-9b-v3-s273",
    settle_s=0.5,  # 真机点击后等待界面动画稳定
)
log = agent.run("把主驾温度调到22度，打开前除霜，副驾座椅加热开到2档，再导航去公司选最快路线。")
print(log.status, log.clicks, log.summary)
```

### 3. 训练与全量基准复现
```bash
# 生成 v3 四合一数据集
python scripts/car_hmi_multistep/build_dataset_v3_sr99.py \
  /mnt/ymtc2t/datasets/carhmi_maigym_v3 \
  --v1-dir /mnt/ymtc2t/datasets/carhmi_maigym_v1 \
  --train-eps 320 --val-eps 24 --workers 4

# 单卡 4090 BF16 LoRA 增量微调（基于 v2 权重）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u scripts/car_hmi_multistep/train_maigym_multistep_lora.py \
  --model /mnt/ymtc2t/models/MAI-Gym-9B-carhmi-v2 \
  --data-dir /mnt/ymtc2t/datasets/carhmi_maigym_v3 \
  --out-dir /mnt/ymtc2t/outputs/maigym9b_carhmi_v3 \
  --merged-dir /mnt/ymtc2t/models/MAI-Gym-9B-carhmi-v3 \
  --epochs 0.36 --grad-accum 16 --lr 7e-5 --lora-r 32 --lora-alpha 64 --eval-steps 45 --bf16

# 392 键单步基准（Clean & Camera）
bin/py scripts/car_hmi_multistep/run_bench_maigym.py \
  --base-url http://127.0.0.1:18809/v1 --model mai-gym-9b-v3-s273 --label maigym9b_v3_s273 --workers 8 --no-think

# 30 项标准 + 刁钻长任务（Clean & Camera）
python scripts/car_hmi_multistep/bench_reactive_all.py \
  --grounder-url http://127.0.0.1:18809/v1 --grounder-model mai-gym-9b-v3-s273 \
  --suite all --cond clean --label reactive_v3_s273_clean
python scripts/car_hmi_multistep/bench_reactive_all.py \
  --grounder-url http://127.0.0.1:18809/v1 --grounder-model mai-gym-9b-v3-s273 \
  --suite all --cond camera --label reactive_v3_s273_camera

# 100 项独立随机地狱压测（seed=99999）
python scripts/car_hmi_multistep/bench_stress_100.py \
  --agent reactive --url http://127.0.0.1:18809/v1 --model mai-gym-9b-v3-s273 --label stress100_v3_s273

# 离线单元测试 & 架构分层测试
TABLET_TAP_NO_HARDWARE=1 bin/py -m pytest -q tests/test_maigym_agent.py tests/test_architecture.py
```
