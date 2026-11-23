# 从 RoboSynChallenge 原样拷来的 PiPER 控制代码

来源：/home/phl/workspace/fmc3-no1-real-stage/RoboSynChallenge，分支 claude/real-stage，提交 0046877b，
拷贝时间 2026-11-23 13:50 北京时间。标 M 的文件在原仓库里有未提交的改动，拷的是工作区里的当前版本（台子上实际在跑的）。
这些文件**没有改动**；要改行为请在 tablet_tap/ 里包一层，升级时整文件覆盖并更新下表。

| 文件 | 原路径 | 原仓库状态 | sha256 前 16 位 |
|---|---|---|---|
| `real_robot/__init__.py` | `policy/fmc3_no1/real_robot/__init__.py` | 已提交 | ad40597d4932d815 |
| `real_robot/hardware.py` | `policy/fmc3_no1/real_robot/hardware.py` | M | 353d8e216ae12278 |
| `real_robot/conversions.py` | `policy/fmc3_no1/real_robot/conversions.py` | M | fc6f823211669f23 |
| `reset_arms.py` | `scripts/sim2real/reset_arms.py` | 已提交 | 863aea04a305934d |
| `dock_switch.py` | `policy/hilserl/dock_switch.py` | ?? | 9ddadc72792300be |
| `../../scripts/setup_piper_can.sh` | `scripts/sim2real/setup_piper_can.sh` | 已提交 | 523b95c628ae6add |
| `../../docs/bench_twin_params.json` | `sim_twin/twin_params.json` | 已提交 | 4849102daf90293b |

各文件的作用：
- `real_robot/hardware.py`：`PiperArm`（经 pyAgxArm 走 CAN 控制一只 PiPER：固件识别、使能、读关节 / 夹爪、`move_j` 关节指令、故障检测、掉线重连）、`RealSenseCamera`、双臂的 `CobotMagicPiper`。
- `real_robot/conversions.py`：关节符号 / 零偏标定、夹爪开度换算、`SafetyFilter`（关节限位 + 每步限幅）、`HardwareFault`。
- `reset_arms.py`：急停、碰撞或示教模式之后把从臂恢复并重新使能（`--status_only` 只看状态，`--no_move` 恢复后不回零）。
- `dock_switch.py`：其中的 `fk`（CobotMagic 机械臂 URDF 链 + 指尖 TCP 的正运动学）是台子上一直在用的算法。程序自己的运动学已改用 Pinocchio + `assets/CobotMagicArm` 的 URDF，这个文件现在只在测试里当对照（`tests/test_kinematics.py` 检查两者一致）。
- `setup_piper_can.sh`：四个 USB-CAN 适配器按序列号固定成 can_l_follower 等名字并拉到 1 Mbps（要 sudo；这台机器已经配好）。
- `bench_twin_params.json`：台子的孪生仿真参数，含顶部 / 腕部相机 640×480 的内参和顶部相机位姿的估计值。
