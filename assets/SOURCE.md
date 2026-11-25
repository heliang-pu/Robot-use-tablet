# 机械臂模型

`CobotMagicArm/` 是主办方 RoboSynChallenge 仓库里的单臂模型，整个文件夹拷来；只改了两个 URDF 里 joint1…joint6 的关节限位（见下面第 1 点），别的没有改动：

- 来源：`/home/phl/workspace/fmc3-no1-real-stage/_remote_sim/RoboSynChallenge/assets/CobotMagicArm`（官方仓库 assets 的本机副本，URDF 修改时间 2026-11-13），拷贝于 2026-11-23 14:37 北京时间。
- `CobotMagicNoGripper.urdf`（md5 19f5a5f392b9…，改限位前 6ddc76a58fd4…）：基座 → joint1…joint6 → link6，6 个转动关节。**程序用的是这个**：
  `tablet_tap/core/kinematics.py` 用 Pinocchio 读它，在 link6 上加一个"笔尖"坐标系（V100 夹爪的指尖 TCP 再沿笔轴伸出笔长），正逆运动学都在这个模型上算。
- `CobotMagicWithGripperV100.urdf`（md5 0b72ec049c47…，改限位前 044258220ff9…）：同一只手臂加上 100 mm 行程的夹爪（夹爪座、两根手指）。目前只作参考，笔夹的指尖定位槽就是按它的手指网格做的。
- `Collision/`（4 MB，凸分解后的 obj）和 `Visual/`（106 MB，dae）是两个 URDF 引用的网格。

两点要知道：

1. **关节限位**：两个 URDF 里 joint1…joint6 的 `<limit>` 在 2026-11-23 改成了**真机固件里的限位**（拿笔的左臂的；
   手臂走到固件限位就停住，规划超出去就到不了位）。规划直接用 URDF 的限位（`kinematics.JOINT_LIMITS_RAD`，再各留 0.025 rad），
   驱动的钳位（`SafetyFilter`）也用同一组（留 0.02 rad）。两只臂的固件限位是当天用 `limits` 命令从手臂上读出来的：
   右臂的 5 号关节只到 70°，和左臂不一样，记在 `calib/setup.json` 的 `arm.firmware_limits_deg` 里，用右臂时按那一行走。
   `effort`、`velocity` 和夹爪的两个关节没有动。

   | 关节 | 1 | 2 | 3 | 4 | 5 | 6 |
   |---|---|---|---|---|---|---|
   | 拷来时（仿真用，rad） | ±2.618 | 0 … 3.14 | −2.967 … 0 | ±1.745 | ±1.3 | ±2.988 |
   | 现在 = 左臂固件（度） | ±150 | 0 … 180 | −170 … 0 | ±100 | −70 … 75 | ±180 |
   | 右臂固件（度） | ±150 | 0 … 180 | −170 … 0 | ±100 | ±70 | ±180 |

   左臂 5 号的上限出厂是 70°，当天用 `limits --joint5_max 75` 改到 75°（笔要垂直够到屏幕远端）。手册上 6 号的运动范围是 ±170°，
   比固件的 ±180° 小；点平板时 6 号在 0° 附近，没有在 160° 以外试过。

2. **网格路径**：URDF 里网格写成 `../Collision/xxx.obj`、`../Visual/xxx.dae`（主办方仿真器的写法），相对 URDF 自己的目录是找不到的。
   只算运动学不需要网格；要加载几何体（碰撞检测、可视化）时，把 `Collision` 子目录作为包目录传进去即可：

   ```python
   import pinocchio as pin
   from tablet_tap.core.kinematics import URDF_PATH
   model = pin.buildModelFromUrdf(str(URDF_PATH))
   geometry = pin.buildGeomFromUrdf(model, str(URDF_PATH), pin.GeometryType.COLLISION,
                                    package_dirs=[str(URDF_PATH.parent / "Collision")])   # 7 个碰撞体
   ```
