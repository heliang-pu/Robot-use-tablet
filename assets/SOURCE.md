# 机械臂模型

`CobotMagicArm/` 是主办方 RoboSynChallenge 仓库里的单臂模型，整个文件夹原样拷来，没有改动：

- 来源：`/home/phl/workspace/fmc3-no1-real-stage/_remote_sim/RoboSynChallenge/assets/CobotMagicArm`（官方仓库 assets 的本机副本，URDF 修改时间 2026-11-13），拷贝于 2026-11-23 14:37 北京时间。
- `CobotMagicNoGripper.urdf`（md5 6ddc76a58fd4…）：基座 → joint1…joint6 → link6，6 个转动关节。**程序用的是这个**：
  `tablet_tap/core/kinematics.py` 用 Pinocchio 读它，在 link6 上加一个"笔尖"坐标系（V100 夹爪的指尖 TCP 再沿笔轴伸出笔长），正逆运动学都在这个模型上算。
- `CobotMagicWithGripperV100.urdf`（md5 044258220ff9…）：同一只手臂加上 100 mm 行程的夹爪（夹爪座、两根手指）。目前只作参考，笔夹的指尖定位槽就是按它的手指网格做的。
- `Collision/`（4 MB，凸分解后的 obj）和 `Visual/`（106 MB，dae）是两个 URDF 引用的网格。

两点要知道：

1. **关节限位**：URDF 里写的是仿真用的限位，和真机驱动实际接受的限位不完全一样（4、5 关节 URDF 更窄，6 关节 URDF 更宽）。
   规划用的是真机驱动的限位（`kinematics.DRIVER_JOINT_LIMITS_RAD`，再各留 0.05 rad），和 RoboSynChallenge 的 `SafetyFilter` 一致；
   URDF 的限位可以用 `kinematics.model_joint_limits()` 读出来对比。

   | 关节 | 1 | 2 | 3 | 4 | 5 | 6 |
   |---|---|---|---|---|---|---|
   | URDF (rad) | ±2.618 | 0 … 3.14 | −2.967 … 0 | ±1.745 | ±1.3 | ±2.988 |
   | 真机驱动 (rad) | ±2.618 | 0 … 3.1416 | −2.9671 … 0 | ±1.81 | ±1.35 | ±2.82 |

2. **网格路径**：URDF 里网格写成 `../Collision/xxx.obj`、`../Visual/xxx.dae`（主办方仿真器的写法），相对 URDF 自己的目录是找不到的。
   只算运动学不需要网格；要加载几何体（碰撞检测、可视化）时，把 `Collision` 子目录作为包目录传进去即可：

   ```python
   import pinocchio as pin
   from tablet_tap.core.kinematics import URDF_PATH
   model = pin.buildModelFromUrdf(str(URDF_PATH))
   geometry = pin.buildGeomFromUrdf(model, str(URDF_PATH), pin.GeometryType.COLLISION,
                                    package_dirs=[str(URDF_PATH.parent / "Collision")])   # 7 个碰撞体
   ```
