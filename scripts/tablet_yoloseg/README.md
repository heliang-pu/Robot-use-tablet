# 无 Tag 码任意姿态平板分割与 4 角点追踪（`YOLO11s-seg`）

面向顶部相机 + 机械臂/灵巧手任意位置、任意倾角（0°～70° 陡立）点击场景，通过 **VLM 框选 + SAM2.1-Base 像素级自动打标 + 人工/Gemini 视觉质检 + 0°～70° 3D 透视合成增强** 在单张 RTX 4090 上训练 `YOLO11s-seg`，彻底替代运行时 ArUco Tag 码。

## 实测核心指标（权重已保存在 `gpu0-4090:/mnt/ymtc2t/models/tablet_yoloseg/yolo11s_tablet_v1.pt`）

| 评估维度 | 实测结果 | 说明 |
|---|---|---|
| **真实留出场次（73 帧，严格按场次隔离）** | **Box mAP50 = 0.994 / mAP50-95 = 0.940** | 覆盖 `tablet` / `overview` / `wrist` 三视角及机械臂遮挡 |
| **真实留出场次掩码精度** | **Mask mAP50 = 0.994 / mAP50-95 = 0.900** | `median Mask IoU = 0.9627`（均 0.9421） |
| **真实无遮挡 4 角点像素误差（@1080p）** | **中位数 2.94 px（约 0.8 mm）** | Huber 四边直线拟合求交点 |
| **38°～68° 陡倾角任意位置压测（80 帧）** | **检出率 80/80 (100.0%)，Mask IoU 0.9878** | 4 角点误差中位数 **1.76 px（约 0.7 mm）**、均值 **3.00 px** |
| **单帧端到端推理耗时（RTX 4090）** | **均值 5.92 ms（P95 6.88 ms，~169 FPS）** | 含 `YOLO11s-seg (retina_masks)` + 四边直线交点 + 透视黑边内缩 |

## 快速接入示例

```python
import cv2
from tablet_tap.vision.tablet_seg_tracker import TabletSegTracker, fit_screen_plane_rgbd

tracker = TabletSegTracker("/mnt/ymtc2t/models/tablet_yoloseg/yolo11s_tablet_v1.pt")
frame = cv2.imread("top_camera.jpg")
rect_screen, det = tracker.rectify_screen(frame, out_size=(1280, 800))
# det.outer_quad  -> 机身外框 4 角点 [TL, TR, BR, BL]
# det.screen_quad -> 内圈发光显示区 4 角点（即使深色模式也严格精准）
# 若有 RealSense 对齐深度图 depth_m 与内参 K：
# plane = fit_screen_plane_rgbd(det, depth_m, K)  # 输出 3D 法向量与 4 个 3D 角点
```

## 目录与流水线

- `tablet_tap/vision/tablet_seg_tracker.py`：推理、四边直线交点拟合、透视黑边反标定与内缩、RealSense RGB-D RANSAC 3D 平面拟合；
- `tests/test_tablet_seg_tracker.py`：离线单元测试（3/3 通过，覆盖 60° 陡倾角遮挡与 RGB-D 平面法向量）；
- `scripts/tablet_yoloseg/autolabel_tablet_sam2.py`：VLM 边界框 + SAM2.1-Base 自动像素级打标与质检拼图；
- `scripts/tablet_yoloseg/build_tablet_yoloseg_dataset.py`：真实数据 + 800 张 0°～70° 3D 透视合成数据（含真实 L8 车机亮/暗屏界面）；
- `scripts/tablet_yoloseg/train_and_eval_yoloseg.py`：4090 上 80 轮训练与真实/陡倾角双基准量化评测。
