"""The wrist-camera flow on the bench in Isaac Sim: the real commands (wrist calib / view / run) against the twin.

    TABLET_TAP_NO_HARDWARE=1 bin/py scripts/isaac/wrist_bench.py [--out runs/isaac_wrist] [--mount tap|kit] [--preview]

Real here: tablet_tap's planner, eye-in-hand calibration, look pose, rectification, tap planning and agent loop; the
arm's dynamics, the bench and the lamps of the RoboSyn twin; the wrist D435's intrinsics at 1080p. Made up, and unknown
to the program: the mount is not quite as drawn, the joint zeros and the pen are off a little, the screen is not exactly
where the top camera said. Focus and noise of the real D435 are assumptions (``--blur``, ``--noise``). The model is a
stand-in that knows what to press (``--llm scripted``); whether the real model reads the pictures is a separate check.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from multiprocessing.connection import Client
import os
import pathlib
import subprocess
import sys
import time

import cv2
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("TABLET_TAP_NO_HARDWARE", "1")

from tablet_tap import config  # noqa: E402
from tablet_tap.agent.gui_agent import GuiAgent  # noqa: E402
from tablet_tap.app import Options  # noqa: E402
from tablet_tap.calibration import page_server  # noqa: E402
from tablet_tap.calibration.eye_in_hand import mount_error  # noqa: E402
from tablet_tap.calibration.hand_eye import seed_joints  # noqa: E402
from tablet_tap.cli.commands.wrist import Wrist  # noqa: E402
from tablet_tap.cli.commands.wrist import WristContext  # noqa: E402
from tablet_tap.core.kinematics import pose  # noqa: E402
from tablet_tap.core.look import HOLDER_BOX  # noqa: E402
from tablet_tap.core.look import MOUNTS  # noqa: E402
from tablet_tap.core.look import camera_in_flange  # noqa: E402
from tablet_tap.core.look import flange_pose  # noqa: E402
from tablet_tap.core.look import tip_in_flange  # noqa: E402
from tablet_tap.core.pinhole import CameraModel  # noqa: E402
from tablet_tap.core.screen_map import fit_screen_map  # noqa: E402
from tablet_tap.robot.wrist import NOMINAL_MODEL  # noqa: E402
from tablet_tap.vision import layout as screen_layout  # noqa: E402
from tablet_tap.web import rehearsal  # noqa: E402

TWIN = pathlib.Path("/home/phl/workspace/fmc3-no1-real-stage/RoboSynChallenge/.worktrees/isaac-twin/sim_isaac")
ISAAC_PYTHON = pathlib.Path.home() / "workspace/venvs/isaacsim51/bin/python"
TABLE_Z = -0.016  # the table top in the left arm's base frame, in the twin
GLASS_M = 0.008  # the tablet's screen above the table


def small_turn(deg: float, axis) -> np.ndarray:
    axis = np.asarray(axis, np.float64)
    return cv2.Rodrigues(np.radians(deg) * axis / np.linalg.norm(axis))[0]


class Isaac:
    """The server process and the calls to it."""

    def __init__(self, out: pathlib.Path, cfg: dict, port: int) -> None:
        (out / "isaac_config.json").write_text(json.dumps(cfg, indent=1))
        env = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "PYTHONPATH", "LD_PRELOAD", "JAX_PLATFORMS")}
        env.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
        self.log = (out / "isaac_server.log").open("w")
        work = out / "isaac_cwd"  # the URDF importer writes into the current directory
        work.mkdir(exist_ok=True)
        self.proc = subprocess.Popen(
            [str(ISAAC_PYTHON), "-u", str(REPO / "scripts/isaac/wrist_bench_server.py"), "--config", str(out / "isaac_config.json"), "--port", str(port)],
            env=env, stdout=self.log, stderr=subprocess.STDOUT, cwd=work,
        )  # fmt: skip
        deadline = time.time() + 420
        while True:
            try:
                self.conn = Client(("127.0.0.1", port), authkey=b"tablet-tap")
                break
            except ConnectionRefusedError:
                if self.proc.poll() is not None or time.time() > deadline:
                    self.proc.kill()
                    raise SystemExit(f"Isaac 没有起来，看 {out / 'isaac_server.log'}") from None
                time.sleep(1.0)

    def call(self, cmd: str, **kwargs):
        self.conn.send({"cmd": cmd, **kwargs})
        reply = self.conn.recv()
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    def close(self) -> None:
        try:
            self.call("close")
        except (OSError, EOFError, RuntimeError):
            pass
        try:
            self.proc.wait(60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


@dataclasses.dataclass
class Truth:
    """What the bench really is."""

    pen: object  # the arm's model
    zero: np.ndarray  # real joints = readings + these
    tool: np.ndarray  # real tip in the model's tip frame
    camera: np.ndarray  # (4, 4) in the flange
    screen: object  # ScreenMap
    model: CameraModel

    def tip(self, q_real: np.ndarray) -> np.ndarray:
        t = self.pen.fk(q_real)
        return t[:3, 3] + t[:3, :3] @ self.tool

    def wrist_pixels(self, q_real: np.ndarray, points: np.ndarray) -> np.ndarray:
        cam = flange_pose(self.pen, q_real) @ self.camera
        return self.model.project((np.asarray(points) - cam[:3, 3]) @ cam[:3, :3])


class IsaacArm:
    """``ArmIO`` of the twin's left arm: readings are off by the joint zeros; the pen presses the tablet."""

    def __init__(self, isaac: Isaac, truth: Truth, tablet) -> None:
        self.isaac, self.truth, self.tablet = isaac, truth, tablet
        self.real = np.asarray(isaac.call("joints")["q"])
        self.down, self.path, self.sent = False, [self.real.copy()], 0

    def read(self) -> np.ndarray:
        return self.real - self.truth.zero

    def command(self, q: np.ndarray) -> None:
        self.real = np.asarray(self.isaac.call("command", q=(np.asarray(q) + self.truth.zero).tolist(), steps=2)["q"])
        self.sent += 1
        if self.sent % 5 == 0:
            self.path.append(self.real.copy())
        u, v, h = self.truth.screen.uv_of(self.truth.tip(self.real))
        pressed = h <= -0.0003 and 0 <= u <= 1 and 0 <= v <= 1
        if pressed and not self.down:
            self.tablet.touch(float(u), float(v))
        elif self.down and not pressed:
            self.tablet.release(float(u), float(v))
        self.down = pressed

    def close(self) -> None:
        pass


class Display:
    """The rehearsal's tablet at the display's resolution, put on the twin's screen."""

    def __init__(self, out: pathlib.Path) -> None:
        rehearsal.PICTURE_PX = (1280, 800)
        self.calib = page_server.CalibState()
        self.tablet = rehearsal.Tablet(self.calib)
        self.folder = out / "screens"
        self.folder.mkdir(exist_ok=True)
        self.shown = None
        self.extra: dict[str, np.ndarray] = {}
        self.isaac: Isaac | None = None

    def refresh(self) -> None:
        if self.tablet.app == "calib":
            name = "calib_markers"
            image = screen_layout.render_layout(screen_layout.default_layout(1.6), 2560)
        elif self.tablet.app in self.extra:
            name, image = self.tablet.app, self.extra[self.tablet.app]
        else:
            name = f"{self.tablet.app}_{int(self.tablet.playing)}_{self.tablet.track}"
            image = cv2.resize(self.tablet.picture(), (2560, 1600), interpolation=cv2.INTER_CUBIC)
        if name != self.shown:
            path = self.folder / f"{name}.png"
            cv2.imwrite(str(path), image)
            self.isaac.call("show", name=name, texture=str(path))
            self.shown = name


class IsaacCamera:
    def __init__(self, isaac: Isaac, display: Display, name: str, blur: float, noise: float) -> None:
        self.isaac, self.display, self.name, self.blur, self.noise = isaac, display, name, blur, noise
        self.rng = np.random.default_rng(3)

    def read(self) -> np.ndarray:
        self.display.refresh()
        image = cv2.cvtColor(self.isaac.call("picture", name=self.name)["image"], cv2.COLOR_RGB2BGR)
        if self.blur > 0:
            image = cv2.GaussianBlur(image, (0, 0), self.blur)
        return np.clip(image + self.rng.normal(0, self.noise, image.shape), 0, 255).astype(np.uint8)

    def close(self) -> None:
        pass


class Page:
    def __init__(self, display: Display) -> None:
        self.display = display

    def set_mode(self, mode: str, **_) -> None:
        self.display.tablet.show("calib")

    def clear(self) -> None:
        pass

    def layout(self):
        return screen_layout.default_layout(1.6)


class BenchContext(WristContext):
    def __init__(self, options: Options, arm: IsaacArm, camera: IsaacCamera, page: Page) -> None:
        super().__init__(options)
        self._arm, self._camera, self._page = arm, camera, page

    def camera(self):
        return self._camera

    def page(self):
        return self._page


class Scripted:
    """Stands in for the model: finds the target in the rectified picture exactly, answers in GUI-Owl's format."""

    def __init__(self, display: Display, truth: Truth, ctx: WristContext, arm: IsaacArm) -> None:
        self.display, self.truth, self.ctx, self.arm, self.aims = display, truth, ctx, arm, []

    def where(self, name: str) -> list[float]:
        u0, v0, u1, v1 = self.display.tablet.box(name)
        px = self.truth.wrist_pixels(self.arm.real, self.truth.screen.point(0.5 * (u0 + u1), 0.5 * (v0 + v1)))
        u, v = self.ctx.device().rectifier.cam_to_norm(px)
        self.aims.append((0.5 * (u0 + u1), 0.5 * (v0 + v1)))
        return [round(1000 * float(u), 1), round(1000 * float(v), 1)]

    def predict_mm(self, messages: list[dict]):
        text, tablet = json.dumps(messages, ensure_ascii=False), self.display.tablet
        done = {"action": "terminate", "status": "success"}
        if "打开设置" in text:
            action = done if tablet.app == "settings" else {"action": "click", "coordinate": self.where("settings")}
        elif "回到桌面" in text:
            action = done if tablet.app == "home" else {"action": "system_button", "button": "Home"}
        elif "暂停" in text:
            action = {"action": "click", "coordinate": self.where("play")} if tablet.playing else done
        elif tablet.app == "home":
            action = {"action": "click", "coordinate": self.where("music")}
        else:
            action = done if tablet.playing else {"action": "click", "coordinate": self.where("play")}
        return "Action: 排练\n<tool_call>\n" + json.dumps({"name": "mobile_use", "arguments": action}) + "\n</tool_call>", None, None


def text_chart() -> np.ndarray:
    """Chinese and Latin text from small to large, as an app would draw it on a 2560 x 1600 display."""
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415

    font_path = next(p for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc") if pathlib.Path(p).exists())
    image = Image.new("RGB", (2560, 1600), (250, 250, 250))
    draw, y = ImageDraw.Draw(image), 30
    for size in (24, 28, 32, 36, 44, 52, 64, 80):
        font = ImageFont.truetype(font_path, size)
        mm = size / 2560 * 236.9
        for x, colour in ((40, (30, 30, 30)), (1300, (130, 130, 130))):
            draw.text((x, y), f"{size}px {mm:.1f}mm 设置 蓝牙已连接 Wi-Fi 5G 播放列表 12:45"[: 34 if size > 44 else 60], font=font, fill=colour)
        y += int(size * 1.9) + 10
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def check_path(pen, path: np.ndarray, camera: np.ndarray, screen, mount_stl: pathlib.Path) -> dict:
    """The smallest gaps along the joints the arm really went through, from the collision meshes (hpp-fcl): independent of
    the point samples the planner uses."""
    import hppfcl  # noqa: PLC0415
    import pinocchio as pin  # noqa: PLC0415
    import struct  # noqa: PLC0415

    from tablet_tap.core.kinematics import URDF_PATH  # noqa: PLC0415
    from tablet_tap.core.look import CAMERA_BOX, POST_BOX  # noqa: PLC0415

    model = pin.buildModelFromUrdf(str(URDF_PATH))
    geom = pin.buildGeomFromUrdf(model, str(URDF_PATH), pin.GeometryType.COLLISION, package_dirs=[str(URDF_PATH.parent / "Collision")])
    data, gdata = model.createData(), geom.createData()
    parts = {g.name: (g, i) for i, g in enumerate(geom.geometryObjects)}

    def box_of(low, high):
        low, high = np.array(low), np.array(high)
        return hppfcl.Box(*(high - low)), pin.SE3(np.eye(3), 0.5 * (low + high))

    raw = mount_stl.read_bytes()
    n = struct.unpack("<I", raw[80:84])[0]
    v = np.frombuffer(raw[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n)["v"].reshape(-1, 3).astype(float) * 1e-3
    v = np.column_stack([-v[:, 1], v[:, 0], v[:, 2]])
    bvh = hppfcl.BVHModelOBBRSS()
    bvh.beginModel(n, 3 * n)
    for k in range(n):
        bvh.addTriangle(v[3 * k], v[3 * k + 1], v[3 * k + 2])
    bvh.endModel()
    tip = float(tip_in_flange(pen)[2, 3])
    tool = {"mount": (bvh, pin.SE3.Identity()), "camera": (lambda b: (b[0], pin.SE3(camera) * b[1]))(box_of(*CAMERA_BOX)),
            "holder": box_of(*HOLDER_BOX), "pen": box_of((-0.0045, -0.0045, 0.16), (0.0045, 0.0045, tip - 0.001))}  # fmt: skip
    s = screen
    frame = np.eye(4)
    frame[:3, :3] = np.column_stack([s.right / np.linalg.norm(s.right), s.down / np.linalg.norm(s.down), -s.normal])
    w, h = s.size_m
    frame[:3, 3] = s.point(0.5, 0.5) - 0.004 * s.normal
    world = {"table": (hppfcl.Box(3.0, 3.0, 0.1), pin.SE3(np.eye(3), np.array([0.6, -0.3, TABLE_Z - 0.05]))),
             "tablet": (hppfcl.Box(w + 0.018, h + 0.018, 0.008), pin.SE3(frame)), "post": box_of(*POST_BOX)}  # fmt: skip
    req, out = hppfcl.DistanceRequest(), {}
    for q in path:
        pin.updateGeometryPlacements(model, data, geom, gdata, q)
        pin.updateFramePlacements(model, data)
        flange = data.oMf[model.getFrameId("link6")]
        moving = {name: (g.geometry, gdata.oMg[i]) for name, (g, i) in parts.items() if not name.startswith(("base", "link1"))}
        mounted = {name: (shape, flange * at) for name, (shape, at) in tool.items()}
        for wn, (ws, wt) in world.items():
            for mn, (ms, mt) in {**moving, **mounted}.items():
                if wn == "tablet" and mn == "pen":
                    continue  # the pen is meant to touch it
                if wn == "post" and mn.startswith("link2"):
                    pass
                d = hppfcl.distance(ms, hppfcl.Transform3f(mt.rotation, mt.translation), ws, hppfcl.Transform3f(wt.rotation, wt.translation), req, hppfcl.DistanceResult())
                key = f"{mn.split('_')[0]}-{wn}"
                out[key] = min(out.get(key, 9.0), d)
        for mn in ("mount", "camera"):
            for ln, (ls, lt) in moving.items():
                if ln.startswith(("link5", "link6")):
                    continue
                ms, mt = mounted[mn]
                d = hppfcl.distance(ms, hppfcl.Transform3f(mt.rotation, mt.translation), ls, hppfcl.Transform3f(lt.rotation, lt.translation), req, hppfcl.DistanceResult())
                key = f"{mn}-{ln.split('_')[0]}"
                out[key] = min(out.get(key, 9.0), d)
    return dict(sorted(out.items(), key=lambda kv: kv[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "runs" / f"isaac_wrist_{time.strftime('%Y%m%d_%H%M%S')}"))
    ap.add_argument("--mount", default="tap", choices=sorted(MOUNTS))
    ap.add_argument("--at", type=float, nargs=3, default=(32.0, 0.0, 2.0), metavar=("前cm", "左cm", "转度"))
    ap.add_argument("--slant", type=float, default=0.0)
    ap.add_argument("--blur", type=float, default=1.0, help="画面的模糊（1080p 像素，高斯标准差）：约等于按 D435 光学参数估的近距离焦")
    ap.add_argument("--noise", type=float, default=2.5)
    ap.add_argument("--nits", type=float, default=900.0)
    ap.add_argument("--light", type=float, default=0.18)
    ap.add_argument("--port", type=int, default=6398)
    ap.add_argument("--preview", action="store_true", help="只在图纸算出的看屏位拍几张，不走流程")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = pathlib.Path(args.out).resolve()
    (out / "calib").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    config.PROJECT_ROOT = out

    setup = config.SetupStore(REPO / "calib" / "setup.json").load()
    size = tuple(setup.screen_size_m or (0.2369, 0.1481))
    true_size = tuple(x * 1.002 for x in size)
    z = TABLE_Z + GLASS_M + 0.5 * size[1] * np.sin(np.radians(args.slant)) + (0.01 if args.slant else 0.0)
    true_screen = rehearsal.tablet_screen((args.at[0] * 1e-2, args.at[1] * 1e-2, z), args.slant, args.at[2], true_size)
    pen = setup.pen_arm()
    drawn = camera_in_flange(*MOUNTS[args.mount])
    wrong = pose(small_turn(1.2, rng.normal(size=3)), 0.0025 * (lambda x: x / np.linalg.norm(x))(rng.normal(size=3)))
    truth = Truth(pen, np.radians([0.25, -0.25, 0.3, 0.3, -0.4, 0.2]), np.array([0.0012, -0.0008, 0.0005]), drawn @ wrong, true_screen, NOMINAL_MODEL)

    # what the program knows: the screen as a touch calibration with these joint zeros would have it
    seed_screen = dataclasses.replace(true_screen, tilt_deg=8.0)
    seed = seed_joints(pen, seed_screen, setup.motion_limits())
    uv, tips, rot = [], [], seed_screen.tool_rotation()
    for v in (0.1, 0.5, 0.9):
        q = seed
        for u in (0.1, 0.5, 0.9):
            target = pose(rot, true_screen.point(u, v) - rot @ truth.tool)
            q = pen.solve(target, q)
            uv.append((u + rng.normal(0, 0.0008), v + rng.normal(0, 0.0012)))
            tips.append(pen.fk(q - truth.zero))
    believed = dataclasses.replace(fit_screen_map(np.array(uv), np.array(tips)), tilt_deg=8.0, tool_x=seed_screen.tool_x)
    setup.screen_map, setup.seed_q = believed, seed.tolist()
    setup.nav = {"back": [0.75, 0.965], "home": [0.5, 0.965]}
    for name in ("live_screen", "tap_correction", "screen_map_source"):
        setup.records.pop(name, None)
    store = config.SetupStore(out / "calib" / "setup.json")
    store.save(setup, backup=False)

    s = true_screen
    frame = np.eye(4)
    frame[:3, :3] = np.column_stack([s.right / np.linalg.norm(s.right), s.down / np.linalg.norm(s.down), -s.normal])
    frame[:3, 3] = s.origin
    m = NOMINAL_MODEL
    cam = {"parent": "link6", "width": m.width, "height": m.height, "fx": m.fx, "fy": m.fy, "cx": m.cx, "cy": m.cy}
    top = setup.camera_model

    def eye_view(eye, target):
        f = np.array(target, float) - eye
        f /= np.linalg.norm(f)
        r = np.cross(f, [0, 0, 1.0])
        r /= np.linalg.norm(r)
        return pose(np.column_stack([r, np.cross(f, r), f]), np.array(eye, float)).tolist()

    c = s.point(0.5, 0.5)
    cfg = {
        "twin_dir": str(TWIN), "scene_dir": str(TWIN / "scenes/click_bell/bench1116L000_random"),
        "options": {"appearance": "bench", "cloth": "orange_check", "base_plate_m": 0.0, "top_camera": "nominal", "start_pose": "episode",
                    "cloth_back_x": 0.232, "table_back_x": 0.13, "table_body": True, "light_scale": args.light},
        "base_link": "left_base_link", "park_q": (np.asarray(setup.park_q) + truth.zero).tolist(), "screen_in_base": frame.tolist(), "screen_size_m": list(true_size),
        "mount_stl": str(REPO / "cad/wrist_camera/out_one_piece/wrist_mount_left.stl"), "wrist_in_flange": truth.camera.tolist(),
        "holder_box": HOLDER_BOX, "tip_z": float(tip_in_flange(pen)[2, 3]), "screen_nits": args.nits,
        "cameras": {"wrist": cam | {"pose": truth.camera.tolist()},
                    "kit": cam | {"pose": camera_in_flange(*MOUNTS["kit"]).tolist()},
                    "top": {"parent": "base", "width": top.width, "height": top.height, "fx": top.fx, "fy": top.fy, "cx": top.cx, "cy": top.cy, "pose": setup.camera_pose},
                    "side": {"parent": "base", "width": 1280, "height": 960, "fx": 1100, "fy": 1100, "cx": 640, "cy": 480, "pose": eye_view(c + [0.35, 0.62, 0.42], c + [-0.08, 0, 0.10])},
                    "front": {"parent": "base", "width": 1280, "height": 960, "fx": 1300, "fy": 1300, "cx": 640, "cy": 480, "pose": eye_view(c + [0.30, -0.42, 0.34], c + [-0.06, 0, 0.11])}},
    }  # fmt: skip
    print("启动 Isaac ……", flush=True)
    isaac = Isaac(out, cfg, args.port)
    report: dict = {"args": vars(args), "mount_true_vs_drawn": mount_error(truth.camera, drawn)}
    try:
        base = np.array(isaac.call("info")["arm_base"])
        display = Display(out)
        display.isaac = isaac
        display.extra["chart"] = text_chart()
        arm = IsaacArm(isaac, truth, display.tablet)
        gap = np.linalg.norm(isaac.call("flange")["pose"][:3, 3] - flange_pose(pen, arm.real)[:3, 3]) * 1e3
        report["isaac_fk_gap_mm"] = float(gap)
        print(f"Isaac 里左臂基座 {base[:3, 3].round(3).tolist()}，法兰位置和运动学差 {gap:.2f} mm", flush=True)
        camera = IsaacCamera(isaac, display, "wrist", args.blur, args.noise)
        options = Options(setup_path=store.path, view_path=out / "calib" / "screen_view.json", calib_url="", fake=True, yes=True)
        ctx = BenchContext(options, arm, camera, Page(display))
        ctx.wrist.mount, ctx.wrist.model = args.mount, NOMINAL_MODEL
        ctx.save_wrist()

        def shot(name: str, which: str, app: str | None = None) -> np.ndarray:
            if app:
                display.tablet.show(app)
            display.refresh()
            image = cv2.cvtColor(isaac.call("picture", name=which, settle=10)["image"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out / f"{name}.png"), image)
            return image

        if args.preview:
            look = ctx.wrist.planner(pen).best(believed, seed, np.asarray(setup.park_q))
            print(look.describe(), flush=True)
            ctx.move(look.q, "看屏位")
            for which in ("wrist", "kit", "side", "front", "top"):
                shot(f"preview_{which}", which, "calib" if which != "kit" else None)
            for app in ("home", "settings", "chart"):
                shot(f"preview_wrist_{app}", "wrist", app)
            return 0

        ns = argparse.Namespace
        wrist = Wrist()
        print("== wrist calib", flush=True)
        wrist._calib(ctx, ns(views=14))  # noqa: SLF001
        found = np.array(ctx.wrist.in_flange)
        off = mount_error(found, truth.camera)
        report["calib"] = {"vs_truth_mm": off[0], "vs_truth_deg": off[1], **{k: ctx.wrist.records["eye_in_hand"][k] for k in ("rms_px", "held_out_px", "left_out", "screen_size_m")}}
        print(f"标定结果离真值 {off[0]:.2f} mm、{off[1]:.2f}°", flush=True)

        print("== wrist view", flush=True)
        wrist._view(ctx, ns(out_width=1792, keep_pose=False))  # noqa: SLF001
        report["view"] = ctx.wrist.records["view"]

        print("== 看屏画面", flush=True)
        robot = ctx.device().robot
        robot.to_look()
        errors = []
        for u in np.linspace(0.03, 0.97, 9):
            for v in np.linspace(0.04, 0.96, 7):
                px = truth.wrist_pixels(arm.real, true_screen.point(u, v))
                uu, vv = ctx.device().rectifier.cam_to_norm(px)
                errors.append(true_screen.error_mm((u, v), (float(uu), float(vv))))
        report["picture_error_mm"] = {"mean": float(np.mean(errors)), "max": float(np.max(errors))}
        print(f"拉正后的画面和真实屏幕的位置差：平均 {np.mean(errors):.2f} mm，最大 {np.max(errors):.2f} mm", flush=True)
        for app in ("home", "player", "settings", "chart"):
            display.tablet.show(app)
            frame_ = camera.read()
            cv2.imwrite(str(out / f"wrist_raw_{app}.png"), frame_)
            cv2.imwrite(str(out / f"wrist_rectified_{app}.png"), ctx.device().rectifier.rectify(frame_))
        for which in ("side", "front", "kit"):
            shot(f"at_look_{which}", which, "home")
        display.tablet.show("calib")
        robot.to_rest()
        for which in ("top", "side"):
            shot(f"at_park_{which}", which, "settings")
        # the same screens through the top camera, rectified from its markers, for comparison
        from tablet_tap.vision.rectifier import ScreenRectifier  # noqa: PLC0415

        top_cam = IsaacCamera(isaac, display, "top", 0.7, args.noise)
        display.tablet.show("calib")
        try:
            top_rect = ScreenRectifier.from_marker_image(top_cam.read(), screen_layout.default_layout(1.6), out_width=1792)
            q_ = top_rect.screen_quad_cam()
            report["top_width_px"] = float(np.linalg.norm(q_[1] - q_[0]))
            for app in ("home", "player", "settings", "chart"):
                display.tablet.show(app)
                cv2.imwrite(str(out / f"top_rectified_{app}.png"), top_rect.rectify(top_cam.read()))
        except Exception as exc:  # noqa: BLE001
            report["top_error"] = str(exc)
            print("顶部相机：", exc, flush=True)

        print("== 点 12 个随机位置", flush=True)
        display.tablet.show("calib")
        robot.to_look()
        taps = []
        for u, v in rng.uniform(0.06, 0.94, (12, 2)):
            seq = len(display.calib.touches()) if hasattr(display.calib, "touches") else None  # noqa: F841
            before = len(arm.path)
            touched: list = []
            original = display.tablet.touch
            display.tablet.touch = lambda a, b: touched.append((a, b))  # noqa: B023
            robot.tap(float(u), float(v))
            display.tablet.touch = original
            robot.to_park()
            if touched:
                taps.append(true_screen.error_mm((u, v), touched[0]))
            del before
        report["taps"] = {"n": len(taps), "mean_mm": float(np.mean(taps)), "max_mm": float(np.max(taps))}
        print(f"12 次点按触发 {len(taps)} 次，平均偏 {np.mean(taps):.2f} mm，最大 {np.max(taps):.2f} mm（没有点按修正）", flush=True)

        print("== 智能体循环（替身模型）", flush=True)
        checks = {"打开音乐，播放音乐": lambda t: t.app == "player" and t.playing, "暂停播放": lambda t: t.app == "player" and not t.playing,
                  "回到桌面": lambda t: t.app == "home", "打开设置": lambda t: t.app == "settings"}  # fmt: skip
        display.tablet.show("home")
        report["runs"] = []
        for k, (text, ok) in enumerate(checks.items()):
            llm = Scripted(display, truth, ctx, arm)
            sent = arm.sent
            agent = GuiAgent(ctx.device(), max_steps=6, settle_s=0.0, llm=llm, run_dir=out / f"agent_{k}")
            result = agent.run(text)
            robot.to_park()
            reached = bool(ok(display.tablet))
            report["runs"].append({"instruction": text, "status": result.status, "steps": result.steps, "reached": reached, "arm_s": (arm.sent - sent) / 50.0})
            print(f"  {text}：{result.status}，{result.steps} 步，平板{'到了' if reached else '没到'}目标状态，机械臂运动 {(arm.sent - sent) / 50:.0f} 秒", flush=True)
        robot.to_rest()

        print("== 平板被挪 15 mm 后应当拒绝", flush=True)
        moved = dataclasses.replace(believed, origin=believed.origin + [0.015, 0, 0])
        report["refuses_moved"] = ctx.wrist.screen_moved(moved)
        print("  ", report["refuses_moved"], flush=True)

        path = np.array(arm.path)
        report["gaps_mm"] = {k: round(v * 1e3, 1) for k, v in list(check_path(pen, path, truth.camera, true_screen, REPO / "cad/wrist_camera/out_one_piece/wrist_mount_left.stl").items())[:14]}
        print(f"全程 {len(path)} 个实际关节位置的最小间隙（mm，碰撞网格独立核对）：", report["gaps_mm"], flush=True)
        jumps = np.abs(np.diff(path, axis=0)).max()
        print(f"最大单次（0.1 s）关节变化 {jumps:.3f} rad", flush=True)
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
        isaac.close()
        print("结果在", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
