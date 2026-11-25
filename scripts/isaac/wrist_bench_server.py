"""The bench in Isaac Sim (RoboSyn's twin, branch claude/isaac-twin, read only) with a tablet, the pen and the new wrist
mount on the left arm; serves joints and pictures to ``wrist_bench.py`` over a local socket.

    ~/workspace/venvs/isaacsim51/bin/python scripts/isaac/wrist_bench_server.py --config <config.json> --port 6398

Run by ``wrist_bench.py``. No hardware. The physics is the twin's (PhysX, EmbodiChain's drive gains); the display is an
emissive, glossy surface, so the lamps and the arm reflect in it.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing.connection import Listener
import pathlib
import sys
import traceback

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--port", type=int, default=6398)
OPTS = ap.parse_args()
CFG = json.loads(pathlib.Path(OPTS.config).read_text())
TWIN_DIR = pathlib.Path(CFG["twin_dir"])
sys.path.insert(0, str(TWIN_DIR))
sys.argv = [sys.argv[0]]
sys.dont_write_bytecode = True

import isaac_server as twin_server  # noqa: E402 - starts Isaac Sim
from isaac_server import APP  # noqa: E402
from isaac_server import Twin  # noqa: E402
from isaac_server import prim_world_matrix  # noqa: E402
from isaac_server import set_xform  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from pxr import Gf  # noqa: E402
from pxr import Sdf  # noqa: E402
from pxr import UsdGeom  # noqa: E402
from pxr import UsdShade  # noqa: E402
import trimesh  # noqa: E402

CV_TO_USD = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV camera axes -> USD's (-z forward, +y up)


class Bench(Twin):
    def build(self) -> dict:
        out = self.load(CFG["scene_dir"], CFG["options"])
        episode = json.loads((pathlib.Path(CFG["scene_dir"]) / "episode_000.json").read_text())
        self.reset(episode)
        stage = self.stage
        for paths in (self.rigid_paths.values(), [a for a in CFG.get("hide", [])]):
            for path in paths:
                prim = stage.GetPrimAtPath(path)
                if prim:
                    UsdGeom.Imageable(prim).MakeInvisible()
        for a in self.art_views.values():
            for key in ("path", "prim_path"):
                if isinstance(a.get(key), str):
                    UsdGeom.Imageable(stage.GetPrimAtPath(a[key])).MakeInvisible()
        for prim in stage.Traverse():  # the twin's top camera housing and bracket: the real lens stands a little elsewhere
            name = str(prim.GetPath()).lower()
            if "cam_high" not in name and ("top_mount" in name or "top_d435" in name or "bracket" in name) and prim.IsA(UsdGeom.Gprim):
                UsdGeom.Imageable(prim).MakeInvisible()
        self.link6 = self.links["left_link6"]
        self.arm_base = prim_world_matrix(stage.GetPrimAtPath(self.links[CFG["base_link"]]))
        self.left = [self.dof_names.index(f"left_joint{i}") for i in range(1, 7)]
        self._tool()
        self._tablet(np.array(CFG["screen_in_base"]), CFG["screen_size_m"])
        self._bench_cameras()
        self.command(CFG["park_q"], 1)
        q = self.robot.get_joint_positions()[0]
        q[self.left] = CFG["park_q"]
        for n in ("left_joint7", "left_joint8"):
            q[self.dof_names.index(n)] = 0.0105
            self.target[self.dof_names.index(n)] = 0.0105
        self.robot.set_joint_positions(q[None])
        self.command(CFG["park_q"], 30)
        for _ in range(12):
            self._render()
        return out | {"arm_base": self.arm_base.tolist()}

    # ---- what is added to the twin -----------------------------------------------------------------------------------
    def _mesh(self, path: str, mesh: trimesh.Trimesh, material) -> None:
        prim = UsdGeom.Mesh.Define(self.stage, path)
        prim.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in mesh.vertices])
        prim.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
        prim.CreateFaceVertexIndicesAttr(mesh.faces.ravel().tolist())
        prim.CreateSubdivisionSchemeAttr("none")
        self._bind(prim.GetPrim(), material)

    def _tool(self) -> None:
        """Mount, D435, holder and pen on the left gripper."""
        link = self.link6
        for child in self.stage.GetPrimAtPath(link).GetChildren():
            if child.GetName() == "cam_left_wrist":  # the twin's own wrist camera housing
                for g in child.GetChildren():
                    UsdGeom.Imageable(g).MakeInvisible()
        black, grey = self._material("tt_mount", (0.03, 0.03, 0.035), None, 0.6), self._material("tt_pen", (0.45, 0.45, 0.47), None, 0.35, metallic=0.5)
        blue = self._material("tt_holder", (0.16, 0.2, 0.3), None, 0.7)
        mount = trimesh.load(CFG["mount_stl"])  # mm: x robot's left, y up, z forward, origin at the flange centre
        v = mount.vertices * 1e-3
        mount.vertices = np.column_stack([-v[:, 1], v[:, 0], v[:, 2]])  # flange frame: -x up, y left, z forward
        self._mesh(link + "/tt_mount", mount, black)
        camera = np.array(CFG["wrist_in_flange"])
        body = camera @ CV_TO_USD
        shift = np.eye(4)
        shift[:3, 3] = (0.0325, 0.0, 0.0085 + 0.004)  # the D435 behind and to the right of its colour lens
        self._box(link + "/tt_d435", (0.090, 0.025, 0.025), body @ shift, self._material("tt_d435", (0.55, 0.56, 0.58), None, 0.35, metallic=0.6), collision=False)
        for name, (low, high) in {"tt_holder": CFG["holder_box"]}.items():
            low, high = np.array(low), np.array(high)
            m = np.eye(4)
            m[:3, 3] = 0.5 * (low + high)
            self._box(f"{link}/{name}", tuple(high - low), m, blue, collision=False)
        tip = float(CFG["tip_z"])
        pen = trimesh.creation.cylinder(0.0042, 0.13, sections=24)
        pen.apply_translation([0, 0, tip - 0.008 - 0.065])
        self._mesh(link + "/tt_pen", pen, grey)
        nib = trimesh.creation.icosphere(2, 0.0035)
        nib.apply_translation([0, 0, tip - 0.0035])
        self._mesh(link + "/tt_nib", nib, black)

    def _screen_material(self, name: str, texture: str) -> UsdShade.Material:
        path = f"/World/Looks/tt_screen_{name}"
        mat = UsdShade.Material.Define(self.stage, path)
        shader = UsdShade.Shader.Define(self.stage, path + "/shader")
        shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        shader.GetImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.004, 0.004, 0.004))
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.06)
        shader.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(0.5)
        shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
        shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 1, 1))
        shader.CreateInput("emissive_color_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
        shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(float(CFG["screen_nits"]))
        mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        return mat

    def _tablet(self, screen_in_base: np.ndarray, size) -> None:
        w, h = size
        UsdGeom.Xform.Define(self.stage, "/World/tablet")
        set_xform(self.stage.GetPrimAtPath("/World/tablet"), matrix=self.arm_base @ screen_in_base)
        b, thick = 0.009, 0.0075
        m = np.eye(4)
        m[:3, 3] = (w / 2, h / 2, thick / 2 + 0.0002)  # the screen frame's z points into the glass
        self._box("/World/tablet/body", (w + 2 * b, h + 2 * b, thick), m, self._material("tt_body", (0.02, 0.02, 0.022), None, 0.25), collision=False)
        mesh = UsdGeom.Mesh.Define(self.stage, "/World/tablet/screen")
        mesh.CreatePointsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(w, 0, 0), Gf.Vec3f(w, h, 0), Gf.Vec3f(0, h, 0)])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 3, 2, 1])  # seen from -z
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        uv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        uv.Set([Gf.Vec2f(0, 1), Gf.Vec2f(1, 1), Gf.Vec2f(1, 0), Gf.Vec2f(0, 0)])
        self.screen_mesh = mesh.GetPrim()
        self.screen_materials = {}

    def _bench_cameras(self) -> None:
        self.tt = {}
        for name, c in CFG["cameras"].items():
            parent = self.link6 if c["parent"] == "link6" else "/World"
            path = f"{parent}/tt_cam_{name}"
            cam = Camera(prim_path=path, resolution=(int(c["width"]), int(c["height"])))
            cam.initialize()
            prim = UsdGeom.Camera(self.stage.GetPrimAtPath(path))
            f, width, height = 2.0, c["width"], c["height"]
            prim.GetFocalLengthAttr().Set(f)
            prim.GetHorizontalApertureAttr().Set(width * f / c["fx"])
            prim.GetVerticalApertureAttr().Set(height * f / c["fy"])
            prim.GetHorizontalApertureOffsetAttr().Set(-(c["cx"] - width / 2) * f / c["fx"])
            prim.GetVerticalApertureOffsetAttr().Set((c["cy"] - height / 2) * f / c["fy"])
            prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.005, 50.0))
            pose = np.array(c["pose"]) @ CV_TO_USD
            set_xform(self.stage.GetPrimAtPath(path), matrix=pose if c["parent"] == "link6" else self.arm_base @ pose)
            self.tt[name] = cam

    # ---- the commands ------------------------------------------------------------------------------------------------
    def show(self, name: str, texture: str) -> dict:
        if name not in self.screen_materials:
            self.screen_materials[name] = self._screen_material(name, texture)
        UsdShade.MaterialBindingAPI.Apply(self.screen_mesh).Bind(self.screen_materials[name], UsdShade.Tokens.strongerThanDescendants)
        for _ in range(40):  # the texture streams in, and what the denoiser kept of the old picture fades
            self._render()
        return {"ok": True}

    def move_tablet(self, screen_in_base) -> dict:
        set_xform(self.stage.GetPrimAtPath("/World/tablet"), matrix=self.arm_base @ np.array(screen_in_base))
        return {"ok": True}

    def command(self, q, steps: int) -> dict:
        self.target[self.left] = np.asarray(q, float)
        self.robot.set_joint_position_targets(self.target[None])
        for _ in range(int(steps)):
            self.world.step(render=False)
        return self.joints()

    def joints(self) -> dict:
        return {"q": self.robot.get_joint_positions()[0][self.left].astype(float)}

    def flange(self) -> dict:
        self.world.step(render=False)
        link = prim_world_matrix(self.stage.GetPrimAtPath(self.link6))
        return {"pose": np.linalg.inv(self.arm_base) @ link}

    def picture(self, name: str, settle: int = 12) -> dict:
        for _ in range(settle):
            self._render()
        rgba = self.tt[name].get_rgba()
        return {"image": None if rgba is None or not rgba.size else np.ascontiguousarray(rgba[..., :3]).astype(np.uint8)}


def serve() -> None:
    bench = Bench()
    info = bench.build()
    listener = Listener(("127.0.0.1", OPTS.port), authkey=b"tablet-tap")
    print("wrist bench ready", flush=True)
    conn = listener.accept()
    while True:
        try:
            req = conn.recv()
        except (EOFError, ConnectionResetError):
            break
        try:
            cmd = req.pop("cmd")
            if cmd == "close":
                conn.send({"ok": True})
                break
            reply = info if cmd == "info" else getattr(bench, {"show": "show", "command": "command", "joints": "joints", "flange": "flange",
                                                                "picture": "picture", "light": "set_light_scale", "tablet": "move_tablet"}[cmd])(**req)  # fmt: skip
        except Exception as exc:  # noqa: BLE001 - reported to the client
            traceback.print_exc()
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        conn.send(reply)


if __name__ == "__main__":
    try:
        serve()
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        APP.close()
