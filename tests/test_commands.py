"""The command registry and the mapping from errors to exit codes."""

import pytest

from tablet_tap import errors
from tablet_tap.app import ArmUse
from tablet_tap.cli import commands
from tablet_tap.cli import main as cli
from tablet_tap.robot.arm import FakeArm


def test_every_command_is_registered_once_with_help():
    registered = commands.commands()
    names = [c.name for c in registered]
    assert names == ["status", "recover", "limits", "tablet", "teach", "view", "board", "place", "probe", "taptest", "park", "hover", "tap", "swipe",
                     "home", "back", "shot", "run", "wrist"]  # fmt: skip
    assert all(c.help for c in registered)
    by_name = {c.name: c for c in registered}
    assert by_name["status"].arm is ArmUse.READS and by_name["view"].arm is ArmUse.NONE
    assert by_name["recover"].arm is ArmUse.NONE and by_name["tap"].arm is ArmUse.MOVES
    assert by_name["place"].arm is ArmUse.NONE and by_name["board"].arm is ArmUse.MOVES
    parser = cli.build_parser()
    assert by_name["teach"].arm_use(parser.parse_args(["teach", "--read_only"])) is ArmUse.READS
    assert by_name["teach"].arm_use(parser.parse_args(["teach"])) is ArmUse.MOVES
    assert parser.parse_args(["tap", "0.2", "0.3", "--park"]).park
    with pytest.raises(SystemExit):
        parser.parse_args(["hover", "--park"])  # hovering and parking at once makes no sense


@pytest.mark.parametrize(
    ("error", "code", "said"),
    [
        (errors.Stopped("x"), 130, "已停止：x"),
        (errors.PathError("x"), 2, "未执行：x"),
        (errors.Unreachable("x"), 2, "未执行：x"),
        (errors.ArmConnectionError("x"), 3, "x"),
        (errors.ArmBusyError("x"), 3, "x"),
        (errors.SetupError("x"), 1, "x"),
        (errors.CalibrationError("x"), 1, "x"),
    ],
)
def test_errors_become_exit_codes(error, code, said, monkeypatch, capsys, tmp_path):
    status = next(c for c in commands.commands() if c.name == "status")

    def fail(ctx, args):
        raise error

    monkeypatch.setattr(type(status), "run", lambda self, ctx, args: fail(ctx, args))
    assert cli.main(["--setup", str(tmp_path / "s.json"), "--fake", "status"]) == code
    assert capsys.readouterr().out.strip() == said


def test_missing_calibration_is_explained_not_a_traceback(tmp_path, capsys, monkeypatch):
    assert cli.main(["--setup", str(tmp_path / "s.json"), "--fake", "tap", "0.5", "0.5"]) == 1
    assert "teach" in capsys.readouterr().out
    monkeypatch.delenv("TABLET_TAP_NO_HARDWARE")  # no channel is given, so nothing can be connected to
    assert cli.main(["--setup", str(tmp_path / "s.json"), "status"]) == 1  # a real arm, but no channel given
    assert "--channel" in capsys.readouterr().out


def test_the_hardware_lock_keeps_the_command_line_off_the_bench(tmp_path, capsys, monkeypatch):
    """With TABLET_TAP_NO_HARDWARE set (as in every test) nothing may open the arm, its CAN channel or a camera."""
    opened = []
    arm = FakeArm([0.0, 0.6, -0.5, 0.0, 0.3, 0.0])
    monkeypatch.setattr("tablet_tap.app.open_arm", lambda *a, **k: (opened.append("arm"), arm)[1])
    monkeypatch.setattr("tablet_tap.app.open_camera", lambda *a, **k: opened.append("camera"))
    monkeypatch.setattr("tablet_tap.cli.commands.status.describe_bus", lambda *a, **k: opened.append("can"))
    (tmp_path / "s.json").write_text('{"park_q": [0, 0.6, -0.5, 0, 0.3, 0]}')
    base = ["--setup", str(tmp_path / "s.json"), "--channel", "can_r_follower", "--camera", "realsense:123:1280x720"]
    for command in (["status"], ["teach", "--read_only", "--no_page"], ["park"], ["view"], ["recover", "--status_only"],
                    ["--dry_run", "status"]):  # fmt: skip
        argv = [*command[:-1], *base, command[-1]] if command[0].startswith("--") else [*base, *command]
        assert cli.main(argv) == 1, command
        assert "TABLET_TAP_NO_HARDWARE" in capsys.readouterr().out
    assert opened == []
    monkeypatch.delenv("TABLET_TAP_NO_HARDWARE")  # without the lock the same command goes on to the arm
    assert cli.main([*base, "status"]) == 0
    assert opened == ["can", "arm"]


def test_the_top_camera_is_the_default(tmp_path):
    from tablet_tap import config
    from tablet_tap.app import AppContext
    from tablet_tap.app import ArmUse
    from tablet_tap.app import Options

    parser = cli.build_parser()

    def context(*argv):
        return AppContext(
            Options.from_args(parser.parse_args(["--setup", str(tmp_path / "s.json"), *argv])), ArmUse.NONE
        )

    assert context("--fake", "view").setup.camera == config.BENCH_TOP_CAMERA == "realsense:262422077281:1920x1080"
    chosen = context("--fake", "--camera", "cv:0", "view")
    assert chosen.setup.camera == "cv:0"
    chosen.store.save(chosen.setup)
    assert context("--fake", "view").setup.camera == "cv:0"  # what the setup file names wins over the default
