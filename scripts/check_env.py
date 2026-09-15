"""Phase 0 regression check for the whole stack.

Verifies that MuJoCo renders offscreen, that robosuite builds a Lift environment
with both cameras, and that the GPU is visible to PyTorch with the VRAM expected.

Run this at the start of every later phase:

    python scripts/check_env.py

Exits 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.envs.robosuite_compat import configure_rendering  # noqa: E402

logger = logging.getLogger("check_env")

GIB = 1024**3
PACKAGES = ("torch", "mujoco", "robosuite", "robomimic", "lerobot", "numpy")


@dataclass
class Report:
    """Accumulates pass/fail rows so one failure does not hide the rest.

    WARN rows flag a working but degraded configuration. They are printed loudly
    and do not affect the exit code, so a performance problem stays visible without
    blocking the phases that only need correctness.
    """

    rows: list[tuple[str, str, str]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, "PASS" if ok else "FAIL", detail))
        return ok

    def warn(self, name: str, detail: str = "") -> None:
        self.rows.append((name, "WARN", detail))

    def fail(self, name: str, exc: BaseException) -> bool:
        return self.add(name, False, f"{type(exc).__name__}: {exc}")

    @property
    def ok(self) -> bool:
        return all(status != "FAIL" for _, status, _ in self.rows)

    @property
    def warnings(self) -> int:
        return sum(status == "WARN" for _, status, _ in self.rows)

    def render(self) -> str:
        width = max(len(n) for n, _, _ in self.rows)
        return "\n".join(
            f"  [{status}] {name:<{width}}  {detail}"
            for name, status, detail in self.rows
        )


def check_versions(report: Report) -> None:
    report.add("python", sys.version_info[:2] >= (3, 10),
               f"{platform.python_version()} on {platform.system()}")
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - report, do not abort
            report.add(f"import {name}", False, f"{type(exc).__name__}: {exc}")
            continue
        report.add(f"import {name}", True, getattr(mod, "__version__", "unknown"))


def check_cuda(report: Report, budget_gb: float) -> None:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        report.fail("cuda available", exc)
        return

    if not report.add("cuda available", torch.cuda.is_available(),
                      f"torch {torch.__version__}"):
        return

    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info(0)
    report.add("gpu device", True, name)
    report.add("gpu vram", total / GIB >= budget_gb,
               f"{total / GIB:.2f} GiB total, {free / GIB:.2f} GiB free "
               f"(budget {budget_gb:.1f} GiB)")
    report.add("cuda runtime", True,
               f"cuda {torch.version.cuda}, capability "
               f"{'.'.join(str(c) for c in torch.cuda.get_device_capability(0))}")


def check_gl_backend(report: Report, backend: str) -> None:
    report.add("mujoco_gl backend", True, backend or "<platform default>")


def check_render_gpu(report: Report) -> None:
    """Report which GPU actually serves robosuite's OpenGL context.

    This must be queried from inside the live robosuite context. Creating a
    separate GLFW probe context answers a different question and can report the
    discrete GPU while robosuite is in fact rendering on the integrated one.

    On a hybrid laptop, landing on the integrated GPU is a large and completely
    silent performance loss. Warn rather than fail: rendering is still correct.
    """
    try:
        from OpenGL import GL

        vendor = GL.glGetString(GL.GL_VENDOR).decode()
        renderer = GL.glGetString(GL.GL_RENDERER).decode()
    except Exception as exc:  # noqa: BLE001
        report.warn("render gpu", f"could not query ({exc})")
        return

    discrete = "nvidia" in vendor.lower() or "nvidia" in renderer.lower()
    if discrete:
        report.add("render gpu", True, renderer)
    else:
        report.warn(
            "render gpu",
            f"{renderer} -- integrated GPU, not the RTX 4050. Rendering is "
            "several times slower and much less stable here. Fix: Windows "
            "Settings > System > Display > Graphics, add this venv's "
            "python.exe and set it to High performance.",
        )


def check_env_step(report: Report, cfg: dict) -> None:
    env_cfg = cfg["env"]
    cameras = list(env_cfg["cameras"])
    res = int(env_cfg["resolution"])

    try:
        import numpy as np
        import robosuite
        from robosuite.controllers import load_controller_config
    except Exception as exc:  # noqa: BLE001
        report.fail("build env", exc)
        return

    try:
        import torch

        cuda = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        cuda = False

    free_before = torch.cuda.mem_get_info(0)[0] if cuda else 0

    try:
        t0 = time.perf_counter()
        env = robosuite.make(
            env_name=env_cfg["name"],
            robots=env_cfg["robot"],
            controller_configs=load_controller_config(
                default_controller=env_cfg["controller"]
            ),
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            use_object_obs=True,
            camera_names=cameras,
            camera_heights=res,
            camera_widths=res,
            control_freq=int(env_cfg["control_freq"]),
            horizon=10_000,
        )
        build_s = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        report.fail("build env", exc)
        return

    report.add("build env", True,
               f"{env_cfg['name']}/{env_cfg['robot']}/{env_cfg['controller']}, "
               f"action_dim={env.action_dim}, built in {build_s:.2f}s")

    try:
        obs = env.reset()
        obs, _, _, _ = env.step(np.zeros(env.action_dim))

        # Only valid once the context exists and has rendered at least once.
        check_render_gpu(report)

        for cam in cameras:
            key = f"{cam}_image"
            if key not in obs:
                report.add(f"camera {cam}", False, f"{key} missing from observation")
                continue
            img = obs[key]
            ok = (
                img.shape == (res, res, 3)
                and img.dtype == np.uint8
                and float(img.std()) > 1.0
            )
            report.add(
                f"camera {cam}", ok,
                f"shape={img.shape} dtype={img.dtype} "
                f"min={img.min()} max={img.max()} std={img.std():.1f}"
                + ("" if ok else "  <- blank or wrong shape/dtype"),
            )

        report.add("proprio state", "robot0_proprio-state" in obs,
                   f"dim={obs.get('robot0_proprio-state', np.empty(0)).shape[0]}")

        if cuda:
            free_after = torch.cuda.mem_get_info(0)[0]
            used = (free_before - free_after) / GIB
            report.add("render vram", used < cfg["vram"]["budget_gb"],
                       f"{used:.2f} GiB consumed by the render context")

        bench = cfg.get("benchmark", {})
        warmup = int(bench.get("warmup_steps", 15))
        timed = int(bench.get("timed_steps", 50))
        if timed > 0:
            action = np.zeros(env.action_dim)
            for _ in range(warmup):
                env.step(action)
            samples = []
            for _ in range(timed):
                t0 = time.perf_counter()
                env.step(action)
                samples.append(time.perf_counter() - t0)
            arr = np.array(samples) * 1000.0
            report.add(
                f"per-step cost ({len(cameras)} cam @ {res}px)", True,
                f"mean={arr.mean():.2f}ms p95={np.percentile(arr, 95):.2f}ms "
                f"-> {arr.mean() * 200 / 1000:.1f}s per 200-step episode",
            )
    except Exception as exc:  # noqa: BLE001
        report.fail("step env", exc)
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs" / "phase0_check.yaml")
    parser.add_argument("--resolution", type=int, default=None,
                        help="override env.resolution")
    parser.add_argument("--no-bench", action="store_true",
                        help="skip the per-step timing loop")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.resolution is not None:
        cfg["env"]["resolution"] = args.resolution
    if args.no_bench:
        cfg["benchmark"]["timed_steps"] = 0

    # Must happen before robosuite is imported anywhere.
    backend = configure_rendering()

    report = Report()
    check_versions(report)
    check_cuda(report, float(cfg["vram"]["budget_gb"]))
    check_gl_backend(report, backend)
    check_env_step(report, cfg)

    print(f"\nenvironment check  ({args.config.name})")
    print(report.render())
    verdict = "ALL CHECKS PASSED" if report.ok else "FAILURES PRESENT"
    if report.warnings:
        verdict += f"  ({report.warnings} warning(s) -- working but degraded)"
    print(f"\n{verdict}\n")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
