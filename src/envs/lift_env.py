"""Construction of the robosuite `Lift` environment used throughout the project.

Kept in one place so that every phase builds an identical environment. The camera
set and the controller are project-level decisions, not per-script ones.
"""

from __future__ import annotations

from typing import Any

# Must run before robosuite is imported anywhere in the process.
from .robosuite_compat import configure_rendering

configure_rendering()

import robosuite  # noqa: E402
from robosuite.controllers import load_controller_config  # noqa: E402

WRIST_CAMERA = "robot0_eye_in_hand"
THIRD_PERSON_CAMERA = "agentview"
DEFAULT_CAMERAS = (WRIST_CAMERA, THIRD_PERSON_CAMERA)


def make_lift_env(
    cameras: tuple[str, ...] = DEFAULT_CAMERAS,
    resolution: int = 256,
    controller: str = "OSC_POSE",
    robot: str = "Panda",
    control_freq: int = 20,
    horizon: int = 200,
    hard_reset: bool = False,
    **kwargs: Any,
):
    """Build the `Lift` environment with offscreen rendering.

    Args:
        cameras: Camera names to render. The wrist view is the primary input; the
            third-person view exists so that rollout failures are debuggable.
        resolution: Square render size. Phase 0 measured render cost as flat from
            128 to 512 px, so this is chosen on correctness grounds, not cost.
        controller: robosuite controller name. `OSC_POSE` gives a 7-dimensional
            action: 3 position deltas, 3 axis-angle rotation deltas, 1 gripper.
        robot: Robot model. Panda is 7-DoF with a parallel-jaw gripper.
        control_freq: Control frequency in Hz.
        horizon: Maximum steps per episode.
        hard_reset: Left **off**, against robosuite's default.

            With robosuite's default of True, every `reset()` destroys the sim and
            rebuilds it from XML, re-reading 66+ Panda mesh files plus the gripper
            and arena assets. On Windows those handles are not released, and after
            about ten episodes MuJoCo fails to open the next mesh with
            `ValueError: Error: resource not found via provider or OS filesystem:
            ...link0_vis_9.obj` -- a file that is present on disk the whole time.
            Observed crashing a 20-episode rollout at episode 10.

            With it off, `reset()` calls `sim.reset()` on the existing simulation.
            Verified over 25 consecutive resets: no failure, and all 25 episodes
            still get distinct cube positions, so domain randomisation is intact.

    Returns:
        A constructed robosuite environment.
    """
    return robosuite.make(
        env_name="Lift",
        robots=robot,
        controller_configs=load_controller_config(default_controller=controller),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=list(cameras),
        camera_heights=resolution,
        camera_widths=resolution,
        control_freq=control_freq,
        horizon=horizon,
        reward_shaping=False,
        hard_reset=hard_reset,
        **kwargs,
    )
