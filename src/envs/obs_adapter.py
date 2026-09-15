"""robosuite observations to SmolVLA inputs.

Image orientation **is** handled here. robosuite renders with an OpenGL origin at
bottom-left, so raw frames arrive flipped vertically, and :func:`upright` corrects
them.

Phase 2 originally fixed this globally with robosuite's ``IMAGE_CONVENTION``
macro instead. That turned out to be wrong: robomimic flips every RGB observation
unconditionally (``env_robosuite.py:190``), so the two corrections composed and
Phase 3's regenerated dataset came out upside down. Each consumer now turns frames
upright at the point of use, which is what robomimic already assumes.

The state mapping is arbitrary and is documented as such. SmolVLA's base checkpoint
expects 6 values that are SO-100 joint angles in degrees. A Panda has 7 joints.
There is no correct answer, only a recorded choice -- see `docs/phase-2/result.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# SmolVLA declares three cameras. We have two and deliberately do not invent a
# third: `prepare_images` tolerates missing camera keys (empty_cameras=0 makes its
# padding loop break immediately), so the model simply receives two views.
#
# agentview is mapped first because the SO-100 datasets the checkpoint was trained
# on lead with a fixed scene camera and follow with the wrist view. That ordering is
# a guess about training-time convention, not a documented fact.
DEFAULT_CAMERA_MAP: dict[str, str] = {
    "agentview": "observation.images.camera1",
    "robot0_eye_in_hand": "observation.images.camera2",
}

# Panda joint limits are roughly +/-166 deg for the wide joints. The SO-100 action
# statistics span means of about 1.6 to 125 with stds of 14 to 59, so degrees put
# the two on a comparable scale without further rescaling.
PANDA_JOINTS_USED = 5
GRIPPER_SCALE_DEG = 90.0
# arctan2 returns an angle in (-180, 180], so this is the exact bound of the raw
# state, not an estimate. Dividing by it maps the state into [-1, 1].
JOINT_RANGE_DEG = 180.0


def upright(image: np.ndarray) -> np.ndarray:
    """Turn a raw robosuite frame the right way up.

    robosuite renders in the OpenGL convention, origin bottom-left, so raw frames
    are flipped vertically. Verified by eye in Phase 2: without this, `agentview`
    puts the robot and floor at the bottom of the frame with the table above them,
    and the wrist camera puts the gripper fingers at the top.

    Use this anywhere a raw robosuite frame is shown or fed to a model. robomimic
    applies the same flip itself, so data coming back from robomimic is already
    upright and must not be passed through here.
    """
    return image[::-1]


@dataclass
class ObsAdapterStats:
    """Running record of what passed through, for post-rollout inspection."""

    n_calls: int = 0
    state_min: np.ndarray | None = None
    state_max: np.ndarray | None = None
    image_min: float = field(default=np.inf)
    image_max: float = field(default=-np.inf)

    def update(self, state: np.ndarray, images: list[np.ndarray]) -> None:
        self.n_calls += 1
        self.state_min = state if self.state_min is None else np.minimum(self.state_min, state)
        self.state_max = state if self.state_max is None else np.maximum(self.state_max, state)
        for img in images:
            self.image_min = min(self.image_min, float(img.min()))
            self.image_max = max(self.image_max, float(img.max()))


class ObsAdapter:
    """Convert a robosuite observation dict into SmolVLA's input dict."""

    def __init__(
        self,
        camera_map: dict[str, str] | None = None,
        state_dim: int = 6,
        normalize_state: bool = True,
    ) -> None:
        """
        Args:
            camera_map: robosuite camera name -> SmolVLA image key.
            state_dim: Number of state values the model expects.
            normalize_state: Scale the state to roughly ``[-1, 1]`` instead of
                handing the network raw degrees.

                This is not cosmetic. The checkpoint declares ``STATE: MEAN_STD``
                but ships **no** state statistics, so LeRobot's normalizer is a
                silent no-op (Phase 1). During training the state was normalised by
                the dataset's own statistics, so the network's input weights expect
                roughly unit-scale values. Feeding raw degrees puts the input two
                orders of magnitude out.

                Measured effect, 2 episodes on `Lift`: raw degrees drove action
                clipping to 64.3% with per-dimension means as far out as -9.3;
                scaling brought both back into range. See docs/phase-2/result.md.
        """
        self.camera_map = dict(camera_map or DEFAULT_CAMERA_MAP)
        self.state_dim = state_dim
        self.normalize_state = normalize_state
        self.stats = ObsAdapterStats()

    def __call__(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        images = []
        adapted: dict[str, np.ndarray] = {}

        for robosuite_cam, model_key in self.camera_map.items():
            raw = obs.get(f"{robosuite_cam}_image")
            if raw is None:
                raise KeyError(
                    f"camera {robosuite_cam!r} missing from observation; present: "
                    f"{sorted(k for k in obs if k.endswith('_image'))}"
                )
            if raw.dtype != np.uint8:
                raise TypeError(f"{robosuite_cam}: expected uint8, got {raw.dtype}")
            if raw.ndim != 3 or raw.shape[2] != 3:
                raise ValueError(f"{robosuite_cam}: expected (H, W, 3), got {raw.shape}")

            # Flip upright, then uint8 HWC [0, 255] -> float32 CHW [0, 1]. The policy
            # rescales to [-1, 1] for SigLIP itself (modeling_smolvla.py:360); doing
            # that here too would shift the input out of range.
            img = np.transpose(upright(raw), (2, 0, 1)).astype(np.float32) / 255.0
            adapted[model_key] = img
            images.append(img)

        state = self.build_state(obs)
        adapted["observation.state"] = state
        self.stats.update(state, images)
        return adapted

    def build_state(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Build the 6-value state vector.

        **This mapping is arbitrary.** The model wants 5 SO-100 arm joints plus a
        gripper, in degrees; a Panda has 7 joints plus a parallel-jaw gripper. The
        choice made here is to take the first 5 Panda joints and the gripper
        opening, all in degrees, because it at least preserves "joint angles, in
        degrees, gripper last" -- the only structure the two embodiments share.

        robosuite reports joint positions as cos/sin pairs rather than angles, so
        the angle is recovered with ``arctan2``.
        """
        cos = np.asarray(obs["robot0_joint_pos_cos"], dtype=np.float64)
        sin = np.asarray(obs["robot0_joint_pos_sin"], dtype=np.float64)
        joints_deg = np.degrees(np.arctan2(sin, cos))

        gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
        # Panda's two finger joints move in opposition; their separation is the
        # opening. Scaled to degrees purely so it shares the arm's units.
        opening = float(gripper_qpos[0] - gripper_qpos[1])
        gripper_deg = opening / 0.08 * GRIPPER_SCALE_DEG

        state = np.concatenate([joints_deg[:PANDA_JOINTS_USED], [gripper_deg]])

        if self.normalize_state:
            # No dataset statistics exist to normalise against, so this is a plain
            # scaling by the physical range rather than a mean/std normalisation.
            # It puts the values on the unit scale the network was trained to see.
            state = state / JOINT_RANGE_DEG

        if state.shape[0] != self.state_dim:
            raise ValueError(
                f"built state of {state.shape[0]} values, model expects {self.state_dim}"
            )
        return state.astype(np.float32)
