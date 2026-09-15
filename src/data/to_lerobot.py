"""Convert a regenerated robomimic HDF5 into a LeRobot dataset.

Two things here are load-bearing and easy to get wrong.

**The state is built with the same code the evaluator uses.** Training and
evaluation must agree on what `observation.state` means, down to the ordering and
the scaling. Rather than restating the construction, this module calls
:meth:`ObsAdapter.build_state`, so a change to one cannot silently desynchronise
the other.

**Images from robomimic are already upright and must not be flipped again.**
robosuite renders bottom-left-origin, and robomimic flips every RGB observation on
its way out (`env_robosuite.py:190`). The rollout path has no such flip, which is
why :func:`obs_adapter.upright` exists there. Applying it here as well would train
the model on upside-down frames and evaluate it on upright ones -- a failure that
shows up only as unexplained poor fine-tuning results.

Videos are disabled (`use_videos=False`). LeRobot's video path needs `torchcodec`,
whose shared libraries do not load on Windows without a system FFmpeg install, and
the frames are stored as PNG instead.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from src.envs.obs_adapter import DEFAULT_CAMERA_MAP, ObsAdapter

logger = logging.getLogger(__name__)

# A single fixed instruction is enough for Lift. The exact string is recorded in
# the dataset card because it must match what is passed at evaluation time.
DEFAULT_INSTRUCTION = "lift the cube"


def _episode_keys(handle: h5py.File) -> list[str]:
    return sorted(handle["data"].keys(), key=lambda s: int(s.split("_")[1]))


def build_features(
    camera_map: dict[str, str], resolution: int, state_dim: int, action_dim: int
) -> dict[str, dict[str, Any]]:
    """LeRobot feature schema for this dataset."""
    features: dict[str, dict[str, Any]] = {}
    for model_key in camera_map.values():
        features[model_key] = {
            "dtype": "image",
            "shape": (resolution, resolution, 3),
            "names": ["height", "width", "channels"],
        }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": [f"state_{i}" for i in range(state_dim)],
    }
    features["action"] = {
        "dtype": "float32",
        "shape": (action_dim,),
        # robosuite OSC_POSE: normalised delta pose plus gripper. Verified against
        # the dataset's own env_kwargs in Phase 3.
        "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"][:action_dim],
    }
    return features


def convert(
    source: Path,
    repo_id: str,
    root: Path,
    instruction: str = DEFAULT_INSTRUCTION,
    camera_map: dict[str, str] | None = None,
    fps: int = 20,
    n_episodes: int | None = None,
    normalize_state: bool = True,
) -> Any:
    """Convert `source` into a LeRobot dataset rooted at `root`.

    Args:
        source: Regenerated robomimic HDF5, from `scripts/regenerate_obs.py`.
        repo_id: LeRobot dataset identifier, e.g. ``local/lift_ph``.
        root: Directory to write the dataset into.
        instruction: Language task attached to every frame.
        camera_map: robosuite camera name -> LeRobot image key.
        fps: Control frequency the demonstrations were recorded at.
        n_episodes: Convert only the first N episodes.
        normalize_state: Passed through to the observation adapter; must match
            what the evaluator uses.

    Returns:
        The created `LeRobotDataset`.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    camera_map = dict(camera_map or DEFAULT_CAMERA_MAP)
    adapter = ObsAdapter(camera_map=camera_map, normalize_state=normalize_state)

    with h5py.File(source, "r") as f:
        episodes = _episode_keys(f)
        if n_episodes is not None:
            episodes = episodes[:n_episodes]

        first = f[f"data/{episodes[0]}"]
        for robosuite_cam in camera_map:
            key = f"{robosuite_cam}_image"
            if key not in first["obs"]:
                raise KeyError(
                    f"{key!r} missing from {source.name}. Regenerate with "
                    f"--cameras {' '.join(camera_map)}"
                )
        resolution = int(first["obs"][f"{next(iter(camera_map))}_image"].shape[1])
        action_dim = int(first["actions"].shape[1])

        sample_state = adapter.build_state({
            k: first["obs"][k][0] for k in ("robot0_joint_pos_cos",
                                            "robot0_joint_pos_sin",
                                            "robot0_gripper_qpos")
        })
        state_dim = int(sample_state.shape[0])

        features = build_features(camera_map, resolution, state_dim, action_dim)
        logger.info("features: %s", {k: v["shape"] for k, v in features.items()})

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=root,
            robot_type="panda",
            use_videos=False,
        )

        for index, episode in enumerate(episodes):
            group = f[f"data/{episode}"]
            actions = group["actions"][()]
            n_frames = actions.shape[0]

            images = {
                robosuite_cam: group["obs"][f"{robosuite_cam}_image"][()]
                for robosuite_cam in camera_map
            }
            joint_cos = group["obs"]["robot0_joint_pos_cos"][()]
            joint_sin = group["obs"]["robot0_joint_pos_sin"][()]
            gripper = group["obs"]["robot0_gripper_qpos"][()]

            for t in range(n_frames):
                frame: dict[str, Any] = {
                    # Already upright out of robomimic. Do not flip.
                    model_key: images[robosuite_cam][t]
                    for robosuite_cam, model_key in camera_map.items()
                }
                frame["observation.state"] = adapter.build_state({
                    "robot0_joint_pos_cos": joint_cos[t],
                    "robot0_joint_pos_sin": joint_sin[t],
                    "robot0_gripper_qpos": gripper[t],
                })
                frame["action"] = actions[t].astype(np.float32)
                frame["task"] = instruction
                dataset.add_frame(frame)

            dataset.save_episode()
            logger.info("episode %d/%d (%s): %d frames",
                        index + 1, len(episodes), episode, n_frames)

    return dataset
