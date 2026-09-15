"""Tests for the Phase 2 observation and action adapters.

The observation tests run against a real robosuite observation rather than a
synthetic dict. Shape and dtype bugs survive synthetic fixtures happily; what they
do not survive is a real environment, which is where camera key names, value ranges
and the cos/sin joint encoding actually come from.

Run: python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.envs.action_adapter import (  # noqa: E402
    ARM_DIMS,
    GRIPPER_INDEX,
    OSC_ACTION_DIM,
    YAW_INDEX,
    ActionAdapter,
)
from src.envs.obs_adapter import DEFAULT_CAMERA_MAP, ObsAdapter  # noqa: E402

RESOLUTION = 128


@pytest.fixture(scope="module")
def real_observation():
    """One observation from a stepped Lift environment."""
    from src.envs.lift_env import make_lift_env

    env = make_lift_env(resolution=RESOLUTION)
    try:
        env.reset()
        # Step a few times so the arm is not in its exact reset pose, which would
        # let a broken joint decode pass by looking plausible.
        for _ in range(10):
            obs, *_ = env.step(np.zeros(env.action_dim))
        yield obs
    finally:
        env.close()


# --- observation adapter ---------------------------------------------------------


def test_produces_every_declared_key(real_observation):
    adapted = ObsAdapter()(real_observation)
    for model_key in DEFAULT_CAMERA_MAP.values():
        assert model_key in adapted
    assert "observation.state" in adapted


def test_images_are_chw_float_unit_range(real_observation):
    adapted = ObsAdapter()(real_observation)
    for model_key in DEFAULT_CAMERA_MAP.values():
        img = adapted[model_key]
        assert img.dtype == np.float32
        assert img.shape == (3, RESOLUTION, RESOLUTION), f"{model_key}: {img.shape}"
        assert 0.0 <= img.min() and img.max() <= 1.0
        # The policy rescales to [-1, 1] itself; anything already negative here
        # means the conversion was applied twice.
        assert img.min() >= 0.0


def test_cameras_are_not_swapped(real_observation):
    """The two views must not be the same array, nor crossed over.

    A swap is invisible to every shape assertion and silently ruins a rollout, so
    it is checked directly against the source observation.
    """
    adapted = ObsAdapter()(real_observation)
    wrist = adapted[DEFAULT_CAMERA_MAP["robot0_eye_in_hand"]]
    scene = adapted[DEFAULT_CAMERA_MAP["agentview"]]
    assert not np.allclose(wrist, scene), "both keys carry the same image"

    def to_chw(raw):
        return np.transpose(raw, (2, 0, 1)).astype(np.float32) / 255.0

    np.testing.assert_allclose(
        wrist, to_chw(real_observation["robot0_eye_in_hand_image"]), rtol=1e-6
    )
    np.testing.assert_allclose(
        scene, to_chw(real_observation["agentview_image"]), rtol=1e-6
    )


def test_images_are_not_blank(real_observation):
    adapted = ObsAdapter()(real_observation)
    for model_key in DEFAULT_CAMERA_MAP.values():
        assert adapted[model_key].std() > 0.01, f"{model_key} looks blank"


def test_state_shape_and_finiteness(real_observation):
    state = ObsAdapter()(real_observation)["observation.state"]
    assert state.shape == (6,)
    assert state.dtype == np.float32
    assert np.all(np.isfinite(state))


def test_raw_state_joints_are_degrees(real_observation):
    """arctan2 returns radians; the adapter must convert. A missing conversion
    leaves every joint inside [-pi, pi], which this catches."""
    state = ObsAdapter(normalize_state=False)(real_observation)["observation.state"]
    assert np.all(np.abs(state[:ARM_DIMS]) <= 180.0 + 1e-6)

    cos = real_observation["robot0_joint_pos_cos"]
    sin = real_observation["robot0_joint_pos_sin"]
    expected = np.degrees(np.arctan2(sin, cos))[:ARM_DIMS]
    np.testing.assert_allclose(state[:ARM_DIMS], expected, rtol=1e-4)


def test_normalized_state_is_unit_scale(real_observation):
    """The default path must hand the network unit-scale values.

    The checkpoint ships no state statistics, so LeRobot's normalizer is a no-op and
    raw degrees would reach the network two orders of magnitude out of range.
    """
    state = ObsAdapter(normalize_state=True)(real_observation)["observation.state"]
    assert np.all(np.abs(state) <= 1.0 + 1e-6), state

    raw = ObsAdapter(normalize_state=False)(real_observation)["observation.state"]
    np.testing.assert_allclose(state, raw / 180.0, rtol=1e-5)


def test_missing_camera_is_an_error(real_observation):
    obs = {k: v for k, v in real_observation.items() if k != "agentview_image"}
    with pytest.raises(KeyError, match="agentview"):
        ObsAdapter()(obs)


# --- action adapter --------------------------------------------------------------


def test_output_shape_and_range():
    adapter = ActionAdapter()
    out = adapter(np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6]))
    assert out.shape == (OSC_ACTION_DIM,)
    assert np.all(out >= -1.0) and np.all(out <= 1.0)


def test_arm_dims_copied_and_gripper_is_last():
    adapter = ActionAdapter()
    action = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    out = adapter(action)
    np.testing.assert_allclose(out[:ARM_DIMS], action[:ARM_DIMS])
    assert out[GRIPPER_INDEX] == pytest.approx(action[ARM_DIMS])


def test_yaw_is_held_at_zero():
    out = ActionAdapter()(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    assert out[YAW_INDEX] == 0.0


def test_clipping_is_applied_and_counted():
    adapter = ActionAdapter()
    out = adapter(np.array([5.0, -5.0, 0.0, 0.0, 0.0, 0.0]))
    assert out[0] == 1.0
    assert out[1] == -1.0
    assert adapter.stats.n_clipped_values == 2
    assert adapter.stats.per_dim_clip_rate[0] == 1.0
    assert adapter.stats.per_dim_clip_rate[2] == 0.0


def test_degree_scale_actions_saturate():
    """Unnormalised SmolVLA output spans roughly [-150, 180]. Fed to a controller
    expecting [-1, 1] it must saturate -- this is the §2.3 diagnostic."""
    adapter = ActionAdapter(action_space="degrees")
    for _ in range(10):
        adapter(np.array([120.0, -95.0, 60.0, -30.0, 45.0, 12.0]))
    assert adapter.stats.clip_rate > 0.8


def test_normalized_actions_mostly_survive():
    adapter = ActionAdapter()
    rng = np.random.default_rng(0)
    for _ in range(200):
        adapter(rng.standard_normal(6) * 0.5)
    assert adapter.stats.clip_rate < 0.05


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        ActionAdapter()(np.zeros((2, 6)))
    with pytest.raises(ValueError):
        ActionAdapter()(np.zeros(3))


def test_stats_track_raw_distribution():
    adapter = ActionAdapter()
    for _ in range(100):
        adapter(np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert adapter.stats.raw_mean[0] == pytest.approx(0.5)
    assert adapter.stats.raw_std[0] == pytest.approx(0.0, abs=1e-9)
