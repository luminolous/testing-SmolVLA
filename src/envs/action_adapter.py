"""SmolVLA actions to robosuite actions.

**Read this before trusting any rollout number produced through this module.**

There is no valid mapping between the two action spaces. SmolVLA's base checkpoint
emits 6 values that are SO-100 arm joint targets in degrees. robosuite's `OSC_POSE`
controller consumes 7 values that are a normalised end-effector delta pose plus a
gripper command:

| robosuite index | meaning            | range   |
| --------------- | ------------------ | ------- |
| 0, 1, 2         | position delta     | [-1, 1] |
| 3, 4, 5         | axis-angle delta   | [-1, 1] |
| 6               | gripper            | [-1, 1] |

Different dimensionality, different semantics, different embodiment. Phase 1
established this from the checkpoint's own statistics.

What this module implements is therefore an *explicitly arbitrary* mapping, kept
only so that the rollout loop can be exercised end to end. The one real
correspondence -- the last dimension is the gripper on both sides -- is preserved,
and the rest is a straight positional copy. Success rates measured through this are
a floor and an integration test, not a measure of transfer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

ActionSpace = Literal["normalized", "degrees"]

OSC_ACTION_DIM = 7
GRIPPER_INDEX = 6
# Model dims 0-4 (SO-100 arm joints) copied onto robosuite dims 0-4
# (dx, dy, dz, drx, dry). robosuite dim 5 (drz) has no counterpart and is held at 0
# rather than being fed an unrelated joint.
ARM_DIMS = 5
YAW_INDEX = 5


@dataclass
class ActionAdapterStats:
    """Clipping frequency is the diagnostic this class exists to produce.

    `phase-2-zero-shot.md` §2.3: frequent clipping is a strong signal that
    denormalisation is wrong. Here it also measures how badly the arbitrary mapping
    fits.
    """

    n_calls: int = 0
    n_clipped_values: int = 0
    n_values: int = 0
    per_dim_clipped: np.ndarray = field(
        default_factory=lambda: np.zeros(OSC_ACTION_DIM, dtype=np.int64)
    )
    raw_sum: np.ndarray = field(default_factory=lambda: np.zeros(OSC_ACTION_DIM))
    raw_sq_sum: np.ndarray = field(default_factory=lambda: np.zeros(OSC_ACTION_DIM))

    def update(self, raw: np.ndarray, clipped_mask: np.ndarray) -> None:
        self.n_calls += 1
        self.n_values += raw.size
        self.n_clipped_values += int(clipped_mask.sum())
        self.per_dim_clipped += clipped_mask.astype(np.int64)
        self.raw_sum += raw
        self.raw_sq_sum += raw**2

    @property
    def clip_rate(self) -> float:
        return self.n_clipped_values / self.n_values if self.n_values else 0.0

    @property
    def per_dim_clip_rate(self) -> np.ndarray:
        if self.n_calls == 0:
            return np.zeros(OSC_ACTION_DIM)
        return self.per_dim_clipped / self.n_calls

    @property
    def raw_mean(self) -> np.ndarray:
        if self.n_calls == 0:
            return np.zeros(OSC_ACTION_DIM)
        return self.raw_sum / self.n_calls

    @property
    def raw_std(self) -> np.ndarray:
        if self.n_calls == 0:
            return np.zeros(OSC_ACTION_DIM)
        var = self.raw_sq_sum / self.n_calls - self.raw_mean**2
        return np.sqrt(np.maximum(var, 0.0))


class ActionAdapter:
    """Map a SmolVLA action onto a robosuite `OSC_POSE` action."""

    def __init__(self, action_space: ActionSpace = "normalized") -> None:
        """
        Args:
            action_space: Which units the incoming action is in.

                ``"normalized"`` -- the checkpoint as shipped, whose output is
                roughly unit-scale. That happens to be the range `OSC_POSE` wants,
                so clipping stays informative.

                ``"degrees"`` -- statistics bound, so values span roughly
                ``[-150, 180]``. Feeding those to a controller expecting ``[-1, 1]``
                saturates essentially every dimension. Included because §2.3 asks
                for clipping frequency as a diagnostic, and running both settings
                turns that into an actual measurement.
        """
        self.action_space = action_space
        self.stats = ActionAdapterStats()

    def __call__(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64).squeeze()
        if action.ndim != 1:
            raise ValueError(f"expected a 1-D action, got shape {action.shape}")
        if action.shape[0] < ARM_DIMS + 1:
            raise ValueError(
                f"expected at least {ARM_DIMS + 1} values, got {action.shape[0]}"
            )

        # Order matters and follows §2.3: units first, then convention, then
        # clipping. There is no denormalisation step here -- that happens upstream
        # in SmolVLAWrapper, which owns the checkpoint statistics.
        raw = np.zeros(OSC_ACTION_DIM, dtype=np.float64)
        raw[:ARM_DIMS] = action[:ARM_DIMS]
        raw[YAW_INDEX] = 0.0
        raw[GRIPPER_INDEX] = action[ARM_DIMS]

        clipped = np.clip(raw, -1.0, 1.0)
        self.stats.update(raw, clipped != raw)
        return clipped.astype(np.float64)

    def describe_mapping(self) -> str:
        """Human-readable record of the mapping, for the run log."""
        lines = [
            f"action space in : {self.action_space}",
            "model dim -> robosuite dim (OSC_POSE)",
        ]
        names = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
        for i in range(ARM_DIMS):
            lines.append(f"    {i} -> {i} ({names[i]})    ARBITRARY")
        lines.append(f"    - -> {YAW_INDEX} ({names[YAW_INDEX]})   held at 0, no counterpart")
        lines.append(f"    {ARM_DIMS} -> {GRIPPER_INDEX} ({names[GRIPPER_INDEX]})"
                     f"  the one real correspondence")
        return "\n".join(lines)
