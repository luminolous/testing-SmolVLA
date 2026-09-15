# Phase 2 — Action Mapping and Zero-Shot Rollout: Results

**Status:** adapters built and tested; loop verified end to end; the 20-episode
baseline is queued for the user to run
**Approach:** option A — Phase 2 is run as an **integration test and a floor**, not
as a measure of transfer. See "Why the baseline cannot measure transfer" below.

---

## Why the baseline cannot measure transfer

Phase 1 established, from the checkpoint's own statistics, that the two action
spaces do not correspond:

| | SmolVLA base | robosuite `Lift` / Panda / `OSC_POSE` |
| --- | --- | --- |
| Action dim | 6 | 7 |
| Meaning | Absolute SO-100 joint angles | End-effector delta pose + gripper |
| Units | Degrees | Normalised `[-1, 1]` |
| Embodiment | SO-100, 5 joints + gripper | Panda, 7 joints + parallel gripper |

`phase-2-zero-shot.md` §2.1 asks for a dimension-by-dimension correspondence. None
exists, and changing robot does not create one — robosuite has no 5-DoF arm, and a
UR5e under joint control is still 6 joints plus a gripper.

The mapping implemented in
[`action_adapter.py`](../../src/envs/action_adapter.py) is therefore **explicitly
arbitrary**, and says so in its own module docstring:

```
model dim -> robosuite dim (OSC_POSE)
    0 -> 0 (dx)     ARBITRARY
    1 -> 1 (dy)     ARBITRARY
    2 -> 2 (dz)     ARBITRARY
    3 -> 3 (drx)    ARBITRARY
    4 -> 4 (dry)    ARBITRARY
    - -> 5 (drz)    held at 0, no counterpart
    5 -> 6 (gripper)  the one real correspondence
```

The single genuine correspondence — the last dimension is the gripper on both sides
— is preserved. Dimension 5 of the robosuite action (`drz`) is held at zero rather
than fed an unrelated joint value.

**What the run does therefore measure:** that the pipeline closes end to end without
crashes, shape errors, NaNs or VRAM creep; that the model produces varied,
in-range output from real simulator images; and a floor for Phase 4 to beat.

---

## The preprocessing bug this phase found

The first smoke rollout came back with **64.3% of all action values clipped** and
per-dimension means as far out as −9.3, on a model whose output Phase 1 measured at
roughly unit scale. Following the §2.6 diagnosis tree, saturated actions point back
at preprocessing, not at a domain gap.

The cause was in the state vector. The checkpoint declares `STATE: MEAN_STD` but
**ships no state statistics at all** (Phase 1), so LeRobot's normalizer is a silent
no-op. During training the state was normalised by the training dataset's own
statistics, so the network's input weights expect roughly unit-scale values. The
adapter was handing it raw joint angles in degrees, spanning ±168 — two orders of
magnitude out.

Scaling the state by its exact bound (`arctan2` returns an angle in (−180, 180], so
dividing by 180 is not an estimate) fixed it. Same seed, same 2 episodes, same
everything else:

| State fed to the model | Clip rate | `dx` mean | `dz` mean | `drx` mean | `grip` mean |
| --- | --- | --- | --- | --- | --- |
| Raw degrees | **64.3%** | 1.473 | −1.538 | −9.307 | 7.641 |
| Scaled to `[-1, 1]` | **9.9%** | 0.501 | −0.467 | −0.986 | 0.209 |

`normalize_state=True` is now the default in
[`obs_adapter.py`](../../src/envs/obs_adapter.py); `--raw-state` reproduces the
broken behaviour for comparison.

**This matters well beyond Phase 2.** Any future use of this checkpoint must supply
an already-scaled state, because LeRobot will not do it. Phase 3's converted dataset
must carry its own statistics.

### One dimension is still saturated

Even after the fix, `drx` sits at a mean of −0.986 with a **55% clip rate** while
every other dimension clips at 0–14%. That is the arbitrary mapping showing through:
model dimension 3 is an SO-100 elbow-ish joint whose distribution simply does not
resemble a normalised rotation delta. It is expected under an arbitrary mapping and
is not evidence of a further bug.

---

## Image orientation — resolved by looking

`phase-2-zero-shot.md` §2.2 is explicit that this must be verified rather than
assumed, and that a human eye is the right instrument. Frames were rendered both
ways and inspected.

robosuite's default `IMAGE_CONVENTION = "opengl"` returns frames **flipped
vertically**: `agentview` put the robot and the floor at the bottom of the frame
with the table above them, and the wrist camera put the gripper fingers at the top.
The flip is confirmed by both cameras.

The fix is `IMAGE_CONVENTION = "opencv"`, set in the generated
`robosuite/macros_private.py` by
[`robosuite_compat.py`](../../src/envs/robosuite_compat.py).

**Set there deliberately, not in the observation adapter.** Phase 3 regenerates
images through robomimic, which never touches our adapter. An adapter-side flip
would leave the rollout images and the training images silently disagreeing — which
is exactly the class of bug that only surfaces as inexplicably poor fine-tuning
results much later.

Sample frames: `results/phase-2/oriented_*.png`.

---

## Adapters

### Observation — [`src/envs/obs_adapter.py`](../../src/envs/obs_adapter.py)

| Aspect | Decision |
| --- | --- |
| Camera mapping | `agentview` → `observation.images.camera1`, `robot0_eye_in_hand` → `camera2` |
| Third camera | **Deliberately not supplied.** `prepare_images` tolerates missing keys (`empty_cameras=0` makes its padding loop break immediately), so the model receives two real views rather than one fabricated one |
| Image format | uint8 HWC `[0,255]` → float32 CHW `[0,1]`. The policy rescales to `[-1,1]` for SigLIP itself (`modeling_smolvla.py:360`); doing it here too would double-apply |
| Orientation | Not handled here — see above |
| State | First 5 Panda joints via `arctan2(sin, cos)` in degrees, plus gripper opening, then scaled by 180. **Arbitrary**, but preserves "joint angles, gripper last", the only shared structure |

The camera ordering is a guess about training-time convention — SO-100 datasets lead
with a fixed scene camera — not a documented fact.

### Action — [`src/envs/action_adapter.py`](../../src/envs/action_adapter.py)

Order follows §2.3: units, then convention, then clipping. There is no
denormalisation step inside the adapter; that belongs upstream in `SmolVLAWrapper`,
which owns the checkpoint statistics.

Two action spaces are selectable so that clipping frequency becomes a measurement
rather than an assertion:

- `normalized` (default) — the checkpoint as shipped, output at roughly unit scale,
  which is the range `OSC_POSE` wants.
- `degrees` — statistics bound, output spanning roughly `[-150, 180]`. Saturates
  essentially every dimension, as §2.3 predicts for wrong denormalisation.

---

## Tests

`python -m pytest tests/ -v` — **16 passed**.

The observation tests run against a **real stepped robosuite observation**, not a
synthetic dict. Camera key names, value ranges and the cos/sin joint encoding all
come from the real environment, and synthetic fixtures happily hide mistakes in all
three. Coverage includes:

- Images are CHW float32 in `[0,1]`, non-blank, and **not swapped** — checked
  against the source observation, since a swap passes every shape assertion.
- Joint decode matches `degrees(arctan2(sin, cos))`, catching a missing
  radians-to-degrees conversion.
- The normalised state path is exactly the raw path over 180.
- Degree-scale actions saturate; normalised ones mostly survive.
- Clipping is applied and counted per dimension.

---

## 2.4 Sanity rollout — passed

2 episodes, 60-step horizon.

| Check | Result |
| --- | --- |
| Loop closes end to end | yes |
| Crashes / shape errors / NaNs | none |
| VRAM across episodes | 1.75 → 1.75 GiB, **flat, no creep** |
| Peak VRAM allocated | 1.752 GiB |
| Device VRAM in use | 2.887 GiB of 6.00 GiB |

### Timing observed during rollout

| Measurement | Rollout | Phase 0/1 in isolation |
| --- | --- | --- |
| Env step | 33.7 ms | 7.4 ms (Phase 0, 256 px) |
| Inference per call | 717.8 ms | 399.7 ms (Phase 1) |

Both are roughly 2–4× worse than measured in isolation. The likely cause is
contention between the CUDA context doing inference and the OpenGL context doing
offscreen rendering on the same GPU, with the small call count (4) also including a
cold first call. Worth confirming against the longer run rather than treating as
settled.

---

## The reset handle leak

The first 20-episode attempt **crashed at episode 10**, inside `env.reset()`:

```
ValueError: Error: resource not found via provider or OS filesystem:
  ...robosuite\models\assets\robots\panda\obj_meshes/link0_vis/link0_vis_9.obj
```

The file is present on disk the whole time, and episodes 0–9 had already used it.
A missing asset would have failed at episode 0, so this is resource exhaustion, not
a broken install.

**Cause.** robosuite defaults to `hard_reset=True`, which destroys the simulation
and rebuilds it from XML on *every* reset, re-reading 66+ Panda meshes plus gripper
and arena assets each time. On Windows those file handles are not released, and
around the tenth episode MuJoCo can no longer open the next mesh.

**Fix.** `hard_reset=False` in
[`lift_env.py`](../../src/envs/lift_env.py), so `reset()` calls `sim.reset()` on the
existing simulation.

Verified on two counts, because the cheap fix here would have quietly destroyed the
evaluation:

| Check | Result |
| --- | --- |
| 25 consecutive resets | all succeeded, no failure |
| Distinct cube positions across those 25 | **25 unique** — randomisation intact |
| 14 episodes through `rollout.py`, past the old crash point | no crash, VRAM flat at 1.75 GiB |

Had randomisation been lost, every episode would have been identical and the
success rate meaningless while still looking like a clean run.

**Carry this into Phase 3.** Dataset regeneration replays hundreds of demonstrations
and resets far more often than 20 times. Any regeneration path that builds its
environment outside `make_lift_env` — robomimic constructs its own — will hit the
same leak, much sooner.

---

## 2.5 Baseline evaluation — **queued for the user**

```bash
python scripts/rollout.py --n-episodes 20 --render
```

| | |
| --- | --- |
| Expected runtime | ~4 minutes (20 × ~10 s, plus 21 s model load) |
| Expected peak VRAM | ~1.8 GiB allocated, ~2.9 GiB device total — well inside the 5 GiB budget |
| Output | `results/phase-2/<timestamp>/` — resolved config, `metrics.json`, `rollout.log`, and camera frames from episode 0 |
| Expected result | **0% success.** Under an arbitrary action mapping this is the honest prediction, not a disappointment |

_Results table to be filled in once the run completes._

For the contrast run that demonstrates the denormalisation diagnostic:

```bash
python scripts/rollout.py --n-episodes 5 --stats-key so100 --action-space degrees
```

---

## 2.6 Diagnosis

_To be written against the 20-episode numbers._

What can already be said: the outcome is **not** the "actions near-constant or
saturated" branch of §2.6 — that branch was entered, traced to the state scaling,
and fixed. Nor can it be the "genuine domain gap" branch in any meaningful sense,
because a domain gap presupposes a shared action space. The correct reading is a
third case the phase document does not enumerate: **structural incompatibility**,
which fine-tuning in Phases 3 and 4 is exactly the remedy for.

---

## Findings carried forward

| Finding | Matters in | Note |
| --- | --- | --- |
| `hard_reset=False` is required on Windows | **Phase 3** | Otherwise mesh file handles leak and MuJoCo fails after ~10 resets. Regeneration resets far more often than that, and robomimic builds its own environment |
| State must be scaled before the model sees it | **Phase 3, 4** | LeRobot will not do it — no statistics ship with the checkpoint. The converted dataset must carry its own |
| `IMAGE_CONVENTION = "opencv"` is required | **Phase 3** | Set globally so regenerated dataset images and rollout images agree. Verify it is still in effect when regenerating |
| Rollout is ~2–4× slower than isolated measurement | Phase 3, 4 | Suspected CUDA/OpenGL contention. Confirm on the longer run before using isolated figures for estimates |
| `drx` clips at 55% even after the fix | — | Expected artefact of the arbitrary mapping, not a bug |
| Two cameras, not three, are supplied | Phase 3, 4 | The model tolerates it. Fine-tuning will define the real camera contract |
