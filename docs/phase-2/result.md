# Phase 2 — Action Mapping and Zero-Shot Rollout: Results

**Status:** complete — 0.0% success over 20 episodes, diagnosed as structural
incompatibility rather than domain gap; proceed to Phase 3
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

| Measurement | 2-episode smoke | 20-episode run | Phase 0/1 in isolation |
| --- | --- | --- | --- |
| Env step | 33.7 ms | **7.30 ms** | 7.4 ms (Phase 0, 256 px) |
| Inference per call | 717.8 ms | **506.5 ms** | 399.7 ms (Phase 1) |

The smoke-test figures suggested a 2–4× slowdown, and the guess offered for it was
contention between the CUDA and OpenGL contexts. **The 20-episode run shows that was
wrong.** Env step comes out at 7.30 ms against Phase 0's 7.38 ms — the same number.
The smoke figure was a 120-step sample dominated by cold-start cost.

Inference at 506.5 ms against 399.7 ms is a real but mild 1.27× overhead, which is
what actual context contention looks like. Use 7.3 ms and 506 ms for planning.

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

## 2.5 Baseline evaluation — complete

`python scripts/rollout.py --n-episodes 20 --render`
→ `results/phase-2/20260915-144404/`

| Metric | Value |
| --- | --- |
| **Success rate** | **0.0%** (0/20) — as predicted |
| Episode length | 200.0 ± 0.0 — every episode ran to the horizon, none terminated early |
| Env step | 7.30 ms |
| Inference | 506.5 ms per call × 80 calls, 10.13 ms amortised per step |
| Action clipping | **2.9%** of all values |
| Peak VRAM | 1.752 GiB allocated, 2.887 GiB device — flat across all 20 episodes |
| Wall clock | 3.6 s per episode, ~93 s total |

### Per-dimension action statistics

| dim | mean | std | clip rate |
| --- | --- | --- | --- |
| dx | **+0.4497** | 0.3165 | 5.2% |
| dy | +0.1631 | 0.5017 | 2.8% |
| dz | −0.0120 | 0.4128 | 1.0% |
| drx | −0.4071 | 0.3912 | 8.0% |
| dry | −0.3717 | 0.4047 | 3.0% |
| drz | 0.0000 | 0.0000 | 0.0% |
| grip | +0.0116 | 0.3041 | 0.0% |

### The denormalisation contrast run

`--stats-key so100 --action-space degrees`, 5 episodes
→ `results/phase-2/20260915-144600/`

| | Normalised (20 ep) | Degrees (5 ep) |
| --- | --- | --- |
| Clip rate | 2.9% | **85.4%** |
| Per-dim clip rate | 0–8% | 97.8–100% on every mapped dimension |
| dy mean | +0.163 | **+119.18** |
| dz mean | −0.012 | **+122.08** |

This is §2.3's diagnostic working exactly as the phase document says it should:
wrong denormalisation produces near-total clipping. It also confirms the
unnormalisation path is arithmetically correct — the degree-scale means track the
checkpoint's stored `so100.buffer.action.mean` of
`[1.60, 119.94, 109.77, 56.71, −27.42, 12.00]`, which is what unnormalising a
roughly zero-mean output must produce.

---

## 2.6 Diagnosis

### What the frames show

Episode 0's saved frames settle this more clearly than any statistic.

| Step | Wrist camera |
| --- | --- |
| 0 | **Cube centred and cleanly framed** between the gripper fingers |
| 20 | Table edge and floor — the arm has already left the task region |
| 60–180 | Wall and floor; `agentview` is identical frame to frame |

The cube never moves in any of the 200 steps.

### Reading against the §2.6 tree

| Branch | Applies? |
| --- | --- |
| Actions near-constant or saturated → preprocessing bug | **No.** Clip rate 2.9%, per-dimension std 0.30–0.50. This branch *was* entered earlier at 64.3% clipping, traced to state scaling, and fixed |
| Gripper never actuates → mapping bug | **No.** `grip` has std 0.3041; it varies |
| Arm moves coherently toward the object but fails to finish | **No.** It moves away immediately |
| Actions varied and plausible but the arm moves incoherently → domain gap | Matches the *symptom*, but the label is wrong here — see below |

### The actual cause

`dx` carries a persistent positive bias: mean +0.4497 against a std of 0.3165, so
the command is positive roughly 92% of the time. That is a constant push in +x. The
arm drives itself off the table within the first 20 steps — inside the very first
50-action chunk — and parks against its workspace limit for the remaining 180.

Two things follow.

**The observation adapter is exonerated.** At step 0 the wrist image is correctly
oriented, correctly scaled and has the cube centred. Whatever is wrong is
downstream of the observation path.

**There is a feedback trap.** Once the arm is off the table, both cameras see wall
and floor. That input is far outside anything SmolVLA was trained on, so the policy
has no information with which to recover, and the bias keeps it pinned. A single
early systematic error is therefore unrecoverable, not merely costly.

### The verdict

`phase-2-zero-shot.md` offers "genuine domain gap" as the expected finding, and the
symptom matches. But **domain gap is the wrong diagnosis**, because it presupposes a
shared action space in which the policy could in principle be right and merely
transfers poorly. Here the policy is emitting SO-100 joint angles and the controller
is reading them as an end-effector delta pose. The mapping is not a poor fit; it is
a category error, and 0% is the arithmetically expected outcome rather than
evidence about SmolVLA.

The honest classification is a case the phase document does not enumerate:
**structural incompatibility**. Fine-tuning in Phases 3 and 4 is precisely the
remedy — it defines an action head for the 7-dimensional robosuite action space,
after which "domain gap" becomes a meaningful thing to measure.

### What this baseline is good for

1. **The pipeline is proven.** Observations, adapters, chunked inference, action
   execution, metrics and logging all run end to end over 20 episodes with VRAM flat
   at 1.75 GiB and no leaks, crashes or NaNs.
2. **A floor for Phase 4.** 0% at 2.9% clipping is the number fine-tuning must beat,
   and any improvement is unambiguous.
3. **Two real bugs were caught here rather than in Phase 4**, where they would have
   shown up as inexplicably poor fine-tuning: the state scaling and the reset handle
   leak.

### Proceeding to Phase 3

**Recommended.** §2.6 says not to enter Phase 3 with an unresolved suspected adapter
bug. There is none: the state-scaling bug was found and fixed, the image orientation
was verified by eye, the camera mapping is covered by tests against a real
observation, and the remaining failure is fully explained by a mapping that is
arbitrary by construction and documented as such.

---

## Findings carried forward

| Finding | Matters in | Note |
| --- | --- | --- |
| `hard_reset=False` is required on Windows | **Phase 3** | Otherwise mesh file handles leak and MuJoCo fails after ~10 resets. Regeneration resets far more often than that, and robomimic builds its own environment |
| State must be scaled before the model sees it | **Phase 3, 4** | LeRobot will not do it — no statistics ship with the checkpoint. The converted dataset must carry its own |
| `IMAGE_CONVENTION = "opencv"` is required | **Phase 3** | Set globally so regenerated dataset images and rollout images agree. Verify it is still in effect when regenerating |
| Env step in rollout is 7.30 ms, matching Phase 0 | Phase 3, 4 | The earlier "2–4× slower" reading was a cold-start artefact of a 120-step sample. Inference carries a real but mild 1.27× overhead (506 ms against 399 ms) |
| A single early action bias is unrecoverable | **Phase 4** | Once off the table the cameras see nothing task-relevant, so the policy cannot recover. Evaluation should report *when* an episode leaves the task region, not only whether it succeeded |
| `drx` clips at 55% even after the fix | — | Expected artefact of the arbitrary mapping, not a bug |
| Two cameras, not three, are supplied | Phase 3, 4 | The model tolerates it. Fine-tuning will define the real camera contract |
