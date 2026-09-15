# Phase 3 — Dataset Acquisition and Conversion: Results

**Status:** complete — 200 episodes regenerated and converted, all validation
checks passing

---

## Headline

Three blockers were found and fixed, two of which invalidated earlier decisions
rather than merely being new obstacles:

1. **mujoco 3.1.6 leaks ~50 Windows file handles per XML load.** This was the real
   cause of the Phase 2 crash, not robosuite's `hard_reset`. Pinning mujoco 3.2.7
   fixes it completely and lets `hard_reset` return to its correct value.
2. **The Phase 2 image-orientation fix was wrong and produced upside-down data.**
   robomimic flips every RGB observation itself, so the global macro set in Phase 2
   composed with it.
3. **`torchcodec` cannot load on Windows**, as flagged in Phase 0. Sidestepped by
   storing frames as PNG rather than video.

---

## 3.1 Download — complete

`python scripts/download_data.py`

| Field | Value |
| --- | --- |
| Source | `lift/ph/demo_v141.hdf5` (proficient human, raw) |
| Size | 29.00 MB |
| SHA-256 | `99ee209282597a0060f767a79818e5d5cfd27b8c85fc225009c0e45fc81d38d5` |
| Episodes | 200 |
| Frames | 9 666 |
| Episode length | 36–64, mean 48.3 |
| Per-demo keys | `actions`, `states`, `controller_info`, `interventions`, … |

`v141` means the demonstrations were recorded with **robosuite 1.4.1**, exactly the
version pinned in Phase 0. Replay fidelity is therefore not in question, which was
the reason for choosing 1.4.1 over 1.5.2 in the first place.

The registry lists **no URL for the `image` variant of any task**, confirming that
images genuinely do not ship and must be regenerated.

### The controller convention, read from the data

The dataset's own `env_kwargs` record how the demonstrations were collected:

```
controller: OSC_POSE, control_delta=True, input [-1, 1],
            output_max [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]
robots: Panda,  control_freq: 20,  reward_shaping: false
```

The stored 7-dimensional actions are therefore a **normalised delta pose plus
gripper** — the same convention the Phase 2 action adapter targets. Phase 4 inherits
a matching action space for free.

---

## 3.2 Regeneration settings

| Setting | Value | Why |
| --- | --- | --- |
| Cameras | `robot0_eye_in_hand`, `agentview` | robomimic defaults to `agentview` alone. The target deployment uses a wrist camera, so a dataset without it is useless here |
| Resolution | 256×256 | What SmolVLA's checkpoint declares. Phase 0 measured render cost as flat from 128 to 512 px, so this is chosen to match the model, not to save time |
| Low-dim observations | stored | Cheap, and the model needs a state vector |
| Compression | off | Optional via `--compress` |

The regenerated file also carries `robot0_joint_pos` in radians directly, rather
than only the cos/sin encoding the live environment exposes.

---

## Blocker 1: the mujoco handle leak

Phase 2 worked around a crash by setting `hard_reset=False`, attributing the leak
to robosuite's reset path. Phase 3 needed 200+ resets, so the cause was measured
properly with `psutil.Process().num_handles()`:

| Path | Handles per reset | Outcome |
| --- | --- | --- |
| `hard_reset=True` | **+50** | fails at reset 10 |
| `hard_reset=False` | 0 | 60/60 fine |
| robomimic's `reset_to(model=…)` | **+50** | fails at reset 12 |

The leak is identical with the offscreen renderer on and off, which rules out the
GL context and points at MuJoCo's own XML/mesh loading. Testing newer versions:

| mujoco | Handles over 20 hard resets |
| --- | --- |
| 3.1.6 | +250 by reset 5, crash at 10 |
| **3.2.7** | **0** |
| 3.3.0 | 0 |

**Pinned to 3.2.7** — the smallest step from the previously pinned version that
fixes the leak, while staying below 3.13, which renames `MjData.qM` to `M` and
breaks robosuite 1.4.1.

### What this corrects in Phase 2

`hard_reset=False` was not a fix, it was a workaround whose cost was hidden.
`Lift` randomises the cube's **size** in `_load_model()`
(`BoxObject(size_min=[0.020]*3, size_max=[0.022]*3)`), which only runs on a hard
reset. So the Phase 2 baseline ran 20 episodes with **one fixed cube size**, varying
only position.

With mujoco 3.2.7, `hard_reset=True` is restored and verified: **60 consecutive
resets, 0 handles leaked, 60 unique cube sizes and 60 unique positions.**

The ±5% size range means the Phase 2 conclusions are unaffected — the arm left the
table within 20 steps regardless — but the baseline was measured with less
randomisation than intended, and that is worth knowing before it is used as the
Phase 4 comparison point.

---

## Blocker 2: the orientation fix was wrong

Phase 2 set `IMAGE_CONVENTION = "opencv"` globally, reasoning that one setting at
the source beats a flip inside the observation adapter, and that an adapter-side
flip would desynchronise rollouts from the Phase 3 dataset.

The reasoning was right about the hazard and wrong about the direction.
**robomimic flips every RGB observation unconditionally:**

```python
# robomimic/envs/env_robosuite.py:190
ret[k] = di[k][::-1]
```

It assumes robosuite's `opengl` default. With the macro set, the two flips composed
and the first regenerated dataset came out upside down — gripper fingers at the top
of the wrist view, confirmed by eye.

**Corrected approach:** leave `IMAGE_CONVENTION` at robosuite's default and turn
frames upright at the point of use. robomimic already does this; the rollout path
now calls `obs_adapter.upright()`, which is also used when saving debug frames so
that what is saved matches what the model sees.

Verified by eye after the change: the wrist camera shows fingers at the bottom with
the cube above, and at the end of the episode the cube grasped between them;
`agentview` shows the arm descending from the top onto the cube.

**Phase 2's numbers are unaffected.** Under the macro the model received upright
images; under the adapter flip it receives upright images. The input is identical.

---

## Blocker 3: torchcodec does not load on Windows

Flagged in Phase 0 as an open risk, and it materialised:

```
OSError: Could not load this library: ...torchcodec\libtorchcodec_core4.dll
```

The wheel installs but its shared library needs a system FFmpeg. PyAV ships FFmpeg
DLLs, but under hashed names (`avcodec-61-1306a7df8fab62262dd7797a8fb9f1cc.dll`),
so torchcodec's lookup cannot find them.

**Sidestepped rather than fought:** the LeRobot dataset is created with
`use_videos=False`, storing frames as PNG. torchcodec is only needed to decode
video. LeRobot logs the load failure, falls back to PyAV, and nothing in this
project's path touches a decoder at all.

This also costs less disk than the HDF5 intermediate: 0.10 MiB/frame against
0.38 MiB/frame.

---

## 3.3 Smoke regeneration — passed

`python scripts/regenerate_obs.py --n 2 --save-samples`

| Check | Result |
| --- | --- |
| Episodes / frames | 2 / 117 |
| Both camera streams present | yes |
| Shape, dtype | `(59, 256, 256, 3)` uint8 per episode |
| Actions | `(59, 7)` |
| Wrist view changes across frames | mean frame delta 2.02, max 4.14 |
| `agentview` changes across frames | mean frame delta 3.42, max 6.84 |
| Orientation | verified by eye, both cameras upright |
| Runtime | 5.5 s for 2 episodes |

The wrist camera's frame delta being *lower* than `agentview`'s is not the "static
view" failure the phase document warns about. The wrist view is mostly featureless
white table, so its pixel differences are small even as the viewpoint moves;
`agentview` watches a structured arm sweep across a fixed scene. The sample frames
confirm the viewpoint genuinely moves — the table edge shifts and the cube ends up
grasped between the fingers.

Sample frames: `results/phase-3/samples/`.

---

## 3.4 Full regeneration — complete

`python scripts/regenerate_obs.py --save-samples`

| | Estimated from 2 episodes | Actual |
| --- | --- | --- |
| Runtime | ~9.2 min | **5.8 min** (347.3 s) |
| Disk | ~3.55 GiB | **3.56 GiB** (3 640.6 MiB) |
| Episodes / frames | — | 200 / 9 666 |

The runtime estimate was 60% pessimistic: the smoke sample's 2.81 s/episode included
one-time environment construction, which amortises to 1.74 s/episode over 200. The
disk projection was accurate to within 0.3%.

Verification passed on all counts: both camera streams present at
`(n, 256, 256, 3)` uint8, actions `(n, 7)`, no static view.

---

## 3.5 / 3.6 Conversion and validation — complete

`python scripts/convert_to_lerobot.py --overwrite`

| Check | Result |
| --- | --- |
| Episode count matches source | **200 / 200** |
| Frame count matches source | **9 666 / 9 666** |
| Instruction round-trips | `"lift the cube"` |
| `observation.images.camera1` | `[3, 256, 256]` float32, `[0.024, 0.996]` |
| `observation.images.camera2` | `[3, 256, 256]` float32, `[0.027, 0.996]` |
| `observation.state` | `[6]`, range `[-0.824, 0.260]` |
| `action` | `[7]`, range `[-1.000, 0.148]` |
| On disk | **986.5 MiB** (PNG, against 3.56 GiB for the HDF5 intermediate) |

### Accepted by the evaluator without adaptation

§3.6 asks that a sample load through the same wrapper Phase 2 uses. Three random
frames were passed **verbatim** — the dataset's own tensors, no reshaping, no dtype
conversion, no rescaling — into `SmolVLAWrapper.predict_action`:

```
frame 8222  task='lift the cube' -> action (6,) range [-0.823, +0.099]  finite=True
frame 6156  task='lift the cube' -> action (6,) range [-0.917, +0.415]  finite=True
frame 4940  task='lift the cube' -> action (6,) range [-0.815, +0.497]  finite=True
```

The dataset and the evaluation path agree on format with nothing in between.

### Action statistics against the Phase 2 rollout

§3.6 asks for this comparison as a check on the Phase 2 action adapter. It turned
out to explain the Phase 2 failure in concrete terms.

| dim | Expert demos (mean ± std) | SmolVLA via the arbitrary mapping |
| --- | --- | --- |
| dx | +0.174 ± 0.256 | **+0.450** ± 0.317 |
| dy | +0.007 ± 0.127 | +0.163 ± 0.502 |
| dz | −0.171 ± 0.492 | −0.012 ± 0.413 |
| drx | +0.004 ± **0.022** | **−0.407** ± **0.391** |
| dry | +0.005 ± **0.062** | **−0.372** ± **0.405** |
| drz | +0.011 ± 0.083 | 0.000 ± 0.000 (held at 0) |
| gripper | **−0.450 ± 0.893** | +0.012 ± **0.304** |

Three readings, none of which were visible from the Phase 2 numbers alone:

1. **The experts barely rotate.** `drx` and `dry` have standard deviations of 0.022
   and 0.062. `Lift` is solved almost entirely by translation with a fixed wrist
   orientation. The arbitrary mapping fed SmolVLA's joint outputs into those
   channels with std 0.39 and 0.40 — roughly **17× more rotation than the task ever
   uses**, with a large constant bias on top. That is the mechanism behind the arm
   tumbling off-task within 20 steps.

2. **The expert gripper is effectively binary.** Mean −0.450 with std 0.893 and
   values at both ±1.0: it is open or closed, rarely in between. SmolVLA's mapped
   gripper hovered at +0.012 with std 0.304 — it never commits to closing. Phase 2
   recorded that the gripper "varies", which was true and misleading; against the
   expert distribution it never actually grasps.

3. **`drz` is used by the experts** (std 0.083, range ±0.29), and the Phase 2
   adapter held it at zero. A small but real loss, and one that disappears in
   Phase 4 where the fine-tuned head emits all seven dimensions.

There is **no discrepancy pointing back at the action adapter itself** — the
adapter faithfully passes through what it is given, clips correctly, and the
dataset's own actions sit inside `[-1, 1]` as expected for `control_delta=True`.
The mismatch is entirely in what SmolVLA emits, which is the known structural
incompatibility rather than a wiring bug.

Loading a sample back returns images already in **CHW float32 `[0, 1]`** — exactly
what SmolVLA's preprocessor expects, so no adaptation sits between the dataset and
the model at training time.

### The state is built by the evaluator's own code

`to_lerobot.py` calls `ObsAdapter.build_state` rather than restating the
construction. Training and evaluation must agree on what `observation.state` means
down to ordering and scaling, and a duplicated implementation is exactly how that
agreement rots. The `normalize_state` setting is recorded in the dataset card
because it has to match at evaluation time.

Images are the one thing *not* routed through `ObsAdapter`: robomimic's output is
already upright, and passing it through `upright()` again would train the model on
inverted frames while evaluating it on correct ones.

---

## Findings carried forward

| Finding | Matters in | Note |
| --- | --- | --- |
| Dataset actions are 7-dim normalised OSC delta pose | **Phase 4** | Matches the Phase 2 action adapter's target, so fine-tuning defines a head the evaluator can already drive |
| `use_videos=False` is required on Windows | Phase 4 | torchcodec cannot load. Any LeRobot training path that assumes video decoding will fail |
| mujoco pin is load-bearing | all | 3.1.6 leaks handles, 3.13 breaks robosuite. Do not move it without re-running the handle measurement |
| Phase 2 baseline used a fixed cube size | Phase 4 | Consider re-running the zero-shot baseline with `hard_reset=True` before using it as the comparison point |
| Instruction string is `"lift the cube"` | Phase 4 | Recorded in the dataset card; must be passed verbatim at evaluation |
| State is 6-dim and scaled to `[-1, 1]` | Phase 4 | Built by `ObsAdapter.build_state`. Fine-tuning may want a richer state; changing it means changing both sides at once |
| Rotation channels carry almost no signal in this task | **Phase 4** | Expert `drx`/`dry` std is 0.022/0.062. A fine-tuned head should learn this quickly, and it is a cheap sanity check on a trained policy: if its rotation output is large, something is wrong |
| Expert gripper is near-binary (±1) | **Phase 4** | Mean −0.450, std 0.893. Worth checking that the fine-tuned policy reproduces the bimodality rather than regressing to the mean, which is the classic behaviour-cloning failure on a binary channel |
| Dataset ready at `data/lerobot/local_lift_ph` | **Phase 4** | 200 episodes, 9 666 frames, 986.5 MiB, validated end to end and accepted by the evaluator unmodified |
