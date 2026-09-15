# Phase 3 — Dataset Acquisition and Conversion: Results

**Status:** download, regeneration and conversion all built and smoke-tested;
the full 200-episode regeneration is queued for the user

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

## 3.4 Full regeneration — **queued for the user**

```bash
python scripts/regenerate_obs.py --save-samples
```

| | |
| --- | --- |
| Expected runtime | **~9.2 minutes** (2.81 s/episode × 200) |
| Expected disk | **~3.55 GiB** (385.6 KiB/frame × 9 666) |
| Output | `data/lift/ph/image_256px.hdf5` plus `image_256px_meta.json` |
| VRAM | negligible; this is rendering only, no model loaded |

Add `--compress` to trade runtime for a smaller file if 3.55 GiB is inconvenient.

Then convert:

```bash
python scripts/convert_to_lerobot.py --overwrite
```

| | |
| --- | --- |
| Expected disk | ~1.0 GiB (0.10 MiB/frame, PNG) |
| Output | `data/lerobot/local_lift_ph/`, plus `docs/phase-3/dataset_card.md` |

---

## 3.5 / 3.6 Conversion and validation — smoke-tested

`python scripts/convert_to_lerobot.py --source data/lift/ph/image_256px_2ep.hdf5 --repo-id local/lift_ph_smoke`

| Check | Result |
| --- | --- |
| Episode count matches source | 2 / 2 |
| Frame count matches source | 117 / 117 |
| Instruction round-trips | `"lift the cube"` |
| `observation.images.camera1` | `[3, 256, 256]` float32, `[0.024, 0.996]` |
| `observation.images.camera2` | `[3, 256, 256]` float32, `[0.027, 0.996]` |
| `observation.state` | `[6]`, range `[-0.824, 0.260]` |
| `action` | `[7]`, range `[-1.000, 0.148]` |
| On disk | 11.9 MiB |

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
