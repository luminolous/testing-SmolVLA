# Phase 1 — Model Load and I/O Sanity Check: Results

**Status:** complete
**Checkpoint:** `lerobot/smolvla_base` (906 MB, `model.safetensors`)
**Stack:** lerobot 0.6.1 · transformers 5.5.4 · torch 2.11.0+cu128

---

## Headline

Two findings, both structural, both established from the checkpoint's own files
rather than from assumption.

1. **The base checkpoint is an SO-100 model, not a general one.** It declares a
   6-dimensional action space whose statistics are SO-100 arm joint angles in
   degrees, and three cameras. Our robosuite target is a 7-DoF Panda under
   OSC_POSE with a 7-dimensional delta-pose action and two cameras. There is no
   dimension-for-dimension mapping between them.

2. **Actions come out of the stock checkpoint in normalised units, silently.** The
   unnormalisation step is a no-op because of a key-naming mismatch inside LeRobot.
   No error, no warning. This is precisely the failure `phase-1-model-io.md` §1.3
   warns is most likely to break Phase 2.

---

## 1.2 Input contract

Source: `config.json` from the checkpoint, and `modeling_smolvla.py`.

| Property | Value | Evidence |
| --- | --- | --- |
| Cameras | **3** — `observation.images.camera1/2/3` | `config.json` `input_features` |
| Declared image shape | `(3, 256, 256)` | `config.json` `input_features` |
| Internal resize | `resize_with_pad` to **512×512** | `config.resize_imgs_with_padding` |
| Pixel range in | float `[0, 1]`, rescaled to `[-1, 1]` for SigLIP | `modeling_smolvla.py:360` |
| State dim | **6**, zero-padded to `max_state_dim=32` | `config.json`; `modeling_smolvla.py:412` |
| State normalisation | `MEAN_STD` declared | `config.normalization_mapping` |
| Observation steps | `n_obs_steps=1` | `config.json` |
| Language key | `task` | `policy_preprocessor.json` step 3 |
| Tokenizer | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | `config.vlm_model_name` |
| Tokenizer max length | 48, right padding, truncation on | `policy_preprocessor.json` step 3 |

The declared `(3, 256, 256)` and the internal 512×512 resize are not in conflict:
the first is what the training datasets stored, the second is what the vision
encoder receives. Feeding a different input resolution is fine — it is padded and
resized either way.

### Preprocessing pipeline

Normalisation is **not** part of the policy. It lives in a separate processor
pipeline loaded from the checkpoint:

```
0  rename_observations_processor   (rename_map: {})
1  to_batch_processor
2  smolvla_new_line_processor
3  tokenizer_processor             (task_key='task', max_length=48)
4  device_processor                (cuda)
5  normalizer_processor            (STATE/ACTION: MEAN_STD, VISUAL: IDENTITY)
```

Postprocessor: `unnormalizer_processor` → `device_processor(cpu)`.

---

## 1.3 Output contract

| Property | Value | Evidence |
| --- | --- | --- |
| Action dim | **6** | `config.json` `output_features` |
| Chunked | **Yes** — `chunk_size=50` | `config.json` |
| Actions executed per inference | `n_action_steps=50` | `config.json` |
| Execution strategy | Chunk is cached in a queue; one action popped per `select_action()` call; the network runs once every 50 calls | `modeling_smolvla.py:262-269` |
| Decoding | Flow matching, `num_steps=10` | `config.num_steps` |
| Internal padding | Padded to `max_action_dim=32`, then sliced back to 6 | `modeling_smolvla.py:217-218` |
| Action normalisation | `MEAN_STD` declared | `config.normalization_mapping` |

### What the 6 dimensions are

The checkpoint ships three action statistic sets, all SO-100:

| Key | mean | std |
| --- | --- | --- |
| `so100.buffer.action` | `[1.60, 119.94, 109.77, 56.71, -27.42, 12.00]` | `[26.39, 52.41, 49.85, 37.00, 59.36, 19.04]` |
| `so100-blue.buffer.action` | `[1.42, 125.72, 125.42, 63.78, -106.54, 3.35]` | `[14.08, 39.43, 18.24, 23.49, 38.11, 5.05]` |
| `so100-red.buffer.action` | `[2.44, 124.94, 123.56, 66.99, -103.88, 2.89]` | `[14.30, 42.77, 18.88, 21.65, 33.53, 4.48]` |

Magnitudes in the ±180 range with per-dimension means far from zero are **absolute
joint angles in degrees**, not deltas and not a normalised control signal. The
SO-100 is a 5-joint arm plus a gripper, which accounts for all 6 dimensions.

**There are no `observation.state` statistics in the checkpoint at all** — only
action statistics, despite `STATE: MEAN_STD` being declared. State normalisation is
therefore also a no-op.

### The silent normalisation failure

`policy_preprocessor_step_5_normalizer_processor.safetensors` stores six tensors,
keyed as `so100.buffer.action.mean`, `so100-blue.buffer.action.std`, and so on.

`NormalizerProcessorStep._apply_transform` looks up the **bare** feature name:

```python
# normalize_processor.py:330
if norm_mode == NormalizationMode.IDENTITY or key not in self._tensor_stats:
    return tensor
```

`key` is `"action"`. The stored keys are dataset-qualified. The lookup misses and
the tensor is returned untouched — no exception, no log line.

Consequence: **`lerobot/smolvla_base` used as shipped returns actions in normalised
units.** An action that looks plausible in shape and dtype is wrong in magnitude by
roughly the per-dimension std, which is 14 to 59 degrees.

[`SmolVLAWrapper`](../../src/model/smolvla_wrapper.py) exposes this as
`action_unnormalised` and can rebind a chosen statistic set onto the bare `action`
key via `dataset_stats_key=`, which makes unnormalisation actually happen.

---

## The action space mismatch

| | SmolVLA base | robosuite `Lift` / Panda / OSC_POSE |
| --- | --- | --- |
| Action dim | 6 | 7 |
| Action meaning | Absolute joint angles, degrees | Delta end-effector pose + gripper |
| Units | Degrees | Normalised `[-1, 1]` |
| Arm | SO-100, 5 joints + gripper | Panda, 7 joints + parallel gripper |
| State dim | 6 | 32 (`robot0_proprio-state`) |
| Cameras | 3 | 2 |

This is not a scaling problem or a convention flip. The dimensionalities, the
semantics, and the embodiment all differ. `phase-2-zero-shot.md` §2.1 asks for a
dimension-by-dimension correspondence between SmolVLA's output and the robosuite
controller; **no honest correspondence exists.**

Note what this does *not* block: the architecture pads actions to
`max_action_dim=32` internally, so a 7-dimensional action space is perfectly
representable. Phases 3 and 4 — fine-tuning on robosuite data with 7-dim actions —
are unaffected. It is specifically the zero-shot baseline in Phase 2 whose premise
needs revisiting.

---

## 1.1 Load cost

| Quantity | Measured |
| --- | --- |
| Load time, cold start | **20.85 s** |
| Parameters | **450.0 M** (99.9 M trainable) |
| Weight bytes | **1.677 GiB** (float32) |
| VRAM attributable to load | **1.766 GiB** |

The 99.9 M trainable figure is the action expert. The VLM backbone is frozen by
default (`freeze_vision_encoder=True`, `train_expert_only=True`), and only 16 of
SmolVLM2's layers are used (`num_vlm_layers=16`) — the loader logs
`Reducing the number of VLM layers to 16`.

> **Corrected in Phase 4.** The conclusion below — that float32 is forced — is
> wrong, and the measurements in this section were taken under that mistake. The
> checkpoint **ships mixed precision**: 474 bfloat16 tensors for the backbone and 26
> float32 ones for the flow-matching projections, which is exactly what line 808
> requires. Leaving that mix alone works. What fails is casting to a *uniform*
> dtype.
>
> | | Forced float32 (below) | Checkpoint mix (correct) |
> | --- | --- | --- |
> | Weight bytes | 1.677 GiB | **0.844 GiB** |
> | VRAM after load | 1.766 GiB | **0.885 GiB** |
> | Peak VRAM | 1.756 GiB | **0.904 GiB** |
> | Latency, mean | 399.71 ms | **353.43 ms** |
> | p95 | 410.97 ms | **367.68 ms** |
>
> Output values are unchanged (`[-2.95, 1.38]`, sensitivity 0.805), so this is
> purely a memory and speed correction. The one requirement is that the
> flow-matching noise match `action_in_proj`'s dtype rather than the backbone's;
> `SmolVLAWrapper.make_noise` now reads it off the module.
>
> The wrapper defaults to the checkpoint's own dtypes. `--float32` reproduces the
> figures below.

### Precision: the original, mistaken conclusion

`modeling_smolvla.py:808` hardcodes an upcast before the output projection:

```python
suffix_out = suffix_out.to(dtype=torch.float32)
v_t = self.action_out_proj(suffix_out)
```

so `action_out_proj` must hold float32 weights. The float32 then propagates around
the flow-matching loop — `v_t` → `x_t` → `action_in_proj` → `action_time_mlp` → the
suffix embeddings fed back to the expert — leaving no clean place to keep half
precision. Casting the policy to bfloat16 fails with
`RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16`.

**`torch.autocast` was measured and is not worth using here:**

| Mode | Latency/chunk | Peak VRAM allocated |
| --- | --- | --- |
| float32 weights, bfloat16 autocast | 401.39 ms | 1.927 GiB |
| float32 weights, no autocast | 399.71 ms | 1.756 GiB |

No speed gain, and 0.17 GiB more VRAM because the float32 weights stay resident
alongside the cast copies. At batch size 1 this workload is overhead-bound, not
compute-bound. Autocast is therefore **off by default** in the wrapper.

There is also a second, separate dtype bug: `VLAFlowMatching.sample_noise()` takes
only a shape and a device, so it always produces float32 noise. The wrapper supplies
noise explicitly instead — `sample_actions` only generates its own when
`noise is None`. This doubles as the seeding mechanism, which matters because
SmolVLA is generative: without a fixed noise tensor, two calls on identical
observations return different actions.

---

## 1.4 Dummy forward pass

Five random observations, one instruction (`"lift the cube"`), **flow-matching noise
held fixed across observations** so that any variation in the output is attributable
to the images rather than to the sampling.

| Property | Value |
| --- | --- |
| Output shape | `(50, 6)` — `(chunk_size, action_dim)` |
| Output dtype | `float32` |
| NaNs / Infs | 0 / 0 |

### The normalisation finding, confirmed empirically

| Configuration | Value range | mean / std |
| --- | --- | --- |
| As shipped | `[-2.9479, 1.3847]` | −0.2933 / 0.8315 |
| `--stats-key so100` | `[-152.3554, 178.8021]` | 35.3158 / 70.7497 |

As shipped the output is approximately unit-scale — normalised units. With the
statistics bound, dimension 1 comes back with a mean of **+107.2**, against the
stored `so100.buffer.action.mean[1] = 119.94`. Degrees, as expected. This is
measurement, not inference from reading the source.

### Not saturated

Per-dimension standard deviation across the five observations, as shipped:

```
dim 0: mean= -0.4425  std=0.7527    dim 3: mean= -0.4211  std=1.0916
dim 1: mean= -0.2429  std=0.6932    dim 4: mean= -0.4493  std=0.8637
dim 2: mean= -0.0408  std=1.2035    dim 5: mean= -0.4534  std=0.2245
```

Mean per-dimension std **0.805** with noise held fixed. The images genuinely reach
the model and influence the output — the failure mode `phase-1-model-io.md` §1.4
warns about (everything saturating at one value) is not present.

---

## 1.5 Latency and memory

50 iterations at batch size 1 after 5 warmup, forcing a full network call each time
via `predict_action_chunk`.

| Metric | Value |
| --- | --- |
| Mean | **399.71 ms** per chunk |
| p95 | 410.97 ms |
| min / max | 379.93 / 437.12 ms |
| Drift (2nd half − 1st half) | −2.85 ms — stable, no leak |
| Amortised per executed action | **7.99 ms** (chunk covers 50 steps) |
| Peak VRAM allocated | **1.756 GiB** (no autocast) |
| Peak VRAM reserved | 1.828 GiB |
| Total device VRAM in use | 2.894 GiB of 6.00 GiB |
| Against the 5.0 GiB budget | **within budget**, 2.1 GiB spare |

### Against the published sub-100 ms expectation

**400 ms is roughly 4× the published figure**, and the phase document asks for this
to be reported rather than glossed over. Two things explain most of it without
implying anything is broken:

- One call runs the VLM prefix pass **plus 10 flow-matching denoise steps**
  (`num_steps=10`), each a full pass through the action expert. It is ten forward
  passes, not one.
- An RTX 4050 Laptop at batch size 1 is overhead-bound. That autocast changed
  nothing is direct evidence: the GPU is not saturated by arithmetic.

**This does not make the model slow in practice.** Because one call produces 50
executable actions, the amortised cost is 8.01 ms per environment step — comparable
to the 7.51 ms simulator step measured in Phase 0. A 200-step episode needs 4
inference calls, about 1.6 s, against roughly 1.5 s of simulation.

### The 50-step open-loop horizon

`n_action_steps=50` means the policy commits to 50 actions before it looks at the
world again. At `control_freq=20` that is **2.5 seconds of open-loop control**.

This is a Phase 2 concern, not a Phase 1 one, but it is worth recording now: for a
contact-rich task like grasping, a 2.5 s open-loop commitment is a long time. If
zero-shot rollouts show the arm reaching plausibly and then failing at contact, this
is a candidate explanation and is worth testing before concluding a domain gap.
`n_action_steps` can be lowered below `chunk_size` to re-plan more often, at
proportionally higher inference cost.

---

## Findings carried forward

| Finding | Matters in | Note |
| --- | --- | --- |
| Action space is 6-dim SO-100 joint degrees; robosuite is 7-dim OSC delta pose | **Phase 2** | No honest dimension mapping exists. The zero-shot baseline's premise needs a decision before §2.5 is run |
| Actions are normalised unless statistics are explicitly bound | Phase 2 | Silent. Use `dataset_stats_key=`, and assert `action_unnormalised` before trusting any rollout number |
| No `observation.state` statistics exist in the checkpoint | Phase 2, 3 | State normalisation is also a no-op. Phase 3's converted dataset must supply its own statistics |
| ~~Half-precision weights are impossible~~ **Corrected in Phase 4** | **Phases 4, 5** | The checkpoint already ships bfloat16 for the backbone and float32 for the flow-matching projections. Keep that mix: 0.844 GiB of weights, 0.904 GiB peak. `phase-4-lora.md` §4.2's bfloat16 requirement is satisfied out of the box. Casting to a uniform dtype is the thing that breaks |
| Autocast buys nothing at batch size 1 | Phases 2, 4 | Off by default. Do not reach for it as an optimisation without measuring |
| `sample_noise()` ignores dtype; noise must be supplied externally | Phase 2 | Already handled in the wrapper, and it is also how runs are made reproducible |
| 50-step open-loop horizon (2.5 s at 20 Hz) | Phase 2 | A candidate explanation for contact-stage failures. Test before concluding domain gap |
| Inference is 400 ms/chunk, 8 ms amortised per step | Phase 2 | Rollouts are not inference-bound. 50 episodes ≈ 2.6 min including simulation |
| Peak VRAM 1.76 GiB, 2.89 GiB device total | Phases 4, 5 | Leaves ~2.1 GiB against the 5 GiB budget for optimizer state and activations |

---

## Reproducing

```bash
python scripts/probe_model.py                    # as shipped
python scripts/probe_model.py --stats-key so100  # with unnormalisation bound
python scripts/probe_model.py --autocast bfloat16  # the autocast comparison
```

Raw output of all three runs: `results/phase-1/probe_model.txt` (gitignored).
