# Phase 4 — LoRA Fine-Tuning: Results

**Status:** setup built, dry-run verified; the training run is queued for the user

---

## 4.1 What is adapted, and why

| Decision | Choice | Reasoning |
| --- | --- | --- |
| Target modules | lerobot's SmolVLA defaults — q/v projections of the action expert, plus `state_proj`, `action_in_proj`, `action_out_proj`, `action_time_mlp_in/out` | These are the modules between the frozen backbone and the action space, which is precisely where the mismatch lives |
| Vision encoder | **Frozen** | Not a default taken on faith. Phase 2's saved frames show the cube correctly framed and correctly oriented in the wrist camera at step 0, so the visual path works. The failure is entirely downstream. Adapting the encoder would spend memory fixing something that is not broken |
| Action head | LoRA, not full training | The projections are **dimension-agnostic**: they operate at `max_action_dim=32` and the slice to the real action dim happens afterwards. Going from the checkpoint's 6 outputs to robosuite's 7 needs **no new parameters at all** |
| Rank / alpha | 16 / 32 | Small first. Rank is the first thing to raise on underfitting |

Trainable: **0.74 M of 450.8 M parameters (0.165%)**.

### No architectural surgery was needed

This was the open question going into Phase 4, given Phase 1 found a 6-dimensional
SO-100 action space against robosuite's 7. It turns out SmolVLA pads state and
action to 32 internally and slices back at the end, so the pretrained projections
carry straight over. Only the declared feature shapes change: three cameras become
two, and the action feature becomes `(7,)`.

---

## The precision finding — Phase 1 corrected

Phase 1 concluded that half-precision weights were impossible and that
`phase-4-lora.md` §4.2's "base weights in bfloat16" was unreachable. **That was
wrong**, and it was found while measuring this phase's dry run: peak VRAM came in
at 1.29 GiB, below what 1.68 GiB of float32 weights would allow.

The checkpoint ships **mixed precision on purpose**: 474 bfloat16 tensors for the
backbone, 26 float32 ones for the flow-matching projections — which is exactly what
the hardcoded upcast at `modeling_smolvla.py:808` requires. Leaving that mix alone
works. What fails is casting to a *uniform* dtype: all-bfloat16 breaks
`action_out_proj`, all-float32 doubles the footprint for nothing.

| | Forced float32 (Phase 1) | Checkpoint mix |
| --- | --- | --- |
| Weight bytes | 1.677 GiB | **0.844 GiB** |
| Inference peak VRAM | 1.756 GiB | **0.904 GiB** |
| Inference latency | 399.7 ms | **353.4 ms** |

Half the memory and 12% faster, with identical outputs. §4.2's bfloat16 requirement
is satisfied out of the box.

The one catch: the flow-matching noise must match `action_in_proj`'s dtype, not the
backbone's. `SmolVLAWrapper.make_noise` now reads it off the module rather than
assuming, since a fine-tuned checkpoint could differ.

---

## 4.2 Memory-conscious setup

| Requirement from §4.2 | Status |
| --- | --- |
| Base weights in bfloat16 | Satisfied by the checkpoint's own dtypes |
| Gradient checkpointing behind a flag | `training.gradient_checkpointing`, `--gradient-checkpointing` |
| Small batch with accumulation | 2 × 8 = 16 effective |
| Explicit peak VRAM logging | Logged every `log_every` steps and in `metrics.json` |
| Checkpointing without a second copy of the model | Adapter only — **2.9 MiB** per checkpoint |
| Clear OOM message naming the knobs | `OutOfMemoryHint`, listing batch size, gradient checkpointing, rank and resolution in the order worth trying |

### Checkpoint size — a bug caught in the dry run

The first implementation saved **868 MB** per checkpoint. `wrap_with_peft` returns
the PEFT wrapper, and saving the *policy* instead writes the full 450M-parameter
model — exactly what §4.2 says not to do. Saving the returned wrapper brings it to
2.9 MiB.

The checkpoint also carries the policy config and the processor pipelines, because
the adapter alone is not loadable: the config holds this dataset's feature shapes
(two cameras, 7-value action) and the processors hold the dataset statistics.
Without those statistics LeRobot's normalizer silently no-ops — the Phase 1 trap —
and here that would be fatal rather than cosmetic, since training uses normalised
actions.

### Batch size was measured, not assumed

Effective batch held at 16 throughout:

| Config | Peak VRAM | s/step | s/sample |
| --- | --- | --- | --- |
| **2 × 8** | **1.292 GiB** | **5.94** | **0.371** |
| 4 × 4 | 1.687 GiB | 6.76 | 0.423 |
| 8 × 2 | 2.476 GiB | 9.27 | 0.579 |

Larger batches are slower *per sample*, not just per step. The GPU is already
saturated at batch 2 and bigger batches only add memory pressure. There is plenty
of VRAM headroom (1.29 of 5.0 GiB) but no throughput to buy with it.

---

## 4.3 Dry run — passed

60 steps, warmup shortened to 10 so the learning rate actually reaches 1e-4 within
the run. (A 10-step dry run at the configured 200-step warmup is uninformative:
the learning rate never leaves 1e-6 and the loss cannot be expected to move.)

| Check | Result |
| --- | --- |
| Loss finite | yes, all 60 steps |
| Loss moving | **4.06 → 1.59**, first-3 mean 3.59 against last-3 mean 1.54 |
| Validation loss | **1.4939** at step 60 |
| Peak VRAM | **1.292 GiB** of a 5.0 GiB budget |
| Device VRAM | 2.4 GiB of 6.00 GiB |
| Time per step | 4.85 s |
| Checkpoint save | 2.9 MiB, adapter + config + processors |
| Checkpoint reload | verified — see below |

### Checkpoint round-trip

Reloaded through `SmolVLAWrapper(adapter_path=...)`, which rebuilds the base policy
under the saved config and merges the LoRA weights:

```
action_dim      : 7
cameras         : ['observation.images.camera1', 'observation.images.camera2']
unnormalised    : True
weights         : 0.844 GiB
ROUND-TRIP OK   : action (7,) range [-3.399, 0.320] finite True
```

`unnormalised: True` is the part that matters. The base checkpoint could not
unnormalise at all (Phase 1); this one can, because the dataset statistics travel
with the adapter.

The adapter is **merged** into the weights at load rather than kept as a wrapper, so
evaluation costs exactly what the base model costs — which matters when one action
chunk is 10 flow-matching passes.

---

## Evaluation path

The fine-tuned policy emits robosuite's 7 dimensions directly, so **the arbitrary
mapping stops existing**. `ActionAdapter` gained an `identity` mapping that clips to
`[-1, 1]` and passes through; `scripts/rollout.py --adapter <checkpoint>` selects it
automatically.

That is the whole point of Phases 3 and 4: Phase 2's baseline was measured through a
mapping that was a category error by construction, and there is now nothing to map.

---

## Log noise — one real bug, one harmless warning

The first full training run printed roughly 200 lines of tracebacks before reaching
step 1. Nothing was wrong — training ran normally — but two things made the output
unreadable, and one was self-inflicted.

**Self-inflicted:** `logging.basicConfig(level=INFO)` configures the **root** logger,
so every third-party library began printing its INFO records. The `HTTP Request`
lines were HuggingFace cache validation (HEAD requests against files already on
disk), not downloads.

**Harmless:** the `torchcodec` probe. `lerobot.utils.import_utils` logs a warning
that embeds torchcodec's full RuntimeError, which itself contains one traceback per
FFmpeg version tried — five per occurrence, three occurrences per run. Expected on
Windows since Phase 3, and irrelevant here: the dataset is PNG, so no video decoder
is ever used.

[`src/log_utils.py`](../../src/log_utils.py) now keeps the project's own loggers at
INFO and pins the known noisy ones above it. Output drops from ~200 lines to one.
`--debug-logs` restores the full firehose when a download or decoder is genuinely
what needs debugging.

---

## 4.4 Training run — **queued for the user**

```bash
python scripts/train_lora.py
```

| | |
| --- | --- |
| Expected runtime | **~4 hours** (3 000 steps at 4.85 s/step) |
| Expected peak VRAM | **1.3 GiB**, against a 5.0 GiB budget |
| Effective batch | 16 (2 × 8 accumulation) |
| Data | 180 train episodes / 8 741 frames, 20 val episodes / 925 frames |
| Epochs | ~5.5 (546 steps per epoch) |
| Output | `results/phase-4/<timestamp>/` — config, `train.log`, `metrics.json`, and adapter checkpoints every 500 steps |

Validation runs every 250 steps and checkpoints are written every 500, so the run
can be stopped early at any checkpoint without losing the work.

If it OOMs — it should not, at 1.3 of 5.0 GiB — the error names the knobs in order.

### Then evaluate

```bash
python scripts/rollout.py --n-episodes 20 --render --adapter results/phase-4/<timestamp>/checkpoint-final
```

Same task, same episode count and the same instruction string as the baseline, which
is required for the comparison to mean anything.

---

## 4.5 Comparison against the zero-shot baseline

_To be filled in after the run._

Baseline: `results/phase-2/20260915-153627/` — **0.0% success (0/20)**, 2.3% action
clipping, full domain randomisation.

## 4.6 Phase 5 decision

_To be written against the trained model's numbers._

Two cheap checks on the trained policy, from the Phase 3 expert statistics:

- **Rotation should be small.** Expert `drx`/`dry` standard deviations are 0.022 and
  0.062 — `Lift` is solved almost entirely by translation. A trained policy emitting
  large rotations has not learned the task.
- **The gripper should be bimodal.** The expert gripper sits at ±1 with mean −0.450
  and std 0.893. Regression to the mean on a near-binary channel is the classic
  behaviour-cloning failure and would show up as a policy that reaches but never
  grasps.
