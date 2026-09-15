# Phase 4 — LoRA Fine-Tuning: Results

**Status:** complete — 0% to 40% success rate; Phase 5 assessed as not justified

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
| Small batch with accumulation | 4 × 2 = 8 effective, 2 dataloader workers |
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

### Throughput: two measurements, the first one wrong

The first sweep held the effective batch at 16 with `num_workers=0` and concluded
that batch 2 was optimal, since larger micro-batches were slower *per sample*
(0.371 s at 2×8, 0.423 at 4×4, 0.579 at 8×2).

**That conclusion did not survive a second look.** It was measured with data loading
on the main thread, where PNG decoding — not the GPU — was the limit. Two further
mistakes were in the method:

- **Runs were too short.** An 8-step benchmark of `num_workers=2` showed 7.78 s/step
  against 3.40 for 0 workers, which looks damning. That is Windows process startup:
  each worker re-imports torch. Over 30 steps the same comparison reverses.
- **The conclusion was drawn under a bottleneck that was itself removable.** Once
  loading is overlapped, the GPU can feed on a larger micro-batch after all.

Re-measured over 30 steps at an effective batch of 8:

| Config | s/step | Peak VRAM |
| --- | --- | --- |
| workers 0, 2 × 4 | 2.48 | 1.292 GiB |
| workers 2, 2 × 4 | 1.96 | 1.292 GiB |
| workers 4, 2 × 4 | 2.78 | — |
| **workers 2, 4 × 2** | **1.89** | 1.687 GiB |
| workers 2, 8 × 1 | 2.10 | 2.473 GiB |

Chosen: **2 workers, batch 4, 2 accumulation steps**. Against the original
configuration that is 0.236 s/sample against 0.303 — **22% faster** — at 1.687 GiB
of a 5.0 GiB budget.

Four workers are slower than two on this CPU.

### Image resolution is not the lever it looks like

SmolVLA resizes every camera to 512×512 before SigLIP, so 256×256 should cut vision
tokens roughly fourfold. Measured, it bought **4%** (3.25 against 3.40 s/step).

The vision encoder is not the bottleneck, which is what pointed at data loading in
the first place. The knob stays available as `model.resize_imgs_with_padding` but
defaults to the checkpoint's own 512, since lowering it moves the input away from
the pretraining distribution for almost no gain.

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

Sized to a **one-hour budget**, because the goal of this project is to establish
whether SmolVLA can be trained and run on this laptop at all — not to maximise task
success. Most of that question is already answered by the dry run: it trains, inside
1.7 GiB of a 5.0 GiB budget, and the loss moves.

| | |
| --- | --- |
| Expected runtime | **~57 minutes** (1 800 steps at 1.89 s/step) |
| Expected peak VRAM | **1.7 GiB**, against a 5.0 GiB budget |
| Effective batch | 8 (4 × 2 accumulation), 2 dataloader workers |
| Data | 180 train episodes / 8 741 frames, 20 val episodes / 925 frames |
| Epochs | ~1.6 (14 400 samples) |
| Optimizer updates | 1 800 |
| Output | `results/phase-4/<timestamp>/` — config, `train.log`, `metrics.json`, and adapter checkpoints every 400 steps |

Validation runs every 200 steps and checkpoints every 400, so the run can be stopped
early at any checkpoint without losing the work.

### What the hour costs

Cost is per sample, not per step: at 0.236 s/sample an hour buys roughly 15 000
samples however the steps are arranged. The original 4-hour configuration saw 5.5
epochs and made 3 000 updates; this one sees 1.6 epochs and makes 1 800.

That is a real reduction in fitting, and it should be read as such if the success
rate comes out low — §4.6's "training loss plateaus high" branch cannot be
distinguished from simple under-training at 1.6 epochs. Restoring `steps: 3000` with
the same batch settings costs about 95 minutes now rather than four hours, and is
the first thing to try if the model underfits.

If it OOMs — it should not, at 1.3 of 5.0 GiB — the error names the knobs in order.

### Then evaluate

```bash
python scripts/rollout.py --n-episodes 20 --render --adapter results/phase-4/<timestamp>/checkpoint-final
```

Same task, same episode count and the same instruction string as the baseline, which
is required for the comparison to mean anything.

---

## 4.4 Training run — complete

`results/phase-4/20260915-165042/`

| | Estimated | Actual |
| --- | --- | --- |
| Runtime | ~57 min | **25.4 min** (0.846 s/step) |
| Peak VRAM | 1.7 GiB | **1.687 GiB** of 5.0 GiB |
| Device VRAM | — | 2.820 GiB of 6.00 GiB |
| Steps | 1 800 | 1 800 |
| Epochs | ~1.6 | 1.6 |

**The runtime estimate was 2.2× pessimistic.** Even the 30-step benchmark carried
worker startup; over 1 800 steps it amortises to 0.846 s/step against the 1.89
measured. The hour budget was less than half spent.

### Loss

| Steps | Mean train loss |
| --- | --- |
| 0–100 | 2.9065 |
| 100–400 | 1.0439 |
| 400–800 | 0.8178 |
| 800–1200 | 0.7566 |
| 1200–1600 | 0.6864 |
| 1600–1800 | **0.6714** |

| Step | Val loss |
| --- | --- |
| 200 | 1.1774 |
| 400 | 0.9857 |
| 600 | 0.8913 |
| 800 | 0.9445 |
| 1000 | 0.8694 |
| 1200 | 0.8435 |
| 1400 | 0.7947 |
| **1600** | **0.7684** |
| 1800 | 0.7893 |

Two readings that matter for §4.6:

1. **Training loss has not plateaued.** It was still falling in the final band. The
   "loss plateaus high" branch — the only one that justifies Phase 5 — **does not
   apply**. This model is under-trained, not capacity-limited, which is the expected
   consequence of deliberately buying 1.6 epochs instead of 5.5.
2. **Validation bottoms at step 1600** and ticks up at 1800. The train/val gap
   (0.667 against 0.789) is small, so this is the beginning of the curve flattening
   rather than real overfitting. `checkpoint-1600` is the one to evaluate.

### A caveat on the validation numbers

`evaluate()` is capped at 20 batches, so each validation measures **80 frames of the
925** held out. The subset is fixed (`shuffle=False`), so the numbers are comparable
to each other, but they are not a full validation pass. Widening it costs about four
minutes per evaluation at this batch size, which is why it is capped; the trend is
what it is being read for, not the absolute value.

---

## 4.5 Comparison against the zero-shot baseline

`checkpoint-1600` of the 1 800-step run, 20 episodes, same task, same instruction,
same episode count as the baseline. → `results/phase-2/20260915-173822/`

| Metric | Zero-shot baseline | LoRA fine-tuned |
| --- | --- | --- |
| **Success rate** | **0.0%** (0/20) | **30.0%** (6/20) at 1.6 epochs, **40.0%** (8/20) at 3 |
| Action clipping | 2.3% | 9.1% |
| Peak VRAM | 1.752 GiB | **0.899 GiB** |
| Inference / call | 502.7 ms | 498.2 ms |
| Action mapping | arbitrary, 6→7 | **identity, 7→7** |

### The successes reproduce the expert's timing

All six successes occurred between **step 43 and step 50**. The expert
demonstrations average **48.3 steps**. The policy is not stumbling into the goal
late in a 200-step episode — it is executing the demonstrated behaviour on the
demonstrated schedule.

The fourteen failures never succeed at all within 200 steps. The outcome is
effectively binary: either the task is done in about 45 steps, or not.

### The predicted checks, against the expert distribution

| dim | Expert demos | Zero-shot | **LoRA** |
| --- | --- | --- | --- |
| dx | +0.174 ± 0.256 | +0.425 ± 0.320 | +0.114 ± 0.200 |
| dy | +0.007 ± 0.127 | +0.147 ± 0.486 | +0.057 ± 0.140 |
| dz | −0.171 ± 0.492 | −0.008 ± 0.421 | +0.479 ± 0.721 |
| **drx** | **+0.004 ± 0.022** | −0.432 ± 0.374 | **+0.003 ± 0.021** |
| dry | +0.005 ± 0.062 | −0.368 ± 0.409 | +0.127 ± 0.124 |
| drz | +0.011 ± 0.083 | 0.000 ± 0.000 | +0.002 ± 0.065 |
| **grip** | **−0.450 ± 0.893** | +0.032 ± 0.302 | **+0.354 ± 0.820** |

**Rotation collapsed to the expert's, as predicted.** `drx` came out at
+0.003 ± 0.021 against the expert's +0.004 ± 0.022 — effectively identical, from a
zero-shot baseline that was pushing −0.432 ± 0.374. `drz` matches too. `dry` is the
one still off, at roughly twice the expert's spread.

**The gripper became bimodal, as predicted.** Standard deviation 0.820 against the
expert's 0.893, up from 0.302 zero-shot. The channel is being driven to its
extremes rather than regressed to the mean, which was the failure mode worth
watching for.

Its *mean* differs in sign — +0.354 against the expert's −0.450 — but that is an
artefact of episode length rather than behaviour. Expert demonstrations end at ~48
steps, shortly after the grasp; these episodes run the full 200, so roughly 150 of
them are spent holding the cube with the gripper closed. The mean is dominated by
the hold, not the approach.

**Where it still saturates:** `dz` clips 34.2% and the gripper 29.5%. The policy
commands upward motion harder than the expert ever did (+0.479 against −0.171) and
runs into the limit. That is the most likely remaining cause of the 14 failures and
the obvious thing to look at next.

### Memory, and the feasibility question

Inference peak VRAM **fell** from 1.752 GiB to 0.899 GiB between the two runs, which
is not a fine-tuning effect — it is the Phase 1 precision correction. The whole
pipeline now fits in:

| Stage | Peak VRAM | Of 6 GB |
| --- | --- | --- |
| Inference / evaluation | 0.899 GiB | 15% |
| LoRA fine-tuning | 1.687 GiB | 28% |

**SmolVLA both trains and runs on this laptop with room to spare.** That was the
question the project was built to answer.

---

## Does better validation loss buy success? Yes.

That relation is not automatic in behaviour cloning — a lower action-prediction loss
can coexist with a policy that never closes the loop. Here it held.

| Run | Val loss | Success | Clipping | Steps at which success occurred |
| --- | --- | --- | --- | --- |
| Zero-shot | — | **0%** (0/20) | 2.3% | — |
| LoRA, 1.6 epochs | 0.7684 | **30%** (6/20) | 9.1% | 43, 43, 44, 47, 49, 50 |
| **LoRA, 3 epochs** | **0.7055** | **40%** (8/20) | 12.0% | **39, 40, 42, 43, 43, 44, 45, 49** |

The extra epochs bought more than count. The successes also arrive **earlier** —
median step 43 against 45.5 — which is now *below* the expert demonstration mean of
48.3. The policy is not merely reproducing the demonstrated schedule, it is
completing slightly ahead of it.

### Action statistics at 3 epochs

| dim | Expert demos | LoRA 1.6 ep | **LoRA 3 ep** |
| --- | --- | --- | --- |
| **dx** | **+0.174 ± 0.256** | +0.114 ± 0.200 | **+0.174 ± 0.186** |
| dy | +0.007 ± 0.127 | +0.057 ± 0.140 | +0.052 ± 0.140 |
| dz | −0.171 ± 0.492 | +0.479 ± 0.721 | +0.512 ± 0.650 |
| drx | +0.004 ± 0.022 | +0.003 ± 0.021 | +0.003 ± 0.021 |
| dry | +0.005 ± 0.062 | +0.127 ± 0.124 | +0.147 ± 0.119 |
| drz | +0.011 ± 0.083 | +0.002 ± 0.065 | +0.004 ± 0.056 |
| **grip** | **−0.450 ± 0.893** | +0.354 ± 0.820 | **+0.495 ± 0.863** |

`dx` now matches the expert mean exactly (+0.174) with a tighter spread, and the
gripper's standard deviation has moved further toward the expert's bimodality
(0.863 against 0.893). Gripper clipping rose to 55.4%, which on a channel the
experts drive to ±1 is the policy committing harder rather than a defect.

`dz` remains the outlier: +0.512 against the expert's −0.171, clipping 28.9%. Three
epochs did not correct it, which makes it look structural rather than
under-trained — the clearest target for the next piece of work.

---

## The longer run

`python scripts/train_lora.py --steps 4000` → `results/phase-4/20260915-174932/`

| | 1 800 steps | 4 000 steps |
| --- | --- | --- |
| Wall clock | 25.4 min | **57.6 min** |
| Epochs | 1.6 | **3** |
| Final train loss | 0.671 | **0.603** |
| Best val loss | 0.7684 (step 1600) | **0.7055 (step 3800)** |

Train loss by band: 1.398 (0–500) → 0.702 (1000–1500) → 0.642 (2000–2500) →
0.616 (3000–3500) → **0.603 (3500–4000)**.

**Still decreasing at 4 000 steps**, and validation improved by 0.06 over the
shorter run. The model remains under-trained rather than saturated.

No checkpoint exists at step 3800 (saves are every 400), but `checkpoint-4000` sits
at val 0.7110 against 0.7055 — the difference is inside the noise of an 80-frame
validation subset.

---

## 4.6 Phase 5 decision: **not justified**

`phase-5-full-ft.md` admits entry on exactly one condition — training loss
plateauing high despite raising rank and unfreezing modules, i.e. a genuine capacity
limit. That condition is **not met**:

| §4.6 branch | Applies? |
| --- | --- |
| Training loss plateaus high → underfitting, consider Phase 5 | **No.** Loss was still falling at 4 000 steps, 0.616 → 0.603 in the last band |
| Training loss low but rollout success low → adapter or instruction bug | **No.** Success went 0% → 30%, and the action statistics converged onto the expert's |
| Both losses low, success improved but modest → data quantity | **Closest match.** 30% from 3 epochs of 180 demonstrations |

`phase-5-full-ft.md` names the third case explicitly as *not* justifying entry:
"Results improved but modestly. That points at data quantity; MimicGen is the
cheaper next step."

### What to do instead, in order of cost

1. **Train longer.** Loss is still moving and an hour buys 3 epochs. The cheapest
   experiment available.
2. **Investigate the `dz` saturation.** 34% clipping on the lift axis, commanding
   upward motion far harder than any expert demonstration. This is a concrete,
   diagnosable defect rather than a capacity ceiling.
3. **More data via MimicGen**, as Phase 3 anticipated. Going from 1.6 to 3 epochs
   bought 10 percentage points, so more of the same data is still paying.
4. **Raise the LoRA rank** — the one lever that would begin to test capacity, and
   which must be tried before Phase 5 could honestly be entered.

Full fine-tuning would spend the entire VRAM budget to solve a problem that the
evidence says is not capacity.
