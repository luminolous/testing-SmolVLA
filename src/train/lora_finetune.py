"""LoRA fine-tuning of SmolVLA on a LeRobot dataset.

Why LoRA and not a full fine-tune: a 450M-parameter model with Adam needs the
weights, the gradients and two optimizer moments resident at once. On a 6 GB card
that is over budget before activations are counted. LoRA trains a small adapter
instead, so optimizer state becomes negligible.

Two things here are specific to this project rather than generic:

* **The checkpoint's own dtypes are left alone.** `phase-4-lora.md` 4.2 asks for
  bfloat16 base weights, and the checkpoint already ships them: 474 bfloat16
  tensors for the backbone plus 26 float32 ones for the flow-matching
  projections, which `modeling_smolvla.py:808` requires to be float32. That is
  0.844 GiB of weights. Casting to a *uniform* dtype is what breaks things --
  all-bfloat16 fails at `action_out_proj`, all-float32 doubles the footprint for
  nothing.

* **Normalisation actually works here, unlike at inference.** Phase 1 found the
  base checkpoint ships no usable statistics, so LeRobot's normalizer silently
  no-ops. Statistics computed from this dataset are passed in explicitly, so the
  trained policy gets real normalisation on both state and action.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

GIB = 2**30


class OutOfMemoryHint(RuntimeError):
    """A CUDA OOM, re-raised naming the knobs that actually reduce memory."""


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[tuple[int, float]] = field(default_factory=list)
    peak_vram_gib: float = 0.0
    seconds_elapsed: float = 0.0


def split_episodes(
    n_episodes: int, val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split episode indices into train and validation.

    Held out **by episode**. Frames within an episode are strongly correlated, so a
    frame-level split leaks validation data into training and produces a validation
    loss that looks good and means nothing.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_episodes)
    n_val = max(1, int(round(n_episodes * val_fraction)))
    return sorted(order[n_val:].tolist()), sorted(order[:n_val].tolist())


def build_policy_and_processors(
    checkpoint: str,
    ds_meta: Any,
    device: str,
    n_action_steps: int,
) -> tuple[Any, Any, Any]:
    """Load the pretrained policy, re-featured for this dataset.

    The checkpoint declares three cameras, a 6-value state and a 6-value action.
    This dataset has two cameras, a 6-value state and a **7**-value action. Only the
    declared feature shapes change: SmolVLA pads state and action to
    `max_state_dim`/`max_action_dim` (32) internally and slices back afterwards, so
    the pretrained projections carry over to 7 dimensions with no new parameters
    and no architectural surgery.
    """
    from lerobot.configs.types import FeatureType
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.utils.feature_utils import dataset_to_policy_features
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    features = dataset_to_policy_features(ds_meta.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    config = SmolVLAConfig.from_pretrained(checkpoint)
    # Must be replaced wholesale. `make_policy` only fills input_features when they
    # are empty, so the checkpoint's three-camera declaration would otherwise
    # survive and the policy would expect a camera the dataset does not have.
    config.input_features = input_features
    config.output_features = output_features
    config.n_action_steps = n_action_steps
    config.device = device
    # `wrap_with_peft` refuses to attach adapters when this is unset, on the
    # reasonable grounds that LoRA on randomly initialised weights is pointless.
    # We do load pretrained weights below; the field just is not populated by
    # `SmolVLAConfig.from_pretrained`.
    config.pretrained_path = checkpoint

    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config)
    policy.to(device)

    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        config=config, dataset_stats=ds_meta.stats
    )
    return policy, preprocessor, postprocessor


def wrap_lora(policy: Any, r: int, alpha: int, dropout: float,
              target_modules: str | list[str] | None) -> Any:
    """Attach LoRA adapters and report what became trainable."""
    from peft import LoraConfig

    targets = target_modules
    if targets is None:
        targets = policy._get_default_peft_targets()["target_modules"]

    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=targets,
        bias="none",
    )
    peft_model = policy.wrap_with_peft(peft_config=peft_config)

    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    logger.info("LoRA: %.2fM trainable of %.1fM (%.3f%%)",
                trainable / 1e6, total / 1e6, 100 * trainable / total)
    return peft_model


def _oom_message(batch_size: int, grad_accum: int, checkpointing: bool) -> str:
    knobs = [
        f"training.batch_size (currently {batch_size}) -- halve it and double "
        f"training.grad_accum_steps (currently {grad_accum}) to keep the effective "
        f"batch the same",
    ]
    if not checkpointing:
        knobs.append("training.gradient_checkpointing: true -- trades compute for "
                     "activation memory, expect a meaningful slowdown per step")
    knobs.append("lora.r (currently set in configs/lora_lift.yaml) -- a smaller rank "
                 "reduces adapter and optimizer state, though this is the smallest "
                 "term here")
    knobs.append("env.resolution in the dataset -- only fixable by regenerating, so "
                 "treat it as a last resort")
    return (
        "CUDA out of memory during training.\n\nReduce, in this order:\n  - "
        + "\n  - ".join(knobs)
    )


@torch.no_grad()
def evaluate(policy: Any, preprocessor: Any, loader: Any, device: str,
             max_batches: int | None = None) -> float:
    """Mean validation loss."""
    policy.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = preprocessor(batch)
        loss, _ = policy.forward(batch)
        total += float(loss.item())
        n += 1
    policy.train()
    return total / max(n, 1)


def train(cfg: dict, run_dir: Path, max_steps: int | None = None) -> TrainState:
    """Run the fine-tune. `max_steps` overrides the config, for dry runs."""
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from torch.utils.data import DataLoader

    ds_cfg, model_cfg = cfg["dataset"], cfg["model"]
    lora_cfg, train_cfg = cfg["lora"], cfg["training"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    steps = max_steps if max_steps is not None else int(train_cfg["steps"])

    meta_only = LeRobotDataset(repo_id=ds_cfg["repo_id"], root=ds_cfg["root"])
    ds_meta = meta_only.meta
    n_episodes = meta_only.num_episodes
    del meta_only

    train_eps, val_eps = split_episodes(
        n_episodes, float(ds_cfg["val_fraction"]), int(ds_cfg["seed"])
    )
    logger.info("episodes: %d train, %d val", len(train_eps), len(val_eps))

    policy, preprocessor, postprocessor = build_policy_and_processors(
        model_cfg["checkpoint"], ds_meta, device, int(model_cfg["n_action_steps"])
    )
    delta_timestamps = resolve_delta_timestamps(policy.config, ds_meta)

    train_ds = LeRobotDataset(repo_id=ds_cfg["repo_id"], root=ds_cfg["root"],
                              episodes=train_eps, delta_timestamps=delta_timestamps)
    val_ds = LeRobotDataset(repo_id=ds_cfg["repo_id"], root=ds_cfg["root"],
                            episodes=val_eps, delta_timestamps=delta_timestamps)
    logger.info("frames: %d train, %d val", train_ds.num_frames, val_ds.num_frames)

    if train_cfg["gradient_checkpointing"]:
        target = getattr(policy.model, "vlm_with_expert", None)
        if target is not None and hasattr(target, "gradient_checkpointing_enable"):
            target.gradient_checkpointing_enable()
            logger.info("gradient checkpointing enabled")
        else:
            logger.warning("gradient checkpointing requested but unavailable here")

    peft_model = wrap_lora(policy, int(lora_cfg["r"]), int(lora_cfg["alpha"]),
                           float(lora_cfg["dropout"]), lora_cfg["target_modules"])

    batch_size = int(train_cfg["batch_size"])
    grad_accum = int(train_cfg["grad_accum_steps"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=int(train_cfg["num_workers"]),
                              drop_last=True, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=int(train_cfg["num_workers"]),
                            drop_last=False, pin_memory=(device == "cuda"))

    params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(train_cfg["lr"]),
                                  weight_decay=float(train_cfg["weight_decay"]))
    warmup = int(train_cfg["warmup_steps"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(warmup, 1))
    )

    state = TrainState()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    policy.train()
    t0 = time.perf_counter()
    iterator = iter(train_loader)
    accumulated = 0.0

    try:
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            micro_losses = []

            for _ in range(grad_accum):
                try:
                    batch = next(iterator)
                except StopIteration:
                    state.epoch += 1
                    iterator = iter(train_loader)
                    batch = next(iterator)

                batch = preprocessor(batch)
                loss, _ = policy.forward(batch)
                (loss / grad_accum).backward()
                micro_losses.append(float(loss.item()))

            torch.nn.utils.clip_grad_norm_(params, float(train_cfg["grad_clip_norm"]))
            optimizer.step()
            scheduler.step()

            step_loss = sum(micro_losses) / len(micro_losses)
            state.train_losses.append(step_loss)
            accumulated += step_loss
            state.step = step + 1

            if device == "cuda":
                state.peak_vram_gib = torch.cuda.max_memory_allocated() / GIB

            if state.step % int(train_cfg["log_every"]) == 0:
                window = int(train_cfg["log_every"])
                logger.info(
                    "step %5d/%d  loss %.4f  lr %.2e  peak VRAM %.2f GiB  %.2f s/step",
                    state.step, steps, accumulated / window,
                    scheduler.get_last_lr()[0], state.peak_vram_gib,
                    (time.perf_counter() - t0) / state.step,
                )
                accumulated = 0.0

            if state.step % int(train_cfg["eval_every"]) == 0:
                val = evaluate(policy, preprocessor, val_loader, device, max_batches=20)
                state.val_losses.append((state.step, val))
                logger.info("step %5d  val loss %.4f", state.step, val)

            if state.step % int(train_cfg["save_every"]) == 0:
                save_adapter(peft_model, run_dir / f"checkpoint-{state.step}",
                             policy.config, preprocessor, postprocessor)

    except torch.cuda.OutOfMemoryError as exc:
        raise OutOfMemoryHint(
            _oom_message(batch_size, grad_accum, train_cfg["gradient_checkpointing"])
        ) from exc

    state.seconds_elapsed = time.perf_counter() - t0
    save_adapter(peft_model, run_dir / "checkpoint-final",
                 policy.config, preprocessor, postprocessor)
    return state


def save_adapter(
    peft_model: Any,
    path: Path,
    policy_config: Any = None,
    preprocessor: Any = None,
    postprocessor: Any = None,
) -> None:
    """Save the LoRA adapter, plus everything evaluation needs to reload it.

    The PEFT wrapper returned by `wrap_with_peft` must be the object saved. Calling
    `save_pretrained` on the *policy* instead writes the full 450M-parameter model
    -- 868 MB per checkpoint, against roughly 3 MB for the adapter -- and defeats
    the requirement in `phase-4-lora.md` 4.2 that checkpointing not hold a second
    copy of the model.

    The policy config and the processor pipelines are written alongside it because
    the adapter alone is not loadable. The config carries this dataset's feature
    shapes (two cameras, a 7-value action) which differ from the base checkpoint's,
    and the processors carry the dataset statistics. Without those statistics
    LeRobot's normalizer silently no-ops -- the Phase 1 trap -- and here that would
    be fatal rather than merely wrong, since training used normalised actions.
    """
    path.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(path))

    if policy_config is not None:
        policy_config.save_pretrained(str(path / "policy_config"))
    if preprocessor is not None:
        preprocessor.save_pretrained(str(path / "processors"))
    if postprocessor is not None:
        postprocessor.save_pretrained(str(path / "processors"))

    size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    logger.info("saved adapter to %s (%.1f MiB)", path, size / 2**20)


def write_metrics(run_dir: Path, state: TrainState, cfg: dict) -> None:
    payload = asdict(state)
    payload["config"] = cfg
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
