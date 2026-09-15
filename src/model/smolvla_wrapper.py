"""Thin wrapper around LeRobot's SmolVLA policy.

Exposes the model's *native* contract. It deliberately does not adapt anything to
robosuite -- that translation is Phase 2's job, and mixing the two would hide which
side a bug came from.

What the pretrained `lerobot/smolvla_base` checkpoint actually expects, read from
its own `config.json` rather than assumed:

* Three camera inputs, ``observation.images.camera{1,2,3}``, each ``(3, 256, 256)``
  float in ``[0, 1]``. The policy resizes with padding to 512x512 and rescales to
  ``[-1, 1]`` for SigLIP internally.
* ``observation.state``: 6 values, padded to ``max_state_dim`` (32) internally.
* ``action``: 6 values per timestep, a chunk of ``chunk_size`` (50) timesteps.
* The language instruction arrives under the batch key ``task``.

The 6 action dimensions are SO-100 arm joint targets in degrees, not a delta pose.
See :attr:`SmolVLAWrapper.action_stats` and docs/phase-1/result.md.

**Normalisation warning.** Normalisation is not part of the policy; it lives in
separate pre/post-processor pipelines. The base checkpoint stores its action
statistics under dataset-qualified keys (``so100.buffer.action.mean``), while the
unnormalizer looks up the bare key ``action``. The lookup misses and
``normalize_processor.py:330`` returns the tensor untouched, with no error and no
warning. So the actions coming out of the stock checkpoint are in **normalised**
units. :attr:`SmolVLAWrapper.action_unnormalised` reports whether the postprocessor
is actually able to unnormalise, and `dataset_stats_key` selects one of the stored
statistic sets to make it do so.

**Precision.** The checkpoint ships a deliberate mix: 474 bfloat16 tensors for the
VLM backbone and 26 float32 ones, which include the flow-matching projections.
That mix is not an accident -- ``modeling_smolvla.py:808`` hardcodes
``suffix_out.to(dtype=torch.float32)`` before ``action_out_proj``, so those
projections have to be float32 while everything upstream can be half precision.

Casting the policy to a *uniform* dtype is what breaks it. All-bfloat16 fails at
``action_out_proj`` (float32 input, bfloat16 weight); all-float32 works but doubles
the weight footprint for nothing. The default here is therefore to leave the
checkpoint's dtypes alone, which measured 0.844 GiB of weights and 0.905 GiB peak
against 1.677 and 1.756 GiB for uniform float32, and ran 359 ms per chunk against
400 ms.

The one thing this requires is that the flow-matching noise match the projections'
dtype rather than the backbone's -- see :meth:`SmolVLAWrapper.make_noise`.
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "lerobot/smolvla_base"

# Stat sets shipped in the base checkpoint, as stored in
# policy_preprocessor_step_5_normalizer_processor.safetensors.
AVAILABLE_STAT_KEYS = ("so100", "so100-blue", "so100-red")


@dataclass
class LoadReport:
    """What loading the checkpoint actually cost, measured rather than estimated."""

    load_seconds: float
    param_count: int
    trainable_param_count: int
    weight_bytes: int
    vram_bytes_after_load: int

    def __str__(self) -> str:
        return (
            f"loaded in {self.load_seconds:.2f}s | "
            f"{self.param_count / 1e6:.1f}M params "
            f"({self.trainable_param_count / 1e6:.1f}M trainable) | "
            f"weights {self.weight_bytes / 2**30:.3f} GiB | "
            f"VRAM after load {self.vram_bytes_after_load / 2**30:.3f} GiB"
        )


class SmolVLAWrapper:
    """Load SmolVLA and run single-observation inference through it."""

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        autocast_dtype: torch.dtype | None = None,
        dataset_stats_key: str | None = None,
        seed: int | None = None,
        adapter_path: str | Path | None = None,
    ) -> None:
        """
        Args:
            checkpoint: Hub id or local path of the SmolVLA checkpoint.
            device: Torch device for the policy.
            dtype: Cast every weight to this dtype. ``None`` (the default) keeps the
                checkpoint's own mixed precision, which is both correct and roughly
                half the memory. Only ``float32`` is a valid cast; a uniform
                half-precision cast breaks ``action_out_proj``.
            autocast_dtype: Compute dtype for inference, via ``torch.autocast``.
                Off by default: measured on this GPU it bought no speed (401.4 ms
                against 399.7 ms per chunk) and cost 0.17 GiB more VRAM, since the
                float32 weights stay resident alongside the cast copies. Inference
                here is overhead-bound at batch size 1, not compute-bound.
            dataset_stats_key: One of :data:`AVAILABLE_STAT_KEYS`. When given, the
                matching stored statistics are rebound to the bare ``action`` key so
                the postprocessor can actually unnormalise. When ``None`` the
                checkpoint is left exactly as shipped, which means actions come back
                normalised.
            seed: Seeds the flow-matching noise. SmolVLA is a generative policy, so
                without this two calls on identical observations differ. Set it when
                comparing runs.
        """
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        if dtype is not None and dtype is not torch.float32:
            raise ValueError(
                f"refusing to cast SmolVLA to {dtype}: modeling_smolvla.py:808 "
                "hardcodes an upcast to float32 before action_out_proj, so a uniform "
                "half-precision cast fails there with 'mat1 and mat2 must have the "
                "same dtype'. Pass dtype=None to keep the checkpoint's own mix, "
                "which is what it was saved with."
            )

        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.dtype = dtype
        self.autocast_dtype = autocast_dtype
        self.dataset_stats_key = dataset_stats_key

        self._generator: torch.Generator | None = None
        if seed is not None:
            self._generator = torch.Generator(device=self.device).manual_seed(seed)

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        vram_before = _vram_used()

        self.adapter_path = Path(adapter_path) if adapter_path else None

        t0 = time.perf_counter()
        if self.adapter_path is None:
            self.policy = SmolVLAPolicy.from_pretrained(checkpoint)
        else:
            self.policy = self._load_with_adapter(checkpoint, self.adapter_path)

        if self.dtype is None:
            self.policy.to(device=self.device)
        else:
            self.policy.to(device=self.device, dtype=self.dtype)
        self.policy.eval()
        load_seconds = time.perf_counter() - t0

        self.config = self.policy.config
        if self.adapter_path is None:
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=self.config,
                pretrained_path=checkpoint,
            )
        else:
            # The fine-tuned checkpoint carries processors built from the training
            # dataset's statistics. Using the base checkpoint's here instead would
            # silently skip normalisation -- see the module docstring -- and the
            # policy was trained on normalised actions, so the output would be
            # wrong by the action std rather than merely unscaled.
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=self.config,
                pretrained_path=str(self.adapter_path / "processors"),
            )

        if dataset_stats_key is not None:
            self._bind_action_stats(dataset_stats_key)

        weight_bytes = sum(p.numel() * p.element_size() for p in self.policy.parameters())
        self.load_report = LoadReport(
            load_seconds=load_seconds,
            param_count=sum(p.numel() for p in self.policy.parameters()),
            trainable_param_count=sum(
                p.numel() for p in self.policy.parameters() if p.requires_grad
            ),
            weight_bytes=weight_bytes,
            vram_bytes_after_load=_vram_used() - vram_before,
        )

        if not self.action_unnormalised:
            logger.warning(
                "postprocessor has no stats bound to the 'action' key: predicted "
                "actions are in NORMALISED units. Pass dataset_stats_key=%r to bind "
                "one of the stored statistic sets.",
                AVAILABLE_STAT_KEYS[0],
            )

    @staticmethod
    def _load_with_adapter(base_checkpoint: str, adapter_path: Path):
        """Load the base policy under the fine-tune's config, then merge the LoRA.

        The adapter alone is not loadable. The saved `policy_config` carries the
        fine-tuning dataset's feature shapes -- two cameras and a 7-value action
        against the base checkpoint's three and six -- and the base weights have to
        be instantiated under those shapes before the adapter will attach.

        The adapter is merged into the weights rather than kept as a wrapper.
        Inference then costs exactly what the base model costs, with no per-layer
        adapter arithmetic, which matters at 50 flow-matching passes per call.
        """
        from peft import PeftModel

        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        config = SmolVLAConfig.from_pretrained(str(adapter_path / "policy_config"))
        config.pretrained_path = base_checkpoint
        policy = SmolVLAPolicy.from_pretrained(base_checkpoint, config=config)

        peft_model = PeftModel.from_pretrained(policy, str(adapter_path))
        merged = peft_model.merge_and_unload()
        logger.info("merged LoRA adapter from %s", adapter_path)
        return merged

    # -- contract ---------------------------------------------------------------

    @property
    def image_keys(self) -> list[str]:
        """Camera keys the checkpoint declares, in config order."""
        return list(self.config.image_features)

    @property
    def image_shape(self) -> tuple[int, int, int]:
        """Declared ``(C, H, W)`` for each camera."""
        return tuple(next(iter(self.config.image_features.values())).shape)

    @property
    def state_dim(self) -> int:
        return int(self.config.robot_state_feature.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.config.action_feature.shape[0])

    @property
    def chunk_size(self) -> int:
        return int(self.config.chunk_size)

    @property
    def n_action_steps(self) -> int:
        """Actions consumed from one chunk before the model is called again."""
        return int(self.config.n_action_steps)

    @property
    def action_stats(self) -> dict[str, dict[str, np.ndarray]]:
        """Every action statistic set stored in the checkpoint."""
        step = self._normalizer_step(self.postprocessor)
        return {} if step is None else dict(step.stats or {})

    @property
    def action_unnormalised(self) -> bool:
        """True when the postprocessor can actually unnormalise actions.

        False means predictions come back in normalised units. See the module
        docstring -- this failure is silent in LeRobot.
        """
        step = self._normalizer_step(self.postprocessor)
        return step is not None and "action" in getattr(step, "_tensor_stats", {})

    # -- inference --------------------------------------------------------------

    @torch.no_grad()
    def predict_action(self, observation: dict[str, Any], instruction: str) -> np.ndarray:
        """Return one action for one observation.

        The policy caches a chunk of :attr:`n_action_steps` actions and only runs the
        network when that queue empties, so most calls are cheap. Call
        :meth:`reset` between episodes or the new episode starts by replaying the
        previous one's leftover actions.

        Args:
            observation: Keys as declared by :attr:`image_keys` plus
                ``observation.state``. Images are float ``[0, 1]`` in ``(C, H, W)``;
                state is a 1-D float vector of :attr:`state_dim` values.
            instruction: Natural-language task, e.g. ``"lift the cube"``.

        Returns:
            A 1-D array of :attr:`action_dim` values. Normalised unless
            :attr:`action_unnormalised` is True.
        """
        batch = self._to_batch(observation, instruction)
        processed = self.preprocessor(batch)
        with self._autocast():
            action = self.policy.select_action(processed, noise=self._make_noise(1))
        return self.postprocessor(action).squeeze(0).float().cpu().numpy()

    @torch.no_grad()
    def predict_action_chunk(
        self,
        observation: dict[str, Any],
        instruction: str,
        noise: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Return the whole action chunk, shape ``(chunk_size, action_dim)``.

        Bypasses the action queue, so every call runs the network. This is the honest
        way to measure inference latency.

        Args:
            noise: Fixed flow-matching noise, from :meth:`make_noise`. Pass the same
                tensor across several observations to attribute output variation to
                the observations rather than to the sampling. Fresh noise is drawn
                when omitted.
        """
        batch = self._to_batch(observation, instruction)
        processed = self.preprocessor(batch)
        if noise is None:
            noise = self._make_noise(1)
        with self._autocast():
            chunk = self.policy.predict_action_chunk(processed, noise=noise)
        return self.postprocessor(chunk).squeeze(0).float().cpu().numpy()

    def make_noise(self, batch_size: int = 1) -> torch.Tensor:
        """Draw a flow-matching noise tensor usable with :meth:`predict_action_chunk`."""
        return self._make_noise(batch_size)

    def reset(self) -> None:
        """Clear the cached action chunk. Call between episodes."""
        self.policy.reset()

    # -- internals --------------------------------------------------------------

    def _autocast(self):
        """Half-precision compute context, or a no-op when autocast is disabled."""
        if self.autocast_dtype is None or self.device.type != "cuda":
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    def _make_noise(self, batch_size: int) -> torch.Tensor:
        """Flow-matching noise in the policy's own dtype.

        The dtype must match ``action_in_proj``, not the backbone. Under the
        checkpoint's mixed precision the backbone is bfloat16 while that projection
        is float32, and getting it wrong fails immediately with "mat1 and mat2 must
        have the same dtype". Reading it off the module is more robust than
        assuming, since a fine-tuned checkpoint may differ.

        Supplying noise at all is necessary because LeRobot's
        ``VLAFlowMatching.sample_noise()`` takes only a shape and a device, so it
        cannot honour any of this. ``sample_actions`` generates its own only when
        ``noise is None``.
        """
        return torch.randn(
            (batch_size, self.config.chunk_size, self.config.max_action_dim),
            device=self.device,
            dtype=self.policy.model.action_in_proj.weight.dtype,
            generator=self._generator,
        )

    def _to_batch(self, observation: dict[str, Any], instruction: str) -> dict[str, Any]:
        batch: dict[str, Any] = {}
        for key, value in observation.items():
            tensor = torch.as_tensor(value) if not torch.is_tensor(value) else value
            # float32 regardless of the weights: the preprocessor normalises in
            # float32 and the model casts to the backbone dtype itself.
            batch[key] = tensor.to(device=self.device, dtype=torch.float32)
        batch["task"] = instruction
        return batch

    @staticmethod
    def _normalizer_step(pipeline: Any) -> Any:
        for step in getattr(pipeline, "steps", []):
            if hasattr(step, "_tensor_stats"):
                return step
        return None

    def _bind_action_stats(self, key: str) -> None:
        """Rebind ``<key>.buffer.action`` statistics onto the bare ``action`` key.

        The checkpoint stores statistics per source dataset. The un/normalizer looks
        up the unqualified feature name, so without this the lookup misses and the
        transform is a silent no-op.
        """
        if key not in AVAILABLE_STAT_KEYS:
            raise ValueError(
                f"unknown dataset_stats_key {key!r}; expected one of {AVAILABLE_STAT_KEYS}"
            )

        source = f"{key}.buffer.action"
        for pipeline in (self.preprocessor, self.postprocessor):
            step = self._normalizer_step(pipeline)
            if step is None:
                continue
            tensor_stats = getattr(step, "_tensor_stats", {})
            if source not in tensor_stats:
                raise KeyError(
                    f"{source!r} not found in checkpoint statistics; available: "
                    f"{sorted(tensor_stats)}"
                )
            tensor_stats["action"] = tensor_stats[source]

            # Both dicts must be patched. On a device or dtype mismatch
            # `_apply_transform` calls `self.to()`, which rebuilds `_tensor_stats`
            # from `stats` -- an alias added only to the former is dropped there and
            # the very next lookup raises KeyError('action').
            if getattr(step, "stats", None) and source in step.stats:
                step.stats["action"] = step.stats[source]

            logger.info("bound %s -> 'action' on %s", source, type(step).__name__)


def _vram_used() -> int:
    if not torch.cuda.is_available():
        return 0
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return total - free
