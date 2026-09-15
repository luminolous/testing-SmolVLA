"""Phase 1 probe: print SmolVLA's full I/O contract and measure its cost.

Runs entirely on synthetic input. No simulator involved -- the point is to know the
model's interface precisely before Phase 2 wires it to robosuite, so that a failure
there can be told apart from a wiring mistake.

    python scripts/probe_model.py
    python scripts/probe_model.py --stats-key so100 --iters 50
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model.smolvla_wrapper import AVAILABLE_STAT_KEYS, SmolVLAWrapper  # noqa: E402

GIB = 2**30
INSTRUCTION = "lift the cube"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def make_observation(model: SmolVLAWrapper, rng: np.random.Generator) -> dict:
    """Synthetic observation matching the declared contract."""
    c, h, w = model.image_shape
    obs = {
        key: rng.random((c, h, w), dtype=np.float32) for key in model.image_keys
    }
    obs["observation.state"] = rng.standard_normal(model.state_dim).astype(np.float32)
    return obs


def report_input_contract(model: SmolVLAWrapper) -> None:
    section("1.2  INPUT CONTRACT")
    cfg = model.config
    c, h, w = model.image_shape

    print(f"cameras expected      : {len(model.image_keys)}")
    for key in model.image_keys:
        print(f"    {key}")
    print(f"declared image shape  : (C={c}, H={h}, W={w})   [config.json input_features]")
    print(f"resized with padding  : {tuple(cfg.resize_imgs_with_padding)}  "
          f"[config.resize_imgs_with_padding]")
    print(f"pixel range in        : [0, 1] float, rescaled to [-1, 1] for SigLIP  "
          f"[modeling_smolvla.py:360]")
    print(f"state dim             : {model.state_dim}  "
          f"(padded to max_state_dim={cfg.max_state_dim}) [modeling_smolvla.py:412]")
    print(f"state normalisation   : {cfg.normalization_mapping['STATE']}")
    print(f"observation steps     : n_obs_steps={cfg.n_obs_steps}")
    print(f"language key          : 'task'  [policy_preprocessor.json step 3]")
    print(f"tokenizer             : {cfg.vlm_model_name}")
    print(f"tokenizer max length  : {cfg.tokenizer_max_length}, padding="
          f"{cfg.pad_language_to}, truncation=True")


def report_output_contract(model: SmolVLAWrapper) -> None:
    section("1.3  OUTPUT CONTRACT")
    cfg = model.config
    print(f"action dim            : {model.action_dim}  "
          f"(padded to max_action_dim={cfg.max_action_dim} internally, then sliced back)")
    print(f"chunked               : YES -- chunk_size={model.chunk_size}, "
          f"n_action_steps={model.n_action_steps}")
    print(f"execution strategy    : select_action() caches the chunk and pops one "
          f"action per call;\n"
          f"                        the network runs once every {model.n_action_steps} "
          f"calls  [modeling_smolvla.py:262-269]")
    print(f"decoding              : flow matching, num_steps={cfg.num_steps}")
    print(f"action normalisation  : {cfg.normalization_mapping['ACTION']} "
          f"(declared in config)")

    stats = model.action_stats
    print(f"\nstatistics stored in checkpoint ({len(stats)} key(s)):")
    for key in sorted(stats):
        entry = stats[key]
        mean = np.asarray(entry.get("mean"))
        std = np.asarray(entry.get("std"))
        print(f"    {key}")
        print(f"        mean {np.array2string(mean, precision=2, suppress_small=True)}")
        print(f"        std  {np.array2string(std, precision=2, suppress_small=True)}")

    print(f"\nunnormalisation active: {model.action_unnormalised}")
    if not model.action_unnormalised:
        print("    ^ actions are returned in NORMALISED units.")
        print("      The checkpoint stores stats under dataset-qualified keys")
        print("      ('so100.buffer.action'), the unnormalizer looks up the bare key")
        print("      'action', the lookup misses, and normalize_processor.py:330")
        print("      returns the tensor untouched -- silently, with no warning.")
        print(f"      Bind one explicitly with --stats-key {AVAILABLE_STAT_KEYS[0]}")


def report_dummy_pass(model: SmolVLAWrapper, n_samples: int = 5) -> np.ndarray:
    section("1.4  DUMMY FORWARD PASS")
    rng = np.random.default_rng(0)

    # Hold the flow-matching noise fixed across observations. SmolVLA is generative:
    # with fresh noise per call, output variation would not be evidence that the
    # images reached the model at all.
    noise = model.make_noise()

    chunks = []
    for i in range(n_samples):
        model.reset()
        chunk = model.predict_action_chunk(
            make_observation(model, rng), INSTRUCTION, noise=noise
        )
        chunks.append(chunk)
        if i == 0:
            print(f"output shape          : {chunk.shape}  "
                  f"(chunk_size, action_dim)")
            print(f"output dtype          : {chunk.dtype}")

    stacked = np.stack(chunks)
    print(f"NaNs                  : {int(np.isnan(stacked).sum())}")
    print(f"Infs                  : {int(np.isinf(stacked).sum())}")
    print(f"value range           : [{stacked.min():.4f}, {stacked.max():.4f}]")
    print(f"mean / std            : {stacked.mean():.4f} / {stacked.std():.4f}")

    print(f"\nper-dimension over {n_samples} random observations "
          f"(first timestep of each chunk):")
    first = stacked[:, 0, :]
    for d in range(first.shape[1]):
        col = first[:, d]
        print(f"    dim {d}: mean={col.mean():+8.4f}  std={col.std():8.4f}  "
              f"range=[{col.min():+8.4f}, {col.max():+8.4f}]")

    # If different observations produce near-identical actions, the images are not
    # reaching the model in a form it can distinguish.
    spread = first.std(axis=0).mean()
    verdict = "varies with input" if spread > 1e-3 else "SATURATED -- suspect preprocessing"
    print(f"\nsensitivity to input  : {verdict} (mean per-dim std {spread:.6f})")
    return stacked


def report_latency(model: SmolVLAWrapper, iters: int, warmup: int) -> None:
    section("1.5  LATENCY AND MEMORY")
    rng = np.random.default_rng(1)
    obs = make_observation(model, rng)

    for _ in range(warmup):
        model.reset()
        model.predict_action_chunk(obs, INSTRUCTION)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        model.reset()
        t0 = time.perf_counter()
        model.predict_action_chunk(obs, INSTRUCTION)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)

    a = np.array(samples) * 1000.0
    print(f"iterations            : {iters} (after {warmup} warmup), batch size 1")
    print(f"mean                  : {a.mean():.2f} ms")
    print(f"p95                   : {np.percentile(a, 95):.2f} ms")
    print(f"min / max             : {a.min():.2f} / {a.max():.2f} ms")

    half = len(a) // 2
    drift = a[half:].mean() - a[:half].mean()
    print(f"drift (2nd - 1st half): {drift:+.2f} ms  "
          f"({'stable' if abs(drift) < 0.1 * a.mean() else 'DRIFTING'})")

    print(f"\nper single action     : {a.mean() / model.n_action_steps:.2f} ms amortised "
          f"over a {model.n_action_steps}-step chunk")
    print(f"published expectation : sub-100 ms per call")
    print(f"verdict               : "
          f"{'within expectation' if a.mean() < 100 else 'ABOVE the published figure'}")

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        free, total = torch.cuda.mem_get_info()
        print(f"\npeak VRAM allocated   : {peak / GIB:.3f} GiB")
        print(f"peak VRAM reserved    : {reserved / GIB:.3f} GiB")
        print(f"device VRAM in use    : {(total - free) / GIB:.3f} GiB of "
              f"{total / GIB:.2f} GiB")
        print(f"budget                : 5.0 GiB -> "
              f"{'within budget' if (total - free) / GIB < 5.0 else 'OVER BUDGET'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="lerobot/smolvla_base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--float32", action="store_true",
                        help="cast all weights to float32 instead of keeping the "
                             "checkpoint mix; doubles the weight footprint")
    parser.add_argument("--autocast", default="none",
                        choices=["bfloat16", "float16", "none"],
                        help="compute dtype via torch.autocast; weights are always "
                             "float32 (see src/model/smolvla_wrapper.py)")
    parser.add_argument("--stats-key", default=None, choices=list(AVAILABLE_STAT_KEYS),
                        help="bind a stored action statistic set so outputs are "
                             "unnormalised")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0,
                        help="seeds the flow-matching noise")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    section("1.1  CHECKPOINT LOAD")
    print(f"checkpoint            : {args.checkpoint}")
    print(f"device                : {args.device}")
    print(f"weight dtype          : "
          f"{'float32 (forced)' if args.float32 else 'checkpoint mix (bf16 + fp32)'}")
    print(f"compute dtype         : {args.autocast} via torch.autocast")
    print(f"stats key             : {args.stats_key or '<none, as shipped>'}")

    model = SmolVLAWrapper(
        checkpoint=args.checkpoint,
        device=args.device,
        dtype=torch.float32 if args.float32 else None,
        autocast_dtype=None if args.autocast == "none" else getattr(torch, args.autocast),
        dataset_stats_key=args.stats_key,
        seed=args.seed,
    )
    r = model.load_report
    print(f"\nload time             : {r.load_seconds:.2f} s")
    print(f"parameters            : {r.param_count / 1e6:.1f} M "
          f"({r.trainable_param_count / 1e6:.1f} M trainable)")
    print(f"weight bytes          : {r.weight_bytes / GIB:.3f} GiB")
    print(f"VRAM after load       : {r.vram_bytes_after_load / GIB:.3f} GiB")

    report_input_contract(model)
    report_output_contract(model)
    report_dummy_pass(model)
    report_latency(model, iters=args.iters, warmup=args.warmup)

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
