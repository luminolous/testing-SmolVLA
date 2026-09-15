"""Phase 4 LoRA fine-tuning entry point.

    python scripts/train_lora.py --dry-run          # 10 steps, verifies the setup
    python scripts/train_lora.py                    # full run

Writes a run directory under `results/phase-4/<timestamp>/` with the resolved
config, `metrics.json`, a log and adapter checkpoints.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.log_utils import configure_logging  # noqa: E402
from src.train.lora_finetune import (  # noqa: E402
    OutOfMemoryHint,
    train,
    write_metrics,
)

GIB = 2**30


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs" / "lora_lift.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="10 steps, to verify loss, VRAM and checkpointing")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="reduce first on OOM")
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--debug-logs", action="store_true",
                        help="do not silence third-party loggers")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.grad_accum_steps is not None:
        cfg["training"]["grad_accum_steps"] = args.grad_accum_steps
    if args.gradient_checkpointing:
        cfg["training"]["gradient_checkpointing"] = True
    if args.rank is not None:
        cfg["lora"]["r"] = args.rank
    if args.warmup_steps is not None:
        cfg["training"]["warmup_steps"] = args.warmup_steps
    if args.num_workers is not None:
        cfg["training"]["num_workers"] = args.num_workers

    steps = args.steps
    if args.dry_run:
        steps = steps or 10
        cfg["training"]["log_every"] = 1
        cfg["training"]["eval_every"] = max(steps, 1)
        cfg["training"]["save_every"] = max(steps, 1)

    run_dir = REPO_ROOT / "results" / "phase-4" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False),
                                         encoding="utf-8")

    configure_logging(log_file=run_dir / "train.log", quiet_third_party=not args.debug_logs)

    print(f"run dir     : {run_dir}")
    print(f"mode        : {'DRY RUN' if args.dry_run else 'full training'}")
    print(f"steps       : {steps or cfg['training']['steps']}")
    print(f"batch       : {cfg['training']['batch_size']} x "
          f"{cfg['training']['grad_accum_steps']} accum = "
          f"{cfg['training']['batch_size'] * cfg['training']['grad_accum_steps']} effective")
    print(f"lora rank   : {cfg['lora']['r']}, alpha {cfg['lora']['alpha']}\n")

    t0 = time.perf_counter()
    try:
        state = train(cfg, run_dir, max_steps=steps)
    except OutOfMemoryHint as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    write_metrics(run_dir, state, cfg)

    losses = state.train_losses
    first = sum(losses[:3]) / max(len(losses[:3]), 1)
    last = sum(losses[-3:]) / max(len(losses[-3:]), 1)
    budget = float(cfg["vram"]["budget_gb"])

    print(f"\n{'=' * 70}\nTRAINING SUMMARY\n{'=' * 70}")
    print(f"steps            : {state.step}")
    print(f"epochs           : {state.epoch}")
    print(f"loss first 3     : {first:.4f}")
    print(f"loss last 3      : {last:.4f}   ({'decreasing' if last < first else 'NOT decreasing'})")
    print(f"all finite       : {all(l == l and abs(l) != float('inf') for l in losses)}")
    if state.val_losses:
        print(f"val loss         : " + ", ".join(
            f"step {s}: {v:.4f}" for s, v in state.val_losses))
    print(f"peak VRAM        : {state.peak_vram_gib:.3f} GiB "
          f"(budget {budget:.1f} GiB, "
          f"{'within' if state.peak_vram_gib < budget else 'OVER'})")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"device VRAM      : {(total - free) / GIB:.3f} GiB of {total / GIB:.2f} GiB")
    print(f"wall clock       : {state.seconds_elapsed:.1f} s "
          f"({state.seconds_elapsed / max(state.step, 1):.2f} s/step)")

    full_steps = int(cfg["training"]["steps"])
    if args.dry_run:
        est = state.seconds_elapsed / max(state.step, 1) * full_steps
        print(f"\nprojection to {full_steps} steps: ~{est / 60:.1f} min")

    print(f"\nwritten to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
