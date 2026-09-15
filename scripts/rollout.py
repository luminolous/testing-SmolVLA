"""Phase 2 zero-shot rollout of SmolVLA on robosuite `Lift`.

    python scripts/rollout.py --n-episodes 2          # smoke test
    python scripts/rollout.py --n-episodes 20         # baseline

Writes a run directory under `results/phase-2/<timestamp>/` containing the resolved
config, `metrics.json`, and a log.

**What this measures.** The action mapping is arbitrary -- SmolVLA's base checkpoint
speaks SO-100 joint degrees, `OSC_POSE` speaks a normalised delta pose, and the two
do not correspond. A low success rate here is the expected and uninformative
outcome. The numbers that carry information are the action statistics and the
clipping frequency, which say whether the model is producing varied, plausible
output at all, and whether the pipeline runs end to end without leaking memory.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.envs.action_adapter import ActionAdapter  # noqa: E402
from src.envs.lift_env import make_lift_env  # noqa: E402
from src.envs.obs_adapter import ObsAdapter  # noqa: E402
from src.model.smolvla_wrapper import SmolVLAWrapper  # noqa: E402

logger = logging.getLogger("rollout")
GIB = 2**30


@dataclass
class EpisodeResult:
    index: int
    success: bool
    length: int
    first_success_step: int | None
    env_seconds: float
    inference_seconds: float
    n_inference_calls: int


@dataclass
class RunMetrics:
    n_episodes: int = 0
    n_success: int = 0
    episode_lengths: list[int] = field(default_factory=list)
    episodes: list[dict] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.n_success / self.n_episodes if self.n_episodes else 0.0


def build_run_dir() -> Path:
    run_dir = REPO_ROOT / "results" / "phase-2" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path, verbose: bool) -> None:
    handlers: list[logging.Handler] = [logging.FileHandler(run_dir / "rollout.log", encoding="utf-8")]
    if verbose:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def run_episode(
    index: int,
    env,
    model: SmolVLAWrapper,
    obs_adapter: ObsAdapter,
    action_adapter: ActionAdapter,
    instruction: str,
    horizon: int,
    frame_dir: Path | None,
    save_frames_every: int,
) -> EpisodeResult:
    obs = env.reset()
    model.reset()

    success = False
    first_success_step: int | None = None
    env_seconds = 0.0
    inference_seconds = 0.0
    n_calls_before = 0
    step = 0

    for step in range(horizon):
        model_obs = obs_adapter(obs)

        t0 = time.perf_counter()
        raw_action = model.predict_action(model_obs, instruction)
        inference_seconds += time.perf_counter() - t0

        env_action = action_adapter(raw_action)

        t0 = time.perf_counter()
        obs, reward, done, _ = env.step(env_action)
        env_seconds += time.perf_counter() - t0

        # reward_shaping is off, so reward is 1.0 exactly on task success.
        if reward >= 1.0 and not success:
            success = True
            first_success_step = step

        if frame_dir is not None and save_frames_every and step % save_frames_every == 0:
            _save_frame(frame_dir, index, step, obs)

        if done:
            break

    # The policy only invokes the network when its action queue empties.
    n_calls = int(np.ceil((step + 1) / model.n_action_steps))
    del n_calls_before

    return EpisodeResult(
        index=index,
        success=success,
        length=step + 1,
        first_success_step=first_success_step,
        env_seconds=env_seconds,
        inference_seconds=inference_seconds,
        n_inference_calls=n_calls,
    )


def _save_frame(frame_dir: Path, episode: int, step: int, obs: dict) -> None:
    from PIL import Image

    from src.envs.obs_adapter import upright

    for cam in ("agentview", "robot0_eye_in_hand"):
        key = f"{cam}_image"
        if key in obs:
            # Saved frames must match what the model sees, or they are worse than
            # useless for debugging.
            Image.fromarray(upright(obs[key])).save(
                frame_dir / f"ep{episode:02d}_s{step:03d}_{cam}.png"
            )


def summarise(
    metrics: RunMetrics,
    obs_adapter: ObsAdapter,
    action_adapter: ActionAdapter,
    results: list[EpisodeResult],
) -> dict:
    lengths = np.array(metrics.episode_lengths, dtype=float)
    env_s = sum(r.env_seconds for r in results)
    inf_s = sum(r.inference_seconds for r in results)
    total_steps = int(lengths.sum())
    total_calls = sum(r.n_inference_calls for r in results)

    stats = action_adapter.stats
    summary = {
        "success_rate": metrics.success_rate,
        "n_success": metrics.n_success,
        "n_episodes": metrics.n_episodes,
        "episode_length_mean": float(lengths.mean()) if total_steps else 0.0,
        "episode_length_std": float(lengths.std()) if total_steps else 0.0,
        "total_steps": total_steps,
        "timing": {
            "env_ms_per_step": env_s / total_steps * 1000 if total_steps else 0.0,
            "inference_ms_per_step": inf_s / total_steps * 1000 if total_steps else 0.0,
            "inference_ms_per_call": inf_s / total_calls * 1000 if total_calls else 0.0,
            "n_inference_calls": total_calls,
        },
        "action": {
            "clip_rate": stats.clip_rate,
            "per_dim_clip_rate": stats.per_dim_clip_rate.tolist(),
            "raw_mean": stats.raw_mean.tolist(),
            "raw_std": stats.raw_std.tolist(),
        },
        "observation": {
            "n_calls": obs_adapter.stats.n_calls,
            "state_min": obs_adapter.stats.state_min.tolist()
            if obs_adapter.stats.state_min is not None
            else None,
            "state_max": obs_adapter.stats.state_max.tolist()
            if obs_adapter.stats.state_max is not None
            else None,
            "image_min": obs_adapter.stats.image_min,
            "image_max": obs_adapter.stats.image_max,
        },
        "episodes": [asdict(r) for r in results],
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        summary["vram"] = {
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
            "device_in_use_gib": (total - free) / GIB,
        }
    return summary


def print_summary(summary: dict, action_adapter: ActionAdapter) -> None:
    names = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]
    print(f"\n{'=' * 72}\nROLLOUT SUMMARY\n{'=' * 72}")
    print(f"success rate        : {summary['success_rate']:.1%} "
          f"({summary['n_success']}/{summary['n_episodes']})")
    print(f"episode length      : {summary['episode_length_mean']:.1f} "
          f"+/- {summary['episode_length_std']:.1f} steps")

    t = summary["timing"]
    print(f"\ntiming per step     : env {t['env_ms_per_step']:.2f} ms | "
          f"inference {t['inference_ms_per_step']:.2f} ms (amortised)")
    print(f"inference per call  : {t['inference_ms_per_call']:.1f} ms "
          f"x {t['n_inference_calls']} calls")

    a = summary["action"]
    print(f"\naction clipping     : {a['clip_rate']:.1%} of all values")
    print(f"{'dim':<6}{'mean':>10}{'std':>10}{'clip rate':>12}")
    for i, name in enumerate(names):
        print(f"{name:<6}{a['raw_mean'][i]:>10.4f}{a['raw_std'][i]:>10.4f}"
              f"{a['per_dim_clip_rate'][i]:>11.1%}")

    o = summary["observation"]
    print(f"\nimage value range   : [{o['image_min']:.3f}, {o['image_max']:.3f}]")
    if o["state_min"] is not None:
        print("state range         : "
              + np.array2string(np.array(o["state_min"]), precision=1, suppress_small=True)
              + " .. "
              + np.array2string(np.array(o["state_max"]), precision=1, suppress_small=True))

    if "vram" in summary:
        v = summary["vram"]
        print(f"\npeak VRAM allocated : {v['peak_allocated_gib']:.3f} GiB")
        print(f"device VRAM in use  : {v['device_in_use_gib']:.3f} GiB")

    print(f"\n{action_adapter.describe_mapping()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs" / "phase2_rollout.yaml")
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--action-space", default=None, choices=["normalized", "degrees"])
    parser.add_argument("--stats-key", default=None, choices=["so100", "so100-blue", "so100-red"])
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render", action="store_true",
                        help="save camera frames from episode 0 into the run dir")
    parser.add_argument("--raw-state", action="store_true",
                        help="feed unscaled degrees instead of a [-1, 1] state")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.n_episodes is not None:
        cfg["rollout"]["n_episodes"] = args.n_episodes
    if args.action_space is not None:
        cfg["rollout"]["action_space"] = args.action_space
    if args.stats_key is not None:
        cfg["model"]["dataset_stats_key"] = args.stats_key
    if args.n_action_steps is not None:
        cfg["model"]["n_action_steps"] = args.n_action_steps
    if args.horizon is not None:
        cfg["env"]["horizon"] = args.horizon
    if args.seed is not None:
        cfg["rollout"]["seed"] = args.seed
    if args.raw_state:
        cfg["rollout"]["normalize_state"] = False
    if args.render and not cfg["rollout"]["save_frames_every"]:
        cfg["rollout"]["save_frames_every"] = 20

    run_dir = build_run_dir()
    setup_logging(run_dir, args.verbose)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    env_cfg, model_cfg, roll_cfg = cfg["env"], cfg["model"], cfg["rollout"]
    np.random.seed(roll_cfg["seed"])
    torch.manual_seed(roll_cfg["seed"])

    print(f"run dir: {run_dir}")
    print(f"loading {model_cfg['checkpoint']} ...")
    model = SmolVLAWrapper(
        checkpoint=model_cfg["checkpoint"],
        device=model_cfg["device"],
        autocast_dtype=None if model_cfg["autocast"] == "none"
        else getattr(torch, model_cfg["autocast"]),
        dataset_stats_key=model_cfg["dataset_stats_key"],
        seed=roll_cfg["seed"],
    )
    model.policy.config.n_action_steps = int(model_cfg["n_action_steps"])
    print(model.load_report)
    print(f"actions unnormalised: {model.action_unnormalised}")

    env = make_lift_env(
        cameras=tuple(env_cfg["cameras"]),
        resolution=int(env_cfg["resolution"]),
        controller=env_cfg["controller"],
        robot=env_cfg["robot"],
        control_freq=int(env_cfg["control_freq"]),
        horizon=int(env_cfg["horizon"]),
    )

    obs_adapter = ObsAdapter(
        state_dim=model.state_dim,
        normalize_state=bool(roll_cfg.get("normalize_state", True)),
    )
    action_adapter = ActionAdapter(action_space=roll_cfg["action_space"])
    logger.info("action mapping:\n%s", action_adapter.describe_mapping())

    frame_dir = None
    if roll_cfg["save_frames_every"]:
        frame_dir = run_dir / "frames"
        frame_dir.mkdir(exist_ok=True)

    metrics = RunMetrics()
    results: list[EpisodeResult] = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        for i in range(int(roll_cfg["n_episodes"])):
            t0 = time.perf_counter()
            result = run_episode(
                index=i,
                env=env,
                model=model,
                obs_adapter=obs_adapter,
                action_adapter=action_adapter,
                instruction=model_cfg["instruction"],
                horizon=int(env_cfg["horizon"]),
                frame_dir=frame_dir if i == 0 else None,
                save_frames_every=int(roll_cfg["save_frames_every"]),
            )
            results.append(result)
            metrics.n_episodes += 1
            metrics.n_success += int(result.success)
            metrics.episode_lengths.append(result.length)

            vram = (torch.cuda.max_memory_allocated() / GIB
                    if torch.cuda.is_available() else 0.0)
            line = (f"episode {i:3d}  success={result.success!s:<5} "
                    f"len={result.length:3d}  {time.perf_counter() - t0:5.1f}s  "
                    f"peak VRAM {vram:.2f} GiB")
            print(line)
            logger.info(line)
    finally:
        env.close()

    summary = summarise(metrics, obs_adapter, action_adapter, results)
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary, action_adapter)
    print(f"\nwritten to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
