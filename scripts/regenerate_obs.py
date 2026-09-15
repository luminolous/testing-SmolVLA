"""Regenerate image observations for a robomimic dataset.

    python scripts/regenerate_obs.py --n 2            # smoke test
    python scripts/regenerate_obs.py                  # full set

robomimic's raw files store simulator states and actions but no images, and its
registry lists no URL for the image variant of any task. Images therefore have to
be produced locally by replaying each stored state and rendering the cameras
offscreen. This is the expensive step of Phase 3.

The actual replay is delegated to `robomimic.scripts.dataset_states_to_obs`, which
is the reference implementation and handles reward and done modes, observation key
naming and the next-obs bookkeeping. This wrapper exists to enforce the parts that
are project decisions rather than robomimic defaults:

* **Both cameras, wrist included.** robomimic defaults to `agentview` alone at
  84x84. The target deployment uses a wrist camera, so a dataset regenerated
  without `robot0_eye_in_hand` is useless here.
* **Resolution set from what SmolVLA declares**, not from robomimic's default.
* Windows rendering configured before robosuite is imported.
* Runtime and disk projected from a sample before the full run is committed to.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Must precede any robosuite or robomimic import.
from src.envs.robosuite_compat import configure_rendering  # noqa: E402

configure_rendering()

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from src.envs.lift_env import DEFAULT_CAMERAS  # noqa: E402

# SmolVLA's checkpoint declares (3, 256, 256) per camera. Phase 0 measured render
# cost as effectively flat from 128 to 512 px, so this is chosen to match the model
# rather than to save time.
DEFAULT_RESOLUTION = 256


def build_args(dataset: Path, output: Path, cameras: list[str], resolution: int,
               n: int | None, compress: bool) -> Namespace:
    """Full argument namespace for robomimic's regeneration entry point."""
    return Namespace(
        dataset=str(dataset),
        output_name=output.name,
        n=n,
        shaped=False,
        camera_names=cameras,
        camera_height=resolution,
        camera_width=resolution,
        done_mode=2,
        copy_rewards=False,
        copy_dones=False,
        compress=compress,
        exclude_next_obs=True,
    )


def verify(output: Path, cameras: list[str], resolution: int) -> dict:
    """Check the regenerated file, including that the wrist view actually moves.

    A static wrist view means the wrong camera was rendered, and every shape
    assertion in the world passes while that is true.
    """
    report: dict = {"cameras": {}}
    with h5py.File(output, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
        report["n_episodes"] = len(demos)
        report["n_frames"] = int(sum(f[f"data/{d}"].attrs["num_samples"] for d in demos))

        first = f[f"data/{demos[0]}"]
        report["obs_keys"] = sorted(first["obs"].keys())
        report["action_shape"] = list(first["actions"].shape)

        for cam in cameras:
            key = f"{cam}_image"
            if key not in first["obs"]:
                report["cameras"][cam] = {"present": False}
                continue
            images = first["obs"][key][()]
            # Frame-to-frame change: a wrist camera rides the end effector, so its
            # view must change as the arm moves. agentview is fixed and changes only
            # where the arm and cube appear.
            diffs = np.abs(np.diff(images.astype(np.int16), axis=0)).mean(axis=(1, 2, 3))
            report["cameras"][cam] = {
                "present": True,
                "shape": list(images.shape),
                "dtype": str(images.dtype),
                "value_range": [int(images.min()), int(images.max())],
                "mean_frame_delta": float(diffs.mean()),
                "max_frame_delta": float(diffs.max()),
                "static": bool(diffs.mean() < 0.5),
            }
            expected = (resolution, resolution, 3)
            if images.shape[1:] != expected:
                report["cameras"][cam]["shape_mismatch"] = (
                    f"expected {expected}, got {images.shape[1:]}"
                )
    return report


def save_samples(output: Path, cameras: list[str], out_dir: Path, n: int = 3) -> None:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "r") as f:
        demo = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))[0]
        for cam in cameras:
            key = f"{cam}_image"
            if key not in f[f"data/{demo}"]["obs"]:
                continue
            images = f[f"data/{demo}"]["obs"][key]
            steps = np.linspace(0, len(images) - 1, n, dtype=int)
            for step in steps:
                Image.fromarray(images[step]).save(out_dir / f"{cam}_s{step:03d}.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=REPO_ROOT / "data" / "lift" / "ph" / "demo_v141.hdf5")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cameras", nargs="+", default=list(DEFAULT_CAMERAS))
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--n", type=int, default=None,
                        help="regenerate only the first N episodes")
    parser.add_argument("--compress", action="store_true",
                        help="gzip the image datasets; smaller on disk, slower to read")
    parser.add_argument("--save-samples", action="store_true",
                        help="write sample frames for visual inspection")
    args = parser.parse_args()

    if not args.dataset.is_file():
        print(f"dataset not found: {args.dataset}\nrun scripts/download_data.py first")
        return 1

    suffix = f"_{args.n}ep" if args.n else ""
    output = args.output or args.dataset.parent / (
        f"image_{args.resolution}px{suffix}.hdf5"
    )

    with h5py.File(args.dataset, "r") as f:
        total_episodes = len(f["data"].keys())
        total_frames = int(sum(
            f[f"data/{d}"].attrs["num_samples"] for d in f["data"].keys()
        ))
    n_episodes = args.n or total_episodes

    print(f"source      : {args.dataset}")
    print(f"output      : {output}")
    print(f"cameras     : {', '.join(args.cameras)}")
    print(f"resolution  : {args.resolution}x{args.resolution}")
    print(f"episodes    : {n_episodes} of {total_episodes}")
    print(f"compress    : {args.compress}\n")

    from robomimic.scripts.dataset_states_to_obs import dataset_states_to_obs

    t0 = time.perf_counter()
    dataset_states_to_obs(build_args(
        args.dataset, output, args.cameras, args.resolution, args.n, args.compress
    ))
    elapsed = time.perf_counter() - t0

    print(f"\nregenerated in {elapsed:.1f} s")

    report = verify(output, args.cameras, args.resolution)
    size_bytes = output.stat().st_size

    print(f"\n{'=' * 70}\nVERIFICATION\n{'=' * 70}")
    print(f"episodes    : {report['n_episodes']}")
    print(f"frames      : {report['n_frames']}")
    print(f"actions     : {report['action_shape']}")
    print(f"obs keys    : {', '.join(report['obs_keys'])}")
    print(f"file size   : {size_bytes / 2**20:.1f} MiB")

    ok = True
    for cam, info in report["cameras"].items():
        if not info["present"]:
            print(f"\n{cam}: MISSING")
            ok = False
            continue
        print(f"\n{cam}")
        print(f"    shape        : {info['shape']} {info['dtype']}")
        print(f"    value range  : {info['value_range']}")
        print(f"    frame delta  : mean {info['mean_frame_delta']:.2f}, "
              f"max {info['max_frame_delta']:.2f}")
        if "shape_mismatch" in info:
            print(f"    SHAPE MISMATCH: {info['shape_mismatch']}")
            ok = False
        if info["static"]:
            print("    STATIC VIEW -- this camera does not change between frames. "
                  "For a wrist camera that means the wrong camera was rendered.")
            ok = False

    per_episode = elapsed / n_episodes
    per_frame_bytes = size_bytes / max(report["n_frames"], 1)
    print(f"\n{'=' * 70}\nPROJECTION TO THE FULL SET ({total_episodes} episodes, "
          f"{total_frames} frames)\n{'=' * 70}")
    print(f"observed    : {per_episode:.2f} s/episode, "
          f"{per_frame_bytes / 1024:.1f} KiB/frame")
    print(f"runtime     : ~{per_episode * total_episodes / 60:.1f} min")
    print(f"disk        : ~{per_frame_bytes * total_frames / 2**30:.2f} GiB")

    if args.save_samples:
        sample_dir = REPO_ROOT / "results" / "phase-3" / "samples"
        save_samples(output, args.cameras, sample_dir)
        print(f"\nsample frames written to {sample_dir}")

    meta = {
        "source": str(args.dataset),
        "output": str(output),
        "cameras": args.cameras,
        "resolution": args.resolution,
        "n_episodes_regenerated": n_episodes,
        "compress": args.compress,
        "elapsed_seconds": elapsed,
        "size_bytes": size_bytes,
        "verification": report,
    }
    (output.parent / f"{output.stem}_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"\n{'OK' if ok else 'PROBLEMS FOUND -- see above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
