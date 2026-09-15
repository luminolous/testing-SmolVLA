"""Fetch robomimic demonstrations.

    python scripts/download_data.py                 # Lift, proficient-human, raw
    python scripts/download_data.py --dataset-type mh

Downloads the **raw** file, which stores simulator states and actions but no
images. robomimic's registry lists no URL for the `image` variant of any task --
images do not ship and must be regenerated locally by replaying the stored states.
That is what `scripts/regenerate_obs.py` does.

Records size, episode count, frame count and a SHA-256 digest into
`data/<task>/<type>/manifest.json`, so a later phase can tell whether it is working
from the file it thinks it is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CHUNK = 1 << 20


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("Content-Length", 0))

    written = 0
    t0 = time.perf_counter()
    with dest.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=CHUNK):
            handle.write(chunk)
            written += len(chunk)
            if total:
                pct = written / total * 100
                mb_s = written / 1e6 / max(time.perf_counter() - t0, 1e-9)
                print(f"\r  {written / 1e6:7.1f} / {total / 1e6:.1f} MB "
                      f"({pct:5.1f}%)  {mb_s:5.1f} MB/s", end="", flush=True)
    print()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect(path: Path) -> dict:
    """Read episode and frame counts, and the robosuite version that wrote it."""
    import h5py

    with h5py.File(path, "r") as f:
        data = f["data"]
        demos = sorted(data.keys())
        lengths = [int(data[d].attrs["num_samples"]) for d in demos]
        env_args = json.loads(data.attrs["env_args"])
        info = {
            "n_episodes": len(demos),
            "n_frames": int(sum(lengths)),
            "episode_length_min": min(lengths),
            "episode_length_max": max(lengths),
            "episode_length_mean": sum(lengths) / len(lengths),
            "env_name": env_args.get("env_name"),
            "env_kwargs": env_args.get("env_kwargs", {}),
            "keys_per_demo": sorted(data[demos[0]].keys()),
        }
        if "total" in data.attrs:
            info["total_attr"] = int(data.attrs["total"])
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="lift")
    parser.add_argument("--dataset-type", default="ph",
                        help="ph = proficient human. Cleaner than mh and the "
                             "resulting policy is easier to interpret")
    parser.add_argument("--hdf5-type", default="raw", choices=["raw", "low_dim"])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--force", action="store_true", help="re-download if present")
    args = parser.parse_args()

    import robomimic

    registry = robomimic.DATASET_REGISTRY
    if args.task not in registry:
        print(f"unknown task {args.task!r}; available: {sorted(registry)}")
        return 1
    entry = registry[args.task][args.dataset_type].get(args.hdf5_type)
    if entry is None or entry.get("url") is None:
        print(f"no URL for {args.task}/{args.dataset_type}/{args.hdf5_type}. "
              f"Image datasets in particular are not distributed -- regenerate them "
              f"locally with scripts/regenerate_obs.py")
        return 1

    url = entry["url"]
    dest_dir = args.out / args.task / args.dataset_type
    dest = dest_dir / Path(url).name

    print(f"task        : {args.task}/{args.dataset_type}/{args.hdf5_type}")
    print(f"url         : {url}")
    print(f"destination : {dest}")

    if dest.is_file() and not args.force:
        print(f"\nalready present ({dest.stat().st_size / 1e6:.1f} MB); "
              f"pass --force to re-download")
    else:
        print()
        download(url, dest)

    print("\nhashing ...")
    digest = sha256(dest)
    info = inspect(dest)

    manifest = {
        "task": args.task,
        "dataset_type": args.dataset_type,
        "hdf5_type": args.hdf5_type,
        "url": url,
        "file": dest.name,
        "size_bytes": dest.stat().st_size,
        "sha256": digest,
        "horizon": entry.get("horizon"),
        **info,
    }
    manifest_path = dest_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nsize          : {manifest['size_bytes'] / 1e6:.2f} MB")
    print(f"sha256        : {digest}")
    print(f"episodes      : {info['n_episodes']}")
    print(f"frames        : {info['n_frames']}")
    print(f"episode length: {info['episode_length_min']}-{info['episode_length_max']} "
          f"(mean {info['episode_length_mean']:.1f})")
    print(f"env           : {info['env_name']}")
    print(f"per-demo keys : {', '.join(info['keys_per_demo'])}")
    print(f"\nmanifest      : {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
