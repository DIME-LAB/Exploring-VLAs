"""Dump a single frame from a LeRobotDataset episode as a PNG.

Used by EpisodeBrowser.app's "Save WRIST" / "Save WORKSPACE" buttons —
spawned via Process() with the args below.

Usage:
    python3 dump_episode_frame.py \\
        --repo-id arjunsinghyadav2/blue_sort_black_bg_colored_cups_v1_440ep \\
        --episode 7 \\
        --timestamp 0.475 \\
        --view wrist \\
        --out /Users/aldrininbaraj/transfer/

`--view` is one of: 'wrist' | 'top' | 'both'. With 'both', two PNGs are
written. Output filename is auto-generated as
    ep<E>_t<MS>ms_<view>.png
where MS is the closest sample's timestamp in milliseconds (rounded).

Prints the absolute path of every written PNG, one per line, on stdout —
parsed by the SwiftUI app to populate the status field.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


VIEW_MAP = {
    "wrist": ["observation.images.wrist"],
    "top":   ["observation.images.top"],
    "both":  ["observation.images.wrist", "observation.images.top"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--repo-id", required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--timestamp", type=float, default=0.0,
                   help="Target timestamp in seconds. Closest sample wins.")
    p.add_argument("--view", choices=list(VIEW_MAP.keys()), default="both")
    p.add_argument("--out", default=str(Path.home() / "transfer"),
                   help="Output directory (created if missing).")
    p.add_argument("--root", default=None,
                   help="Local dataset root directory (skips HF Hub fetch). "
                        "When set, --repo-id is just a label.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Defensive: empty / whitespace-only repo_id used to crash through
    # huggingface_hub.utils._validators with a confusing traceback (the
    # GUI sometimes persisted "\n" through @AppStorage).
    repo_id = (args.repo_id or "").strip()
    if not repo_id:
        if args.root:
            leaf = Path(args.root).expanduser().name or "dataset"
            repo_id = f"local/{leaf}"
            print(f"[dump_frame] empty --repo-id, synthesised '{repo_id}' from --root",
                  file=sys.stderr)
        else:
            print("ERROR: --repo-id is empty/whitespace and no --root was given. "
                  "Pass a valid HuggingFace repo id or a local --root path.",
                  file=sys.stderr)
            return 2

    ds_kwargs: dict = {"episodes": [args.episode]}
    if args.root:
        ds_kwargs["root"] = str(Path(args.root).expanduser().resolve())
    ds = LeRobotDataset(repo_id, **ds_kwargs)

    # Closest-sample search by timestamp
    best_idx, best_delta = 0, float("inf")
    for i in range(len(ds)):
        t = float(ds[i]["timestamp"])
        d = abs(t - args.timestamp)
        if d < best_delta:
            best_delta, best_idx = d, i

    sample = ds[best_idx]
    chosen_t = float(sample["timestamp"])
    ms = int(round(chosen_t * 1000))

    keys = VIEW_MAP[args.view]
    written: list[str] = []
    for k in keys:
        if k not in sample:
            print(f"WARN: key {k!r} not in sample (have: "
                  f"{[kk for kk in sample if 'image' in kk]})",
                  file=sys.stderr)
            continue
        v = sample[k]
        # CHW float [0,1] -> HWC uint8 BGR
        img = (v.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        view_short = k.split(".")[-1]  # 'wrist' or 'top'
        if view_short == "top":
            view_short = "workspace"  # match the side-by-side labelling

        out_path = out_dir / f"ep{args.episode}_t{ms}ms_{view_short}.png"
        cv2.imwrite(str(out_path), img_bgr)
        written.append(str(out_path))
        print(out_path)

    if not written:
        print("ERROR: no images written", file=sys.stderr)
        return 1
    print(
        f"# episode={args.episode} target_t={args.timestamp:.4f} "
        f"chosen_t={chosen_t:.4f} (delta={best_delta*1000:.1f} ms) "
        f"frame_idx={best_idx}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
