#!/usr/bin/env python3
"""Upload a local lerobot dataset to the Hugging Face Hub under a new repo_id.

Usage:
    pixi run --manifest-path linux-env/pixi.toml \
        python linux-env/scripts/upload_dataset_to_hf.py \
        --local-name rec_222812 \
        --repo-id inbarajaldrin/soarm101_blue_lego_pick_v1 \
        [--private] [--no-videos]
"""
import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--local-name", required=True,
                   help="Folder name under ~/.cache/huggingface/lerobot/local/")
    p.add_argument("--repo-id", required=True,
                   help="Target HF repo (e.g. user/dataset_name)")
    p.add_argument("--private", action="store_true")
    p.add_argument("--no-videos", action="store_true",
                   help="Skip uploading videos/ (saves bandwidth)")
    p.add_argument("--tags", nargs="*",
                   default=["lerobot", "robotics", "so-arm101", "isaac-sim", "pick-and-place"])
    args = p.parse_args()

    root = Path.home() / ".cache/huggingface/lerobot/local" / args.local_name
    if not (root / "meta" / "info.json").exists():
        raise SystemExit(f"no dataset at {root}")

    print(f"Loading {root} as {args.repo_id} …")
    ds = LeRobotDataset(repo_id=args.repo_id, root=str(root))
    print(f"  episodes={ds.meta.total_episodes} frames={ds.meta.total_frames}")

    print(f"Pushing to https://huggingface.co/datasets/{args.repo_id} "
          f"(private={args.private}, videos={not args.no_videos}) …")
    ds.push_to_hub(
        tags=args.tags,
        private=args.private,
        push_videos=not args.no_videos,
    )
    print("done.")


if __name__ == "__main__":
    main()
