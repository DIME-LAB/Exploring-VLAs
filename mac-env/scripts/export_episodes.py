"""Extract one or more episodes from a LeRobotDataset into a new standalone
dataset that preserves the v3.0 schema, then verify it loads.

Implementation strategy: use lerobot's own `delete_episodes` API on the
COMPLEMENT of the kept set. delete_episodes handles sharded video re-
encoding, parquet re-indexing, stats regen, and episode-metadata rewrite —
all the v3.0 plumbing we'd otherwise have to reproduce by hand.

Usage:
    python3 export_episodes.py \\
        --repo-id arjunsinghyadav2/blue_sort_black_bg_colored_cups_v1_440ep \\
        --episodes 7 \\
        --out-dir ~/transfer/blue_sort_ep7 \\
        --out-repo-id local/blue_sort_ep7

    python3 export_episodes.py --episodes 7,12,42 --out-dir ~/transfer/three_eps ...

Stdout (consumed by the SwiftUI EpisodeBrowser app):
    OUT_DIR=<path>            # final dataset root
    KEPT=<count>              # number of episodes in the output
    FRAMES=<count>            # total frames in the output
    VERIFY=PASS|FAIL          # round-trip load result
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--repo-id", required=True)
    p.add_argument(
        "--episodes",
        required=True,
        help="Comma-separated episode indices to KEEP, e.g. '7' or '7,12,42'.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Root directory for the new standalone dataset.",
    )
    p.add_argument(
        "--out-repo-id",
        default=None,
        help="Repo id for the output dataset metadata. "
             "Defaults to '<source_repo_id>__ep<list>'.",
    )
    p.add_argument(
        "--root",
        default=None,
        help="Local dataset root directory (skips HF Hub fetch). "
             "When set, --repo-id is just a label.",
    )
    return p.parse_args()


def parse_episode_list(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError(f"empty episode list: {s!r}")
    return sorted(set(out))


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    keep = parse_episode_list(args.episodes)

    repo_owner_safe = args.repo_id.replace("/", "_")
    if args.out_repo_id is None:
        ep_str = "_".join(str(e) for e in keep) if len(keep) <= 6 else f"{len(keep)}eps"
        out_repo_id = f"local/{repo_owner_safe}__ep{ep_str}"
    else:
        out_repo_id = args.out_repo_id

    print(f"[export] source: {args.repo_id}", file=sys.stderr)
    print(f"[export] keep:   {keep}", file=sys.stderr)
    print(f"[export] out:    {out_dir}", file=sys.stderr)
    print(f"[export] new id: {out_repo_id}", file=sys.stderr)

    # Defensive: same trim/synthesise logic as dump_episode_frame.py — an
    # empty repo_id used to break HF Hub validation.
    repo_id = (args.repo_id or "").strip()
    if not repo_id:
        if args.root:
            leaf = Path(args.root).expanduser().name or "dataset"
            repo_id = f"local/{leaf}"
            print(f"[export] empty --repo-id, synthesised '{repo_id}' from --root",
                  file=sys.stderr)
        else:
            print("ERROR: --repo-id is empty/whitespace and no --root was given.",
                  file=sys.stderr)
            return 2

    src_kwargs: dict = {}
    if args.root:
        src_kwargs["root"] = str(Path(args.root).expanduser().resolve())
    src = LeRobotDataset(repo_id, **src_kwargs)
    total = src.meta.total_episodes
    print(f"[export] source has {total} episodes", file=sys.stderr)

    invalid = [e for e in keep if e < 0 or e >= total]
    if invalid:
        print(f"ERROR: invalid episode indices for source dataset: {invalid}",
              file=sys.stderr)
        return 2

    drop = [i for i in range(total) if i not in set(keep)]
    print(f"[export] deleting {len(drop)} episodes (the complement)…",
          file=sys.stderr)

    if out_dir.exists():
        print(f"[export] WARNING: out_dir exists and will be overwritten by "
              f"delete_episodes: {out_dir}", file=sys.stderr)

    new_ds = delete_episodes(
        dataset=src,
        episode_indices=drop,
        output_dir=str(out_dir),
        repo_id=out_repo_id,
    )

    n_eps = new_ds.meta.total_episodes
    n_frames = new_ds.meta.total_frames
    print(f"[export] OK. wrote {n_eps} episode(s), {n_frames} frame(s)",
          file=sys.stderr)

    # ---- Round-trip verification ----
    print("[verify] re-loading new dataset from disk…", file=sys.stderr)
    verify_pass = True
    try:
        check = LeRobotDataset(out_repo_id, root=str(out_dir))
        if check.meta.total_episodes != n_eps:
            print(f"[verify] FAIL: total_episodes mismatch "
                  f"({check.meta.total_episodes} != {n_eps})", file=sys.stderr)
            verify_pass = False
        if check.meta.total_frames != n_frames:
            print(f"[verify] FAIL: total_frames mismatch "
                  f"({check.meta.total_frames} != {n_frames})", file=sys.stderr)
            verify_pass = False
        # Try to read frame 0 — exercises both parquet and video decoders
        sample = check[0]
        for k in ("observation.images.wrist", "observation.images.top"):
            if k in sample and hasattr(sample[k], "shape"):
                if sample[k].shape != (3, 480, 640):
                    print(f"[verify] FAIL: {k} shape unexpected: "
                          f"{tuple(sample[k].shape)}", file=sys.stderr)
                    verify_pass = False
        if verify_pass:
            print("[verify] PASS", file=sys.stderr)
    except Exception as e:
        print(f"[verify] FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        verify_pass = False

    # Machine-parseable summary on stdout (consumed by the SwiftUI app)
    print(f"OUT_DIR={out_dir}")
    print(f"OUT_REPO_ID={out_repo_id}")
    print(f"KEPT={n_eps}")
    print(f"FRAMES={n_frames}")
    print(f"VERIFY={'PASS' if verify_pass else 'FAIL'}")
    return 0 if verify_pass else 1


if __name__ == "__main__":
    sys.exit(main())
