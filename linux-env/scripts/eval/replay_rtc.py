#!/usr/bin/env python3
"""Replay real episode observations through PolicyServer, without ROS or Isaac."""

import argparse
import csv
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.transport import services_pb2
from run_async_inference_sim import RECORDED_FPS


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--fps", type=float, default=RECORDED_FPS)
    p.add_argument("--stride", type=int, default=25)
    p.add_argument(
        "--model", default="anirudhrani/smolvla_sim_100ep_fft__10ksteps_h200"
    )
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ds = LeRobotDataset(
        "inbarajaldrin/soarm101_blue_lego_pick_v1",
        root=a.root,
        episodes=[a.episode],
        video_backend="pyav",
    )
    server = PolicyServer(PolicyServerConfig(fps=a.fps))
    specs = RemotePolicyConfig("smolvla", a.model, ds.features, 50, "cuda")
    os.environ["LEROBOT_RTC_ENABLED"] = "1"
    os.environ["LEROBOT_RTC_DEBUG"] = "1"
    server.SendPolicyInstructions(
        services_pb2.PolicySetup(data=pickle.dumps(specs)),
        type("Context", (), {"peer": lambda _: "offline-replay"})(),
    )
    frames = list(range(0, len(ds) - 50, a.stride))
    raw_obs = []
    names = ds.features["observation.state"]["names"]
    for frame in frames:
        sample = ds[frame]
        raw = {name: float(v) for name, v in zip(names, sample["observation.state"])}
        for key in ds.meta.camera_keys:
            raw[key.removeprefix("observation.images.")] = (
                (sample[key].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            )
        raw["task"] = "Pick a blue lego and place it in blue cup"
        raw_obs.append(raw)
    outputs, latencies, guidance = {}, {}, []
    for mode in ("off", "on"):
        server._reset_server()
        server.shutdown_event.clear()
        server.policy.config.rtc_config.enabled = mode == "on"
        server.policy.init_rtc_processor()
        outputs[mode], latencies[mode] = [], []
        for i, (frame, raw) in enumerate(zip(frames, raw_obs)):
            torch.manual_seed(42 + i)
            if mode == "on":
                server.policy.rtc_processor.reset_tracker()
            obs = TimedObservation(
                timestamp=time.time(), timestep=frame, observation=raw, must_go=True
            )
            t0 = time.perf_counter()
            chunk = server._predict_action_chunk(obs)
            latency = time.perf_counter() - t0
            outputs[mode].append(np.stack([x.action.numpy() for x in chunk]))
            latencies[mode].append(latency)
            if mode == "on":
                corrections = [
                    s.correction.norm().item()
                    for s in server.policy.rtc_processor.get_all_debug_steps()
                    if s.correction is not None
                ]
                guidance.append(max(corrections, default=0.0))
            print(f"REPLAY {mode} frame={frame} latency={latency:.3f}", flush=True)
    rows = []
    for i in range(1, len(frames)):
        # Common arrival schedule keeps the OFF/ON boundary comparison paired.
        # Exclude cold load from delay: model was loaded before prediction.
        delay = math.ceil(max(latencies[m][i] for m in outputs) * a.fps)
        if a.stride + delay >= 50:
            raise RuntimeError(
                f"Inference overran overlap at boundary {i}: delay={delay}"
            )
        for mode in outputs:
            old, new = outputs[mode][i - 1], outputs[mode][i]
            last = old[a.stride + delay - 1]
            first = new[delay]
            blended = 0.3 * old[a.stride + delay] + 0.7 * first
            rows.append(
                {
                    "mode": mode,
                    "frame": frames[i],
                    "delay": delay,
                    "raw_jump_l2_deg": float(np.linalg.norm(first - last)),
                    "blended_jump_l2_deg": float(np.linalg.norm(blended - last)),
                    "guidance_correction_l2": guidance[i] if mode == "on" else 0.0,
                }
            )
    with (a.out / "boundaries.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "episode": a.episode,
        "dataset_frames": len(ds),
        "frames": frames,
        "fps": a.fps,
        "stride": a.stride,
        "seed": 42,
        "model": a.model,
        "max_guidance_correction_l2": max(guidance),
        "modes": {},
    }
    for mode in outputs:
        selected = [r for r in rows if r["mode"] == mode]
        summary["modes"][mode] = {
            k: float(np.mean([r[k] for r in selected]))
            for k in ("raw_jump_l2_deg", "blended_jump_l2_deg")
        }
        summary["modes"][mode]["mean_latency_s"] = float(np.mean(latencies[mode][1:]))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        a.out / "chunks.npz", **{m: np.array(v) for m, v in outputs.items()}
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for mode in outputs:
        rr = [r for r in rows if r["mode"] == mode]
        for ax, key in zip(axes, ("raw_jump_l2_deg", "blended_jump_l2_deg")):
            ax.plot(
                [r["frame"] for r in rr],
                [r[key] for r in rr],
                marker=".",
                label=f"RTC {mode}",
            )
            ax.set_ylabel("Boundary jump L2 (degrees)")
            ax.grid(alpha=0.25)
            ax.legend()
    axes[0].set_title(
        "Recorded episode replay: first executable new action − last executed old action"
    )
    axes[1].set_title("Same boundary after client 0.3 old + 0.7 new aggregation")
    axes[1].set_xlabel("Recorded observation frame (episode 0)")
    fig.tight_layout()
    fig.savefig(a.out / "rtc_discontinuity.png", dpi=160)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
