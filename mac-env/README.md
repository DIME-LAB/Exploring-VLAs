# mac-env — pixi environment for SO-ARM101 × lerobot on macOS

Reproducible Mac (Apple Silicon) environment for the recording pipeline described in
`vla_SO-ARM101/docs/LEROBOT_ROS2_MAC_SETUP.md`.

Everything lerobot/ROS2/smoke-test-related on macOS lives here:

- `pixi.toml` — conda + pip environment definition
- `pixi.lock` — exact versions for reproducibility
- `cyclonedds.xml` — localhost-only DDS config (FastDDS discovery fails on macOS)
- `scripts/` — smoke tests for the ROS2 camera + dataset round-trip

## Quick setup

```bash
# Inside this directory:
pixi install

# Install the rebased lerobot fork editable into the env
pixi run pip install -e ../lerobot

# Optional: env vars that consistently make macOS behave
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$(pwd)/cyclonedds.xml"
export KMP_DUPLICATE_LIB_OK=TRUE   # torch + conda libomp coexistence
```

See `vla_SO-ARM101/docs/LEROBOT_ROS2_MAC_SETUP.md` for the full guide.

## Smoke tests

```bash
# Terminal A — synthetic image publisher
pixi run python -u scripts/smoke_publisher.py --topic /smoke/image --fps 30

# Terminal B — L1: ROS2Camera subscribes, reads N frames
pixi run python -u scripts/smoke_l1_camera.py --topic /smoke/image --frames 10

# Terminal B — L2: full LeRobotDataset write + reload from ROS2 topics
pixi run python -u scripts/smoke_l2_dataset.py --frames 30 --fps 15
```

## What is NOT in this directory

- The actual built environment (`.pixi/`) — gitignored. Regenerate via `pixi install`.
- The lerobot source — lives in `../lerobot/` (submodule).
- SO-ARM101 ROS2 packages — live in `../vla_SO-ARM101/`.
