# In-process SO-ARM101 prototype

This standalone collector imports the existing geometry, DART, recovery, physical
success gate, tick audit, fault policy and exporter. It is a prototype awaiting a
lead-granted simulation window. CPU checks do not establish physical grasp yield,
camera freshness, renderer fidelity or throughput. See
[`docs/notes/inprocess-collector.md`](../../../docs/notes/inprocess-collector.md).
No shared ROS collector or extension module is changed.

Run the CPU checks without launching Isaac:

```bash
cd ~/Downloads/Exploring-VLAs
OPENBLAS_NUM_THREADS=1 ~/env_isaacsim61/bin/python -m unittest discover \
  -s linux-env/scripts/collect_inproc -p 'test_*.py' -v
~/env_isaacsim61/bin/python -m compileall -q linux-env/scripts/collect_inproc
```

## Scene input and window

The first experiment consumes recorded ROS receipts (`params`, `initial_physical`)
to reproduce their exact seeded initial scene. Supply at least five distinct normal
and five distinct re-grasp fixtures for the combined acceptance experiment. The
runner selects fixtures of the same recovery kind, preserving the ROS comparison
seeds even when an earlier attempt is rejected. This prototype does not reproduce the
extension's random-layout generator: original ROS snapshots are required inputs.
Old receipts with different body inventories refuse restore. Do not use superseded
pilot receipts as evidence of correctness; they are useful only as pose fixtures.

Inside the explicitly assigned, held ROS collector Kit, load this helper with
`run_python` and call (supply the actual camera prim paths):

```python
import sys
sys.path.insert(0, '/home/aaugus11/Downloads/Exploring-VLAs/linux-env/scripts/collect_inproc')
from snapshot_scene import export_scene
result = export_scene('/mnt/a4500-backup/exploring-vlas/inproc/scene-v1')
```

The helper requires `collection_runtime.runtime()` already installed and held.
It resolves actual camera relationships from the current collector's render-product
graph inputs, as in `configure_clock.py`; no old camera paths are assumed.
It exports a flattened scene, native baseline, joint/primitive geometry and source
digests. It refuses unknown enabled collision shapes. It does not launch or stop
the shared sim. Stop the ROS instance only through its owner before standalone work.

Inspect a launch first; `preflight` never boots Isaac:

With a real prior Kit log, it can report GPU mapping before scene export, but
`inputs_ready` remains false until the exported scene, manifest and matching seed
fixtures exist. A missing-input preflight is not clearance to start collection.

```bash
python3 linux-env/scripts/collect_inproc/launch.py preflight \
  --runtime /mnt/a4500-backup/exploring-vlas/runtime/inproc-31 \
  --gpu 0 --renderer-log /PATH/TO/PRIOR/OWNED/KIT.log -- \
  --scene /mnt/a4500-backup/exploring-vlas/inproc/scene-v1/scene.usdc \
  --manifest /mnt/a4500-backup/exploring-vlas/inproc/scene-v1/manifest.json \
  --fixture /PATH/TO/ROS/attempt-0000/receipt.json \
  --output /mnt/a4500-backup/exploring-vlas/datasets/inproc-v1 \
  --kind both --per-kind 5 --max-attempts 30
```

After the lead grants a window, replace `preflight` with `start --window WINDOW_ID`.
Repeat `--fixture` for the other seeds. The prior Kit log must be under the canonical
storage root. Renderer selection resolves UUID/PCI independently of CUDA; startup
rechecks the current Kit table and owned process GPU allocations. `status` and
`stop` need only `--runtime`; they use the shared identity-bound process supervisor.
Never stop another collector or prompt2assembly process.

The raw attempt layout matches the ROS collector. On a successfully finalized live
run, use the **existing** pixi export entrance:

```bash
pixi run --manifest-path linux-env/pixi.toml python linux-env/scripts/collect_inproc/validate.py \
  /mnt/a4500-backup/exploring-vlas/datasets/inproc-v1 \
  --repo-id local/soarm101_inproc_v1 \
  --report-dir /mnt/a4500-backup/exploring-vlas/reports/inproc-v1
```

This entrance calls `collect/export.py` unchanged, then checks every numeric
state/action/timestamp against the raw episode rows. It uses the pinned LeRobot v3
writer and validates degree features, camera shapes, fps, timing and recovery.
Do not export physics-only candidates or terminal/incomplete runs.

## Deferred rendering and comparison

`run.py --physics-only` creates no camera products. It records all measured link/
body/camera poses, physics contracts and labels; its completion marker is
`PHYSICS_CAPTURE_FINISHED`, accepted-for-training remains false. `render_replay.py`
loads that pose log in a separate Kit, disables physical bodies/joints and poses
links explicitly. It requires the same window/GPU environment and never promotes
candidate receipts. This experimental adapter still needs live clock, scale,
pixel-fidelity and shutdown verification.

Run `compare.py` inside the same pixi environment as validation (the Isaac venv's
OpenCV cannot decode H.264 on this host).
`compare.py --reference ROS_ATTEMPT --candidate INPROC_ATTEMPT --output REPORT`
checks seed/target identity and writes a two-camera side-by-side and stage summary.
It labels this a trajectory comparison, since different expert paths can produce
different poses at the same frame index. For a live-versus-replayed rendering
comparison use `--replay`, which requires identical raw physical rows and reports
PSNR/SSIM and RGB errors. Both tools are experiment scaffolding until exercised on
the granted sim run. Browser rendering is evaluated in the design; no browser
backend is admitted as training-equivalent.

The offline collision model verifies USD-joint FK against physical link poses,
uses the exact enabled boxes and SRDF pairs, and calls shared target/sweep checks.
Only named finger-pad/selected-block contact is allowed. The carried-block model
is a collision envelope, never a physical attachment. Continuous collision proof
beyond the existing 0.01 rad sampling policy is not claimed. The expert currently
tries direct joint paths and rejects blocked paths; retract/pan/settle or OMPL
fallback is a proposed extension if live yield requires it.
