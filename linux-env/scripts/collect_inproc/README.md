# In-process SO-ARM101 collector

The isolated prototype reuses the existing expert geometry, DART, recovery,
physical success gate, tick contract, fault policy and exporter. The production
asset's fixed-base correction is verified through the real ROS subscriber and
ArticulationController: six unchanged joint names/order, native fixed_base=true,
and zero root drift over 272 physics samples. Shared collector and extension code
are unchanged. The asset repair and owned ROS verification entrances are
`isaac/fix_articulation_root.py` and `fixed_root_validation.py`.

The 32-tick diagnostic proves that SDG CameraParams matches the retained native
observation through motion/reset; SDK drawable camera metadata is older. Admission
now requires fresh CameraParams and checks its render-camera pose against the
frozen physical snapshot on every capture before finalizing a tick. Invalid
physics handles are terminal. The final native boot passed 32 SDG-gated ticks, a paused second reset and
successful owner finalization at 9d4762c; see the final-native probe receipt. No accepted episodes, 5+5 dataset, export/readback or collector
RTF result is claimed. See [design](../../../docs/notes/inprocess-collector.md)
and [fixed-base evidence](../../../docs/notes/inprocess-collector-base-root.md).

## Offline checks

These commands do not launch Isaac:

```bash
cd ~/Downloads/Exploring-VLAs
OPENBLAS_NUM_THREADS=1 ~/env_isaacsim61/bin/python -m unittest discover \
  -s linux-env/scripts/collect_inproc -p 'test_*.py' -v
~/env_isaacsim61/bin/python -m compileall -q linux-env/scripts/collect_inproc
```

Tests include actual native packet fixtures, shared tick proofs, both control/
image delivery orders, camera metadata freshness/pose, deterministic DART,
primitive collision geometry, real FFmpeg encode/decode and owned finalization.
CPU checks do not prove physical grasp/recovery yield or native exit.

## Held scene and fixtures

The current fixed-base export is
`/mnt/a4500-backup/exploring-vlas/inproc/scene-fixed-root-20261114/`.
Its manifest contains the native articulation audit, physics baseline, joint/box
geometry, asset digests and actual cameras resolved from the collector's own
render-product configuration. Scene/geometry digest mismatches refuse startup.
No camera path is hard-coded from an older document.

To regenerate, only the production owner in its explicitly assigned held Kit may
call this through its own bridge:

```python
import sys
sys.path.insert(0, '/home/aaugus11/Downloads/Exploring-VLAs/linux-env/scripts/collect_inproc')
from snapshot_scene import export_scene
result = export_scene('/mnt/a4500-backup/exploring-vlas/inproc/NEW_SCENE_EXPORT')
```

It requires an initialized, held `collection_runtime`. It never launches/stops a
sim or saves changes to the production stage. Shut down that owned instance before
starting standalone capture. The current asset authors its root API on root_joint;
`articulation_audit.readback(stage, articulation)` checks the native PhysX flag and
composed schema placement without changing the stage.

The prototype uses exact ROS receipts (`params`, `initial_physical`) for seeded
layout reproduction; it has no autonomous layout generator. Fresh read-only
fixed-base fixtures appear under
`/mnt/a4500-backup/exploring-vlas/datasets/smoke-descent-clearance-fixed-v1/`.
The final normal set has five accepted receipts, attempts 0000–0004,
seeds 330000–330004. Re-grasp fixtures are pending. Supply at least five distinct accepted comparison seeds per kind for the
combined acceptance experiment. Earlier floating-base receipts do not fulfill it.
Never write into the ROS output root.

## Boot before episodes

The sim is shared. Never launch without an explicit lead window. Preflight only
prints the reviewed command and input/GPU readiness; it never starts Isaac:

```bash
python3 linux-env/scripts/collect_inproc/launch.py preflight --probe \
  --runtime /mnt/a4500-backup/exploring-vlas/runtime/inproc-final-native-next \
  --gpu 0 \
  --renderer-log /mnt/a4500-backup/exploring-vlas/runtime/inproc-camera-params-20261114-v1/kit.log -- \
  --scene /mnt/a4500-backup/exploring-vlas/inproc/scene-fixed-root-20261114/scene.usdc \
  --manifest /mnt/a4500-backup/exploring-vlas/inproc/scene-fixed-root-20261114/manifest.json \
  --output /mnt/a4500-backup/exploring-vlas/inproc/probe-final-native-next \
  --instance 2 --frames 30 --step-mode playing --motion-check
```

After a grant, replace `preflight` with `start --window WINDOW_ID`. Always use fresh
runtime/output paths. Instance 2 uses domain92 and ports 8781/8782, GPU 0UUID plus
independent Vulkan renderer pinning, CPU PhysX, nice10, persistent caches and
identity-bound process cleanup. GPU 1 is reserved for other agents by the lead, even when all processes share
the same Unix user; a process is owned only through this launcher's identity receipt. `status` and `stop`
require only the owned `--runtime`; they never adopt or stop another instance.

Playing mode retains one paused viewport engine, two fixed render products,
30 Hz sensors and four 120 Hz physical substeps per app update. It explicitly enables
PhysX Fabric pose publication and disables USD pose/velocity writeback, recording
those settings. Current poses come from valid tensor views. Publication before
an update must not advance either clock, physics callbacks or render counters.
Both captureOnPlay flags are enabled; the empty-list annotator gate is bypassed
and replaced by strict per-product counter/stamp/pose checks.

After each immediate verified restore, discard at most 30 unrecorded updates while
holding restored drives, requiring three consecutive correct previous-tick camera
pairs. Retain all priming evidence and raw restore proof. A fresh recording ledger
begins at the measured end of priming; its first ordinary observation is one tick
later. Original ReferenceTime stamps are retained. CameraParams and RGB compute
counts must advance exactly 1 per tick. Exactly one actual target write/ACK at T
belongs to each frozen state, and image T joins within zero or one update. The
caller returns only after exactly one four-substep advance; gates/writers never
receive a row before both pose-checked image join and advance succeed.

With `--frames 30`, the probe proves 32 frames total: 30 before an explicit paused
restore and 2 after. It requires collision-checked 0.12 rad pan motion, measured
physical response, native fixed-base mode and paired capture without CPU work
between updates generating renders. Inspect the boundary PNGs. `camera_pose_phase`
compares SDG metadata to physical poses; `drawable_pose_phase` retains SDK metadata.
Historical reports before the SDG correction used `camera_pose_phase` for SDK and
must not be compared by key alone.

A successful probe must have `BOOT_PROBE_FINISHED`, `kind=boot_probe`, episodes=0,
a successful `inproc-watchdog.json`, affirmative finalized certificate and retired
owned descendants. A worker exit 0 alone is insufficient. The final boot at 9d4762c passed with all these records. Episode collection still
requires its own lead grant; the next window is after the ROS 50-episode pilot
(approximately 2–2.5 hours). No episode launch is admitted by this README.

## Episodes and unchanged export

Only after the complete native boot passes and an episode window is granted,
preflight a two-normal pilot with the fresh fixtures (repeat `--fixture` for each
seed). For the full experiment use `--kind both --per-kind 5` with five normal and
five re-grasp receipts; accepted seeds are excluded from retries. Example pilot:

```bash
python3 linux-env/scripts/collect_inproc/launch.py preflight \
  --runtime /mnt/a4500-backup/exploring-vlas/runtime/inproc-normal-fixed-next \
  --gpu 0 --renderer-log /mnt/a4500-backup/exploring-vlas/runtime/inproc-final-native-20261114-v1/kit.log -- \
  --scene /mnt/a4500-backup/exploring-vlas/inproc/scene-fixed-root-20261114/scene.usdc \
  --manifest /mnt/a4500-backup/exploring-vlas/inproc/scene-fixed-root-20261114/manifest.json \
  --fixture /mnt/a4500-backup/exploring-vlas/datasets/smoke-descent-clearance-fixed-v1/normal/attempt-0000/receipt.json \
  --fixture /mnt/a4500-backup/exploring-vlas/datasets/smoke-descent-clearance-fixed-v1/normal/attempt-0001/receipt.json \
  --output /mnt/a4500-backup/exploring-vlas/datasets/inproc-normal-fixed-next \
  --kind normal --per-kind 2 --max-attempts 6 --step-mode playing --instance 2
```

Only finalized accepted runs go through the existing export entrance:

```bash
pixi run --manifest-path linux-env/pixi.toml python linux-env/scripts/collect_inproc/validate.py \
  /mnt/a4500-backup/exploring-vlas/datasets/inproc-normal-fixed-next \
  --repo-id local/soarm101_inproc_normal_fixed_next \
  --report-dir /mnt/a4500-backup/exploring-vlas/reports/inproc-normal-fixed-next
```

This calls `collect/export.py` unchanged, then reads back every numeric state,
action and timestamp through the pinned LeRobot v3 dataset. Features, degree units,
deployment order,30fps,tasks,cameras and recovery metadata remain unchanged.

The expert uses shared `make_reachable_grasp_plan` yaw-specific IK, TCP offset and
jaw values together; no separate yaw fallback is allowed. Recovery uses the same
shared selection for its clean/miss paths. Receipts retain effective grasp specs
and DART programs. Direct joint paths and sampled primitive sweeps are implemented;
retract/pan/settle or OMPL fallback remains proposed if yield requires it. A carried
block is a planning envelope, never an attachment or teleported object.

## Deferred rendering and comparison

`--physics-only` uses held physics stepping with no camera products and refuses
playing mode. Pose logs contain all measured links/bodies/cameras and exact row
hashes; physics candidates remain non-exportable. `render_replay.py` is an
experimental separate-Kit replay, with physics disabled and explicit poses. It
needs its own fidelity/time/shutdown proof and cannot promote candidates yet.
Browser rendering is evaluated in the design; it is not an interchangeable RTX
backend and has no SO-ARM101 speed or image-gap measurements.

Run the comparison in pixi (the Isaac venv's OpenCV cannot decode H.264 here):

```bash
pixi run --manifest-path linux-env/pixi.toml python linux-env/scripts/collect_inproc/compare.py \
  --reference /mnt/a4500-backup/exploring-vlas/datasets/smoke-descent-clearance-fixed-v1/normal/attempt-0000 \
  --candidate /mnt/a4500-backup/exploring-vlas/datasets/inproc-normal-fixed-next/attempt-0000 \
  --output /mnt/a4500-backup/exploring-vlas/reports/inproc-normal-fixed-next-comparison
```

Both inputs are single attempt directories containing receipt.json, frames.jsonl
and both MP4s. Choose the candidate attempt with the matching seed and target;
it may have another attempt number after rejections. The output directory must
not already exist. The command checks seed/target identity and writes
side-by-sides and stage profiles. It is a trajectory
comparison unless the physical poses match. `--replay` requires identical raw
physical rows and reports PSNR/SSIM/RGB errors. Neither benchmark is claimed until
exercised on the accepted sim experiment.


Boot logs are written to `<runtime>/kit.log`. The completed owned boot used
`/mnt/a4500-backup/exploring-vlas/runtime/inproc-final-native-20261114-v1/kit.log`.
A future boot using the preflight example's new runtime will write its log beneath
that new runtime instead. Runtime/output/report directories are always fresh.
