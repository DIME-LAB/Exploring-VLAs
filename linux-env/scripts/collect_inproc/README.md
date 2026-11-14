# In-process SO-ARM101 prototype

This standalone collector reuses shared geometry, DART, recovery, physical success,
tick audit, fault policy and export. Bounded unrecorded priming stabilized the
camera pipeline live; sixteen joined ticks passed before revealing a floating-base
scene. The production asset now places its root API on the world fixed joint.
Native production readback verifies fixed_base=true and unchanged six-name joint
order; ROS pan/lift sweeps measured zero root drift over272 physics samples.
The shared collector runtime initializes, and a fresh fixed-base held scene is
available. Standalone completed32 tick proofs including motion and paused reset but failed
the moving-camera pose gate (one extra physical pose of lag). Its finalization
and the5+5 dataset acceptance remain pending; no episodes have been accepted.

Collision FK uses measured rigid root transforms and unchanged descendant checks;
probe/collector refuse a floating native base flag. See
[`inprocess-collector.md`](../../../docs/notes/inprocess-collector.md) and
[`inprocess-collector-base-root.md`](../../../docs/notes/inprocess-collector-base-root.md).
Shared ROS collector and extension code remain unchanged; the asset correction
preserves all opinions other than the two root API fields. The offline reproducer
is `isaac/fix_articulation_root.py` and the owned ROS verification entrance is
`fixed_root_validation.py --port OWNED_PORT --output OWNED_REPORT` from a sourced
ROS environment. It requires the owner's running production scene and sim grant.

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
`run_python` and call:

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

Inspect a launch first; `preflight` never boots Isaac.

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
  --kind both --per-kind 5 --max-attempts 30 --step-mode playing --instance 2
```

Only after the joined boot probe passes and the lead grants an episode window,
replace `preflight` with `start --window WINDOW_ID`.
Repeat `--fixture` for the other seeds. The prior Kit log must be under the canonical
storage root. Renderer selection resolves UUID/PCI independently of CUDA; startup
rechecks the current Kit table and owned process GPU allocations. `status` and
`stop` need only `--runtime`; they use the shared identity-bound process supervisor.
Never stop another collector or prompt2assembly process.

The lead's fresh read-only inputs currently live under
`datasets/smoke-primitive-reset-v4/` in canonical storage: five accepted normal
receipts (seeds330000/330002/330003/330004/330005). Re-grasp inputs are pending.
A normal pilot uses `--kind normal --per-kind 2` with those two receipts after
boot acceptance. Accepted seeds are excluded from retries. It is a pilot, not 5+5.

## Boot probe before episodes

The boot probe needs no ROS episode fixtures. Preflight is safe outside a window:

```bash
python3 linux-env/scripts/collect_inproc/launch.py preflight --probe \
  --runtime /mnt/a4500-backup/exploring-vlas/runtime/inproc-playing-next \
  --gpu 0 \
  --renderer-log /mnt/a4500-backup/exploring-vlas/runtime/inproc-playing-20261114-v11/kit.log -- \
  --scene /mnt/a4500-backup/exploring-vlas/inproc/scene-window-20261114/scene.usdc \
  --manifest /mnt/a4500-backup/exploring-vlas/inproc/scene-window-20261114/manifest.json \
  --output /mnt/a4500-backup/exploring-vlas/inproc/probe-playing-next \
  --instance 2 --frames 30 --step-mode playing --motion-check
```

Only after a new grant, replace `preflight` with `start --window WINDOW_ID`. This
plays the timeline at 30 Hz, using four CPU physics substeps per app update and
retaining the existing paused viewport's Hydra engine. It refuses missing/extra
render completions, old producer identities and mismatched clocks without polling.
Both persistent and runtime capture-on-play flags are explicitly enabled and
checked, because Orchestrator.reset copies the former over the latter. The probe
records both flag readbacks, CPU RGB buffer shapes/compute counts and Fabric's
external physics time. The internal empty-list annotation filter is bypassed;
the collector verifies the original stamp, drawable frame and copy count itself.
Texture async preferences are explicitly disabled and read back after warmup;
actual one-update delivery is still a required live check. After every verified
restore, priming discards at most 30 physical/render updates and requires three
consecutive pairs whose original ReferenceTime equals the previous physical tick,
with increasing drawable IDs and exactly one new RGB compute per update. No new
targets are issued during priming. Physics/clock/drive faults are terminal; camera
instability exhausts the explicit bound. Raw restore proof and every discarded
update are retained. The first boundary image after priming also passes the strict
ledger before being discarded; recorded gaps remain terminal.
The immediate restore is checked against the preceding shared tick contract, then
a fresh local recording contract begins at the measured settled boundary. This
permits explicitly unrecorded settling without inventing commands or relaxing
the shared validator. Priming after a paused later reset must pass the same live
probe as priming after startup.
The timeline follows the native Fabric clock; World time is checked independently
for four-substep progression. `Capture.prepare` issues the actual command at t;
`Capture.complete` joins the camera frames stamped t to the retained state and
ACK within zero or one update. An already buffered frame may complete before the
advance; the caller still returns only after exactly one advance. No row reaches
gates or the writer before both join and advance succeed.
`probe.json` records physical advance, paired consumption, two post-reset
frames and no progress during CPU work between updates. A successful probe
finalizes with `BOOT_PROBE_FINISHED`, `kind=boot_probe`, episodes=0;
it is never an accepted dataset. Owner-side finalization follows
worker exit and verified descendant cleanup. Check `inproc-watchdog.json`, the
finalization certificate and `launch.py status`; a worker exit alone is insufficient.

`--motion-check` uses the same collision-checked clean-correction entrance for a
0.12 rad pan movement and records both native camera view and measured poses.
It requires real motion and the wrist view to match the observation pose more
closely than the arrival pose. Inspect the boundary PNGs as well. This physical
pixel-phase check is required before collecting episodes; the option needs at
least eight frames in playing mode. Its native result is still pending.

Both adapters currently default to diagnostic `--step-mode held`; its RGB path
failed live. Select `--step-mode playing` explicitly for the next probe. Do not
start episode collection until that probe passes. For episodes the same option
belongs after `--` with the run arguments. `--physics-only` uses held stepping and
refuses `--step-mode playing`. Fresh ROS-smoke normal/re-grasp fixtures and an
explicit episode window remain prerequisites for the 5+5 acceptance experiment.

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
