# SO-ARM101 articulation base: diagnostic proposal

The production robot asset authors ArticulationRootAPI on its world fixed joint
root_joint. The asset-only correction preserves all joint names, limits, drives,
gains, colliders, visuals, masses, transforms and external references. Its offline
reproducer is `isaac/fix_articulation_root.py`; repeated application is idempotent.
The extension load and GUI code are unchanged.

Live production verification on 2026-11-14 confirms native fixed_base=true, zero
root DOFs, six joint DOFs and eleven links. The exact six-name positional order is
unchanged. Six ROS pan/lift goals passed through the production subscriber and
ArticulationController; 272 actual physics callback samples measured maximum base
drift 0 m and 0 rad. The shared collector runtime initialized with the same six
joint names, and a fresh held fixed-base scene was exported. The standalone
joined/reset/success-exit probe on that export is a separate acceptance gate.

## Evidence

The 2026-11-14 10:48Z owned boot at 77c0aa1 passed bounded priming after seven
updates (camera completion counts 0,0,1,2,1,1,1 per camera). Sixteen subsequent
joined boot ticks each advanced four physics steps and produced one correctly
stamped camera pair. The first pan response moved the measured joint from
-0.0000344 to 0.0350221 rad. Its wrist drawable view matched the retained
observation camera pose with maximum matrix error 2.04e-7; its error against the
arrival pose was 0.03727. These are partial boot results, not dataset acceptance.

The next command stopped because the collision model's FK anchored its root to
the exported baseline, while the measured base moved. Inverting the USD joint
chain from the recorded measured wrist transform and measured arrival joints
infers approximately 0.107 mm translation and 6.27 mrad rotation relative to that
exported root. This is an inference from the wrist, not a direct root readback;
the failed probe did not retain the full failing snapshot. The new probe retains
that snapshot and root-motion diagnostics. Neither later reset nor successful
producer finalization was reached. No episodes ran. The sim was released at
10:49:22Z with no owned processes or GPU contexts and ports8781/8782 closed.

Read-only OpenUSD inspection found:

| Source | ArticulationRootAPI | Enabled world fixed joint | SHA256 |
| --- | --- | --- | --- |
| Held production export `inproc/scene-window-20261114/scene.usdc` | `/World/SO_ARM101/base_link` | sibling `/World/SO_ARM101/root_joint`, body0 empty, body1 base_link | `613813660d0fdf42bc972c7e9b389867c00bd6ba51cf68aeb4ceb1a228e1c22d` |
| Production asset `isaac/exts/soarm101-dt/assets/SO-ARM101-USD.usd` | `/so101_new_calib/base_link` | sibling `/so101_new_calib/root_joint`, body0 empty, body1 base_link | `996d6e00d5ab716e607fa87c59b32907fc108bd4df4d1c9de54563de0c75de30` |

The extension's `load_robot` references that production asset at
`/World/SO_ARM101` and creates its articulation view at the robot path. The held
export was flattened from the actual collector stage; no standalone setup moves
its root API. These establish asset/export authoring, not the current live
production PhysX flag.

NVIDIA documents fixed-base authoring on the world joint or an ancestor containing
that joint; applying the root API directly to a rigid-body link selects a floating
root. The observed placement therefore suggests a floating articulation held by
an external constraint. Confirm it with the native flag rather than inferring it
from six controlled joints. [Omni Physics 110 articulation documentation](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/rigid_bodies_articulations/articulations.html).

## Native reads and acceptance gate

Installed `omni.physics.tensors/api.py` documents
`ArticulationView.shared_metatype.fixed_base`, `dof_count` and `link_count`.
`collect_inproc/articulation_audit.py` reads these and the composed USD root API
and world-joint relationships without modifying the stage. The six joint DOFs are
reported separately; root DOFs are explicitly derived from the native base flag.
The standalone probe and collector refuse a floating flag before motion/episodes.
Its startup warmup remains unrecorded. The scene export helper includes the same
readback in future manifests.

In the production owner's next explicitly assigned held sim, after the runtime is
initialized, the owner can call through its own Kit bridge:

```python
import sys
sys.path.insert(0, '/home/aaugus11/Downloads/Exploring-VLAs/linux-env/scripts/collect_inproc')
from articulation_audit import readback
import collection_runtime
r = collection_runtime.runtime()
result = readback(r.stage, r.art)
```

The production runtime readback was captured after installing the shared collector
runtime in the owned production sim. The earlier owned export audit independently
confirmed the pre-fix floating flag. Keep this readback in future smoke receipts.
Offline tests do not replace either measurement.

## Asset correction and verification

The production asset moves only the root API from base_link to root_joint.
Independent full Sdf serialization comparison found exactly four differing API
schema lines; all other opinions and relative references are identical. This
choice preserves extension loading, joint controller wiring and GUI behavior.
The corrected asset SHA256 is
`5fab6704255a2f1fa3b52a1fe710b332dc2456fa20657b2c78d46bda2c4f6123`.
The native production flag was read directly; before correction the owned export
returned fixed_base=false with six joint DOFs, eleven links and six root DOFs
derived from that flag. After correction the production runtime returns
fixed_base=true with the same joint DOFs/links and root DOFs zero.

ROS verification published deployment-order JointState messages on domain92 and
checked the actual subscriber output, ArticulationController indices[0..5], native
applied drive targets and measured convergence for pan+/−0.12/home and
lift+/−0.12/home. Native base poses were sampled on every physical callback during
the sweep. Report: `inproc/fixed-root-verification-20261114/ros-pan-lift-sweep.json`.
Runtime installation and held export receipts are beside it. The new held export
is `inproc/scene-fixed-root-20261114/{scene.usdc,manifest.json}`; its manifest
contains the production native articulation audit. Existing primitive collider
rest/contact offsets also read back unchanged. Independent CPU checks passed56
collector tests,12 GUI parity tests and6 geometry parity tests.

Before accepting either collector: prove native fixed_base=true, six joint DOFs,
zero physical root response under the bounded pan motion, unchanged descendant FK,
reset/root restoration, and camera phase. Run the ROS smoke again on the corrected
scene and then the in-process boot and matched-seed episodes. The lead will perform the live GUI check before merging to main.

## Collision FK follows measured physical roots

Independently of scene correction, collision FK now starts from each measured
root-link transform in the current physical snapshot. Target sweeps use that same
root and world-frame object poses; all descendant-link comparisons retain the
existing maximum-entry tolerance 0.0005. Invalid/nonrigid roots and inconsistent
child FK remain terminal. Root motion relative to the export is reported, rather
than hidden by a larger tolerance. This improves geometric correspondence but
does not establish fixed-base physics fidelity or permit floating-base episodes.


The standalone probe on the corrected export completed32 tick proofs with native
fixed base, measured motion, successful priming at both epochs and the explicit
paused second restore. It refused overall acceptance because moving wrist view
metadata lagged the retained physical pose by a further tick despite matching
ReferenceTime. That camera synchronization problem is independent of the fixed
root correction; no episodes or successful producer finalization are claimed.
Production and standalone owned groups were both released at 11:13:15Z, with
identity-bound remaining=[], closed ports and no owned NVML contexts/errors.
