# SO-ARM101 articulation base: diagnostic proposal

Status: proposal only; no extension or shared collector changes. The native PhysX
fixed-base flag has not yet been read from either the owned standalone or the next
live production scene. Do not accept motion episodes until those reads and the
resulting scene decision are resolved.

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

The lead has asked the production owner to include that result in its smoke
receipt. A separately granted <=5-minute owned standalone boot will capture the
other flag; it stops before motion if floating. Neither measurement is claimed
by the offline tests.

## Proposed scene correction if floating is confirmed

Have the scene owner relocate ArticulationRootAPI from base_link to the enabled
world fixed joint root_joint (or an appropriate ancestor encompassing that joint),
preserving the six drives, geometry and robot path. Validate wrapper/view discovery
at `/World/SO_ARM101` after reparsing. Do not merely stiffen the external joint or
increase FK tolerances. This affects production ROS collection too, so update the
asset through its owner, regenerate the held export/manifest, and collect fresh
fixtures against the corrected physics. Existing scene/asset digests must refuse
mixing old fixtures and new geometry.

Before accepting either collector: prove native fixed_base=true, six joint DOFs,
zero physical root response under the bounded pan motion, unchanged descendant FK,
reset/root restoration, and camera phase. Run the ROS smoke again on the corrected
scene and then the in-process boot and matched-seed episodes. The proposed API
relocation is not applied by this worker.

## Collision FK follows measured physical roots

Independently of scene correction, collision FK now starts from each measured
root-link transform in the current physical snapshot. Target sweeps use that same
root and world-frame object poses; all descendant-link comparisons retain the
existing maximum-entry tolerance 0.0005. Invalid/nonrigid roots and inconsistent
child FK remain terminal. Root motion relative to the export is reported, rather
than hidden by a larger tolerance. This improves geometric correspondence but
does not establish fixed-base physics fidelity or permit floating-base episodes.
