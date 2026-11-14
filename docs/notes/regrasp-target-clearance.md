# Deliberate-miss target clearance

A recovery miss must leave the target untouched. MoveIt omits the target while
planning the clean corrective expert, because its intended grasp includes contact.
A separate native finger-box oracle checks the executed miss against that target.
Normal collection and GUI execution keep their established grasp/jaw behavior.

The held-stage reader serializes the six named joint frames and every enabled
native cube on both fingers. It normalizes quaternions, uses body-local collider
transforms and refuses unknown geometry. Native `RigidPrim` shape tensors supply
all four effective contact/rest offsets on each finger. The selected block's
native offsets are added to this model; both sets participate in its hash. Each
guarded episode starts with a fresh native finger/block offset read under the
epoch/held barrier; exact float32 identity must match startup. Changed native
values refuse before target removal, cap installation or motion.
Missing/unresolved offsets, nonzero rest offsets, or a skin sum exceeding the
clearance policy refuse recovery. The 1 mm executed margin exceeds the measured
0.1 mm + 0.1 mm contact skin plus 20 um geometric allowance. The 0.2 mm clean-open
margin includes the separate 20 um geometric allowance. No force is inferred.
The fixed-base articulation audit is a prerequisite.

Search tries the seeded 2–4 cm offset first, then bounded direction/magnitude
alternatives. Requested and +/-180-degree yaws are checked in order, with each
candidate's projected width, jaw angles and TCP compensation recomputed together.
Every physical stage, including initial open, close and reopen, needs MoveIt
endpoint/path preflight and a nominal target-clearance path check. The planned
physical path is buffered by the next-tick reach radius at that stage. Clean
open-jaw labels are checked separately; closed corrective labels intentionally
aim at the block and never become the miss's executed trajectory.

The geometric certificate does **not** assume common interpolation progress or
position convergence implies rest. Before each command, it constructs the joint
box `q_measured +/- (v_limit * (1/30 s + 1 us) + 5e-6 rad)` independently in all
six coordinates. This includes arbitrary drive progress, opposite-direction
motion and overshoot within the next tick under the verified native speed limit.
The 5 urad pad is the declared experimental allowance (four local 1 urad
budgets plus rounding), not an established bound on coupled solver residuals; measured speed must already be within the new limit plus
1 urad/s before it can be installed. A high inward velocity cannot be relabelled
as a low cap. Command endpoints are independently checked, too.

For each joint-box cell, the SAT separating-axis gap at its centre is reduced by
`sum(joint_point_radius * half_joint_interval)`. The radius bounds are checked
against the actual serialized chain. This is a bound on every finger point, so
a fixed separating axis certifies every combination of positions in the cell.
Uncertain cells split on their largest weighted coordinate. A cell unresolved
at 0.25 mm motion resolution refuses; a count limit is an explicit capacity
failure. The nominal planner path is only preflight. The per-tick reachable box,
not a nominal common-fraction segment, authorizes physical stepping.

The recovery-only native caps are 0.5 rad/s for initial opening, 0.1 rad/s for
high approach, and 0.05 rad/s for descent, close, lift and reopen. They cover the
entire deliberate-miss prefix because initial opening/approach and residual
motion between operations can also contact the target. The higher initial caps
avoid applying the near-block rate to all of that prefix. Arm operation duration
is `max(original duration, 2 * maximum joint displacement / stage cap)`;
gripper duration uses the same rule with a 3 s original duration. This asks the
planner for at most half the physical cap to leave tracking headroom. Receipts
record each original/new duration and their ratio; these are simulated planned
costs, not a measured wall-throughput claim. Retained seed 340000 gives a CPU planned-prefix estimate of 17.50 ->33.38 simulated
seconds (1.91x), with approach 2.83 ->10.32, descent 2.83 ->6.09 and lift
2.83 ->7.96 s; this uses fixture offsets and fake target-free MoveIt answers,
not live throughput. No cap applies to the corrective
shared Quickstart grasp/drop or the rest of the episode.

The target must be measurably stationary before every guarded advance and at
every substep check: linear speed <=0.1 mm/s and angular speed <=0.005 rad/s.
The latter admits the retained resting fixture's 0.0015 rad/s angular residual.
At a 20 mm corner radius these speeds contribute about 6.7 um over one frame
if maintained; that estimate does not bound unmeasured acceleration. The
reviewed moving-target hold (0.456 m/s) refuses at the actual command entrance
and native kickoff. Target pose/corner stability is still checked from the
pre-approach anchor. This is a verified sampled precondition, not a force model
or a claim that the target is frozen between samples.

Caps use the installed Isaac Articulation `set_max_joint_velocities` and
`get_joint_max_velocities`, with name-based DOF indices. These wrap the native
PhysX DOF maximum-velocity tensors (installed source: `isaacsim.core.prims/impl/
articulation.py`, methods around lines 2163–2240 in 6.1).
[PhysX joint schema](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/schemas/physxschema.html)
documents the articulation joint-space impulses enforcing `maxJointVelocity`.
Setter/getter agreement alone does not establish a preventive bound for coupled
PhysX/TGS solving. The **live re-grasp smoke is the required experiment**. Each
pre-step callback observes the preceding integration; a final snapshot closes
the frame. For every named joint, samples retain measured position/velocity,
native limit, local excess above `cap * physics_dt`, excess above the full-frame
cap range, and excess outside its **5 urad pad**. Sources distinguish
`before_advance`, `pre_physics_callback` and `final_snapshot`. A local residual
>1 urad, frame displacement outside the 5 urad pad, speed violation or failed
target-stationarity check is terminal. A detected failure rejects the episode;
it cannot undo contact already produced by an integration. No native solver
residual is claimed from CPU tests. The smoke must report measured maxima per
joint against the pad and stop/redesign if a bound is exceeded.

The cap is installed/read back while held before targets and advancing; its
state must match the command ACK. Starting an operation also has a checked
kickoff: the unchanged prior drive target is certified from the held snapshot,
then the native runtime validates epoch/stamp/state/drives and installs the cap
before the first advance. All six kickoff proofs/ACKs are retained. Arm/gripper
completion in miss stages additionally requires all measured speeds <=0.0125
rad/s; the ordinary collector/GUI completion rules remain unchanged. Native limit installation saves the originals and marks the state dirty before
the first write. Partial writes/readback errors restore and verify those saved
originals; originals are never replaced while dirty. Kickoff, command, cancel
and reset failures use this retirement path. Failed rollback remains terminal
and reports `on=None, status=dirty_unverified`, never off. Only a verified
restoration clears dirty state; a runtime fault still requires restart and
cannot resume or reset physical poses merely because cleanup later succeeds.
Cancellation restores the original native limits, and successful reopen restores/read-verifies
them before normal corrective planning. There are no gain, torque, collider or
asset changes.

Each row records `measured_miss_velocity_cap` (on/off, values and substep checks)
and `command_miss_velocity_cap` (the cap acknowledged for the next step). Normal
rows explicitly report off. Full recovery metadata includes policy, model/hash,
phase, original/new operation durations, kickoff boundaries and cap restoration.
The training merge can audit which deliberate-miss steps ran under a cap.
Rows include all substep samples. Receipts summarize signed per-joint maximum
local/frame/certificate excess in `miss_solver_residuals`. A native fault is
atomically published before pause/rollback in this instance's unique
`native-motion-faults/<runtime-id>.json`; its path is returned at installation.
The collector copies matching-epoch evidence into the attempt receipt, even
when native cancellation itself fails. The first error remains primary and
rollback diagnostics are secondary. No marker or runtime error is cleared to
permit another attempt.
The dataset itself also carries int64 `collection_phase` (`[1]`): 0 normal,
1 miss_prefix, 2 corrective; `meta/collection_phases.json` records its mapping and
counts. Export checks flags against the native reopen boundary and reads all
stored values back. Train on normal/corrective by default, masking/excluding the
slowed miss prefix. Miss-prefix actions still mean clean expert corrections
toward the true target, not the intentionally offset executed commands. The
phase feature is selection metadata, not a deployed policy observation.

Target movement is anchored before opening/approach. All eight target corners
must remain within 0.2 mm of that anchor throughout the miss. The empty-lift
check examines every close/lift tick, including transient lifts returning to the
original pose. Reopen remains guarded. Epoch, episode, model/policy hashes,
physical poses and whole-miss row hashes bind evidence to the recorded episode;
export recomputes command and kickoff proofs. Hashes exclude only separately
audited post-capture wall profiles. Copied identities/poses remain exact;
independently computed geometry uses the shared 1e-9 arithmetic comparison.

Selected-plan jaw authority and collision-rejected equivalent-yaw search are
restricted to the deliberate recovery path that actually commands those angles.
Shared `_cmd_grasp_move` uses its original actual-jaw validity route. Shared
object jaw commands retain the min/max bbox-width rule, including swapped bbox
dimensions and operation without an object pose. Default pure geometry outputs
remain unchanged. Entry-point tests cover both GUI validity predicates with an
actual jaw different from the candidate opening, both bbox orders/cross modes,
and pin the verified shared methods' source text.

Retained regressions include the 350007/340000 target overlaps, the reviewed
mixed independent-joint reopen posture (nominal lines pass, intermediate overlap
0.271 mm), a 2.28 rad/s inward hold, full close/reopen sweeps, cap application and
restoration, substep violations, and changed/missing effective offsets.

Offline checks (no Isaac app or GPU):

```bash
env CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none ROS_DOMAIN_ID=96 \
  ROS_LOCALHOST_ONLY=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python3 -B -m unittest discover -s linux-env/scripts/collect -p 'test_*.py'
```

The anonymous-session USD reader probe tests authored geometry only. It cannot
supply effective PhysX offsets or demonstrate the speed constraint. A separately
authorized live recovery smoke must freshly read both, verify the physical
no-contact corridor and cap restoration, inspect miss/recovery videos, validate
the dataset and prove owned cleanup before recovery episodes qualify.
