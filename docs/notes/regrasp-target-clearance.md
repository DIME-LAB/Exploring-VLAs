The recovery strategy is a **synthetic post-miss reset**, followed by the shared
Quickstart correction at ordinary dynamics and speed. Setup is unrecorded and
all recorded rows carry `collection_phase=corrective` (dataset code 2). No
failed grasp motion is claimed. This is conditional state augmentation for the
established single-observation policy; history-conditioned models need a
separate evaluation of that approximation.

The seeded policy requests 50% closed jaws, 30% partly closed jaws and 20% open
hover starts. Lateral TCP offset from the true grasp is 26–38 mm, yaw jitter is
at most 12 degrees, and hover ranges are 3–15 mm for closed/partly-closed or
18–35 mm for open starts. Up to 24 deterministic direction/magnitude candidates
retain the requested phenotype. Rejections and MoveIt contact pairs are logged;
accepted distributions are measured, rather than assumed to match the weights.
The separately sampled hover contributes to the 3-D distance, which can exceed
38 mm.

Each candidate uses the shared calibrated 5-DOF IK and jaw-width convention.
All six joint limits, full-scene MoveIt validity with the target still present,
and independent enabled native robot-box geometry must pass **before** native
restore. The box model includes all collision links and is hashed with the
fresh effective finger/block offsets. Each robot box is checked against all
block boxes and all thirteen primitive boxes of every cup, in their physical
frames. Required separation is 1 mm plus 20 um arithmetic allowance. Finger
boxes also require that floor clearance; MoveIt keeps the full robot ground and
self-collision policy. This is a static configuration check, with no moving
miss sweep or velocity-derived reachable-set claim.

The held native restore already supports arbitrary six-DOF states by named
joint mapping, installs matching drive targets and zero actual/target velocity,
and reads back the result immediately. The collector rebases epoch and command
history and installs the restored target as its idle hold. It then advances six
unrecorded normal-dynamics ticks. Every tick must pass the same clock/state/image/
command contract and fresh geometry/scene checks. A native protected-scene guard
anchors every cup and block (target and neighbours) to the pre-restore world:
linear speed <=0.7 mm/s, angular speed <=0.012 rad/s and maximum corner motion
<=0.2 mm. Block corners use measured body-local geometry; cup corners use all
thirteen authored primitive boxes. Both fixed base frames must remain within
the shared 1 um float32-readback rule (quaternions compared up to sign).
The velocity screens are empirical contact/impulse diagnostics, not solver
displacement bounds. The checked-in
[rest-velocity-floor.json](../../linux-env/scripts/collect/fixtures/rest-velocity-floor.json) records
59 real episodes and pose-only rest classification; clean pre-contact maxima
were0.34691 mm/s and0.0056004 rad/s. Limits give about2x margin. Displacement
remains0.2 mm. Receipts retain per-body120Hz substep peaks; resting peaks above
half a velocity limit require reporting before production.

The guard reads native poses and velocities in every 120 Hz pre-physics callback
and again after the final solve of every 30 Hz frame. Four callback samples plus
the last solved boundary cover each solve-to-solve excursion; a transient motion
cannot disappear in the end-of-frame pose. Raw World clocks, callback order/dt,
body/base state and held drive targets are retained; no substep timestamps are
fabricated. The installed Isaac 6.1 SimulationContext registers user callbacks
with `pre_step=True` (simulation_context.py:1053), and advances its World clock
in the post-step timer (:1464, :1494). Thus pre-step World deltas are 0, 1/120,
2/120, 3/120 and the final boundary is 4/120, checked with the shared 1 us clock
rule. Native frame progression keeps the 10 ns native-clock rule. Physics gaps,
missing/extra callbacks, moved bodies, excessive velocities or changed bases
are terminal and retain the complete native guard evidence in the durable fault
journal. Independent receipt reconstruction verifies all 24 callback samples,
six final samples, original anchor hash and final certificate **before** the
writer opens. These are sampled no-disturbance bounds; no claim is made about
unobservable sub-substep contact that produces no above-tolerance motion.
The graph clock need not tick at120Hz: each raw native substep stamp must stay
nondecreasing inside its held frame's [start,start+1/30s] interval under the
10ns native rule. The independently measured World clock supplies substep
cadence. The first corrective row's fixed robot base is bound to the certified
last settle boundary. Replay binds the current robot model to the saved model
(geometry and kinematics, using the shared arithmetic readback tolerance) before
any synthetic write.

The writer opens only after successful settle and scene synchronization.
Reset or settle verification failures publish the original native error durably
and terminate the run; a failure before the writer produces zero training rows.
Exhausting statically valid candidates is a planner quality refusal.

Receipt validation reconstructs the seeded candidates and IK, static proofs,
native reset/readback and checked settle snapshots. It binds every restored body
and both bases to the saved anchor, restore epoch/clock/drive baseline to the
first settle proof, every later substep/frame, and the recording boundary to the
first corrective tick. Replays use the saved synthetic world anchor (all body
poses and geometry) and selected joints, rather than adopting the current scene.
An observed neighbour shift is deliberately reset to the saved pose; a different
fixed world/robot base or body geometry is refused before any native write. No
fixed base is teleported. Zero velocity and matching drives are read back after
the reset. Export rejects
setup rows and requires synthetic certificates for the current strategy. Normal
and GUI code remain on the shared expert path; normal batch qualification is
specific to its frozen source cut.
Relative to `bf7336d`, NORMAL exports also add `collection_phase=0` and runtime
pause behaviour uses the committed-pause barrier. An unchanged shared motion
module does not qualify these runtime/schema changes: NORMAL batches on the
current cut require their own startup/clock and end-to-end qualification.

The CPU integration probe constructs the actual Collector and PickPlaceNode,
uses real ROS/image transport with fake physics/MoveIt, executes synthetic reset
and settle before the real shared corrective Quickstart, and checks successful
finalization. Separate partial-reset, moving-target and late-native-fault cases
prove rejection, durable evidence, no setup data and completed cleanup. This
qualifies the pipeline, not physical success; an owned three-episode C smoke is
required before production recovery collection.

A gravity-disabled position-set prefix is a documented fallback only. Installed
Isaac 6.1 Articulation setters teleport joint state and targets; they leave the
articulation dynamically simulated. Its native per-link gravity getters/setters
and PhysX pre/post-step callbacks permit an experiment with readback before any
corrective setter. Such an experiment would need its own explicit measured
post-solve deviation term and held, read-verified gravity restoration. True
kinematic articulation links are unsupported by [PhysX](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Articulations.html)
and [Omniverse requirement JT.ART.003](https://docs.omniverse.nvidia.com/kit/docs/asset-requirements/1.11.2/capabilities/physics_bodies/physics_joints/requirements/articulation-not-on-kinematic-body.html).
Velocity-cap solving was rejected by the retained v4 diagnostic: +138.789 urad
local position excess at a 0.1 rad/s cap despite reported velocity below cap.
Synthetic setup makes no solver-residual claim and does not alter native limits,
gravity, solver settings, articulation topology or physical collision shapes.
