The recovery strategy is a **synthetic post-miss reset**, followed by the shared
Quickstart correction at ordinary dynamics and speed. Setup is unrecorded and
all recorded rows carry `collection_phase=corrective` (dataset code 2). No
failed grasp motion is claimed. This is conditional state augmentation for the
established single-observation policy; history-conditioned models need a
separate evaluation of that approximation.

The seeded policy requests 50% closed jaws, 30% partly closed jaws and 20% open
hover starts. TCP displacement from the true grasp is 26–38 mm, yaw jitter is
at most 12 degrees, and hover ranges are 3–15 mm for closed/partly-closed or
18–35 mm for open starts. Up to 24 deterministic direction/magnitude candidates
retain the requested phenotype. Rejections and MoveIt contact pairs are logged;
accepted distributions are measured, rather than assumed to match the weights.

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
command contract, fresh geometry/scene checks and the stationary-target gate:
linear speed <=0.1 mm/s, angular speed <=0.005 rad/s and maximum corner motion
<=0.2 mm. The writer opens only after successful settle and scene synchronization.
Reset or settle verification failures publish the original native error durably
and terminate the run; a failure before the writer produces zero training rows.
Exhausting statically valid candidates is a planner quality refusal.

Receipt validation reconstructs the seeded candidates and IK, static proofs,
native reset/readback and checked settle snapshots. It binds episode/epoch,
recording boundary and first corrective tick to the setup contract. Replays use
the recorded selected pose and revalidate it without resampling. Export rejects
setup rows and requires synthetic certificates for the current strategy. Normal
and GUI code remain on the shared expert path; normal batch qualification is
specific to its frozen source cut.

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
