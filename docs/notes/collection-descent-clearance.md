# Grasp descent and measured support contact

A grasp yaw and its TCP compensation are one plan. `make_reachable_grasp_plan`
tries the requested narrow-face axis and its +/-pi equivalents. For each yaw it
recomputes projected block width, TCP jaw offset, opening margin and all IK
poses. It refuses a diagonal or long-face substitute. The chosen plan is carried
as immutable JSON on the operation, so an obsolete prevalidator cannot replace a
new operation's target, event or geometry metadata. GUI and collection use the
same helper. Basic `make_grasp_plan` math retains its legacy parity fixtures;
actual execution uses the complete, yaw-consistent helper.

The retained seed 330001 command used a 22.5-degree fallback without changing the
original jaw offset/opening. Its moving finger intersects the initial block in
the authored primitive reconstruction. Seed 350010 has the same problem at 45
degrees. Seeds 350007/350009 instead have initially clear targets, then small
block displacement during descent. All four are retained in
`linux-env/scripts/collect/fixtures/descent-obstructions.json`, with authored joint
frames/boxes, initial and final measured body poses, source hash and scope.
These poses came from the floating-base scene: reconstruction is geometric
comparison evidence, not a live force or solver-link-deflection measurement.

An approach has at least 25 mm vertical clearance above its final TCP. Lateral
approach offsets are projected into the available open-jaw corridor: the fixed
jaw must approach from outside, and the moving jaw must retain its clearance.
Requested and effective offsets, requested/chosen yaw, projected width, opening
and margin remain in operation/episode metadata. The retained whole-descent
fixtures verify target clearance and at least 1.2 mm table clearance.

DART remains nonzero near the block. `ClearanceDart` bounds pad-point movement
using conservative pivot-to-pad radii and the available height/corridor. The
near-block budget is at most 0.2 mm and 20% of the declared fixed-jaw clearance;
the budget grows smoothly above the block. A minimum-jerk fade covers the last
3 mm of descent/contact, in addition to the seeded program's quiet tail. One
fixed noise-only rescale enforces sampled velocity/acceleration limits. Expert
planner timing and clean labels do not change. Receipts retain policy,
derivative rescale, and per-phase nonzero tick counts and amplitude bounds.
Large 2–4 cm near-block misses belong to the separately validated re-grasp set.

A measured starting state may have support contact within 20 um, using the same
rest/float bound as the cup gate (9 um observed + 10 um variability + 1 um
transform/float allowance; native contact offset 0.1 mm, rest zero). This exception
applies only to the first sample of the measured-state segment. Every commanded
target, previous-command segment and swept interior sample remains strict.
Ground contacts must lie on the ground's upper surface with a vertical normal.
Cup contacts must lie on its local floor inside all wall half-planes with a floor
normal. Unknown frames/bodies, side/rim contacts, absent contact evidence, mixed
unapproved contacts and deeper penetration are refused. The raw query contacts,
context and classification proof are recorded; the row audit rechecks that proof.
This does not change final whole-block containment or settled/lift requirements.

MoveIt Humble reports contact positions/normals in its planning frame and sets
that frame explicitly in the validity response; see the
[2.5.10 state-validation service](https://raw.githubusercontent.com/moveit/moveit2/2.5.10/moveit_ros/move_group/src/default_capabilities/state_validation_service_capability.cpp).
The collector requires the declared `base` frame rather than guessing one.

Collection requires native `fixed_base=true`. Installation calls the shared
`collect_inproc/articulation_audit.readback(stage, articulation)` and retains its
source hash, native flag, counts, root placement and name-to-index mapping. Each
held snapshot also reads `base_link` through a read-only PhysX rigid-body view.
Episode receipts report drift from the initial native base pose. `/World/base`
remains the ROS reference frame; it is not relabelled as a measured robot link.
No audit/measurement function relocates a root, changes drives or edits the scene.
The fixed-base asset repair is owned by the in-process scene work.

Recovery preflight searches up to sixteen seeded direction/magnitude candidates.
It checks all clean/executed approach, descent, close, lift and reopen segments
against MoveIt before choosing the first passing candidate. Refused candidates
keep their contact pairs. Query transport failure or cancellation never means
"try another physical candidate". Planning has a separate bounded wall allowance;
it does not add synthetic simulation time or relax convergence limits. Replay
and export both reject episode recovery kinds that contradict the run mode.
