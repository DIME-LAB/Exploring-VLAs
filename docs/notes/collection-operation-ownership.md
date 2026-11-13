# Motion request ownership during sim collection

Planning and ROS validity checks run on wall time with the simulation held. They
must not own the generation lock. The lock covers short UI/state commits,
command dispatch/ACK/recording, and scene mutation transactions. A planning
result is accepted only after checking its original epoch again.

`MotionOperation` is pure Python. Its frozen fields contain a copied tuple of
joint targets, the original completion-event identity and epoch, duration,
speed, motion tag and planner settings. Retargeting returns another request;
it never publishes a target through an instance field. The event verdict is
mutable because physical execution or cancellation completes it. Every
consumer checks that event and request origins agree with its worker.

The shared grasp prevalidator passes an approach request and a descent request
directly to `_ik_apply_and_act`. Its wait closure captures those requests; it
cannot read another prevalidator's final target or event. There are no
`_grasp_final_joints`, `_grasp_motion_event`, `_grasp_arm_duration`, or
`_last_ompl_error` handoff fields in the implementation. OMPL returns its error
alongside the trajectory. Home speed overrides belong to the request instead
of temporarily changing a setting and restoring it after a planning wait.

The source sweep covers writes by nested workers and their callers, including
attribute, mapping and setting updates:

| State | Writer / reader | Ownership |
| --- | --- | --- |
| Approach/descent target, event, duration, speed, tag, planner settings | Prevalidator, continuation, deterministic/OMPL planners, Servo | Frozen request passed explicitly; epoch validated at consumption and execution. No shared continuation slots. Trajectories are newly allocated per plan and never mutated after dispatch. |
| Current event | Command setup, status/wait helpers, child threads | `_begin_motion` publishes under the guard and stores a thread-local event. Children inherit that identity before thread start. Status/verdict helpers use the thread's event or an explicit request event. `_motion_event` remains a UI/service inventory and terminal-abort wake-up slot; it is not a target or continuation input in collection. |
| `joint_positions`, `_ik_planned_target`, `_ik_valid`, slider/group state | UI callback and deterministic planner setup | Short guarded commits from copied request targets; all validity/planning calls occur after releasing ownership. The collector reads complete measured state, so UI fallback positions cannot supply a physical start state. Manual GUI-only controls keep their ordinary UI semantics. |
| Attached object name/offset, collision names, ACM, detach flag | Attach/detach and scene callbacks | Guarded scene transactions through service acknowledgement. Stale workers cannot mutate bookkeeping or scene. |
| `_last_grasped_object`, selected object/cup, motion tag | Parent command setup / later parent steps | Short guarded request bookkeeping or the single cycle parent. The collector joins that parent before another attempt; a parent that does not retire aborts the run. Worker target/event/tag inputs are captured rather than reread from these fields. |
| Cycle trace and home trace bookkeeping | Prevalidator / home trace workers | Trace mutations use the originating worker's guard. A retired planner cannot close or add to a later cycle's trace. |
| Display motion status | Shared status helper / UI and diagnostics | Guarded; decisions in collection use per-event verdicts. GUI action callbacks without an adapter retain the status fallback. Wrong-origin status events refuse before any status write. |
| FJT handles / diagnostic trajectory path | GUI action callbacks | The collector returns through its Servo adapter before this fallback. Remaining full-trajectory callbacks and their handle/path writes also retain the request guard; diagnostic tags come from the request. |
| Physical positions, velocities, object/cup poses | Joined capture feedback / planner snapshot | Capture owns the feedback commit and validates the native observation epoch. Pose subscribers are bypassed for collection; direct measured feedback owns caches. Bounding-box catalog callbacks are sensor input, not a worker-to-worker handoff. |
| Servo plan, reference, RNG, parameters, phase | Capture, dispatch, cancellation, main attempt setup | Serialized by generation ownership and the Servo lock. Main setup copies parameters. Replay state is committed while held under ownership. Plan labels use the request tag. |
| Cycle start | Main attempt thread / spawned cycle parent | Object name and epoch are bound before starting the thread; delayed thread start cannot acquire a later epoch. `motion_call` restores its prior thread context and refuses work after stop. |
| Cycle lifecycle flags | Single cycle parent / main controller | Main joins the parent before reset/another attempt. An unretired parent is fatal. Asynchronous child workers never supply later operation targets through these flags. |
| Collector writer, last joined snapshot, processed stamp | Capture / main attempt thread | Guarded capture and writer/tick conditions; cancellation waits for capture retirement before scene reset. |

`StepGeneration` invokes its configured stall handler when any ownership
acquisition times out. `abort_stalled_owner` is the common terminal path for capture, scene/status,
and explicit cancellation requesters. It sets failure/stop/QS abort, performs
setter-free native cancellation with a bounded RPC, and emits the same
structured diagnostic. An abort lock prevents duplicate emergency actions and
preserves the first failure, including emergency-RPC failure. Generation poison
prevents subsequent reset, advance or installation. A publication stall can
therefore abort; finite planning outside ownership cannot be mistaken for a
stalled owner merely because its serial validity checks take longer than the
retirement deadline.

Motion completion waits release ownership while polling operation-owned
simulation budgets. Each timing tree carries its immutable origin; timing
updates validate under the guard and cancellation is revalidated after every
wait and before status writes. Servo's cumulative dispatched-duration counters
are guarded and serve only episode watchdog sizing. Timeout diagnostics retain
their epoch; receipts select only their attempt's motion epochs. Attach requires
an already held physical snapshot rather than waiting under scene ownership.
See [motion wait clocks](collection-wait-clocks.md) for the complete sweep.

CPU regressions exercise the actual old-prevalidator/current-approach
interleaving, selected descent target and event identity, stale origin
consumption, request immutability, all 51 validity calls concurrent with capture,
and capture-requested emergency abort. The earlier gripper cancellation,
physical-state, geometry, timing, reset and confinement tests remain in the
suite. These checks do not replace corrected physical smoke/pilot validation.

## Scene request completion and reset authority

A local wait ending does not cancel an ApplyPlanningScene request already sent
to MoveIt. `SceneMutationClient` wraps the shared node's scene client for all
attach/detach, world ADD/REMOVE and ACM requests. It retains pending futures.
A result is conclusive only after the future completes with a response. A wait
expiry, missing response or transport exception sets the node's terminal scene
error and invokes the common collection abort. Typed terminal errors propagate
through ordinary scene fallback handlers. A subsequent mutation/reset refuses;
an already completed future may be deliberately drained before further work.
Completion arriving after a terminal wait does not clear the error or generation
poison. A false service response is conclusive, but local bookkeeping cannot
establish which partial effects occurred; reset still queries remote state.

`SceneSafetyMarker` atomically records unresolved requests in the instance's
confined runtime directory. A new collector refuses the same control process
(identity uses PID, process start ticks and boot ID). An unknown identity cannot
prove a restart. Unresolved Kit pose/reset RPCs are also terminal and require
both the instance's control and Isaac processes to have changed. Restart the
assigned instance with collect_sim.sh down/up; leave other instances alone.
The record remains for diagnosis and is never cleared by an ordinary retry.
If the recorded identity is unknown, another restart cannot prove replacement.
The error names the marker, original failure and affected identity; collection
stays stopped until the operator performs the manual scene reconciliation in
README's "Collecting sim data" section. Malformed/unreadable records also refuse
startup. Marker leaf reads use O_NOFOLLOW; temporary writes use O_NOFOLLOW and
O_EXCL, then flush/fsync and atomic rename. No marker-leaf target is followed.

At startup and every scene reset, `SceneReconciler` queries all planning-scene
components (mask 1023). It checks pending request completion, removes every
remote attached object, then queries again because detach can create a world
object. It removes all remote world objects and replaces ACM policy using the
checked-in SRDF's disabled robot-link pairs. Base/ground and object/ground rest
contacts are the only extra pair allowances. Stale default ACM allowances and
unknown names are removed. Local attached names and collision trackers are
cleared only after the remote postcondition passes. An unexpected occupancy
map refuses setup and requires a fresh control stack.

The scene is rebuilt with the shared ground/cup/block geometry methods. A full
query must then show exactly the expected world IDs, no attached objects,
matching object/cup poses and physical joints, the expected ground geometry,
the rebuilt primitive dimensions or mesh identity from the shared shape builders,
empty occupancy, and canonical off-diagonal ACM policy. Missing false-only names
and self-pairs have no effect on that policy; changed robot/world permissions or
defaults reject. The fake service models MoveIt 2.5.10's constructor/serializer
instead of copying the input matrix. Mesh identity comparison tolerates vertex
and triangle reordering/winding and numeric differences up to 1e-7 m, while
requiring the same vertices and triangle content. `first-reset-scene.json`
retains the actual first query's matrix, IDs, occupancy and verification results
for smoke evidence. Every accepted receipt contains cleanup and
verification results. This is independent of the node's local attached name or
collision-name lists. MoveIt's [Humble planning-scene implementation](https://moveit.picknik.ai/humble/api/html/moveit__core_2planning__scene_2src_2planning__scene_8cpp_source.html)
treats world and attached-object changes separately and replaces an explicitly
provided ACM; the reset follows those semantics and verifies its service result.

Pose subscribers check the physical-cache bypass flag under their respective
map lock. Collection enables the physical source before ROS spin; the handover
also holds both map locks when replacing caches. A callback already waiting on
a map lock cannot overwrite the physical replacement afterward.

Controller-reference rows include the original source header stamp, callback
receipt epoch/wall time, inferred source epoch and its inference basis, same-tick
flag and sim-time age. The raw topic does not carry an epoch: stamps at/before a
reset or ahead of the recorded tick are marked unknown. This diagnostic never
supplies physical state, clean action or success. Its callback reads an atomic
epoch/barrier tuple without waiting for generation ownership, preserving ACK
callback progress.

Static mesh-loader caches contain only fixed asset geometry, not episode poses
or targets. Raw GUI action callbacks are outside collection's adapter path;
only full-trajectory fallback callbacks retain operation guards. Bbox catalog
input is assumed to describe the fixed scene assets. Native singleton/source
hashes and command ledger stay on Kit's main thread. Encoder state is owned by
each attempt's writer and joined before acceptance or another writer is used.
