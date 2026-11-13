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

CPU regressions exercise the actual old-prevalidator/current-approach
interleaving, selected descent target and event identity, stale origin
consumption, request immutability, all 51 validity calls concurrent with capture,
and capture-requested emergency abort. The earlier gripper cancellation,
physical-state, geometry, timing, reset and confinement tests remain in the
suite. These checks do not replace corrected physical smoke/pilot validation.
