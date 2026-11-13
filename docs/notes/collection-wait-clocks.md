# Motion wait clocks

Collection uses measured simulation time for robot motion. Planning and ROS
transport consume wall time with physics held. `MotionWaiter` activates only
through the collector's injected adapter; the GUI keeps its Event/FJT wall
waits, including the 10 s release wait and duration+10 s blocking action-result
wait. No decision is based on the dataset's claimed fps or a synthetic clock.

Each completion event owns a timing budget with the same immutable epoch as
its operation. A forecast is registered before a worker starts; actual executed
trajectory duration increases the forecast when necessary. Approach timing is
a child of the final descent event, so Quickstart's composite wait includes
both stages. These objects contain timing only, never motion targets. Budget
registration validates origin under generation ownership. Waiting releases
ownership, revalidates after every poll and rechecks before status/scene effects.
A stale timeout cannot write a newer operation's status.

The simulated limit is planned duration plus 5 simulated seconds per operation.
The independent wall watchdog is `max(legacy minimum, sim limit / RTF floor +
30 wall seconds)`. The default floor is 0.1. Whole-cycle timing uses the sum of
actually dispatched trajectory and gripper durations, plus per-operation sim
margins; it grows as the script dispatches subsequent motions. The CLI's
`--episode-deadline` is a minimum wall watchdog. A legitimate slow episode can
exceed that minimum while its planned budget supports it. Resource starvation
below the configured RTF floor can still trigger the wall watchdog; the
structured diagnostic records the floor, planned/elapsed times, epoch and kind.
Policy is recorded and validated on resume.
Every incomplete sample is also compared with the previous sample; partial
backward jumps above the origin raise `clock_reversed` with both samples logged.
An already-completed valid operation retains readiness-first behavior.

| Wait / window in pick_place.py or collect.py | Collector clock / reason | GUI behavior |
| --- | --- | --- |
| `_qs_wait_for_step` motion completion | Operation tree sim budget; separate wall watchdog. | Original 60 s wall loop and 0.2 s pause/completion polling. |
| Approach completion before descent | Approach event's own sim budget; final event includes child timing. | Original 60 s Event wait. |
| Home/data-home and drop trace completion | Motion event sim budget. Timeout raises before a completion trace is emitted. | Original 30 s Event waits. |
| Release outer wait | Gripper event's planned sim duration and convergence margin. | Original 10 s Event wait. |
| `Servo.gripper` blocking inner wait | Its own gripper budget/verdict; origin validated after wait. | Collector adapter absent. Shared FJT helper retains GUI handling. |
| Blocking FJT trajectory/gripper result | Sim budget when collector adapter exists; goal acceptance remains wall. Quickstart normally uses Servo instead of this fallback. | Original duration+10 s result wait. |
| `clock-sanity` FJT result | Installs the same sim-time waiter. Duration verdict logs one clock tick plus the 50 ms wall polling quantum scaled by measured RTF (at least 1), added to the 0.4 sim-second controller/transport allowance. | Separate diagnostic node; GUI unchanged. |
| `_wait_arm_at_rest` at attach | Requires a proved held physical snapshot after Servo convergence. An unheld snapshot refuses immediately: attach owns the scene transaction, so waiting there would block capture. Collector settling uses the sim-tick checks below. | Original 1.5 s wall window / 0.15 s wall dwell. |
| `Servo.tick` arm/gripper convergence | Native sim stamps: actual planned duration + the same 5 s sim margin. | Collector-only. |
| Pre-episode home | Home event sim budget; legacy 90 s becomes a minimum wall watchdog. | Collector-only. |
| Cycle thread join | Growing executed-motion sim budget; configured minimum wall watchdog. Any wait exit retires/cancels the cycle before another attempt. | Collector-only. |
| Explicit verification/reset settling | Six/15 native sim ticks, not wall sleeps. Each tick has its own bounded transport wait. | Collector-only. |
| Goal acceptance, IK/FK/validity/OMPL futures | Wall: remote computation/service response, not robot execution. `_wait_future` polling remains wall. | Unchanged. |
| Planning-scene query/apply, attachment/world/ACM completion and service discovery | Wall: remote scene mutation completion; unresolved mutations retain terminal quarantine. | Unchanged. |
| Physical command ACK (10 s), joined frame read (30 s), one-frame handshake (20 s) | Wall: dead/stalled transport must fail even when /clock stops. No motion-duration cap. | Collector-only. |
| Controller graph readiness (60 s), subscriber discovery / scene read-back (15 s) | Wall: discovery and transport propagation, with no robot-duration assumption. | Collector-only. |
| Generation ownership (15 s), emergency RPC (30 s), cancelled parent retirement (5 s) | Wall: lifecycle/resource protection after motion cancellation. An unretired parent is terminal. | GUI adapter absent. |
| Encoder queue/drain/process shutdown and ROS executor shutdown | Wall: host I/O/process lifecycle. | Collector-only. |
| Interactive pause/resume | User event, not motion duration. Collector cycle watchdog still bounds an unattended pause. | Unchanged pause semantics. |
| 10/20/50/200 ms polling sleeps | Wall scheduling quanta; the enclosing deadline selects the clock above. | Unchanged scheduling. |

CPU tests execute the actual shared release method and Servo gripper path at
RTF 0.27 and 0.15, including a 15 sim-second gripper move that exceeds the old
60 wall-second inner wait. A stopped clock trips the logged wall watchdog; an
advancing nonconvergent clock trips the sim limit. Tests also cover composite
budgets, episodes longer than 180 wall seconds, GUI wall parity, cancellation
while waiting and cancellation between timeout logging and a status commit.
These tests do not replace the fresh headless smoke.
