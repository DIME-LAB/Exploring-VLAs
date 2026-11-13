# Battery-style collector: pointers from plan47 (2026-11-13), all on dual-a4500 under ~/prompt2assembly (read only)
Note: `plan-anything/scripts/battery_simulator.py` is NOT a battery (it's a publish-time Isaac-version check).

(a) Standalone batch loop with deterministic stepping
- assembly-ingest/simcheck/launcher.py: headless SimulationApp, fresh world per check, bodies posed as USD ops BEFORE world.reset (a runtime set_world_pose doesn't move a kinematic body's collider); Stepper seam in simcheck/settle_node.py. It isolates HOME/XDG per process (cold shader cache); for long collection, share ~/.cache/ov instead.
- plan-anything/spikes/088-physics-battery/push_battery.py: the full shape (scripted episode, receipt, judgement, RTF); no articulation.
- plan-anything/research/cpu-regime-probes-20261020/overlay/square_table_arm_battery.py: closest for an arm (research code).
- No Lula/RMPflow anywhere in their code; use NVIDIA standalone examples for in-process IK.

(b) Cameras without a viewport
- sim-instrumentation/sim_instrumentation/lab/incell_camera.py: own camera prim + render product, move by authoring the transform; create once and REUSE (they cost GPU memory and leak if recreated).
- assembly-ingest/simcheck/lifecycle.py (render products in a standalone lifecycle); assembly-ingest/simcheck/capture.py (framed PNG capture).

(c) Dispatch and scale
- sim-instrumentation/sim_instrumentation/standalone_shell.py: cache isolation helper, interpreter resolver, cgroup slice (memory/pid, no cpuset), teardown by process group, receipt written before anything can refuse.
- plan-anything/research/standalone-dispatch-method-design-20260918.md (pa dispatch standalone).
- sim-instrumentation/sim_instrumentation/concurrency_battery.py: per-instance allocation (roots, ports, DDS island, cpu list).
- plan-anything/research/isaac-concurrency-limit-20260906.md (MEASURED, read the verdict):
  - dual, one GPU: N=1 RTF 6.26; N=2 each ~-10% RTF (~1.8x total), identical digests
  - rtx6000: N=3 2.43x total; N=4 failed (one process segfaulted at startup)
  - pinning or capping CPUs made N=2 worse (87% pinned vs 95% unpinned)
  - cap threads per process; don't pin
- Seeds: sim-instrumentation/sim_instrumentation/duct_thread_battery.py. One receipt per run, always written (knobs, Isaac stamp, code head, input digests, time series, terminal state).

Pitfalls (handbook plan-anything/handbook/isaacsim-operating-handbook.md)
1. A standalone script EXITS 0 on a module-level exception: require your own end marker plus the receipt (~l.1161).
2. Replicator orchestrator wedges: an unfinished capture holds stepping; a sync stop outside its async path parks it in STOPPING forever. Give each capture step its own deadline (l.154).
3. Capture GPU memory only grows (4.9 to 13.0 GB over a resolution sweep): fix the resolution, reuse render products (l.154).
4. Render products and session layers live as long as the stage: recreate after new_stage(). NEVER insert or remove a rigid body beside a live articulation: it invalidates the articulation view (l.1358).
5. World() resets the run-loop rate to 60: set it after, read it back (l.249).
6. Arm contact reporting before play (l.154).
7. No disableWindowOnLoad with cameras; captureOnPlay=1 with Replicator writers.

## In-process IK for the SO-ARM101 (plan47, 2026-11-13, measured on dual ~09:15Z)
- robo-manipulate (~/Documents/robo-manipulate, branch franka-embodiment @3016a531) is NOT in-process: a legacy
  verb-substrate MCP server driving Isaac over the bridge. Analytic solvers only (UR5e closed form, FR3 He-Liu);
  neither transfers to a 5-DOF arm. Treat every robo-manipulate* dir on dual as READ-ONLY
  (robo-manipulate-published-fr3-20261002 is pinned by published records). Worked example of adding an arm:
  ~/Documents/robo-manipulate/docs/franka-embodiment.md; survey: ~/prompt2assembly/plan-anything/codebase-survey/
  robo-manipulate-prismatic-20260814.md.
- prompt2assembly/DroidStack carry NO planner. Robots join as a ROS 2 contract plus a USD realization
  (~/platform/plugins/droidstack/skills/onboard-robot/SKILL.md, onboard-end-effector/SKILL.md). No in-process IK there.
- Options for 3.1f IK:
  (1) REUSE our own control_gui IK/grasp maths, once 3.1a extracts it into a GUI-free module, if it is plain Python
      (not MoveIt calls). Demonstrations then stay identical to the verified GUI path; only MoveIt execution and cup
      avoidance need an in-process replacement.
  (2) Isaac's Lula kinematics solver with a robot description generated from
      vla_SO-ARM101/src/so_arm101_description (NVIDIA's documented standalone route; handles the 5-DOF chain;
      plan47's recommendation).
  Keep the gripper semantics the same as the ROS controller so collected data matches deployment.
