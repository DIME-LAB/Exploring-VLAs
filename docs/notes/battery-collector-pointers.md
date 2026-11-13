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
