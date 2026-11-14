# Next in-process performance experiment

The accepted baseline is five normal episodes (seeds 330000–330004), 3,300 frames / 110 sim seconds, at code `5a3a719`. Collection and owned cleanup took 454.210 s: **0.242×** accepted-output RTF. All episodes passed the unchanged exporter and full numeric readback. Re-grasp is unqualified. No flags below were changed or benchmarked in this window.

The run used RTX 6000 Ada, CPU PhysX, two 640×480 cameras at 30 Hz sim time, and a base-Python Kit through SimulationApp. RoboSort's measurements use another scene, a full Kit experience and usually no camera at 10 Hz app updates. Its RTF values are evidence for experiments, not SO-ARM101 predictions. Keep our 30 Hz captures and four 120 Hz physics steps per tick.

## Measured cost

| Stage | Mean ms | p95 ms | Scope |
| --- | ---: | ---: | --- |
| App update | 33.497 | 33.452 | Inside wrapper |
| PhysX → Fabric publication | 0.228 | 0.312 | Inside wrapper |
| Both RGB host reads | 4.040 | 4.644 | Inside wrapper |
| Physics/render/read wrapper | 38.002 | 38.659 | Main thread |
| Snapshot copy | 0.654 | 0.742 | Capture.prepare copy; not all native reads |
| Clean candidate | 0.486 | 0.427 | Main thread |
| Primitive swept collision checks | 7.672 | 11.184 | Main thread |
| Target write / ACK | 0.187 | 0.221 | Main thread |
| Join / hashes / contract | 4.389 | 5.011 | Main thread |
| Enqueue | 0.448 | 0.511 | 3,300 recorded rows |
| Writer queue residence | 1.263 | 1.675 | Overlaps main thread; includes producer deepcopy/enqueue |
| Writer encoder/JSON writes | 5.666 | 6.952 | Overlaps main thread |

Foreground means cover 3,405 ticks in the five accepted attempts, including unrecorded home/containment ticks. Do not add nested wrapper stages or asynchronous writer times again. Non-overlapping timed foreground stages alone total about 51.82 ms per tick. Arm/DART preflight adds 18.907 s over the five episodes; final writer drains total 0.075 s. Some pose acquisition, serialization, reset and host work remains unattributed but is included in the 454.210 s denominator.

Recording-only episode RTF ranged 0.497–0.541×. Native startup to run-config publication took 230.922 s on the original RTX filesystem (includes shader compilation and warmup). The remainder of the worker after that publication gives 0.493×; it is a derived slice, not a warm relaunch benchmark. Sequential collection plus the 165.887 s CPU export/readback takes 620.098 s, or 0.177×, excluding transfer and manual preparation. Export ran on dual, collection on RTX. GPU/CPU tenant samples are retained in `rtx-normal-window/contention-context.jsonl`; contention was not controlled continuously.

The app update's median 33.431 ms is close to the 30 Hz wall budget. Pacing is a hypothesis, not a measured GPU render cost: the baseline did not retain these pacing settings' runtime readback.

## Flag audit against RoboSort

Read-only sources: `dual-a4500:~/Documents/RoboSort-twin/sim/PERFORMANCE.md`, `sim/launch/launch`, and its referenced `sim/launch/kit_env.sh`. Their SHA256 values at inspection are listed below. Our source is `collect_inproc/{shared,native,playing}.py` at `5a3a719`; our Kit log and `run.json.native_pipeline_settings` provide the readbacks noted here.

| RoboSort lever | Native baseline | Proposed treatment |
| --- | --- | --- |
| `--no-window` | Already present in actual Kit command, supplied by `SimulationApp(headless=True)`; hide UI and viewport updates disabled | Keep |
| Disable livestream app/WebRTC/native extensions | Not explicitly disabled; no livestream startup entry found in retained Kit log | Audit loaded extensions, then disable explicitly in a paired arm; no predicted saving |
| `/app/runLoopsGlobal/syncToPresent=false` and `/app/runLoops/main/syncToPresent=false` | Not authored or retained as readback | First pacing A/B |
| `/exts/omni.kit.renderer.core/present/enabled=false` | Not authored or retained as readback | Same presentation family, then isolate if useful |
| `/app/runLoops/main/rateLimitUseBusyLoop=false` | Not authored/read back | Separate busy-wait A/B; measure CPU as well as RTF |
| `/app/runLoops/main/rateLimitEnabled=false` | Authored in Native before `World.reset`; final/reset values unrecorded | Read back after initialization and every restore; do not assume it stayed false |
| `/app/player/useFixedTimeStepping=true` | Already authored; native tick admission verifies exactly four physics callbacks per advance | Keep; never trade away native clock/cadence proof |
| `/persistent/physics/numThreads=1` | CLI specifies **4**; final readback unrecorded | Paired 4 versus 1 worker test |
| `carb.tasking.threadCount=6` | Already explicit in actual command | Keep |
| SRTX disabled, core throttling async disabled | Already explicit; SRTX read back false, legacy annotators and synchronous products verified | Keep |
| CPU PhysX + MBP | Already selected | Keep |
| Fabric on, USD transform/velocity writeback off | Already enabled and read back; explicit pre-render Fabric publication | Keep the complete publication path and pose proof |
| Renderer focus/minimize sleeps disabled | RoboSort also sets skipWhileMinimized=false and both sleepMs values to0; native does not author/read them | Optional follow-up after presentation/busy-loop tests |

Do not copy RoboSort's GPU index, domains, paths, 10 Hz app cadence, solver iterations or collision changes. Its Fabric camera trials were invalid because moving physical state produced nearly unchanged pixels. Our accepted run uses a different explicit publication path, authoritative SDG CameraParams, per-row pose checks, native valid-handle reads and advancing RGB/SDG counters. Its largest camera matrix error was 4.14e-7, below the existing 2e-4 threshold; the same-seed sheets show moving robot/object images. Preserve all of these checks. The paused invisible viewport is needed to create Hydra; disabling presentation must not remove the engine.

## Proposed paired window

A new lead grant is required. Use one owned instance, the same qualified environment and scene, GPU UUID/renderer pin, CPU physics, nice10, persistent caches, and approved domain/ports. Prefer matched quiet RTX periods; otherwise retain all compute tenants, their CPU usage, GPU memory/utilization and host load in every arm and report the limitation.

1. **A0, instrumentation only.** Log effective pacing/presentation/focus settings and loaded livestream extensions after app creation, World.reset, camera setup/warmup and each restore. Record the loop runner's manual-step capability. The installed RenderingManager.set_dt source has a rate-limit fallback when a manual runner is unavailable, so verify the final state rather than inferring it from launch arguments. Keep dt fixed at 1/30 and physical callbacks at 4.
2. **A/B/A presentation.** A is unchanged source 5a3a719 plus instrumentation. B sets both syncToPresent keys and renderer-core present/enabled false. Keep the paused viewport/Hydra engine. First run 32 moving SDG-gated ticks and paused restore; stop if counts, native stamps, image pose/freshness or cleanup fails. If it passes, run the same five normal seeds in each arm. If the family helps, isolate its three settings before making a production recommendation.
3. **Busy loop, then livestream.** Against the passing best presentation arm, test rateLimitUseBusyLoop=false alone, then explicit disable of the three livestream extensions alone. Confirm whether each extension was loaded at all. Report CPU reduction even if RTF does not change. Check final rateLimitEnabled after every reset; an explicit post-reset unthrottled arm is a separate experiment if readback shows re-enabling.
4. **CPU worker count.** Compare4 versus1 with every other setting fixed. Keep physics120 Hz, rendering30 Hz, solver settings, contacts, colliders, drives, seeds and labels fixed. Thread changes can alter physical trajectories; all five physical gates and the unchanged export/readback must still pass.

Alternate arms in a warm-cache A/B/A order; retain cold startup separately. For each arm report full accepted-output sim/wall including preflight, resets, failures, writing and cleanup, plus recording-only timing and separately timed export. Use the same video/row and SDG gates, no stationary-row filtering or quality-threshold changes. The first short window can cover A0 and a32-tick A/B boot; the full five-seed pairs need an additional bounded allocation. These are proposals, with no claimed speedup.

After pacing is resolved, profile duplicate swept-query work, RGB copies and hash/proof cost without weakening safety. Two-pass Isaac replay and browser rendering still need SO-ARM101 live/replay pose/pixel and whole-pipeline measurements. Faster physics-only stepping is insufficient; browser training also needs the deployment-domain strategy in the design.

Source SHA256 at read:

- PERFORMANCE.md: `d3799ad61a60701c32a22c51a7d64ed520b4c8c8e7a95bfa35c0f4b27735c88e`
- launch: `0bf835ec8cbe84bd0b29acf5995d512c27a81383bd7b13ec5386e1fa3d880d10`
- kit_env.sh: `d4dcf461099ea752633a3d8095a65a2ffee447d04b5fd28a595b59257d6c8180`
