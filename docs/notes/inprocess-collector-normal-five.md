# Five normal in-process episodes

At code `5a3a719`, all five fixed-base normal fixtures (seeds 330000–330004) were accepted on the first attempt. The export ran from the dual checkout at `cbf60a8`; `export.py`, `contract.py` and `validate.py` are byte-identical to the native code cut. The unchanged `linux-env/scripts/collect/export.py` produced a LeRobot v3 dataset, and `collect_inproc/validate.py` read back every numeric state, action, timestamp and episode index: **5 episodes, 3,300 frames, 30 fps, both 640×480 RGB cameras, three tasks**. No recovery episode was attempted or qualified.

| Seed | Color | Frames | Recording-only RTF | Stationary fraction | Physical/contract/export |
| --- | --- | ---: | ---: | ---: | --- |
| 330000 | red | 665 | 0.497× | 0.75% | pass |
| 330001 | green | 640 | 0.541× | 0.78% | pass |
| 330002 | blue | 670 | 0.527× | 1.20% | pass |
| 330003 | red | 645 | 0.537× | 0.93% | pass |
| 330004 | green | 680 | 0.534× | 0.88% | pass |

Exported episode order follows native attempt order, **not seed order**:

| Episode index | Native attempt | Seed | ROS reference attempt |
| ---: | --- | ---: | --- |
| 0 | attempt-0000 | 330000 | attempt-0000 |
| 1 | attempt-0001 | 330002 | attempt-0002 |
| 2 | attempt-0002 | 330004 | attempt-0004 |
| 3 | attempt-0003 | 330003 | attempt-0003 |
| 4 | attempt-0004 | 330001 | attempt-0001 |

The measured states and clean correction actions have six deployment-ordered float32 degree features. Both videos decode completely with no surplus frames; the exporter verifies their layout and reloads the dataset. Tasks, DART specifications, grasp specifications and physical/recovery metadata are retained. Existing acceptance thresholds were used throughout. The earlier three-attempt diagnostic was correctly rejected because a redundant already-open gripper operation produced stationary frames; `cbf60a8` completes a natively settled jaw goal without advancing physics, and preserves every executed recorded tick.

## Timing and limits

- Collection plus owner cleanup: **110 sim seconds / 454.210 wall seconds = 0.242×**. Five accepted episodes in five attempts; this includes startup, preflight, priming/reset, writing and cleanup.
- Startup to run-config publication:230.922 s, measured from original RTX filesystem metadata and the invocation certificate, retained in `startup-stat.json` and reproducible with `derive-profile.py`; includes cold RTX pipeline compilation and warmup. The post-publication remainder gives 0.493×, a derived slice rather than a warm relaunch measurement.
- Individual recording-only audits:0.497–0.541×. They exclude time before the first or after the last recorded row.
- CPU export and full readback on dual:165.887 s. Sequential collection+export:620.098 s /0.177×, excluding data transport/manual preparation. Different hosts were used for collection and export.
- No >1× result, controlled hardware comparison, or two-pass renderer speed/fidelity result is claimed. The original5+5 bar remains incomplete until five re-grasp episodes pass.

The foreground profile covers 3,405 ticks including unrecorded home/containment ticks. Mean physics/render/read wrapper 38.002 ms (app 33.497,Fabric 0.228,camera reads 4.040 nested); collision 7.672 ms, join/hash/contract 4.389 ms, snapshot copy 0.654 ms, clean candidate 0.486 ms, target/ACK 0.187 ms and enqueue 0.448 ms. These timed foreground stages total 51.82 ms/tick. The asynchronous writer averages5.666 ms write time and 1.263 ms queue residence, overlapping the main thread. Arm/DART preflight totals 18.907 s; final writer drains 0.075 s. Some native read/pose/serialization/reset work remains unattributed but stays in end-to-end wall. See `rtx-normal-window/profile-summary.json` for the derived summary and `reports/normal-five-v2/accepted-profile.json` for the unchanged original stage report, with sample counts, p95 and per-episode breakdown, and the performance proposal for the next experiments.

RTX 6000 Ada GPU0 was UUID-pinned (`GPU-c2772797-ad71-f07e-0101-db0f188f3504`), renderer0, CPU PhysX, nice10, domain 98 localhost-only, ports 8793/8794, isolated checkout/runtime and persistent caches. Other ROS compute processes ran concurrently for part of the batch; NVML, per-process CPU and uptime samples are retained. This was not a continuously controlled contention experiment. Source qualification uses the exact shared host-root/preload patches (`7b7026f`/`69e4797`, from w_rtx's `a37662f`/`a7d9e36`), with all ten already-loaded charset 3.3.2 paths and file hashes checked after Kit initialization. No shared venv files or acceptance checks were altered.

## Images and scene provenance

Five `compare-seed-*` directories contain ROS-reference|native side-by-side sheets at common frame indexes, seed/target identity checks, frame counts and maximum joint-state differences. The ROS inputs are read-only `smoke-descent-clearance-fixed-v1/normal/attempt-0000..0004`. The initial views match closely on manual inspection. Intermediate/end-of-common-prefix poses differ because the scripted trajectories have different durations: these are same-seed trajectory comparisons, **not pose-matched pixel fidelity tests**; PSNR/SSIM is not reported. The native clips have640–680 frames and ROS clips have different lengths, so the last common index need not be both episodes' endpoints.

The held scene preserves the verified production fixed base and runtime-derived cameras. Moving it to RTX changed exactly three ArUco texture AssetPath strings; a full authored-layer diff verified all other opinions unchanged, and all three PNG hashes match dual. Original scene SHA `42b1a65a7dcea7a7be2f8ca73583efea2d850d9b724650b0f248ab944f5ff245`; relocated scene SHA `28e69c0235e21eb353ea4656217eefda5693a912b01a71742fe1a73485303a6f`. The largest admitted SDG camera matrix error was4.14e-7, below the unchanged 2e-4 threshold. Native stamps, advancing RGB/CameraParams counters, one render per capture and four physical callbacks per advance remain admission checks.

## Cleanup and locations

Sim released **2026-11-14T15:04:53.113004Z**. Worker exit 0 and owner finalization completed/passed with cleanup_errors[]. Seventeen owned PIDs retired, remaining[], ports 8793/8794 closed, NVML compute+graphics verified with no owned contexts/errors. A post-interruption check reconfirmed no owned process/GPU context alive; foreign processes were left alone.

- Original raw run: `rtx6000:/mnt/storage3/aaugus11/exploring-vlas/inproc/datasets/normal-five-v2`.
- Exported dataset: `dual-a4500:/mnt/extreme4tb/aaugus11/exploring-vlas/inproc/rtx-normal-five-v2-transfer/datasets/normal-five-v2/dataset`.
- Export/readback report and five comparisons: siblings `export-report` and `compare-seed-*` under the same transfer root.
- Mac deliverable: `coursework-eee598/roadmap-2026-11/phase 3/3.1f-inprocess-collector/rtx-normal-window/`, including raw evidence, exported dataset archive, validation reports, sample videos, comparisons, qualification and cleanup proof.

Raw evidence archive SHA256: `bff37b2d391b4457621798baa0d5ec692dc5aefb76d4b2bba1a9c7739617c198`. `full-readback.json` is the affirmative3300-row result. Source remains tied to5a3a719; documentation commits do not change that measured code cut.
