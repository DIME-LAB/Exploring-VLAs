# Collection comparison rules

Agreement uses `linux-env/scripts/collect/comparisons.py`. The rules distinguish
an exact copy from a value independently sampled or calculated. Timestamps are
never replaced, rounded, resampled or normalized by validation. Epochs, IDs,
hashes and monotonic ordering stay exact. Physical pass/fail bounds remain
physical requirements, separate from numeric agreement tolerances.

| Comparison / entrance | Rule and reason |
| --- | --- |
| Sensors join: camera, measured JointState and native pose stamp | `stamp_ns`: absolute 1,000 ns; the capture contract's existing bound for independent ROS camera/native producers. The coarse join bucket is only an index; the full stamps are checked afterward. |
| Sensors controller-reference diagnostic / ROS held-clock join | `stamp_ns`; the diagnostic retains original stamp/epoch and age. Future clocks beyond this bound fail. |
| Capture state / velocity versus native articulation snapshot | `state_rad`: absolute 1e-7 in the quantity's units, preserving the original capture bound. State is radians; velocity is radians/s. |
| Consecutive native or camera ticks | `native_ns`: 10 ns about 1/30 s, allowing integer-nanosecond quantization. Strictly positive ordering; missing/duplicate frames fail. |
| First camera frame after reset | `stamp_ns` about one 1/30 s step from the native floor; subsequent frames use the tighter same-producer increment rule. |
| World time progression | `world_s`: minimum 1 microsecond, plus the existing four float32-substep and double-ULP accumulation bound. Positive progression, exactly four physics steps per camera frame; a frozen, backward, missing or extra substep fails. Absolute World/native equality is deliberately absent. |
| Runtime after-update / TickContract / trailing-verdict World progression | The same `world_step_errors` calculation and shared numeric comparator; no separate 100 ns native-runtime bound. |
| Native command apply versus camera observation stamp | `stamp_ns`: these are independent producers. A stale or future observation outside 1 microsecond is refused while held. |
| ACK application latency versus camera observation | Shared interval comparator, zero to one tick with `stamp_ns` at the endpoints; the five-second-late probe still fails. |
| Expert observation, ACK observation echo, declared command stamp, command candidate, geometry proof stamp | `copy`: all originate from the same camera row; exact equality. |
| Command clock versus held row | Shared interval comparator, zero to one tick with `stamp_ns`; the live held-clock join is checked before command computation. |
| PhysX ACK target / persisted target versus commanded target | `target_rad`: absolute 1e-6, zero relative tolerance, retaining the float32 drive readback bound. |
| Clean corrective action / planner plus DART arithmetic | `arithmetic`: absolute 1e-9; recomputed from the same measured row. Noisy gripper and nonzero expert-path filter deltas still fail exactly. |
| Derived command velocity | `velocity_rad_s`: absolute 1e-4, preserving the original provenance check. |
| Restore baseline stamp, World value, targets versus its returned snapshot | `copy`: these fields are assigned from that same readback. No tolerance hides receipt substitution. |
| Pause baseline versus pause result | `copy` for epoch/stamp identity. World/native progression is reconciled against the last row or last boundary; repeated boundaries cannot erase that anchor. |
| Held cancellation versus previous held boundary | `native_ns` for native clock quantization; `world_s` for independent World accumulation. Epoch strictly increases, native time cannot rewind; at most the one already-in-flight frame may finish, with matching World progression. Multi-frame gaps fail. |
| Restore physical positions/orientations/zero velocities | `float32_readback`: absolute 1e-6 in the field's units; quaternion comparison also admits q/-q. Six arm positions and velocities are read back. These are float32 write/read agreements, not the geometric success envelope. |
| Snapshot native stamp versus row.pose_stamp_ns | `copy`: both are copied from that native snapshot. This replaces the incorrect exact comparison to the camera row stamp. |
| Snapshot native stamp versus camera row stamp | `stamp_ns`, as in capture. The row retains both original values. |
| Snapshot state/velocity versus row.physics_state/physics_velocity | `copy`: both come from that native snapshot. |
| Snapshot state/velocity versus ROS row.state/velocity | `state_rad`, as in capture; the native copies above remain exact. |
| Snapshot World time, bodies, state source, epoch versus proof | `copy`; these are direct copies, not independent clock or pose calculations. |
| Verdict proof identity and raw frame hash | Exact SHA-256/identity. Numeric tolerances do not weaken proof hashes. |
| Home/settle and success gate contract | `snapshot_contracts` runs all rules before any physical quality verdict. Native-native window progression uses `native_ns` and the same World-step check. |
| Re-grasp close/lift window and endpoints | Select and reconstruct from copied native `pose_stamp_ns`, `physics_state` and `physical_bodies`, then compare exact copies. Camera stamps remain unchanged in raw rows and their hashes. This also binds stage endpoints and peak-miss detection to the same native clock. |
| Recovery stage report / geometry proof / resume and export | Exact copied native endpoint/epoch/episode/hash identity; every raw row is re-audited with the shared rules. |
| Servo planned versus executed duration agreement | `world_s` (1 microsecond) for two computed duration values; physical motion deadlines still use the operation's own sim budget and wall watchdog. |
| Receipt provenance, reset floors, epoch cancellation, ROS arrival ordering | Exact identity / strict ordering. These are not independently computed approximate equality tests. |
| Cup profile, floor/rim/interior clearance, lift, uprightness, settled speed, home spread and convergence, re-grasp peak displacement | Physical bounds remain unchanged. They are intended task requirements rather than equality between two producers; the numeric comparator does not add a physical allowance. Planning-scene pose/geometry checks retain their existing documented spatial tolerances. |

A failed verdict retains the complete snapshots and tick proofs plus structured
agreement failures (`field`, `left`, `right`, `delta`, `rule`, `tolerance`). Receipt
`pipeline_contract_failure` preserves this evidence, so a later cancellation
cannot erase the original disagreement. A short one-frame audit returns an
unavailable RTF rather than dividing by zero; duration/idle quality rules still
prevent its acceptance as an episode.

## Retained evidence and offline regression

`fixtures/retained_verdict_snapshots.json` contains the **26 unmodified** native
initial/episode-start snapshots from the 13 accepted diagnostic pilot receipts.
The earlier clock-pair fixture remains unchanged. `test_comparisons.py` feeds each
retained snapshot through actual Collector/Servo/native-apply/ACK capture with a
2 ns camera offset, then through observation/proof, home, containment-contract,
restore-boundary and dataset timing contracts. This does not claim that a single
held tick is a physical home/containment success or a valid recorded episode.
The dedicated actual-Collector construction test likewise publishes camera
stamps 2 ns beyond the fake native tick. Copy changes, >1 microsecond joins,
clock gaps and mutated recovery evidence are negative fixtures.

The v2 failure did **not** retain its failing home sample or proof. Its final
native cancel stamp was 12,416,666,667 ns, but that alone cannot reconstruct the
sample's joint values or camera stamp. The new 2 ns fixtures reproduce the
producer-boundary failure without claiming to reconstruct that missing live
tick. V3 must verify the physical run; retained v2 files remain diagnostic only.
