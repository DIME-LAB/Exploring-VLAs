# Full collector world at reset barriers

The retained primitive smoke succeeds on red and green placements, then stops at
the next pre-randomization reset: green_2x3_0 is missing from MoveIt and its ACM.
The log explicitly says skipped1in-cup. This happens in reset_release's sync_scene,
before randomize_scene; it is not evidence of a failed teleport or stale native
pose. The retained initial snapshots across seeds show real object/cup diversity.

GUI's ordinary lego refresh deliberately omits in-cup blocks to avoid a false
start collision on its post-drop home path. That verified GUI behavior remains
unchanged. Collector reset verification requires the entire measured world,
including a correctly placed block that has not been randomized yet. Its own
SceneReconciler.rebuild_legos builds every physical lego from the native snapshot
using the shared geometry factory, then installs the full collision policy and
runs the unchanged pose/shape/attachment/occupancy/ACM checks. Missing geometry or
an unsuccessful ADD is terminal; no comparator or physical tolerance is loosened.
The retained post-green-drop fixture reproduces the old exclusion and verifies
the full rebuild, then still rejects a mutated shape dimension.

Attempt finalization records FailureStreak exactly once after normal/fallback
assembly. If terminal-marker publication itself fails, the fallback preserves the
original fault and publication error and still writes its receipt/ledger. Such a
run is terminal, not accepted or a quality reject. The regression exercises the
actual finalization code with a marker OSError and asserts one streak observation.

Unresolved camera groups have a separate durable-boundary check described in
collection-image-identity.md. The bounded publication wait never advances physics.
The collector-only changes introduce no GUI scene waits or mode inference.
