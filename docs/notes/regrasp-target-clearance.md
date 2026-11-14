# Deliberate-miss target clearance

A recovery miss must leave the target untouched. MoveIt omits the target while
planning the clean corrective expert, because its intended grasp includes
contact. That omission does not authorize contact on the separate executed miss.
The collector therefore uses a second, read-only finger-box oracle for that target.

`collection_runtime.install` serializes all enabled native finger cubes and their
joint frames from the held USD stage. `finger_geometry` maps the six joints by
name, normalizes authored quaternions, rejects unsupported finger geometry, and
extracts collider transforms relative to their rigid body. The model's hash is
independent of the body's current USD pose. It includes both finger frames,
hinges and all pad pieces, not just tip endpoints. No stage/drive/body setter is
part of this reader. The fixed-base audit remains a separate native prerequisite.

A bounded search tries the seeded 2–4 cm direction/magnitude candidates. For each,
it tries requested and +/-180-degree equivalent grasp yaws, rebuilding TCP
centering, projected width and jaw angles. Each equivalent must pass IK, MoveIt
endpoint/path checks and target clearance before it can be selected. The actual
measured-jaw-to-open sweep is checked too. All executed approach, descent, close,
lift and reopen segments need **1 mm target clearance**, plus **20 um numerical
allowance**. This is deliberately larger than two 0.1 mm contact offsets; it is
an engineering separation margin, not an inferred force threshold. The clean
open-jaw approach/descent also need 0.2 mm plus numerical allowance. Clean closed
corrections may intentionally aim to grasp the target and are labels, never the
physical miss's trajectory; they retain their separate MoveIt checks.

The SAT gap is a lower bound on Euclidean box separation. A triangle-inequality
bound for every finger point under joint motion lets the checker certify a whole
linear commanded interval from its nearest sampled endpoint. Uncertain intervals
are bisected; an interval that cannot prove the margin at 0.25 mm motion resolution
is refused. A per-segment check-count ceiling is terminal capacity failure.
Conservative joint-point radius bounds are verified against the serialized joint
tree, including gripper rotation. This certifies commanded rigid geometry, not a
force model or an unobserved physical trajectory between feedback ticks.

During execution, an immutable epoch-bound guard checks the exact final command,
both measured-state and previous-target segments, and the current target pose at
every held tick before publication. Cancellation retires the guard. Current
native target/base poses and checks survive in both successful and failed proofs;
`miss_clearance_ms` is separately profiled, including failure ticks. A missing or
invalid geometry response cannot be converted to a candidate refusal and retried.

Target movement is anchored **before opening/approach**, not after descent.
Maximum corner displacement must stay within 0.2 mm throughout the entire miss,
including rotation. This small contact-band ceiling tolerates rest/float residuals
but refuses a displaced block even if it subsequently stays still. The separate
empty-lift detector examines every recorded miss tick through lift. Its row-window
starts at the first recorded tick; the held pre-miss anchor may be unrecorded, but
must have its own pipeline proof and be either that tick or exactly one native
frame earlier. Reopen is also covered by the guard and whole-miss row hashes.

Hashes use tick fields excluding the separately audited post-capture `wall_profile`.
Copied native identities/poses remain exact. Independently recomputed geometry
uses the shared 1e-9 arithmetic rule, with structured field/value/rule differences,
so Python/NumPy export round-off cannot create a false proof mismatch. Model,
policy, epoch, episode, stage and initial-snapshot evidence remain bound to the
recorded episode; export recomputes the physical clearance from raw rows.

Jaw commands share the same projected-width rule as yaw-specific TCP compensation.
The admitted +/-pi equivalents have exactly the same canonical projection, so
opening before yaw selection and closing afterward use the selected plan's angles.
Tests enter the actual shared grasp/open/close methods for both bbox dimension
orders and both cross settings, verify native-box/table clearance, and exercise
collision-rejected yaw fallback. Legacy 1,000-case math parity is separate from
these execution tests.

Two retained failures exercise the guard: the review's seed 350007 overlap, and
the actual seed 340000 smoke miss. The latter requested a 4.12 mm finger/target
intersection, reconstructed 1.54 mm at its held state, and displaced a target corner
4.98 mm from the pre-miss pose. The first miss never reached acceptance. These are
rigid-box and measured-state observations; contact forces were not recorded.

Offline checks (no Isaac app):

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python3 -m unittest discover -s linux-env/scripts/collect -p 'test_*.py'
~/env_isaacsim61/bin/python -B linux-env/scripts/collect/finger_model_probe.py \
  --output /mnt/a4500-backup/exploring-vlas/runtime/instance-5/finger-model-check.json
```

The USD probe uses anonymous session opinions only; it does not emulate effective
PhysX offsets. A new owned live recovery smoke must verify those, physical miss
clearance/target stability, recovery success, dataset readback and cleanup before
recovery episodes qualify. The normal-only pilot at its frozen cut is independent
of this worktree repair.
