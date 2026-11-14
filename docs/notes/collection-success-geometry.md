# Primitive cup success geometry

The authoritative asset is `isaac/exts/soarm101-dt/assets/sorting_collision_geometry.json`.
Both the collector success gate and its MoveIt cup objects read this file. Each
cup consists of twelve tapered wall boxes plus one floor box; sizes/centres are
metres in CUP-BASE coordinates, and rotations are wxyz. `/drop_poses` describes
the body centre, 48.26 mm above the base along cup Z, rather than the primitive
origin. The collector transforms the complete compound into the planning base
frame. The GUI scene implementation is unchanged.

Every corner of the oriented physical block must satisfy every inner wall plane
`n.p <= offset`. Normals in the asset are NOT normalized. Physical signed distance
is `(offset - n.p) / ||n||`; normalizing n without scaling the offset changes the
geometry. Unit tests derive the inner faces from the rotated primitive boxes and
check that they coincide with all twelve authored planes. A box and the allowed
interior are convex; checking all eight corners also bounds edges and faces.

The floor top is 1.2 mm and the rim is 96.52 mm. Admission allows at most 20 um of
wall/floor penetration: 9 um measured rest-contact residual, 10 um allowance for
solver variability and 1 um for float/transforms. This is a bounded rest tolerance,
not the 0.1 mm predictive-contact distance; rest offsets are zero. The 20 um bound
is smaller than contact distance and is explicit in metadata. A 0.1 mm side or
floor violation fails; rim admission requires all corners below rim minus 0.1 mm.
No radial-cylinder/CAD envelope or 0.7 mm inward margin is used. The old radial
calibrator/profile files are removed; earlier Mac diagnostic reports retain the
historical measurements for superseded runs.

Success also requires an upright cup, block/cup speed below 0.015 m/s, angular
speed below 0.15 rad/s for fifteen consecutive contract-valid 30 Hz ticks, and
25 mm of lift during the episode. A contract-invalid physical verdict is terminal.

MoveIt uses the exact thirteen boxes, without mesh padding. Native reset readback
checks each box's dimensions and transformed pose together, including parent and
world/base transforms. A tolerance-aware one-to-one match accepts message-order
changes while refusing missing/duplicate shapes, altered sizes, bad orientations
or the 48.26 mm origin mistake. Dimensional tolerance is 0.1 um; pose tolerance is
10 um for ROS/service readback (separate from physical containment tolerances).
The native ACM/attachment/octomap and measured-state checks remain strict.

CPU dynamics are explicit at World construction and verified from the current
PhysxSceneAPI enableGPUDynamics=false attribute. Unknown settings refuse collection.
Runtime metadata and each receipt retain the scene setting; effective tensor
contact/rest offsets and shape counts must match the primitive contract (13 per
cup, 1 per lego, 0.1 mm contact and zero rest). The next smoke must confirm these
settings, floor/wall contacts, reset receipts, videos and timing against the new
scene; no live yield or throughput is inferred from offline checks.
