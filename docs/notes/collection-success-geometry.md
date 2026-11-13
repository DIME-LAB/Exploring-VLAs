# Cup-local success geometry

The success gate measures the complete oriented block box in the cup's physical
body frame. Each corner must lie above 0.6 mm, below 94 mm, and within the
height-dependent interior radius in `cup_profile.json`. The block and cup must
stay settled for 15 consecutive 30 Hz simulated ticks, with the cup upright.
The episode also needs at least 25 mm of measured lift.

The radius is a concave piecewise affine lower envelope of the full CAD shell's
interior, expressed in metres. Four nodes describe the floor fillet and tapered
wall. The first sampling plane is 10 nm above the 1.2 mm authored floor to avoid
a coplanar STL slice. The calibration takes the smaller of the actual slice
inradius and active triangle supporting-plane distance, then subtracts 0.1 mm
at the nodes. Across all analytical critical heights the envelope's minimum
radial plane separation is about 0.0598 mm; it is not a uniform 0.1 mm clearance.
A further 0.7 mm inward geometric allowance leaves at least about 0.7598 mm
radial separation from the authored supporting planes. The floor keeps its
separate, explicit 0.6 mm geometric allowance.

Native RigidBody tensor getters resolve every block to contact offset
1.3625002466 mm and rest offset 0. All 16 shapes of each cup resolve to contact
0.0999999975 mm and rest 0. These effective values supersede USD `-Infinity`
sentinels and UI/default parameters. The 0.7 mm allowance is a geometric design
margin; it is not a sum of contact/rest offsets. [PhysX](https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/_api_build/class_px_shape.html)
uses contact offsets to generate predictive contacts and rest offsets to set
resting separation. Positive contact offsets do not authorize overlap; zero
rest offsets imply no nominal resting overlap.

For each nonhorizontal STL triangle, calibration verifies the envelope against
its infinite supporting plane over that triangle's complete active height
interval. Both sides are affine between envelope nodes and a plane's sign
change, so checking interval endpoints, nodes and sign changes proves the
inequality between them. All 24,010 triangles pass. The source mesh hash is
checked at collection startup. Concavity means the admissible volume is an
intersection of convex frusta; testing all eight box corners bounds the whole
box, including edges and faces.

The returned cooked inner-wall intersections lie inward of the authored shell at the measured
corner directions, restricting the physically reachable settled interior.
The envelope certifies containment in the authored cup; measured physical
settlement checks behavior under those cooked collisions. The geometric radial
allowance does not promise uniform normal distance from every cooked face.

Reproduce calibration without ROS or Isaac:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python3 -B \
  linux-env/scripts/collect/calibrate_cup_profile.py \
  --output /mnt/a4500-backup/exploring-vlas/runtime/instance-0/cup-calibration.json
```

The script prints the measured slice radii, conservative envelope, margins,
source hash and triangle verification. It refuses a nonconcave or unverified
result. Changing the cup asset also requires remeasuring the floor/rim and
checking the live collider; regenerating a profile alone is insufficient.

The checked-in physical fixture records seed 310000, including all 15 settled
snapshots. Its maximum-radius corner is index 1 at height 15.706255 mm and
radius 25.245268 mm. Its corresponding bottom corner is at 2.924815 mm with
nearly identical radius. At these directions/heights, the authored mesh wall
radii are 30.787616 mm and 29.051482 mm. The PhysX cooking interface returned
16 convex pieces; its side-wall ray intersections are 26.486565 mm and
26.486631 mm. The block centre is 6.783824 mm off axis.

The returned cooked bottom slab reaches 2.924848 mm, matching the measured
block bottom to less than 0.05 micrometres. This supports contact with an inflated
cooked floor, rather than a wall-taper perch. The authored floor is 1.2 mm;
convex decomposition is a coarse approximation, not the smooth CAD shell. The
radial success profile bounds the authored interior. Returned cooked convexes
provide contact diagnostics and must not be represented as one circular wall.
The exact fixture passes with 0.594024 mm minimum radial admission clearance.
Rim-resting, outside, protruding, below-floor, moving and tilted-cup fixtures
still reject. This CPU regression does not turn the historical failed attempt
into an accepted episode or replace the fresh five-success smoke.

`--diagnostic` retains failed videos and writes final recorded-frame stills with
frame index and native stamp in the receipt. Each PNG must exist, be nonempty,
and decode before `saved=True`; a missing/corrupt output rejects diagnostic
acceptance through the existing failure path. Older stills are removed before
extraction so a zero-output FFmpeg success cannot reuse them. It adds no simulation steps. The
settled verification tail is explicitly outside the training video. Production
keeps its success-only video retention and export always excludes failures.
