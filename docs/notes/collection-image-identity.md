# Camera packet identity and held re-renders

An Isaac app update need not advance physics. The supplied primitive-scene probe
observed 7,440 app updates and 7,439 render/SDG completions per camera while only
four physics substeps advanced (one 30 Hz frame). Same-stamp wrist RGB digests
were different. The retained scalar/hash evidence is in
`collect/fixtures/held-rerender-boundary.json`; original RGB payloads were not
provided, so tests reconstruct different pixels rather than claiming those bytes.
Its image stamp 178316666666 ns is 1 ns below the cancellation floor.

Before identity checks, discard pre-epoch camera packets and raw stamps at/below
floor + 1 us (the independent-producer tolerance). Valid next physics ticks are
33,333,333 ns later. For active frames, require nonzero per-packet RMW identity:
implementation plus the 24-byte GID. Bind each camera stream for this subscription
lifetime. Different source, unknown identity, changed layout or distinct nearby
stamp is terminal. For an exact stamp from the same source and same layout, keep
the first taken pixels permanently. Count identical repeats separately from
changed-pixel re-renders; retain first/last digests and count in receipts. No late
re-render can replace a pending or consumed image. The DDS subscription order,
under its existing mutually exclusive callback group, defines first taken.

Startup and every episode boundary also query each camera topic: require exactly
one graph endpoint with a nonzero GID, and retain its type, node and QoS. These
graph GUIDs are diagnostics in their own namespace. Per-packet identity guards
every taken image without adding per-tick graph queries.

Graph polling alone cannot establish this. The actual Humble/CycloneDDS graph
counterexample published the first same-stamp 640x480 rgb8 image from source B,
which vanished after 0.20073 s. Graph observations at 0/1/2 s all contained only A;
the intermediate observer saw A+B. The metadata is checked in as
`fixtures/publisher-graph-gap.json`; rerun `publisher_graph_gap_regression.py
--output <canonical-fresh-path>` on domain 90 with its instance down. It starts
only ROS test nodes on an isolated probe topic, never Isaac/control motion.

[Humble rclpy](https://github.com/ros2/rclpy/blob/humble/rclpy/src/rclpy/subscription.cpp#L124-L127)
exports only source/receive timestamps from take_message, and its executor drops
that info before invoking a callback. `image_take.c` uses
rcl_take_serialized_message on the SAME subscription pointer while Python holds
its normal borrowed handle. It creates/destroys no ROS entities. The serialized
buffer and an owned copy have explicit bounded allocation/free paths; Python's
normal deserialize_message constructs the Image. A collector-only executor
retains that packet metadata and delegates every unmarked subscription to Humble.
It validates the tested executor API shape, and transport failures enter the
existing protected callback and capture terminal paths rather than killing the
executor. GUI/control executors are unchanged.

The local native helper is compiled using the installed Humble headers/libs,
with strict warnings, into the canonical runtime cache. A checked-in ABI manifest covers the full compiler header dependency set, loaded
RCL/RMW/CycloneDDS/rclpy native libraries and their dependency closure, and the
Humble executor implementation. Unknown or changed inputs refuse startup; cached
helpers must match both the current fingerprint and their complete build receipt.
C/Python source hashes and the verified library hash identify the adapter.
The C compilation input is copied into that owned cache before compiling, so a
shared-checkout edit cannot change the file while the compiler reads it. Compiler
errors refuse startup and preserve their traceback. No system library is changed.
Unit tests cover buffer release after success/deserialization failure, empty/error
takes and executor delegation/error forwarding. Real ROS tests exercise the actual
native ABI, same-source changed pixels, stale floor frames, and identical pixels
from two distinct publishers. Actual Collector/PickPlaceNode boot uses this same
adapter; no testing-only recording entrance exists.

[CycloneDDS Humble](https://github.com/ros2/rmw_cyclonedds/blob/humble/rmw_cyclonedds_cpp/src/rmw_node.cpp#L2914-L2928)
places its local DDS publication_handle in received publisher_gid, while graph
endpoints carry GUIDs. Packet IDs are stable/distinct within the receiving
subscription/context; they are not compared with graph IDs. This implementation
reports unsupported sequence numbers and receive timestamps; those are null in
metadata, not invented clocks. Image header sim stamps remain unchanged. RMW
source time is wall-clock diagnostics only and never a dataset timestamp.

`publisher_identity_regression.py --output <canonical-fresh-path>` exercises real
ROS images (640x480 rgb8), confirms first pixels survive re-renders, rejects a
second publisher with identical pixels and verifies the cancellation-floor case.
It also samples 50 warm native take/decode costs. It requires isolated domain 90
and the private Isaac/control instance to be down. These are transport tests,
not physical episode or GPU performance evidence.

The viewport/render-product SDK exposes update-disable controls. Correct render
completion ordering has not yet been validated on this collector's products;
render pausing remains a separate measured optimization on an authorized own
instance. First-packet correctness does not depend on it.
