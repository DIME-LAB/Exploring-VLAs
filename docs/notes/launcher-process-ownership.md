# Launcher process ownership

`collect/managed_process.py` owns the process lifetime for indexed/manual Isaac,
collection control, and the control/policy processes started by `_lib.sh` and
`deploy_sim.sh`. The `.pid` file names a small supervisor. Its sibling
`.identity.json` records the supervisor and workload PID/start-time/PGID, launch
command, original PID-file path, instance root and nonce, discovered descendants
and final cleanup result. Identity leaf reads refuse symlinks; receipts are
written atomically with file and directory fsync.

Start serializes on a PID-file lock and refuses a live PID without its matching
receipt, a reused PID, or a receipt for another launcher path. The startup helper
and supervisor both become child subreapers. Only children proven through kernel
ancestry enter the pidfd ledger; children of every task/thread are considered.
If startup fails before readiness, the startup helper retires its own tree.
Once ready, the supervisor retains those process handles until shutdown. An
exiting workload triggers cleanup of its orphaned descendants automatically.

Stop verifies the birth identity again, consults the receipt's instance claim,
and connects to its private abstract Unix control socket. Socket peer credentials
must match the recorded supervisor PID/UID, the immutable request nonce must
match, and the caller waits on a pidfd for that supervisor's exit. The supervisor
signals only its recorded child pidfds, checking birth identity before each
signal. It first allows SIGINT, then SIGTERM, then a bounded final SIGKILL of
surviving owned processes. No process-name matching or numeric process-group
signal fallback exists. A stalled control channel refuses shutdown rather than
expanding the target set. Final cleanup must report no surviving recorded child.

Indexed `close`, `kill`, `restart` and collection `down` take the same instance
claim as launch/up. Restart claims **before** close. Deployment/control shutdown
also checks the claim when an instance root is configured. The direct managed
stop API independently checks the root recorded in its receipt. A runner can
reuse its inherited lock descriptor after inode validation. A manually started
supervisor closes the operation claim before starting the workload, allowing a
later manual close; the smoke runner retains its own claim for the whole smoke.
Unindexed manual instances use their component identity/lock and do not lock
other runtimes, GPUs or users.

`deploy_sim` retains the original stack PID file as the authority: its copied
PID marker must still agree with that file before stop. A legacy PID file cannot
be upgraded into ownership from a command-line pattern, port, `kill -0`, or PGID.
Such files fail clearly. Logs and operator inspection may establish which
process was originally started; these scripts deliberately do not infer missing
birth identities or stop an unrelated process to recover stale bookkeeping.

`launcher_process_regression.py` starts only short Python trees. It proves normal
child cleanup and automatic orphan cleanup, refusal of direct stop while another
owner holds the instance claim, four production close/restart/down claim refusals,
and five production entrypoint refusals of foreign PID-only files. The CPU suite
adds missing-receipt, reused-PID and mismatched-group cases. It starts no ROS,
Isaac or GPU work.

A final escalation also drains newly discovered proven descendants within its
bounded deadline. If any recorded descendant survives that deadline, the
supervisor keeps `cleanup_failed` and its identities. Missing supervisor/marker
alone never proves cleanup: status/down/start refuse while any recorded birth
identity remains, including an unresolved late orphan. They become available
only after those identities are proven gone; no retry silently adopts them.

Every stop branch, including an already-readable supervisor pidfd, re-reads the
final identity receipt and requires the original nonce, birth identity, stopped
state and empty descendant cleanup. An exited supervisor alone cannot report
success. Runner cleanup errors and port-probe errors produce a failed smoke
receipt; an outer finally releases the instance claim even if finalization fails.
Receipts published at producer exit are reconciled before starting another stage.

Indexed GPU launches use the assigned physical card UUID as CUDA_VISIBLE_DEVICES.
CUDA ordinals become zero within that process; the renderer uses its Kit-table index, resolved against physical UUID and PCI
bus. A previous owned log is only a launch hint; the current log must prove the
sole Active device is the assigned card. The collector records startup and before/after-attempt NVML memory
for launcher-owned birth identities. Other users' contexts are ignored; any
positive owned memory on another card fails. Persistent shader/OptiX/CUDA caches
are private to this project and keyed by hostname plus card UUID. Camera warm-up
precedes recording; neither the shader directory nor other projects' caches are
cleared.

GPU attribution enumerates both nvmlDeviceGetComputeRunningProcesses_v3 and
nvmlDeviceGetGraphicsRunningProcesses_v3 for every card. Shared PID/card memory
uses the larger value rather than a sum. API errors and unknown memory stay
unverified. Kernel ancestry rooted at verified supervisors and the collector
includes descendants; birth/PGID sets are reconciled on both sides of sampling.
Unstable samples retry at most three times and never report a confirmed allocation
against a reused PID. Renderer table mapping follows [NVIDIA Kit Q9](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/110.1.5/guide/linux_troubleshooting.html#q9-how-to-specify-what-gpus-to-run-omniverse-apps-on).
The NVML ABI is pinned to the installed nvml.h nvmlProcessInfo_v2_t definition;
queries do not load a CUDA runtime. These are sampled allocation checks, not a
kernel device-access boundary.

Resume validates the predecessor certificate before beginning another invocation.
Incomplete predecessor evidence is refused and preserved. A valid predecessor
certificate and its exact summary text are archived before status changes.
Cleanup before an invocation began writes separate instance runtime diagnostics.
Shutdown requires executor.shutdown=True and a retired spin thread; failed/unknown
results poison the generation and fail certification. Certificate schema and
invocation checks detect accidental mixes; these are local consistency receipts,
not authentication of a same-permission writer.
