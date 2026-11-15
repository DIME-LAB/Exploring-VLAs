The execution-cadence diagnostic is a separate 24-episode experiment on bank-v1's OOD cup arrangement. It uses the staged primary30k checkpoint and seeds900000–900005, two per colour, at 30 actions per sim-second. Each arm uses its own warmed persistent native policy/client: re-observe10 RTC-on (a traced diagnostic repeat),25 RTC-on,25 RTC-off,50 RTC-off. Original MAIN results are immutable controls; these diagnostic repeats never replace them.

After an explicit GPU0 GO and START announcement, run:

    PYTHONDONTWRITEBYTECODE=1 python3 linux-env/scripts/eval/run_execution_diagnostic.py \
      --out /mnt/a4500-backup/exploring-vlas/eval/runs/execution-horizon-diagnostic-v1

The runner requires a new output, pins bank/model SHA, launches only instance2/domain92 with CycloneDDS loopback, and uses GPU0 for both Isaac and policy. It warms100 non-actuating actions per arm, saves all four manifests/configs/results/videos, applies a74-minute watchdog, stops its own child process groups on timeout, and runs instance-scoped cleanup in finally. Send STOP after checking the release receipt and GPU table. A failed arm stops the diagnostic without retrying an attempted scene.

`run_episodes.py --diagnostic-reobserve-actions {10,25,50}` selects this eval-only cadence. Default MAIN remains10. The50 cadence requires RTC-off. Its forced-empty retry is disabled: at a full-chunk boundary, normal inference may consume >=10 sim ticks; retrying there would overlap/blend a second chunk.10/25 retain the existing10-empty-slot recovery. Report boundary starvation and unusually long gaps separately; a failed chunk can leave the50 arm idle to its finite episode deadline. Native RPC/camera health checks remain in force.

`--diagnostic-prediction-trace` records every postprocessed50x6 chunk in degrees, plus its capture sim stamp and observation step. The server attaches it to the existing TimedAction RPC metadata, and the receiver saves it before native queue aggregation. There is no new work in the dispatch tick. Posthoc, join capture stamps to physical snapshots to compute each predicted endpoint, the nearest block, endpoint-to-target distances, and switches between re-plans. Endpoint-nearest labels are geometric diagnostics, not claims about visual attention or intention.

Keep the original Phase3 deployment rate, camera/normalization/RPC paths, eager backend,50-action chunks, RTC horizon10/weight10 and measured feedback. Cadence is the diagnostic intervention; queue starvation remains an outcome. Use the normal validity checks with the explicitly requested diagnostic cadence, rather than the frozen MAIN reporter's hardcoded10 requirement. Run the short timing checks with no GPUs visible and TMPDIR on backup:

    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 ROS_DOMAIN_ID=100 \
      TMPDIR=/mnt/a4500-backup/exploring-vlas/eval/cache/tmp \
      python3 -m unittest discover -s linux-env/scripts/eval -p test_sim_time.py

Ten CPU timing tests pass, including fake clocks at RTF0.3/1 and a real paced-loop test with13-tick inference latency that executes sequential50-step chunks without retries or blending. Independent review verified the adapter and telemetry placement. Live diagnostic execution and endpoint-switching analysis remain pending.
