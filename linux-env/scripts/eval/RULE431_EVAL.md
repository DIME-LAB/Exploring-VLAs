# Rule431 evaluation operator

The operator prepares CPU evidence and prints inert, reviewable window plans. It never starts a GPU window automatically. All inputs and outputs on dual live under `/mnt/a4500-backup/exploring-vlas/eval/`; the private runtime is `/mnt/a4500-backup/exploring-vlas/eval-native-rule431`. The collector implementation remains the exact bf7336d cut. Use dual GPU0 only and domain92 loopback. Announce START/STOP through the lead after ABI verification and at least 60 seconds without foreign GPU0 compute. Existing own7GiB/global79% thresholds and foreign-process guards apply.

## Checkpoint preparation (Mac)

Copy the four sealed reference JSONs to `eval/rule431/contracts/` once. The checkpoint stager publishes `eval/checkpoints/rule431-base25k/NNNNNN/handoff.json` only after completion-marker, stable-size and SHA verification. Every checkpoint must load its OWN saved crop processor and subset pre-state/post-action normalizers, with its directory on server PYTHONPATH; pass raw 640x480 top/wrist images and never apply a second crop.

Run `python3 rule431_eval.py prepare --step 5000 --processor-proof <actual-saved-crop-005000-proof.json> --local-state <new-proof-dir>`. This validates the family/source identity, reference six hashes, state/config files and exact saved pre/post float32 stats; runs the production CPU RPC smoke; and generates source preflights for rule15, held0, held1 and a one-seed unscored smoke. A READY JSON is CPU readiness only.

## Qualification and scoring plans

Print an inert plan with `python3 rule431_eval.py plan --mode qualify --suite rule15 --stop-by-utc <UTC>`. Qualify the whole new15 bank; also qualify seen5 as the same-source bootstrap control, then held0 (ALL30 held layouts). Qualifications are invalidated by any file in `native_environment.QUALIFICATION_FILES` changing. Verify all source files against the reviewed dev commit before launching an owner argv.

After qualification, `--mode smoke --suite rule15 --step 5000` prints a separate UNSCORED one-seed policy plan. `--mode score --suite rule15 --step 5000` prints the first scored seen15 DEFAULT arm. For 10k/20k/25k, also run held0 and held1, each fixed15 seeds with its own warmup and STOP. Use the actual UTC hard-stop time, allowing the existing owner three minutes for cleanup. Do not merge a new attempt into an interrupted root.

Default protocol: 45 sim seconds, native ACK held ticks30Hz, eager, warm100 held non-actuating actions, cadence10/RTC-on/weighted queue, paired flow repeat0 by layout. Rule-target selection is fixed from the live baseline before rollout, never passed as model input. Rule-seen15 must match the recorded target; held-bank comparison is diagnostic only. Report live candidate distances/margins, ambiguous <=5mm cases, all-valid and unambiguous-only rule success separately. Contact telemetry and opt-in lossless wrist retention around confirmed closures are enabled. Wrist output is encoded only at episode end.

Audit a stopped root with `python3 report_rule431.py <root> --out <new-audit-dir>`. Report requested-colour and rule-target success separately, wrong-colour/same-colour-wrong-instance picks, invalid attempts, Wilson95 intervals, per-colour counts, and actual-native CONFIRMED>=2degree closure geometry. Yaw compares world-horizontal TCP local+Y to block local+X modulo180degrees; record tilt/degenerate status. TCP is the fixed jaw tip, not aperture centre. The old29.88mm first-close SD is descriptive for a different metric/cohort and cannot calibrate these new comparisons.

Qualification version is explicit: use `--attempt 2 --qualification-attempt 2` for fresh v2 qualification plans and `--qualification-attempt 2 --revision v2` for scored/smoke plans bound to the new CPU proof set. The default remains v1. The first seen5-v1 passed; rule15-v1 stopped before launch with an import-path error, so the corrected registry requires fresh seen5/rule15/held0 v2 qualifications. These old attempts stay recorded.
