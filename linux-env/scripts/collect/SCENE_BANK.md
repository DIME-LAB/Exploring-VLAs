Scene-only sampling uses the real Collector constructor, startup, reset, homing, cup/LEGO sampler, camera checks and retirement path. It stops before expert execution and dataset writing. Normal collection uses the same shared preparation functions. The opt-in bank mode refuses resume/recovery/replay and stops at its first failing seed; it never substitutes a layout.

After an announced domain-92/GPU-0 window is authorized, bring up a fresh collect-mode instance with the existing wrapper and loopback configuration. Keep the policy off during scene generation. Output, runtime and destination must remain on the backup filesystem.

    export SOARM_EVAL_LOCALHOST_ONLY=1 ISAACSIM_GPU=0
    export SOARM_COLLECTION_ROOT=/mnt/extreme4tb/aaugus11/exploring-vlas/eval/runtime33
    export SOARM_INSTANCE_ROOT=$SOARM_COLLECTION_ROOT/instance92
    bash linux-env/scripts/collect_sim.sh up 2
    bash linux-env/scripts/collect_sim.sh run 2 \
      --scene-bank receipts \
      --scene-bank-reference-run /mnt/extreme4tb/aaugus11/exploring-vlas/datasets/rtx6000-normal-i3-b00/normal/run.json \
      --scene-bank-receipts /mnt/extreme4tb/aaugus11/exploring-vlas/datasets/rtx6000-normal-i3-b00/normal/attempt-000{0,1,2,3,4}/receipt.json \
      --output /mnt/extreme4tb/aaugus11/exploring-vlas/eval/bank-generation/golden-training-five \
      --scene-bank-output /mnt/extreme4tb/aaugus11/exploring-vlas/eval/scene-bank-golden-training-five

Golden mode uses each accepted receipt's actual colour, not seed modulo three. It checks the parameter recipe, exact deterministic cup-sampler output, and settled body poses against the original receipt (0.1 mm position, quaternion-component tolerance 1e-4 with sign aliasing). A failed golden check blocks adoption; preserve its receipt rather than relaxing tolerance silently.

After the golden check passes, use a new output/destination with `--scene-bank eval` and omit `--scene-bank-receipts`. This fixes all 30 seeds to 900000–900029, one attempt each, colour=(red,green,blue)[seed%3], normal/no recovery. It restores the selected recorded training run's object reset baseline before invoking the native sampler. The fresh live object-name set and CUP_LAYOUT must match the reference; live startup positions are recorded separately because startup baselines differed across training batches.

The sealed bank contains bodies-only `seed-N.json` files and `baseline.json`, which is an alias of the first sampled layout. The actual sampler reset baseline is in metadata/provenance.json. Metadata includes full per-seed receipts, visibility, parameters, source hashes, reference-run hash and finalization certificate. Publication requires all planned layouts and successful Collector.close()/require_finalized; draft and destination must share a filesystem. Publication refuses an existing destination and uses an atomic rename. A partial bank cannot be accepted by its absence of bank-complete.json and complete manifest.

Generation tests do not qualify deployment loading. Before scoring a new bank, pin its manifest SHA, verify every file, replay all scenes in deploy mode, check physical read-back and both cameras, and retain that qualification. Reusing a seed number across different banks does not create a paired layout. The boundary tests run with TMPDIR under backup storage:

    PYTHONDONTWRITEBYTECODE=1 TMPDIR=/mnt/extreme4tb/aaugus11/exploring-vlas/eval/cache/tmp \
      python3 -m unittest discover -s linux-env/scripts/collect -p test_scene_bank.py

Seven boundary tests and independent static review pass. Live golden resampling and new-bank generation remain pending; these are not claims of live parity. The default collection path and sampler are checked against a sealed AST fixture; empty Python 3.12 type_params are normalized for the Python 3.10 deployment host.

Bank destinations use the qualified host profile canonical root plus /eval: /mnt/extreme4tb/aaugus11/exploring-vlas/eval on dual and /mnt/storage3/aaugus11/exploring-vlas/eval on rtx6000. Both hosts pass the same seven CPU boundary/AST checks. The private rtx6000 bank worktree stays at bf7336d with this scene-only extraction; it does not modify the shared training checkout or the active native policy worktree. Live golden sampling and SHA sealing still require their own announced simulation window.
