# RTC replay and client timing

The deployed `anirudhrani/smolvla_sim_100ep_fft__10ksteps_h200` policy was run on GPU 1 against actual decoded wrist/top images and joint observations from episode 0 of `rec_222812` (617 frames). The replay calls `PolicyServer.SendPolicyInstructions` and the production `PolicyServer._predict_action_chunk` pipeline, including its observation conversion, processors, normalized-prefix tracking and action postprocessing. No ROS, Isaac or synthetic policy output is involved. The network transport and camera blocking are not exercised by this offline experiment.

## Paired result

23 observation frames (0, 25, …, 550) produce 50-action chunks, yielding 22 overlapping chunk boundaries. Each paired OFF/ON prediction gets the same random seed (`42 + chunk index`). Runtime arrival delay is `ceil(max(OFF latency, ON latency) × 17.4)` for the paired boundary, so both curves use the same execution schedule. The boundary is the first **executable** new action after that delay minus the previous chunk's last executed action; it is not the stale action at index zero that async execution discards. The metric is the Euclidean norm over six joint targets, in degrees.

| Mean boundary jump | RTC off | RTC on | Reduction |
|---|---:|---:|---:|
| Raw policy chunk | 9.0592° | 4.7737° | 47.3% |
| With client `0.3 old + 0.7 new` blending | 6.8831° | 4.0850° | 40.7% |

Mean warm inference pipeline time: OFF 0.4016 s, ON 0.4699 s. The RTC debug tracker recorded a maximum correction L2 norm of **18.0334** in normalized latent space. This is direct evidence that guidance was applied, beyond setting an enabled flag. `rtc_discontinuity.png`, `boundaries.csv` and `summary.json` contain the plot and measurements; `chunks.npz` retains all six-dimensional output chunks on the remote host in `~/w_policy_phase1/rtc17.4/`.

RTC reduces the mean jump in this replay; some individual boundaries worsen. This does not establish task success or a general performance gain. It does not resimulate the changed observations that a different action would cause. No full item-1.9 evaluation was run.

## Implementation and intended alignment

The server retains a detached copy of the last **sent**, truncated chunk before action unnormalization; RTC compares predictions in the same model space. It slices that chunk at `next_action_index − previous_chunk_start` and supplies `prev_chunk_left_over` plus `inference_delay` to `predict_action_chunk`. Empty/exhausted tails become `None`. Client observations now identify the next unexecuted action (`latest_action + 1`), correcting the prior one-step alignment error.

Measured end-to-end prediction time (including preprocessing, CPU transfer, postprocessing and observation age) is converted into steps using measured action-index advancement per observation timestamp interval, capped at configured client fps. The first prediction has no tail and delay zero. Episode Ready/reset clears the prefix, delay, similarity filter, predicted indices and policy queues under the same lock as inference, while retaining model weights. The env switch `LEROBOT_RTC_ENABLED` remains available; the launcher explicitly sets it to 0 for `--no-rtc`. Debug correction tracking is opt-in with `LEROBOT_RTC_DEBUG=1`.

A prior replay at 19.4 Hz is retained in `replay-19.4/` alongside the current 17.4 Hz evidence. It measured 9.0486→4.4486° raw and 6.6993→3.7737° blended mean jumps. These runs use measured latency schedules from different wall-clock runs and are not a controlled between-rate comparison.

The implementation follows the checked-out fork's `policies/rtc/modeling_rtc.py`, `examples/rtc/eval_dataset.py`, and `examples/rtc/eval_with_real_robot.py`. The [upstream RTC documentation](https://huggingface.co/docs/lerobot/rtc) describes passing the unexecuted prefix and measured delay together. The fork's implementation is the API authority; newer upstream docs also describe modes absent from this checkpoint.

## Dataset fps, real cadence and async timing (1.3)

`meta/info.json` says **30 fps**. Those metadata timestamps index frames and model action steps; they do not set deployment wall-clock speed and do not resample async predictions. The lead selected **17.4 Hz** from item 1.1's calibrated motion-phase estimate of 17.43 Hz (95% CI 17.27–17.58). The whole-run wall estimate is 19.35 Hz with a much wider interval; historical RTF remains unresolved. `RECORDED_FPS` in `run_async_inference_sim.py` is the single deployment default. Both 17.4 and 19.4 remain selectable for the lead's later evaluation sweep. A model step remains one recorded frame/action pair. Playing 30 steps/s runs the sequence 1.724× faster than the selected 17.4 Hz default (or 1.546× faster than the earlier 19.4 Hz whole-run estimate).

`--fps` is a float and reaches **both** server and robot client. The server spaces TimedAction timestamps by `1/fps`; the client sleeps to a target period `1/fps`. The client dequeues one action per iteration, discarding arriving actions whose indices are already executed; timestamps do not schedule individual actions. Matching indices are aggregated by default with `0.3 old + 0.7 new`. This averaging is independent of RTC guidance.

With the default 50 actions and `chunk_size_threshold=0.5`, the client starts sending observations when at most 25 queued actions remain. At 17.4 Hz, 50 steps represent 2.874 s and the 25-step lead represents 1.437 s (at 30 Hz: 1.667 s and 0.833 s). It can send multiple observations below the threshold; the server retains only the latest queued observation and filters similar/already-predicted ones. `--action-duration-s` separately sets the ROS FollowJointTrajectory target horizon; its default is one requested client period, **0.057471 s**, and it can be overridden (for example 0.1 s for longer interpolation).

**The robot client performs action dispatch and observation capture in one thread.** Fresh-frame camera reads block that thread. Tick duration is therefore `max(1/fps, action dispatch + camera reads + observation transmission)`, and observed action rate can be below the target. Camera config fps does not force the ROS publisher cadence. Dataset fps=30 is deliberately preserved; no metadata is rewritten. The launcher now requests `rgb8` for both camera topics to match the sim worker's ROS2CameraHelper output. RGB order is preserved: the decoder's rgb8 path is already RGB, while the recording's rgba8 path dropped alpha to RGB.

## Reproduction

From the repository on dual-a4500:

```bash
# Dataset replay requires LeRobot's dataset extra in the existing pixi env.
~/.pixi/bin/pixi run --manifest-path linux-env/pixi.toml pip install -e './lerobot[dataset]'
CUDA_VISIBLE_DEVICES=GPU-ba9bc29d-f43c-30f0-af4e-22dce230810e HF_HUB_OFFLINE=1 \
  ~/.pixi/bin/pixi run --manifest-path linux-env/pixi.toml python \
  linux-env/scripts/eval/replay_rtc.py \
  --root ~/a4500-backup-20260828/transfer/rec_222812 \
  --out ~/w_policy_phase1/rtc-reproduce
python3 linux-env/scripts/run_async_inference_sim.py --dry-run
python3 linux-env/scripts/run_async_inference_sim.py --dry-run --fps 17.2 --action-duration-s .08
```

Verification: 35 async server/client tests pass in the pixi environment, including four new checks for normalized prefix preservation, action-index slicing, measured delay, exhausted tails, RTC off, fractional rates and episode reset. An additional regression verifies that a chunk arriving during robot dispatch cannot requeue the executing action; dequeue/index advancement and chunk replacement share one lock. Both default and overridden command construction were inspected. Shell syntax and Python compilation pass. Live harness verification is documented separately after the coordinated two-episode smoke run.
