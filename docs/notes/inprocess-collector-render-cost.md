# Native versus four-ROS GPU render budget: offline estimate

**The existing evidence does not measure GPU service time, so it cannot establish that replacing one saturated ROS instance would increase accepted episodes/hour.** It does identify a credible source of avoidable GPU work: native produces roughly 694 two-camera capture updates per accepted episode, while ROS has about 677 logical capture ticks plus 3,947 observed held-stamp camera publications. Whether those publications correspond to additional completed GPU renders is not instrumented. Keep the overnight 900 on ROS; test the hypothesis only in a later granted window.

This note uses the accepted native `5a3a719` batch (five normal episodes) and w_rtx's existing four-instance, 24/24 accepted ROS evidence. It makes no source, scene, flag, process or GPU changes. The seeds and instance loads differ. The previously proposed Kit-flag A/B remains unrun and on hold for the lead's later dual GPU 0 GO.

## Counts and measured rates

| Quantity | Native, 5 accepted | Four ROS, 24 accepted |
| --- | ---: | ---: |
| Stored frame pairs | 3,300; 660/episode | 14,683; 611.79/episode |
| Full foreground/profile ticks | 3,405; 681/episode | 16,247; 676.96/episode |
| Foreground camera completions | Exactly one/camera/tick, 6,810 total | Not counted at SDG/GPU completion boundary |
| Additional logged native priming completions | 100 camera completions, 20/episode | Not separated from other work in this comparison |
| Held-stamp camera re-publication classifications | No ROS transport; not an equivalent counter | Wrist 46,017 + top 48,703 = 94,720, or 3,946.67/episode |
| Duplicate transport messages | No ROS transport | 13,138; excluded from all draw hypotheses |
| Complete native render-budget estimate | About 694 two-camera updates / 1,388 camera images per episode, before one-time startup | About 1,353.92 logical camera images/episode, plus unmeasured extra draws |

The native 694 estimate is 681 foreground updates + 10 logged priming completion pairs + two boundary advances + approximately one cancellation/held update. The boundary steps each enforce one fresh completion per camera in source; the cancellation increment is inferred from the counter gaps between episodes. Directly represented foreground+priming logs contain 3,455 completion pairs; 694 is an approximate recurrent budget, not an exact whole-process draw count. RGB annotator compute counters reach 3,586 per camera by the last profile, including startup. The four measurable first-profile to next-first-profile copy-counter deltas are 699,704,714,679: each equals the preceding episode’s foreground tick count + 13 exactly. The fifth cycle is not observed after final teardown. This supports the recurrent count model, but a copy count is not a GPU timestamp measurement or independently instrumented GPU draw count.

For ROS, the 16,247 ticks are logical requests and the 94,720 re-publications are transport/join classifications of distinct same-publisher held-stamp publications. Neither is an instrumented GPU draw count. Do not count duplicate messages as extra draws, and do not assume there were no unpublished/background renders. The native and ROS protocols expose different counters.

Four ROS per-instance batch rates were 30.208, 29.903, 29.376 and 29.162 accepted/hour. Their sum is **118.65 accepted/hour**; this is a sum of per-instance batch rates, not a single exactly synchronized cohort measurement. w_rtx reports approximately 99% card utilization with four ROS instances and no training. Native was measured under fewer concurrent ROS tenants. Its historical post-initialization rate is about 80.61 accepted/hour, but that cannot be added to three saturated ROS rates as a prediction.

## Wall-time proxies, not GPU milliseconds

Timings below use recorded rows on both paths. Native foreground-only values differ negligibly from these means.

| Stage boundary | Mean ms per two-camera tick | Recorded-row sum per accepted episode |
| --- | ---: | ---: |
| Native app update (CPU physics, render scheduling/wait/pacing) | 33.499 | 22.109 s |
| Native physics/render/read wrapper | 38.006 | 25.084 s |
| Native RGB host reads, nested in wrapper | 4.041 | 2.667 s |
| Native join/hash/contract, CPU work | 4.391 | 2.898 s |
| ROS advance request → Kit post-update | 26.270 | 16.072 s |
| ROS image transport/header join | 70.950 | 43.406 s |
| ROS step request → both camera callbacks | 93.868 | 57.428 s |
| ROS validity service work | 50.172 | 30.695 s |

The native app proxy is **higher**, not lower, than ROS's Kit post-update proxy. The native complete read path is shorter than the ROS camera callback path. These facts can coexist: ROS post-update is not a GPU fence, and its later join contains GPU completion, publication/DDS and header matching. Native app time contains CPU physics and possible 30 Hz wall pacing; the median is 33.431 ms. Its camera data is delivered zero or one update later. None of these timers isolate average GPU busy time per camera image. Their boundaries overlap, so do not add Kit+join or subtract the rows to claim GPU savings. These wall proxies do not provide a useful measurement-backed bound for actual GPU service ms/image.

Extrapolating the native app proxy over 694 updates gives about 23.25 app-wall seconds/episode; its wrapper gives about 26.38 seconds/episode. These are scheduling/readback proxies, **not GPU-service estimates**. The native cold startup cost is one-time and must not be multiplied into every long-run episode.

## Conditional GPU-budget estimate

Let `rho` be native GPU busy service per rendered camera image divided by ROS's, and let `alpha` be an assumed mapping from each observed held-stamp publication to an additional camera draw. Neither is measured. Holding acceptance yield and per-image work constant, a count-only model gives

```
kappa = native GPU work per accepted episode / ROS GPU work per accepted episode
      ≈ 1388 * rho / (1353.9167 + alpha * 3946.6667)
```

`alpha=0` and `alpha=1` are illustrative hypotheses, not proven bounds. Unpublished renders, different per-camera work, CUDA copies, changing wait-time re-renders and scheduling feedback can invalidate this simplified model.

- With no extra draws attributable to those publications and equal image cost (`alpha=0,rho=1`), native requires about **1.025×** ROS's work: approximately parity, slightly worse.
- If every observed held-stamp publication represents an additional draw at equal image cost (`alpha=1,rho=1`), native requires about **0.262×** ROS's work. That is a promising count-based hypothesis, not an observed 74% GPU saving.
- At equal image cost, only about **0.86%** of those extra publications need to represent additional draws to offset native's slightly larger logical image budget. Conversely, with `alpha=1`, native image service could be up to about **3.82×** ROS's before the count advantage disappears. These are break-even algebra, not evidence of the unknown mapping or service ratio.

If one additionally assumes the card is solely GPU-service limited, 118.65 accepted/hour corresponds to 30.34 full-card capacity-seconds per accepted ROS episode. Calibrating the two contradictory count hypotheses above would assign native about **31.1** or **7.95** capacity-seconds/episode respectively. Neither is measured GPU time, and this is not a confidence interval. The difference is exactly why the existing data cannot settle the swap decision.

## Aggregate accepted/hour sensitivity

If native gets GPU busy-time share `s`, all processes keep the same GPU work per accepted episode, and none is limited by CPU/feed or faults, then

```
H_mix = H_4ROS * ((1-s) + s/kappa)
```

For the illustrative `s=1/4` case:

| Native/ROS GPU work per accepted episode, kappa | Model mixed accepted/hour | Change |
| ---: | ---: | ---: |
| 0.50 | 148.31 | +25.0% |
| 0.75 | 128.54 | +8.3% |
| 1.00 | 118.65 | 0% |
| 1.25 | 112.72 | −5.0% |
| 1.50 | 108.76 | −8.3% |

This is a sensitivity table, not a throughput forecast. A four-process scheduler does not promise equal GPU shares. ROS held rendering can expand while it waits, so its work/episode is not necessarily constant after a swap. Native's one-update image-join bound has not been qualified beside three fully saturated ROS instances; stalls or terminal faults could reduce its accepted yield. Its measured 80.61/hour under lighter contention is neither a guaranteed feed ceiling nor a rate available under saturation. At the optimistic kappa=0.262 and one-quarter share, the model would require about 113 native accepted episodes/hour, above any native rate measured so far (80.61/hour under lighter contention). The raw count advantage therefore cannot by itself realise that model’s peak rate without better feed performance. Faster orchestration alone is not proof of aggregate GPU efficiency.

The break-even condition is simply **kappa < 1 while native sustains valid output and does not harm peer progress**. Present evidence supports investigating that condition, not changing the overnight collection composition.

## Evidence and next qualification

Reproducible inputs are under the Mac deliverable's `render-cost/`: `ros-four-instance-proxies.json`, `estimate.py` and `estimate.json`. The script derives counts and weighted means from that frozen JSON and the five native receipts, retains input hashes, and runs without importing CUDA/Isaac. The ROS JSON points to SHA-verified backups under `/mnt/a4500-backup/exploring-vlas/datasets/rtx6000-normal-i{3,4}-b01/normal/attempt-*` and `rtx6000-normal-i{5,6}-b00/...`. The original extraction was supplied by w_rtx; no production processes were queried or changed for this estimate.

A later granted experiment should count actual per-product NEW_FRAME/SDG completions across recording, held waits, priming/reset and idle periods on both paths, and obtain GPU timestamp/service measurements (for example renderer GPU timestamps or a Vulkan/GPU trace), rather than re-labeling CPU callbacks as GPU fences. GPU utilization alone cannot attribute service to a process or camera. After the existing 32-tick motion/reset/freshness gates pass, a controlled four-ROS versus three-ROS+one-native pair with the same seed distribution, warm caches and all instances' accepted counts/wall time would directly answer the aggregate question. The overnight 900 and all current GPUs remain untouched; no such run is authorized by this note.
