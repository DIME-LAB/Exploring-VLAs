# LeRobot Training & Deployment Guide
**SO-101 Arm | SmolVLA / Pi0.5 | Windows + RTX 4090**

---

## Prerequisites

```bash
conda activate lerobot
pip install -e ".[smolvla]"   # SmolVLA
pip install -e ".[pi]"        # Pi0.5 (installs custom transformers branch)
```

## 1. Hardware Setup

| Device | Port | Notes |
|--------|------|-------|
| Follower arm | COM3 | SO-101 follower |
| Leader arm | COM4 | SO-101 leader (teleoperation) |
| Wrist camera | OpenCV index 0 | 1280x720 or 640x480, MJPG |
| Top camera | Intel RealSense 747612060071 | 640x480 |

Find cameras:
```bash
lerobot-find-cameras OpenCV
```

---

## 2. Teleoperation (manual control)

```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=COM3 --robot.id=my_awesome_follower_arm \
  --teleop.type=so101_leader --teleop.port=COM4 --teleop.id=my_awesome_leader_arm
```

---

## 3. Record Data

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=COM3 --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}, top: {type: intelrealsense, serial_number_or_name: 747612060071, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader --teleop.port=COM4 --teleop.id=my_awesome_leader_arm \
  --display_data=true \
  --dataset.repo_id=arjunsinghyadav2/YOUR_DATASET_NAME \
  --dataset.num_episodes=20 \
  --dataset.single_task="Pick a blue lego and place it in blue cup"
```

**Controls during recording:**
- Right Arrow: end episode early
- Left Arrow: discard + redo current episode
- Esc: stop session

---

## 4. Replay a Recorded Episode

```bash
lerobot-replay \
  --robot.type=so101_follower --robot.port=COM3 --robot.id=my_awesome_follower_arm \
  --dataset.repo_id=arjunsinghyadav2/YOUR_DATASET_NAME \
  --dataset.episode=0
```

---

## 5. Training

### SmolVLA (full fine-tune, vision + action expert)

```bash
lerobot-train \
  --dataset.repo_id=arjunsinghyadav2/blue_sort_black_bg_colored_cups_v1_440ep \
  --dataset.image_transforms.enable=true \
  --rename_map='{"observation.images.wrist": "observation.images.camera1", "observation.images.top": "observation.images.camera2"}' \
  --policy.type=smolvla \
  --policy.empty_cameras=1 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.train_state_proj=true \
  --policy.device=cuda \
  --policy.optimizer_lr=3e-05 \
  --policy.scheduler_warmup_steps=2000 \
  --policy.scheduler_decay_steps=140000 \
  --policy.scheduler_decay_lr=1e-06 \
  --batch_size=8 \
  --steps=150000 \
  --num_workers=8 \
  --save_freq=25000 \
  --log_freq=100 \
  --eval_freq=0 \
  --seed=42 \
  --output_dir=outputs/train/smolvla_blue_sort_v1 \
  --job_name=smolvla_blue_sort_v1 \
  --policy.repo_id=arjunsinghyadav2/smolvla_blue_sort_v1 \
  --policy.push_to_hub=true \
  --wandb.enable=true \
  --wandb.project=lego-sort
```

**Key flags:**
- `train_expert_only=false` + `freeze_vision_encoder=false` = full fine-tune (~403M params, ~17GB VRAM)
- `train_expert_only=true` = action expert only (~100M params, ~6GB VRAM) **but vision encoder is also frozen**
- VRAM: ~17GB at batch=8

### Pi0.5 (expert-only, frozen PaliGemma-2B)

```bash
lerobot-train \
  --dataset.repo_id=arjunsinghyadav2/blue_sort_black_bg_colored_cups_v1_440ep \
  --dataset.image_transforms.enable=true \
  --rename_map='{"observation.images.wrist": "observation.images.camera1", "observation.images.top": "observation.images.camera2"}' \
  --policy.type=pi05 \
  --policy.paligemma_variant=gemma_2b \
  --policy.action_expert_variant=gemma_300m \
  --policy.dtype=bfloat16 \
  --policy.use_amp=true \
  --policy.gradient_checkpointing=true \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=false \
  --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}' \
  --policy.device=cuda \
  --policy.optimizer_lr=2.5e-05 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=90000 \
  --policy.scheduler_decay_lr=2.5e-06 \
  --batch_size=32 \
  --steps=100000 \
  --num_workers=8 \
  --save_freq=5000 \
  --log_freq=100 \
  --eval_freq=0 \
  --seed=42 \
  --output_dir=outputs/train/pi05_blue_sort_v1 \
  --job_name=pi05_blue_sort_v1 \
  --policy.repo_id=arjunsinghyadav2/pi05_blue_sort_v1 \
  --policy.push_to_hub=true \
  --wandb.enable=true \
  --wandb.project=lego-sort
```

**Key flags:**
- `gradient_checkpointing=true` + `dtype=bfloat16` + `use_amp=true` = required to fit on 4090
- Expert-only at batch=32: ~15GB VRAM
- Full fine-tune at batch=4: ~24GB VRAM (tight)

### Resume Training from Checkpoint

```bash
lerobot-train --resume=true \
  --config_path=outputs/train/YOUR_RUN/checkpoints/025000/pretrained_model/train_config.json
```

---

## 6. Evaluation (Sync)

Policy controls the arm autonomously (no teleop):

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=COM3 --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 30, fourcc: MJPG}, top: {type: intelrealsense, serial_number_or_name: 747612060071, width: 640, height: 480, fps: 30}}" \
  --policy.path=outputs/train/smolvla_blue_sort_v1/checkpoints/075000/pretrained_model \
  --policy.device=cuda \
  --dataset.repo_id=arjunsinghyadav2/eval_blue_sort \
  --dataset.num_episodes=5 \
  --dataset.single_task="pick a blue LEGO and place it into blue bin" \
  --dataset.episode_time_s=30 \
  --dataset.push_to_hub=false \
  --display_data=true
```

---

## 7. Evaluation (Async) — Smoother, No Idle Gaps

Decouples action execution from inference. The robot keeps moving while the GPU predicts the next action chunk.

### Terminal 1 — Policy Server (GPU):
```bash
python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080 --fps=30
```

### Terminal 2 — Robot Client:
```bash
python -m lerobot.async_inference.robot_client \
  --robot.type=so101_follower --robot.port=COM3 \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 30, fourcc: MJPG}, top: {type: intelrealsense, serial_number_or_name: 747612060071, width: 640, height: 480, fps: 30}}" \
  --robot.id=my_awesome_follower_arm \
  --task="pick a blue LEGO and place it into blue bin" \
  --server_address=127.0.0.1:8080 \
  --policy_type=smolvla \
  --pretrained_name_or_path=C:/Users/singh/Documents/RAS/Lerobot/lerobot/data_collection/outputs/train/smolvla_blue_sort_v2_lora/checkpoints/050000/pretrained_model \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average
```

**Tuning async:**
- `actions_per_chunk=50` — max chunk size (model outputs 50 actions per inference)
- `chunk_size_threshold=0.5` — request new inference when queue drops below 50%
- `aggregate_fn_name` — how overlapping chunks are blended: `weighted_average` (0.3 old + 0.7 new), `latest_only`, `average`, `conservative`

**Sync vs Async:**
- **Sync**: predict 50 actions → execute all → idle while predicting next chunk → repeat
- **Async**: execute actions from queue while GPU predicts next chunk in background. No idle gaps.

---

## 8. Merge Batch Datasets

If data was collected in multiple batches, merge into one:

```bash
python data_collection/merge_blue_sort.py           # merge locally
python data_collection/merge_blue_sort.py --push     # merge + push to HF
python data_collection/merge_blue_sort.py --dry-run  # preview
```

---

## 9. Key Gotchas

### Camera Rename Map
SmolVLA base expects `camera1/camera2/camera3`. Your dataset has `wrist/top`. Always pass:
```
--rename_map='{"observation.images.wrist": "observation.images.camera1", "observation.images.top": "observation.images.camera2"}'
--policy.empty_cameras=1
```

### train_expert_only Overrides freeze_vision_encoder
In SmolVLA, `train_expert_only=true` freezes the **entire VLM** (including vision encoder), regardless of `freeze_vision_encoder`. To train the vision encoder, **both** must be false.

### torch.compile Doesn't Work on Windows
Triton has cache directory bugs on Windows. Training runs in eager mode (no speed penalty, just no kernel fusion optimization).

### Pi0.5 Requires Custom Transformers
```bash
pip install -e ".[pi]"   # installs transformers@fix/lerobot_openpi branch
```
This may break SmolVLA. Switch back with `pip install -e ".[smolvla]"` if needed.

### LoRA Checkpoints
LoRA saves only the adapter (~3-9MB), not the full model. At load time, the base model is downloaded from HF cache and the adapter is applied on top.

---

## 10. Custom Training Scripts

| Script | Description |
|--------|-------------|
| `data_collection/train_blue_sort.py` | SmolVLA full fine-tune on blue sort dataset |
| `data_collection/train_pi05_blue_sort.py` | Pi0.5 training on blue sort dataset |
| `data_collection/run_eval_blue_sort.py` | Sync eval for SmolVLA |
| `data_collection/run_eval_pi05_blue_sort.py` | Sync eval for Pi0.5 |
| `data_collection/merge_blue_sort.py` | Merge batch datasets |
| `data_collection/plot_training.py` | Plot loss curves from log file |

---

## 11. VRAM Reference (RTX 4090, 24GB)

| Config | Params | VRAM |
|--------|--------|------|
| SmolVLA expert-only (batch=8) | 100M | ~6GB |
| SmolVLA full fine-tune (batch=8) | 403M | ~17GB |
| Pi0.5 expert-only (batch=32, bf16+grad_ckpt) | 693M | ~15GB |
| Pi0.5 full fine-tune (batch=4, bf16+grad_ckpt) | 3.6B | ~24GB |
