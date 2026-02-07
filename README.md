# VLA CartPole: Vision-Language-Action Model for CartPole Control

A minimal, self-contained implementation of a **Vision-Language-Action (VLA)** agent that learns to balance a pole on a cart using raw pixel observations and a natural-language instruction. The project combines a CNN-based vision encoder, a bag-of-words language encoder, and an Advantage Actor-Critic (A2C) reinforcement learning algorithm into a single ~31k-parameter model.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Quick Start](#quick-start)
4. [File-by-File Reference](#file-by-file-reference)
5. [Model Architecture](#model-architecture)
6. [Reinforcement Learning Internals](#reinforcement-learning-internals)
7. [Environment Details](#environment-details)
8. [Data Formats and I/O](#data-formats-and-io)
9. [Reward Shaping](#reward-shaping)
10. [Visualization Guide](#visualization-guide)
11. [Hyperparameter Reference](#hyperparameter-reference)
12. [Reproducibility](#reproducibility)

---

## Project Overview

The goal is to train a small neural network that:

1. **Sees** a 256x64 RGB image (width x height) of a cart with a pole balanced on top.
2. **Reads** a natural-language instruction (e.g. `"keep the pole upright"`).
3. **Acts** by choosing to push the cart left or right at every timestep.

Training uses **A2C with Generalized Advantage Estimation (GAE)** across 1 000 parallel environments to produce stable gradient updates.

---

## Directory Structure

```
vla_cartpole/
├── __init__.py                     # Package exports (MiniCartPoleVisionEnv, MiniVLA, make_bow_instruction)
├── train.py                        # Training entry point
├── eval.py                         # Evaluation entry point
├── visualize.py                    # CLI for rollout visualization, GIF/PNG export
├── requirements.txt                # Python dependencies
│
├── env/
│   ├── __init__.py
│   └── cartpole_vision.py          # Gymnasium CartPole environment with pixel rendering
│
├── models/
│   ├── __init__.py
│   └── vla.py                      # MiniVLA actor-critic network
│
├── training/
│   ├── __init__.py
│   └── reinforce.py                # A2C training loop, GAE, parallel env rollouts
│
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py                # Greedy evaluation over N episodes
│
└── utils/
    ├── __init__.py
    ├── text.py                     # Bag-of-words text encoding (SHA-256 hashing)
    ├── visualization.py            # Matplotlib plotting helpers
    └── runtime.py                  # Device selection (CUDA/MPS/CPU), RNG seeding
```

---

## Quick Start

```bash
# Install dependencies
pip install -r vla_cartpole/requirements.txt

# Train (saves model.pth and checkpoints/)
python -m vla_cartpole.train

# Evaluate a trained model
python -m vla_cartpole.eval

# Visualize a rollout (interactive plot + optional GIF)
python -m vla_cartpole.visualize --save-gif episode.gif --save-plot summary.png
```

---

## File-by-File Reference

### `vla_cartpole/__init__.py`

Package initialization. Exports the three core objects so they can be imported directly:

```python
from vla_cartpole import MiniCartPoleVisionEnv, MiniVLA, make_bow_instruction
```

**Version:** `0.1.0`

---

### `vla_cartpole/env/cartpole_vision.py`

**Class: `MiniCartPoleVisionEnv(gymnasium.Env)`**

A self-contained CartPole environment that returns **256x64 RGB pixel observations** (width x height) instead of the standard 4-element state vector. All physics and rendering are implemented in pure NumPy (no external renderer).

**Observation space:** `Box(0, 255, shape=(64, 256, 3), dtype=uint8)`
**Action space:** `Discrete(2)` — `0` = push left, `1` = push right

**Physics state (internal, not exposed as observation):**

| Variable     | Description                    | Units   |
|-------------|-------------------------------|---------|
| `x`          | Cart horizontal position       | meters  |
| `x_dot`      | Cart velocity                  | m/s     |
| `theta`      | Pole angle (0 = upright)       | radians |
| `theta_dot`  | Pole angular velocity          | rad/s   |

**Physics constants:**

| Constant           | Value  |
|-------------------|--------|
| Gravity            | 9.8 m/s² |
| Cart mass          | 1.0 kg |
| Pole mass          | 0.1 kg |
| Pole half-length   | 0.5 m  |
| Applied force      | 10.0 N |
| Integration step   | 0.02 s |

**Termination conditions:**
- Pole angle exceeds ±0.5 rad (~29 degrees)
- Cart position exceeds ±1.2 m
- Episode reaches `max_episode_steps` (default 200 — truncation, not termination)

**Reset:** State is initialized with small random perturbations (±0.02 for position/velocity, ±0.05 for angle/angular velocity).

**Rendering:** A white 256x64 canvas (width x height) with a black rectangle (cart) and a red line (pole). Rendering uses NumPy array operations and a custom `_draw_line` rasterizer — no PIL or OpenGL dependencies.

**Example usage:**

```python
env = MiniCartPoleVisionEnv(max_episode_steps=200)
obs, info = env.reset(seed=42)         # obs.shape == (64, 256, 3)
obs, reward, terminated, truncated, info = env.step(1)  # push right
# info == {'x': 0.003, 'x_dot': 0.19, 'theta': -0.04, 'theta_dot': -0.28}
```

---

### `vla_cartpole/models/vla.py`

**Class: `MiniVLA(nn.Module)`**

The core neural network. It is a multimodal **actor-critic** model with three components: a vision encoder, a text encoder, and two output heads (policy and value).

**Constructor parameters:**

| Parameter      | Default | Description                                 |
|---------------|---------|---------------------------------------------|
| `num_actions`  | 2       | Discrete action count                       |
| `vocab_size`   | 1000    | Bag-of-words vocabulary dimension           |
| `embed_dim`    | 32      | Text embedding output dimension             |

**Forward input:**

| Input      | Shape              | Description                          |
|-----------|--------------------|--------------------------------------|
| `image`    | `[B, 64, 256, 3]`   | RGB uint8 image in HWC layout        |
| `bow_text` | `[B, vocab_size]`   | Bag-of-words encoded instruction     |

**Forward output (default):** Action probabilities, shape `[B, num_actions]`
**Forward output (`return_logits=True`):** Raw logits, shape `[B, num_actions]`

**Key methods:**

| Method                     | Returns                      | Use case                          |
|---------------------------|------------------------------|-----------------------------------|
| `forward(img, bow)`       | Action probabilities `[B,2]` | Greedy evaluation                 |
| `forward(img, bow, return_logits=True)` | Logits `[B,2]`  | Sampling during training          |
| `get_action_and_value(img, bow)` | `(logits, value)`      | Training (actor + critic)         |
| `get_value(img, bow)`     | Value `[B]`                  | GAE bootstrapping                 |

**Weight initialization:**
- All Conv2d and Linear layers: orthogonal initialization with ReLU gain
- Policy output layer: orthogonal with gain=**0.01** (small logits encourage uniform initial exploration)
- Value output layer: orthogonal with gain=**1.0**
- All biases: zeros

**Total parameters:** ~31,000

---

### `vla_cartpole/training/reinforce.py`

Contains the full A2C training loop and advantage estimation functions.

#### `compute_returns(rewards, gamma=0.99)`

Simple discounted return calculation (backward cumulative sum). Used by the legacy single-trajectory path.

```
G(t) = r(t) + gamma * G(t+1)
```

#### `compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95)`

Generalized Advantage Estimation for a single trajectory. Returns `(advantages, returns)` tensors.

#### `compute_gae_batch(rewards_t, values_t, next_value_t, dones_t, gamma=0.99, lam=0.95)`

Batched GAE for parallel environments. All inputs have shape `[T, N]` where `T` = rollout length and `N` = number of environments.

**GAE formula (per timestep, computed in reverse):**

```
delta(t) = r(t) + gamma * V(t+1) * (1 - done(t)) - V(t)
A(t)     = delta(t) + gamma * lambda * (1 - done(t)) * A(t+1)
Return(t)= A(t) + V(t)
```

Advantages are normalized across the entire `[T, N]` batch before computing the policy gradient.

#### `train_vla(...)` — Main Training Function

**Key parameters:**

| Parameter                    | Default   | Description                                   |
|-----------------------------|-----------|-----------------------------------------------|
| `num_episodes`               | 1000      | Total training episodes                       |
| `num_envs`                   | 1         | Parallel environments                         |
| `rollout_steps`              | 32        | Steps collected before each gradient update    |
| `lr`                         | 3e-4      | Shared encoder learning rate                  |
| `actor_lr`                   | same as lr| Policy head learning rate                     |
| `critic_lr`                  | same as lr| Value head learning rate                      |
| `gamma`                      | 0.99      | Discount factor                               |
| `gae_lambda`                 | 0.95      | GAE smoothing parameter                       |
| `entropy_coef`               | 0.01      | Entropy regularization weight                 |
| `value_coef`                 | 0.5       | Critic loss weight                            |
| `max_steps`                  | 200       | Max steps per episode                         |
| `checkpoint_every_steps`     | 1000      | Save numbered checkpoint every N env steps    |
| `checkpoint_latest_every_steps`| same    | Overwrite latest weights every N env steps    |
| `eval_every_steps`           | 1000      | Run evaluation every N env steps              |
| `seed`                       | None      | Base seed (env *i* gets seed+*i*)             |

**Training loop (one iteration):**

1. Collect `rollout_steps` transitions from all `num_envs` environments in parallel.
2. At each step, run `model.get_action_and_value()` on the batched observations to get logits and value estimates.
3. Sample actions from `Categorical(logits=logits)`. Store log-probs, values, entropies, rewards, done flags.
4. After the rollout, bootstrap the final value with `model.get_value()` on the last observation.
5. Compute GAE advantages and returns using `compute_gae_batch()`.
6. Normalize advantages: `(A - mean) / (std + 1e-8)`.
7. Compute losses:
   - **Actor loss:** `-mean(log_prob * advantage)`
   - **Critic loss:** `smooth_l1_loss(values, returns)`
   - **Entropy loss:** `-mean(entropy)` (negative because we *maximize* entropy)
   - **Total:** `actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss`
8. Backpropagate and clip gradients:
   - Actor/critic parameters: `max_norm=0.5`
   - Vision parameters: `max_norm=1.0`
9. Adam step with separate learning rates for shared, actor, and critic parameter groups.

**Optimizer setup:**

```python
optimizer = Adam([
    {'params': vision_and_text_params, 'lr': lr},       # shared encoders
    {'params': actor_params,           'lr': actor_lr},  # policy head
    {'params': critic_params,          'lr': critic_lr},  # value head
])
```

**Checkpointing:**
- Numbered checkpoints: `checkpoints/model_step_<N>.pth`
- Latest weights: overwritten at the specified path (e.g. `model.pth`)

**Returns:** `(episode_rewards, episode_lengths)` — lists of floats/ints across all completed episodes.

#### `collect_episode(...)` — Legacy Function

Single-trajectory collection for backward compatibility. Returns a dict with `observations`, `actions`, `rewards`, `log_probs`, and `episode_length`.

---

### `vla_cartpole/evaluation/evaluator.py`

**Function: `evaluate_model(env, model, instruction, ...)`**

Runs `num_episodes` greedy rollouts (no sampling — uses `argmax` over action probabilities) and reports statistics.

**Returns:**

```python
{
    'rewards':    [float, ...],  # per-episode total rewards
    'lengths':    [int, ...],    # per-episode step counts
    'avg_reward': float,
    'std_reward': float,
    'avg_length': float,
    'std_length': float,
}
```

**Example output:**

```
Episode 1: Reward=387.500, Length=200
Episode 2: Reward=390.123, Length=200
Episode 3: Reward=385.000, Length=200
...
Average Reward: 387.541 ± 2.12
Average Length: 200.0 ± 0.0
```

---

### `vla_cartpole/utils/text.py`

**Function: `make_bow_instruction(text, vocab_size=1000)`**

Converts a text string into a fixed-size bag-of-words tensor using **SHA-256 feature hashing**.

**Algorithm:**

1. Lowercase the input and split on whitespace.
2. For each word, compute `SHA256(word_bytes)`.
3. Take the first 8 bytes, interpret as a little-endian unsigned integer.
4. Modulo `vocab_size` to get the bucket index.
5. Increment the count at that index.
6. Return a `torch.Tensor` of shape `(vocab_size,)`.

**Why SHA-256?** Python's built-in `hash()` is randomized across interpreter restarts (unless `PYTHONHASHSEED=0`). SHA-256 produces identical hashes on every machine and every run.

**Example:**

```python
bow = make_bow_instruction("keep the pole upright")
# bow.shape == torch.Size([1000])
# bow is mostly zeros, with a few 1.0 entries at hashed indices
```

---

### `vla_cartpole/utils/visualization.py`

Three plotting functions for analysis and debugging.

#### `plot_observation_and_probs(obs, probs, action, reward, step_num=None)`

Side-by-side view of one timestep: the 256x64 RGB frame on the left, and a bar chart of P(left) vs P(right) on the right.

#### `plot_training_progress(rewards, lengths, window=20)`

Two-panel figure showing episode rewards and episode lengths over training, each with a 20-episode moving average overlay. Used at the end of `train.py`.

#### `plot_episode_summary(...)`

A comprehensive 3-row × 2-column figure summarizing a single rollout:

| Row | Left Panel | Right Panel |
|-----|-----------|-------------|
| 1   | Frame strip (8 sampled frames across the episode) | — (spans full width) |
| 2   | Reward vs. Value estimate over time | Policy probabilities P(left), P(right) with action scatter |
| 3   | Cart position `x` and pole angle `theta` over time | — (spans full width) |

---

### `vla_cartpole/utils/runtime.py`

#### `pick_device()`

Returns the best available PyTorch device in priority order: **CUDA > MPS (Apple Silicon) > CPU**. Lazy-imports `torch` to avoid loading it when the module is imported by non-training code.

#### `seed_everything(seed)`

Seeds Python's `random`, NumPy, and all PyTorch RNGs (CPU + all CUDA devices). Does **not** force deterministic CUDA kernels (which would require `torch.use_deterministic_algorithms(True)` and can degrade performance).

---

### `vla_cartpole/train.py`

Main training entry point. Wires together the environment, model, and training loop with production hyperparameters.

**Default configuration:**

```python
instruction       = "keep the pole upright"
num_episodes      = 6000
num_envs          = 1000        # parallel environments
rollout_steps     = 32
lr (shared)       = 5e-4
actor_lr          = 5e-4
critic_lr         = 1e-3        # higher for faster value convergence
gamma             = 0.99
gae_lambda        = 0.95
entropy_coef      = 0.002
value_coef        = 0.5
max_steps         = 200
seed              = 0
```

**What it does:**

1. Ensures `PYTHONHASHSEED=0` (restarts the process if needed).
2. Selects device, seeds RNGs.
3. Creates `MiniCartPoleVisionEnv` and `MiniVLA`, prints parameter count (~31k).
4. Calls `train_vla()` with the above hyperparameters.
5. Plots training curves via `plot_training_progress()`.
6. Prints final statistics (last 20 episodes, best episode).
7. Saves weights to `model.pth`.

---

### `vla_cartpole/eval.py`

Loads `model.pth`, runs 5 greedy evaluation episodes (max 100 steps each), and prints mean/std of rewards and lengths.

---

### `vla_cartpole/visualize.py`

CLI tool for post-training analysis. Runs a single episode, captures per-step data, and produces rich visualizations.

**CLI arguments:**

| Argument             | Default                  | Description                         |
|---------------------|--------------------------|-------------------------------------|
| `--weights`          | `model.pth`              | Path to weights file                |
| `--checkpoint-step`  | None                     | Load `checkpoints/model_step_<N>.pth` |
| `--instruction`      | `"keep the pole upright"` | Text instruction                    |
| `--max-steps`        | 200                      | Episode length cap                  |
| `--seed`             | None                     | Environment seed                    |
| `--save-plot`        | None                     | Save summary plot to PNG            |
| `--save-gif`         | None                     | Save episode as animated GIF        |
| `--gif-fps`          | 20                       | GIF frame rate                      |
| `--no-show`          | False                    | Suppress matplotlib windows         |

**Per-step data captured:**

| Data           | Type    | Description                             |
|---------------|--------|-----------------------------------------|
| Frame          | ndarray | 64x256x3 RGB observation                 |
| Action         | int     | 0 (left) or 1 (right)                   |
| Reward         | float   | Shaped reward for this step             |
| Value          | float   | Critic's value estimate V(s)            |
| P(left)        | float   | Policy probability of action 0          |
| P(right)       | float   | Policy probability of action 1          |
| x              | float   | Cart position from `info`               |
| theta          | float   | Pole angle from `info`                  |

**Example:**

```bash
# Interactive plot
python -m vla_cartpole.visualize

# Save outputs, no display
python -m vla_cartpole.visualize --save-gif rollout.gif --save-plot summary.png --no-show

# Visualize a specific checkpoint
python -m vla_cartpole.visualize --checkpoint-step 1000000
```

---

### `vla_cartpole/requirements.txt`

```
gymnasium
matplotlib
numpy
torch          (with CUDA 12.4 index)
torchvision
torchaudio
pillow
ipykernel
jupyter
```

---

## Model Architecture

```
                   ┌─────────────────────────┐
                   │   256x64x3 RGB Image     │
                   └────────────┬────────────┘
                                │
                   ┌────────────▼────────────┐
                   │     Vision Encoder       │
                   │                          │
                   │  Conv2d(3→32, 3x3, s=2)  │  256x64 → 128x32
                   │  ReLU                    │
                   │  Conv2d(32→64, 3x3, s=2) │  128x32 → 64x16
                   │  ReLU                    │
                   │  Conv2d(64→64, 3x3, s=2) │  64x16 → 32x8
                   │  ReLU                    │
                   │  Flatten (64*32*8=16384) │
                   │  Linear(16384→256)       │
                   │  ReLU                    │
                   └────────────┬────────────┘
                                │ 256-dim
                                │
 ┌──────────────┐               │
 │  Instruction │               │
 │  (text)      │               │
 └──────┬───────┘               │
        │                       │
 ┌──────▼───────┐               │
 │  BoW Encode  │               │
 │  (SHA-256    │               │
 │   hashing)   │               │
 └──────┬───────┘               │
        │ 1000-dim              │
 ┌──────▼───────┐               │
 │ Text Encoder │               │
 │ Linear(1000  │               │
 │   → 32)      │               │
 │ ReLU         │               │
 └──────┬───────┘               │
        │ 32-dim                │
        │                       │
        └───────┬───────────────┘
                │
       ┌────────▼────────┐
       │   Concatenate   │
       │  [256 + 32]     │
       │   = 288-dim     │
       └───┬─────────┬───┘
           │         │
   ┌───────▼───┐ ┌───▼───────┐
   │   Actor   │ │  Critic   │
   │ (Policy)  │ │ (Value)   │
   │           │ │           │
   │ Lin(288   │ │ Lin(288   │
   │   →64)    │ │   →64)    │
   │ Tanh      │ │ Tanh      │
   │ Lin(64→2) │ │ Lin(64→1) │
   └───────┬───┘ └───┬───────┘
           │         │
           ▼         ▼
     Action Logits  V(s)
      [B, 2]       [B]
```

**Forward pass detail:**

1. The raw `[B, 64, 256, 3]` uint8 image is normalized to `[0, 1]` and permuted to NCHW.
2. Three strided convolutions halve the spatial dimensions at each layer (256x64 → 128x32 → 64x16 → 32x8), producing 64 feature maps.
3. The 32x8x64 feature volume is flattened to 16,384 and projected to a 256-dim vector.
4. The instruction string is hashed to a 1000-dim BoW vector and projected to 32 dims.
5. Vision and text features are concatenated into a 288-dim fused representation.
6. The **actor head** maps 288→64→2 (with Tanh activation) to produce action logits.
7. The **critic head** maps 288→64→1 (with Tanh activation) to produce a scalar value estimate.

---

## Reinforcement Learning Internals

### Algorithm: Advantage Actor-Critic (A2C)

A2C is a synchronous policy-gradient method that uses a learned value function as a variance-reducing baseline.

**Core idea:**
- The **actor** (policy network) learns a distribution over actions: `pi(a|s)`.
- The **critic** (value network) learns the expected return from a state: `V(s)`.
- The **advantage** `A(s,a) = Q(s,a) - V(s)` tells us whether an action was better or worse than average. We estimate it using GAE rather than computing Q directly.

### Generalized Advantage Estimation (GAE)

GAE smooths the advantage estimate using an exponential weighting controlled by `lambda`:

```
delta(t)     = r(t) + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
A^GAE(t)     = delta(t) + (gamma * lambda) * (1 - done_t) * A^GAE(t+1)
Return(t)    = A^GAE(t) + V(s_t)
```

- `lambda=0` reduces to the 1-step TD advantage (low variance, high bias).
- `lambda=1` reduces to the Monte Carlo advantage (high variance, low bias).
- `lambda=0.95` (the default) is a practical middle ground.

### Parallel Environment Collection

Instead of collecting one trajectory at a time, the training loop runs **1,000 environments simultaneously**:

```
For each of 32 rollout steps:
    obs_batch  = stack observations from all 1000 envs     → [1000, 64, 256, 3]
    logits, V  = model.get_action_and_value(obs_batch)     → [1000, 2], [1000]
    actions    = Categorical(logits).sample()               → [1000]
    step all 1000 envs with their respective actions
    store: log_probs[t], values[t], rewards[t], dones[t]   → each [1000]
```

This produces `32 * 1000 = 32,000` transitions per gradient update, leading to much more stable learning than single-trajectory REINFORCE.

### Loss Computation

```
L_actor   = -mean( log pi(a_t | s_t)  *  A^GAE(t) )     # policy gradient
L_critic  = smooth_L1( V(s_t),  Return(t) )               # value regression
L_entropy = -mean( H[pi(. | s_t)] )                       # entropy bonus (negated)

L_total   = L_actor  +  0.5 * L_critic  +  0.002 * L_entropy
```

- **Actor loss** pushes the policy toward actions that had positive advantage.
- **Critic loss** (Huber/smooth-L1) trains the value function to predict returns.
- **Entropy loss** prevents the policy from collapsing to a deterministic action too early.

### Gradient Clipping

Separate max-norm clipping is applied to different parts of the network:

| Parameter Group | Max Norm |
|----------------|----------|
| Actor head      | 0.5      |
| Critic head     | 0.5      |
| Vision encoder  | 1.0      |

### Optimizer

A single Adam optimizer with three parameter groups, each with its own learning rate:

| Group              | Default LR | Purpose                              |
|-------------------|-----------|--------------------------------------|
| Shared (vision+text) | 5e-4    | Feature extraction                   |
| Actor               | 5e-4    | Policy learning                      |
| Critic              | 1e-3    | Value learning (higher for stability)|

The critic learning rate is set 2x higher than the actor rate. This is a common practice: the value function needs to converge faster so it can provide a reliable baseline for policy gradient updates.

---

## Environment Details

### Physics Model

The environment implements a classic inverted pendulum (CartPole). The cart moves along a 1D track and a rigid pole is attached to the top of the cart by an unactuated joint.

**Equations of motion (Euler integration, dt=0.02s):**

```
total_mass       = cart_mass + pole_mass
pole_mass_length = pole_mass * pole_half_length

temp      = (F + pole_mass_length * theta_dot^2 * sin(theta)) / total_mass
theta_acc = (g * sin(theta) - cos(theta) * temp) /
            (pole_half_length * (4/3 - pole_mass * cos(theta)^2 / total_mass))
x_acc     = temp - pole_mass_length * theta_acc * cos(theta) / total_mass

x         += dt * x_dot
x_dot     += dt * x_acc
theta     += dt * theta_dot
theta_dot += dt * theta_acc
```

### Rendering

The 256x64 RGB observation (width x height) is rendered entirely in NumPy:

- **Background:** white (255, 255, 255)
- **Cart:** black rectangle, 16 pixels wide, 11 pixels tall, centered at row 40-50
- **Pole:** red line (255, 0, 0), 20 pixels long, 3 pixels thick, drawn from the top-center of the cart at angle `theta`
- **Cart position mapping:** the physical range `[-1.2, 1.2]` maps to pixel columns `[32, 224]` (±96 pixels from center 128)

---

## Data Formats and I/O

### Training Data Flow

```
                          Per rollout step (×32 steps, ×1000 envs)
                          ─────────────────────────────────────────
Environment   ──obs──►   obs_batch     [1000, 64, 256, 3]  uint8
                          │
                          ▼
Model         ──────►    logits        [1000, 2]           float32
                          value         [1000]              float32
                          │
                          ▼
Sampling      ──────►    actions       [1000]              int64
                          log_probs     [1000]              float32
                          entropy       [1000]              float32
                          │
                          ▼
Environment   ──step──►  rewards       [1000]              float32
                          dones         [1000]              float32 (0 or 1)

              After 32 steps, stack to [32, 1000]:
              ─────────────────────────────────────
              log_probs_t   [32, 1000]
              values_t      [32, 1000]
              rewards_t     [32, 1000]
              dones_t       [32, 1000]
              entropies_t   [32, 1000]

              GAE computation → advantages_t [32, 1000], returns_t [32, 1000]
```

### Checkpoint Format

Checkpoints are PyTorch `state_dict` files (`.pth`) containing all model weights:

```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model.load_state_dict(torch.load('model.pth', map_location=device))
```

### Evaluation Output

```python
{
    'rewards':    [387.5, 390.1, 385.0, 392.3, 388.7],
    'lengths':    [200, 200, 200, 200, 200],
    'avg_reward': 388.72,
    'std_reward': 2.45,
    'avg_length': 200.0,
    'std_length': 0.0,
}
```

---

## Reward Shaping

The reward function provides a dense signal to accelerate learning:

```python
if not terminated:
    alive_reward   = 1.0                                          # survive
    angle_bonus    = 0.5 * (1.0 - abs(theta) / theta_threshold)  # stay upright
    position_bonus = 0.5 * (1.0 - abs(x) / x_threshold)          # stay centered
    reward = alive_reward + angle_bonus + position_bonus
else:
    reward = 0.0
```

| Component        | Min  | Max  | When maximized               |
|-----------------|------|------|------------------------------|
| Alive reward     | 1.0  | 1.0  | Always (if not terminated)   |
| Angle bonus      | 0.0  | 0.5  | Pole perfectly vertical      |
| Position bonus   | 0.0  | 0.5  | Cart perfectly centered      |
| **Total/step**   | **1.0** | **2.0** |                           |

For a perfect 200-step episode: max total reward = **400.0**.

---

## Visualization Guide

### Training Progress (`plot_training_progress`)

Generated automatically at the end of training. Two side-by-side plots:

- **Left:** Episode reward over training with a 20-episode moving average (red line).
- **Right:** Episode length over training with a 20-episode moving average.

A successful training run shows rewards climbing from ~20-50 (random policy) to ~350-400 (converged), and episode lengths increasing from ~10-20 to 200 (the max).

### Episode Summary (`plot_episode_summary`)

A 3-row diagnostic figure produced by `visualize.py`:

**Row 1 — Frame Strip:** 8 frames sampled uniformly across the episode, concatenated horizontally. Shows the visual progression of the cart and pole.

**Row 2, Left — Reward vs Value:** Time series comparing the actual per-step reward (blue) against the critic's value estimate (orange). A well-trained critic should track the reward curve. If the value consistently overshoots or undershoots, the critic may need more training.

**Row 2, Right — Policy Probabilities:** P(left) and P(right) over time with a scatter plot of the actions actually taken. A trained policy typically oscillates between left and right to maintain balance, with probabilities near 0.5 when the pole is centered.

**Row 3 — State:** Cart position `x` and pole angle `theta` over time. A trained agent keeps both close to zero. Oscillations are normal — the agent constantly corrects small perturbations.

### GIF Export

```bash
python -m vla_cartpole.visualize --save-gif episode.gif --gif-fps 20
```

Produces an animated GIF of the 256x64 observations at 20 frames per second, showing the cart and pole for the full episode.

---

## Hyperparameter Reference

All defaults as configured in `train.py`:

| Hyperparameter         | Value    | Category        | Notes                                           |
|-----------------------|----------|-----------------|--------------------------------------------------|
| `num_episodes`         | 6000     | Training        | Total episodes across all envs                   |
| `num_envs`             | 1000     | Training        | Parallel environments per update                 |
| `rollout_steps`        | 32       | Training        | Steps before each gradient update                |
| `lr` (shared)          | 5e-4     | Optimization    | Vision + text encoder learning rate              |
| `actor_lr`             | 5e-4     | Optimization    | Policy head learning rate                        |
| `critic_lr`            | 1e-3     | Optimization    | Value head learning rate (2x shared)             |
| `gamma`                | 0.99     | RL              | Discount factor                                  |
| `gae_lambda`           | 0.95     | RL              | GAE bias-variance trade-off                      |
| `entropy_coef`         | 0.002    | RL              | Exploration incentive weight                     |
| `value_coef`           | 0.5      | RL              | Critic loss weight in total loss                 |
| `max_steps`            | 200      | Environment     | Episode length cap                               |
| `vocab_size`           | 1000     | Model           | BoW hashing buckets                              |
| `embed_dim`            | 32       | Model           | Text embedding dimension                         |
| `num_actions`          | 2        | Model           | Left / Right                                     |
| `grad_clip_actor`      | 0.5      | Optimization    | Max gradient norm for actor                      |
| `grad_clip_critic`     | 0.5      | Optimization    | Max gradient norm for critic                     |
| `grad_clip_vision`     | 1.0      | Optimization    | Max gradient norm for vision encoder             |
| `seed`                 | 0        | Reproducibility | Base seed for all RNGs                           |

---

## Reproducibility

The project takes several steps to ensure reproducible results:

1. **`PYTHONHASHSEED=0`** — Enforced at process startup in `train.py`, `eval.py`, and `visualize.py`. If the environment variable is not set, the script re-executes itself with it set. This ensures SHA-256-based BoW encoding produces identical indices across runs.

2. **`seed_everything(seed)`** — Seeds Python `random`, NumPy, and PyTorch (CPU + all CUDA devices).

3. **Per-environment seeding** — Environment *i* in the parallel batch is seeded with `base_seed + i`, ensuring diverse but deterministic initial conditions.

4. **Limitation:** GPU non-determinism is **not** addressed. CUDA operations like `atomicAdd` in convolutions can introduce small floating-point variations. To fully eliminate this, you would need `torch.use_deterministic_algorithms(True)`, which is not enabled by default due to performance cost.
