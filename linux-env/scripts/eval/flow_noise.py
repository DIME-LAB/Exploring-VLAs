"""Opt-in checkpoint-independent noise for paired native simulation rollouts."""
import hashlib
import json

VERSION = 'native-flow-pair-v1'


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


def _seed(parts):
    encoded = json.dumps([VERSION, *parts], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], 'big') & ((1 << 63) - 1)


def episode_identity(layout_seed, repeat_index):
    layout_seed = _integer(layout_seed, 'layout_seed')
    repeat_index = _integer(repeat_index, 'repeat_index')
    return dict(version=VERSION, layout_seed=layout_seed, repeat_index=repeat_index,
                episode_seed=_seed(['episode', layout_seed, repeat_index]))


def validate_identity(identity):
    if not isinstance(identity, dict):
        raise ValueError('Missing paired-noise identity')
    expected = episode_identity(identity.get('layout_seed'), identity.get('repeat_index'))
    if identity != expected:
        raise ValueError('Paired-noise identity does not match its layout/repeat')
    return expected


def prediction_noise(identity, action_step, batch_size, chunk_size, action_dim, device):
    """Use the production explicit-noise API without changing any global RNG.

    CPU-generated float32 N(0,1) noise is transferred to the policy device.
    The seed depends on the episode and executed-action index, never weights,
    wall time, warmup draws or preceding requests. Dimensions come from the
    actual processed batch and model config (including padded action dims).
    """
    import torch
    identity = validate_identity(identity)
    step = _integer(action_step, 'action_step')
    shape = tuple(_integer(x, 'noise dimension') for x in (batch_size, chunk_size, action_dim))
    if not all(shape):
        raise ValueError('Noise dimensions must be positive')
    seed = _seed(['chunk', identity['episode_seed'], step])
    generator = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.normal(0.0, 1.0, size=shape, generator=generator, dtype=torch.float32)
    proof = dict(**identity, action_step=step, chunk_seed=seed, shape=list(shape),
                 dtype='float32', generator_device='cpu', torch_version=str(torch.__version__),
                 noise_sha256=hashlib.sha256(noise.numpy().tobytes()).hexdigest())
    return noise.to(device=device), proof


def validate_episode_noise(done, identity, chunk_size, action_dim):
    """Refuse a dropped or altered proof anywhere in the production RPC path."""
    if done.get('flow_noise') != identity:
        raise ValueError('Actor paired-noise identity differs from requested episode')
    by_step = {}
    for collection in ('requests', 'chunks'):
        for item in done[collection]:
            proof = (item.get('native_rtc') or {}).get('flow_noise')
            if identity is None:
                if proof is not None:
                    raise ValueError('Unpaired episode returned paired-noise proof')
            else:
                validate_identity(identity)
                step = _integer(item['step'], 'action_step')
                expected = dict(**identity, action_step=step,
                                chunk_seed=_seed(['chunk', identity['episode_seed'], step]),
                                shape=[1, chunk_size, action_dim], dtype='float32', generator_device='cpu')
                if not isinstance(proof, dict) or any(proof.get(k) != v for k, v in expected.items()):
                    raise ValueError('Missing or incorrect per-request flow-noise proof')
                digest = proof.get('noise_sha256', '')
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest) or not isinstance(proof.get('torch_version'), str) or not proof['torch_version']:
                    raise ValueError('Missing noise hash or Torch version')
                if collection == 'requests':
                    if step in by_step: raise ValueError('Duplicate noise request step')
                    by_step[step] = proof
                elif by_step.pop(step, None) != proof:
                    raise ValueError('Chunk noise proof differs from request')
    if identity is not None and (not done['requests'] or by_step):
        raise ValueError('Paired requests and chunks do not match')
