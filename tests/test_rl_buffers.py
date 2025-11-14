import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("gymnasium")
from gymnasium.spaces import Box

from mtrl.rl.buffers import ReplayBuffer


def create_replay_buffer(capacity: int) -> ReplayBuffer:
    obs_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    return ReplayBuffer(
        capacity=capacity,
        env_obs_space=obs_space,
        env_action_space=action_space,
        seed=0,
    )


def test_replay_buffer_sets_full_flag_after_single_transitions() -> None:
    buffer = create_replay_buffer(capacity=4)

    for idx in range(buffer.capacity):
        value = float(idx)
        obs = np.full((3,), value, dtype=np.float32)
        next_obs = np.full((3,), value + 1.0, dtype=np.float32)
        action = np.full((2,), -value, dtype=np.float32)
        reward = np.array([value], dtype=np.float32)
        done = np.array([idx % 2], dtype=np.float32)

        buffer.add(obs=obs, next_obs=next_obs, action=action, reward=reward, done=done)

    assert buffer.full is True
    assert buffer.pos == 0


def test_replay_buffer_sets_full_flag_after_batched_transitions() -> None:
    buffer = create_replay_buffer(capacity=5)

    batch_size = buffer.capacity
    obs = np.arange(batch_size * 3, dtype=np.float32).reshape(batch_size, 3)
    next_obs = obs + 1.0
    action = np.arange(batch_size * 2, dtype=np.float32).reshape(batch_size, 2)
    reward = np.arange(batch_size, dtype=np.float32)
    done = np.zeros(batch_size, dtype=np.float32)

    buffer.add(obs=obs, next_obs=next_obs, action=action, reward=reward, done=done)

    assert buffer.full is True
    assert buffer.pos == 0

    buffer.add(
        obs=obs[:2],
        next_obs=next_obs[:2],
        action=action[:2],
        reward=reward[:2],
        done=done[:2],
    )

    assert buffer.full is True
    assert buffer.pos == 2
