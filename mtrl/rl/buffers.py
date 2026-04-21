from collections import deque

from jaxtyping import Float, Int

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from mtrl.types import (
    Action,
    Observation,
    ReplayBufferCheckpoint,
    ReplayBufferSamples,
    Rollout,
    AtariReplayBufferSamples
)
from scipy.ndimage import gaussian_filter1d


class ReplayBuffer:
    """Replay buffer for the single-task environments.

    Each sampling step, it samples a batch for each task, returning a batch of shape (batch_size,).
    When pushing samples to the buffer, the buffer accepts inputs of arbitrary batch dimensions.
    """

    obs: Float[Observation, " buffer_size"]
    actions: Float[Action, " buffer_size"]
    rewards: Float[npt.NDArray, "buffer_size 1"]
    next_obs: Float[Observation, " buffer_size"]
    dones: Float[npt.NDArray, "buffer_size 1"]
    pos: int

    def __init__(
        self,
        capacity: int,
        env_obs_space: gym.Space,
        env_action_space: gym.Space,
        seed: int | None = None,
    ) -> None:
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)
        self._obs_shape = np.array(env_obs_space.shape).prod()
        self._action_shape = np.array(env_action_space.shape).prod()
        self.full = False

        self.reset()  # Init buffer

    def reset(self):
        """Reinitialize the buffer."""
        self.obs = np.zeros((self.capacity, self._obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self._action_shape), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self._obs_shape), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.pos = 0

    def checkpoint(self) -> ReplayBufferCheckpoint:
        return {
            "data": {
                "obs": self.obs,
                "actions": self.actions,
                "rewards": self.rewards,
                "next_obs": self.next_obs,
                "dones": self.dones,
                "pos": self.pos,
                "full": self.full,
            },
            "rng_state": self._rng.__getstate__(),
        }

    def load_checkpoint(self, ckpt: ReplayBufferCheckpoint) -> None:
        for key in ["data", "rng_state"]:
            assert key in ckpt

        for key in ["obs", "actions", "rewards", "next_obs", "dones", "pos", "full"]:
            assert key in ckpt["data"]
            setattr(self, key, ckpt["data"][key])

        self._rng.__setstate__(ckpt["rng_state"])

    def _advance_position(self, steps: int) -> None:
        """Advance the write pointer and update the full flag when wrapping."""

        if steps <= 0:
            return

        new_pos = self.pos + steps
        if new_pos >= self.capacity:
            self.full = True

        self.pos = new_pos % self.capacity

    def add(
        self,
        obs: Observation,
        next_obs: Observation,
        action: Action,
        reward: Float[npt.NDArray, " *batch"],
        done: Float[npt.NDArray, " *batch"],
    ) -> None:
        """Add a batch of samples to the buffer."""
        if obs.ndim >= 2:
            assert (
                obs.shape[0] == action.shape[0] == reward.shape[0] == done.shape[0]
            ), "Batch size must be the same for all transition data."

            # Flatten any batch dims
            flat_obs = obs.reshape(-1, obs.shape[-1])
            flat_next_obs = next_obs.reshape(-1, next_obs.shape[-1])
            flat_action = action.reshape(-1, action.shape[-1])
            flat_reward = reward.reshape(
                -1, 1
            )  # Keep the last dim as 1 for consistency
            flat_done = done.reshape(-1, 1)  # Keep the last dim as 1 for consistency

            # Calculate number of new transitions
            n_transitions = len(flat_obs)

            # Handle buffer wraparound
            start = self.pos
            indices = np.arange(start, start + n_transitions) % self.capacity

            # Store the transitions
            self.obs[indices] = flat_obs
            self.next_obs[indices] = flat_next_obs
            self.actions[indices] = flat_action
            self.rewards[indices] = flat_reward
            self.dones[indices] = flat_done

            self._advance_position(n_transitions)
        else:
            self.obs[self.pos] = obs.copy()
            self.actions[self.pos] = action.copy()
            self.next_obs[self.pos] = next_obs.copy()
            self.dones[self.pos] = done.copy().reshape(-1, 1)
            self.rewards[self.pos] = reward.copy().reshape(-1, 1)

            self._advance_position(1)

    def sample(
        self,
        batch_size: int | np.ndarray,
    ) -> ReplayBufferSamples:
        """Sample a batch with optional per-task batch size control.

        Args:
            batch_size: The total batch size. Must be divisible by number of tasks
                       if per_task_batch_sizes is None.
            per_task_batch_sizes: Optional array of shape [num_tasks] specifying
                                 how many samples to draw from each task.
                                 If provided, must sum to batch_size.

        Returns:
            ReplayBufferSamples: A batch of samples of batch shape (batch_size,).
        """
        buffer_size = self.pos if not self.full else self.capacity
        if isinstance(batch_size, np.ndarray):
            # Adaptive sampling - different batch sizes per task
            per_task_batch_sizes = np.asarray(batch_size, dtype=int)
            assert len(per_task_batch_sizes) == self.num_tasks, \
                f"per_task_batch_sizes length ({len(per_task_batch_sizes)}) must equal num_tasks ({self.num_tasks})"
            assert per_task_batch_sizes.sum() == (128 * self.num_tasks), \
                f"per_task_batch_sizes must sum to batch_size ({batch_size}), got {per_task_batch_sizes.sum()}"
            # Sample different amounts from each task
            all_obs = []
            all_actions = []
            all_next_obs = []
            all_dones = []
            all_rewards = []
            for task_idx in range(self.num_tasks):
                task_batch_size = per_task_batch_sizes[task_idx]
                if task_batch_size > 0:
                    # Sample indices for this task
                    sample_idx = self._rng.integers(
                        low=0,
                        high=buffer_size,
                        size=(task_batch_size,),
                    )
                    # Gather samples for this task
                    all_obs.append(self.obs[sample_idx, task_idx])
                    all_actions.append(self.actions[sample_idx, task_idx])
                    all_next_obs.append(self.next_obs[sample_idx, task_idx])
                    all_dones.append(self.dones[sample_idx, task_idx])
                    all_rewards.append(self.rewards[sample_idx, task_idx])
            # Concatenate all task samples
            batch = (
                np.concatenate(all_obs, axis=0),
                np.concatenate(all_actions, axis=0),
                np.concatenate(all_next_obs, axis=0),
                np.concatenate(all_dones, axis=0),
                np.concatenate(all_rewards, axis=0),
            )
        else:
            # Uniform sampling - same batch size for each task (original behavior)
            assert batch_size % self.num_tasks == 0, \
                f"Batch size must be divisible by the number of tasks. batch: {batch_size}, num tasks: {self.num_tasks}"
            single_task_batch_size = batch_size // self.num_tasks
            sample_idx = self._rng.integers(
                low=0,
                high=max(
                self.pos if not self.full else self.capacity, single_task_batch_size
                ),
                size=(single_task_batch_size,),
            )
            batch = (
                self.obs[sample_idx],
                self.actions[sample_idx],
                self.next_obs[sample_idx],
                self.dones[sample_idx],
                self.rewards[sample_idx],
            )

            mt_batch_size = single_task_batch_size * self.num_tasks
            batch = map(lambda x: x.reshape(mt_batch_size, *x.shape[2:]), batch)

        return ReplayBufferSamples(*batch)


class MultiTaskReplayBuffer:
    """Replay buffer for the multi-task benchmarks.
 
    Each sampling step, it samples a batch for each task, returning a batch of shape (batch_size, num_tasks,).
    When pushing samples to the buffer, the buffer only accepts inputs with batch shape (num_tasks,).
    """
 
    obs: Float[Observation, "buffer_size task"]
    actions: Float[Action, "buffer_size task"]
    rewards: Float[npt.NDArray, "buffer_size task 1"]
    next_obs: Float[Observation, "buffer_size task"]
    dones: Float[npt.NDArray, "buffer_size task 1"]
    pos: int
 
    def __init__(
        self,
        total_capacity: int,
        num_tasks: int,
        env_obs_space: gym.Space,
        env_action_space: gym.Space,
        seed: int | None = None,
        max_steps: int = 500,
        normalize_rewards: bool = False,
        reward_norm_eps: float = 1e-8,
        reward_filter: str | None = None,
        sigma: float | None = None,
        alpha: float | None = None,
        delta: float | None = None,
        filter_mode: str | None = None,
        # ── new: return-based normalization ──────────────────────
        returns_normalization: bool = False,
        discount: float = 0.99,
        v_max: float = 10.0,
    ) -> None:
        assert (
            total_capacity % num_tasks == 0
        ), "Total capacity must be divisible by the number of tasks."
        self.capacity = total_capacity // num_tasks
        self.num_tasks = num_tasks
        self._rng = np.random.default_rng(seed)
        self._obs_shape = np.array(env_obs_space.shape).prod()
        self._action_shape = np.array(env_action_space.shape).prod()
        self.full = False
 
        self.normalize_rewards = normalize_rewards
        self._min_rewards = np.full(num_tasks, np.inf, dtype=np.float64)
        self._max_rewards = np.full(num_tasks, -np.inf, dtype=np.float64)
        self.reward_norm_eps = reward_norm_eps
 
        # ── return-based normalization state ─────────────────────
        # Separate flag so both modes can coexist without confusion.
        # When use_return_normalization=True, sample() uses return-based
        # normalization instead of the per-step min-max approach.
        self.use_return_normalization = returns_normalization
        self.discount = discount
        self.v_max = v_max
        self.effective_horizon = 1.0 / (1.0 - discount)
 
        # Running min/max of observed *discounted returns* per task.
        # Initialised to ±inf so the first episode always updates them.
        self._returns_min = np.full(num_tasks, np.inf,  dtype=np.float64)
        self._returns_max = np.full(num_tasks, -np.inf, dtype=np.float64)
 
        # Per-task reward accumulators for the in-progress episode.
        # We collect raw rewards step-by-step and compute the return
        # backwards when the episode ends.
        self._episode_rewards: list[list[float]] = [[] for _ in range(num_tasks)]
 
        self.reset()
 
    # ── existing methods unchanged ────────────────────────────────
 
    def reset(self):
        """Reinitialize the buffer."""
        self.obs = np.zeros(
            (self.capacity, self.num_tasks, self._obs_shape), dtype=np.float32
        )
        self.actions = np.zeros(
            (self.capacity, self.num_tasks, self._action_shape), dtype=np.float32
        )
        self.rewards = np.zeros((self.capacity, self.num_tasks, 1), dtype=np.float32)
        self.next_obs = np.zeros(
            (self.capacity, self.num_tasks, self._obs_shape), dtype=np.float32
        )
        self.dones = np.zeros((self.capacity, self.num_tasks, 1), dtype=np.float32)
        self.pos = 0
 
    def checkpoint(self) -> ReplayBufferCheckpoint:
        return {
            "data": {
                "obs": self.obs,
                "actions": self.actions,
                "rewards": self.rewards,
                "next_obs": self.next_obs,
                "dones": self.dones,
                "pos": self.pos,
                "full": self.full,
                # persist normalization state so a resumed run doesn't
                # lose the return statistics it already computed
                "returns_min": self._returns_min,
                "returns_max": self._returns_max,
            },
            "rng_state": self._rng.__getstate__(),
        }
 
    def load_checkpoint(self, ckpt: ReplayBufferCheckpoint) -> None:
        for key in ["data", "rng_state"]:
            assert key in ckpt
        for key in ["obs", "actions", "rewards", "next_obs", "dones", "pos", "full"]:
            assert key in ckpt["data"]
            setattr(self, key, ckpt["data"][key])
        # restore normalization stats if present (backwards-compatible)
        self._returns_min = ckpt["data"].get("returns_min", self._returns_min)
        self._returns_max = ckpt["data"].get("returns_max", self._returns_max)
        self._rng.__setstate__(ckpt["rng_state"])
 
    def _advance_position(self, steps: int) -> None:
        if steps <= 0:
            return
        new_pos = self.pos + steps
        if new_pos >= self.capacity:
            self.full = True
        self.pos = new_pos % self.capacity
 
    # ── new: return normalizer helpers ───────────────────────────
 
    def _compute_discounted_returns(
        self,
        rewards: np.ndarray,   # (T,) raw rewards for one episode
        truncated: bool,
    ) -> tuple[float, float]:
        """
        Compute discounted return for every timestep in the episode,
        then return the (min, max) over those values.
 
        If the episode was truncated (time limit, not true terminal),
        we bootstrap the tail with mean_reward * effective_horizon
        instead of 0 — same approach as BRO.
        """
        T = len(rewards)
        values = np.zeros(T, dtype=np.float64)
        bootstrap = float(rewards.mean()) * self.effective_horizon if truncated else 0.0
        for i in reversed(range(T)):
            values[i] = rewards[i] + self.discount * bootstrap
            bootstrap  = values[i]
        return float(values.min()), float(values.max())
 
    def _update_return_stats(
        self,
        rewards: np.ndarray,    # (num_tasks,) reward at this step
        terminal: np.ndarray,   # (num_tasks,) bool
        truncated: np.ndarray,  # (num_tasks,) bool
    ) -> None:
        """
        Accumulate per-task rewards and, when an episode ends, compute
        the discounted return and update the running min/max.
        Called inside add() when use_return_normalization is True.
        """
        for task_idx in range(self.num_tasks):
            self._episode_rewards[task_idx].append(float(rewards[task_idx]))
 
            done = bool(terminal[task_idx]) or bool(truncated[task_idx])
            if done:
                ep_rewards = np.array(self._episode_rewards[task_idx], dtype=np.float64)
                v_min, v_max = self._compute_discounted_returns(
                    ep_rewards, truncated=bool(truncated[task_idx])
                )
                self._returns_min[task_idx] = min(self._returns_min[task_idx], v_min)
                self._returns_max[task_idx] = max(self._returns_max[task_idx], v_max)
                self._episode_rewards[task_idx] = []   # reset for next episode
 
    def _normalize_rewards_by_return(self, rewards: np.ndarray) -> np.ndarray:
        """
        Scale rewards so that observed discounted returns map to [-v_max, v_max].
 
        denominator = max(|returns_min|, returns_max) / v_max
 
        Dividing rewards by this denominator means:
            normalized_return = raw_return / denominator
                              ≈ raw_return * v_max / max_observed_return
        which keeps returns within [-v_max, v_max] by construction.
 
        Falls back to per-step min-max if return stats aren't available yet
        (i.e. no episode has completed).
        """
        no_data = np.isinf(self._returns_min) | np.isinf(self._returns_max)
 
        # denominator shape: (num_tasks,)
        denominator = np.where(
            self._returns_max >= np.abs(self._returns_min),
            self._returns_max,
            np.abs(self._returns_min),
        )
        denominator = denominator / self.v_max
 
        # avoid divide-by-zero on tasks with no completed episode yet
        denominator = np.where(no_data | (denominator < self.reward_norm_eps),
                               1.0, denominator)
 
        # rewards shape coming in: (sample, num_tasks, 1)
        # denominator needs to broadcast to that shape
        return rewards / denominator[np.newaxis, :, np.newaxis]
 
    # ── add: wire in return stat tracking ────────────────────────
 
    def add(
        self,
        obs: Float[Observation, " task"],
        next_obs: Float[Observation, " task"],
        action: Float[Action, " task"],
        reward: Float[npt.NDArray, " task"],
        done: Float[npt.NDArray, " task"],
        # ── new optional args for return normalization ──────────
        terminal: Float[npt.NDArray, " task"] | None = None,
        truncated: Float[npt.NDArray, " task"] | None = None,
    ) -> None:
        """Add a batch of samples to the buffer.
 
        terminal and truncated are only needed when use_return_normalization=True.
        If not provided, done is treated as terminal (truncated=False).
        """
        assert (
            obs.ndim == 2 and action.ndim == 2 and reward.ndim <= 2 and done.ndim <= 2
        )
        assert (
            obs.shape[0]
            == action.shape[0]
            == reward.shape[0]
            == done.shape[0]
            == self.num_tasks
        )
 
        self.obs[self.pos]     = obs.copy()
        self.actions[self.pos] = action.copy()
        self.next_obs[self.pos] = next_obs.copy()
        self.dones[self.pos]   = done.copy().reshape(-1, 1)
        self.rewards[self.pos] = reward.reshape(-1, 1).copy()
 
        # original per-step min-max tracking (unchanged)
        if self.normalize_rewards:
            self._min_rewards = np.minimum(self._min_rewards, reward)
            self._max_rewards = np.maximum(self._max_rewards, reward)
 
        # new: trajectory return tracking
        if self.use_return_normalization:
            _terminal  = terminal  if terminal  is not None else done
            _truncated = truncated if truncated is not None else np.zeros_like(done)
            self._update_return_stats(
                reward.flatten(),
                _terminal.flatten().astype(bool),
                _truncated.flatten().astype(bool),
            )
 
        self._advance_position(1)
 
    # ── sample: swap in return-based normalization ────────────────
 
    def single_task_sample(self, task_idx: int, batch_size: int) -> ReplayBufferSamples:
        assert task_idx < self.num_tasks, "Task index out of bounds."
        sample_idx = self._rng.integers(
            low=0,
            high=max(self.pos if not self.full else self.capacity, batch_size),
            size=(batch_size,),
        )
        batch = (
            self.obs[sample_idx][task_idx],
            self.actions[sample_idx][task_idx],
            self.next_obs[sample_idx][task_idx],
            self.dones[sample_idx][task_idx],
            self.rewards[sample_idx][task_idx],
        )
        return ReplayBufferSamples(*batch)
 
    def sample(self, batch_size: int | np.ndarray) -> ReplayBufferSamples:
        buffer_size = self.pos if not self.full else self.capacity
        if isinstance(batch_size, np.ndarray):
            assert len(batch_size) == self.num_tasks
            assert batch_size.sum() == (128 * self.num_tasks)
            all_obs, all_actions, all_next_obs, all_dones, all_rewards = [], [], [], [], []
            for task_idx in range(self.num_tasks):
                task_batch_size = batch_size[task_idx]
                if task_batch_size > 0:
                    sample_idx = self._rng.integers(
                        low=0,
                        high=max(self.pos if not self.full else self.capacity, task_batch_size),
                        size=(task_batch_size,),
                    )
                    all_obs.append(self.obs[sample_idx, task_idx])
                    all_actions.append(self.actions[sample_idx, task_idx])
                    all_next_obs.append(self.next_obs[sample_idx, task_idx])
                    all_dones.append(self.dones[sample_idx, task_idx])
                    all_rewards.append(self.rewards[sample_idx, task_idx])
            batch = (
                np.concatenate(all_obs,     axis=0),
                np.concatenate(all_actions, axis=0),
                np.concatenate(all_next_obs, axis=0),
                np.concatenate(all_dones,   axis=0),
                np.concatenate(all_rewards, axis=0),
            )
        else:
            assert batch_size % self.num_tasks == 0
            single_task_batch_size = batch_size // self.num_tasks
            sample_idx = self._rng.integers(
                low=0,
                high=max(self.pos if not self.full else self.capacity, single_task_batch_size),
                size=(single_task_batch_size,),
            )
 
            rewards = self.rewards[sample_idx]   # (sample, num_tasks, 1)
 
            if self.use_return_normalization:
                # ── new: scale rewards so returns fit in [-v_max, v_max]
                rewards = self._normalize_rewards_by_return(rewards)
            elif self.normalize_rewards:
                # original per-step min-max normalization (unchanged)
                mn = self._min_rewards[np.newaxis, :, np.newaxis]
                mx = self._max_rewards[np.newaxis, :, np.newaxis]
                rewards = (rewards - mn) / (mx - mn + self.reward_norm_eps)

            batch = (
                self.obs[sample_idx],
                self.actions[sample_idx],
                self.next_obs[sample_idx],
                self.dones[sample_idx],
                rewards,
            )
            mt_batch_size = single_task_batch_size * self.num_tasks
            batch = map(lambda x: x.reshape(mt_batch_size, *x.shape[2:]), batch)
        return ReplayBufferSamples(*batch)



class MultiTaskRolloutBuffer:
    num_rollout_steps: int
    num_tasks: int
    pos: int

    observations: Float[Observation, "task timestep"]
    actions: Float[Action, "task timestep"]
    rewards: Float[npt.NDArray, "task timestep 1"]
    dones: Float[npt.NDArray, "task timestep 1"]

    values: Float[npt.NDArray, "task timestep 1"]
    log_probs: Float[npt.NDArray, "task timestep 1"]
    means: Float[Action, "task timestep"]
    stds: Float[Action, "task timestep"]

    def __init__(
        self,
        num_rollout_steps: int,
        num_tasks: int,
        env_obs_space: gym.Space,
        env_action_space: gym.Space,
        seed: int | None = None,
    ) -> None:
        self.num_rollout_steps = num_rollout_steps
        self.num_tasks = num_tasks
        self._rng = np.random.default_rng(seed)
        self._obs_shape = np.array(env_obs_space.shape).prod()
        self._action_shape = np.array(env_action_space.shape).prod()
        self.reset()  # Init buffer

    def reset(self) -> None:
        """Reinitialize the buffer."""
        self.observations = np.zeros(
            (self.num_rollout_steps, self.num_tasks, self._obs_shape), dtype=np.float32
        )
        self.actions = np.zeros(
            (self.num_rollout_steps, self.num_tasks, self._action_shape),
            dtype=np.float32,
        )
        self.rewards = np.zeros(
            (self.num_rollout_steps, self.num_tasks, 1), dtype=np.float32
        )
        self.dones = np.zeros(
            (self.num_rollout_steps, self.num_tasks, 1), dtype=np.float32
        )

        self.log_probs = np.zeros(
            (self.num_rollout_steps, self.num_tasks, 1), dtype=np.float32
        )
        self.values = np.zeros_like(self.rewards)
        self.means = np.zeros_like(self.actions)
        self.stds = np.zeros_like(self.actions)
        self.pos = 0

    @property
    def ready(self) -> bool:
        return self.pos == self.num_rollout_steps

    def add(
        self,
        obs: Float[Observation, " task"],
        action: Float[Action, " task"],
        reward: Float[npt.NDArray, " task"],
        done: Float[npt.NDArray, " task"],
        value: Float[npt.NDArray, " task"] | None = None,
        log_prob: Float[npt.NDArray, " task"] | None = None,
        mean: Float[Action, " task"] | None = None,
        std: Float[Action, " task"] | None = None,
    ):
        # NOTE: assuming batch dim = task dim
        assert (
            obs.ndim == 2 and action.ndim == 2 and reward.ndim <= 2 and done.ndim <= 2
        )
        assert (
            obs.shape[0]
            == action.shape[0]
            == reward.shape[0]
            == done.shape[0]
            == self.num_tasks
        )

        self.observations[self.pos] = obs.copy()
        self.actions[self.pos] = action.copy()
        self.rewards[self.pos] = reward.copy().reshape(-1, 1)
        self.dones[self.pos] = done.copy().reshape(-1, 1)

        if value is not None:
            self.values[self.pos] = value.copy()
        if log_prob is not None:
            self.log_probs[self.pos] = log_prob.reshape(-1, 1).copy()
        if mean is not None:
            self.means[self.pos] = mean.copy()
        if std is not None:
            self.stds[self.pos] = std.copy()

        self.pos += 1

    def get(
        self,
        compute_advantages: bool,
        last_values: Float[npt.NDArray, " task"] | None = None,
        dones: Float[npt.NDArray, " task"] | None = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.97,
    ) -> Rollout:
        if compute_advantages:
            assert (
                last_values is not None
            ), "Must provide final value estimates if compute_advantages=True."
            assert (
                dones is not None
            ), "Must provide final value estimates if compute_advantages=True."
            assert not np.all(
                self.values == np.zeros_like(self.values)
            ), "Values must have been pushed to the buffer if compute_advantages=True."
            last_values = last_values.reshape(-1, 1)
            dones = dones.reshape(-1, 1)

            advantages = np.zeros_like(self.rewards)

            # Adapted from https://github.com/openai/baselines/blob/master/baselines/ppo2/runner.py
            last_gae_lamda = 0
            for timestep in reversed(range(self.num_rollout_steps)):
                if timestep == self.num_rollout_steps - 1:
                    next_nonterminal = 1.0 - self.dones
                    next_values = last_values
                else:
                    next_nonterminal = 1.0 - self.dones[timestep + 1]
                    next_values = self.values[timestep + 1]
                delta = (
                    self.rewards[timestep]
                    + next_nonterminal * gamma * next_values
                    - self.values[timestep]
                )
                advantages[timestep] = last_gae_lamda = (
                    delta + next_nonterminal * gamma * gae_lambda * last_gae_lamda
                )
            returns = (advantages + self.values).transpose(1, 0)
            advantages = advantages.transpose(1, 0)
        else:
            returns = None
            advantages = None

        return Rollout(
            self.observations.transpose(1, 0),
            self.actions.transpose(1, 0),
            self.rewards.transpose(1, 0),
            self.dones.transpose(1, 0),
            self.log_probs.transpose(1, 0),
            self.means.transpose(1, 0),
            self.stds.transpose(1, 0),
            self.values.transpose(1, 0),
            returns,
            advantages,
        )


class AtariMultiTaskReplayBuffer(MultiTaskReplayBuffer):

    obs: Float[Observation, "buffer_size task"]
    actions: Float[Action, "buffer_size task"]
    rewards: Float[npt.NDArray, "buffer_size task 1"]
    next_obs: Float[Observation, "buffer_size task"]
    truncates: Float[npt.NDArray, "buffer_size task 1"]
    dones: Float[npt.NDArray, "buffer_size task 1"]
    pos: int

    def __init__(
        self,
        total_capacity: int,
        num_tasks: int,
        env_obs_space: gym.Space,
        env_action_space: gym.Space,
        seed: int | None = None,
        max_steps: int = 500,
        reward_filter: str | None = None,
        sigma: float | None = None,
        alpha: float | None = None,
        delta: float | None = None,
        filter_mode: str | None = None,
        normalize_rewards: bool = False,
        reward_norm_eps: float = 1e-8,
        nstep: int = 3,
        gamma: float = 0.99,
    ) -> None:
        print(total_capacity%num_tasks)
        assert (
            total_capacity % num_tasks == 0
        ), "Total capacity must be divisible by the number of tasks."
        self.capacity = total_capacity // num_tasks
        self.num_tasks = num_tasks
        self._rng = np.random.default_rng(seed)
        self._obs_shape = env_obs_space
        self._action_shape = 1
        self.full = False
        self.normalize_rewards = normalize_rewards
        self.reward_norm_eps = reward_norm_eps
        self.nstep = nstep
        self.gamma = gamma

        self._nstep_buffer: deque = deque(maxlen=nstep)

        # Per-task min-max reward normalization
        self._min_rewards = np.full(num_tasks, np.inf, dtype=np.float64)
        self._max_rewards = np.full(num_tasks, -np.inf, dtype=np.float64)

        self.reset()

    def reset(self):
        """Reinitialize the buffer."""
        self.obs = np.zeros(
            (self.capacity, self.num_tasks, *self._obs_shape.shape), dtype=np.uint8
        )
        self.actions = np.zeros(
            (self.capacity, self.num_tasks), dtype=np.int32
        )
        self.rewards = np.zeros((self.capacity, self.num_tasks, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.num_tasks, *self._obs_shape.shape), dtype=np.uint8)
        self.dones = np.zeros((self.capacity, self.num_tasks, 1), dtype=np.float32)
        self.truncations = np.zeros((self.capacity, self.num_tasks, 1), dtype=np.float32)
        self.pos = 0

    def _get_nstep_info(self):
        """Compute n-step return from the current nstep_buffer (all tasks vectorized).

        Returns obs and action from the oldest transition, accumulated discounted
        reward, and the terminal next_obs/done for each task.
        """
        obs_0, next_obs_0, action_0, reward_0, truncate_0, done_0 = self._nstep_buffer[0]

        # Start from the last transition
        _, next_obs, _, reward, truncate, done = self._nstep_buffer[-1]
        reward = reward.copy().reshape(-1)
        next_obs = next_obs.copy()
        done = done.copy().reshape(-1)

        # Accumulate backwards through the window (excluding the last entry)
        for _, next_obs_i, _, rew_i, _, done_i in reversed(list(self._nstep_buffer)[:-1]):
            rew_i = rew_i.reshape(-1)
            done_i = done_i.reshape(-1)
            reward = rew_i + self.gamma * reward * (1 - done_i)
            # Where this step ended an episode, use its terminal next_obs/done
            mask = done_i.astype(bool)
            if mask.any():
                next_obs = np.where(mask[:, np.newaxis, np.newaxis, np.newaxis], next_obs_i, next_obs)
                done = np.where(mask, done_i, done)

        return obs_0, next_obs, action_0, reward, truncate_0.reshape(-1), done

    def add(
        self,
        obs,
        next_obs,
        action,
        reward,
        truncate,
        done,
    ) -> None:
        assert (
            obs.shape[0]
            == action.shape[0]
            == reward.shape[0]
            == truncate.shape[0]
            == done.shape[0]
            == self.num_tasks
        )

        self._nstep_buffer.append((
            obs.copy(),
            next_obs.copy(),
            action.copy(),
            reward.reshape(-1).copy(),
            truncate.reshape(-1).copy(),
            done.reshape(-1).copy(),
        ))

        if len(self._nstep_buffer) < self.nstep:
            return

        n_obs, n_next_obs, n_action, n_reward, n_truncate, n_done = self._get_nstep_info()

        self.obs[self.pos] = n_obs
        self.actions[self.pos] = n_action
        self.next_obs[self.pos] = n_next_obs
        self.dones[self.pos] = n_done.reshape(-1, 1)
        self.rewards[self.pos] = n_reward.reshape(-1, 1)
        self.truncations[self.pos] = n_truncate.reshape(-1, 1)

        if self.normalize_rewards:
            self._min_rewards = np.minimum(self._min_rewards, n_reward)
            self._max_rewards = np.maximum(self._max_rewards, n_reward)

        self._advance_position(1)

    def sample(
        self,
        batch_size: int | np.ndarray,
    ) -> AtariReplayBufferSamples:
        """Sample a balanced batch (equal samples per task).

        Args:
            batch_size: The total batch size. Must be divisible by number of tasks.

        Returns:
            AtariReplayBufferSamples: A batch of samples of batch shape (batch_size,).
        """

        assert batch_size % self.num_tasks == 0, \
                "Batch size must be divisible by the number of tasks."
        single_task_batch_size = batch_size // self.num_tasks
        sample_idx = self._rng.integers(
            low=0,
            high=max(
            self.pos if not self.full else self.capacity, single_task_batch_size
            ),
            size=(single_task_batch_size,),
        )


        rewards = self.rewards[sample_idx]
        if self.normalize_rewards:
            # Per-task min-max normalization (shape: [sample, task, 1])
            mn = self._min_rewards[np.newaxis, :, np.newaxis]
            mx = self._max_rewards[np.newaxis, :, np.newaxis]
            rewards = (rewards - mn) / (mx - mn + self.reward_norm_eps)

        batch = (
            self.obs[sample_idx],
            self.actions[sample_idx],
            self.next_obs[sample_idx],
            self.truncations[sample_idx],
            self.dones[sample_idx],
            rewards,
        )
        mt_batch_size = single_task_batch_size * self.num_tasks
        batch = map(lambda x: x.reshape(mt_batch_size, *x.shape[2:]), batch)

        return AtariReplayBufferSamples(*batch)

    def sample_unbalanced(
        self,
        batch_size: int,
    ) -> AtariReplayBufferSamples:
        """Sample with Dirichlet-weighted unbalanced task proportions.

        Uses np.random.dirichlet to vary per-task batch sizes, matching the
        Archive reference implementation.
        """
        buffer_size = self.pos if not self.full else self.capacity
        weights = self._rng.dirichlet([1] * self.num_tasks)
        task_sizes = np.floor(weights * batch_size).astype(np.int32)
        # Distribute remainder to ensure total is exactly batch_size
        remainder = batch_size - task_sizes.sum()
        if remainder > 0:
            top_indices = np.argsort(-weights)[:remainder]
            task_sizes[top_indices] += 1

        all_obs = []
        all_actions = []
        all_next_obs = []
        all_truncations = []
        all_dones = []
        all_rewards = []
        all_task_ids = []

        for i in range(self.num_tasks):
            n = task_sizes[i]
            if n > 0:
                idx = self._rng.integers(0, buffer_size, size=(n,))
                all_obs.append(self.obs[idx, i])
                all_actions.append(self.actions[idx, i])
                all_next_obs.append(self.next_obs[idx, i])
                all_truncations.append(self.truncations[idx, i])
                all_dones.append(self.dones[idx, i])
                r = self.rewards[idx, i]
                if self.normalize_rewards:
                    mn = self._min_rewards[i]
                    mx = self._max_rewards[i]
                    r = (r - mn) / (mx - mn + self.reward_norm_eps)
                all_rewards.append(r)
                all_task_ids.append(np.full(n, i, dtype=np.int32))

        return AtariReplayBufferSamples(
            np.concatenate(all_obs).astype(np.uint8),
            np.concatenate(all_actions).astype(np.int32),
            np.concatenate(all_next_obs).astype(np.uint8),
            np.concatenate(all_truncations).astype(np.float32),
            np.concatenate(all_dones).astype(np.float32),
            np.concatenate(all_rewards).astype(np.float32),
            np.concatenate(all_task_ids).astype(np.int32),
        )
