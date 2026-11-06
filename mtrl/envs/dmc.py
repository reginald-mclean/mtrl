from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import override

import numpy as np
import numpy.typing as npt
import gymnasium as gym
from dm_control import suite

from mtrl.types import Agent
from .base import EnvConfig


# ---------------------------------------------------------------------------
# Low-level gymnasium wrapper around dm_control.suite
# ---------------------------------------------------------------------------

class DMCSuiteEnv(gym.Env):
    """Thin stateless gymnasium wrapper around a dm_control suite environment."""

    metadata: dict = {"render_modes": []}

    def __init__(self, domain: str, task: str) -> None:
        self._env = suite.load(domain, task)

        obs_spec = self._env.observation_spec()
        obs_dim = sum(
            int(np.prod(v.shape)) if v.shape else 1 for v in obs_spec.values()
        )
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float64
        )

        act_spec = self._env.action_spec()
        self.action_space = gym.spaces.Box(
            act_spec.minimum.astype(np.float64),
            act_spec.maximum.astype(np.float64),
            dtype=np.float64,
        )

    def _flatten_obs(self, time_step) -> npt.NDArray[np.float64]:
        return np.concatenate([
            np.atleast_1d(np.array(v, dtype=np.float64)).flatten()
            for v in time_step.observation.values()
        ])

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[npt.NDArray[np.float64], dict]:
        super().reset(seed=seed)
        ts = self._env.reset()
        return self._flatten_obs(ts), {}

    def step(
        self, action: npt.NDArray
    ) -> tuple[npt.NDArray[np.float64], float, bool, bool, dict]:
        ts = self._env.step(action)
        obs = self._flatten_obs(ts)
        reward = float(ts.reward) if ts.reward is not None else 0.0
        terminated = ts.last()
        return obs, reward, terminated, False, {}

    def close(self) -> None:
        self._env.close()


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

class VelocityGoalWrapper(gym.Wrapper):
    """Replaces reward with -|v - v_target| and appends v_target to obs.

    Forward velocity is read from obs[forward_vel_obs_idx], which avoids
    having to reach into dm_control physics objects.
    """

    def __init__(
        self,
        env: DMCSuiteEnv,
        v_min: float,
        v_max: float,
        forward_vel_obs_idx: int,
        success_threshold: float = 0.5,
    ) -> None:
        super().__init__(env)
        self._v_min = v_min
        self._v_max = v_max
        self._forward_vel_obs_idx = forward_vel_obs_idx
        self._success_threshold = success_threshold
        self._v_target: float = (v_min + v_max) / 2.0

        base_dim = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(base_dim + 1,), dtype=np.float64
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        if self._v_min == self._v_max:
            self._v_target = self._v_min
        else:
            self._v_target = float(self.np_random.uniform(self._v_min, self._v_max))
        return np.append(obs, self._v_target), info

    def step(self, action):
        obs, _native_reward, terminated, truncated, info = self.env.step(action)
        v = float(obs[self._forward_vel_obs_idx])
        reward = -abs(v - self._v_target)
        info = {**info, "forward_velocity": v, "v_target": self._v_target,
                "success": float(abs(v - self._v_target) < self._success_threshold)}
        return np.append(obs, self._v_target), reward, terminated, truncated, info

    @property
    def v_target(self) -> float:
        return self._v_target


class OneHotTaskWrapper(gym.ObservationWrapper):
    """Appends a one-hot task encoding to the observation."""

    def __init__(self, env: gym.Env, task_idx: int, num_tasks: int) -> None:
        super().__init__(env)
        self._one_hot = np.zeros(num_tasks, dtype=np.float64)
        self._one_hot[task_idx] = 1.0

        base_dim = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(base_dim + num_tasks,), dtype=np.float64
        )

    def observation(self, obs: npt.NDArray) -> npt.NDArray[np.float64]:
        return np.append(obs, self._one_hot)


# ---------------------------------------------------------------------------
# Forward velocity obs indices (position in flattened obs)
# ---------------------------------------------------------------------------

# Walker: [orientations(14), height(1), velocity(9)] → velocity[0] = idx 15
_WALKER_FWD_VEL_IDX = 15
# Hopper: [position(6), velocity(7), touch(2)] → velocity[0] = idx 6
_HOPPER_FWD_VEL_IDX = 6


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _evaluate_locomotion(
    envs: gym.vector.VectorEnv,
    agent: Agent,
    num_episodes: int,
    task_names: list[str],
) -> tuple[float, float, dict[str, float]]:
    """Runs num_episodes per task; success = fraction of steps near v_target."""
    num_tasks = envs.num_envs
    episode_returns = [[] for _ in range(num_tasks)]
    episode_successes = [[] for _ in range(num_tasks)]

    episodes_done = [0] * num_tasks
    current_return = np.zeros(num_tasks)
    current_success_steps = np.zeros(num_tasks)
    current_length = np.zeros(num_tasks, dtype=int)

    obs, _ = envs.reset()
    task_ids = np.arange(num_tasks)

    while any(n < num_episodes for n in episodes_done):
        actions = agent.eval_action(obs, task_ids)
        obs, rewards, terminations, truncations, infos = envs.step(actions)
        dones = terminations | truncations

        current_return += rewards
        if "success" in infos:
            current_success_steps += np.array(infos["success"])
        current_length += 1

        for i in range(num_tasks):
            if dones[i] and episodes_done[i] < num_episodes:
                episode_returns[i].append(float(current_return[i]))
                episode_successes[i].append(
                    float(current_success_steps[i] / max(current_length[i], 1))
                )
                episodes_done[i] += 1
                current_return[i] = 0.0
                current_success_steps[i] = 0.0
                current_length[i] = 0

    per_task_success = {
        name: float(np.mean(episode_successes[i]))
        for i, name in enumerate(task_names)
    }
    mean_success = float(np.mean(list(per_task_success.values())))
    mean_return = float(np.mean([np.mean(r) for r in episode_returns]))
    return mean_success, mean_return, per_task_success


# ---------------------------------------------------------------------------
# Task config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DMCTaskConfig:
    domain: str
    task: str
    task_name: str
    v_min: float
    v_max: float
    success_threshold: float = 0.5


# ---------------------------------------------------------------------------
# EnvConfig implementations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DMCLocomotionMTConfig(EnvConfig):
    """Multi-task DMC locomotion with per-episode velocity goals.

    Args:
        tasks: ordered tuple of task configs (one env per task)
        forward_vel_obs_idx: index in the flattened native obs that gives
            forward velocity (domain-specific, set by subclasses)
        num_eval_episodes: episodes per task during evaluation
    """

    tasks: tuple[DMCTaskConfig, ...] = ()
    forward_vel_obs_idx: int = _WALKER_FWD_VEL_IDX
    num_eval_episodes: int = 10

    @cached_property
    @override
    def observation_space(self) -> gym.Space:
        sample = DMCSuiteEnv(self.tasks[0].domain, self.tasks[0].task)
        native_dim = sample.observation_space.shape[0]
        sample.close()
        # native obs + v_target (1) + one_hot (num_tasks if use_one_hot)
        total = native_dim + 1 + (len(self.tasks) if self.use_one_hot else 0)
        return gym.spaces.Box(-np.inf, np.inf, shape=(total,), dtype=np.float64)

    @cached_property
    @override
    def action_space(self) -> gym.Space:
        sample = DMCSuiteEnv(self.tasks[0].domain, self.tasks[0].task)
        space = sample.action_space
        sample.close()
        return space

    def _make_single_env(self, task_cfg: DMCTaskConfig, task_idx: int) -> gym.Env:
        env = DMCSuiteEnv(task_cfg.domain, task_cfg.task)
        env = VelocityGoalWrapper(
            env,
            v_min=task_cfg.v_min,
            v_max=task_cfg.v_max,
            forward_vel_obs_idx=self.forward_vel_obs_idx,
            success_threshold=task_cfg.success_threshold,
        )
        if self.use_one_hot:
            env = OneHotTaskWrapper(env, task_idx, len(self.tasks))
        env = gym.wrappers.TimeLimit(env, max_episode_steps=self.max_episode_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    @override
    def spawn(self, seed: int = 1) -> gym.vector.VectorEnv:
        fns = [
            (lambda cfg, idx: lambda: self._make_single_env(cfg, idx))(task_cfg, i)
            for i, task_cfg in enumerate(self.tasks)
        ]
        return gym.vector.AsyncVectorEnv(fns)

    @override
    def evaluate(
        self, envs: gym.vector.VectorEnv, agent: Agent
    ) -> tuple[float, float, dict[str, float]]:
        return _evaluate_locomotion(
            envs, agent, self.num_eval_episodes,
            [t.task_name for t in self.tasks],
        )


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DMCWalkerConfig(DMCLocomotionMTConfig):
    """Walker stand / walk / run with sampled velocity goals.

    Velocity ranges chosen to correspond to each gait's natural regime:
      stand: 0 – 0.5 m/s  (upright, near-stationary)
      walk:  0.5 – 2.0 m/s
      run:   2.0 – 4.0 m/s
    """

    tasks: tuple[DMCTaskConfig, ...] = (
        DMCTaskConfig("walker", "stand", "walker_stand", 0.0, 0.5, 0.2),
        DMCTaskConfig("walker", "walk",  "walker_walk",  0.5, 2.0, 0.4),
        DMCTaskConfig("walker", "run",   "walker_run",   2.0, 4.0, 0.5),
    )
    forward_vel_obs_idx: int = _WALKER_FWD_VEL_IDX
    env_id: str = "DMC-Walker"


@dataclass(frozen=True)
class DMCHopperConfig(DMCLocomotionMTConfig):
    """Hopper stand / hop with sampled velocity goals."""

    tasks: tuple[DMCTaskConfig, ...] = (
        DMCTaskConfig("hopper", "stand", "hopper_stand", 0.0, 0.2, 0.1),
        DMCTaskConfig("hopper", "hop",   "hopper_hop",   0.5, 2.0, 0.4),
    )
    forward_vel_obs_idx: int = _HOPPER_FWD_VEL_IDX
    env_id: str = "DMC-Hopper"


# ---------------------------------------------------------------------------
# Reacher — goal already in obs natively, no reward modification needed
# ---------------------------------------------------------------------------

def _evaluate_reacher(
    envs: gym.vector.VectorEnv,
    agent: Agent,
    num_episodes: int,
    task_names: list[str],
    success_threshold: float,
) -> tuple[float, float, dict[str, float]]:
    num_tasks = envs.num_envs
    episode_returns = [[] for _ in range(num_tasks)]
    episode_successes = [[] for _ in range(num_tasks)]
    episodes_done = [0] * num_tasks
    current_return = np.zeros(num_tasks)
    current_success_steps = np.zeros(num_tasks)
    current_length = np.zeros(num_tasks, dtype=int)

    obs, _ = envs.reset()
    while any(n < num_episodes for n in episodes_done):
        actions = agent.eval_action(obs)
        obs, rewards, terminations, truncations, infos = envs.step(actions)
        dones = terminations | truncations
        # Reacher obs: [position(2), to_target(2), velocity(2)]
        # to_target = obs[2:4]; distance = norm(to_target)
        distances = np.linalg.norm(obs[:, 2:4], axis=1)
        current_return += rewards
        current_success_steps += (distances < success_threshold).astype(float)
        current_length += 1

        for i in range(num_tasks):
            if dones[i] and episodes_done[i] < num_episodes:
                episode_returns[i].append(float(current_return[i]))
                episode_successes[i].append(
                    float(current_success_steps[i] / max(current_length[i], 1))
                )
                episodes_done[i] += 1
                current_return[i] = 0.0
                current_success_steps[i] = 0.0
                current_length[i] = 0

    per_task = {
        name: float(np.mean(episode_successes[i]))
        for i, name in enumerate(task_names)
    }
    return (
        float(np.mean(list(per_task.values()))),
        float(np.mean([np.mean(r) for r in episode_returns])),
        per_task,
    )


@dataclass(frozen=True)
class DMCReacherConfig(EnvConfig):
    """Reacher easy / hard.  Goal (to_target) is already in the native obs.

    'easy' and 'hard' differ only in target radius, so they share the same
    obs/action space.  No reward modification is needed.
    """

    env_id: str = "DMC-Reacher"
    num_eval_episodes: int = 10
    success_threshold: float = 0.05  # metres

    _TASK_NAMES: tuple[str, ...] = ("reacher_easy", "reacher_hard")

    @cached_property
    @override
    def observation_space(self) -> gym.Space:
        # native obs(6) + one_hot(2 if use_one_hot)
        dim = 6 + (2 if self.use_one_hot else 0)
        return gym.spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float64)

    @cached_property
    @override
    def action_space(self) -> gym.Space:
        sample = DMCSuiteEnv("reacher", "easy")
        space = sample.action_space
        sample.close()
        return space

    def _make_single_env(self, task: str, task_idx: int) -> gym.Env:
        env = DMCSuiteEnv("reacher", task)
        if self.use_one_hot:
            env = OneHotTaskWrapper(env, task_idx, 2)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=self.max_episode_steps)
        return env

    @override
    def spawn(self, seed: int = 1) -> gym.vector.VectorEnv:
        fns = [
            (lambda t, i: lambda: self._make_single_env(t, i))(task, idx)
            for idx, task in enumerate(("easy", "hard"))
        ]
        return gym.vector.AsyncVectorEnv(fns)

    @override
    def evaluate(
        self, envs: gym.vector.VectorEnv, agent: Agent
    ) -> tuple[float, float, dict[str, float]]:
        return _evaluate_reacher(
            envs, agent, self.num_eval_episodes,
            list(self._TASK_NAMES), self.success_threshold,
        )
