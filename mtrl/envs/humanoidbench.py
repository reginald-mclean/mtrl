# pyright: reportAttributeAccessIssue=false, reportIncompatibleMethodOverride=false, reportOptionalMemberAccess=false
# TODO: all of this will be in actual MW in a future release
from dataclasses import dataclass
from functools import cached_property
from typing import override

import gymnasium as gym
import numpy as np

import humanoid_bench

from mtrl.types import Agent

from .base import EnvConfig

from metaworld.wrappers import OneHotWrapper

@dataclass(frozen=True)
class HumanoidBenchConfig(EnvConfig):
    eval_episodes: int = 10

    @cached_property
    @override
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(
            np.array([-1]*19, dtype=np.float32),
            np.array([1]*19, dtype=np.float32),
        )

    @cached_property
    @override
    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(
            np.array([-np.inf]*60, dtype=np.float32),
            np.array([np.inf]*60, dtype=np.float32),
        )
    @override
    def evaluate(
        self, envs: gym.vector.VectorEnv, agent: Agent
    ) -> tuple[float, float, dict[str, float]]:
        mean_returns, mean_hns, mean_return_per_task, hns_per_task = evaluation(agent, envs, num_episodes=self.eval_episodes)
        return mean_hns, mean_returns, hns_per_task

    @override
    def spawn_eval(self, seed: int = 1) -> gym.vector.VectorEnv:
        names = ['walk', 'stand', 'run', 'stair', 'crawl', 'pole', 'slide', 'hurdle', 'maze']
        env_fns = [lambda : gym.wrappers.RecordEpisodeStatistics(
                OneHotWrapper(
                    gym.make(f'h1-{name}-v0'), 
                    idx, 
                    len(names),
                )
            ) for idx, name in enumerate(names)]
        return gym.vector.SyncVectorEnv(env_fns)

    @override
    def spawn(self, seed: int = 1) -> gym.vector.VectorEnv:
        names = ['walk', 'stand', 'run', 'stair', 'crawl', 'pole', 'slide', 'hurdle', 'maze']
        env_fns = [lambda : gym.wrappers.RecordEpisodeStatistics(
                OneHotWrapper(
                    gym.make(f'h1-{name}-v0'), 
                    idx, 
                    len(names),
                )
            ) for idx, name in enumerate(names)]
        return gym.vector.SyncVectorEnv(env_fns)


HUMANOIDBENCH_SCORES = {
    "h1-crawl-v0":  {"random": 272.658, "optimal": 700.0},
    "h1-hurdle-v0": {"random": 2.214,   "optimal": 700.0},
    "h1-maze-v0":   {"random": 106.441, "optimal": 1200.0},
    "h1-pole-v0":   {"random": 20.090,  "optimal": 700.0},
    "h1-run-v0":    {"random": 2.020,   "optimal": 700.0},
    "h1-slide-v0":  {"random": 3.191,   "optimal": 700.0},
    "h1-stair-v0":  {"random": 3.112,   "optimal": 700.0},
    "h1-stand-v0":  {"random": 10.545,  "optimal": 800.0},
    "h1-walk-v0":   {"random": 2.377,   "optimal": 700.0},
}

def normalize_score(task_name, episode_return):
    scores = HUMANOIDBENCH_SCORES[task_name]
    return (episode_return - scores["random"]) / (scores["optimal"] - scores["random"])


def evaluation(
    agent: Agent,
    eval_envs: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
    num_episodes: int = 5,
) -> tuple[float, dict[str, float]]:
    obs: npt.NDArray[np.float64]
    obs, _ = eval_envs.reset()
    agent.reset(np.ones(eval_envs.num_envs, dtype=np.bool_))

    task_names = None

    #if self.env_id == 'medium':
    task_names = ['walk', 'stand', 'run', 'stair', 'crawl', 'pole', 'slide', 'hurdle', 'maze']

    episodic_returns: dict[str, list[float]] = {
        task_name: [] for task_name in task_names
    }

    def eval_done(returns):
        return all(len(r) >= num_episodes for r in returns.values())

    task_ids = np.arange(eval_envs.num_envs)
    while not eval_done(episodic_returns):
        actions = agent.eval_action(obs, task_ids)
        obs, _, terminations, truncations, infos = eval_envs.step(actions)

        dones = np.logical_or(terminations, truncations)
        agent.reset(dones)

        for i, env_ended in enumerate(dones):
            if env_ended:
                episodic_returns[task_names[i]].append(
                    float(infos["episode"]["r"][i])
                )

    mean_return_per_task = {
        task_name: float(np.mean(returns[:num_episodes]))
        for task_name, returns in episodic_returns.items()
    }
    mean_returns = float(np.mean(list(mean_return_per_task.values())))

    hns_per_task = {
        task_name: normalize_score(f'h1-{task_name}-v0', mean_return)
        for task_name, mean_return in mean_return_per_task.items()
    }
    mean_hns = float(np.median(list(hns_per_task.values())))

    return mean_returns, mean_hns, mean_return_per_task, hns_per_task
