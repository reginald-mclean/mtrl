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


@dataclass(frozen=True)
class HumanoidBenchConfig(EnvConfig):
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
            np.array([-np.inf]*51, dtype=np.float32),
            np.array([np.inf]*51, dtype=np.float32),
        )

    @override
    def evaluate(
        self, envs: gym.vector.VectorEnv, agent: Agent
    ) -> tuple[float, float, dict[str, float]]:
        assert isinstance(envs, gym.vector.AsyncVectorEnv) or isinstance(
            envs, gym.vector.SyncVectorEnv
        )
        print("This will not throw an error, but it is not implemented!!!")
        return 0.,0.,{0: 0.}

    @override
    def spawn(self, seed: int = 1) -> gym.vector.VectorEnv:
        names = ['walk', 'stand', 'run', 'stair', 'crawl', 'pole', 'slide', 'hurdle', 'maze']
        env_fns = [lambda : gym.wrappers.RecordEpisodeStatistics(gym.make(f'h1-{name}-v0')) for name in names]
        print([f'h1-{name}-v0' for name in names])
        return gym.vector.SyncVectorEnv(env_fns)
