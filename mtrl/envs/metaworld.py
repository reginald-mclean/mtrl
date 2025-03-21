# pyright: reportAttributeAccessIssue=false, reportIncompatibleMethodOverride=false, reportOptionalMemberAccess=false
# TODO: all of this will be in actual MW in a future release
from dataclasses import dataclass
from functools import cached_property
from typing import override

import gymnasium as gym
import numpy as np

from mtrl.types import Agent

from .base import EnvConfig
from metaworld.evaluation import evaluation


@dataclass(frozen=True)
class MetaworldConfig(EnvConfig):
    reward_func_version: str = "v2"
    num_eval_episodes: int = 50
    num_goals: int = 50
    reward_normalization_method: str | None = None
    task_name: str | None = None
    sparse_rewards: bool = False
    sparse_magnitude: int = 0

    @cached_property
    @override
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(
            np.array([-1, -1, -1, -1], dtype=np.float32),
            np.array([+1, +1, +1, +1], dtype=np.float32),
        )

    @cached_property
    @override
    def observation_space(self) -> gym.Space:
        _HAND_SPACE = gym.spaces.Box(
            np.array([-0.525, 0.348, -0.0525]),
            np.array([+0.525, 1.025, 0.7]),
            dtype=np.float64,
        )

        goal_low = (-0.1, 0.85, 0.0)
        goal_high = (0.1, 0.9 + 1e-7, 0.0)

        goal_space = gym.spaces.Box(
            np.array(goal_low) + np.array([0, -0.083, 0.2499]),
            np.array(goal_high) + np.array([0, -0.083, 0.2501]),
            dtype=np.float64,
        )
        obs_obj_max_len = 14
        obj_low = np.full(obs_obj_max_len, -np.inf)
        obj_high = np.full(obs_obj_max_len, +np.inf)
        goal_low = goal_space.low
        goal_high = goal_space.high
        gripper_low = -1.0
        gripper_high = +1.0

        env_obs_space = gym.spaces.Box(
            np.hstack(
                (
                    _HAND_SPACE.low,
                    gripper_low,
                    obj_low,
                    _HAND_SPACE.low,
                    gripper_low,
                    obj_low,
                    goal_low,
                )
            ),
            np.hstack(
                (
                    _HAND_SPACE.high,
                    gripper_high,
                    obj_high,
                    _HAND_SPACE.high,
                    gripper_high,
                    obj_high,
                    goal_high,
                )
            ),
            dtype=np.float64,
        )

        if self.use_one_hot and self.env_id != "MT1":
            num_tasks = 1
            if self.env_id == "MT10":
                num_tasks = 10
            if self.env_id == "MT50":
                num_tasks = 50
            if self.env_id == "MT25":
                num_tasks = 25
            one_hot_ub = np.ones(num_tasks)
            one_hot_lb = np.zeros(num_tasks)

            env_obs_space = gym.spaces.Box(
                np.concatenate([env_obs_space.low, one_hot_lb]),
                np.concatenate([env_obs_space.high, one_hot_ub]),
                dtype=np.float64,
            )

        return env_obs_space

    @override
    def evaluate(
        self, envs: gym.vector.VectorEnv, agent: Agent
    ) -> tuple[float, float, dict[str, float]]:
        assert isinstance(envs, gym.vector.AsyncVectorEnv) or isinstance(
            envs, gym.vector.SyncVectorEnv
        )
        return evaluation(agent, envs, num_episodes=self.num_eval_episodes)[:3]

    @override
    def spawn(self, seed: int = 1) -> gym.vector.VectorEnv:
        if self.env_id == "MT25":
            envs_list = [
                "reach-v3",
                "push-v3",
                "pick-place-v3",
                "door-open-v3",
                "drawer-open-v3",
                "drawer-close-v3",
                "button-press-topdown-v3",
                "peg-insert-side-v3",
                "window-open-v3",
                "window-close-v3",
                "coffee-pull-v3",
                "pick-out-of-hole-v3",
                "disassemble-v3",
                "pick-place-wall-v3",
                "basketball-v3",
                "stick-pull-v3",
                "button-press-wall-v3",
                "faucet-open-v3",
                "door-lock-v3",
                "lever-pull-v3",
                "sweep-into-v3",
                "faucet-close-v3",
                "coffee-button-v3",
                "button-press-topdown-wall-v3",
                "dial-turn-v3",
            ]
            return gym.make_vec(
                "Meta-World/custom-mt-envs",
                seed=seed,
                envs_list=envs_list,
                use_one_hot=self.use_one_hot,
                terminate_on_success=self.terminate_on_success,
                vector_strategy="async",
                reward_function_version=self.reward_func_version,
                num_goals=self.num_goals,
                reward_normalization_method=self.reward_normalization_method,
                sparse_rewards=self.sparse_rewards,
                sparse_magnitude=self.sparse_magnitude,
            )
        elif self.env_id == "MT1":
            assert self.task_name is not None, "task_name must be specified for MT1"
            return gym.vector.SyncVectorEnv(
                [
                    lambda: gym.make(
                        "Meta-World/MT1",
                        env_name=self.task_name,
                        use_one_hot=False,
                        seed=seed,
                        sparse_rewards=self.sparse_rewards,
                        sparse_magnitude=self.sparse_magnitude,
                        terminate_on_success=self.terminate_on_success,
                        reward_function_version=self.reward_func_version,
                        num_goals=self.num_goals,
                        reward_normalization_method=self.reward_normalization_method,
                    )
                ]
            )
        else:
            return gym.make_vec(
                f"Meta-World/{self.env_id}",
                seed=seed,
                use_one_hot=self.use_one_hot,
                sparse_rewards=self.sparse_rewards,
                sparse_magnitude=self.sparse_magnitude,
                terminate_on_success=self.terminate_on_success,
                vector_strategy="async",
                reward_function_version=self.reward_func_version,
                num_goals=self.num_goals,
                reward_normalization_method=self.reward_normalization_method,
            )
