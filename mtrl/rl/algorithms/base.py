import abc
import time
from collections import deque
from typing import Deque, Generic, Self, TypeVar, override

import gymnasium as gym
import numpy as np
import orbax.checkpoint as ocp
import wandb
from flax import struct

from mtrl.checkpoint import get_checkpoint_save_args
from mtrl.config.rl import (
    AlgorithmConfig,
    OffPolicyTrainingConfig,
    OnPolicyTrainingConfig,
    TrainingConfig,
)
from mtrl.config.utils import Metrics
from mtrl.envs import EnvConfig
from mtrl.rl.buffers import MultiTaskReplayBuffer, MultiTaskRolloutBuffer
from mtrl.types import (
    Action,
    Agent,
    CheckpointMetadata,
    LogDict,
    LogProb,
    Observation,
    ReplayBufferCheckpoint,
    ReplayBufferSamples,
    Rollout,
    Value,
)

AlgorithmConfigType = TypeVar("AlgorithmConfigType", bound=AlgorithmConfig)
TrainingConfigType = TypeVar("TrainingConfigType", bound=TrainingConfig)
DataType = TypeVar("DataType", ReplayBufferSamples, Rollout)


class Algorithm(
    abc.ABC,
    Agent,
    Generic[AlgorithmConfigType, TrainingConfigType, DataType],
    struct.PyTreeNode,
):
    """Based on https://github.com/kevinzakka/nanorl/blob/main/nanorl/agent.py"""

    num_tasks: int = struct.field(pytree_node=False)

    @staticmethod
    @abc.abstractmethod
    def initialize(
        config: AlgorithmConfigType, env_config: EnvConfig, seed: int = 1
    ) -> "Algorithm":
        ...

    @abc.abstractmethod
    def update(self, data: DataType) -> tuple[Self, LogDict]:
        ...

    @abc.abstractmethod
    def get_metrics(self, metrics: Metrics, data: DataType) -> tuple[Self, LogDict]:
        ...

    @abc.abstractmethod
    def get_num_params(self) -> dict[str, int]:
        ...

    @abc.abstractmethod
    def sample_action(self, observation: Observation) -> tuple[Self, Action]:
        ...

    @abc.abstractmethod
    def eval_action(self, observations: Observation) -> Action:
        ...

    @abc.abstractmethod
    def on_task_change(self, task_idx: int) -> None:
        ...

    @abc.abstractmethod
    def should_early_terminate(self) -> None:
        ...

    @abc.abstractmethod
    def _handle_task_change(self, current_task_idx: int) -> None:
        ...

    @abc.abstractmethod
    def on_task_end(self, current_task_idx: int) -> None:
        ...

    @abc.abstractmethod
    def train(
        self,
        config: TrainingConfigType,
        envs: gym.vector.VectorEnv,
        env_config: EnvConfig,
        run_timestamp: str,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self:
        ...


class OffPolicyAlgorithm(
    Algorithm[AlgorithmConfigType, OffPolicyTrainingConfig, ReplayBufferSamples],
    Generic[AlgorithmConfigType],
):
    def spawn_replay_buffer(
        self, env_config: EnvConfig, config: OffPolicyTrainingConfig, seed: int = 1,
    ) -> MultiTaskReplayBuffer:
        return MultiTaskReplayBuffer(
            total_capacity=config.buffer_size,
            num_tasks=self.num_tasks,
            env_obs_space=env_config.observation_space,
            env_action_space=env_config.action_space,
            seed=seed,
            reward_filter=config.reward_filter,
            alpha=config.reward_filter_alpha,
            sigma=config.reward_filter_sigma,
            delta=config.reward_filter_delta,
            filter_mode=config.reward_filter_mode,
        )

    @override
    def on_task_end(self, current_task_idx: int) -> None:
        pass

    @override
    def train(
        self,
        config: OffPolicyTrainingConfig,
        envs: gym.vector.VectorEnv,
        eval_envs: gym.vector.VectorEnv,
        env_config: EnvConfig,
        run_timestamp: str,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self:
        global_episodic_return: Deque[float] = deque([], maxlen=20 * self.num_tasks)
        global_episodic_length: Deque[int] = deque([], maxlen=20 * self.num_tasks)

        obs, _ = envs.reset()

        if isinstance(envs, gym.vector.VectorEnv):
            num_envs = envs.num_envs
        else:
            num_envs = envs.get_wrapper_attr('num_envs')

        done = np.full((num_envs,), False)
        start_step, episodes_ended = 0, 0

        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes_ended"]

        replay_buffer = self.spawn_replay_buffer(env_config, config, seed)

        if buffer_checkpoint is not None:
            replay_buffer.load_checkpoint(buffer_checkpoint)

        start_time = time.time()

        current_task_step = 0
        global_step = 0
        curr_seq_idx = 0

        if isinstance(envs, gym.vector.VectorEnv):
            pass
        else:
            curr_seq_idx = envs.get_wrapper_attr('cur_seq_idx')

        while global_step < (config.total_steps // num_envs):
            if self.should_early_terminate():
                print(f'Task end {current_task_idx}')
                # so each method will have a different method of handling what happens on task end
                self.on_task_end(curr_seq_idx)

                added_steps = self.env.do_next_task()
                global_timestep += (
                    added_steps  # Need to add steps, otherwise not env steps
                )


            if not isinstance(envs, gym.vector.VectorEnv) and curr_seq_idx != envs.get_wrapper_attr('cur_seq_idx'):
                print('task change')
                current_task_step = 0
                curr_seq_idx = envs.get_wrapper_attr('cur_seq_idx')
                print(curr_seq_idx, envs.get_wrapper_attr('num_envs'))
                if curr_seq_idx == len(envs.get_wrapper_attr('envs')):
                    print("Finished running through all envs")
                    break

                self = self._handle_task_change(curr_seq_idx)
                obs, _ = envs.reset()

            total_steps = global_step * num_envs
            if current_task_step < config.warmstart_steps:
                actions = np.array(envs.action_space.sample()) # for _ in range(num_envs)])
            else:
                self, actions = self.sample_action(obs)

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            done = np.logical_or(terminations, truncations)

            buffer_obs = next_obs
            if "final_obs" in infos:
                buffer_obs = np.where(
                    done[:, None], np.stack(infos["final_obs"]), next_obs
                )
                current_task_step = 0

            replay_buffer.add(obs, buffer_obs, actions, rewards, done)

            obs = next_obs

            if not isinstance(done, np.ndarray):
                done = np.array([done])

            for i, env_ended in enumerate(done):
                if env_ended:
                    if eval_envs:
                        global_episodic_return.append(
                            infos["episode"]["r"]
                        )
                        global_episodic_length.append(
                            infos["episode"]["l"]
                        )
                    else:
                        global_episodic_return.append(
                            infos["episode"]["r"][i]
                        )
                        global_episodic_length.append(
                            infos["episode"]["l"][i]
                        )

            if global_step % 500 == 0 and global_episodic_return:
                print(
                    f"global_step={total_steps}, mean_episodic_return={np.mean(list(global_episodic_return))}"
                )
                if track:
                    wandb.log(
                        {
                            "charts/mean_episodic_return": np.mean(
                                list(global_episodic_return)
                            ),
                            "charts/mean_episodic_length": np.mean(
                                list(global_episodic_length)
                            ),
                        },
                        step=total_steps,
                    )

            if current_task_step > config.warmstart_steps:
                data = replay_buffer.sample(config.batch_size)
                self, logs = self.update(data)

                # Logging
                if global_step % 100 == 0:
                    sps_steps = (global_step - start_step) * num_envs
                    sps = int(sps_steps / (time.time() - start_time))
                    print("SPS:", sps)

                    if track:
                        wandb.log({"charts/SPS": sps} | logs, step=total_steps)

                # Evaluation
                if (
                    config.evaluation_frequency > 0
                    and current_task_step % config.evaluation_frequency == 0
                    and current_task_step > 0
                ):
                    (
                        mean_success_rate,
                        mean_returns,
                        mean_success_per_task,
                    ) = env_config.evaluate(envs if not eval_envs else eval_envs, self)
                    eval_metrics = {
                        "charts/mean_success_rate": float(mean_success_rate),
                        "charts/mean_evaluation_return": float(mean_returns),
                    } | {
                        f"charts/{task_name}_success_rate": float(success_rate)
                        for task_name, success_rate in mean_success_per_task.items()
                    }
                    print(
                        f"total_steps={total_steps}, mean evaluation success rate: {mean_success_rate:.4f}"
                        + f" return: {mean_returns:.4f}"
                    )

                    if track:
                        wandb.log(eval_metrics, step=global_step)

                    if config.compute_network_metrics.value != 0:
                        self, network_metrics = self.get_metrics(
                            config.compute_network_metrics, data
                        )
                        print(network_metrics)
                        if track:
                            wandb.log(network_metrics, step=total_steps)
                    # Checkpointing
                    if checkpoint_manager is not None:
                        if not done.all():
                            raise NotImplementedError(
                                "Checkpointing currently doesn't work for the case where evaluation is run before all envs have finished their episodes / are about to be reset."
                            )

                        checkpoint_manager.save(
                            total_steps,
                            args=get_checkpoint_save_args(
                                self,
                                envs,
                                global_step,
                                episodes_ended,
                                run_timestamp,
                                buffer=replay_buffer,
                            ),
                            metrics={
                                k.removeprefix("charts/"): v
                                for k, v in eval_metrics.items()
                            },
                        )

                    # Reset envs again to exit eval mode
                    obs, _ = envs.reset()

            if not isinstance(envs, gym.vector.VectorEnv) and current_task_step == envs.get_wrapper_attr('steps_per_env'):
                print(f"Task end. Current task idx: {curr_seq_idx}")
                if envs.get_wrapper_attr('cur_seq_idx') == (len(envs.get_wrapper_attr('envs'))):
                    print("Finished running through all envs")
                    break
                curr_seq_idx = envs.get_wrapper_attr('cur_seq_idx')
                self.on_task_end(curr_seq_idx)
                obs, _ = envs.reset()

            global_step += 1
            current_task_step += 1
        return self


class OnPolicyAlgorithm(
    Algorithm[AlgorithmConfigType, OnPolicyTrainingConfig, Rollout],
    Generic[AlgorithmConfigType],
):
    @abc.abstractmethod
    def sample_action_dist_and_value(
        self, observation: Observation
    ) -> tuple[Self, Action, LogProb, Action, Action, Value]:
        ...

    def spawn_rollout_buffer(
        self,
        env_config: EnvConfig,
        training_config: OnPolicyTrainingConfig,
        seed: int | None = None,
    ) -> MultiTaskRolloutBuffer:
        return MultiTaskRolloutBuffer(
            training_config.rollout_steps,
            self.num_tasks,
            env_config.observation_space,
            env_config.action_space,
            seed,
        )

    @override
    def train(
        self,
        config: OnPolicyTrainingConfig,
        envs: gym.vector.VectorEnv,
        env_config: EnvConfig,
        run_timestamp: str,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self:
        global_episodic_return: Deque[float] = deque([], maxlen=20 * self.num_tasks)
        global_episodic_length: Deque[int] = deque([], maxlen=20 * self.num_tasks)

        obs, _ = envs.reset()

        episode_started = np.ones((num_envs,))
        start_step, episodes_ended = 0, 0

        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes_ended"]

        rollout_buffer = self.spawn_rollout_buffer(env_config, config, seed)

        start_time = time.time()

        for global_step in range(start_step, config.total_steps // num_envs):
            total_steps = global_step * num_envs

            (
                self,
                actions,
                log_probs,
                means,
                stds,
                value,
            ) = self.sample_action_dist_and_value(obs)
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            rollout_buffer.add(
                obs,
                actions,
                rewards,
                episode_started,
                value=value,
                log_prob=log_probs,
                mean=means,
                std=stds,
            )

            episode_started = np.logical_or(terminations, truncations)
            obs = next_obs

            for i, env_ended in enumerate(episode_started):
                if env_ended:
                    global_episodic_return.append(
                        infos["final_info"]["episode"]["r"][i]
                    )
                    global_episodic_length.append(
                        infos["final_info"]["episode"]["l"][i]
                    )
                    episodes_ended += 1

            if global_step % 500 == 0 and global_episodic_return:
                print(
                    f"global_step={total_steps}, mean_episodic_return={np.mean(list(global_episodic_return))}"
                )
                if track:
                    wandb.log(
                        {
                            "charts/mean_episodic_return": np.mean(
                                list(global_episodic_return)
                            ),
                            "charts/mean_episodic_length": np.mean(
                                list(global_episodic_length)
                            ),
                        },
                        step=total_steps,
                    )

            # Logging
            if global_step % 1_000 == 0:
                sps_steps = (global_step - start_step) * num_envs
                sps = int(sps_steps / (time.time() - start_time))
                print("SPS:", sps)

                if track:
                    wandb.log({"charts/SPS": sps}, step=total_steps)

            if rollout_buffer.ready:
                rollouts = rollout_buffer.get(
                    compute_advantages=True,
                    last_values=value,
                    dones=terminations,
                    gamma=config.gamma,
                    gae_lambda=config.gae_lambda,
                )
                self, logs = self.update(rollouts)
                rollout_buffer.reset()

                if track:
                    wandb.log(logs, step=total_steps)

                # Evaluation
                if (
                    config.evaluation_frequency > 0
                    and episodes_ended % config.evaluation_frequency == 0
                    and episode_started.any()
                    and global_step > 0
                ):
                    (
                        mean_success_rate,
                        mean_returns,
                        mean_success_per_task,
                    ) = env_config.evaluate(envs, self)
                    eval_metrics = {
                        "charts/mean_success_rate": float(mean_success_rate),
                        "charts/mean_evaluation_return": float(mean_returns),
                    } | {
                        f"charts/{task_name}_success_rate": float(success_rate)
                        for task_name, success_rate in mean_success_per_task.items()
                    }
                    print(
                        f"total_steps={total_steps}, mean evaluation success rate: {mean_success_rate:.4f}"
                        + f" return: {mean_returns:.4f}"
                    )

                    if track:
                        wandb.log(eval_metrics, step=total_steps)

                    # Checkpointing
                    if checkpoint_manager is not None:
                        if not episode_started.all():
                            raise NotImplementedError(
                                "Checkpointing currently doesn't work for the case where evaluation is run before all envs have finished their episodes / are about to be reset."
                            )

                        checkpoint_manager.save(
                            total_steps,
                            args=get_checkpoint_save_args(
                                self,
                                envs,
                                global_step,
                                episodes_ended,
                                run_timestamp,
                            ),
                            metrics={
                                k.removeprefix("charts/"): v
                                for k, v in eval_metrics.items()
                            },
                        )

                    # Reset envs again to exit eval mode
                    obs, _ = envs.reset()
                    episode_started = np.ones((num_envs,))

        return self
