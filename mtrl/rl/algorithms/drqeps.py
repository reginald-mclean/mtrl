"""Inspired by https://github.com/kevinzakka/robopianist-rl/blob/main/sac.py"""

import dataclasses
from functools import partial
from typing import Self, override

import distrax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.flatten_util as flatten_util
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from jaxtyping import Array, Float, PRNGKeyArray

from mtrl.config.networks import ImpalaDQNConfig
#from mtrl.config.nn import ImpalaEncoderConfig
from mtrl.config.optim import OptimizerConfig
from mtrl.config.rl import AlgorithmConfig, OffPolicyTrainingConfig
from mtrl.config.utils import Metrics
from mtrl.envs import EnvConfig
from mtrl.monitoring.metrics import (
    compute_srank,
    extract_activations,
    get_dormant_neuron_logs,
)
from mtrl.rl.buffers import AtariMultiTaskReplayBuffer
from mtrl.rl.networks import ContinuousActionPolicy, Ensemble, QValueFunction, ImpalaDQN
from mtrl.types import (
    Action,
    Intermediates,
    LayerActivationsDict,
    LogDict,
    Observation,
    AtariReplayBufferSamples,
)

from .base import OffPolicyAlgorithm

class CriticTrainState(TrainState):
    target_params: FrozenDict | None = None

@partial(jax.jit, static_argnames=("v_min", "v_max", "n_atoms", "eps_start", "eps_end", "eps_decay_steps"))
def _sample_action(
    critic: CriticTrainState,
    observation: Observation,
    task_ids: jnp.ndarray,
    key: PRNGKeyArray,
    step: jnp.ndarray,
    v_min: float,
    v_max: float,
    n_atoms: int,
    eps_start: float,
    eps_end: float,
    eps_decay_steps: int,
) -> tuple[jnp.ndarray, PRNGKeyArray]:
    key, sample_key, random_key = jax.random.split(key, 3)
    logits = critic.apply_fn(critic.params, observation, task_ids)    
    logits = jnp.mean(logits, axis=0)
    exp_q_vals = jax.nn.softmax(logits, axis=-1)
    support = jnp.linspace(v_min, v_max, n_atoms)
    q_values = jnp.sum(exp_q_vals * support, axis=-1)
    t = jnp.minimum(step / eps_decay_steps, 1.0)
    epsilon = eps_start + t * (eps_end - eps_start)
    uniform_samples = jax.random.uniform(sample_key, shape=(observation.shape[0],))
    random_actions = jax.random.randint(random_key, shape=(observation.shape[0],), minval=0, maxval=18) # this is hardcoded for Atari
    greedy_actions = jnp.argmax(q_values, axis=-1)
    actions = jnp.where(uniform_samples < epsilon, random_actions, greedy_actions)
    return actions, key

@partial(jax.jit, static_argnames=("v_min", "v_max", "n_atoms"))
def _eval_action(
    critic: CriticTrainState,
    observation: Observation,
    task_ids: jnp.ndarray,
    v_min: float,
    v_max: float,
    n_atoms: int,
) -> jnp.ndarray:
    logits = critic.apply_fn(critic.params, observation, task_ids)
    logits = jnp.mean(logits, axis=0)
    support = jnp.linspace(v_min, v_max, n_atoms)
    q_values = jnp.sum(jax.nn.softmax(logits, axis=-1) * support, axis=-1)
    return jnp.argmax(q_values, axis=-1)
    

@dataclasses.dataclass(frozen=True)
class DrQConfig(AlgorithmConfig):
    critic_config: ImpalaDQNConfig = ImpalaDQNConfig()
    tau: float = 0.005
    num_critics: int = 2
    v_min: float = 0.0
    v_max: float = 10.0
    n_atoms: int = 51
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 10_000
    num_tasks: int = 26


class DrQ(OffPolicyAlgorithm[DrQConfig]):
    critic: CriticTrainState
    key: PRNGKeyArray
    step: jnp.ndarray
    gamma: float = struct.field(pytree_node=False)
    tau: float = struct.field(pytree_node=False)
    num_critics: int = struct.field(pytree_node=False)
    v_min: float = struct.field(pytree_node=False)
    v_max: float = struct.field(pytree_node=False)
    n_atoms: int = struct.field(pytree_node=False)
    eps_start: float = struct.field(pytree_node=False)
    eps_end: float = struct.field(pytree_node=False)
    eps_decay_steps: int = struct.field(pytree_node=False)
    num_tasks: int = struct.field(pytree_node=False)

    def spawn_replay_buffer(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, env_config: EnvConfig, config: OffPolicyTrainingConfig, seed: int = 1
    ) -> AtariMultiTaskReplayBuffer:
        return AtariMultiTaskReplayBuffer(
            total_capacity=config.buffer_size,
            num_tasks=config.num_tasks,
            env_obs_space=env_config.observation_space,
            env_action_space=env_config.action_space,
            seed=seed,
        )

    @override
    @staticmethod
    def initialize(config: DrQConfig, env_config: EnvConfig, seed: int = 1) -> "DrQ":
        assert isinstance(env_config.action_space, gym.spaces.Discrete), (
            "DQN is only used for discrete action spaces."
        )
        assert isinstance(env_config.observation_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )

        master_key = jax.random.PRNGKey(seed)
        algorithm_key, critic_init_key = (
            jax.random.split(master_key, 2)
        )

        dummy_obs = jnp.array(
            [env_config.observation_space.sample() for _ in range(config.num_tasks)],
            dtype=jnp.uint8
        )

        task_ids = jnp.arange(config.num_tasks)

        critic_net = ImpalaDQN(
            impala_config=config.critic_config.impala_config, 
            q_function_config=config.critic_config.q_function_config,
            task_embed_config=config.critic_config.task_embed_config,
            action_dim=env_config.action_space.n
        )

        critic_init_params = critic_net.init(critic_init_key, dummy_obs, task_ids)
        critic = CriticTrainState.create(
            apply_fn=critic_net.apply,
            params=critic_init_params,
            target_params=critic_init_params,
            tx=config.critic_config.q_function_config.network_config.optimizer.spawn(),
        )

        print("Critic Arch:", jax.tree_util.tree_map(jnp.shape, critic.params))
        print("Critic Params:", sum(x.size for x in jax.tree.leaves(critic.params)))

        return DrQ(
            critic=critic,
            key=algorithm_key,
            num_tasks=config.num_tasks,
            step=jnp.zeros((), dtype=jnp.int32),
            gamma=config.gamma,
            tau=config.tau,
            num_critics=config.num_critics,
            v_min=config.v_min,
            v_max=config.v_max,
            n_atoms=config.n_atoms,
            eps_start=config.eps_start,
            eps_end=config.eps_end,
            eps_decay_steps=config.eps_decay_steps,
        )
    def reset(self, env_mask) -> None:
        pass

    @override
    def get_num_params(self) -> dict[str, int]:
        return {
            "critic_num_params": sum(
                x.size for x in jax.tree.leaves(self.critic.params)
            ),
        }

    @override
    def sample_action(self, observation: Observation, task_ids: jax.Array) -> tuple[Self, Action]:
        actions, key = _sample_action(
            self.critic, observation, task_ids, self.key,
            self.step, self.v_min, self.v_max, self.n_atoms,
            self.eps_start, self.eps_end, self.eps_decay_steps,
        )
        return self.replace(key=key, step=self.step + 1), actions # jax.device_get(actions)

    @override
    def eval_action(self, observation: Observation, task_ids: jnp.ndarray) -> Action:
        return jax.device_get(
            _eval_action(self.critic, observation, task_ids, self.v_min, self.v_max, self.n_atoms)
        )

    @jax.jit
    def _update_inner(self, data: AtariReplayBufferSamples) -> tuple[Self, LogDict]:
        # --- Critic loss ---
        print("TRACING")
        task_ids = data.task_ids

        next_obs_logits = self.critic.apply_fn(self.critic.params, data.next_observations, task_ids).mean(axis=0)
        support = jnp.linspace(self.v_min, self.v_max, self.n_atoms)
        next_q_values = jnp.sum(jax.nn.softmax(next_obs_logits, axis=-1) * support, axis=-1)

        target_logits = self.critic.apply_fn(self.critic.target_params, data.next_observations, task_ids)
        target_probs = jax.nn.softmax(target_logits, axis=-1).mean(axis=0)
        
        next_actions = jnp.argmax(next_q_values, axis=-1)
        target_dist = target_probs[jnp.arange(data.observations.shape[0]), next_actions]

        tz_j = jnp.clip(
            data.rewards + self.gamma * (1 - data.truncations) * support, 
            self.v_min,
            self.v_max,
        )
        delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
        b = (tz_j - self.v_min) / delta_z        # (B, n_atoms)
        l = jnp.floor(b).astype(jnp.int32)       # (B, n_atoms)
        u = jnp.ceil(b).astype(jnp.int32)        # (B, n_atoms)
        
        def critic_loss(
                params: FrozenDict,
        ) -> tuple[Float[Array, ""], Float[Array, ""]]:
            B = data.observations.shape[0]
            m = jnp.zeros((B, self.n_atoms))
            m = m.at[jnp.arange(B)[:, None], l].add(target_dist * (u - b))
            m = m.at[jnp.arange(B)[:, None], u].add(target_dist * (b - l))
            m = jax.lax.stop_gradient(m)
   
            online_logits = self.critic.apply_fn(params, data.observations, task_ids)
            online_logits = online_logits[:, jnp.arange(B), data.actions, :]
            log_probs = jax.nn.log_softmax(online_logits, axis=-1)
            loss = -(m * log_probs).sum(axis=-1).mean()
            return loss, {"losses/online_logits": online_logits.mean()}

        (critic_loss_value, logs), critic_grads = jax.value_and_grad(
            critic_loss, has_aux=True
        )(self.critic.params)
        critic = self.critic.apply_gradients(grads=critic_grads)

        flat_grads, _ = flatten_util.ravel_pytree(critic_grads)
        logs["metrics/critic_grad_magnitude"] = jnp.linalg.norm(flat_grads)

        flat_params_crit, _ = flatten_util.ravel_pytree(self.critic.params)
        logs["metrics/critic_params_norm"] = jnp.linalg.norm(flat_params_crit)

        critic: CriticTrainState
        critic = critic.replace(
            target_params=optax.incremental_update(
                critic.params,
                critic.target_params,  # pyright: ignore [reportArgumentType]
                self.tau,
            )
        )

        self = self.replace(
            critic=critic,
        )

        return (self, {**logs, "losses/critic_loss": critic_loss_value})

    @override
    def update(self, data: AtariReplayBufferSamples) -> tuple[Self, LogDict]:
        return self._update_inner(data)

    def _split_critic_activations(
        self, critic_acts: LayerActivationsDict
    ) -> tuple[LayerActivationsDict, ...]:
        return tuple(
            {key: value[i] for key, value in critic_acts.items()}
            for i in range(self.num_critics)
        )

    @jax.jit
    def _get_intermediates(
        self, data: AtariReplayBufferSamples
    ) -> tuple[Self, Intermediates, Intermediates]:
        key, critic_activations_key = jax.random.split(self.key, 2)

        actions_dist: distrax.Distribution
        batch_size = data.observations.shape[0]
        actions_dist, actor_state = self.actor.apply_fn(
            self.actor.params, data.observations, mutable="intermediates"
        )
        actions = actions_dist.sample(seed=critic_activations_key)

        _, critic_state = self.critic.apply_fn(
            self.critic.params, data.observations, actions, mutable="intermediates"
        )

        actor_intermediates = jax.tree.map(
            lambda x: x.reshape(batch_size, -1), actor_state["intermediates"]
        )
        critic_intermediates = jax.tree.map(
            lambda x: x.reshape(self.num_critics, batch_size, -1),
            critic_state["intermediates"]["VmapQValueFunction_0"],
        )

        self = self.replace(key=key)

        # HACK: Explicitly using the generated name of the Vmap Critic module here.
        return (
            self,
            actor_intermediates,
            critic_intermediates,
        )

    @override
    def get_metrics(
        self, metrics: Metrics, data: AtariReplayBufferSamples
    ) -> tuple[Self, LogDict]:
        self, actor_intermediates, critic_intermediates = self._get_intermediates(data)

        actor_acts = extract_activations(actor_intermediates)
        critic_acts = extract_activations(critic_intermediates)
        critic_acts = self._split_critic_activations(critic_acts)

        # TODO: None of the dormant neuron logs / srank compute are jitted at the top level
        logs: LogDict
        logs = {}
        if metrics.is_enabled(Metrics.DORMANT_NEURONS):
            logs.update(
                {
                    f"metrics/dormant_neurons_actor_{log_name}": log_value
                    for log_name, log_value in get_dormant_neuron_logs(
                        actor_acts
                    ).items()
                }
            )
        if metrics.is_enabled(Metrics.SRANK):
            for key, value in actor_acts.items():
                logs[f"metrics/srank_actor_{key}"] = compute_srank(value)

        for i, acts in enumerate(critic_acts):
            if metrics.is_enabled(Metrics.DORMANT_NEURONS):
                logs.update(
                    {
                        f"metrics/dormant_neurons_critic_{i}_{log_name}": log_value
                        for log_name, log_value in get_dormant_neuron_logs(acts).items()
                    }
                )
            if metrics.is_enabled(Metrics.SRANK):
                for key, value in acts.items():
                    logs[f"metrics/srank_critic_{i}_{key}"] = compute_srank(value)

        return self, logs
