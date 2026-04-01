"""Inspired by https://github.com/kevinzakka/robopianist-rl/blob/main/sac.py"""

import dataclasses
from functools import partial
from typing import Self, override

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
from mtrl.rl.networks import ImpalaDQN
from mtrl.types import (
    Action,
    Intermediates,
    LayerActivationsDict,
    LogDict,
    Observation,
    AtariReplayBufferSamples,
)

from mtrl.nn.augmentation import augment

from .base import OffPolicyAlgorithm
from .utils import compute_conflict_metrics, vmap_cos_sim

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
    # logits = critic.apply_fn(critic.target_params, observation, task_ids)
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
    support = jnp.linspace(v_min, v_max, n_atoms)
    q_values = jnp.sum(jax.nn.softmax(logits, axis=-1) * support, axis=-1)
    return jnp.argmax(q_values, axis=-1)
    

@dataclasses.dataclass(frozen=True)
class DrQConfig(AlgorithmConfig):
    critic_config: ImpalaDQNConfig = ImpalaDQNConfig()
    tau: float = 0.005
    v_min: float = -10.0
    v_max: float = 10.0
    n_atoms: int = 51
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 5_000
    num_tasks: int = 26


class DrQ(OffPolicyAlgorithm[DrQConfig]):
    critic: CriticTrainState
    key: PRNGKeyArray
    step: jnp.ndarray
    gamma: float = struct.field(pytree_node=False)
    tau: float = struct.field(pytree_node=False)
    nstep: int = struct.field(pytree_node=False)
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
            normalize_rewards=config.normalize_rewards,
            nstep=config.nstep,
            gamma=self.gamma,
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
        algorithm_key, critic_init_key, shrink_key = (
            jax.random.split(master_key, 3)
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
            action_dim=env_config.action_space.n,
            use_layer_norm=config.critic_config.use_layer_norm,
        )

        critic_init_params = critic_net.init(critic_init_key, dummy_obs, task_ids)
        critic = CriticTrainState.create(
            apply_fn=critic_net.apply,
            params=critic_init_params,
            target_params=critic_init_params,
            tx=config.critic_config.q_function_config.network_config.optimizer.spawn(),
        )

        # Shrink-and-perturb initialization (matches Archive)
        fresh_params = critic_net.init(shrink_key, dummy_obs, task_ids)
        shrink_rate = 0.5

        def interpolate(old_param, new_param):
            return old_param * (1 - shrink_rate) + new_param * shrink_rate

        combined_inner = {}
        for key in critic_init_params['params']:
            if 'ImpalaEncoder' in key:
                combined_inner[key] = jax.tree_util.tree_map(
                    interpolate, critic_init_params['params'][key], fresh_params['params'][key]
                )
            else:
                combined_inner[key] = fresh_params['params'][key]
        combined_params = {'params': combined_inner}
        critic = critic.replace(params=combined_params, target_params=combined_params)

        print("Critic Arch:", jax.tree_util.tree_map(jnp.shape, critic.params))
        print("Critic Params:", sum(x.size for x in jax.tree.leaves(critic.params)))

        return DrQ(
            critic=critic,
            key=algorithm_key,
            num_tasks=config.num_tasks,
            step=jnp.zeros((), dtype=jnp.int32),
            gamma=config.gamma,
            tau=config.tau,
            nstep=config.nstep if hasattr(config, 'nstep') else 3,
            v_min=config.v_min,
            v_max=config.v_max,
            n_atoms=config.n_atoms,
            eps_start=config.eps_start,
            eps_end=config.eps_end,
            eps_decay_steps=config.eps_decay_steps,
        )
    def shrink_and_perturb(self, dummy_obs: jnp.ndarray, critic_config: ImpalaDQNConfig, action_dim: int, shrink_rate: float = 0.5) -> "DrQ":
        key, shrink_key = jax.random.split(self.key)
        task_ids = jnp.arange(self.num_tasks)

        critic_net = ImpalaDQN(
            impala_config=critic_config.impala_config,
            q_function_config=critic_config.q_function_config,
            task_embed_config=critic_config.task_embed_config,
            action_dim=action_dim,
            use_layer_norm=critic_config.use_layer_norm,
        )
        fresh_params = critic_net.init(shrink_key, dummy_obs, task_ids)

        def interpolate(old_param, new_param):
            return old_param * (1 - shrink_rate) + new_param * shrink_rate

        combined_inner = {}
        for param_key in self.critic.params['params']:
            if 'ImpalaEncoder' in param_key:
                combined_inner[param_key] = jax.tree_util.tree_map(
                    interpolate, self.critic.params['params'][param_key], fresh_params['params'][param_key]
                )
            else:
                combined_inner[param_key] = fresh_params['params'][param_key]
        combined_params = {'params': combined_inner}

        critic = CriticTrainState.create(
            apply_fn=self.critic.apply_fn,
            params=combined_params,
            target_params=optax.incremental_update(combined_params, self.critic.target_params, 1),
            tx=critic_config.q_function_config.network_config.optimizer.spawn(),
        )

        return self.replace(critic=critic, key=key)

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
        key, aug_key = jax.random.split(self.key)
        aug_obs, _ = augment(observation, aug_key)
        actions, key = _sample_action(
            self.critic, aug_obs, task_ids, key,
            self.step, self.v_min, self.v_max, self.n_atoms,
            self.eps_start, self.eps_end, self.eps_decay_steps,
        )
        return self.replace(key=key, step=self.step + self.num_tasks), actions

    @override
    def eval_action(self, observation: Observation, task_ids: jnp.ndarray) -> Action:
        rng = jax.random.PRNGKey(np.random.randint(0, 2**31))
        aug_obs, _ = augment(observation, rng)
        return jax.device_get(
            _eval_action(self.critic, aug_obs, task_ids, self.v_min, self.v_max, self.n_atoms)
        )

    @jax.jit
    def _update_inner(self, data: AtariReplayBufferSamples) -> tuple[Self, LogDict]:
        # --- Critic loss ---
        print("TRACING")
        task_ids = data.task_ids

        next_obs_logits = self.critic.apply_fn(self.critic.params, data.next_observations, task_ids)
        support = jnp.linspace(self.v_min, self.v_max, self.n_atoms)
        next_q_values = jnp.sum(jax.nn.softmax(next_obs_logits, axis=-1) * support, axis=-1)

        target_logits = self.critic.apply_fn(self.critic.target_params, data.next_observations, task_ids)
        target_probs = jax.nn.softmax(target_logits, axis=-1)

        next_actions = jnp.argmax(next_q_values, axis=-1)
        target_dist = target_probs[jnp.arange(data.observations.shape[0]), next_actions]

        tz_j = jnp.clip(
            data.rewards + (self.gamma ** self.nstep) * (1 - data.dones) * support,
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
            online_logits = online_logits[jnp.arange(B), data.actions, :]
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
        key, aug_key1, aug_key2 = jax.random.split(self.key, 3)
        aug_obs, _ = augment(data.observations, aug_key1)
        aug_next_obs, _ = augment(data.next_observations, aug_key2)
        aug_data = data._replace(observations=aug_obs, next_observations=aug_next_obs)
        self = self.replace(key=key)
        return self._update_inner(aug_data)

    @jax.jit
    def compute_weights(self, data: AtariReplayBufferSamples) -> tuple[Self, LogDict]:
        B = data.observations.shape[0]
        per_task_batch = B // self.num_tasks

        # Sort by task_id and reshape to (num_tasks, per_task_batch, ...)
        sorted_idx = jnp.argsort(data.task_ids)

        def split(x):
            x_sorted = x[sorted_idx]
            return x_sorted.reshape(self.num_tasks, per_task_batch, *x.shape[1:])

        obs        = split(data.observations)       # (T, B/T, C, H, W)
        next_obs   = split(data.next_observations)  # (T, B/T, C, H, W)
        rewards    = split(data.rewards)            # (T, B/T, 1)
        actions    = split(data.actions)            # (T, B/T)
        dones      = split(data.dones)               # (T, B/T, 1)
        task_ids   = split(data.task_ids)           # (T, B/T)

        support  = jnp.linspace(self.v_min, self.v_max, self.n_atoms)
        delta_z  = (self.v_max - self.v_min) / (self.n_atoms - 1)

        def per_task_loss(params, t_obs, t_next_obs, t_rewards, t_actions, t_dones, t_task_ids):
            next_logits_online = self.critic.apply_fn(params, t_next_obs, t_task_ids)
            next_q = jnp.sum(jax.nn.softmax(next_logits_online, axis=-1) * support, axis=-1)
            next_actions = jnp.argmax(next_q, axis=-1)

            target_logits = self.critic.apply_fn(self.critic.target_params, t_next_obs, t_task_ids)
            target_probs = jax.nn.softmax(target_logits, axis=-1)
            target_dist = target_probs[jnp.arange(per_task_batch), next_actions]

            tz_j = jnp.clip(
                t_rewards + (self.gamma ** self.nstep) * (1 - t_dones) * support,
                self.v_min, self.v_max,
            )
            b = (tz_j - self.v_min) / delta_z
            l = jnp.floor(b).astype(jnp.int32)
            u = jnp.ceil(b).astype(jnp.int32)

            m = jnp.zeros((per_task_batch, self.n_atoms))
            m = m.at[jnp.arange(per_task_batch)[:, None], l].add(target_dist * (u - b))
            m = m.at[jnp.arange(per_task_batch)[:, None], u].add(target_dist * (b - l))
            m = jax.lax.stop_gradient(m)

            online_logits = self.critic.apply_fn(params, t_obs, t_task_ids)
            online_logits = online_logits[jnp.arange(per_task_batch), t_actions, :]
            log_probs = jax.nn.log_softmax(online_logits, axis=-1)
            return -(m * log_probs).sum(axis=-1).mean()

        _, critic_grads = jax.vmap(
            jax.value_and_grad(per_task_loss),
            in_axes=(None, 0, 0, 0, 0, 0, 0),
            out_axes=0,
        )(self.critic.params, obs, next_obs, rewards, actions, dones, task_ids)

        flat_critic_grads = jax.vmap(
            lambda x: jax.flatten_util.ravel_pytree(x)[0]
        )(critic_grads)

        critic_avg_cos_sim, critic_cos_sim_mat = vmap_cos_sim(flat_critic_grads, self.num_tasks)
        critic_avg_grad_magnitude = jnp.linalg.norm(flat_critic_grads, axis=1).mean()
        critic_conflict_metrics = compute_conflict_metrics(critic_cos_sim_mat, flat_critic_grads)

        return self, {
            "critic_avg_cos_sim":             critic_avg_cos_sim,
            "critic_avg_grad_magnitude":      critic_avg_grad_magnitude,
            "critic_conflict_rate":           critic_conflict_metrics["conflict_rate"],
            "critic_mean_conflict_magnitude": critic_conflict_metrics["mean_conflict_magnitude"],
            "critic_mean_conflict_angle":     critic_conflict_metrics["mean_conflict_angle"],
            "critic_per_task_conflict_rate":  critic_conflict_metrics["per_task_conflict_rate"],
            "critic_per_task_grad_magnitude": critic_conflict_metrics["per_task_grad_magnitude"],
            "critic_pairwise_conflict":       critic_conflict_metrics["pairwise_conflict"],
            "critic_pairwise_cos_sim":        critic_conflict_metrics["pairwise_cos_sim"],
            "critic_pairwise_angle":          critic_conflict_metrics["pairwise_angle"],
        }

    @jax.jit
    def _get_intermediates(
        self, data: AtariReplayBufferSamples
    ) -> tuple[Self, Intermediates]:
        _, critic_state = self.critic.apply_fn(
            self.critic.params, data.observations, data.task_ids, mutable="intermediates"
        )
        return self, critic_state["intermediates"]

    @override
    def get_metrics(
        self, metrics: Metrics, data: AtariReplayBufferSamples
    ) -> tuple[Self, LogDict]:
        self, critic_intermediates = self._get_intermediates(data)

        critic_acts = extract_activations(critic_intermediates)

        logs: LogDict = {}
        if metrics.is_enabled(Metrics.DORMANT_NEURONS):
            logs.update(
                {
                    f"metrics/dormant_neurons_critic_0_{log_name}": log_value
                    for log_name, log_value in get_dormant_neuron_logs(critic_acts).items()
                }
            )
        if metrics.is_enabled(Metrics.SRANK):
            for key, value in critic_acts.items():
                logs[f"metrics/srank_critic_0_{key}"] = compute_srank(value)

        return self, logs
