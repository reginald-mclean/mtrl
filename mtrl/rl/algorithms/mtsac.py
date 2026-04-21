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
import numpy.typing as npt
import optax
from flax import struct
from flax.core import FrozenDict
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from mtrl.config.networks import ContinuousActionPolicyConfig, QValueFunctionConfig, BroQConfig, BroActorConfig
from mtrl.config.optim import OptimizerConfig
from mtrl.config.rl import AlgorithmConfig
from mtrl.config.utils import Metrics
from mtrl.envs import EnvConfig
from mtrl.monitoring.metrics import (
    compute_srank,
    extract_activations,
    get_dormant_neuron_logs,
)
from mtrl.optim.pcgrad import PCGradState
from mtrl.optim.gradnorm import GradNormState
from mtrl.optim.cagrad import CAGradState

from mtrl.rl.networks import ContinuousActionPolicy, Ensemble, QValueFunction, BRCActor, BRCCritic
from mtrl.types import (
    Action,
    Intermediates,
    LayerActivationsDict,
    LogDict,
    Observation,
    ReplayBufferSamples,
)

from .base import OffPolicyAlgorithm
from .utils import TrainState, vmap_cos_sim, compute_conflict_metrics


class MultiTaskTemperature(nn.Module):
    num_tasks: int
    initial_temperature: float = 1.0

    def setup(self):
        self.log_alpha = self.param(
            "log_alpha",
            init_fn=lambda _: jnp.full(
                (self.num_tasks,), jnp.log(self.initial_temperature)
            ),
        )

    def __call__(
        self, task_ids: Float[Array, "... num_tasks"]
    ) -> Float[Array, "... 1"]:
        return jnp.exp(task_ids @ self.log_alpha.reshape(-1, 1))


class CriticTrainState(TrainState):
    target_params: FrozenDict | None = None


@jax.jit
def _sample_action(
    actor: TrainState, observation: Observation, key: PRNGKeyArray
) -> tuple[Float[Array, "... action_dim"], PRNGKeyArray]:
    key, action_key = jax.random.split(key)
    dist = actor.apply_fn(actor.params, observation)
    action = dist.sample(seed=action_key)
    return action, key


@jax.jit
def _eval_action(
    actor: TrainState, observation: Observation
) -> Float[Array, "... action_dim"]:
    return actor.apply_fn(actor.params, observation).mode()

@jax.jit
def _sample_action_bro(
    actor: TrainState, observation: Observation, task_ids: jax.Array, key: PRNGKeyArray
) -> tuple[Float[Array, "... action_dim"], PRNGKeyArray]:
    key, action_key = jax.random.split(key)
    dist = actor.apply_fn(actor.params, observation, task_ids)
    action = dist.sample(seed=action_key)
    return action, key


@jax.jit
def _eval_action_bro(
    actor: TrainState, observation: Observation, task_ids: jax.Array
) -> Float[Array, "... action_dim"]:
    return actor.apply_fn(actor.params, observation, task_ids).mode()


def extract_task_weights(
    alpha_params: FrozenDict, task_ids: Float[np.ndarray, "... num_tasks"]
) -> Float[Array, "... 1"]:
    log_alpha: jax.Array
    task_weights: jax.Array

    log_alpha = alpha_params["params"]["log_alpha"]  # pyright: ignore [reportAssignmentType]
    task_weights = jax.nn.softmax(-log_alpha)
    task_weights = task_ids @ task_weights.reshape(-1, 1)  # pyright: ignore [reportAssignmentType]
    task_weights *= log_alpha.shape[0]
    return task_weights


@dataclasses.dataclass(frozen=True)
class MTSACConfig(AlgorithmConfig):
    actor_config: ContinuousActionPolicyConfig = ContinuousActionPolicyConfig()
    critic_config: QValueFunctionConfig = QValueFunctionConfig()
    temperature_optimizer_config: OptimizerConfig = OptimizerConfig(max_grad_norm=None)
    initial_temperature: float = 1.0
    num_critics: int = 2
    tau: float = 0.005
    use_task_weights: bool = False
    v_min: float = -10.0
    v_max: float = 10.0
    n_atoms: int = 51


class MTSAC(OffPolicyAlgorithm[MTSACConfig]):
    actor: TrainState
    critic: CriticTrainState
    alpha: TrainState
    key: PRNGKeyArray
    gamma: float = struct.field(pytree_node=False)
    tau: float = struct.field(pytree_node=False)
    target_entropy: float = struct.field(pytree_node=False)
    use_task_weights: bool = struct.field(pytree_node=False)
    split_actor_losses: bool = struct.field(pytree_node=False)
    split_critic_losses: bool = struct.field(pytree_node=False)
    num_critics: int = struct.field(pytree_node=False)
    return_split_actor_losses: bool = struct.field(pytree_node=False)
    return_split_critic_losses: bool = struct.field(pytree_node=False)
    explore: bool = struct.field(pytree_node=False)
    clip: bool = struct.field(pytree_node=False)
    actor_network_type: str = struct.field(pytree_node=False)
    critic_network_type: str = struct.field(pytree_node=False)
    v_min: float = struct.field(pytree_node=False)
    v_max: float = struct.field(pytree_node=False)
    n_atoms: int = struct.field(pytree_node=False)

    @override
    @staticmethod
    def initialize(
        config: MTSACConfig, env_config: EnvConfig, seed: int = 1
    ) -> "MTSAC":
        assert isinstance(env_config.action_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )
        assert isinstance(env_config.observation_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )

        master_key = jax.random.PRNGKey(seed)
        algorithm_key, actor_init_key, critic_init_key, alpha_init_key = (
            jax.random.split(master_key, 4)
        )
        
        actor_network_type = 'vanilla'
        if isinstance(config.actor_config, BroActorConfig):
            actor_net = BRCActor(
                bro_config = config.actor_config.bro_config,
                actor_config=config.actor_config.actor_config,
                task_embed_config=config.actor_config.task_embed_config,
                action_dim=int(np.prod(env_config.action_space.shape)),
            )
            dummy_obs = jnp.array(
                [env_config.observation_space.sample() for _ in range(config.num_tasks)]
            )

            task_ids = jnp.arange(config.num_tasks)

            actor = TrainState.create(
                apply_fn=actor_net.apply,
                params=actor_net.init(actor_init_key, dummy_obs, task_ids),
                tx=config.actor_config.actor_config.network_config.optimizer.spawn(),
            )
            actor_network_type = 'bro'

        else:
            actor_net = ContinuousActionPolicy(
                int(np.prod(env_config.action_space.shape)), config=config.actor_config
            )
            dummy_obs = jnp.array(
                [env_config.observation_space.sample() for _ in range(config.num_tasks)]
            )
            actor = TrainState.create(
                apply_fn=actor_net.apply,
                params=actor_net.init(actor_init_key, dummy_obs),
                tx=config.actor_config.network_config.optimizer.spawn(),
            )

        print("Actor Arch:", jax.tree_util.tree_map(jnp.shape, actor.params))
        print("Actor Params:", sum(x.size for x in jax.tree.leaves(actor.params)))

        critic_network_type = 'mse'
        n_atoms = config.n_atoms
        if isinstance(config.critic_config, BroQConfig):

            critic_cls = partial(BRCCritic, bro_config=config.critic_config.bro_config, task_embed_config=config.critic_config.task_embed_config, q_function_config=config.critic_config.q_function_config)
            critic_net = Ensemble(critic_cls, num=config.num_critics)
            dummy_action = jnp.array(
                [env_config.action_space.sample() for _ in range(config.num_tasks)]
            )
            dummy_task_ids = jnp.arange(config.num_tasks)

            critic_init_params = critic_net.init(critic_init_key, dummy_obs, dummy_action, dummy_task_ids)
            critic = CriticTrainState.create(
                apply_fn=critic_net.apply,
                params=critic_init_params,
                target_params=critic_init_params,
                tx=config.critic_config.q_function_config.network_config.optimizer.spawn(),
            )
            critic_network_type = 'c51'
            n_atoms = config.critic_config.q_function_config.num_atoms

        else:
            if config.critic_config.use_classification:
                critic_network_type = 'c51'
                n_atoms = config.critic_config.num_atoms

            critic_cls = partial(QValueFunction, config=config.critic_config, action_dim=int(np.prod(env_config.action_space.shape)) if config.critic_config.use_classification else None)
            critic_net = Ensemble(critic_cls, num=config.num_critics)
            dummy_action = jnp.array(
                [env_config.action_space.sample() for _ in range(config.num_tasks)]
            )
            critic_init_params = critic_net.init(critic_init_key, dummy_obs, dummy_action)
            critic = CriticTrainState.create(
                apply_fn=critic_net.apply,
                params=critic_init_params,
                target_params=critic_init_params,
                tx=config.critic_config.network_config.optimizer.spawn(),
            )
        

        print("Critic Arch:", jax.tree_util.tree_map(jnp.shape, critic.params))
        print("Critic Params:", sum(x.size for x in jax.tree.leaves(critic.params)))

        alpha_net = MultiTaskTemperature(config.num_tasks, config.initial_temperature)
        dummy_task_ids = jnp.array(
            [np.ones((config.num_tasks,)) for _ in range(config.num_tasks)]
        )
        alpha = TrainState.create(
            apply_fn=alpha_net.apply,
            params=alpha_net.init(alpha_init_key, dummy_task_ids),
            tx=config.temperature_optimizer_config.spawn(),
        )

        target_entropy = -np.prod(env_config.action_space.shape).item()

        

        return MTSAC(
            num_tasks=config.num_tasks,
            actor=actor,
            critic=critic,
            alpha=alpha,
            key=algorithm_key,
            gamma=config.gamma,
            tau=config.tau,
            target_entropy=target_entropy,
            use_task_weights=config.use_task_weights,
            num_critics=config.num_critics,
            split_actor_losses=False if isinstance(config.actor_config, BroActorConfig) else config.actor_config.network_config.optimizer.requires_split_task_losses,
            split_critic_losses=False if isinstance(config.critic_config, BroQConfig) else config.critic_config.network_config.optimizer.requires_split_task_losses,
            return_split_actor_losses=False if isinstance(config.actor_config, BroActorConfig) else config.weights_actor_loss,
            return_split_critic_losses=False if isinstance(config.critic_config, BroQConfig) else config.weights_critic_loss or config.weights_qf_vals,
            explore=False,
            clip=config.clip,
            actor_network_type=actor_network_type,
            critic_network_type=critic_network_type,
            v_min=config.v_min,
            v_max=config.v_max,
            n_atoms=n_atoms,
        )

    def reset(self, env_mask) -> None:
        pass

    @override
    def get_num_params(self) -> dict[str, int]:
        return {
            "actor_num_params": sum(x.size for x in jax.tree.leaves(self.actor.params)),
            "critic_num_params": sum(
                x.size for x in jax.tree.leaves(self.critic.params)
            ),
        }

    @override
    def sample_action(self, observation: Observation, task_ids: jax.Array = None) -> tuple[Self, Action]:
        if self.actor_network_type == 'vanilla':
            action, key = _sample_action(self.actor, observation, self.key)
        elif self.actor_network_type == 'bro':
            action, key = _sample_action_bro(self.actor, observation, task_ids, self.key)
        return self.replace(key=key), jax.device_get(action)

    @override
    def eval_action(self, observations: Observation, task_ids: jax.Array = None) -> Action:
        if self.actor_network_type == 'vanilla':
            return jax.device_get(_eval_action(self.actor, observations))
        elif self.actor_network_type == 'bro':
            return jax.device_get(_eval_action_bro(self.actor, observations, task_ids))

    def split_data_by_tasks(
        self,
        data: PyTree[Float[Array, "batch data_dim"]],
        task_ids: Float[npt.NDArray, "batch num_tasks"],
    ) -> PyTree[Float[Array, "num_tasks per_task_batch data_dim"]]:
        tasks = jnp.argmax(task_ids, axis=1)
        sorted_indices = jnp.argsort(tasks)

        def group_by_task_leaf(
            leaf: Float[Array, "batch data_dim"],
        ) -> Float[Array, "task task_batch data_dim"]:
            leaf_sorted = leaf[sorted_indices]
            return leaf_sorted.reshape(self.num_tasks, -1, leaf.shape[1])

        return jax.tree.map(group_by_task_leaf, data), sorted_indices

    def unsplit_data_by_tasks(
        self,
        split_data: PyTree[Float[Array, "num_tasks per_task_batch data_dim"]],
        sort_indices: jax.Array,
    ) -> PyTree[Float[Array, "batch data_dim"]]:
        def reconstruct_leaf(
            leaf: Float[Array, "num_tasks per_task_batch data_dim"],
        ) -> Float[Array, "batch data_dim"]:
            batch_size = leaf.shape[0] * leaf.shape[1]
            flat = leaf.reshape(batch_size, leaf.shape[-1])
            # Create inverse permutation
            inverse_indices = jnp.zeros_like(sort_indices)
            inverse_indices = inverse_indices.at[sort_indices].set(
                jnp.arange(batch_size)
            )
            return flat[inverse_indices]

        return jax.tree.map(reconstruct_leaf, split_data)

    def update_critic(
        self,
        data: ReplayBufferSamples,
        alpha_val: Float[Array, "*batch 1"],
        task_weights: Float[Array, "*batch 1"] | None = None,
    ) -> tuple[Self, LogDict]:
        key, critic_loss_key = jax.random.split(self.key)
        if self.critic_network_type == 'c51':
            # --- C51 distributional critic update ---
            if self.actor_network_type == 'vanilla':
                next_actions, next_action_log_probs = self.actor.apply_fn(
                    self.actor.params, data.next_observations
                ).sample_and_log_prob(seed=critic_loss_key)

                target_logits = self.critic.apply_fn(
                    self.critic.target_params, data.next_observations, next_actions
                )  # [num_critics, batch, n_atoms]

            elif self.actor_network_type == 'bro':
                next_actions, next_action_log_probs = self.actor.apply_fn(
                    self.actor.params, data.next_observations, data.task_ids
                ).sample_and_log_prob(seed=critic_loss_key)

                target_logits = self.critic.apply_fn(
                    self.critic.target_params, data.next_observations, next_actions, data.task_ids
                )  # [num_critics, batch, n_atoms]

            support = jnp.linspace(self.v_min, self.v_max, self.n_atoms)
            target_probs = jax.nn.softmax(target_logits, axis=-1)

            # Min expected Q across critics (pessimistic)
            expected_Q = jnp.sum(target_probs * support, axis=-1)  # [num_critics, batch]
            min_Q_next = jnp.min(expected_Q, axis=0)  # [batch]

            # Soft backup target
            next_action_log_probs = next_action_log_probs.reshape(-1)
            target_value = min_Q_next - alpha_val.reshape(-1) * next_action_log_probs
            scalar_target = data.rewards.reshape(-1) + (1 - data.dones.reshape(-1)) * self.gamma * target_value

            # Project scalar target onto support (Dirac delta projection)
            delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
            tz = jnp.clip(scalar_target, self.v_min, self.v_max)
            b = (tz - self.v_min) / delta_z
            l = jnp.floor(b).astype(jnp.int32)
            u = jnp.ceil(b).astype(jnp.int32)

            B = data.observations.shape[0]
            m = jnp.zeros((B, self.n_atoms))
            m = m.at[jnp.arange(B), l].add(u - b)
            m = m.at[jnp.arange(B), u].add(b - l)
            m = jax.lax.stop_gradient(m)

            def critic_loss_c51(
                params: FrozenDict,
            ) -> tuple[Float[Array, ""], Float[Array, ""]]:
                
                online_logits = self.critic.apply_fn(
                    params, data.observations, data.actions, # data.task_ids
                )  # [num_critics, batch, n_atoms]
                log_probs = jax.nn.log_softmax(online_logits, axis=-1)
                if task_weights is not None:
                    loss = -(task_weights.reshape(1, -1, 1) * m * log_probs).sum(axis=-1).mean()
                else:
                    loss = -(m * log_probs).sum(axis=-1).mean()
                q_vals = jnp.sum(jax.nn.softmax(online_logits, axis=-1) * support, axis=-1)
                return loss, q_vals.mean()

            (critic_loss_value, qf_values), critic_grads = jax.value_and_grad(
                critic_loss_c51, has_aux=True
            )(self.critic.params)
            flat_grads, _ = flatten_util.ravel_pytree(critic_grads)

            key, optimizer_key = jax.random.split(key)
            critic = self.critic.apply_gradients(
                grads=critic_grads,
                optimizer_extra_args={
                    "task_losses": critic_loss_value,
                    "key": optimizer_key,
                },
            )
            critic = critic.replace(
                target_params=optax.incremental_update(
                    critic.params,
                    critic.target_params,  # pyright: ignore [reportArgumentType]
                    self.tau,
                )
            )
            flat_params_crit, _ = flatten_util.ravel_pytree(critic.params)

            return self.replace(critic=critic, key=key), {
                "losses/qf_values": qf_values,
                "losses/qf_loss": critic_loss_value,
                "metrics/critic_grad_magnitude": jnp.linalg.norm(flat_grads),
                "metrics/critic_params_norm": jnp.linalg.norm(flat_params_crit),
            }

        # --- MSE critic update ---
        # Sample a'
        if self.split_critic_losses:
            next_actions, next_action_log_probs = jax.vmap(
                lambda x: self.actor.apply_fn(self.actor.params, x).sample_and_log_prob(
                    seed=critic_loss_key
                )
            )(data.observations)
            q_values = jax.vmap(self.critic.apply_fn, in_axes=(None, 0, 0))(
                self.critic.target_params, data.next_observations, next_actions
            )
        else:
            if self.actor_network_type == 'vanilla':
                next_actions, next_action_log_probs = self.actor.apply_fn(
                    self.actor.params, data.next_observations
                ).sample_and_log_prob(seed=critic_loss_key)
            elif self.actor_network_type == 'bro':
                next_actions, next_action_log_probs = self.actor.apply_fn(
                    self.actor.params, data.next_observations, data.task_ids
                ).sample_and_log_prob(seed=critic_loss_key)
            q_values = self.critic.apply_fn(
                self.critic.target_params, data.next_observations, next_actions
            )

        def critic_loss(
            params: FrozenDict,
            _data: ReplayBufferSamples,
            _q_values: Float[Array, "#batch 1"],
            _alpha_val: Float[Array, "#batch 1"],
            _next_action_log_probs: Float[Array, " #batch"],
            _task_weights: Float[Array, "#batch 1"] | None = None,
        ) -> tuple[Float[Array, ""], Float[Array, ""]]:
            # next_action_log_probs is (B,) shaped because of the sum(axis=1), while Q values are (B, 1)
            min_qf_next_target = jnp.min(
                _q_values, axis=0
            ) - _alpha_val * _next_action_log_probs.reshape(-1, 1)

            next_q_value = jax.lax.stop_gradient(
                _data.rewards + (1 - _data.dones) * self.gamma * min_qf_next_target
            )

            q_pred = self.critic.apply_fn(params, _data.observations, _data.actions)

            # HACK: Clipping Q values to approximate theoretical maximum for Metaworld
            if self.clip: # or (not isinstance(self.actor.opt_state[0], PCGradState) and not isinstance(self.actor.opt_state[0], GradNormState)):
                next_q_value = jnp.clip(next_q_value, -5000, 5000)
                q_pred = jnp.clip(q_pred, -5000, 5000)

            if _task_weights is not None:
                loss = (_task_weights * (q_pred - next_q_value) ** 2).mean()
            else:
                loss = ((q_pred - next_q_value) ** 2).mean()
            return loss, q_pred.mean()

        if self.split_critic_losses:
            (critic_loss_value, qf_values), critic_grads = jax.vmap(
                jax.value_and_grad(critic_loss, has_aux=True),
                in_axes=(None, 0, 0, 0, 0, 0),
                out_axes=0,
            )(
                self.critic.params,
                data,
                q_values,
                alpha_val,
                next_action_log_probs,
                task_weights,
            )
            if not isinstance(self.actor.opt_state[0], PCGradState) and not isinstance(self.actor.opt_state[0], GradNormState) and not isinstance(self.actor.opt_state[0], CAGradState):
                critic_grads = jax.tree.map(lambda x: x.mean(axis=0), critic_grads)
            flat_grads, _ = flatten_util.ravel_pytree(
                jax.tree.map(lambda x: x.mean(axis=0), critic_grads)
            )
        else:
            (critic_loss_value, qf_values), critic_grads = jax.value_and_grad(
                critic_loss, has_aux=True
            )(
                self.critic.params,
                data,
                q_values,
                alpha_val,
                next_action_log_probs,
                task_weights,
            )
            flat_grads, _ = flatten_util.ravel_pytree(critic_grads)

        key, optimizer_key = jax.random.split(key)
        critic = self.critic.apply_gradients(
            grads=critic_grads,
            optimizer_extra_args={
                "task_losses": critic_loss_value,
                "key": optimizer_key,
            },
        )
        critic = critic.replace(
            target_params=optax.incremental_update(
                critic.params,
                critic.target_params,  # pyright: ignore [reportArgumentType]
                self.tau,
            )
        )
        flat_params_crit, _ = flatten_util.ravel_pytree(critic.params)

        return self.replace(critic=critic, key=key), {
            "losses/qf_values": qf_values.mean(),
            "losses/qf_loss": critic_loss_value.mean(),
            "metrics/critic_grad_magnitude": jnp.linalg.norm(flat_grads),
            "metrics/critic_params_norm": jnp.linalg.norm(flat_params_crit),
        }

    def update_actor(
        self,
        data: ReplayBufferSamples,
        alpha_val: Float[Array, "batch 1"],
        task_weights: Float[Array, "batch 1"] | None = None,
    ) -> tuple[Self, Float[Array, " batch"], LogDict]:
        key, actor_loss_key = jax.random.split(self.key)

        def actor_loss(
            params: FrozenDict,
            _data: ReplayBufferSamples,
            _alpha_val: Float[Array, "batch 1"],
            _task_weights: Float[Array, "batch 1"] | None = None,
            _explore: bool = True
        ):
            if self.actor_network_type == 'vanilla':
                action_samples, log_probs = self.actor.apply_fn(
                    params, _data.observations
                ).sample_and_log_prob(seed=actor_loss_key)
            elif self.actor_network_type == 'bro':
                action_samples, log_probs = self.actor.apply_fn(
                    params, _data.observations, _data.task_ids
                ).sample_and_log_prob(seed=actor_loss_key)


            log_probs = log_probs.reshape(-1, 1)

            if self.critic_network_type == 'c51':
                logits = self.critic.apply_fn(
                    self.critic.params, _data.observations, action_samples, # _data.task_ids
                )
                support = jnp.linspace(self.v_min, self.v_max, self.n_atoms)
                q_values = jnp.sum(jax.nn.softmax(logits, axis=-1) * support, axis=-1)  # [num_critics, batch]
                min_qf_values = jnp.min(q_values, axis=0).reshape(-1, 1)
            else:
                q_values = self.critic.apply_fn(
                    self.critic.params, _data.observations, action_samples
                )
                min_qf_values = jnp.min(q_values, axis=0)
            if _task_weights is not None:
                loss = (task_weights * (_alpha_val * log_probs - min_qf_values)).mean()
            else:
                loss = (_alpha_val * log_probs - min_qf_values).mean()

            if _explore:
                exp_loss = jnp.mean(jnp.square(_data.actions - action_samples))
            else:
                exp_loss = 0.0

            loss -= exp_loss

            return loss, (log_probs, exp_loss)

        if self.split_actor_losses:
            (actor_loss_value, (log_probs, exp_loss)), actor_grads = jax.vmap(
                jax.value_and_grad(actor_loss, has_aux=True),
                in_axes=(None, 0, 0, 0),
                out_axes=0,
            )(self.actor.params, data, alpha_val, task_weights)
            if not isinstance(self.actor.opt_state[0], PCGradState) and not isinstance(self.actor.opt_state[0], GradNormState) and not isinstance(self.actor.opt_state[0], CAGradState):
                actor_grads = jax.tree.map(lambda x: x.mean(axis=0), actor_grads)
            flat_grads, _ = flatten_util.ravel_pytree(
                jax.tree.map(lambda x: x.mean(axis=0), actor_grads)
            )
        else:
            (actor_loss_value, (log_probs, exp_loss)), actor_grads = jax.value_and_grad(
                actor_loss, has_aux=True
            )(self.actor.params, data, alpha_val, task_weights, self.explore)
            flat_grads, _ = flatten_util.ravel_pytree(actor_grads)

        key, optimizer_key = jax.random.split(key)
        actor = self.actor.apply_gradients(
            grads=actor_grads,
            optimizer_extra_args={
                "task_losses": actor_loss_value,
                "key": optimizer_key,
            },
        )

        flat_params_act, _ = flatten_util.ravel_pytree(actor.params)
        logs = {
            "losses/actor_loss": actor_loss_value.mean(),
            "metrics/actor_grad_magnitude": jnp.linalg.norm(flat_grads),
            "metrics/actor_params_norm": jnp.linalg.norm(flat_params_act),
            "metrics/explore_loss": exp_loss
        }

        return (self.replace(actor=actor, key=key), log_probs, logs)

    def update_alpha(
        self,
        log_probs: Float[Array, " batch"],
        task_ids: Float[npt.NDArray, " batch num_tasks"],
    ) -> tuple[Self, LogDict]:
        def alpha_loss(params: FrozenDict) -> Float[Array, ""]:
            log_alpha: jax.Array
            log_alpha = task_ids @ params["params"]["log_alpha"].reshape(-1, 1)  # pyright: ignore [reportAttributeAccessIssue]
            return (-log_alpha * (log_probs + self.target_entropy)).mean()

        alpha_loss_value, alpha_grads = jax.value_and_grad(alpha_loss)(
            self.alpha.params
        )
        alpha = self.alpha.apply_gradients(grads=alpha_grads)

        return self.replace(alpha=alpha), {
            "losses/alpha_loss": alpha_loss_value,
            "alpha": jnp.exp(alpha.params["params"]["log_alpha"]).sum(),  # pyright: ignore [reportArgumentType]
        }

    @jax.jit
    def compute_weights(self, data: ReplayBufferSamples) -> tuple[Self, LogDict]:
        task_ids = data.observations[..., -self.num_tasks :]
        alpha_vals = self.alpha.apply_fn(self.alpha.params, task_ids)
        if self.use_task_weights:
            task_weights = extract_task_weights(self.alpha.params, task_ids)
        else:
            task_weights = None

        split_data, _ = self.split_data_by_tasks(data, task_ids)
        split_alpha_vals, alpha_val_indices = self.split_data_by_tasks(
            alpha_vals, task_ids
        )
        split_task_weights, _ = (
            self.split_data_by_tasks(task_weights, task_ids)
            if task_weights is not None
            else (None, None)
        )

        # --- Critic ---
        key, critic_loss_key = jax.random.split(self.key)
        self = self.replace(key=key)

        if self.critic_network_type == 'c51':
            support = jnp.linspace(self.v_min, self.v_max, self.n_atoms)
            delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

            def critic_loss_c51(
                params, _data, _alpha_val, _task_weights=None
            ):
                if self.actor_network_type == 'bro':
                    next_actions, next_action_log_probs = self.actor.apply_fn(
                        self.actor.params, _data.next_observations, _data.task_ids
                    ).sample_and_log_prob(seed=critic_loss_key)
                else:
                    next_actions, next_action_log_probs = self.actor.apply_fn(
                        self.actor.params, _data.next_observations
                    ).sample_and_log_prob(seed=critic_loss_key)

                target_logits = self.critic.apply_fn(
                    self.critic.target_params, _data.next_observations, next_actions, _data.task_ids
                )
                target_probs = jax.nn.softmax(target_logits, axis=-1)
                expected_Q = jnp.sum(target_probs * support, axis=-1)
                min_Q_next = jnp.min(expected_Q, axis=0)

                next_lp = next_action_log_probs.reshape(-1)
                target_value = min_Q_next - _alpha_val.reshape(-1) * next_lp
                scalar_target = _data.rewards.reshape(-1) + (1 - _data.dones.reshape(-1)) * self.gamma * target_value

                tz = jnp.clip(scalar_target, self.v_min, self.v_max)
                b = (tz - self.v_min) / delta_z
                l = jnp.floor(b).astype(jnp.int32)
                u = jnp.ceil(b).astype(jnp.int32)

                per_task_batch = _data.observations.shape[0]
                m = jnp.zeros((per_task_batch, self.n_atoms))
                m = m.at[jnp.arange(per_task_batch), l].add(u - b)
                m = m.at[jnp.arange(per_task_batch), u].add(b - l)
                m = jax.lax.stop_gradient(m)

                online_logits = self.critic.apply_fn(
                    params, _data.observations, _data.actions, _data.task_ids
                )
                log_probs = jax.nn.log_softmax(online_logits, axis=-1)
                if _task_weights is not None:
                    loss = -(_task_weights.reshape(1, -1, 1) * m * log_probs).sum(axis=-1).mean()
                else:
                    loss = -(m * log_probs).sum(axis=-1).mean()
                return loss

            _, critic_grads = jax.vmap(
                jax.value_and_grad(critic_loss_c51),
                in_axes=(None, 0, 0, 0),
                out_axes=0,
            )(
                self.critic.params,
                split_data,
                split_alpha_vals,
                split_task_weights,
            )
        else:
            next_actions, next_action_log_probs = jax.vmap(
                lambda x: self.actor.apply_fn(self.actor.params, x).sample_and_log_prob(
                    seed=critic_loss_key
                )
            )(split_data.observations)
            q_values = jax.vmap(self.critic.apply_fn, in_axes=(None, 0, 0))(
                self.critic.target_params, split_data.next_observations, next_actions
            )

            def critic_loss(
                params, _data, _q_values, _alpha_val, _next_action_log_probs, _task_weights=None
            ):
                min_qf_next_target = jnp.min(
                    _q_values, axis=0
                ) - _alpha_val * _next_action_log_probs.reshape(-1, 1)
                next_q_value = jax.lax.stop_gradient(
                    _data.rewards + (1 - _data.dones) * self.gamma * min_qf_next_target
                )
                q_pred = self.critic.apply_fn(params, _data.observations, _data.actions)
                if self.clip:
                    next_q_value = jnp.clip(next_q_value, -5000, 5000)
                    q_pred = jnp.clip(q_pred, -5000, 5000)
                if _task_weights is not None:
                    loss = (_task_weights * (q_pred - next_q_value) ** 2).mean()
                else:
                    loss = ((q_pred - next_q_value) ** 2).mean()
                return loss, q_pred.mean()

            (_, _), critic_grads = jax.vmap(
                jax.value_and_grad(critic_loss, has_aux=True),
                in_axes=(None, 0, 0, 0, 0, 0),
                out_axes=0,
            )(
                self.critic.params,
                split_data,
                q_values,
                split_alpha_vals,
                next_action_log_probs,
                split_task_weights,
            )
        flat_critic_grads = jax.vmap(
            lambda x: jax.flatten_util.ravel_pytree(x)[0]
        )(critic_grads)
        critic_avg_cos_sim, critic_cos_sim_mat = vmap_cos_sim(flat_critic_grads, self.num_tasks)
        critic_avg_grad_magnitude = jnp.linalg.norm(flat_critic_grads, axis=1).mean()
        critic_conflict_metrics = compute_conflict_metrics(critic_cos_sim_mat, flat_critic_grads)

        # --- Actor ---
        key, actor_loss_key = jax.random.split(self.key)
        self = self.replace(key=key)

        def actor_loss(params, _data, _alpha_val, _task_weights=None):
            action_samples, log_probs = self.actor.apply_fn(
                params, _data.observations
            ).sample_and_log_prob(seed=actor_loss_key)
            log_probs = log_probs.reshape(-1, 1)
            q_values = self.critic.apply_fn(
                self.critic.params, _data.observations, action_samples
            )
            min_qf_values = jnp.min(q_values, axis=0)
            if _task_weights is not None:
                loss = (_task_weights * (_alpha_val * log_probs - min_qf_values)).mean()
            else:
                loss = (_alpha_val * log_probs - min_qf_values).mean()
            return loss, log_probs

        (_, _), actor_grads = jax.vmap(
            jax.value_and_grad(actor_loss, has_aux=True),
            in_axes=(None, 0, 0, 0),
            out_axes=0,
        )(self.actor.params, split_data, split_alpha_vals, split_task_weights)

        flat_actor_grads = jax.vmap(
            lambda x: jax.flatten_util.ravel_pytree(x)[0]
        )(actor_grads)
        actor_avg_cos_sim, actor_cos_sim_mat = vmap_cos_sim(flat_actor_grads, self.num_tasks)
        actor_avg_grad_magnitude = jnp.linalg.norm(flat_actor_grads, axis=1).mean()
        actor_conflict_metrics = compute_conflict_metrics(actor_cos_sim_mat, flat_actor_grads)

        return self, {
            # Existing metrics
            "critic_avg_cos_sim":                   critic_avg_cos_sim,
            "critic_avg_grad_magnitude":             critic_avg_grad_magnitude,
            "actor_avg_cos_sim":                     actor_avg_cos_sim,
            "actor_avg_grad_magnitude":              actor_avg_grad_magnitude,
            # Existing critic conflict metrics
            "critic_conflict_rate":                  critic_conflict_metrics["conflict_rate"],
            "critic_mean_conflict_magnitude":        critic_conflict_metrics["mean_conflict_magnitude"],
            "critic_mean_conflict_angle":            critic_conflict_metrics["mean_conflict_angle"],
            "critic_per_task_conflict_rate":         critic_conflict_metrics["per_task_conflict_rate"],
            "critic_per_task_grad_magnitude":        critic_conflict_metrics["per_task_grad_magnitude"],
            # Existing actor conflict metrics
            "actor_conflict_rate":                   actor_conflict_metrics["conflict_rate"],
            "actor_mean_conflict_magnitude":         actor_conflict_metrics["mean_conflict_magnitude"],
            "actor_mean_conflict_angle":             actor_conflict_metrics["mean_conflict_angle"],
            "actor_per_task_conflict_rate":          actor_conflict_metrics["per_task_conflict_rate"],
            "actor_per_task_grad_magnitude":         actor_conflict_metrics["per_task_grad_magnitude"],
            # Existing pairwise metrics
            "critic_pairwise_conflict":              critic_conflict_metrics["pairwise_conflict"],
            "critic_pairwise_cos_sim":               critic_conflict_metrics["pairwise_cos_sim"],
            "critic_pairwise_angle":                 critic_conflict_metrics["pairwise_angle"],
            "actor_pairwise_conflict":               actor_conflict_metrics["pairwise_conflict"],
            "actor_pairwise_cos_sim":                actor_conflict_metrics["pairwise_cos_sim"],
            "actor_pairwise_angle":                  actor_conflict_metrics["pairwise_angle"],
            # New elementwise critic metrics
            "critic_avg_interference_rate":          critic_conflict_metrics["avg_interference_rate"],
            "critic_interference_asymmetry":         critic_conflict_metrics["interference_asymmetry"],
            "critic_per_task_interference_in":       critic_conflict_metrics["per_task_interference_in"],
            "critic_per_task_interference_out":      critic_conflict_metrics["per_task_interference_out"],
            "critic_pairwise_interference_rate":     critic_conflict_metrics["pairwise_interference_rate"],
            "critic_avg_participation_ratio":        critic_conflict_metrics["avg_participation_ratio"],
            "critic_per_task_participation_ratio":   critic_conflict_metrics["per_task_participation_ratio"],
            "critic_effective_rank":                 critic_conflict_metrics["effective_rank"],
            # New elementwise actor metrics
            "actor_avg_interference_rate":           actor_conflict_metrics["avg_interference_rate"],
            "actor_interference_asymmetry":          actor_conflict_metrics["interference_asymmetry"],
            "actor_per_task_interference_in":        actor_conflict_metrics["per_task_interference_in"],
            "actor_per_task_interference_out":       actor_conflict_metrics["per_task_interference_out"],
            "actor_pairwise_interference_rate":      actor_conflict_metrics["pairwise_interference_rate"],
            "actor_avg_participation_ratio":         actor_conflict_metrics["avg_participation_ratio"],
            "actor_per_task_participation_ratio":    actor_conflict_metrics["per_task_participation_ratio"],
            "actor_effective_rank":                  actor_conflict_metrics["effective_rank"],
        }

    @jax.jit
    def _update_inner(self, data: ReplayBufferSamples) -> tuple[Self, LogDict]:
        task_ids = data.observations[..., -self.num_tasks :]

        alpha_vals = self.alpha.apply_fn(self.alpha.params, task_ids)
        if self.use_task_weights:
            task_weights = extract_task_weights(self.alpha.params, task_ids)
        else:
            task_weights = None

        actor_data = critic_data = data
        actor_alpha_vals = critic_alpha_vals = alpha_vals
        actor_task_weights = critic_task_weights = task_weights
        alpha_val_indices = None

        if self.split_critic_losses or self.split_actor_losses:
            split_data, _ = self.split_data_by_tasks(data, task_ids)
            split_alpha_vals, alpha_val_indices = self.split_data_by_tasks(
                alpha_vals, task_ids
            )
            split_task_weights, _ = (
                self.split_data_by_tasks(task_weights, task_ids)
                if task_weights is not None
                else (None, None)
            )

            if self.split_critic_losses:
                critic_data = split_data
                critic_alpha_vals = split_alpha_vals
                critic_task_weights = split_task_weights

            if self.split_actor_losses:
                actor_data = split_data
                actor_alpha_vals = split_alpha_vals
                actor_task_weights = split_task_weights

        self, critic_logs = self.update_critic(
            critic_data, critic_alpha_vals, critic_task_weights
        )
        self, log_probs, actor_logs = self.update_actor(
            actor_data, actor_alpha_vals, actor_task_weights
        )

        if self.split_actor_losses:
            assert alpha_val_indices is not None
            log_probs = self.unsplit_data_by_tasks(log_probs, alpha_val_indices)
        self, alpha_logs = self.update_alpha(log_probs, task_ids)

        # HACK: PCGrad logs
        assert isinstance(self.critic.opt_state, tuple)
        assert isinstance(self.actor.opt_state, tuple)
        critic_optim_logs = (
            {
                f"metrics/critic_{key}": value
                for key, value in self.critic.opt_state[0]._asdict().items()
            }
            if isinstance(self.critic.opt_state[0], PCGradState) or isinstance(self.actor.opt_state[0], GradNormState) or isinstance(self.actor.opt_state[0], CAGradState)
            else {}
        )
        actor_optim_logs = (
            {
                f"metrics/actor_{key}": value
                for key, value in self.actor.opt_state[0]._asdict().items()
            }
            if isinstance(self.actor.opt_state[0], PCGradState) or isinstance(self.actor.opt_state[0], GradNormState) or isinstance(self.actor.opt_state[0], CAGradState)
            else {}
        )

        return self, {
            **critic_logs,
            **actor_logs,
            **alpha_logs,
            **critic_optim_logs,
            **actor_optim_logs,
        }

    @override
    def update(self, data: ReplayBufferSamples) -> tuple[Self, LogDict]:
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
        self, data: ReplayBufferSamples
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
        self, metrics: Metrics, data: ReplayBufferSamples
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
