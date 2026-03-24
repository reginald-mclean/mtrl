from collections.abc import Callable

from functools import partial

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from mtrl.config.networks import (
    ContinuousActionPolicyConfig,
    QValueFunctionConfig,
    ValueFunctionConfig,
)
from mtrl.nn import get_nn_arch_for_config
from mtrl.nn.distributions import TanhMultivariateNormalDiag
from mtrl.nn.initializers import uniform

from mtrl.config.nn import ImpalaEncoderConfig, TaskEmbeddingConfig

class ContinuousActionPolicy(nn.Module):
    """A Flax module representing the policy network for continous action spaces."""

    action_dim: int
    config: ContinuousActionPolicyConfig
    last_act = None

    @nn.compact
    def __call__(self, x: jax.Array) -> distrax.Distribution:
        x = get_nn_arch_for_config(self.config.network_config)(
            config=self.config.network_config,
            head_dim=self.action_dim * 2,
            head_kernel_init=uniform(1e-3),
            head_bias_init=uniform(1e-3),
        )(x)

        mean, log_std = jnp.split(x, 2, axis=-1)
        log_std = jnp.clip(
            log_std, a_min=self.config.log_std_min, a_max=self.config.log_std_max
        )
        std = jnp.exp(log_std)

        if self.config.squash_tanh:
            return TanhMultivariateNormalDiag(loc=mean, scale_diag=std)
        return distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)


class QValueFunction(nn.Module):
    """A Flax module approximating a Q-Value function."""

    config: QValueFunctionConfig
    action_dim: int | None = None

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array | None = None) -> jax.Array:
        # NOTE: certain NN architectures that make use of task IDs will be looking for them
        # at the last N_TASKS dimensions of their input. So while normally concat(state,action) makes more sense
        # we'll go with (action, state) here

        if not self.config.use_classification:
            x = jnp.concatenate((action, state), axis=-1)
            return get_nn_arch_for_config(self.config.network_config)(
                config=self.config.network_config,
                head_dim=1,
                head_kernel_init=uniform(3e-3),
                head_bias_init=uniform(3e-3),
            )(x)
        else:
            assert self.action_dim is not None, 'Need to pass action_dim to QValueFunction'
            return get_nn_arch_for_config(self.config.network_config)(
                config=self.config.network_config,
                head_dim=self.config.num_atoms * self.action_dim,
                head_kernel_init=uniform(3e-3),
                head_bias_init=uniform(3e-3),
            )(state).reshape(state.shape[0], self.action_dim, self.config.num_atoms)


class ImpalaDQN(nn.Module):
    impala_config: ImpalaEncoderConfig
    q_function_config: QValueFunctionConfig
    task_embed_config: TaskEmbeddingConfig
    action_dim: int

    @nn.compact
    def __call__(self, x: jax.Array, task_ids: jax.Array): 
        enc = get_nn_arch_for_config(self.impala_config)(self.impala_config)(x)
        embed = get_nn_arch_for_config(self.task_embed_config)(config=self.task_embed_config)(task_ids)
        x = jnp.concatenate([enc, embed], axis=-1)
        critic_cls = partial(
            QValueFunction,
            config=self.q_function_config,
            action_dim=self.action_dim,
        )
        return Ensemble(critic_cls, num=self.impala_config.num_critics)(x)

class ValueFunction(nn.Module):
    """A Flax module approximating a Q-Value function."""

    config: ValueFunctionConfig

    @nn.compact
    def __call__(self, state: jax.Array) -> jax.Array:
        if not self.config.use_classification:
            return get_nn_arch_for_config(self.config.network_config)(
                config=self.config.network_config,
                head_dim=1,
                head_kernel_init=uniform(3e-3),
                head_bias_init=uniform(3e-3),
            )(state)
        else:
            raise NotImplementedError(
                "Value prediction as classification is not supported yet."
            )


class Ensemble(nn.Module):
    net_cls: nn.Module | Callable[..., nn.Module]
    num: int = 2

    @nn.compact
    def __call__(self, *args):
        ensemble = nn.vmap(
            self.net_cls,
            variable_axes={"params": 0, "intermediates": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,  # pyright: ignore [reportArgumentType]
            out_axes=0,
            axis_size=self.num,
        )
        return ensemble()(*args)
