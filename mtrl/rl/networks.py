from collections.abc import Callable

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from mtrl.config.networks import ContinuousActionPolicyConfig, QValueFunctionConfig
from mtrl.nn import get_nn_arch_for_config
from mtrl.nn.distributions import TanhMultivariateNormalDiag
from mtrl.nn.initializers import uniform


class ContinuousActionPolicy(nn.Module):
    """A Flax module representing the policy network for continous action spaces."""

    action_dim: int
    config: ContinuousActionPolicyConfig

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

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        x = jnp.concatenate((state, action), axis=-1)

        if not self.config.use_classification:
            return get_nn_arch_for_config(self.config.network_config)(
                config=self.config.network_config,
                head_dim=1,
                head_kernel_init=uniform(3e-3),
                head_bias_init=uniform(3e-3),
            )(x)
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
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,  # pyright: ignore [reportArgumentType]
            out_axes=0,
            axis_size=self.num,
        )
        return ensemble()(*args)
