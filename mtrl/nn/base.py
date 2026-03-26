from collections.abc import Callable

import flax.linen as nn
import jax

from mtrl.config.nn import VanillaNetworkConfig, UVFA_VanillaNetworkConfig

from .utils import name_prefix


class MLP(nn.Module):
    """A Flax Module to represent an MLP feature extractor."""

    head_dim: int

    depth: int = 3
    # TODO: Support variable width?
    width: int = 400

    activation_fn: Callable[[jax.typing.ArrayLike], jax.Array] = jax.nn.relu
    kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_uniform()
    bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.constant(0.1)
    use_bias: bool = True

    head_kernel_init: jax.nn.initializers.Initializer | None = None
    head_bias_init: jax.nn.initializers.Initializer | None = None
    activate_last: bool = False

    use_skip_connections: bool = False
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for i in range(self.depth):
            if self.use_layer_norm and i != 0:
                # Don't normalize inputs
                x = nn.LayerNorm()(x)
            dense_out = nn.Dense(
                self.width,
                name=f"layer_{i}",
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                use_bias=self.use_bias,
            )(x)
            dense_out = self.activation_fn(dense_out)
            if self.use_skip_connections and x.shape[-1] == self.width:
                x = x + dense_out
            else:
                x = dense_out
            self.sow("intermediates", f"{name_prefix(self)}layer_{i}", x)

        if self.use_layer_norm and self.depth != 0:
            x = nn.LayerNorm()(x)
        last_out = nn.Dense(
            self.head_dim,
            name=f"layer_{self.depth}",
            kernel_init=self.head_kernel_init or self.kernel_init,
            bias_init=self.head_bias_init or self.bias_init,
            use_bias=self.use_bias,
        )(x)
        if self.activate_last:
            last_out = self.activation_fn(last_out)
        return last_out


class VanillaNetwork(nn.Module):
    config: VanillaNetworkConfig

    head_dim: int
    head_kernel_init: jax.nn.initializers.Initializer | None = None
    head_bias_init: jax.nn.initializers.Initializer | None = None
    activate_last: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return MLP(
            head_dim=self.head_dim,
            width=self.config.width,
            depth=self.config.depth,
            activation_fn=self.config.activation,
            kernel_init=self.config.kernel_init(),
            bias_init=self.config.bias_init(),
            use_bias=self.config.use_bias,
            head_kernel_init=self.head_kernel_init,
            head_bias_init=self.head_bias_init,
            activate_last=self.activate_last,
            use_skip_connections=self.config.use_skip_connections,
            use_layer_norm=self.config.use_layer_norm,
        )(x)


class UVFA_QFunction(nn.Module):
    """A Flax Module to represent a Universal Value Function Approximator"""

    config: UVFA_VanillaNetworkConfig

    head_kernel_init: jax.nn.initializers.Initializer | None = None
    head_bias_init: jax.nn.initializers.Initializer | None = None
    activate_last: bool = False

    depth: int = 3
    # TODO: Support variable width?
    state_action_width: int = 300
    goal_width: int = 100

    embedding_dim: int = 128

    activation_fn: Callable[[jax.typing.ArrayLike], jax.Array] = jax.nn.relu
    kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_uniform()
    bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.constant(0.1)
    use_bias: bool = True

    head_kernel_init: jax.nn.initializers.Initializer | None = None
    head_bias_init: jax.nn.initializers.Initializer | None = None
    activate_last: bool = False

    use_skip_connections: bool = False
    use_layer_norm: bool = False

    merge_method: str = 'dot'

    @nn.compact
    def __call__(self, s: jax.Array, a: jax.Array, g: jax.Array) -> jax.Array:
        x = jax.numpy.concatenate([s, a], axis=-1)
        for i in range(self.depth):
            dense_out = nn.Dense(
                self.state_action_width,
                name=f"state_action_layer_{i}",
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                use_bias=self.use_bias,
            )(x)
            dense_out = self.activation_fn(dense_out)
            self.sow("intermediates", f"{name_prefix(self)}layer_{i}", x)
            x = dense_out

        state_action_rep = nn.Dense(
            self.embedding_dim,
            name=f"state_action_layer_{self.depth}",
            kernel_init=self.head_kernel_init or self.kernel_init,
            bias_init=self.head_bias_init or self.bias_init,
            use_bias=self.use_bias,
        )(x)


        h = g
        for i in range(self.depth):
            dense_out = nn.Dense(
                self.goal_width,
                name=f"goals_layer_{i}",
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                use_bias=self.use_bias,
            )(h)
            dense_out = self.activation_fn(dense_out)
            self.sow("intermediates", f"{name_prefix(self)}layer_{i}", x)
            h = dense_out

        goal_rep = nn.Dense(
            self.embedding_dim,
            name=f"goals_layer_{self.depth}",
            kernel_init=self.head_kernel_init or self.kernel_init,
            bias_init=self.head_bias_init or self.bias_init,
            use_bias=self.use_bias,
        )(h)


        if self.merge_method == 'dot':
            return (state_action_rep * goal_rep).sum(axis=-1, keepdims=True)
        elif self.merge_method == 'mlp':
            return nn.Dense(1,
                name='layer_merge',
                kernel_init=self.head_kernel_init or self.kernel_init,
                bias_init=self.head_bias_init or self.bias_init,
                use_bias=self.use_bias,
            )(jnp.concatenate([state_action_rep, goal_rep]))
