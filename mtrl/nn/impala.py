from collections.abc import Callable
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from mtrl.config.nn import ImpalaEncoderConfig

from .utils import name_prefix


class ImpalaBlock(nn.Module):
    channels: int
    blocks: int
    use_max_pooling: bool = True
    initializer: Callable = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, x):
        conv_out = nn.Conv(self.channels, (3,3), 1, kernel_init=self.initializer, padding='SAME')(x)
        if self.use_max_pooling:
            conv_out = nn.max_pool(conv_out, (3,3), padding='SAME', strides=(2,2))
        for _ in range(self.blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(self.channels, (3,3), 1, padding='SAME')(conv_out)
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(self.channels, (3,3), 1, padding='SAME')(conv_out)
            conv_out += block_input
        return conv_out

class ImpalaEncoder(nn.Module):
    config: ImpalaEncoderConfig

    def setup(self):
        self.stack = [ImpalaBlock(self.config.scale * stack, blocks=self.config.blocks) for stack in self.config.stacks]

    def __call__(self, x: jax.Array) -> jax.Array:
        if x.dtype == jnp.uint8:
            x = jnp.transpose(x, (0, 2, 3, 1))
            x = (x.astype(jnp.float32) / 255.0 - 0.5) * 2
        for stack in self.stack:
            x = stack(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        return x
