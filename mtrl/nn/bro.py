from collections.abc import Callable
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from mtrl.config.nn import BroConfig

from .utils import name_prefix


class BroBlock(nn.Module):
    initializer: Callable = nn.initializers.xavier_uniform()
    output_size: int = 256
    block_num: int = 0

    @nn.compact
    def __call__(self, x):
        skip = x
        x = nn.Dense(self.output_size,
            name=f'dense_{self.block_num}_0',
            kernel_init=self.initializer,
            bias_init=nn.initializers.zeros,
            use_bias=True,
        )(x)
        x = nn.LayerNorm()(x)
        x = nn.Relu()(x)
        x = nn.Dense(self.output_size,
            name=f'dense_{self.block_num}_1',
            kernel_init=self.initializer,
            bias_init=nn.initializers.zeros,
            use_bias=True,
        )(x)
        x = nn.LayerNorm()(x)
        x = x + skip
        return x


class BroNet(nn.Module):
    config: BroConfig

    def setup(self):
        self.blocks = [BroBlock(output_size=self.config.width, block_num=x) for x in self.config.num_blocks]

    def __call__(self, x):
        x = nn.Dense(self.config.width, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        for b in self.blocks:
            x = b(x)

        return x
