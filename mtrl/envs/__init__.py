from .base import EnvConfig
from .metaworld import MetaworldConfig
from .dmc import (
    DMCLocomotionMTConfig,
    DMCWalkerConfig,
    DMCHopperConfig,
    DMCReacherConfig,
    DMCTaskConfig,
)
#from .atari import AtariConfig, ATARI_26_GAMES
from .humanoidbench import HumanoidBenchConfig

__all__ = [
    "EnvConfig",
    "MetaworldConfig",
    "DMCLocomotionMTConfig",
    "DMCWalkerConfig",
    "DMCHopperConfig",
    "DMCReacherConfig",
    "DMCTaskConfig",
]
