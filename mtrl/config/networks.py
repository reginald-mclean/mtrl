from .nn import NeuralNetworkConfig, VanillaNetworkConfig, ImpalaEncoderConfig, TaskEmbeddingConfig, BroConfig
from dataclasses import dataclass


@dataclass(frozen=True)
class ContinuousActionPolicyConfig:
    network_config: NeuralNetworkConfig = VanillaNetworkConfig(width=400, depth=3)
    """The config for the neural network to use for function approximation."""

    squash_tanh: bool = True
    """Whether or not to squash the outputs with tanh."""

    log_std_min: float = -20.0
    """The minimum possible log standard deviation for each action distribution."""

    log_std_max: float = 2.0
    """The maximum possible log standard deviation for each action distribution."""


@dataclass(frozen=True)
class QValueFunctionConfig:
    network_config: NeuralNetworkConfig = VanillaNetworkConfig(width=400, depth=3)
    """The config for the neural network to use for function approximation."""

    use_classification: bool = False
    """Whether or not to use classification instead of regression."""

    num_atoms: int | None = None
    """If using use_classification, describes the number of atoms to use per action dimension."""

    dueling: bool = False
    """Whether to use a dueling architecture (separate value and advantage streams)."""

@dataclass(frozen=True)
class ValueFunctionConfig(QValueFunctionConfig): ...

@dataclass(frozen=True)
class ImpalaDQNConfig:
    impala_config: ImpalaEncoderConfig = ImpalaEncoderConfig()
    q_function_config: QValueFunctionConfig = QValueFunctionConfig(use_classification=True, num_atoms=101)
    task_embed_config: TaskEmbeddingConfig = TaskEmbeddingConfig()
    use_layer_norm: bool = True

@dataclass(frozen=True)
class BroQConfig:
    bro_config: BroConfig = BroConfig()
    q_function_config: QValueFunctionConfig = QValueFunctionConfig(use_classification=True, num_atoms=101)
    task_embed_config: TaskEmbeddingConfig = TaskEmbeddingConfig()


@dataclass(frozen=True)
class BroActorConfig:
    bro_config: BroConfig = BroConfig()
    actor_config: ContinuousActionPolicyConfig = ContinuousActionPolicyConfig()
    task_embed_config: TaskEmbeddingConfig = TaskEmbeddingConfig()
