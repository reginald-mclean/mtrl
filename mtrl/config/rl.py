from dataclasses import dataclass

from .utils import Metrics

from mtrl.config.nn import ImpalaEncoderConfig, TaskEmbeddingConfig
from mtrl.config.optim import OptimizerConfig

from mtrl.config.networks import (
    QValueFunctionConfig,
    VanillaNetworkConfig,
)

from .utils import Optimizer

@dataclass(frozen=True)
class AlgorithmConfig:
    num_tasks: int
    gamma: float = 0.99
    weights_critic_loss: bool = False
    weights_actor_loss: bool = False
    weights_qf_vals: bool = False
    clip: bool = False

@dataclass(frozen=True, kw_only=True)
class TrainingConfig:
    total_steps: int
    evaluation_frequency: int = 200_000 // 500
    compute_network_metrics: Metrics = Metrics.ALL

    # TODO: Maybe put into its own RewardFilterConfig()?
    reward_filter: str | None = None
    reward_filter_sigma: float | None = None
    reward_filter_alpha: float | None = None
    reward_filter_delta: float | None = None
    reward_filter_mode: str | None = None
    sampler_type: str | None = None
    update_weights_every: int = 500
    weights_critic_loss: bool = False
    weights_actor_loss: bool = False
    weights_qf_vals: bool = False
    state_coverage:bool = False
    normalize_rewards: bool = False

@dataclass(frozen=True)
class OffPolicyTrainingConfig(TrainingConfig):
    warmstart_steps: int = int(4e3)
    buffer_size: int = int(1e6)
    batch_size: int = 1280


@dataclass(frozen=True)
class DrQTrainingConfig(OffPolicyTrainingConfig):
    warmstart_steps: int = 1_000
    buffer_size: int = 100_000  # per-task buffer size (shared indexing)
    batch_size: int = 256

    encoder_config: ImpalaEncoderConfig = ImpalaEncoderConfig(scale=2)
    critic_config: QValueFunctionConfig = QValueFunctionConfig(
        use_classification=True,
        num_atoms=101,
        network_config=VanillaNetworkConfig(width=512, depth=2),
    )
    task_embed_config: TaskEmbeddingConfig = TaskEmbeddingConfig()
    num_critics: int = 2
    tau: float = 0.005
    # Epsilon schedule
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 5_000  # 5% of 100k total_timesteps
    # Categorical support
    v_min: float = -10.0
    v_max: float = 10.0
    num_tasks: int = 26
    normalize_rewards: bool = True
    nstep: int = 3
    replay_ratio: int = 2
    eval_step_frequency: int = 10_000
    augment: bool = True
    shrink_perturb_frequency: int = 50000  # steps between shrink-and-perturb resets (Archive default)
    shrink_rate: float = 0.5  # interpolation rate: old*(1-rate) + new*rate

@dataclass(frozen=True)
class OnPolicyTrainingConfig(TrainingConfig):
    rollout_steps: int = 10_000
    num_epochs: int = 16
    num_gradient_steps: int = 32

    compute_advantages: bool = True
    gae_lambda: float = 0.97
    target_kl: float | None = None
