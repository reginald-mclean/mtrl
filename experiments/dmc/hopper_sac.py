from dataclasses import dataclass
from pathlib import Path

import tyro

from mtrl.config.networks import (
    BroActorConfig,
    BroQConfig,
    ContinuousActionPolicyConfig,
    QValueFunctionConfig,
    TaskEmbeddingConfig,
)
from mtrl.config.nn import BroConfig, VanillaNetworkConfig
from mtrl.config.optim import OptimizerConfig
from mtrl.config.rl import OffPolicyTrainingConfig
from mtrl.envs.dmc import DMCHopperConfig
from mtrl.experiment import Experiment
from mtrl.rl.algorithms import MTSACConfig

NUM_TASKS = 2


@dataclass(frozen=True)
class Args:
    seed: int = 1
    track: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    data_dir: Path = Path("./results")
    resume: bool = False


def main() -> None:
    args = tyro.cli(Args)

    experiment = Experiment(
        exp_name="dmc_hopper_sac",
        seed=args.seed,
        data_dir=args.data_dir,
        env=DMCHopperConfig(max_episode_steps=1000),
        algorithm=MTSACConfig(
            num_tasks=NUM_TASKS,
            gamma=0.99,
            v_min=-10.0,
            v_max=10.0,
            n_atoms=101,
            actor_config=BroActorConfig(
                bro_config=BroConfig(num_blocks=1, width=256),
                task_embed_config=TaskEmbeddingConfig(num_tasks=NUM_TASKS),
                actor_config=ContinuousActionPolicyConfig(
                    network_config=VanillaNetworkConfig(
                        optimizer=OptimizerConfig(max_grad_norm=1.0)
                    )
                ),
            ),
            critic_config=BroQConfig(
                bro_config=BroConfig(num_blocks=2, width=256),
                task_embed_config=TaskEmbeddingConfig(num_tasks=NUM_TASKS),
                q_function_config=QValueFunctionConfig(
                    num_atoms=101,
                    network_config=VanillaNetworkConfig(
                        optimizer=OptimizerConfig(max_grad_norm=1.0)
                    ),
                ),
            ),
            num_critics=2,
        ),
        training_config=OffPolicyTrainingConfig(
            total_steps=int(5e6),
            buffer_size=int(1e6),
            batch_size=512,
            evaluation_frequency=200,
            returns_normalization=True,
            nstep=3,
        ),
        best_checkpoint_metric="mean_success_rate",
        checkpoint=True,
        resume=args.resume,
    )

    if args.track:
        assert args.wandb_project is not None and args.wandb_entity is not None
        experiment.enable_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=experiment,
            resume="allow",
        )

    experiment.run()


if __name__ == "__main__":
    main()
