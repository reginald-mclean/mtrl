from dataclasses import dataclass
from pathlib import Path

import tyro

from mtrl.config.networks import ContinuousActionPolicyConfig, QValueFunctionConfig, ImpalaDQNConfig
from mtrl.config.nn import VanillaNetworkConfig
from mtrl.config.optim import OptimizerConfig
from mtrl.config.rl import OffPolicyTrainingConfig, DrQTrainingConfig
from mtrl.envs import AtariConfig
from mtrl.experiment import Experiment
from mtrl.rl.algorithms import DrQConfig

@dataclass(frozen=True)
class Args:
    seed: int = 1
    track: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    data_dir: Path = Path("./experiment_results")
    resume: bool = False


def main() -> None:
    args = tyro.cli(Args)

    num_tasks = 26

    experiment = Experiment(
        exp_name="atari_26_games",
        seed=args.seed,
        data_dir=args.data_dir,
        env=AtariConfig(
        ),
        algorithm=DrQConfig(
            num_tasks=num_tasks,
            gamma=0.99,
            num_critics=2,
            critic_config=ImpalaDQNConfig(),
        ),
        training_config=DrQTrainingConfig(
            total_steps=num_tasks*int(100_000)
        ),
        checkpoint=False,
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
