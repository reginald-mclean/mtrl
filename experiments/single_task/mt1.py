from dataclasses import dataclass
from pathlib import Path

import tyro

from mtrl.config.networks import ContinuousActionPolicyConfig, QValueFunctionConfig
from mtrl.config.nn import VanillaNetworkConfig
from mtrl.config.optim import OptimizerConfig
from mtrl.config.rl import OffPolicyTrainingConfig
from mtrl.envs import MetaworldConfig
from mtrl.experiment import Experiment
from mtrl.rl.algorithms import SACConfig


@dataclass(frozen=True)
class Args:
    seed: int = 1
    track: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    data_dir: Path = Path("./experiment_results")
    resume: bool = False
    width: int = 400
    depth: int = 3
    env_name: str | None = None
    sparse_rewards: bool = False
    sparse_magnitude: int = 0

def main() -> None:
    args = tyro.cli(Args)

    num_tasks = 1
    width = args.width
    depth = args.depth

    experiment = Experiment(
        exp_name=f"mt1_{args.env_name}",
        seed=args.seed,
        data_dir=args.data_dir,
        env=MetaworldConfig(
            env_id="MT1",
            task_name=f"{args.env_name}",
            terminate_on_success=False,
            sparse_rewards=args.sparse_rewards,
            sparse_magnitude=args.sparse_magnitude,
        ),
        algorithm=SACConfig(
            num_tasks=num_tasks,
            gamma=0.99,
            actor_config=ContinuousActionPolicyConfig(
                network_config=VanillaNetworkConfig(
                    width=width,
                    depth=depth,
                    optimizer=OptimizerConfig(max_grad_norm=1.0)
                )
            ),
            critic_config=QValueFunctionConfig(
                network_config=VanillaNetworkConfig(
                    width=width,
                    depth=depth,
                    optimizer=OptimizerConfig(max_grad_norm=1.0)
                )
            ),
            num_critics=2,
        ),
        training_config=OffPolicyTrainingConfig(
            evaluation_frequency=40,
            total_steps=int(2_000_000),
            buffer_size=int(100_000),
            batch_size=128,
        ),
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
