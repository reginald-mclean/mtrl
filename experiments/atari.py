from dataclasses import dataclass
from pathlib import Path

import tyro

from mtrl.config.networks import ContinuousActionPolicyConfig, QValueFunctionConfig, ImpalaDQNConfig
from mtrl.config.nn import VanillaNetworkConfig, ImpalaEncoderConfig
from mtrl.config.optim import OptimizerConfig
from mtrl.config.rl import OffPolicyTrainingConfig, DrQTrainingConfig
from mtrl.config.utils import Optimizer
from mtrl.envs import AtariConfig
from mtrl.experiment import Experiment
from mtrl.rl.algorithms import DrQConfig

@dataclass(frozen=True)
class Args:
    seed: int = 1
    track: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    data_dir: Path = Path("./results")
    resume: bool = False
    normalize_rewards: bool = True
    scale: int = 1


def main() -> None:
    args = tyro.cli(Args)

    num_tasks = 26

    experiment = Experiment(
        exp_name=f"atari_26_games_dueling_dqn_augmentation_updates_eval_envs_scale_{args.scale}",
        seed=args.seed,
        data_dir=args.data_dir,
        env=AtariConfig(),
        algorithm=DrQConfig(
            num_tasks=num_tasks,
            gamma=0.99,
            critic_config=ImpalaDQNConfig(
                impala_config = ImpalaEncoderConfig(scale=args.scale),
                q_function_config=QValueFunctionConfig(
                    use_classification=True,
                    num_atoms=101,
                    network_config=VanillaNetworkConfig(
                        optimizer=OptimizerConfig(
                            lr=1e-4,
                            optimizer=Optimizer.AdamW,
                            eps=1.5e-4,
                            weight_decay=0.05,
                        ),
                    ),
                ),
            ),
        ),
        training_config=DrQTrainingConfig(
            total_steps=num_tasks*int(1e5),
            normalize_rewards=args.normalize_rewards,
            buffer_size=int(num_tasks*10_000)
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
