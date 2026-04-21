from dataclasses import dataclass
from pathlib import Path

import tyro


from mtrl.config.networks import BroQConfig, BroActorConfig, QValueFunctionConfig, TaskEmbeddingConfig, ContinuousActionPolicyConfig
from mtrl.config.nn import VanillaNetworkConfig, BroConfig
from mtrl.config.optim import OptimizerConfig
from mtrl.config.rl import OffPolicyTrainingConfig
from mtrl.envs import HumanoidBenchConfig
from mtrl.experiment import Experiment
from mtrl.rl.algorithms import MTSACConfig, SACConfig


@dataclass(frozen=True)
class Args:
    seed: int = 1
    track: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    data_dir: Path = Path("./results")
    resume: bool = False
    reward_func_version: str = 'v2'
    width: int = 512
    l2_norm: bool = False



def main() -> None:
    args = tyro.cli(Args)

    WIDTH = args.width
    num_tasks = 9

    experiment = Experiment(
        exp_name=f"humanoidbench-medium_width_{args.width}_norm_{args.l2_norm}",
        seed=args.seed,
        data_dir=args.data_dir,
        env=HumanoidBenchConfig('medium'),
        algorithm=MTSACConfig(
            clip=False,
            num_tasks=num_tasks,
            gamma=0.99,
            v_min=-10.0,
            v_max=10.0,
            n_atoms=101,
            actor_config=BroActorConfig(
                bro_config=BroConfig(
                    width=256,
                    num_blocks=1,
                ),
                task_embed_config=TaskEmbeddingConfig(num_tasks=num_tasks),
                actor_config=ContinuousActionPolicyConfig(
                    network_config=VanillaNetworkConfig(
                        width=WIDTH,
                        depth=1,
                        optimizer=OptimizerConfig(max_grad_norm=1.0) #  if args.reward_func_version == 'v2' else None),
                     )
                )
            ),
            critic_config=BroQConfig(
                bro_config=BroConfig(num_blocks=2, width=1024),
                task_embed_config=TaskEmbeddingConfig(num_tasks=num_tasks),
                q_function_config=QValueFunctionConfig(
                    num_atoms=101,
                    network_config=VanillaNetworkConfig(
                        width=WIDTH,
                        optimizer=OptimizerConfig(max_grad_norm=1.0) #  if args.reward_func_version == 'v2' else None),
                    )
                )
            ),
            num_critics=2,
        ),
        training_config=OffPolicyTrainingConfig(
            returns_normalization=True,
            total_steps=int(2_000_000 * num_tasks),
            buffer_size=int(100_000 * num_tasks),
            batch_size=int(128*num_tasks),
            evaluation_frequency=int(450_000//500),
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
