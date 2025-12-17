from dataclasses import dataclass
from pathlib import Path

import tyro

from mtrl.config.networks import ContinuousActionPolicyConfig, QValueFunctionConfig
from mtrl.config.nn import MultiHeadConfig
from mtrl.config.optim import OptimizerConfig
from mtrl.config.rl import OffPolicyTrainingConfig
from mtrl.envs import MetaworldConfig
from mtrl.experiment import Experiment
from mtrl.rl.algorithms import MTSACConfig


@dataclass(frozen=True)
class Args:
    seed: int = 1
    track: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    data_dir: Path = Path("./experiment_results")
    resume: bool = False
    weights_critic_loss: bool = False
    weights_actor_loss: bool = False
    weights_qf_vals: bool = False
    update_weights_every: int | None = None
    sampler_type: str| None = None
    state_coverage : bool = False

def main() -> None:
    args = tyro.cli(Args)

    WIDTH = 512

    weights_critic_loss=args.weights_critic_loss
    weights_actor_loss=args.weights_actor_loss
    weights_qf_vals=args.weights_qf_vals


    base = "test_"
    if args.sampler_type:
        assert args.sampler_type and args.update_weights_every, "need to set both sampler and sample interval"
        base += f"{args.sampler_type}_update_int_{args.update_weights_every}_"
    if weights_critic_loss:
        base += "weights_critic_loss"
    elif weights_actor_loss:
        base += "weights_actor_loss"
    elif weights_qf_vals:
        base += "weights_qf_vals"

    experiment = Experiment(
        exp_name=f"no_exploration_mt10_{WIDTH}_"+base,
        seed=args.seed,
        data_dir=args.data_dir,
        env=MetaworldConfig(
            num_eval_episodes=10,
            env_id="MT10",
            terminate_on_success=False,
        ),
        algorithm=MTSACConfig(
            num_tasks=10,
            gamma=0.99,
            weights_critic_loss=weights_critic_loss,
            weights_actor_loss=weights_actor_loss,
            weights_qf_vals=weights_qf_vals,
            actor_config=ContinuousActionPolicyConfig(
                network_config=MultiHeadConfig(
                    width=WIDTH,
                    num_tasks=10,
                    optimizer=OptimizerConfig(max_grad_norm=1.0),
                )
            ),
            critic_config=QValueFunctionConfig(
                network_config=MultiHeadConfig(
                    width=WIDTH,
                    num_tasks=10,
                    optimizer=OptimizerConfig(max_grad_norm=1.0),
                )
            ),
            num_critics=2,
        ),
        training_config=OffPolicyTrainingConfig(
            total_steps=int(2e7),
            evaluation_frequency=int(1_000_000),
            buffer_size=int(1e6),
            batch_size=1280,
            sampler_type=args.sampler_type,
            update_weights_every=args.update_weights_every, # this is really * num_envs -> so 1000 is really 1000 steps per env aka 10,000 steps in mt10
            weights_critic_loss=weights_critic_loss,
            weights_actor_loss=weights_actor_loss,
            weights_qf_vals=weights_qf_vals,
            state_coverage=args.state_coverage,
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
