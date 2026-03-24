import abc
from dataclasses import dataclass
from functools import cached_property, partial

import ale_py
import gymnasium as gym
from .base import EnvConfig
from mtrl.types import Agent

import numpy as np
import numpy.typing as npt

gym.register_envs(ale_py)

from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
    TransformReward,
)

ATARI_26_GAMES = [
    "alien",
    "amidar",
    "assault",
    "asterix",
    "bank_heist",
    "battle_zone",
    "boxing",
    "breakout",
    "chopper_command",
    "crazy_climber",
    "demon_attack",
    "freeway",
    "frostbite",
    "gopher",
    "hero",
    "jamesbond",
    "kangaroo",
    "krull",
    "kung_fu_master",
    "ms_pacman",
    "pong",
    "private_eye",
    "qbert",
    "road_runner",
    "seaquest",
    "up_n_down",
]

# ALE game name mapping (gymnasium uses CamelCase)
_GAME_TO_ALE = {
    "alien":           "ALE/Alien-v5",
    "amidar":          "ALE/Amidar-v5",
    "assault":         "ALE/Assault-v5",
    "asterix":         "ALE/Asterix-v5",
    "bank_heist":      "ALE/BankHeist-v5",
    "battle_zone":     "ALE/BattleZone-v5",
    "boxing":          "ALE/Boxing-v5",
    "breakout":        "ALE/Breakout-v5",
    "chopper_command": "ALE/ChopperCommand-v5",
    "crazy_climber":   "ALE/CrazyClimber-v5",
    "demon_attack":    "ALE/DemonAttack-v5",
    "freeway":         "ALE/Freeway-v5",
    "frostbite":       "ALE/Frostbite-v5",
    "gopher":          "ALE/Gopher-v5",
    "hero":            "ALE/Hero-v5",
    "jamesbond":       "ALE/Jamesbond-v5",
    "kangaroo":        "ALE/Kangaroo-v5",
    "krull":           "ALE/Krull-v5",
    "kung_fu_master":  "ALE/KungFuMaster-v5",
    "ms_pacman":       "ALE/MsPacman-v5",
    "pong":            "ALE/Pong-v5",
    "private_eye":     "ALE/PrivateEye-v5",
    "qbert":           "ALE/Qbert-v5",
    "road_runner":     "ALE/RoadRunner-v5",
    "seaquest":        "ALE/Seaquest-v5",
    "up_n_down":       "ALE/UpNDown-v5",
}

def make_atari_env(
    game: str,
    seed: int = 0,
    frame_stack: int = 4,
    obs_size: int = 84,
    eval_mode: bool = False,
    max_episode_steps: int = 27_000,
) -> gym.Env:
    """
    Build a preprocessed Atari environment following DrQ-ε conventions.

    Key choices:
      - frame_skip = 4 (built into AtariPreprocessing)
      - grayscale + resize to obs_size × obs_size
      - EpisodicLife during training (not eval)
      - Reward clipping during training (not eval)
      - FrameStack of 4
    """
    ale_id = _GAME_TO_ALE.get(game)
    if ale_id is None:
        raise ValueError(f"Unknown game: {game!r}. "
                         f"Available: {list(_GAME_TO_ALE.keys())}")

    env = gym.make(
        ale_id,
        frameskip=1,           # AtariPreprocessing handles skipping
        repeat_action_probability=0.25,
        full_action_space=True,
        render_mode=None,
    )

    # AtariPreprocessing: noop_max, frame_skip, grayscale, scale_obs (False → uint8)
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=obs_size,
        terminal_on_life_loss=not eval_mode,  # EpisodicLife for training
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,          # keep uint8, encoder normalises
    )

    if not eval_mode:
        # Clip rewards to {-1, 0, +1} during training
        env = TransformReward(env, np.sign)

    # Frame stacking: produces (frame_stack, H, W) tensor of uint8
    env = FrameStackObservation(env, frame_stack)

    # Time limit
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps // 4)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.reset(seed=seed)
    return env


def evaluation(
    agent: Agent,
    eval_envs: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
    num_episodes: int = 50,
) -> tuple[float, float, dict[str, float], dict[str, list[float]]]:
    obs: npt.NDArray[np.float64]
    obs, _ = eval_envs.reset()
    agent.reset(np.ones(eval_envs.num_envs, dtype=np.bool_))

    task_names = ATARI_26_GAMES
    successes = {task_name: 0 for task_name in set(task_names)}
    episodic_returns: dict[str, list[float]] = {
        task_name: [] for task_name in set(task_names)
    }

    def eval_done(returns):
        return all(len(r) >= num_episodes for _, r in returns.items())

    while not eval_done(episodic_returns):
        actions = agent.eval_action(obs)
        obs, _, terminations, truncations, infos = eval_envs.step(actions)

        dones = np.logical_or(terminations, truncations)
        agent.reset(dones)

        for i, env_ended in enumerate(dones):
            if env_ended:
                episodic_returns[task_names[i]].append(
                    float(infos["final_info"][i]["episode"]["r"])
                )

    episodic_returns = {
        task_name: returns[:num_episodes]
        for task_name, returns in episodic_returns.items()
    }

    success_rate_per_task = {
        task_name: task_successes / num_episodes
        for task_name, task_successes in successes.items()
    }
    mean_success_rate = np.mean(list(success_rate_per_task.values()))
    mean_returns = np.mean(list(episodic_returns.values()))

    return (
        float(mean_success_rate),
        float(mean_returns),
        success_rate_per_task,
        episodic_returns,
    )


def get_human_scores() -> dict[str, float]:
    return {
        "alien":          7127.7,  "amidar":         1719.5,
        "assault":        742.0,   "asterix":        8503.3,
        "bank_heist":     753.1,   "battle_zone":    37187.5,
        "boxing":         12.1,    "breakout":       30.5,
        "chopper_command": 7387.8, "crazy_climber":  35829.4,
        "demon_attack":   1971.0,  "freeway":        29.6,
        "frostbite":      4334.7,  "gopher":         2412.5,
        "hero":           30826.4, "jamesbond":      302.8,
        "kangaroo":       3035.0,  "krull":          2665.5,
        "kung_fu_master": 22736.3, "ms_pacman":      6951.6,
        "pong":           14.6,    "private_eye":    69571.3,
        "qbert":          13455.0, "road_runner":    7845.0,
        "seaquest":       42054.7, "up_n_down":      11693.2,
    }


def get_random_scores() -> dict[str, float]:
    return {
        "alien":          227.8,   "amidar":         5.8,
        "assault":        222.4,   "asterix":        210.0,
        "bank_heist":     14.2,    "battle_zone":    2360.0,
        "boxing":         0.1,     "breakout":       1.7,
        "chopper_command": 811.0,  "crazy_climber":  10780.5,
        "demon_attack":   152.1,   "freeway":        0.0,
        "frostbite":      65.2,    "gopher":         257.6,
        "hero":           1027.0,  "jamesbond":      29.0,
        "kangaroo":       52.0,    "krull":          1598.0,
        "kung_fu_master": 258.5,   "ms_pacman":      307.3,
        "pong":          -20.7,    "private_eye":    24.9,
        "qbert":          163.9,   "road_runner":    11.5,
        "seaquest":       68.4,    "up_n_down":      533.4,
    }


@dataclass(frozen=True)
class AtariConfig(EnvConfig):
    seed: int = 1
    eval_episodes: int = 3
    env_id: str = "Atari26"
    frame_stack: int = 4

    @cached_property
    def action_space(self) -> gym.Space:
        return gym.spaces.Discrete(18)

    @cached_property
    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(0, 255, (self.frame_stack, 84, 84), np.uint8)

    def spawn(self, seed: int = 1) -> gym.vector.VectorEnv:
        return gym.vector.AsyncVectorEnv([partial(make_atari_env, game=x, seed=seed) for x in ATARI_26_GAMES])

    def spawn_eval(self, seed: int = 1) -> gym.vector.VectorEnv:
        return gym.vector.AsyncVectorEnv([partial(make_atari_env, game=x, seed=seed, eval_mode=True) for x in ATARI_26_GAMES])

    def evaluate(
        self, envs: gym.vector.VectorEnv, agent: Agent
    ) -> tuple[float, float, dict[str, float]]: 
        return evaluation(agent, envs)[:3]
