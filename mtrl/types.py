from typing import NamedTuple, TypedDict, Any, Protocol

import numpy as np
import numpy.typing as npt

from jaxtyping import Float, Array

type LogDict = dict[str, float | Float[Array, ""]]

Action = Float[np.ndarray, "... action_dim"]
Value = Float[np.ndarray, "... 1"]
LogProb = Float[np.ndarray, "... 1"]
Observation = Float[np.ndarray, "... obs_dim"]
AuxPolicyOutputs = dict[str, npt.NDArray]


class Agent(Protocol):
    def eval_action(
        self, observation: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray]]: ...


class ReplayBufferSamples(NamedTuple):
    observations: Float[Observation, " batch"]
    actions: Float[Action, " batch"]
    next_observations: Float[Observation, " batch"]
    dones: Float[np.ndarray, "batch 1"]
    rewards: Float[np.ndarray, "batch 1"]


class Rollout(NamedTuple):
    # Standard timestep data
    observations: Float[Observation, "task timestep"]
    actions: Float[Action, "task timestep"]
    rewards: Float[np.ndarray, "task timestep 1"]
    dones: Float[np.ndarray, "task timestep 1"]

    # Auxiliary policy outputs
    log_probs: Float[LogProb, "task timestep"] | None = None
    means: Float[Action, "task timestep"] | None = None
    stds: Float[Action, "task timestep"] | None = None

    # Computed statistics about observed rewards
    values: Float[np.ndarray, "task timestep 1"] | None = None
    returns: Float[np.ndarray, "task timestep 1"] | None = None
    advantages: Float[np.ndarray, "task timestep 1"] | None = None


class CheckpointMetadata(TypedDict):
    step: int
    episodes: int


class RNGCheckpoint(TypedDict):
    python_rng_state: tuple[Any, ...]
    global_numpy_rng_state: dict[str, Any]


class ReplayBufferCheckpoint(TypedDict):
    data: dict[str, npt.NDArray[np.float32] | int | bool]
    rng_state: Any


type EnvCheckpoint = list[tuple[str, dict[str, Any]]]
