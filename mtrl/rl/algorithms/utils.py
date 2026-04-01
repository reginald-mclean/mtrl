from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT
from flax.training.train_state import TrainState as FlaxTrainState
from jaxtyping import Array, Float


class TrainState(FlaxTrainState):
    def apply_gradients(
        self,
        *,
        grads,
        optimizer_extra_args: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        if OVERWRITE_WITH_GRADIENT in grads:
            grads_with_opt = grads["params"]
            params_with_opt = self.params["params"]
        else:
            grads_with_opt = grads
            params_with_opt = self.params

        if optimizer_extra_args is None:
            optimizer_extra_args = {}

        updates, new_opt_state = self.tx.update(
            grads_with_opt, self.opt_state, params_with_opt, **optimizer_extra_args
        )
        new_params_with_opt = optax.apply_updates(params_with_opt, updates)

        if OVERWRITE_WITH_GRADIENT in grads:
            new_params = {
                "params": new_params_with_opt,
                OVERWRITE_WITH_GRADIENT: grads[OVERWRITE_WITH_GRADIENT],
            }
        else:
            new_params = new_params_with_opt
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )


def vmap_cos_sim(
    grads: Float[Array, "num_tasks num_params"],
    num_tasks: int,
) -> tuple[Float[Array, ""], Float[Array, "num_tasks num_tasks"]]:
    def calc_cos_sim(selected_grad, grads):
        return jnp.array(
            [
                jnp.sum(selected_grad * grads, axis=1)
                / (
                    jnp.linalg.norm(selected_grad)
                    * jnp.linalg.norm(grads, axis=1)
                    + 1e-8
                )
            ]
        )

    cos_sim_mat = jax.vmap(calc_cos_sim, in_axes=(0, None), out_axes=-1)(grads, grads)
    mask = jnp.triu(jnp.ones((num_tasks, num_tasks)), k=1)
    num_unique = mask.sum()
    masked_cos_sim = mask * cos_sim_mat
    avg_cos_sim = masked_cos_sim.flatten().sum() / (num_unique + 1e-8)
    return avg_cos_sim, cos_sim_mat


def compute_conflict_metrics(
    cos_sim_mat: Float[Array, "num_tasks num_tasks"],
    flat_grads: Float[Array, "num_tasks num_params"],
) -> dict[str, Float[Array, ""]]:
    num_tasks = flat_grads.shape[0]
    diag_mask = jnp.eye(num_tasks, dtype=bool)
    off_diag = ~diag_mask
    conflict_mask = (cos_sim_mat < 0) & off_diag
    n_pairs = num_tasks * (num_tasks - 1)

    angles = jnp.degrees(jnp.arccos(jnp.clip(cos_sim_mat, -1.0, 1.0)))

    conflict_rate = conflict_mask.sum() / n_pairs
    mean_conflict_magnitude = jnp.where(
        conflict_mask, jnp.abs(cos_sim_mat), 0.0
    ).sum() / (conflict_mask.sum() + 1e-8)
    mean_conflict_angle = jnp.where(
        conflict_mask, angles, 0.0
    ).sum() / (conflict_mask.sum() + 1e-8)

    per_task_conflict_rate = conflict_mask.sum(axis=1) / (num_tasks - 1)
    per_task_grad_magnitude = jnp.linalg.norm(flat_grads, axis=1)

    pairwise_conflict = conflict_mask.astype(jnp.float32)
    pairwise_cos_sim = jnp.where(off_diag, cos_sim_mat, 0.0)
    pairwise_angle = jnp.where(off_diag, angles, 0.0)

    return {
        "conflict_rate":           conflict_rate,
        "mean_conflict_magnitude": mean_conflict_magnitude,
        "mean_conflict_angle":     mean_conflict_angle,
        "per_task_conflict_rate":  per_task_conflict_rate,
        "per_task_grad_magnitude": per_task_grad_magnitude,
        "pairwise_conflict":       pairwise_conflict,
        "pairwise_cos_sim":        pairwise_cos_sim,
        "pairwise_angle":          pairwise_angle,
    }
