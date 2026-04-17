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


def compute_sparsity_mismatch(flat_grads, eps=1e-3, tau=1.0):
    """
    For each ordered pair of tasks (A, B), compute the fraction of elements
    where task A's gradient is near-zero but task B's is large.
    Entry [i, j] = I(i->j): fraction of i's near-zero params strongly updated by j.
    Diagonal is zeroed — a task cannot interfere with itself.
    """
    num_tasks = flat_grads.shape[0]
    near_zero = jnp.abs(flat_grads) < eps   # (num_tasks, num_params)
    large     = jnp.abs(flat_grads) > tau   # (num_tasks, num_params)

    # broadcast to (num_tasks, num_tasks, num_params): near_zero[i] AND large[j]
    mismatch = near_zero[:, None, :] & large[None, :, :]
    near_zero_counts = near_zero.sum(axis=1).clip(min=1)    # (num_tasks,)
    interference_rate = mismatch.sum(axis=-1) / near_zero_counts[:, None]
    interference_rate = interference_rate * (1 - jnp.eye(num_tasks))
    return interference_rate  # (num_tasks, num_tasks)


def compute_participation_ratio(flat_grads):
    """
    Per-task participation ratio — measures effective gradient dimensionality.
    Near 1: dense/uniform gradient. Near 0: sparse/concentrated (overparameterized).
    """
    n   = flat_grads.shape[1]
    l1  = jnp.abs(flat_grads).sum(axis=1)          # (num_tasks,)
    l2_sq = (flat_grads ** 2).sum(axis=1)           # (num_tasks,)
    return (l1 ** 2) / (n * l2_sq.clip(min=1e-10))  # (num_tasks,)


def compute_effective_rank(flat_grads):
    """
    Effective rank of the joint gradient matrix via singular value entropy.
    Computed on the (num_tasks x num_tasks) Gram matrix for efficiency.
    Low effective rank => tasks are collapsing onto a shared gradient subspace.
    """
    G  = flat_grads @ flat_grads.T                          # (num_tasks, num_tasks)
    sv = jnp.linalg.svd(G, compute_uv=False)               # (num_tasks,)
    sv_dist = sv / sv.sum().clip(min=1e-10)
    entropy = -(sv_dist * jnp.log(sv_dist + 1e-10)).sum()
    return jnp.exp(entropy)


def compute_conflict_metrics(cos_sim_mat, flat_grads, eps=1e-3, tau=1.0):
    num_tasks  = flat_grads.shape[0]
    off_diag   = 1 - jnp.eye(num_tasks)                    # float mask, (num_tasks, num_tasks)

    # --- existing metrics ---
    conflict_mask           = (cos_sim_mat < 0).astype(jnp.float32)

    num_off_diag            = num_tasks * (num_tasks - 1)
    conflict_rate           = (conflict_mask * off_diag).sum() / num_off_diag

    magnitudes              = jnp.linalg.norm(flat_grads, axis=1)
    outer_mag               = magnitudes[:, None] * magnitudes[None, :]
    conflict_magnitude      = jnp.where(
                                  (conflict_mask * off_diag).astype(bool),
                                  jnp.abs(cos_sim_mat) * outer_mag,
                                  0.0
                              )
    mean_conflict_magnitude = (conflict_magnitude * off_diag).sum() / num_off_diag

    angles                  = jnp.degrees(jnp.arccos(jnp.clip(cos_sim_mat, -1.0, 1.0)))
    mean_conflict_angle     = (angles * off_diag).sum() / num_off_diag

    per_task_conflict_rate  = (conflict_mask * off_diag).sum(axis=1) / (num_tasks - 1)
    per_task_grad_magnitude = magnitudes

    # --- new elementwise metrics ---
    interference_rate           = compute_sparsity_mismatch(flat_grads, eps=eps, tau=tau)
    avg_interference_rate       = (interference_rate * off_diag).sum() / num_off_diag
    asymmetry                   = (
                                      jnp.abs(interference_rate - interference_rate.T) * off_diag
                                  ).sum() / num_off_diag
    per_task_interference_in    = (interference_rate * off_diag).sum(axis=0) / (num_tasks - 1)
    per_task_interference_out   = (interference_rate * off_diag).sum(axis=1) / (num_tasks - 1)

    participation_ratios        = compute_participation_ratio(flat_grads)
    avg_participation_ratio     = participation_ratios.mean()

    effective_rank              = compute_effective_rank(flat_grads)

    return {
        # existing
        "conflict_rate":                conflict_rate,
        "mean_conflict_magnitude":      mean_conflict_magnitude,
        "mean_conflict_angle":          mean_conflict_angle,
        "per_task_conflict_rate":       per_task_conflict_rate,
        "per_task_grad_magnitude":      per_task_grad_magnitude,
        "pairwise_conflict":            conflict_mask,
        "pairwise_cos_sim":             cos_sim_mat,
        "pairwise_angle":               angles,
        # new
        "avg_interference_rate":        avg_interference_rate,
        "interference_asymmetry":       asymmetry,
        "per_task_interference_in":     per_task_interference_in,
        "per_task_interference_out":    per_task_interference_out,
        "pairwise_interference_rate":   interference_rate,
        "avg_participation_ratio":      avg_participation_ratio,
        "per_task_participation_ratio": participation_ratios,
        "effective_rank":               effective_rank,
    }
