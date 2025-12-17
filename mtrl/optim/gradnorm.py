from typing import TYPE_CHECKING, Any, NamedTuple

import chex
import jax
import jax.flatten_util
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PyTree

if TYPE_CHECKING:
    from mtrl.config.optim import OptimizerConfig

# NOTE: GRADNORM ALGORITHM:
# Initialize $w_i(0)=1 \forall i$
# Initialize network weights $\mathcal{W}$
# Pick value for $\alpha>0$ and pick the weights $W$ (usually the
#     final layer of weights which are shared between tasks)
# for $t=0$ to max_train_steps $^{-10}$
#     Input batch $x_i$ to compute $L_i(t) \forall i$ and
#         $L(t)=\sum_i w_i(t) L_i(t)$ [standard forward pass]
#     Compute $G_W^{(i)}(t)$ and $r_i(t) \forall i$
#     Compute $\bar{G}_W(t)$ by averaging the $G_W^{(i)}(t)$
#     Compute $L_{\text {grad }}=\sum_i\left|G_W^{(i)}(t)-\bar{G}_W(t) \times\left[r_i(t)\right]^\alpha\right|_1$
#     Compute GradNorm gradients $\nabla_{w_i} L_{\text {grad }}$, keeping
#         targets $\bar{G}_W(t) \times\left[r_i(t)\right]^\alpha$ constant
#     Compute standard gradients $\nabla_{\mathcal{W}} L(t)$
#     Update $w_i(t) \mapsto w_i(t+1)$ using $\nabla_{w_i} L_{\text {grad }}$
#     Update $\mathcal{W}(t) \mapsto \mathcal{W}(t+1)$ using $\nabla_{\mathcal{W}} L(t)$ [standard
#         backward pass]
#     Renormalize $w_i(t+1)$ so that $\sum_i w_i(t+1)=T$
# end for


class GradNormState(NamedTuple):
    opt_state: optax.OptState
    task_weights: Float[Array, " num_tasks"]
    original_losses: Float[Array, " num_tasks"]


def clip_per_task_grads(task_grads, max_norm=1.0):
    """Clip each task's gradients independently.
    
    Args:
        task_grads: PyTree with shape prefix (num_tasks, ...)
        max_norm: Maximum gradient norm per task
    
    Returns:
        Clipped gradients with same structure
    """
    def clip_single_task(grad_tree):
        flat, unravel = jax.flatten_util.ravel_pytree(grad_tree)
        norm = jnp.linalg.norm(flat)
        clip_coef = jnp.minimum(1.0, max_norm / (norm + 1e-8))
        clipped = flat * clip_coef
        return unravel(clipped)
    
    # Vmap over the task dimension (first axis)
    return jax.vmap(clip_single_task)(task_grads)


def gradnorm(
    num_tasks: int,
    optim: "OptimizerConfig",
    asymmetry: float = 0.12,
    initial_weights: Array | None = None,
    max_grad_norm: float | None = None,
) -> optax.GradientTransformationExtraArgs:
    _optim = optim.spawn()

    def init_fn(params: optax.Params) -> GradNormState:
        del params
        if initial_weights is None:
            weights = jnp.ones(num_tasks)
        else:
            weights = initial_weights

        opt_state = _optim.init(weights)
        # Normalize weights to sum to num_tasks
        weights = (weights / weights.sum()) * num_tasks
        return GradNormState(
            opt_state=opt_state,
            task_weights=weights,
            original_losses=jnp.full((num_tasks,), jnp.inf),
        )

    def update_fn(
        updates: PyTree[Float[Array, " num_tasks ..."]],
        state: GradNormState,
        params: optax.Params | None = None,
        **extra_args: Any,
    ) -> tuple[optax.Updates, GradNormState]:
        del params

        task_losses: Float[Array, "num_tasks"] | None
        task_losses = extra_args.get("task_losses")
        assert task_losses is not None, "task_losses must be provided"

        chex.assert_shape(task_losses, (num_tasks,))
        chex.assert_tree_shape_prefix(updates, (num_tasks,))



        if max_grad_norm:
            updates = clip_per_task_grads(updates)

        original_losses = jnp.where(
            jnp.isinf(state.original_losses),
            task_losses,
            state.original_losses,
        )

        improvement = task_losses / (original_losses + 1e-12)  # \tilde{L}_i^(t)
        rel_inverse_rate = improvement / (jnp.mean(improvement) + 1e-12)  # r_i(t)

        '''def gradnorm_loss(params) -> Float[Array, ""]:
            def compute_task_grad_norm(  # G_W^(i)(t)
                task_weight: Float[Array, ""],
                task_grads: PyTree[Float[Array, " ..."]],
            ) -> jax.Array:
                flat_grads, _ = jax.flatten_util.ravel_pytree(task_grads)
                return jnp.linalg.norm(task_weight * flat_grads)

            grad_norms = jax.vmap(compute_task_grad_norm, in_axes=(0, 0))(
                params, updates
            )
            avg_grad_norm = jnp.mean(grad_norms)  # \bar{G}_W(t)
            grad_norm_targets = avg_grad_norm * rel_inverse_rate**asymmetry
            return jnp.sum(jnp.abs(grad_norms - grad_norm_targets))'''

        def gradnorm_loss(params) -> Float[Array, ""]:
            def compute_task_grad_norm(task_grads: PyTree[Float[Array, " ..."]]) -> jax.Array:
                flat_grads, _ = jax.flatten_util.ravel_pytree(task_grads)
                return jnp.linalg.norm(flat_grads)  # Don't multiply by weights!

            grad_norms = jax.vmap(compute_task_grad_norm)(updates)  # G_W^(i)(t)
            avg_grad_norm = jnp.mean(grad_norms)  # \bar{G}_W(t)
            grad_norm_targets = avg_grad_norm * rel_inverse_rate**asymmetry
            return jnp.sum(jnp.abs(grad_norms - grad_norm_targets))


        _, gradnorm_grad = jax.value_and_grad(gradnorm_loss)(state.task_weights)
        gradnorm_updates, new_opt_state = _optim.update(
            gradnorm_grad, state.opt_state, state.task_weights
        )


        new_weights = optax.apply_updates(state.task_weights, gradnorm_updates)
        assert isinstance(new_weights, jax.Array)
        new_weights = (new_weights / new_weights.sum()) * num_tasks

        weighted_updates = jax.tree.map(
            lambda x: jnp.einsum("i, i... -> ...", new_weights, x), updates
        )

        new_state = GradNormState(
            opt_state=new_opt_state,
            task_weights=new_weights,
            original_losses=original_losses,
        )

        return weighted_updates, new_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)  # pyright: ignore [reportArgumentType]
