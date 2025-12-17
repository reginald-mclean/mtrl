from functools import partial
from typing import Any, NamedTuple

import chex
import jax
import jax.flatten_util
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float


class CAGradState(NamedTuple):
    task_weights: Float[Array, "num_tasks"]
    avg_grad_magnitude: Float[Array, ""]
    avg_grad_magnitude_before_surgery: Float[Array, ""]
    avg_cosine_similarity: Float[Array, ""]
    cagrad_objective: Float[Array, ""]


def cagrad(
    num_tasks: int,
    c: float = 0.5,
    num_iterations: int = 21,
    learning_rate: float = None,
    momentum: float = 0.5,
    cosine_sim_logs: bool = False,
) -> optax.GradientTransformation:
    """ CAGrad (Conflict-Averse Gradient Descent) optimizer.
    Args:
        num_tasks: Number of tasks
        c: Hyperparameter controlling trade-off between conflict avoidance and staying near average
        num_iterations: Number of SGD iterations to optimize task weights (default 21)
        learning_rate: Learning rate for weight optimization. If None, uses 25 for <50 tasks, 50 otherwise
        momentum: Momentum for SGD weight optimization
        cosine_sim_logs: Whether to log cosine similarity metrics
    """
    
    if learning_rate is None:
        learning_rate = 25.0 if num_tasks < 50 else 50.0

    def cagrad_init(params: optax.Params) -> CAGradState:
        del params
        base_state = CAGradState(
            task_weights=jnp.ones(num_tasks) / num_tasks,
            avg_grad_magnitude=jnp.array(0.0),
            avg_grad_magnitude_before_surgery=jnp.array(0.0),
            avg_cosine_similarity=jnp.array(0.0)
            if cosine_sim_logs
            else jnp.array(jnp.nan),
            cagrad_objective=jnp.array(0.0),
        )
        return base_state

    @jax.jit
    def optimize_weights(flat_task_gradients: Float[Array, "num_tasks num_params"]) -> tuple[Float[Array, " num_tasks"], Float[Array, ""]]:
        """Optimize task weights using SGD to minimize CAGrad objective.
        
        Returns:
            task_weights: Optimized task weights (softmax normalized)
            obj_best: Best objective value found
        """
        # Compute Gram matrix GG = grads @ grads.T
        GG = jnp.matmul(flat_task_gradients, flat_task_gradients.T)  # [num_tasks, num_tasks]
        
        # Scale for numerical stability
        scale = jnp.sqrt(jnp.diag(GG) + 1e-4).mean()
        GG = GG / (scale ** 2)
        
        # Compute statistics for objective
        Gg = jnp.mean(GG, axis=1, keepdims=True)  # [num_tasks, 1] - mean over columns
        gg = jnp.mean(Gg, axis=0, keepdims=True)  # [1, 1] - overall mean
        
        # Regularization strength
        c_normalized = jnp.sqrt(gg + 1e-4) * c
        
        def objective(w: Float[Array, "num_tasks 1"]) -> Float[Array, ""]:
            """CAGrad objective: min_w w^T Gg + c * sqrt(w^T GG w)"""
            ww = w / (w.sum() + 1e-8)
            term1 = jnp.matmul(ww.T, Gg)
            term2 = c_normalized * jnp.sqrt(jnp.matmul(jnp.matmul(ww.T, GG), ww) + 1e-4)
            return (term1 + term2).squeeze()
        
        # Initialize weights and optimizer state
        w = jnp.zeros((num_tasks, 1))
        velocity = jnp.zeros((num_tasks, 1))
        w_best = w
        obj_best = jnp.array(jnp.inf)
        
        def sgd_step(carry, i):
            w, velocity, w_best, obj_best = carry
            
            # Compute objective and gradient
            obj_val, grad_w = jax.value_and_grad(objective)(w)
            
            # Track best weights
            is_better = obj_val < obj_best
            w_best = jnp.where(is_better, w, w_best)
            obj_best = jnp.where(is_better, obj_val, obj_best)
            
            # SGD update with momentum (only for first num_iterations-1 steps)
            velocity = momentum * velocity + grad_w
            w = w - learning_rate * velocity
            
            return (w, velocity, w_best, obj_best), None
        
        # Run SGD for num_iterations-1 steps (last iteration just evaluates)
        (_, _, w_best, obj_best), _ = jax.lax.scan(
            sgd_step,
            (w, velocity, w_best, obj_best),
            jnp.arange(num_iterations - 1)
        )
        
        # Final evaluation
        obj_val = objective(w)
        is_better = obj_val < obj_best
        w_best = jnp.where(is_better, w, w_best)
        obj_best = jnp.where(is_better, obj_val, obj_best)
        
        # Convert to probability distribution using softmax
        task_weights = jax.nn.softmax(w_best.squeeze(), axis=0)
        
        return task_weights, obj_best

    @jax.jit
    def compute_cagrad_gradient(
        flat_task_gradients: Float[Array, "num_tasks num_params"],
        task_weights: Float[Array, " num_tasks"],
    ) -> Float[Array, " num_params"]:
        """Compute final CAGrad gradient from task weights.
        
        Args:
            flat_task_gradients: Flattened per-task gradients
            task_weights: Optimized task weights (from softmax)
            
        Returns:
            Combined gradient vector
        """
        # Recompute Gram matrix (cached by XLA)
        GG = jnp.matmul(flat_task_gradients, flat_task_gradients.T)
        scale = jnp.sqrt(jnp.diag(GG) + 1e-4).mean()
        GG = GG / (scale ** 2)
        
        # Compute gradient norm with optimal weights
        gw_norm = jnp.sqrt(
            jnp.matmul(
                jnp.matmul(task_weights.reshape(1, -1), GG),
                task_weights.reshape(-1, 1)
            ) + 1e-4
        )
        
        # Compute regularization parameter
        Gg = jnp.mean(GG, axis=1, keepdims=True)
        gg = jnp.mean(Gg)
        c_normalized = jnp.sqrt(gg + 1e-4) * c
        lmbda = c_normalized / (gw_norm + 1e-4)
        
        # Combine gradients: (1/T + w*lambda) * grads / (1 + c^2)
        weights_combined = (1.0 / num_tasks + task_weights * lmbda.squeeze())
        g = jnp.sum(weights_combined.reshape(-1, 1) * flat_task_gradients, axis=0) / (1 + c ** 2)
        
        return g

    @jax.jit
    def cagrad_update(
        updates: optax.Updates,
        state: CAGradState,
        params: optax.Params | None = None,
        **extra_args: Any,
    ) -> tuple[optax.Updates, CAGradState]:
        del extra_args
        chex.assert_tree_shape_prefix(updates, (num_tasks,))
        assert params is not None

        # Flatten all task gradients
        flat_task_gradients = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])(
            updates
        )
        # Shape: (num_tasks, num_params)
        
        # Clip each task's gradients to prevent extreme values
        def clip_single_grad(grad_vec):
            norm = jnp.linalg.norm(grad_vec)
            clip_coef = jnp.minimum(1.0, 1.0 / (norm + 1e-8))
            return grad_vec * clip_coef
        
        flat_task_gradients = jax.vmap(clip_single_grad)(flat_task_gradients)
        
        # Optimize task weights
        task_weights, obj_best = optimize_weights(flat_task_gradients)
        
        # Compute final gradient
        combined_grad = compute_cagrad_gradient(flat_task_gradients, task_weights)
        
        # Compute metrics
        new_state = CAGradState(
            task_weights=task_weights,
            avg_grad_magnitude=jnp.linalg.norm(combined_grad),
            avg_grad_magnitude_before_surgery=(
                jnp.linalg.norm(flat_task_gradients, axis=1)
            ).mean(),
            avg_cosine_similarity=jnp.array(jnp.nan),
            cagrad_objective=obj_best,
        )

        # Compute cosine similarity metrics if requested
        if cosine_sim_logs:
            def calc_avg_cos_sim(grads):
                def calc_cos_sim(selected_grad, all_grads):
                    cos_sims = jnp.sum(selected_grad * all_grads, axis=1) / (
                        jnp.linalg.norm(selected_grad)
                        * jnp.linalg.norm(all_grads, axis=1)
                        + 1e-8
                    )
                    return cos_sims

                cos_sim_mat = jax.vmap(calc_cos_sim, in_axes=(0, None), out_axes=-1)(
                    grads, grads
                )
                # Get upper triangle (avoid diagonal)
                mask = jnp.triu(jnp.ones((num_tasks, num_tasks)), k=1)
                num_unique = mask.sum()

                masked_cos_sim = mask * cos_sim_mat
                avg_cos_sim = masked_cos_sim.flatten().sum() / (num_unique + 1e-8)
                return avg_cos_sim

            avg_cos_sim = calc_avg_cos_sim(flat_task_gradients)
            new_state = new_state._replace(avg_cosine_similarity=avg_cos_sim)

        # Unravel back to pytree structure
        _, unravel_fn = jax.flatten_util.ravel_pytree(params)
        return unravel_fn(combined_grad), new_state

    return optax.GradientTransformationExtraArgs(
        init=cagrad_init,
        update=cagrad_update,
    )
