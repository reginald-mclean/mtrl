class AdaptiveSampler:
    def __init__(self, n_tasks, ema_alpha=0.1, temperature=1.0):
        self.n_tasks = n_tasks
        self.ema_alpha = ema_alpha
        self.temperature = temperature
        self.task_losses_ema = jnp.ones(n_tasks)  # Initialize equally

    def update(self, task_losses):
        """Update EMA of task losses"""
        self.task_losses_ema = (
            self.ema_alpha * task_losses + 
            (1 - self.ema_alpha) * self.task_losses_ema
        )

    def get_sampling_weights(self):
        """Get sampling probabilities based on losses"""
        # Softmax with temperature
        logits = self.task_losses_ema / self.temperature
        weights = jnp.exp(logits) / jnp.exp(logits).sum()
        return weights

    def sample_batch_sizes(self, task_losses, total_batch_size):
        """Allocate batch size across tasks"""
        self.update(task_losses)
        weights = self.get_sampling_weights()
        batch_sizes = (weights * total_batch_size).astype(int)
        # Handle rounding to ensure sum = total_batch_size
        remainder = total_batch_size - batch_sizes.sum()
        if remainder > 0:
            # Add remainder to tasks with highest weights
            top_indices = jnp.argsort(weights)[-remainder:]
            batch_sizes = batch_sizes.at[top_indices].add(1)
        return batch_sizes


class Sampler:
    def sample_batch_sizes(self, task_losses, total_batch_size):
        # Normalize losses to get sampling probabilities
        loss_weights = task_losses / task_losses.sum()
        batch_sizes = (loss_weights * total_batch_size).astype(int)
        # Handle rounding to ensure total_batch_size is met
        remainder = total_batch_size - batch_sizes.sum()
        batch_sizes[:remainder] += 1
        return batch_sizes
