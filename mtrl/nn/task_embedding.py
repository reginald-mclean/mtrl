import flax.linen as nn
import jax.numpy as jnp
from mtrl.config.nn import TaskEmbeddingConfig

class TaskEmbedding(nn.Module):
    config: TaskEmbeddingConfig

    @nn.compact
    def __call__(self, x):
        emb = nn.Embed(num_embeddings=self.config.num_tasks, features=self.config.embed_dim)(x)
        norm = jnp.linalg.norm(emb, axis=-1, keepdims=True)
        return emb / (norm + 1e-8)

