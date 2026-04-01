"""DrQ-style image augmentation: random crop + intensity perturbation.

Matches the reference implementation in Archive/multi/augmentation.py.
"""

import functools

import jax
import jax.numpy as jnp


@functools.partial(jax.vmap, in_axes=(0, 0, 0, None))
def _crop_with_indices(img, x, y, cropped_shape):
    cropped_image = jax.lax.dynamic_slice(img, [x, y, 0], cropped_shape[1:])
    return cropped_image


def _per_image_random_crop(key, img, cropped_shape):
    batch_size, width, height = cropped_shape[:-1]
    key_x, key_y = jax.random.split(key, 2)
    x = jax.random.randint(key_x, shape=(batch_size,), minval=0, maxval=img.shape[1] - width)
    y = jax.random.randint(key_y, shape=(batch_size,), minval=0, maxval=img.shape[2] - height)
    return _crop_with_indices(img, x, y, cropped_shape)


def _intensity_aug(key, x, scale=0.05):
    r = jax.random.normal(key, shape=(x.shape[0], 1, 1, 1))
    noise = 1.0 + (scale * jnp.clip(r, -2.0, 2.0))
    return x * noise


def drq_image_augmentation(key, obs, img_pad=4):
    flat_obs = obs.reshape(-1, *obs.shape[-3:])
    paddings = [(0, 0), (img_pad, img_pad), (img_pad, img_pad), (0, 0)]
    cropped_shape = flat_obs.shape
    flat_obs = jnp.pad(flat_obs, paddings, 'edge')
    key1, key2 = jax.random.split(key, num=2)
    cropped_obs = _per_image_random_crop(key2, flat_obs, cropped_shape)
    aug_obs = _intensity_aug(key1, cropped_obs)
    return aug_obs.reshape(*obs.shape)


@jax.jit
def augment(x, rng):
    """Augment uint8 observations: normalize to [-1,1] then apply DrQ augmentation.

    Args:
        x: observations in (B, C, H, W) uint8 format (frame-stacked)
        rng: JAX PRNG key

    Returns:
        Augmented float32 observations in (B, H, W, C) format, range ~ [-1, 1]
        Updated rng key
    """
    rng, key = jax.random.split(rng, 2)
    images = jnp.transpose(x, (0, 2, 3, 1))
    images = (images.astype(jnp.float32) / 255.0 - 0.5) * 2
    out = drq_image_augmentation(key, images)
    return out, rng
