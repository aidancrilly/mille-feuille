import jax
import jax.numpy as jnp
import equinox as eqx

from typing import Tuple

class Actor(eqx.Module):
    mlp: eqx.module
    action_low: jax.Array = eqx.field(static=True)
    action_high: jax.Array = eqx.field(static=True)

    def __init__(self, key: jax.random.KeyArray, state_dim: int, action_dim: int, hidden: Tuple[int, ...], action_low: jax.Array, action_high: jax.Array):
        key, _ = jax.random.split(key)
        layers = [eqx.nn.Linear(state_dim, hidden[0], key), eqx.nn.Lambda(jax.nn.relu)]

        if(len(hidden) > 1):
            for h1, h2 in zip(hidden[1:-1],hidden[2:]):
                key, _ = jax.random.split(key)
                layers.append(eqx.nn.Linear(h1, h2, key))
                layers.append(eqx.nn.Lambda(jax.nn.relu))

        key, _ = jax.random.split(key)
        layers.append(eqx.nn.Linear(hidden[-1],action_dim,key))
        layers.append(eqx.nn.Lambda(jax.nn.sigmoid))

        self.mlp = jax.vmap(eqx.nn.Sequential(layers))
        self.action_low = action_low
        self.action_high = action_high

    def __call__(self, s: jnp.ndarray) -> jnp.ndarray:
        # outputs in [0, 1] then scale to [low, high]
        u = self.mlp(s)  # shape (..., action_dim)
        return self.action_low + u * (self.action_high - self.action_low)
