from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp


class Actor(eqx.Module):
    mlp: eqx.Module
    action_low: Tuple = eqx.field(static=True, converter=tuple)
    action_high: Tuple = eqx.field(static=True, converter=tuple)

    def __init__(
        self,
        key,
        state_dim: int,
        action_dim: int,
        hidden: Tuple[int, ...],
        action_low: jax.Array,
        action_high: jax.Array,
    ):
        key, _ = jax.random.split(key)
        layers = [eqx.nn.Linear(state_dim, hidden[0], key=key), eqx.nn.Lambda(jax.nn.relu)]

        if len(hidden) > 1:
            for h1, h2 in zip(hidden[1:-1], hidden[2:], strict=True):
                key, _ = jax.random.split(key)
                layers.append(eqx.nn.Linear(h1, h2, key=key))
                layers.append(eqx.nn.Lambda(jax.nn.relu))

        key, _ = jax.random.split(key)
        layers.append(eqx.nn.Linear(hidden[-1], action_dim, key=key))
        layers.append(eqx.nn.Lambda(jax.nn.sigmoid))

        self.mlp = eqx.nn.Sequential(layers)
        self.action_low = action_low
        self.action_high = action_high

    def __call__(self, s: jnp.ndarray) -> jnp.ndarray:
        # outputs in [0, 1] then scale to [low, high]
        u = jax.vmap(self.mlp)(s)  # shape (..., action_dim)
        return jnp.array(self.action_low) + u * (jnp.array(self.action_high) - jnp.array(self.action_low))
