from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp


class Critic(eqx.Module):
    mlp: eqx.Module

    def __init__(self, key, state_dim: int, action_dim: int, hidden: Tuple[int, ...]):
        key, _ = jax.random.split(key)
        layers = [eqx.nn.Linear(state_dim + action_dim, hidden[0], key=key), eqx.nn.Lambda(jax.nn.relu)]

        if len(hidden) > 1:
            for h1, h2 in zip(hidden[1:-1], hidden[2:], strict=True):
                key, _ = jax.random.split(key)
                layers.append(eqx.nn.Linear(h1, h2, key=key))
                layers.append(eqx.nn.Lambda(jax.nn.relu))

        key, _ = jax.random.split(key)
        layers.append(eqx.nn.Linear(hidden[-1], 1, key=key))

        self.mlp = eqx.nn.Sequential(layers)

    def __call__(self, s: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([s, a], axis=-1)
        return jax.vmap(self.mlp)(x)
