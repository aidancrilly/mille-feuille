import jax
import equinox as eqx

from typing import Tuple
from jaxtyping import Array, PRNGKeyArray

from .actor import Actor
from .critic import Critic

class Agent(eqx.Module):
    actor: Actor
    critic: Critic
    actor_tgt: Actor
    critic_tgt: Critic
    gamma: float
    tau: float

def copy_module(old):
    params, static = eqx.partition(old, eqx.is_array)
    new_params = jax.tree.map(lambda x : x, params)
    return eqx.combine(new_params, static)

def make_agent(
    key: PRNGKeyArray,
    state_dim: int,
    action_dim: int,
    action_low: Array,
    action_high: Array,
    hidden: Tuple[int, ...] = (256, 256),
    gamma: float = 0.99,
    tau: float = 0.005,
) -> Agent:
    k1, k2 = jax.random.split(key, 2)

    actor = Actor(k1, state_dim, action_dim, hidden, action_low, action_high)
    critic = Critic(k2, state_dim, action_dim, hidden)

    actor_tgt = copy_module(actor)
    critic_tgt = copy_module(critic)

    return Agent(
        actor=actor,
        critic=critic,
        actor_tgt=actor_tgt,
        critic_tgt=critic_tgt,
        gamma=gamma,
        tau=tau,
    )