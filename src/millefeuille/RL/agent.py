import jax
import equinox as eqx
import optax
import copy

from dataclasses import dataclass
from typing import Tuple

from .actor import Actor
from .critic import Critic

@dataclass
class Agent:
    actor: Actor
    critic: Critic
    actor_tgt: Actor
    critic_tgt: Critic
    opt_actor: optax.GradientTransformation
    opt_critic: optax.GradientTransformation
    opt_state_actor: optax.OptState
    opt_state_critic: optax.OptState
    gamma: float
    tau: float
    rng: jax.random.KeyArray


def make_agent(
    key: jax.random.KeyArray,
    state_dim: int,
    action_dim: int,
    action_low: jax.Array,
    action_high: jax.Array,
    hidden: Tuple[int, ...] = (256, 256),
    lr_actor: float = 3e-4,
    lr_critic: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 0.005,
) -> Agent:
    k1, k2, k3 = jax.random.split(key, 3)

    actor = Actor(k1, state_dim, action_dim, hidden, action_low, action_high)
    critic = Critic(k2, state_dim, action_dim, hidden)

    actor_tgt = copy.deepcopy(actor)
    critic_tgt = copy.deepcopy(critic)

    opt_actor = optax.adam(lr_actor)
    opt_critic = optax.adam(lr_critic)

    opt_state_actor = opt_actor.init(eqx.filter(actor, eqx.is_array))
    opt_state_critic = opt_critic.init(eqx.filter(critic, eqx.is_array))

    return Agent(
        actor=actor,
        critic=critic,
        actor_tgt=actor_tgt,
        critic_tgt=critic_tgt,
        opt_actor=opt_actor,
        opt_critic=opt_critic,
        opt_state_actor=opt_state_actor,
        opt_state_critic=opt_state_critic,
        gamma=gamma,
        tau=tau,
        rng=k3,
    )