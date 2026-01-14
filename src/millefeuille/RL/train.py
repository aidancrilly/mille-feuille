import jax
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax

from typing import Dict, Tuple

from .actor import Actor
from .critic import Critic
from .agent import Agent
from .buffer import ReplayBuffer
from .environment import Env

def actor_loss_fn(actor: Actor, critic: Critic, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    pa = actor(batch["s"])            # shape (B,action_dim)
    q = critic(batch["s"], pa)
    return -jnp.mean(q)

def critic_loss_fn(critic: Critic, critic_tgt: Critic, actor_tgt: Agent, 
                   batch: Dict[str, jnp.ndarray], gamma: float) -> jnp.ndarray:
    na = actor_tgt(batch["ns"])                     # shape (B,action_dim)
    q_tgt = critic_tgt(batch["ns"], na)             # shape (B,1)
    y = batch["r"] + gamma * (1.0 - batch["d"]) * q_tgt      # shape (B,1)
    q = critic(batch["s"], batch["a"])
    return jnp.mean((q - jax.lax.stop_gradient(y)) ** 2)

def soft_update(src, dst, tau: float):
    """
    Polyak averaging: dst <- tau*src + (1-tau)*dst
    """
    tar = eqx.filter(src, eqx.is_array)
    on, off = eqx.partition(dst, eqx.is_array)
    new = jax.tree.map(lambda t, o : t * (1 - tau) + o * tau, tar, on)
    return eqx.combine(new,off)

def train(
    env: Env,
    agent: Agent,
    total_steps: int = 50_000,
    warmup_steps: int = 1_000,
    batch_size: int = 256,
    replay_size: int = 1_000_000,
    explore_noise: float = 0.1,
    lr_actor: float = 3e-4,
    lr_critic: float = 3e-4,
    update_every: int = 1,
    updates_per_step: int = 1,
    print_every: int = 100,
    seed: int = 420,
) -> Agent:

    rb = ReplayBuffer(replay_size, env.cfg.state_dim, env.cfg.action_dim)
    rng = np.random.default_rng(seed)
    jax_key = jax.random.PRNGKey(seed)

    s = env.reset()
    episode_return = 0.0
    episode_len = 0

    opt_actor = optax.adam(lr_actor)
    opt_critic = optax.adam(lr_critic)

    opt_state_actor = opt_actor.init(eqx.filter(agent.actor, eqx.is_array))
    opt_state_critic = opt_critic.init(eqx.filter(agent.critic, eqx.is_array))

    @eqx.filter_jit
    def ddpg_update(agent: Agent, opt_state_a, opt_state_c, batch: Dict[str, jnp.ndarray]) -> Tuple[Agent, Dict[str, jnp.ndarray]]:

        # ----- Critic update -----
        critic_loss, critic_grads = eqx.filter_value_and_grad(critic_loss_fn)(
            agent.critic, agent.critic_tgt, agent.actor_tgt, 
            batch, agent.gamma
            )
        critic_params = eqx.filter(agent.critic, eqx.is_array)
        critic_updates, new_opt_state_c = opt_critic.update(critic_grads, opt_state_c, critic_params)
        new_critic = eqx.apply_updates(agent.critic, critic_updates)

        # ----- Actor update -----
        actor_loss, actor_grads = eqx.filter_value_and_grad(actor_loss_fn)(agent.actor, new_critic, batch)
        actor_params = eqx.filter(agent.actor, eqx.is_array)
        actor_updates, new_opt_state_a = opt_actor.update(actor_grads, opt_state_a, actor_params)
        new_actor = eqx.apply_updates(agent.actor, actor_updates)

        # ----- Target networks (Polyak averaging) -----
        new_actor_tgt = soft_update(new_actor, agent.actor_tgt, agent.tau)
        new_critic_tgt = soft_update(new_critic, agent.critic_tgt, agent.tau)

        # ----- Apply updates to agent ----
        agent = eqx.tree_at(lambda agent: agent.actor, agent, new_actor)
        agent = eqx.tree_at(lambda agent: agent.critic, agent, new_critic)
        agent = eqx.tree_at(lambda agent: agent.actor_tgt, agent, new_actor_tgt)
        agent = eqx.tree_at(lambda agent: agent.critic_tgt, agent, new_critic_tgt)

        metrics = {"critic_loss": critic_loss, "actor_loss": actor_loss}
        return agent, new_opt_state_a, new_opt_state_c, metrics

    # Training loop
    for i in range(total_steps):
        # Choose action
        if i < warmup_steps:
            a = rng.uniform(env.cfg.action_low, env.cfg.action_high)
        else:
            jax_key, _ = jax.random.split(jax_key)
            a = np.squeeze(np.array(agent.actor(jnp.asarray(s[None,:]))),axis=0)
            a += rng.normal(scale=explore_noise,size=env.cfg.action_dim)
            a = np.clip(a, env.cfg.action_low, env.cfg.action_high)

        ns, r, done, info = env.step(a)
        rb.add(s, a, r, ns, done)

        s = ns
        episode_return += r
        episode_len += 1

        if done:
            print(f"[episode done] return={episode_return:.3f} len={episode_len} last_Y={float(np.squeeze(info['Y'])):.6f}")
            s = env.reset()
            episode_return = 0.0
            episode_len = 0

        # Updates
        if rb.size >= batch_size and i >= warmup_steps and (i % update_every == 0):
            for _ in range(updates_per_step):
                batch_np = rb.sample(batch_size)
                batch = {k: jnp.asarray(v) for k, v in batch_np.items()}
                agent, opt_state_actor, opt_state_critic, metrics = ddpg_update(agent, opt_state_actor, opt_state_critic, batch)

            if i % print_every == 0:
                cl = float(metrics["critic_loss"])
                al = float(metrics["actor_loss"])
                print(f"step={i} critic_loss={cl:.6f} actor_loss={al:.6f} rb={rb.size}")

    return agent
