import jax
import equinox as eqx
import jax.numpy as jnp
import numpy as np

from typing import Dict, Tuple

from .actor import Actor
from .critic import Critic
from .agent import Agent
from .buffer import ReplayBuffer
from .environment import Env

def soft_update(src, dst, tau: float):
    """
    Polyak averaging: dst <- tau*src + (1-tau)*dst
    """
    src_p = eqx.filter(src, eqx.is_array)
    dst_p = eqx.filter(dst, eqx.is_array)
    new_p = jax.tree_util.tree_map(lambda s, d: tau * s + (1.0 - tau) * d, src_p, dst_p)
    return eqx.combine(new_p, dst)

@eqx.filter_jit
def ddpg_update(agent: Agent, batch: Dict[str, jnp.ndarray]) -> Tuple[Agent, Dict[str, jnp.ndarray]]:
    s = batch["s"]
    a = batch["a"]
    r = batch["r"]
    ns = batch["ns"]
    d = batch["d"]

    # ----- Critic update -----
    def critic_loss_fn(critic: Critic) -> jnp.ndarray:
        na = agent.actor_tgt(ns)                     # shape (B,action_dim)
        q_tgt = agent.critic_tgt(ns, na)             # shape (B,1)
        y = r + agent.gamma * (1.0 - d) * q_tgt      # shape (B,1)
        q = critic(s, a)
        return jnp.mean((q - jax.lax.stop_gradient(y)) ** 2)

    critic_loss, critic_grads = eqx.filter_value_and_grad(critic_loss_fn)(agent.critic)
    critic_params = eqx.filter(agent.critic, eqx.is_array)
    critic_updates, new_opt_state_c = agent.opt_critic.update(critic_grads, agent.opt_state_critic, critic_params)
    new_critic = eqx.apply_updates(agent.critic, critic_updates)

    # ----- Actor update -----
    def actor_loss_fn(actor: Actor) -> jnp.ndarray:
        pa = actor(s)            # shape (B,action_dim)
        q = new_critic(s, pa)
        return -jnp.mean(q)

    actor_loss, actor_grads = eqx.filter_value_and_grad(actor_loss_fn)(agent.actor)
    actor_params = eqx.filter(agent.actor, eqx.is_array)
    actor_updates, new_opt_state_a = agent.opt_actor.update(actor_grads, agent.opt_state_actor, actor_params)
    new_actor = eqx.apply_updates(agent.actor, actor_updates)

    # ----- Target networks (Polyak averaging) -----
    new_actor_tgt = soft_update(new_actor, agent.actor_tgt, agent.tau)
    new_critic_tgt = soft_update(new_critic, agent.critic_tgt, agent.tau)

    new_agent = Agent(
        actor=new_actor,
        critic=new_critic,
        actor_tgt=new_actor_tgt,
        critic_tgt=new_critic_tgt,
        opt_actor=agent.opt_actor,
        opt_critic=agent.opt_critic,
        opt_state_actor=new_opt_state_a,
        opt_state_critic=new_opt_state_c,
        gamma=agent.gamma,
        tau=agent.tau,
        rng=agent.rng,
    )

    metrics = {"critic_loss": critic_loss, "actor_loss": actor_loss}
    return new_agent, metrics

def train(
    env: Env,
    agent: Agent,
    total_steps: int = 50_000,
    warmup_steps: int = 1_000,
    batch_size: int = 256,
    replay_size: int = 1_000_000,
    explore_noise: float = 0.1,
    update_every: int = 1,
    updates_per_step: int = 1,
    print_every: int = 100,
    seed: int = 420,
) -> Agent:

    rb = ReplayBuffer(replay_size, env.cfg.state_dim)
    rng = np.random.default_rng(seed)

    s = env.reset()
    episode_return = 0.0
    episode_len = 0

    for i in range(total_steps):
        # Choose action
        if i < warmup_steps:
            a = rng.uniform(env.cfg.action_low, env.cfg.action_high)
        else:
            agent.rng, k = jax.random.split(agent.rng)
            a = np.array(agent.actor(jnp.asarray(s[None,:])))
            a += rng.normal(scale=explore_noise,size=env.action_dim)
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
                agent, metrics = ddpg_update(agent, batch)

            if i % print_every == 0:
                cl = float(metrics["critic_loss"])
                al = float(metrics["actor_loss"])
                print(f"step={i} critic_loss={cl:.6f} actor_loss={al:.6f} rb={rb.size}")

    return agent
