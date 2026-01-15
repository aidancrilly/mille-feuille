#%%
import millefeuille.RL as RL 
import numpy as np
import jax
import jax.numpy as jnp

from scipy.integrate import odeint
from millefeuille.simulator import PythonSimulator

class Pendulum(PythonSimulator):

    def __init__(self, L, dt, state_scale = np.pi, action_scale = 10.0):
        self.dt = dt
        self.g = 9.8
        self.L = L
        self.state_scale = state_scale
        self.action_scale = action_scale

    def dydt(self, x, t, a):
        dthetadt = x[1]
        dthetadotdt = -self.g/self.L*np.sin(x[0]) + self.action_scale * a[0]
        return np.array([dthetadt,dthetadotdt])

    def solve(self, Xs, As):
        Xs_next = []
        for i in range(Xs.shape[0]):
            Xs_next.append(odeint(self.dydt, Xs[i], [0.0, self.dt], args=(As[i],))[-1,:])
        return np.array(Xs_next).reshape(Xs.shape)

    def __call__(self, indices, Xs, Ss = None):
        # Xs is state, action, time
        x = Xs[:,:2] * self.state_scale
        a = Xs[:,2:-1]
        x_next = self.solve(x,a)
        # Wrap phase
        x_next[:,0] = (x_next[:,0] + np.pi) % (2 * np.pi) - np.pi
        Ps = x_next / self.state_scale
        cos_theta = np.cos(x_next[:,0])
        exp_arg = (cos_theta+1)**2 + (0.25 * x_next[:,1])**2
        Ys = 2*np.exp(-exp_arg) - 1
        return Ps, Ys
    
# %%

sim = Pendulum(L = 1.0, dt = 0.1)
scheduler = None

cfg = RL.SimEnvConfig(state_dim=2, action_dim=1, action_low=(-1.0,), action_high=(1.0,), horizon=100)
env = RL.Env(sim, scheduler, cfg)

key = jax.random.PRNGKey(0)
agent = RL.make_agent(key=key,
    state_dim=cfg.state_dim,
    action_dim=cfg.action_dim,
    action_low=cfg.action_low,
    action_high=cfg.action_high,
    hidden=(32,32),
    )

#%%

agent = RL.train(env, agent, explore_noise=0.01, total_steps=16000, warmup_steps=2000, updates_per_step=10, lr_actor=1e-3, lr_critic=1e-3)
print("done")

#%%

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

s = env.reset()

fig, ax = plt.subplots()
R = float(sim.L) * 1.1
ax.set_xlim(-R, R)
ax.set_ylim(-R, R)
ax.set_aspect("equal", adjustable="box")
ax.grid(True)

(rod,) = ax.plot([], [], "k-", lw=2)
(bob,) = ax.plot([], [], "bo", markersize=8)

def init():
    rod.set_data([], [])
    bob.set_data([], [])
    return (rod, bob)

epi_return = 0.0
def update(frame):
    global s, epi_return
    a = np.squeeze(np.array(agent.actor(jnp.asarray(s[None, :]))), axis=0)
    ns, r, _, _ = env.step(a)
    epi_return += r

    x = float(sim.L) * np.sin(sim.state_scale * float(s[0]))
    y = -float(sim.L) * np.cos(sim.state_scale * float(s[0]))

    rod.set_data([0.0, x], [0.0, y])
    bob.set_data([x], [y])
    print(epi_return)

    s = ns
    return (rod, bob)

ani = FuncAnimation(fig, update, frames=100, init_func=init, interval=50, blit=True)
ani.save("pendulum_rollout.gif", fps=20)

# %%
