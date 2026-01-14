import millefeuille.RL as RL 
import numpy as np
import jax

from scipy.integrate import odeint
from millefeuille.simulator import PythonSimulator

class Pendulum(PythonSimulator):

    def __init__(self, L, dt):
        self.dt = dt
        self.g = 9.8
        self.L = L

    def dydt(self, x, t, a):
        dthetadt = x[1]
        dthetadotdt = -self.g/self.L*np.sin(x[0]) + a[0]
        return np.array([dthetadt,dthetadotdt])

    def solve(self, Xs, As):
        Xs_next = []
        for i in range(Xs.shape[0]):
            Xs_next.append(odeint(self.dydt, Xs[i], [0.0, self.dt], args=(As[i],))[-1,:])
        return np.array(Xs_next).reshape(Xs.shape)

    def __call__(self, indices, Xs, Ss = None):
        # Xs is state, action, time
        x = Xs[:,:2]
        a = Xs[:,2:-1]
        x_next = self.solve(x,a)
        Ps = x_next
        cos_theta = np.cos(Ps[:,0])
        exp_arg = (cos_theta+1)**2
        Ys = 2*np.exp(-exp_arg) - 1
        return Ps, Ys
    
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

trained = RL.train(env, agent, total_steps=10_000, warmup_steps=500, updates_per_step=10)
print("done")