from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np


@dataclass
class SimEnvConfig:
    state_dim: int
    action_dim: int
    action_low: Tuple[float]
    action_high: Tuple[float]
    horizon: int = 50
    gamma: float = 0.99
    # Reward transform: simulator objective Y -> reward
    reward_fn: Callable[[np.ndarray], float] = lambda Y: float(np.squeeze(Y))


class Env:
    """
    Environment interface:
      reset() -> state (np.ndarray shape [N])
      step(action) -> next_state, reward, done, info

    Where reward is computed by calling an external simulator

    """

    def __init__(
        self,
        simulator,
        scheduler,
        cfg: SimEnvConfig,
        state_transition_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None,
        x_builder: Callable[[np.ndarray, np.ndarray, int], np.ndarray] | None = None,
        seed: int = 0,
    ):
        self.simulator = simulator
        self.scheduler = scheduler
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        self.t = 0
        self.run_index = 0

        # By default, p == s
        self.state_transition_fn = state_transition_fn or (lambda s, a, p: p.flatten())
        # By default, pass state, action and time as simulator inputs
        self.x_builder = x_builder or (lambda s, a, t: np.concatenate([s, a, np.array([t])]))

        self.state = np.zeros((cfg.state_dim,))

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        self.t = 0
        if state is None:
            self.state = np.zeros(self.cfg.state_dim)
        else:
            self.state = np.asarray(state).reshape((self.cfg.state_dim,))
        return self.state.copy()

    def step(self, action: np.typing.ArrayLike) -> Tuple[np.ndarray, float, bool, Dict]:
        action = np.clip(action, self.cfg.action_low, self.cfg.action_high)
        x = self.x_builder(self.state, action, self.t)

        idx = self.run_index
        self.run_index += 1

        # --- Coupling pattern matching the simulator ---
        if self.scheduler is None:
            P, Y = self.simulator(np.array([idx]), x.reshape(1, -1))
        else:
            P, Y = self.simulator(np.array([idx]), x.reshape(1, -1), self.scheduler)

        reward = self.cfg.reward_fn(Y)
        next_state = self.state_transition_fn(self.state, action, P)

        self.t += 1
        done = self.t >= self.cfg.horizon

        info = {"P": P, "Y": Y, "index": idx}
        self.state = next_state
        return self.state.copy(), reward, done, info
