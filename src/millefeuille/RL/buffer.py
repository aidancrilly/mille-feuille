from typing import Dict

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.size = 0
        self.ptr = 0
        self.rng = np.random.default_rng(0)

    def add(self, s: np.ndarray, a: np.ndarray, r: float, ns: np.ndarray, done: bool) -> None:
        i = self.ptr
        self.states[i] = s
        self.actions[i] = a
        self.rewards[i, 0] = r
        self.next_states[i] = ns
        self.dones[i, 0] = 1.0 if done else 0.0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        batch_size = int(batch_size)
        idx = self.rng.integers(0, self.size, size=(batch_size,))
        return {
            "s": self.states[idx],
            "a": self.actions[idx],
            "r": self.rewards[idx],
            "ns": self.next_states[idx],
            "d": self.dones[idx],
        }
