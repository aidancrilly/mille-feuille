import numpy as np
from millefeuille.domain import InputDomain
from millefeuille.simulator import PythonSimulator

sampler = np.random.default_rng(seed=12345)


class ForresterFunction(PythonSimulator):
    def f(self, Xs):
        ys = (6 * Xs - 2) ** 2 * np.sin(12 * Xs + 4)
        return ys

    def ABC_values(self, Ss):
        if Ss is not None:
            A = 1.0 - (1 - Ss) * 0.5
            B = (1 - Ss) * 10.0
            C = (1 - Ss) * 5.0
        else:
            A = 1.0
            B = 0.0
            C = 0.0
        return A, B, C

    def __call__(self, indices, Xs, Ss):
        A, B, C = self.ABC_values(Ss)
        return A * self.f(Xs) + B * (Xs - 0.5) + C


ForresterDomain = InputDomain(dim=1, b_low=np.array([0.0]), b_up=np.array([1.0]), steps=np.array([0.0]))
