import numpy as np
import torch
from gpytorch.means import Mean
from millefeuille.domain import InputDomain
from millefeuille.simulator import PythonSimulator
from scipy.stats import qmc

TEST_NUM_RESTARTS = 1
TEST_RAW_SAMPLES = 32

_rng = np.random.default_rng(seed=12345)

ForresterDomain = InputDomain(dim=1, b_low=np.array([0.0]), b_up=np.array([1.0]), steps=np.array([0.0]))

ForresterSampler = qmc.LatinHypercube(ForresterDomain.dim, rng=_rng)


class PythonForresterFunction(PythonSimulator):
    """
    Multi-fidelity Forrestor function (negated for maximisation)

    """

    def f(self, Xs):
        ys = (6 * Xs - 2) ** 2 * np.sin(12 * Xs - 4)
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

    def __call__(self, indices, Xs, Ss=None):
        A, B, C = self.ABC_values(Ss)
        return None, -(A * self.f(Xs) + B * (Xs - 0.5) + C)


class LowFidelityForresterMean(Mean):
    """

    A mean function that returns the low fidelity estimator of the Forrestor

    """

    def __init__(self, output_transform=None):
        super().__init__()
        self.output_transform = output_transform
        self.register_parameter(name="raw_constant", parameter=torch.nn.Parameter(torch.zeros(1)))

    def f(self, Xs):
        ys = (6 * Xs - 2) ** 2 * torch.sin(12 * Xs - 4)
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

    def get_low_fid_ABC(self):
        return self.ABC_values(0.0)

    def forward(self, Xs):
        A, B, C = self.get_low_fid_ABC()
        Y = -(A * self.f(Xs) + B * (Xs - 0.5) + C) + self.raw_constant
        Y = self.output_transform.transform(Y).squeeze(-1) if self.output_transform is not None else Y.squeeze(-1)
        return Y


class Uniform(qmc.QMCEngine):
    def __init__(self, d, rng=None):
        super().__init__(d=d, seed=rng)

    def _random(self, n=1, *, workers=1):
        return self.rng.random((n, self.d))

    def reset(self):
        super().__init__(d=self.d, seed=self.rng_seed)
        return self

    def fast_forward(self, n):
        self.random(n)
        return self
