import numpy as np
import sys

from typing import Optional
import numpy.typing as npt

sys.path.append('./src/')
import millefeuille as mf

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

import numpy as np

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP

from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

# Default GP hyperparameters
DEFAULT_NOISE_INTERVAL = [1e-8,1e-5]
DEFAULT_LENGTHSCALE_INTERVAL = [0.005,4.0]

from abc import ABC, abstractmethod
import numpy.typing as npt

class TestMultiFidelityGPSurrogate:

    def __init__(self):
        self.model = None
        self.likelihood = None

    def get_XY(self,state):
        X_torch,Y_torch = state.transform_XY()
        assert X_torch[:,:-1].min() >= 0.0 and X_torch[:,:-1].max() <= 1.0 and torch.all(torch.isfinite(Y_torch))

        return X_torch,Y_torch

    def fit(self,state,noise_interval=DEFAULT_NOISE_INTERVAL,lengthscale_interval=DEFAULT_LENGTHSCALE_INTERVAL,approx_mll=False):
        X_torch,Y_torch = self.get_XY(state)

        self.likelihood = GaussianLikelihood(noise_constraint=Interval(*noise_interval))

        self.model = mf.Hierarchical_MultiFidelity_MaternGP(
                X_torch, Y_torch, num_fidelities = 2, nu = 2.5, lengthscale_interval = lengthscale_interval, likelihood=self.likelihood
            )
        
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        # Fit the model
        fit_gpytorch_mll(mll, approx_mll=approx_mll)

    def predict(self, state, Xs):
        Xs_unit = state.transform_X(Xs)
        test_X = torch.tensor(Xs_unit, dtype=torch.double, device=device)
        with torch.no_grad():
            post = self.likelihood(self.model(test_X))
            mean = post.mean.cpu().numpy().reshape(-1, state.fidelity_domain.num_fidelities)
            var = post.variance.cpu().numpy().reshape(-1, state.fidelity_domain.num_fidelities)
            std = np.sqrt(var)
        mean, std = state.inverse_transform_Y(mean, std)
        return {'mean' : mean, 'std' : std}

def ForresterFunction(x,s):
    # Modified to be over [0,2]
    # Two fidelities s, 0 and 1

    As = np.ones_like(s,dtype=float)
    Bs = np.zeros_like(s,dtype=float)
    Cs = np.zeros_like(s,dtype=float)

    As[s == 0] = 0.5
    Bs[s == 0] = 5.0
    Cs[s == 0] = 5.0
    
    return -(As*(3*x-2)**2*np.sin(6*x-4)+Bs*(x-1)+Cs)

def SingleFidelityForresterFunction(x):
    return ForresterFunction(x,np.ones_like(x))

np.random.seed(88)
N_init = 100

# Multi-fidelity case
input_domain = mf.InputDomain(1,np.zeros(1),2*np.ones(1),np.zeros(1))
fidelity_domain = mf.FidelityDomain(2,costs=[0.5,1.0])

X_init = np.random.uniform(low=0.0,high=2.0,size=(N_init,1))
S_init = np.random.randint(low=0,high=fidelity_domain.num_fidelities,size=(N_init,1))

S_init = np.zeros_like(X_init,dtype=int)
S_init[-2:] = 1

Y_init = ForresterFunction(X_init,S_init)

state  = mf.State(input_domain,np.arange(N_init),X_init,Y_init,
                  Ss=S_init,fidelity_domain=fidelity_domain,
                  X_names=['L'],Y_names=['Height'])

state.to_csv('test.csv')

import matplotlib.pyplot as plt

X_test = np.linspace(0.0,2.0,100).reshape(-1,1)

plt.plot(X_test,ForresterFunction(X_test,np.zeros_like(X_test)),'b')
plt.plot(X_test,ForresterFunction(X_test,np.ones_like(X_test)),'r')


# surrogate = TestMultiFidelityGPSurrogate()

# surrogate.fit(state)

# Y_pred = surrogate.predict(state,X_test)

# plt.plot(X_test,Y_pred['mean'][:,0],'k')
# plt.plot(X_test,Y_pred['mean'][:,1],'g')

# surrogate = mf.MultiFidelityGPSurrogate()

# surrogate.fit(state)

# Y_pred = surrogate.predict(state,X_test)

# plt.plot(X_test,Y_pred['mean'][:,0],'k--')
# plt.plot(X_test,Y_pred['mean'][:,1],'g--')

surrogate = mf.MultiFidelityGPSurrogate()
surrogate.init(state)
surrogate.fit(state)

surrogate.save('test.pth')

Y_pred = surrogate.predict(state,X_test)

plt.plot(X_test,Y_pred[0]['mean'],'k:')
plt.plot(X_test,Y_pred[1]['mean'],'g:')

plt.fill_between(X_test.flatten(),Y_pred[0]['mean']-Y_pred[0]['std'],Y_pred[0]['mean']+Y_pred[0]['std'])
plt.fill_between(X_test.flatten(),Y_pred[1]['mean']-Y_pred[1]['std'],Y_pred[1]['mean']+Y_pred[1]['std'])

plt.scatter(X_init,Y_init,c=S_init)
print(surrogate.model.state_dict())
surrogate = mf.MultiFidelityGPSurrogate()
surrogate.init(state)
surrogate.load('test.pth')
print(surrogate.model.state_dict())
Y_pred = surrogate.predict(state,X_test)

surrogate.fit(state)

plt.plot(X_test,Y_pred[0]['mean'],'k--')
plt.plot(X_test,Y_pred[1]['mean'],'g--')

plt.show()