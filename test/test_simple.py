import numpy as np
import sys

from typing import Optional
import numpy.typing as npt

sys.path.append('../src/')
import millefeuille as mf

def ForresterFunction(x,s):
    # Modified to be over [0,2]
    # Two fidelities s, 0 and 1

    As = np.ones_like(s)
    Bs = np.zeros_like(s)
    Cs = np.zeros_like(s)

    As[s == 0] = 0.5
    Bs[s == 0] = 5.0
    Cs[s == 0] = 5.0
    
    return -(As*(3*x-2)**2*np.sin(6*x-4)+Bs*(x-1)+Cs)

def SingleFidelityForresterFunction(x):
    return ForresterFunction(x,np.ones_like(x))

# Wrap in mille-feuille simulator class
class ForresterSimulator(mf.PythonSimulator):

    def __call__(self, indices : npt.NDArray, Xs : npt.NDArray, Ss : None | npt.NDArray = None) -> tuple[Optional[npt.NDArray], npt.NDArray]:
        if(Ss is None):
            return None, SingleFidelityForresterFunction(Xs)
        else:
            return None, ForresterFunction(Xs,Ss)

input_domain = mf.InputDomain(1,np.zeros(1),2*np.ones(1),np.zeros(1))
simulator = ForresterSimulator()

# Single fidelity case
N_init = 4
X_init = np.random.uniform(low=0.0,high=2.0,size=(N_init,1))
Y_init = SingleFidelityForresterFunction(X_init)
state  = mf.State(input_domain,np.arange(N_init),X_init,Y_init)
surrogate = mf.SingleFidelityGPSurrogate()
acq_function = 'qLogExpectedImprovement'

Nsamples = 5
state = mf.singlefidelity_serial_BO_run(Nsamples,acq_function,state,surrogate,simulator)
print(state.Xs,state.Ys)

# Multi-fidelity case
fidelity_domain = mf.FidelityDomain(2,costs=[0.5,1.0])

S_init = np.random.randint(low=0,high=fidelity_domain.num_fidelities-1,size=(N_init,1))
Y_init = ForresterFunction(X_init,S_init)
state  = mf.State(input_domain,np.arange(N_init),X_init,Y_init,
                  Ss=S_init,fidelity_domain=fidelity_domain)

surrogate = mf.MultiFidelityGPSurrogate()
acq_function = mf.generate_MFKG_acqf
cost_model = mf.generate_multifidelity_cost_model(fidelity_domain.costs)

Nsamples = 5
state = mf.multifidelity_serial_BO_run(Nsamples,acq_function,cost_model,state,surrogate,simulator)
print(state.Xs,state.Ys,state.Ss)
