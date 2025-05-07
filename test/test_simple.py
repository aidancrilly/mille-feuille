import numpy as np
import sys
sys.path.append('../src/')

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

import millefeuille as mf

# Single fidelity case
domain = mf.Domain(1,np.zeros(1),2*np.ones(1),np.zeros(1))

N_init = 4
X_init = np.random.uniform(low=0.0,high=2.0,size=(N_init,1))
Y_init = SingleFidelityForresterFunction(X_init)
state  = mf.State(domain,None,X_init,Y_init)
surrogate = mf.SingleFidelityGPSurrogate()

Nsample = 5
print('Single fidelity updates')
for _ in range(Nsample):
    X_next = mf.suggest_next_locations(1,state,surrogate,
    acq_function='qLogExpectedImprovement')
    Y_next = SingleFidelityForresterFunction(X_next)
    print(X_next,Y_next)
    state.update(X_next,Y_next)

# Multi-fidelity case
S_init = np.random.randint(low=0,high=1,size=(N_init,1))
Y_init = ForresterFunction(X_init,S_init)
state  = mf.State(domain,None,X_init,Y_init,S_init,costs=[0.5,1.0])
surrogate = mf.MultiFidelityGPSurrogate()

Nsample = 5
print('Multi fidelity updates')
for _ in range(Nsample):
    X_next,S_next = mf.suggest_next_locations(1,state,surrogate,
    acq_function=mf.generate_MFKG_acqf,
    cost_model=mf.generate_multifidelity_cost_model(state.costs))
    Y_next = ForresterFunction(X_next,S_next)
    print(X_next,Y_next,S_next)
    state.update(X_next,Y_next,S_next)

