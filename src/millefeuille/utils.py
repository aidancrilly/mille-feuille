"""
Defines some useful utility functions which do not fit into the defined classes
"""
import numpy as np

from .simulator import *
from .optimise import *

def singlefidelity_serial_BO_run(Nsamples,acq_function,state,surrogate,simulator,scheduler=None):
    if(isinstance(simulator,ExectuableSimulator) and scheduler is None):
        print('If simulator is an ExecutableSimulator, you must provide a scheduler')
        raise Exception

    batch_size = 1
    for _ in range(Nsamples):
        X_next = suggest_next_locations(batch_size,state,surrogate,
        acq_function=acq_function)

        index_next = np.array([state.index[-1]+1])
        if(isinstance(simulator,ExectuableSimulator)):
            P_next, Y_next = simulator(index_next, X_next, scheduler)
        elif(isinstance(simulator,PythonSimulator)):
            P_next, Y_next = simulator(index_next, X_next)
        else:
            print('simulator class not recognised, inherit for mille-feuille Simulator classes...')

        state.update(index_next,X_next=X_next,Y_next=Y_next,P_next=P_next)

    return state

def multifidelity_serial_BO_run(Nsamples,acq_function,cost_model,state,surrogate,simulator,scheduler=None):
    if(isinstance(simulator,ExectuableSimulator) and scheduler is None):
        print('If simulator is an ExecutableSimulator, you must provide a scheduler')
        raise Exception

    batch_size = 1
    for _ in range(Nsamples):
        X_next,S_next = suggest_next_locations(batch_size,state,surrogate,
        acq_function=acq_function,
        cost_model=cost_model)

        index_next = np.array([state.index[-1]+1])
        if(isinstance(simulator,ExectuableSimulator)):
            P_next, Y_next = simulator(index_next, X_next, scheduler, Ss = S_next)
        elif(isinstance(simulator,PythonSimulator)):
            P_next, Y_next = simulator(index_next, X_next, Ss = S_next)
        else:
            print('simulator class not recognised, inherit for mille-feuille Simulator classes...')

        state.update(index_next,X_next=X_next,Y_next=Y_next,P_next=P_next,S_next=S_next)

    return state
