import numpy as np
import matplotlib.pyplot as plt
from batpint.timestepping import timestepper
from batpint.timestepping.timestepper import TimeStepper, TimeStepperPropagator
from batpint.problems.dahlquist import Dahlquist, DahlquistBE, DahlquistExact

# Dahlquist test problem parameters
t_start = 0
u_start = 1+0j
pStart = 1
dtNum = 0.0001
dtEx = 0.0001

# define lam and alpha
def lam(cycle):
    return 1j*(1 + 0.01*cycle)
alpha = 6.001   # to avoid resonance regime

# define event and termination functions
def event(t, u, u_event):
    return np.imag(u) - np.imag(u_event)

def terminate(t, u, alpha, lam, cycle):
    return abs(alpha**2 + lam(cycle)**2) < 1e-10

# Create Dahlquist problem and Backward Euler method
problem = Dahlquist(t_start=t_start, u_start=u_start, lam=lam, alpha=alpha, cycle=pStart,u_event=u_start, event=event, terminate=terminate)
def reset():
    problem.params['cycle'] = pStart
    problem.params['t_start'] = t_start
    problem.params['u_start'] = u_start

methodNum = DahlquistBE(problem.params)
methodEx = DahlquistExact(problem.params)

# solve the problem
def solve_dahlquist(dt, num_events, method, save_history=True):
    """
    solve_dahlquist solves the Dahlquist problem using the specified time stepper method.
    Args:
        dt (float): time step size
        num_events (int): number of events to simulate
        method (TimeStepper): time stepping method to use
        save_history (bool, optional): whether to save the history of the time stepping. Defaults to True.

    Returns:
        tuple: (t_all, u_all, t_ev, u_ev) where t_all and u_all are the time and solution arrays for the entire simulation, 
        and t_ev and u_ev are the time and solution arrays at the event points.
    """
    reset()

    timestepper = TimeStepper(problem=problem, method=method, dt=dt, save_history=save_history)
    propagator = TimeStepperPropagator(timestepper=timestepper, direction=1)

    t_ev = np.zeros(num_events+1)
    t_ev[0] = problem.t_start
    u_ev = np.zeros(num_events+1, dtype=complex)
    u_ev[0] = problem.u_start
    
    t_all = [problem.t_start]
    u_all = [problem.u_start]
    
    for ev in range(num_events):
        if problem.termination_value(t_ev[ev], u_ev[ev]):
            break
        
        t_new, u_new = propagator.propagate(t_ev[ev], u_ev[ev])
        t_ev[ev + 1] = t_new
        u_ev[ev + 1] = u_new
        
        # extract solution for the current cycle
        t_local = np.asarray(propagator.history['t'])
        u_local = np.asarray(propagator.history['u'])
        
        # save the solution for the current cycle
        t_all.extend(t_local[1:])
        u_all.extend(u_local[1:])
        
        problem.params['cycle'] += 1
        problem.params['t_event'] = t_new
        problem.params['u_event'] = u_new
        
    return np.asarray(t_all), np.asarray(u_all), np.asarray(t_ev), np.asarray(u_ev)

num_events = 1 # number of cycles
    
tNum, uNum, t_evNum, u_evNum = solve_dahlquist(dt=dtNum, num_events=num_events, method=methodNum, save_history=True)
tEx, uEx, t_evEx, u_evEx = solve_dahlquist(dt=dtEx, num_events=num_events, method=methodEx, save_history=True)

#%%
    
