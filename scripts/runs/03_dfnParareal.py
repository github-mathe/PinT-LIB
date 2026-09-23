import pybamm
import numpy as np
from time import time
from pprint import pp as pprint
from batpint import plt
import batpint.parameter.javid as bpar
from batpint.problems.battery import batteryExperiment
from batpint.parareal.pybamm_propagator import PybammPropagator

# Script parameters
half_cell = True
exp = "CCCV" # GITT or CCCV
nCycles = 1
num_cycles = 10

# Run script
args = ({"working electrode": "positive"},) if half_cell else ()
model = pybamm.lithium_ion.DFN(*args)

# Space discretization
var_pts = {
    "x_n": 1,   # points in the negative electrode
    "x_s": 30,  # points in the separator
    "x_p": 30,  # points in the positive electrode
    "r_p": 100  # points in the radius of positive electrode
}

# Setup experiment
parameter_values=pybamm.ParameterValues(
    bpar.Li_half.PARAMS if half_cell else bpar.Li_full.PARAMS)

# construct the simulation for the stepwise solving from (cycle to cycle)
experiment = batteryExperiment(nCycles, expType=exp)
solver = pybamm.IDAKLUSolver()

propagator = PybammPropagator(model, experiment, parameter_values, var_pts)


#%%
# solve
solutions = {}
start_sol = None
for cycle in range(num_cycles):
    solution, final_state = propagator.propagate(starting_state=start_sol)
    solutions[cycle] = solution
    if cycle>0:
        y0_new_state = start_sol.y
        y0_actual = solution.first_state.y
        err_y0 = np.max(np.abs(y0_new_state-y0_actual))
        print("Change in initialization: ", err_y0)
    start_sol = final_state


#%%
# construct the simulation for the sequential solving for all cycles in one solve
experiment_seq = batteryExperiment(num_cycles, expType=exp)
model_seq = pybamm.lithium_ion.DFN(*args)
sim_seq = pybamm.Simulation(
    model_seq,
    experiment=experiment_seq,
    parameter_values=parameter_values,
    solver=solver,
    var_pts=var_pts,)
sol_seq = sim_seq.solve()

#%%
# Check if the computed solutions are identical
last_states = {}
for cycle in range(num_cycles):
    sol_step = solutions[cycle]
    
    last_states[cycle] = sol_step.last_state
    sol = sol_seq.cycles[cycle]
    print(f"Cycle: {cycle}","tStart_diff: ", sol_step.t[0]-sol.t[0],"tEnd_diff", sol_step.t[-1] - sol.t[-1])
    t_min = np.maximum(sol_step.t[0], sol.t[0])
    t_max = np.minimum(sol_step.t[-1], sol.t[-1])
    t = np.linspace(t_min, t_max,1000)
    
    err_V = np.max(np.abs(sol_step["Voltage [V]"](t) - sol["Voltage [V]"](t)))
    print(f"Cycle {cycle}: Error in voltage values: ", err_V)
    
    y_step = np.asarray(sol_step.last_state.y).reshape(-1)
    y_seq = np.asarray(sol.last_state.y).reshape(-1)

    err_y = np.max(np.abs(y_step - y_seq))
    print(f"Cycle {cycle}: state error = {err_y}")
    
    





































