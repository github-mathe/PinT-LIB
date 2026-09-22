import pybamm
import numpy as np
from time import time
from pprint import pp as pprint
from batpint import plt
import batpint.parameter.javid as bpar
from batpint.problems.battery import batteryProblem

# Script parameters
half_cell = True
exp = "CCCV" # GITT or CCCV
nCycles = 1
showTimeSteps = True

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

experiment = batteryProblem(nCycles, expType=exp)
solver = pybamm.IDAKLUSolver()
sim = pybamm.Simulation(
    model,
    experiment=experiment,
    parameter_values=parameter_values,
    solver=solver,
    var_pts=var_pts,)

sim.build_for_experiment()

# stepwise implementation
solutions = {}
num_cycles = 10
for cycle in range(num_cycles):
    if cycle==0:
        sol = sim.solve()
    else:
        sol = sim.solve(starting_solution=solutions[cycle-1].cycles[-1].last_state)
    solutions[cycle] = sol
    print("Started at t = ", solutions[cycle].t[0])
    print("Terminated at t = ", solutions[cycle].t[-1], "s with status", solutions[cycle].termination)

# full implementation
experiment_all = batteryProblem(num_cycles, expType=exp)
sim_all = pybamm.Simulation(
    model,
    experiment=experiment_all,
    parameter_values=parameter_values,
    solver=solver,
    var_pts=var_pts,)
sim_all.build_for_experiment()
sol_all = sim_all.solve()

#%%
for i, sol in enumerate(sol_all.cycles):
    print("Started at t = ", sol.t[0])
    print("Terminated at t = ", sol.t[-1], "s with status", sol.termination)

#%%
for cycle in range(num_cycles):
    sol_step = solutions[cycle]
    sol = sol_all.cycles[cycle]
    print("Cycle: ", cycle,"tStart_diff: ", sol_step.t[0]-sol.t[0],"tEnd_diff", sol_step.t[-1] - sol.t[-1])
    t = np.linspace(sol.t[0], sol.t[-1],1000)
    
    err_V = np.max(np.abs(sol_step["Voltage [V]"](t) - sol["Voltage [V]"](t)))
    print("Error in voltage values: ", err_V)
    
    y_step = np.asarray(sol_step.last_state.y).reshape(-1)
    y_seq = np.asarray(sol.last_state.y).reshape(-1)

    err_y = np.max(np.abs(y_step - y_seq))

    print(f"Cycle {i}: state error = {err_y}")
