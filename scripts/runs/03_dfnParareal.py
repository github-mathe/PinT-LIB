import pybamm
import numpy as np
from time import time
import batpint.parameter.javid as bpar
from batpint.problems.battery import batteryExperiment
from batpint.parareal.pybamm_propagator import PybammPropagator
from batpint.parareal.parareal import PararealModified


# Script parameters
half_cell = True
exp = "CCCV" # GITT or CCCV
nCycles = 1
num_cycles = 2

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
sim_step = pybamm.Simulation(
    model,
    experiment=experiment,
    parameter_values=parameter_values,
    solver=solver,
    var_pts=var_pts,
    experiment_model_mode="unified")
#propagator = PybammPropagator(model, experiment, parameter_values, var_pts)


#%%
# solve
# check the starting states 
solutions = {}
start_sol = None
for cycle in range(num_cycles):
    solution = sim_step.solve(starting_solution=start_sol)
    #t,u,solution = propagator.propagate_with_solution(starting_state=start_sol)
    solutions[cycle] = solution.cycles[-1]
    if cycle>0:
        y0_new_state = start_sol.y
        y0_actual = solution.first_state.y
        err_y0 = np.max(np.abs(y0_new_state-y0_actual))
        print("Change in initialization: ", err_y0)
    start_sol = solution.last_state
    
#%%
# construct the simulation for the sequential solving for all cycles in one solve
experiment_seq = batteryExperiment(num_cycles, expType=exp)
model_seq = pybamm.lithium_ion.DFN(*args)
solver_seq = pybamm.IDAKLUSolver() 
sim_seq = pybamm.Simulation(
    model_seq,
    experiment=experiment_seq,
    parameter_values=parameter_values,
    solver=solver_seq,
    var_pts=var_pts,
    experiment_model_mode="unified")
sol_seq = sim_seq.solve()
tEnd_cycle = [sol.last_state.t for sol in sol_seq.cycles]
tStart_cycle = [sol.first_state.t for sol in sol_seq.cycles]

uEnd_cycle = [np.asarray(sol.last_state.y).reshape(-1) for sol in sol_seq.cycles]
uStart_cycle = [np.asarray(sol.first_state.y).reshape(-1) for sol in sol_seq.cycles]


#%%
# Check if the computed solutions are identical
last_states = {}
for cycle in range(num_cycles):
    sol_step = solutions[cycle]
    
    last_states[cycle] = sol_step.last_state
    sol = sol_seq.cycles[cycle]
    print(f"Cycle: {cycle}","tStart_diff: ", sol_step.t[0]-sol.t[0],\
          "tEnd_diff", sol_step.t[-1] - sol.t[-1])
    t_min = np.maximum(sol_step.t[0], sol.t[0])
    t_max = np.minimum(sol_step.t[-1], sol.t[-1])
    t = np.linspace(t_min, t_max,1000)
    
    err_V = np.max(np.abs(sol_step["Voltage [V]"](t) - sol["Voltage [V]"](t)))
    print(f"Cycle {cycle}: Error in voltage values: ", err_V)
    
    y_step = np.asarray(sol_step.last_state.y).reshape(-1)
    y_seq = np.asarray(sol.last_state.y).reshape(-1)

    err_y = np.max(np.abs(y_step - y_seq))
    print(f"Cycle {cycle}: state error = {err_y}")
    
#%%

# Parareal

N = num_cycles
K = N*1

t_start = solutions[0].first_state.t
u_start = np.asarray(solutions[0].first_state.y).reshape(-1).copy()

propagatorF = PybammPropagator(model, experiment, parameter_values, var_pts,experiment_model_mode="unified")
propagatorG = PybammPropagator(model, experiment, parameter_values, var_pts, experiment_model_mode="unified")


make_state= lambda t,u,n: propagatorF.make_state(t,u)
parareal = PararealModified(propagatorF, propagatorG, make_state=make_state)

try:
    TT, U = parareal.solve(t0=t_start, u0=u_start, K=K, N=N)
except Exception as e:
    print(f"Parareal execution failed: {e}")
#%%
err_TT = []
err_U = []
for k in range(1,K):
    err_TT.append(np.abs(TT[k,1:]- tEnd_cycle))
   # err_U.append(np.linalg.norm(U[k,1:]- uEnd_cycle))

    















