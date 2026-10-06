import pybamm
import numpy as np
from time import time
import batpint.parameter.javid as bpar
from batpint.problems.battery import batteryExperiment
from batpint.parareal.pybamm_propagator import PybammPropagator
from batpint.parareal.parareal import PararealModified
import matplotlib.pyplot as plt

# Script parameters
half_cell = True
exp = "CCCV" # GITT or CCCV
nCycles = 1
num_cycles = 200

# Run script
args = ({"working electrode": "positive"},) if half_cell else ()
model_last = pybamm.lithium_ion.DFN(*args)
model_new = pybamm.lithium_ion.DFN(*args)
# Space discretization
var_pts = {
    "x_n": 1,   # points in the negative electrode
    "x_s": 30,  # points in the separator
    "x_p": 30,  # points in the positive electrode
    "r_p": 30  # points in the radius of positive electrode
}

# Setup experiment
parameter_values=pybamm.ParameterValues(
    bpar.Li_half.PARAMS if half_cell else bpar.Li_full.PARAMS)

# construct the simulation for the stepwise solving from (cycle to cycle)
experiment_last = batteryExperiment(nCycles, expType=exp)
experiment_new = batteryExperiment(nCycles,expType=exp)
solver = pybamm.IDAKLUSolver()
sim_new = pybamm.Simulation(model_new,
                            experiment=experiment_new,
                            parameter_values=parameter_values,
                            var_pts=var_pts,
                            solver=solver)

sim_last = pybamm.Simulation(model_last,
                            experiment=experiment_last,
                            parameter_values=parameter_values,
                            var_pts=var_pts,
                            solver=solver)


# solve with pybamm propagator
# no pybamm time correction
init_sol = None
solution_from_last = []
solution_from_new = []

t_last = []
t_new = []

t_delta_last = 0.0
t_delta_new = 0.0
dummy_sol = None
for cycle in range(num_cycles):
    
    sol_last = sim_last.solve(starting_solution=init_sol)

    cycle_sol = sol_last.cycles[-1]
    t_global_last = cycle_sol.t + t_delta_last
    t_last.append(t_global_last)
    t_delta_last = t_global_last[-1]
    solution_from_last.append(cycle_sol)
    
    
    init_sol = cycle_sol.last_state.copy()
    # change the times
    init_sol.t[-1] = 0
    init_sol.t_eval[-1] = 0
    init_sol.all_ts[-1][-1] = 0
    init_sol.all_t_evals[-1][-1] = 0

start_sol = solution_from_last[0].first_state.copy()
dummy_sol = pybamm.Solution(start_sol.all_ts, 
                            start_sol.all_ys,
                            start_sol.all_models,
                            start_sol.all_inputs,
                            all_yps=start_sol.all_yps,
                            t_event=start_sol.t_event,
                            y_event=start_sol.y_event,
                            options=start_sol.options,
                            all_t_evals=start_sol.all_t_evals
                            )


for cycle in range(num_cycles):
    # change the states 
    sol_new = sim_new.solve(starting_solution=dummy_sol)
    solution_from_new.append(sol_new.cycles[-1])
    sol = sol_new.cycles[-1].last_state.copy()
    t_global_new = sol_new.cycles[-1].t + t_delta_new
    t_new.append(t_global_new)
    t_delta_new = t_global_new[-1]
    
    dummy_sol = pybamm.Solution(start_sol.all_ts, 
                                sol.all_ys,
                                sol.all_models,
                                sol.all_inputs,
                                all_yps=sol.all_yps,
                                t_event=start_sol.t_event,
                                y_event=sol.y_event,
                                options=sol.options,
                                all_t_evals=start_sol.all_t_evals
                                )

#%%    

# check the time periods
# lengths
err_t = [np.abs(t1-t2) for t1, t2 in zip(t_last, t_new)]
err_t_per_cycle = [np.max(t) for t in err_t]
err_tStart_per_cycle = [t[0] for t in err_t]
err_tEnd_per_cycle = [t[-1] for t in err_t]

err_y = [np.linalg.norm(sol1.y-sol2.y, ord=np.inf,axis = 0) for sol1, sol2 in zip(solution_from_last, solution_from_new)]
err_yStart_per_cycle = [y[0] for y in err_y]
err_yEnd_per_cycle = [y[-1] for y in err_y]
err_yp = [np.linalg.norm(sol1.yp-sol2.yp, ord=np.inf,axis = 0) for sol1, sol2 in zip(solution_from_last, solution_from_new)]
err_ypStart_per_cycle = [yp[0] for yp in err_yp]
err_ypEnd_per_cycle = [yp[-1] for yp in err_yp]

#PB_propagator = PybammPropagator(model_step, experiment_step, parameter_values, var_pts)
#%%
#tStart = PB_propagator._t_start
#uStart = PB_propagator._u_start
#upStart = PB_propagator._up_start
#UStart = np.vstack((uStart,upStart))

# solve for one cycle
#T1, U1 = PB_propagator.propagate(t = tStart, u=UStart, cycle=0)
