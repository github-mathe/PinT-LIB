# -*- coding: utf-8 -*-

import pybamm
import numpy as np
import batpint.parameter.javid as bpar
from batpint.problems.battery import batteryExperiment
from batpint.parareal.pybamm_propagator import PybammPropagator

# Script parameters
half_cell = True
exp = "CCCV" # GITT or CCCV
nCycles = 1
num_cycles = 50

# Space discretization
var_pts = {
    "x_n": 1,   # points in the negative electrode
    "x_s": 30,  # points in the separator
    "x_p": 30,  # points in the positive electrode
    "r_p": 30  # points in the radius of positive electrode
}

#%%
# Run script
def make_simulation():
    """Create an independent DFN simulation."""
    
    args = ({"working electrode": "positive"},) if half_cell else ()
    model = pybamm.lithium_ion.DFN(*args)
    parameter_values = pybamm.ParameterValues(bpar.Li_half.PARAMS if half_cell else bpar.Li_full.PARAMS)
    experiment = batteryExperiment(nCycles, expType=exp)
    solver = pybamm.IDAKLUSolver()

    return pybamm.Simulation(
        model,
        experiment=experiment,
        parameter_values=parameter_values,
        var_pts=var_pts,
        solver=solver,
    )

sim = make_simulation()

# solve with pybamm propagator
# no pybamm time correction

# starting solutions are none for the first cycle
init_sol = None

# store cycle solutions
solution = []
solution_propagator=[]
# store global times per cycle
t_all = []
t_propagator = [0]
# time diff between cycles 
t_delta = 0.0

# %%

pybamm_propagator = PybammPropagator(
    model = pybamm.lithium_ion.DFN(*({"working electrode": "positive"},) if half_cell else ()),
    experiment = batteryExperiment(nCycles, expType=exp),
    parameter_values = pybamm.ParameterValues(bpar.Li_half.PARAMS if half_cell else bpar.Li_full.PARAMS),
    var_pts = var_pts)
uStart = pybamm_propagator.u_full
solution_propagator.append(uStart)

# compute
for cycle in range(num_cycles):
    
    tNext, sol_propagator = pybamm_propagator.propagate(
        t = t_propagator[cycle],
        u = solution_propagator[cycle],
        cycle = cycle)
    t_propagator.append(tNext)
    solution_propagator.append(sol_propagator)


    sol = sim.solve(starting_solution=init_sol)
    cycle_sol = sol.cycles[-1]
    solution.append(cycle_sol)
    sol_end = cycle_sol.last_state.copy()
    # global times of the cycle
    t_global = cycle_sol.t + t_delta
    t_all.append(t_global)
    t_delta = t_global[-1]
    # create a solution object 
    # with times 0 and states from the last state of the present solution cycle
    init_sol = pybamm.Solution(
        all_ts=np.array([0.0]), 
        all_ys=np.array(sol_end.all_ys[-1],copy=True),
        all_models=sol_end.all_models[-1],
        all_inputs=sol_end.all_inputs[-1],
        all_yps=np.array(sol_end.all_yps[-1],copy=True) if sol_end.all_yps is not None else None,
        t_event=None,
        y_event=None,
        options=sol_end.user_options,
        all_t_evals=None)

#%%    
# check cycle end times 
t_all_ends = [t[-1] for t in t_all]
diff_t_ends = [t1-t2 for t1,t2 in zip(t_all_ends, t_propagator[1:])]
not_zero_t = np.nonzero(diff_t_ends)
print("Difference in end times of cycles: ", not_zero_t)
#check end states
u_all_ends = [sol.all_ys[-1][:,-1] for sol in solution]
up_all_ends = [sol.all_yps[-1][:,-1] for sol in solution]
u_propagator = [u[0] for u in solution_propagator]
up_propagator = [up[1] for up in solution_propagator]

diff_u_ends = [np.linalg.norm(u1-u2, ord=np.inf) for u1,u2 in zip(u_all_ends,u_propagator[1:])]
diff_up_ends = [np.linalg.norm(u1-u2, ord=np.inf) for u1,u2 in zip(up_all_ends,up_propagator[1:])]
not_zero_u = np.nonzero(diff_u_ends)
not_zero_up = np.nonzero(diff_up_ends)
print("Difference in end states, derivatives of states: ", not_zero_u, not_zero_up)