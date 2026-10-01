import pybamm
import numpy as np
from time import time
import batpint.parameter.javid as bpar
from batpint.problems.battery import batteryExperiment
from batpint.parareal.pybamm_propagator import PybammPropagator
from batpint.parareal.parareal import PararealModified
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

# Script parameters
half_cell = True
exp = "CCCV" # GITT or CCCV
nCycles = 1
num_cycles = 100

# Run script
args = ({"working electrode": "positive"},) if half_cell else ()

# Space discretization
var_pts = {
    "x_n": 1,   # points in the negative electrode
    "x_s": 30,  # points in the separator
    "x_p": 30,  # points in the positive electrode
    "r_p": 30  # points in the radius of positive electrode
}

# parameters
parameter_values=pybamm.ParameterValues(
    bpar.Li_half.PARAMS if half_cell else bpar.Li_full.PARAMS)
# solver
solver = pybamm.IDAKLUSolver()

# models
model_step = pybamm.lithium_ion.DFN(*args)
model_all = pybamm.lithium_ion.DFN(*args)

# experiments
experiment_step = batteryExperiment(nCycles, expType=exp)
experiment_all = batteryExperiment(num_cycles,expType=exp)

# simulations
sim_all = pybamm.Simulation(model_all,
                            experiment=experiment_all,
                            parameter_values=parameter_values,
                            var_pts=var_pts,
                            solver=solver)
sim_step = pybamm.Simulation(model_step,
                            experiment=experiment_step,
                            parameter_values=parameter_values,
                            var_pts=var_pts,
                            solver=pybamm.IDAKLUSolver())

#solve for all cycles
solution_all = sim_all.solve()

# solve stepwise
# no pybamm time correction
init_sol = None
solution_step = []
t_step = []
t_delta = 0.0
for cycle in range(num_cycles):
    sol = sim_step.solve(starting_solution=init_sol)
    cycle_sol = sol.cycles[-1]
    t_global = cycle_sol.t + t_delta
    t_step.append(t_global)
    t_delta = t_global[-1]
    solution_step.append(cycle_sol)
    init_sol = cycle_sol.last_state.copy()
    init_sol.t[-1] = 0
    init_sol.all_ts[-1][-1] = 0
    print("cycle_sol changed:", cycle_sol.t[-1] != t_global[-1])
    
print(f"Solved {num_cycles} cycles.")
#%%
# check the size of the states
# processed models
built_model_all = solution_all.cycles[0].first_state.all_models[0]
built_model_step = solution_step[0].first_state.all_models[0]
len_t_all = [sol.t.shape[0] for sol in solution_all.cycles]
len_t_step = [t.shape[0] for t in t_step]
print("Same state vector size: ", built_model_all.len_rhs_and_alg== built_model_step.len_rhs_and_alg)
print("Same differential state vector size: ", built_model_all.len_rhs== built_model_step.len_rhs)
print("Same algebraic state vector size: ", built_model_all.len_alg== built_model_step.len_alg)
print("Same time lengths: ", len_t_all == len_t_step)
# data at the time interface = time at the switch between cycles

#%%
# interface times
t_start_all = [sol.t[0] for sol in solution_all.cycles]
t_last_all = [sol.t[-1] for sol in solution_all.cycles]
t_switch_all = [t1-t2 for t1,t2 in zip(t_start_all[1:],t_last_all[:-1])]

t_last_step = [t[-1] for t in t_step]
t_start_step = [t[0] for t in t_step]
t_switch_step = [t1-t2 for t1,t2 in zip(t_start_step[1:], t_last_step[:-1])]

# difference in time at the time interface
err_t_last = [t1-t2 for t1,t2 in zip(t_last_all,t_last_step)]
err_t_start = [t1-t2 for t1,t2 in zip(t_start_all,t_start_step)]

#interface states

y_all = [sol.y for sol in solution_all.cycles]
y_last_all = [sol.y[:,-1] for sol in solution_all.cycles]
y_start_all = [sol.y[:,0] for sol in solution_all.cycles]
y_switch_all = [y1-y2 for y1,y2 in zip(y_last_all[:-1], y_start_all[1:])]
err_y_switch_all = [np.linalg.norm(y_switch, ord=np.inf,axis = 0) for y_switch in y_switch_all]

y_diff_last_all = [sol.y[:built_model_all.len_rhs,-1] for sol in solution_all.cycles]
y_diff_start_all = [sol.y[:built_model_all.len_rhs,0] for sol in solution_all.cycles]
y_diff_switch_all = [y1-y2 for y1,y2 in zip(y_diff_last_all[:-1],y_diff_start_all[1:])]
err_y_diff_switch_all = [np.linalg.norm(y_switch, ord=np.inf,axis = 0) for y_switch in y_diff_switch_all]

y_alg_last_all = [sol.y[built_model_all.len_rhs:,-1] for sol in solution_all.cycles]
y_alg_start_all = [sol.y[built_model_all.len_rhs:,0] for sol in solution_all.cycles]
y_alg_switch_all = [y1-y2 for y1,y2 in zip(y_alg_last_all[:-1],y_alg_start_all[1:])]
err_y_alg_switch_all = [np.linalg.norm(y_switch, ord=np.inf,axis = 0) for y_switch in y_alg_switch_all]


y_step = [sol.y for sol in solution_step]
y_last_step = [sol.y[:,-1] for sol in solution_step]
y_start_step = [sol.y[:,0] for sol in solution_step]
y_switch_step = [y1-y2 for y1,y2 in zip(y_last_step[:-1],y_start_step[1:])]
err_y_switch_step = [np.linalg.norm(y_switch, ord=np.inf,axis = 0) for y_switch in y_switch_step]


y_diff_last_step = [sol.y[:built_model_step.len_rhs,-1] for sol in solution_step]
y_diff_start_step = [sol.y[:built_model_step.len_rhs,0] for sol in solution_step]
y_diff_switch_step = [y1-y2 for y1,y2 in zip(y_diff_last_step[:-1],y_diff_start_step[1:])]
err_y_diff_switch_step = [np.linalg.norm(y_switch, ord=np.inf,axis = 0) for y_switch in y_diff_switch_step]


y_alg_last_step =  [sol.y[built_model_step.len_rhs:,-1] for sol in solution_step]
y_alg_start_step = [sol.y[built_model_step.len_rhs:,0] for sol in solution_step]
y_alg_switch_step = [y1-y2 for y1,y2 in zip(y_alg_last_step[:-1],y_alg_start_step[1:])]
err_y_alg_switch_step = [np.linalg.norm(y_switch, ord=np.inf,axis = 0) for y_switch in y_alg_switch_step]

# L_Inf error of the state vectors at all times 
err_y = [np.linalg.norm(y1-y2, ord=np.inf, axis = 0) for y1,y2 in zip(y_all,y_step)]
err_y_diff = [np.linalg.norm(y1[:built_model_step.len_rhs,:]-y2[:built_model_step.len_rhs,:], ord=np.inf, axis = 0) for y1,y2 in zip(y_all,y_step)]
err_y_alg = [np.linalg.norm(y1[built_model_step.len_rhs:,:]-y2[built_model_step.len_rhs:,:], ord=np.inf, axis = 0) for y1,y2 in zip(y_all,y_step)]

#%%
print("\n" + "=" * 50)
print("CYCLE COMPARISON between sequential and step solutions")
print("=" * 50)


for cycle in range(num_cycles):
    print("\n" + "=" * 50)
    print("Cycle {cycle}")
    print("starting time difference:", err_t_start[cycle])
    print("ending time difference:", err_t_last[cycle])
    
    print("starting state difference:", err_y[cycle][0])
    print("starting differential state difference:", err_y_diff[cycle][0])
    print("starting algebraic state difference:", err_y_alg[cycle][0])
    
    print("ending state difference:", err_y[cycle][-1])
    print("ending differential state difference:", err_y_diff[cycle][-1])
    print("ending algebraic state difference:", err_y_alg[cycle][-1])
    
    #start_error = np.linalg.norm(y_start_step[cycle] - y_start_all[cycle], ord=np.inf)
    #end_error = np.linalg.norm(y_last_step[cycle] - y_last_all[cycle], ord=np.inf)
    #print(start_error == err_y[cycle][0], end_error == err_y[cycle][-1])
    print("=" * 50)  
    
#%%    
plt.figure()
plt.semilogy(t_switch_all, "-o", color="red")
plt.semilogy(t_switch_step, "-o")
plt.xlabel(r"$t_{\Gamma}$")
plt.ylabel(r"$t_+ - t_-$")

plt.figure()
plt.semilogy(err_y_diff_switch_all, "-o", color = "red")
plt.semilogy(err_y_diff_switch_step, "-o")
plt.xlabel(r"$t_{\Gamma}$ [s]")
plt.ylabel(r"$\|y_{+,d} - y_{-,d}\|_{\infty}$")

plt.figure()
plt.semilogy(err_y_alg_switch_all, "-o", color="red")
plt.semilogy(err_y_alg_switch_step, "-o")
plt.xlabel(r"$t_{\Gamma}$ [s]")
plt.ylabel(r"$\|y_{+,a} - y_{-,a}\|_{\infty}$")

plt.show()

#%%
# ------------------------------------------------------------
# Mesh points
# ------------------------------------------------------------
# Spatial coordinates
mesh = sim_step.mesh

r_p = mesh["positive particle"].nodes
x_p = mesh["positive electrode"].nodes
x_s = mesh["separator"].nodes
x = mesh["separator", "positive electrode"].nodes

results = {"all": solution_all.cycles[:], "step": solution_step[:]}

state_variable = lambda cycle, sol_type, state: results[sol_type][cycle][state]
c_s_p = lambda cycle, sol_type: state_variable(cycle,sol_type, "Positive particle concentration [mol.m-3]")
eps_c_e = lambda cycle, sol_type: state_variable(cycle,sol_type, "Porosity times concentration [mol.m-3]")
ce = lambda cycle, sol_type: state_variable(cycle,sol_type, "ELectrolyte concentration [mol.m-3]")

phi_s_p = lambda cycle, sol_type: state_variable(cycle,sol_type,"Positive electrode potential [V]")
phi_e = lambda cycle, sol_type: state_variable(cycle,sol_type,"Electrolyte potentiial [V]")
Q_abs = lambda cycle, sol_type: state_variable(cycle,sol_type,"Throughput capacity [A.h]")
Q_disch = lambda cycle, sol_type: state_variable(cycle,sol_type,"Discharge capacity [A.h]")

t0 = 1000
c_s_profile = c_s_p(0,"all")(t=t0, x=x0, r=r_p)
plt.plot(r_p, c_s_profile, label=fr"$x={x0:.3e}$ m")