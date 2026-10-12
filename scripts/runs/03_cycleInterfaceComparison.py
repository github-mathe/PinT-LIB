#%%
import pybamm
import numpy as np
import batpint.parameter.javid as bpar
from batpint.problems.battery import batteryExperiment
import matplotlib.pyplot as plt


# Script parameters
half_cell = True
exp = "CCCV" # GITT or CCCV
nCycles = 1
num_cycles = 150

# Space discretization
var_pts = {
    "x_n": 1,   # points in the negative electrode
    "x_s": 30,  # points in the separator
    "x_p": 30,  # points in the positive electrode
    "r_p": 30  # points in the radius of positive electrode
}

def make_simulation(nCycles):
    """Create an independent DFN simulation."""

    args = ({"working electrode": "positive"},) if half_cell else ()

    model = pybamm.lithium_ion.DFN(*args)

    parameter_values=pybamm.ParameterValues(
        bpar.Li_half.PARAMS if half_cell else bpar.Li_full.PARAMS)

    experiment = batteryExperiment(nCycles, expType=exp)
    solver = pybamm.IDAKLUSolver()

    return pybamm.Simulation(
        model,
        experiment=experiment,
        parameter_values=parameter_values,
        var_pts=var_pts,
        solver=solver,
    )


# simulations
sim_all = make_simulation(num_cycles)   # all cycles at once
sim_from_last= make_simulation(nCycles)   # one step at a time starting from last state
sim_from_new = make_simulation(nCycles)    # one step at a time constracting new state every cycle

#solve for all cycles
solution_all = sim_all.solve()

# starting solutions are none for the first cycle
init_sol_last = None
init_sol_new = None

# store cycle solutions
solution_from_last = []
solution_from_new = []

# store global times per cycle
t_last = []
t_new = []

# time diff between cycles 
t_delta_last = 0.0
t_delta_new = 0.0

# first solving type to solve using the solution from the previous cycle
# without time correction by resetting times of the startinng solution to 0
 
for cycle in range(num_cycles):
    
    sol_last = sim_from_last.solve(starting_solution=init_sol_last)

    cycle_sol = sol_last.cycles[-1]
    
    # global times of the cycle
    t_global_last = cycle_sol.t + t_delta_last
    # store the global time per cycle
    t_last.append(t_global_last)
    # adjust for time diff between cycles
    t_delta_last = t_global_last[-1]
    # store cycle soltuion
    solution_from_last.append(cycle_sol)
    # use last state of the present cycle solution
    # as starting solution for the next
    init_sol_last = cycle_sol.last_state.copy()
    # reset starting solution time to 0
    init_sol_last._t = None
    init_sol_last._t_eval = None
    init_sol_last._all_ts = [np.array([0.0])]
    init_sol_last._all_t_evals=[np.array([0.0])]

# this can be commented if we start from "None" solution for the first cycle
# if not commented: we start from new solution constructed from the first_state
# of the very first cycle computed before
#%%
start_sol = solution_from_last[0].first_state.copy()
init_sol_new = pybamm.Solution(all_ts=np.array([0.0]), 
                              all_ys=np.array(start_sol.all_ys[-1],copy=True),
                              all_models=start_sol.all_models[-1],
                              all_inputs=start_sol.all_inputs[-1],
                              all_yps=np.array(start_sol.all_yps[-1],copy=True) if start_sol.all_yps is not None else None,
                              t_event=None,
                              y_event=None,
                              options=start_sol.user_options,
                              all_t_evals=None
                              )

# %%
# compute again
for cycle in range(num_cycles):
    
    sol_new = sim_from_new.solve(starting_solution=init_sol_new)
    
    cycle_sol = sol_new.cycles[-1]
    
    solution_from_new.append(cycle_sol)
    
    sol_end = cycle_sol.last_state.copy()
    # global times of the cycle
    t_global_new = cycle_sol.t + t_delta_new
    t_new.append(t_global_new)
    t_delta_new = t_global_new[-1]
    # create a solution object 
    # with times 0 and states from the last state of the present solution cycle
    init_sol_new = pybamm.Solution(all_ts=np.array([0.0]), 
                                  all_ys=np.array(sol_end.all_ys[-1],copy=True),
                                  all_models=sol_end.all_models[-1],
                                  all_inputs=sol_end.all_inputs[-1],
                                  all_yps=np.array(sol_end.all_yps[-1],copy=True) if sol_end.all_yps is not None else None,
                                  t_event=None,
                                  y_event=None,
                                  options=sol_end.user_options,
                                  all_t_evals=None)
    
results = {
    "all": solution_all.cycles[:],
    "from_last": solution_from_last[:],
    "from_new": solution_from_new[:],                           
    }

times = {
    "all": [sol.t for sol in solution_all.cycles],
    "from_last": t_last[:],
    "from_new": t_new[:],                           
    }

print(f"Solved {num_cycles} cycles.")
#%%
# check the size of the states
# processed models
built_model_all = solution_all.cycles[0].first_state.all_models[0]
built_model_last = solution_from_last[0].first_state.all_models[0]
built_model_new = solution_from_new[0].first_state.all_models[0]

len_t_all = np.array([sol.t.shape[0] for sol in solution_all.cycles])
len_t_last = np.array([t.shape[0] for t in t_last])
len_t_new = np.array([t.shape[0] for t in t_new])

print("Same state vector size: ", built_model_all.len_rhs_and_alg == built_model_last.len_rhs_and_alg == built_model_new.len_rhs_and_alg)
print("Same differential state vector size: ", built_model_all.len_rhs == built_model_last.len_rhs == built_model_new.len_rhs)
print("Same algebraic state vector size: ", built_model_all.len_alg == built_model_last.len_alg == built_model_new.len_alg)
print("Differ in time lengths: ", np.where(len_t_all != len_t_last) , np.where(len_t_last != len_t_new))
# data at the time interface = time at the switch between cycles
len_rhs = built_model_all.len_rhs
len_alg = built_model_all.len_alg


#%%
# interface times
tStart_per_cycle = np.zeros((num_cycles, 3), dtype=float)
tEnd_per_cycle = np.zeros((num_cycles, 3), dtype=float)
tSwitch_per_cycle = np.zeros((num_cycles, 3), dtype=float)

for i in range(num_cycles):
    for position, (key, tCycle) in enumerate(times.items()):
        tStart_per_cycle[i, position] = tCycle[i][0]
        tEnd_per_cycle[i, position] = tCycle[i][-1]
        if i==0: continue
        tSwitch_per_cycle[i, position] = tStart_per_cycle[i, position] - tEnd_per_cycle[i-1, position]

diff_tStart = np.abs(tStart_per_cycle[:,1:] - tStart_per_cycle[:,[0]])
diff_tEnd = np.abs(tEnd_per_cycle[:,1:] - tEnd_per_cycle[:,[0]])

yStart_per_cycle = np.zeros((num_cycles, 3), dtype=object)
yEnd_per_cycle = np.zeros((num_cycles, 3), dtype=object)
ySwitch_per_cycle = np.zeros((num_cycles, 3), dtype=object)

ypStart_per_cycle = np.zeros((num_cycles, 3), dtype=object)
ypEnd_per_cycle = np.zeros((num_cycles, 3), dtype=object)
ypSwitch_per_cycle = np.zeros((num_cycles, 3), dtype=object)


err_ySwitch = np.zeros((num_cycles,3), dtype=float)
err_yDiffSwitch = np.zeros((num_cycles,3), dtype=float)
err_yAlgSwitch = np.zeros((num_cycles,3), dtype=float)

err_ypSwitch = np.zeros((num_cycles,3), dtype=float)
err_ypDiffSwitch = np.zeros((num_cycles,3), dtype=float)
err_ypAlgSwitch = np.zeros((num_cycles,3), dtype=float)

for i in range(num_cycles):
    for position, (key, sol) in enumerate(results.items()):
        yStart_per_cycle[i, position] = sol[i].y[:,0]
        yEnd_per_cycle[i, position] = sol[i].y[:,-1]
        
        ypStart_per_cycle[i, position] = sol[i].yp[:,0]
        ypEnd_per_cycle[i, position] = sol[i].yp[:,-1]
        if i==0: continue
        ySwitch_per_cycle[i, position] = yStart_per_cycle[i, position] - yEnd_per_cycle[i-1, position]
        ypSwitch_per_cycle[i, position] = ypStart_per_cycle[i, position] - ypEnd_per_cycle[i-1, position]
        
        err_ySwitch[i, position] = np.linalg.norm(ySwitch_per_cycle[i, position], ord=np.inf)
        err_yDiffSwitch[i, position] = np.linalg.norm(ySwitch_per_cycle[i, position][:len_rhs], ord=np.inf)
        err_yAlgSwitch[i, position] = np.linalg.norm(ySwitch_per_cycle[i, position][len_rhs:], ord=np.inf)
        
        err_ypSwitch[i, position] = np.linalg.norm(ypSwitch_per_cycle[i, position], ord=np.inf)
        err_ypDiffSwitch[i, position] = np.linalg.norm(ypSwitch_per_cycle[i, position][:len_rhs], ord=np.inf)
        err_ypAlgSwitch[i, position] = np.linalg.norm(ypSwitch_per_cycle[i, position][len_rhs:], ord=np.inf)

#%%    
# Interface differences at cycle interfaces.
colors = ["#D55E00", "#0072B2", "#009E73"]
solver_labels = [f"Solver '{key}'" for key in results]

fig1, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
for position, label in enumerate(solver_labels):
    style = {
        "color": colors[position % len(colors)],
        "marker": "o",
        "linewidth": 1.8,
        "markersize": 4,
        "label": label,
    }
    ax.plot(tSwitch_per_cycle[:, position], **style)
    ax1.plot(err_yDiffSwitch[:, position], **style)
    ax2.plot(err_yAlgSwitch[:, position], **style)

for current_ax in (ax, ax1, ax2):
    current_ax.set_yscale("symlog", linthresh=1e-14)
    current_ax.set_xlabel("Cycles")
    current_ax.grid(True, which="both", linestyle=":", alpha=0.6)
    current_ax.set_xticks(range(num_cycles))
    current_ax.set_xticklabels(
        [str(cycle) for cycle in range(1, num_cycles + 1)], fontsize=8
    )
    current_ax.tick_params(axis="y", labelsize=8)
    current_ax.xaxis.label.set_fontsize(9)
    current_ax.yaxis.label.set_fontsize(9)
    current_ax.legend(frameon=False, fontsize=8)

ax.set_ylabel(r"$t_+ - t_-$")
ax.set_title("Interface time difference")
ax1.set_ylabel(r"$\|y_{+,d} - y_{-,d}\|_\infty$")
ax1.set_title("Differential-state")
ax2.set_ylabel(r"$\|y_{+,a} - y_{-,a}\|_\infty$")
ax2.set_title("Algebraic-state")
for current_ax in (ax, ax1, ax2):
    current_ax.title.set_fontsize(10)
fig1.suptitle("Time mismatch at cycle interfaces")
fig2.suptitle("State mismatch at cycle interfaces")
plt.show()


#%%
# difference in states between solvers at the time interface
# difference between interface times for solvers all and step
fig, (ax_last, ax_new) = plt.subplots(1, 2, sharey=True, layout="constrained")

ax_last.semilogy(diff_tStart[:, 0], "o", label=r"$t_{-, all} - t_{-, from last}$")
ax_last.semilogy(diff_tEnd[:, 0], "x", label=r"$t_{+, all} - t_{+, from last}$")
ax_last.set_title("From last")
ax_new.semilogy(diff_tStart[:, 1], "s", label=r"$t_{-, all} - t_{-, from new}$")
ax_new.semilogy(diff_tEnd[:, 1], "d", label=r"$t_{+, all} - t_{+, from new}$")
ax_new.set_title("From new")

for current_ax in (ax_last, ax_new):
    current_ax.set_yscale("symlog", linthresh=1e-14)
    current_ax.set_xlabel(r"$t_{\Gamma}$")
    current_ax.set_xticks(range(0, num_cycles, round(num_cycles / 5)))
    current_ax.legend()

ax_last.set_ylabel(r"$\Delta t_{\Gamma}$")
fig.suptitle("Interface time difference per cycle")
plt.show()

#%%
#TODO
diff_tStart_new_last = np.abs(tStart_per_cycle[:,1] - tStart_per_cycle[:,2])
diff_tEnd_new_last = np.abs(tEnd_per_cycle[:,1] - tEnd_per_cycle[:,2])
diff_yStart_new_last = np.array([np.linalg.norm(yStart_per_cycle[i,1] - yStart_per_cycle[i,2], ord=np.inf) for i in range(num_cycles)])
diff_yEnd_new_last = np.array([np.linalg.norm(yEnd_per_cycle[i,1] - yEnd_per_cycle[i,2], ord=np.inf) for i in range(num_cycles)])
diff_ypStart_new_last = np.array([np.linalg.norm(ypStart_per_cycle[i,1] - ypStart_per_cycle[i,2], ord=np.inf) for i in range(num_cycles)])
diff_ypEnd_new_last = np.array([np.linalg.norm(ypEnd_per_cycle[i,1] - ypEnd_per_cycle[i,2], ord=np.inf) for i in range(num_cycles)])
#%%
fig, ax = plt.subplots(2,2, sharex="all")
ax[0,0].semilogy(diff_tStart_new_last, label = r"$\Delta t_{start}$", color="blue")
ax[0,0].semilogy(diff_tEnd_new_last, label = r"$\Delta t_{end}$", color="orange")
ax[0,0].set_ylabel(r"$\Delta t$")

ax[0,1].semilogy(diff_yStart_new_last, label=r"$\Delta y_{start}$",color="blue")
ax[0,1].semilogy(diff_yEnd_new_last, label=r"$\Delta y_{end}$",color="orange")
ax[0,1].set_ylabel(r"$\Delta y$")

ax[1,0].semilogy(diff_ypStart_new_last, label=r"$\Delta y'_{start}$",color="blue")
ax[1,0].set_xlabel("cycles")
ax[1,0].set_ylabel(r"$\Delta y'$")

ax[1,1].semilogy(diff_ypEnd_new_last, label=r"$\Delta y'_{end}$",color="orange")
ax[1,1].set_xlabel("cycles")
ax[1,1].set_ylabel(r"$\Delta y'$")

fig.legend(["start of the cycle", "end of the cycle"])
fig.suptitle("Difference between solvers 'from last' and 'from new' at the cycle interfaces")
plt.show()

#%%
# plot states
cycle = num_cycles - 1
state_variable = lambda cycle, sol_type, state: results[sol_type][cycle][state]
c_s_p = "Positive particle concentration [mol.m-3]"
eps_ce = "Porosity times concentration [mol.m-3]"
ce = "Electrolyte concentration [mol.m-3]"

phi_s_p = "Positive electrode potential [V]"
phi_e = "Electrolyte potential [V]"
Q_abs = "Throughput capacity [A.h]"
Q_discharge = "Discharge capacity [A.h]"
current_profile = "Current [A]"
# interface states
# solid 

def plot_state(ax, cycle, sol_type, t_id, state_var, x_id=None):
    var = state_variable(cycle, sol_type, state_var)
    if var.dimensions == 2:
        nodes = var.mesh.nodes*1e6
        values = var.entries[:, x_id, t_id]
        ax.plot(nodes, values)
        ax.set_xlabel(r"$\mu m$")
        ax.set_ylabel("c_s")

    elif var.dimensions == 1:
        nodes = var.mesh.nodes*1e6
        values = var.entries[:, t_id]
        ax.plot(nodes, values)
        ax.set_xlabel(r"$\mu m$")
        ax.set_ylabel(state_var)

    else:
        values = var.entries[t_id]
        ax.plot(values,"o")
        ax.set_ylabel(state_var)
    return ax

t_id = -1
x_id = 0
#%%
labels = ["" for _ in range(len(results))]
for state_var in [c_s_p, eps_ce, phi_s_p, phi_e, current_profile, Q_discharge]:
    fig, ax = plt.subplots()
    ax.set_title(state_var)
    for position, key in enumerate(results):
        plot_state(ax, cycle, key, t_id, state_var, x_id=x_id)
        labels[position] = key + f" at t = {times[key][cycle][t_id]:.2f} s"
        hold = True
    ax.legend(labels)

plt.show()
