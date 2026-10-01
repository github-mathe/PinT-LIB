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
num_cycles = 2

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

t_last_all = [sol.t[-1] for sol in solution_all.cycles]
t_start_all = [sol.t[0] for sol in solution_all.cycles]
t_switch_all = [t1-t2 for t1,t2 in zip(t_last_all[:-1],t_start_all[1:])]

t_last_step = [t[-1] for t in t_step]
t_start_step = [t[0] for t in t_step]
t_switch_step = [t1-t2 for t1,t2 in zip(t_last_step[:-1],t_start_step[1:])]

err_t_last = [t1-t2 for t1,t2 in zip(t_last_all,t_last_step)]
err_t_start = [t1-t2 for t1,t2 in zip(t_start_all,t_start_step)]

#interface states

y_all = [sol.y for sol in solution_all.cycles]
y_last_all = [sol.y[:,-1] for sol in solution_all.cycles]
y_start_all = [sol.y[:,0] for sol in solution_all.cycles]
y_switch_all = [y1-y2 for y1,y2 in zip(y_last_all[:-1],y_start_all[1:])]

y_diff_last_all = [sol.y[:built_model_all.len_rhs,-1] for sol in solution_all.cycles]
y_diff_start_all = [sol.y[:built_model_all.len_rhs,0] for sol in solution_all.cycles]
y_diff_switch_all = [y1-y2 for y1,y2 in zip(y_diff_last_all[:-1],y_diff_start_all[1:])]

y_alg_last_all = [sol.y[built_model_all.len_rhs:,-1] for sol in solution_all.cycles]
y_alg_start_all = [sol.y[built_model_all.len_rhs:,0] for sol in solution_all.cycles]
y_alg_switch_all = [y1-y2 for y1,y2 in zip(y_alg_last_all[:-1],y_alg_start_all[1:])]


y_step = [sol.y for sol in solution_step]
y_last_step = [sol.y[:,-1] for sol in solution_step]
y_start_step = [sol.y[:,0] for sol in solution_step]
y_switch_step = [y1-y2 for y1,y2 in zip(y_last_step[:-1],y_start_step[1:])]

y_diff_last_step = [sol.y[:built_model_step.len_rhs,-1] for sol in solution_step]
y_diff_start_step = [sol.y[:built_model_step.len_rhs,0] for sol in solution_step]
y_diff_switch_step = [y1-y2 for y1,y2 in zip(y_diff_last_step[:-1],y_diff_start_step[1:])]

y_alg_last_step =  [sol.y[built_model_step.len_rhs:,-1] for sol in solution_step]
y_alg_start_step = [sol.y[built_model_step.len_rhs:,0] for sol in solution_step]
y_alg_switch_step = [y1-y2 for y1,y2 in zip(y_alg_last_step[:-1],y_alg_start_step[1:])]

err_y = [np.linalg.norm(y1-y2, ord=np.inf, axis = 0) for y1,y2 in zip(y_all,y_step)]
err_y_diff = [np.linalg.norm(y1[:built_model_step.len_rhs,:]-y2[:built_model_step.len_rhs,:], ord=np.inf, axis = 0) for y1,y2 in zip(y_all,y_step)]
err_y_alg = [np.linalg.norm(y1[built_model_step.len_rhs:,:]-y2[built_model_step.len_rhs:,:], ord=np.inf, axis = 0) for y1,y2 in zip(y_all,y_step)]

#%%
print("\n" + "=" * 50)
print("CYCLE COMPARISON between sequential and step solutions")
print("=" * 50)


for cycle in range(num_cycles):
    print("\n" + "=" * 50)
    print(f"Cycle {cycle}")
    print(f"starting time difference:", err_t_start[cycle])
    print(f"ending time difference:", err_t_last[cycle])
    
    print(f"starting state difference:", err_y[cycle][0])
    print(f"starting differential state difference:", err_y_diff[cycle][0])
    print(f"starting algebraic state difference:", err_y_alg[cycle][0])
    
    print(f"ending state difference:", err_y[cycle][-1])
    print(f"ending differential state difference:", err_y_diff[cycle][-1])
    print(f"ending algebraic state difference:", err_y_alg[cycle][-1])
    
    start_error = np.linalg.norm(y_start_step[cycle] - y_start_all[cycle], ord=np.inf)
    end_error = np.linalg.norm(y_last_step[cycle] - y_last_all[cycle], ord=np.inf)
    print(start_error == err_y[cycle][0], end_error == err_y[cycle][-1])
    print("=" * 50)
    

#%%
# state variables of the model
for cycle in range(num_cycles-1):
    fig, axes = plt.subplots(2, 3, figsize=(12, 4))

    all_old = solution["all"]["cycles"][cycle]
    all_new = solution["all"]["cycles"][cycle + 1]
    step_old = solution["step"]["cycles"][cycle]
    step_new = solution["step"]["cycles"][cycle + 1]

    # Switch times
    t_minus_all = all_old.t[-1]
    t_plus_all = all_new.t[0]

    t_minus_step = step_old.t[-1]
    t_plus_step = step_new.t[0]

    # ------------------------------------------------------------
    # State variables - sequential
    # ------------------------------------------------------------
    csp_all_old = all_old["Positive particle concentration [mol.m-3]"]
    csp_all_new = all_new["Positive particle concentration [mol.m-3]"]

    eps_ce_all_old = all_old["Porosity times concentration [mol.m-3]"]
    eps_ce_all_new = all_new["Porosity times concentration [mol.m-3]"]

    ce_all_old = all_old["Electrolyte concentration [mol.m-3]"]
    ce_all_new = all_new["Electrolyte concentration [mol.m-3]"]

    phi_sp_all_old = all_old["Positive electrode potential [V]"]
    phi_sp_all_new = all_new["Positive electrode potential [V]"]

    phi_e_all_old = all_old["Electrolyte potential [V]"]
    phi_e_all_new = all_new["Electrolyte potential [V]"]

    I_all_old = all_old["Current [A]"]
    I_all_new = all_new["Current [A]"]

    # ------------------------------------------------------------
    # State variables - stepwise
    # ------------------------------------------------------------

    csp_step_old = step_old["Positive particle concentration [mol.m-3]"]
    csp_step_new = step_new["Positive particle concentration [mol.m-3]"]

    eps_ce_step_old = step_old["Porosity times concentration [mol.m-3]"]
    eps_ce_step_new = step_new["Porosity times concentration [mol.m-3]"]

    ce_step_old = step_old["Electrolyte concentration [mol.m-3]"]
    ce_step_new = step_new["Electrolyte concentration [mol.m-3]"]

    phi_sp_step_old = step_old["Positive electrode potential [V]"]
    phi_sp_step_new = step_new["Positive electrode potential [V]"]

    phi_e_step_old = step_old["Electrolyte potential [V]"]
    phi_e_step_new = step_new["Electrolyte potential [V]"]

    I_step_old = step_old["Current [A]"]
    I_step_new = step_new["Current [A]"]

    # ------------------------------------------------------------
    # Mesh points
    # ------------------------------------------------------------

    r_p = all_old["r_p [m]"].entries[:, 0, 0]
    x_p = all_old["x_p [m]"].entries[:, 0]

    x_eps_ce = eps_ce_all_old.mesh.nodes
    x_ce = ce_all_old.mesh.nodes

    # ============================================================
    # Particle concentration
    # ============================================================
    ax = axes[0, 0]
    ax.plot(r_p * 1e6, csp_all_old.entries[:, -1, -1], "-", label=fr"All $t_s^-$")
    ax.plot(r_p * 1e6, csp_all_new.entries[:, -1, 0], "--", label=fr"All $t_s^+$")
    ax.plot(r_p * 1e6, csp_step_old.entries[:, -1, -1], ":", label=fr"Step $t_s^-$")
    ax.plot(r_p * 1e6, csp_step_new.entries[:, -1, 0], "-.",label=fr"Step $t_s^+$")
    ax.set_xlabel(r"$r$ [$\mu$m]")
    ax.set_ylabel(r"$c_{s,p}$ [mol m$^{-3}$]")
    ax.set_title("Particle concentration")
    ax.grid()

    # ============================================================
    # Porosity times electrolyte concentration
    # ============================================================
    ax = axes[0, 1]
    ax.plot(x_eps_ce * 1e6, eps_ce_all_old.entries[:, -1], "-")
    ax.plot(x_eps_ce * 1e6, eps_ce_all_new.entries[:, 0], "--")
    ax.plot(x_eps_ce * 1e6, eps_ce_step_old.entries[:, -1], ":")
    ax.plot(x_eps_ce * 1e6, eps_ce_step_new.entries[:, 0],"-.")
    ax.set_xlabel(r"$x$ [$\mu$m]")
    ax.set_ylabel(r"$\epsilon c_e$ [mol m$^{-3}$]")
    ax.set_title("Porosity times concentration")
    ax.grid()

    # ============================================================
    # Electrolyte concentration
    # ============================================================
    ax = axes[0, 2]
    ax.plot(x_ce * 1e6, ce_all_old.entries[:, -1], "-")
    ax.plot(x_ce * 1e6, ce_all_new.entries[:, 0], "--")
    ax.plot(x_ce * 1e6, ce_step_old.entries[:, -1], ":")
    ax.plot(x_ce * 1e6, ce_step_new.entries[:, 0], "-.")
    ax.set_xlabel(r"$x$ [$\mu$m]")
    ax.set_ylabel(r"$c_e$ [mol m$^{-3}$]")
    ax.set_title("Electrolyte concentration")
    ax.grid()

    # ============================================================
    # Positive-electrode potential
    # ============================================================
    ax = axes[1, 0]
    ax.plot(x_p * 1e6, phi_sp_all_old.entries[:, -1], "-")
    ax.plot(x_p * 1e6, phi_sp_all_new.entries[:, 0], "--")
    ax.plot(x_p * 1e6, phi_sp_step_old.entries[:, -1], ":")
    ax.plot( x_p * 1e6, phi_sp_step_new.entries[:, 0], "-.")
    ax.set_xlabel(r"$x_p$ [$\mu$m]")
    ax.set_ylabel(r"$\phi_{s,p}$ [V]")
    ax.set_title("Positive-electrode potential")
    ax.grid()

    # ============================================================
    # Electrolyte potential
    # ============================================================
    ax = axes[1, 1]
    ax.plot(x_ce * 1e6, phi_e_all_old.entries[:, -1], "-")
    ax.plot(x_ce * 1e6, phi_e_all_new.entries[:, 0], "--")
    ax.plot(x_ce * 1e6, phi_e_step_old.entries[:, -1],":")
    ax.plot(x_ce * 1e6, phi_e_step_new.entries[:, 0], "-.")
    ax.set_xlabel(r"$x$ [$\mu$m]")
    ax.set_ylabel(r"$\phi_e$ [V]")
    ax.set_title("Electrolyte potential")
    ax.grid()

    # ============================================================
    # Applied current
    # ============================================================

    ax = axes[1, 2]
    # sequential
    ax.scatter(0, I_all_old.entries[-1], s=80, marker="o")
    ax.scatter(1, I_all_new.entries[0], s=80, marker="o")
    ax.plot([0, 1], [I_all_old.entries[-1], I_all_new.entries[0]], "-")
    # stepwise
    ax.scatter(0, I_step_old.entries[-1], s=80, marker="x")
    ax.scatter(1, I_step_new.entries[0], s=80, marker="x")
    ax.plot([0, 1], [I_step_old.entries[-1], I_step_new.entries[0]], "--")
    ax.set_xticks([0, 1], [r"$t_s^-$", r"$t_s^+$"])
    ax.set_ylabel(r"$I_{\mathrm{app}}$ [A]")
    ax.set_title("Applied current")
    ax.grid()


    # ============================================================
    # Title and common legend
    # ============================================================
    fig.suptitle((
            f"Cycle {cycle + 1} → {cycle + 2} switch\n"
            f"$t_s^-={t_minus_all:.16e}$ s, "
            f"$t_s^+={t_plus_all:.16e}$ s"
        ),
        fontsize=14,
        y=0.99,
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="lower center", ncol=4,)
    fig.subplots_adjust( wspace=1, hspace=1, top=0.82, bottom=0.18)
    plt.show()
# %%
