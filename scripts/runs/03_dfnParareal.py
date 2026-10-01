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
model_step = pybamm.lithium_ion.DFN(*args)
model_all = pybamm.lithium_ion.DFN(*args)
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
experiment_step = batteryExperiment(nCycles, expType=exp)
experiment_all = batteryExperiment(num_cycles,expType=exp)
solver = pybamm.IDAKLUSolver()
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
init_sol = None
solution_step = []
for cycle in range(num_cycles):
    sol = sim_step.solve(starting_solution=init_sol)
    cycle_sol = sol.cycles[-1]
    solution_step.append(cycle_sol)
    init_sol = cycle_sol.last_state


#%%
print(f"Solved {num_cycles} cycles.")
print(f"Number of stored cycles: {len(solution_all.cycles)}\n")

# extract the differential and algebraic states
solution = {"step":{}, "all":{}}

solution["step"]["cycles"] = solution_step[:]
solution["all"]["cycles"] = solution_all.cycles[:]
for key, data in solution.items():
    cycles = data["cycles"]
    built_model = cycles[0].first_state.all_models[0]
    data["built_model"] = built_model
    data["size_y"] = cycles[0].first_state.y.shape[0]
    data["n_diff"] = built_model.concatenated_rhs.size
    data["n_alg"] = built_model.concatenated_algebraic.size
    data["t_minus"] = [cycles[cycle_id].t[-1] for cycle_id in range(num_cycles-1)]
    data["t_plus"] = [cycles[cycle_id+1].t[0] for cycle_id in range(num_cycles-1)]
    data["t_correction"] = [cycles[cycle_id+1].t[0]-cycles[cycle_id].t[-1] for cycle_id in range(num_cycles-1)]
    data["y_minus"] = [cycles[cycle_id].y[:,-1] for cycle_id in range(num_cycles-1)]
    data["y_plus"] = [cycles[cycle_id+1].y[:,0] for cycle_id in range(num_cycles-1)]
    data["y_correction"] = [cycles[cycle_id+1].y[:,0]-cycles[cycle_id].y[:,-1] for cycle_id in range(num_cycles-1)]
    data["y_diff_minus"] = [cycles[cycle_id].y[:built_model.concatenated_rhs.size,-1] for cycle_id in range(num_cycles-1)]
    data["y_alg_minus"] = [cycles[cycle_id].y[built_model.concatenated_rhs.size:,-1] for cycle_id in range(num_cycles-1)]
    data["y_diff_plus"] = [cycles[cycle_id+1].y[:built_model.concatenated_rhs.size,0] for cycle_id in range(num_cycles-1)]
    data["y_alg_plus"] = [cycles[cycle_id+1].y[built_model.concatenated_rhs.size:,0] for cycle_id in range(num_cycles-1)]
    data["y_diff_correction"] = [cycles[cycle_id+1].y[:built_model.concatenated_rhs.size,0]-cycles[cycle_id].y[:built_model.concatenated_rhs.size,-1] for cycle_id in range(num_cycles-1)]
    data["y_alg_correction"] = [cycles[cycle_id+1].y[built_model.concatenated_rhs.size:,0]-cycles[cycle_id].y[built_model.concatenated_rhs.size:,-1] for cycle_id in range(num_cycles-1)]

# check the size
print("Same state vector size: ", solution["all"]["size_y"] == solution["step"]["size_y"])
print("Same differential state vector size: ", solution["all"]["n_diff"] == solution["step"]["n_diff"])
print("Same algebraic state vector size: ", solution["all"]["n_alg"] == solution["step"]["n_alg"])

print("\n" + "=" * 50)
print("CYCLE SWITCH COMPARISON")
print("=" * 50)

for key, dict_sol in solution.items():
    print(key, "solver")
    print("Difference in times: ", solution[key]["t_correction"])
    print("Difference in states: ", np.max(np.abs(solution[key]["y_correction"])))
    print("Difference in diff states: ", np.max(np.abs(solution[key]["y_diff_correction"])))
    print("Difference in alg states: ", np.max(np.abs(solution[key]["y_alg_correction"])))
    print()
# step vs sequantial solver
print("Differences in states between step and sequential solutions\n")
print("Delta t_minus:", [solution["step"]["t_minus"][num]-solution["all"]["t_minus"][num] for num in range(len(solution["all"]["t_minus"]))])
print("Delta t_plus:", [solution["step"]["t_plus"][num]-solution["all"]["t_plus"][num] for num in range(len(solution["all"]["t_plus"]))])
print("Delta y_minus:", np.max(np.abs([solution["step"]["y_minus"][num]-solution["all"]["y_minus"][num] for num in range(len(solution["all"]["y_minus"]))])))
print("Delta y_plus:", np.max(np.abs([solution["step"]["y_plus"][num]-solution["all"]["y_plus"][num] for num in range(len(solution["all"]["y_plus"]))])))
print("Delta y_diff_plus:", np.max(np.abs([solution["step"]["y_diff_plus"][num]-solution["all"]["y_diff_plus"][num] for num in range(len(solution["all"]["y_diff_plus"]))])))
print("Delta y_alg_plus:", np.max(np.abs([solution["step"]["y_alg_plus"][num]-solution["all"]["y_alg_plus"][num] for num in range(len(solution["all"]["y_diff_plus"]))])))

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
