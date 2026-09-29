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
                            solver=solver)

#solve for all cycles
solution_all = sim_all.solve()

#%%
print(f"Solved {num_cycles} cycles.")
print(f"Number of stored cycles: {len(solution_all.cycles)}")

cycle0 = solution_all.cycles[0]

built_model = cycle0.first_state.all_models[0]
y = cycle0.first_state.y
n_diff = built_model.concatenated_rhs.size
n_alg = built_model.concatenated_algebraic.size

print("\n" + "=" * 80)
print("DAE STATE STRUCTURE")
print("=" * 80)

print(f"Differential states : {n_diff}")
print(f"Algebraic states    : {n_alg}")
print(f"Total states        : {n_diff + n_alg}")

print(
    "Solution y size   : "
    f"{y.shape[0]}"
)

for eq_type, equations in [
    ("DIFFERENTIAL", built_model.rhs),
    ("ALGEBRAIC", built_model.algebraic),
]:
    print(eq_type)
    for variable in equations:
        slices = built_model.y_slices[variable]

        for slc in slices:
            print(
                f"{variable.name:70s} "
                f"[{slc.start}:{slc.stop}] "
                f"size = {slc.stop - slc.start}"
            )

    print()

for cycle_index in range(len(solution_all.cycles) - 1):
    cycle_old = solution_all.cycles[cycle_index]
    cycle_new = solution_all.cycles[cycle_index + 1]
    
    if cycle_old is None or cycle_new is None:
        continue

    y_old = np.asarray(
        cycle_old.last_state.y[:, -1]
    ).reshape(-1)

    y_new = np.asarray(
        cycle_new.first_state.y[:, -1]
    ).reshape(-1)

    y_old_diff = y_old[:n_diff]
    y_new_diff = y_new[:n_diff]

    y_old_alg = y_old[n_diff:]
    y_new_alg = y_new[n_diff:]

    diff_jump = y_new_diff - y_old_diff
    alg_jump = y_new_alg - y_old_alg

    print(
        f"\nCycle {cycle_index + 1} -> "
        f"Cycle {cycle_index + 2}"
    )

    print(
        "  Time end old cycle:   "
        f"{cycle_old.t[-1]:.16e}"
    )

    print(
        "  Time start new cycle: "
        f"{cycle_new.t[0]:.16e}"
    )

    print(
        "  Time difference:      "
        f"{cycle_new.t[0] - cycle_old.t[-1]:.16e}"
    )

    print(
        "  Differential jump:    "
        f"{np.max(np.abs(diff_jump)):.16e}"
    )

    print(
        "  Algebraic jump:       "
        f"{np.max(np.abs(alg_jump)):.16e}"
    )
    
#%%
# state variables of the model
c_s_p = solution_all["Positive particle concentration [mol.m-3]"]
eps_c_e = solution_all["Porosity times concentration [mol.m-3]"]    
ce = solution_all["Electrolyte concentration [mol.m-3]"]
phi_s_p = solution_all["Positive electrode potential [V]"]
phi_e = solution_all["Electrolyte potential [V]"]

r_p = solution_all["r_p [m]"].entries[:, 0, 0]
x_n = solution_all["x_n [m]"].entries[0]
x = solution_all["x [m]"].entries[:,-1]
x_s = solution_all["x_s [m]"].entries[:,-1]
x_p = solution_all["x_p [m]"].entries[:,0]
t = solution_all["Time [s]"].entries

k_t = 1000
k_x = 20

c_particle = c_s_p.entries[:, k_x, k_t]

plt.plot(r_p, c_particle)
plt.xlabel(r"$r_p$ [m]")
plt.ylabel(r"$c_{s,p}$ [mol m$^{-3}$]")


