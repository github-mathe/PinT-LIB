from pprint import pp as pprint
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pybamm

from batpint.plotting import configure_matplotlib
from batpint.problems.battery import (
    DEFAULT_ATOL,
    batteryExperiment,
    make_battery_model,
    make_parameter_values,
)

configure_matplotlib()

# Script parameters
model_name = "DFN"  # DFN or SPMe
atol = DEFAULT_ATOL

half_cell = True
exp = "CCCV"  # GITT or CCCV
nCycles = 30
showTimeSteps = True
sType = "IDAKLU"

# Run script
model = make_battery_model(model_name, half_cell=half_cell)

print("Model variables :")
pprint(model.default_var_pts)

# Space discretization
var_pts = {
    "x_n": 3,  # points in the negative electrode
    "x_s": 30,  # points in the separator
    "x_p": 30,  # points in the positive electrode
    "r_n": 3,  # points in the radius of negative electrode
    "r_p": 100,  # points in the radius of positive electrode
}

# Setup experiment
parameter_values = make_parameter_values(half_cell)

experiment = batteryExperiment(nCycles, expType=exp)

# Setup solver
if sType == "CASADI":
    solver = pybamm.CasadiSolver(mode="safe", atol=atol)
elif sType == "IDAKLU":
    solver = pybamm.IDAKLUSolver(atol=atol)
else:
    raise NotImplementedError(f"{sType=}")


# Run simulation
print("Running simulation ...")
tBeg = time()
sim = pybamm.Simulation(model, experiment=experiment, parameter_values=parameter_values, solver=solver, var_pts=var_pts)
sol = sim.solve()
tEnd = time()
tComp = tEnd - tBeg
print(f" -- done in {tComp:1.2f}s ({tComp / nCycles:1.2f}s per cycle)")

# Extract data
print("Extracting data ...")
times = sol["Time [h]"].entries
voltage = sol["Voltage [V]"].entries
capacity = sol["Discharge capacity [A.h]"].entries
print(" -- done")

# %% Plots
label = f"{model.name}, {exp} ({'HC' if half_cell else 'FC'})"

plt.figure("voltage vs capacity")
plt.plot(capacity, voltage, label=label)
plt.xlabel("Capacity [Ah]"), plt.ylabel("Voltage [V]")
plt.grid(True), plt.legend(), plt.tight_layout()


plt.figure("cycles")
plt.plot(times, voltage, label=f"{label}")
plt.xlabel("Time [h]"), plt.ylabel("Voltage [V]")
plt.grid(True), plt.legend()

if showTimeSteps:
    deltas = np.ediff1d(np.concatenate(sol.all_ts))
    deltas /= voltage.max() * 10
    deltas += 2

    ax = plt.gca().twinx()
    ax.set_ylabel("Time steps size")
    ax.plot(times[1:], deltas, ":", color="gray")

plt.tight_layout()
