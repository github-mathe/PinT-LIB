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
nCycles = 30
showTimeSteps = True
sType = "IDAKLU"

# Run script
args = ({"working electrode": "positive"},) if half_cell else ()
model = pybamm.lithium_ion.DFN(*args)

print("Model variables :")
pprint(model.default_var_pts)

# Space discretization
var_pts = {
    "x_n": 3,   # points in the negative electrode
    "x_s": 30,  # points in the separator
    "x_p": 30,  # points in the positive electrode
    "r_n": 3,   # points in the radius of negative electrode
    "r_p": 100  # points in the radius of positive electrode
}

# Setup experiment
parameter_values=pybamm.ParameterValues(
    bpar.Li_half.PARAMS if half_cell else bpar.Li_full.PARAMS)

experiment = batteryProblem(nCycles, expType=exp)

# Setup solver
if sType == "CASADI":
    solver = pybamm.CasadiSolver(mode="safe")
elif sType == "IDAKLU":
    solver = pybamm.IDAKLUSolver()
else:
    raise NotImplementedError(f"{sType=}")


# Run simulation
print("Running simulation ...")
tBeg = time()
sim = pybamm.Simulation(
    model,
    experiment=experiment,
    parameter_values=parameter_values,
    solver=solver,
    var_pts=var_pts)
sol = sim.solve()
tEnd = time()
tComp = tEnd-tBeg
print(f" -- done in {tComp:1.2f}s ({tComp/nCycles:1.2f}s per cycle)")

# Extract data
print("Extracting data ...")
times = sol["Time [h]"].entries
voltage = sol["Voltage [V]"].entries
capacity = sol["Discharge capacity [A.h]"].entries
print(" -- done")

# %% Plots
label = f"{exp} ({'HC' if half_cell else 'FC'})"

plt.figure("voltage vs capacity")
plt.plot(capacity, voltage, label=label)
plt.xlabel('Capacity [Ah]'), plt.ylabel('Voltage [V]')
plt.grid(True), plt.legend(), plt.tight_layout()



plt.figure("cycles")
plt.plot(times, voltage, label=f"{label}")
plt.xlabel('Time [h]'), plt.ylabel('Voltage [V]')
plt.grid(True), plt.legend()

if showTimeSteps:
    deltas = np.ediff1d(np.concat(sol.all_ts))
    deltas /= voltage.max()*10
    deltas += 2

    ax = plt.gca().twinx()
    ax.set_ylabel("Time steps size")
    ax.plot(times[1:], deltas, ':', color="gray")

plt.tight_layout()
