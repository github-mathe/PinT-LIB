import pybamm
import numpy as np
from time import time

from batpint import plt
import batpint.parameter.javid as bpar

# Script parameters
half_cell = False
exp = "CCCV" # GITT or CCCV
nCycles = 2
showTimeSteps = True

# Run script
args = ({"working electrode": "positive"},) if half_cell else ()
model = pybamm.lithium_ion.DFN(*args)

print("Model variables :")
print(model.default_var_pts)

# Space discretization
var_pts = {
    "x_n": 3,
    "x_s": 30,
    "x_p": 30,
    "r_n": 3,
    "r_p": 100
}

# Setup experiment
parameter_values=pybamm.ParameterValues(
    bpar.Li_half.PARAMS if half_cell else bpar.Li_full.PARAMS)

if exp == "GITT":
    # define GITT experiment: short pulse followed by long rest period
    pulse_duration = "10 minutes"  # duration of current pulse
    rest_duration = "2 hours"      # relaxation time
    current_rate = "0.05C"         # current during pulse
    voltage_min = 2.5              # V - discharge cutoff

    # WARNING : this parameter is not used !
    voltage_max = 4.2              # V - charge cutoff

    # Create GITT discharge experiment
    # Tuple groups discharge+rest as ONE cycle
    experiment = pybamm.Experiment(
        [
            (
                f"Discharge at {current_rate} for {pulse_duration}",
                f"Rest for {rest_duration}"
            )
        ] * nCycles,
        termination=f"{voltage_min} V"
    )
elif exp == "CCCV":
    # define CCVV experiment: many cycles of discharge / charge
    experiment = pybamm.Experiment(
        [
            (
            #"Rest for 5 minutes",
            "Discharge at 2C until 3.5 V",
            "Rest for 5 minutes",
            "Charge at 0.1C until 4.2 V",
            "Rest for 10 minutes"
            )
        ]
        * nCycles
        )
else:
    raise NotImplementedError(f"{exp=}")

# Setup solver
solver = pybamm.CasadiSolver(mode="safe")

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
print(f" -- done in {tEnd-tBeg:1.2f}s")

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
