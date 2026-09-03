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
nCycles = 1
showTimeSteps = True

# Run script
args = ({"working electrode": "positive"},) if half_cell else ()
model = pybamm.lithium_ion.DFN(*args)

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
solver = pybamm.IDAKLUSolver()
sim = pybamm.Simulation(
    model,
    experiment=experiment,
    parameter_values=parameter_values,
    solver=solver,
    var_pts=var_pts)
sol1 = sim.solve()
state1 = sol1.last_state.y
print(len(sol1.cycles))

# second cycle 
new_model = model.set_initial_conditions_from(sol1,inplace=False)
sim_new = pybamm.Simulation(
    new_model,
        experiment=experiment,
    parameter_values=parameter_values,
    solver=solver,
    var_pts=var_pts)

sol2 =sim_new.solve()
print(len(sol2.cycles))
