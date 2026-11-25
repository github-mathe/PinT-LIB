import copy
import time

import matplotlib.pyplot as plt
import numpy as np

from batpint.parareal.errors import compute_parareal_errors
from batpint.problems.battery import DEFAULT_ATOL, make_propagator

# Script parameters
model_name = "DFN"  # DFN or SPMe
atol = DEFAULT_ATOL

half_cell = True
exp = "CCCV"  # GITT or CCCV
nCycles = 1

# Space discretization
var_pts = {
    "x_n": 1,  # points in the negative electrode
    "x_s": 10,  # points in the separator
    "x_p": 10,  # points in the positive electrode
    "r_n": 10,  # points in the radius of negative electrode
    "r_p": 10,  # points in the radius of positive electrode
}


# %%
num_cycles = 4
# check cost and accuracy
rtolCoarse = 1 / (10 ** np.arange(2, 7))
rtolFine = 1e-6  # default

fine_propagator = make_propagator(
    half_cell=half_cell, exp=exp, nCycles=nCycles, var_pts=var_pts, rtol=rtolFine, atol=atol, model=model_name
)
uStart = copy.deepcopy(fine_propagator.u_full)
solutionFine = np.empty((num_cycles + 1, *uStart.shape), dtype=uStart.dtype)
solutionCoarse = np.empty((len(rtolCoarse), num_cycles + 1, *uStart.shape), dtype=uStart.dtype)
solutionFine[0] = uStart
len_rhs = fine_propagator._last_step_model.len_rhs
state_size = uStart.shape[0]

coarse_propagators = [
    make_propagator(
        half_cell=half_cell, exp=exp, nCycles=nCycles, var_pts=var_pts, rtol=rtol_c, atol=atol, model=model_name
    )
    for rtol_c in rtolCoarse
]

for i in range(len(rtolCoarse)):
    solutionCoarse[i, 0] = copy.deepcopy(coarse_propagators[i].u_full)

tStart = 0

tCoarse_cycles = np.zeros((len(rtolCoarse), num_cycles + 1), dtype=float)
tFine_cycles = np.zeros(num_cycles + 1, dtype=float)

err_t = np.zeros((len(rtolCoarse), num_cycles + 1), dtype=float)
err_u = np.zeros((len(rtolCoarse), num_cycles + 1), dtype=float)
err_u_diff = np.zeros((len(rtolCoarse), num_cycles + 1), dtype=float)
err_up = np.zeros((len(rtolCoarse), num_cycles + 1), dtype=float)
time_Fine_cycles = np.zeros(num_cycles + 1)
time_Coarse_cycles = np.zeros((len(rtolCoarse), num_cycles + 1))

for cycle in range(num_cycles):
    # fine propagator
    time_start_Fine = time.perf_counter()
    tFine, solFine = fine_propagator.propagate(t=tFine_cycles[cycle], u=solutionFine[cycle], cycle=cycle)
    total_time_Fine = time.perf_counter() - time_start_Fine

    time_Fine_cycles[cycle + 1] = total_time_Fine
    tFine_cycles[cycle + 1] = tFine
    solutionFine[cycle + 1] = solFine

    for i, rtolC in enumerate(rtolCoarse):
        # coarse propagator
        time_start_coarse = time.perf_counter()
        tCoarse, solCoarse = coarse_propagators[i].propagate(
            t=tCoarse_cycles[i, cycle], u=solutionCoarse[i, cycle], cycle=cycle
        )
        time_Coarse_cycles[i, cycle + 1] = time.perf_counter() - time_start_coarse

        tCoarse_cycles[i, cycle + 1] = tCoarse
        solutionCoarse[i, cycle + 1] = solCoarse

err_t = compute_parareal_errors(tCoarse_cycles, tFine_cycles)
err_u = compute_parareal_errors( solutionCoarse[:,:,:,0], solutionFine[:, :, 0])
err_u_diff = compute_parareal_errors(solutionCoarse[:,:,:len_rhs,0], solutionFine[:, :len_rhs, 0])
err_up = compute_parareal_errors(solutionCoarse[:,:,:,1], solutionFine[:, :, 1])

errors = {
    "Time": err_t,
    "State": err_u,
    "Differential state": err_u_diff,
    "Derivative": err_up,
}
# %%

fine_cost = np.sum(time_Fine_cycles[1:])
coarse_cost = np.sum(time_Coarse_cycles[:, 1:], axis=1)
time_cost = coarse_cost / fine_cost

ratio = rtolCoarse / rtolFine
time_cost = (
    time_Coarse_cycles[:, 1:].sum(axis=1)
    / time_Fine_cycles[1:].sum()
)

for name, error in errors.items():
    fig, ax = plt.subplots()

    ax.plot(ratio, error[:, 1:].max(axis=1), "o-", label=name)
    ax.set_xscale("log")
    #ax.set_yscale("symlog", linthresh=1e-14)
    ax.set_xlabel("Coarse rtol / fine rtol")
    ax.set_ylabel("Maximum error over cycles")
    ax.set_title(name)
    ax.invert_xaxis()

    cost_ax = ax.twinx()
    cost_ax.plot(ratio, time_cost, "r--", label="Relative cost")
    cost_ax.set_ylabel("Coarse / fine propagation time")

    fig.legend()

plt.show()
