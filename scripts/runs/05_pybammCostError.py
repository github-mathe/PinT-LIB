import copy
import time

#from batpint.parareal.parareal import PararealModified
import matplotlib.pyplot as plt
import numpy as np

from batpint.problems.battery import make_propagator

# Script parameters
half_cell = True
exp = "CCCV" # GITT or CCCV
nCycles = 1

# Space discretization
var_pts = {
    "x_n": 1,   # points in the negative electrode
    "x_s": 30,  # points in the separator
    "x_p": 30,  # points in the positive electrode
    "r_p": 30  # points in the radius of positive electrode
}


#%%
num_cycles = 4
# check cost and accuracy
rtolCoarse = 1/(10**np.arange(2,7))
rtolFine = 1e-6 # default

solutionFine = np.empty(num_cycles+1, dtype=object)
solutionCoarse = np.empty((len(rtolCoarse), num_cycles+1),dtype=object)

fine_propagator = make_propagator(half_cell=half_cell, exp=exp, nCycles=nCycles, var_pts=var_pts, rtol=rtolFine)
uStart = copy.deepcopy(fine_propagator.u_full)
solutionFine[0] = uStart
len_rhs = fine_propagator._last_step_model.len_rhs

coarse_propagators = [
    make_propagator(half_cell=half_cell, exp=exp, nCycles=nCycles, var_pts=var_pts, rtol=rtol_c)
    for rtol_c in rtolCoarse
]

for i in range(len(rtolCoarse)):
    solutionCoarse[i, 0] = copy.deepcopy(coarse_propagators[i].u_full)

tStart = 0

tCoarse_cycles = np.zeros((len(rtolCoarse), num_cycles+1), dtype=float)
tFine_cycles = np.zeros(num_cycles+1, dtype=float)

err_t = np.zeros((len(rtolCoarse), num_cycles+1), dtype=float)
err_u = np.zeros((len(rtolCoarse), num_cycles+1), dtype=float)
err_u_diff = np.zeros((len(rtolCoarse), num_cycles+1), dtype=float)
err_up = np.zeros((len(rtolCoarse), num_cycles+1), dtype=float)
time_Fine_cycles = np.zeros(num_cycles+1)
time_Coarse_cycles = np.zeros((len(rtolCoarse), num_cycles+1))

for cycle in range(num_cycles):

    # fine propagator
    time_start_Fine = time.perf_counter()
    tFine, solFine = fine_propagator.propagate(
        t=tFine_cycles[cycle],
        u=solutionFine[cycle],
        cycle=cycle)

    total_time_Fine = time.perf_counter() - time_start_Fine

    time_Fine_cycles[cycle+1] = total_time_Fine
    tFine_cycles[cycle+1] = tFine
    solutionFine[cycle+1] = solFine


    for i, rtolC in enumerate(rtolCoarse):
        err_t[i, 0] = np.abs(tFine_cycles[0]-tCoarse_cycles[i,0])
        err_u[i, 0] = np.linalg.norm(solutionFine[0][0] - solutionCoarse[i,0][0], ord=np.inf)
        err_u_diff[i, 0] =  np.linalg.norm(solutionFine[0][0][:len_rhs] - solutionCoarse[i,0][0][:len_rhs], ord=np.inf)
        err_up[i, 0] = np.linalg.norm(solutionFine[0][1] - solutionCoarse[i,0][1], ord=np.inf)

        # coarse propagator
        time_start_coarse = time.perf_counter()
        tCoarse, solCoarse = coarse_propagators[i].propagate(
            t=tCoarse_cycles[i,cycle],
            u=solutionCoarse[i,cycle],
            cycle=cycle)

        time_Coarse_cycles[i,cycle+1] = time.perf_counter() - time_start_coarse

        tCoarse_cycles[i,cycle+1] = tCoarse
        solutionCoarse[i,cycle+1] = solCoarse
        err_t[i, cycle+1] = np.abs(tFine-tCoarse)
        err_u[i, cycle+1] = np.linalg.norm(solFine[0] - solCoarse[0], ord=np.inf)
        err_u_diff[i, cycle+1] = np.linalg.norm(solFine[0][:len_rhs] - solCoarse[0][:len_rhs], ord=np.inf)
        err_up[i, cycle+1] = np.linalg.norm(solFine[1] - solCoarse[1], ord=np.inf)

#%%

fine_cost = np.sum(time_Fine_cycles[1:])
coarse_cost = np.sum(time_Coarse_cycles[:, 1:],axis=1)
time_cost = coarse_cost/fine_cost


fig, ax = plt.subplots()
ax.loglog([rtolC/rtolFine for rtolC in rtolCoarse], np.max(err_u[:,1:], axis=1) , color="blue", label = r"$\|u_{coarse}-u_{fine}\|_{\infty}$")
ax1 = ax.twinx()
ax.invert_xaxis()
ax1.semilogx([rtolC/rtolFine for rtolC in rtolCoarse], time_cost, color="red", label = "cost")
fig.legend()

fig, ax = plt.subplots()
ax.loglog([rtolC/rtolFine for rtolC in rtolCoarse], np.max(err_up[:,1:], axis=1), label = r"$\|up_{coarse}-up_{fine}\|_{\infty}$")
ax1 = ax.twinx()
ax.invert_xaxis()
ax1.semilogx([rtolC/rtolFine for rtolC in rtolCoarse], time_cost, color="red", label = "cost")
fig.legend()

fig, ax = plt.subplots()
ax.semilogx([rtolC/rtolFine for rtolC in rtolCoarse], np.max(err_t[:,1:], axis=1), label = r"$\|t_{coarse}-t_{fine}\|_{\infty}$")
ax1 = ax.twinx()
ax.invert_xaxis()
ax1.semilogx([rtolC/rtolFine for rtolC in rtolCoarse], time_cost, color="red", label = "cost")
fig.legend()

fig, ax = plt.subplots()
ax.semilogx([rtolC/rtolFine for rtolC in rtolCoarse], np.max(err_u_diff[:,1:], axis=1), label = r"$\|u_{diff,coarse}-u_{diff,fine}\|_{\infty}$")
ax1 = ax.twinx()
ax.invert_xaxis()
ax1.semilogx([rtolC/rtolFine for rtolC in rtolCoarse], time_cost, color="red", label = "cost")
fig.legend()

plt.show()
