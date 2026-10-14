#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:32:00 2026
@author: yessima
"""
import pybamm
import numpy as np
import batpint.parameter.javid as bpar
from batpint.problems.battery import batteryExperiment
from batpint.parareal.pybamm_propagator import PybammPropagator
from batpint.parareal.parareal import PararealModified

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import copy

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

# Parareal
N = 4 # number of cycles
K = N+1 # number of iterations

rtolCoarse = 1e-5
atolCoarse = 1e-6

rtolFine = 1e-6
atolFine = 1e-6

fine_propagator = PybammPropagator(
    model = pybamm.lithium_ion.DFN(*({"working electrode": "positive"},) if half_cell else ()),
    experiment = batteryExperiment(nCycles, expType=exp),
    parameter_values = pybamm.ParameterValues(bpar.Li_half.PARAMS if half_cell else bpar.Li_full.PARAMS),
    var_pts = var_pts,
    solver=pybamm.IDAKLUSolver(rtol=rtolFine, atol=atolFine))

tStart = 0
uStart = copy.deepcopy(fine_propagator.u_full)

len_rhs = fine_propagator._last_step_model.len_rhs

coarse_propagator = PybammPropagator(
        model = pybamm.lithium_ion.DFN(*({"working electrode": "positive"},) if half_cell else ()),
        experiment = batteryExperiment(nCycles, expType=exp),
        parameter_values = pybamm.ParameterValues(bpar.Li_half.PARAMS if half_cell else bpar.Li_full.PARAMS),
        var_pts = var_pts,
        solver=pybamm.IDAKLUSolver(rtol=rtolCoarse, atol=atolCoarse)
        )

parareal = PararealModified(fine=fine_propagator, coarse=coarse_propagator)
try:
    TT, U = parareal.solve(t0=tStart, u0=uStart, K=K, N=N)
except Exception as e:
    print(f"Parareal execution failed: {e}")

#%%
# exact numerical solution
solutionFine = np.empty(N+1, dtype=object)
tFine_cycles = np.zeros(N+1, dtype=float)
solutionFine[0] = uStart
for cycle in range(N):
    # fine propagator
    tFine, solFine = fine_propagator.propagate(
        t=tFine_cycles[cycle],
        u=solutionFine[cycle],
        cycle=cycle)    
    tFine_cycles[cycle+1] = tFine
    solutionFine[cycle+1] = solFine
    
# error at the event points for the Parareal solver
err_tParareal = np.zeros((K+1, N+1), dtype=float) # error in event times per cycle
err_uParareal = np.zeros((K+1, N+1), dtype=object) # error in event values per cycle
err_uDiffParareal = np.zeros((K+1, N+1), dtype=object) # error in event values per cycle
err_uAlgParareal = np.zeros((K+1, N+1), dtype=object)
err_upParareal = np.zeros((K+1, N+1), dtype=object)
for k in range(K+1):
    for cycle in range(N+1):
        err_tParareal[k, cycle] = np.abs(TT[k, cycle] - tFine_cycles[cycle])
        err_uParareal[k, cycle] = np.linalg.norm(U[k, cycle][0] - solutionFine[cycle][0], ord=np.inf)
        err_uDiffParareal[k, cycle] = np.linalg.norm(U[k, cycle][0][:len_rhs] - solutionFine[cycle][0][:len_rhs], ord=np.inf)
        err_uAlgParareal[k, cycle] = np.linalg.norm(U[k, cycle][0][len_rhs:] - solutionFine[cycle][0][len_rhs:], ord=np.inf)
        err_upParareal[k, cycle] = np.linalg.norm(U[k, cycle][1] - solutionFine[cycle][1], ord=np.inf)

#%%
err_tPararealmax = np.max(err_tParareal, axis=1) # error in event times for the Parareal solver
err_uPararealmax = np.max(err_uParareal, axis=1) # error in event values for the Parareal solver
err_uDiffPararealmax = np.max(err_uDiffParareal, axis=1) 
err_upPararealmax = np.max(err_upParareal, axis=1)
# %%
fig, (ax_left, ax_right) = plt.subplots(1,2, figsize=(12, 8), sharex=True)
fig.suptitle(f"Adaptive Parareal Error for {N+1} cycles")

ax_left.semilogy(range(K+1), err_tPararealmax, label=r'$\|t_{Parareal} - t_{FineNum}\|_\infty$', color="red")
ax_left.set_yscale('symlog', linthresh=1e-14)
ax_left.set_xlabel("Parareal iteration k"), ax_left.set_ylabel("Error"), ax_left.grid();
ax_left.set_xticks(range(K+1))

ax_right.semilogy(range(K+1), err_uDiffPararealmax, label=r'$\|u_{Parareal} - u_{FineNum}\|_\infty$')
ax_twin = ax_right.twinx()
ax_right.set_yscale('symlog', linthresh=1e-14)
ax_twin.semilogy(range(K+1), err_upPararealmax,'--', label=r'$\|up_{Parareal} - up_{FineNum}\|_\infty$')
ax_right.set_xlabel("Parareal iteration k"), ax_right.set_ylabel("Error"), ax_right.grid();
fig.legend(loc = "center")  
plt.tight_layout()  
plt.show()

#%%
# plotting the error vs N points
fig = plt.figure(figsize=(12, 8))
fig.suptitle(f"Adaptive Parareal Error for {N+1} cycles")
for k in range(K+1):
    plt.semilogy(range(N+1), err_tParareal[k,:], label=f"k={k}")
plt.xticks(range(N+1))
plt.yscale('symlog', linthresh=1e-14)
plt.xlabel("Cycles"), plt.ylabel("Error"), plt.grid();
plt.legend()

fig, ax = plt.subplots(figsize=(12, 8), sharex=True)
fig.suptitle(f"Adaptive Parareal Error for {N+1} cycles")
ax1 = ax.twinx()
for k in range(K+1):
    ax.semilogy(range(N+1), err_uDiffParareal[k,:], label=f"k={k}")
    ax.semilogy(range(N+1), err_upParareal[k,:],'--', label=f"k={k}")

ax.set_xticks(range(N+1))
ax.set_yscale('symlog', linthresh=1e-14)
ax.set_xlabel("Cycles"), plt.ylabel("Error"), plt.grid();
legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label=r'$\|u_{Parareal} - u_{FineNum}\|_\infty$'),
    Line2D([0], [0], color='black', linestyle='--', label=r'$\|up_{Parareal} - up_{FineNum}\|_\infty$')
]
ax.legend(handles = legend_elements)
plt.tight_layout()  
plt.show()