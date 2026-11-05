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
from batpint.parareal.errors import compute_parareal_errors
from batpint.parareal.plot import plot_error_per_iteration, plot_error_per_cycle
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
    "x_s": 10,  # points in the separator
    "x_p": 10,  # points in the positive electrode
    "r_p": 30  # points in the radius of positive electrode
}

# Parareal
N = 5 # number of cycles
K = N+1 # number of iterations

rtolCoarse = 1e-9
rtolFine = 1e-10

# define the fine and coarse propagators
fine_propagator = PybammPropagator(
    model=pybamm.lithium_ion.DFN(*({"working electrode": "positive"},) if half_cell else ()),
    experiment=batteryExperiment(nCycles, expType=exp),
    parameter_values=pybamm.ParameterValues(bpar.Li_half.PARAMS if half_cell else bpar.Li_full.PARAMS),
    var_pts=var_pts,
    solver=pybamm.IDAKLUSolver(rtol=rtolFine))

coarse_propagator = PybammPropagator(
        model=pybamm.lithium_ion.DFN(*({"working electrode": "positive"},) if half_cell else ()),
        experiment=batteryExperiment(nCycles, expType=exp),
        parameter_values=pybamm.ParameterValues(bpar.Li_half.PARAMS if half_cell else bpar.Li_full.PARAMS),
        var_pts=var_pts,
        solver=pybamm.IDAKLUSolver(rtol=rtolCoarse)
        )

# initial conditions
tStart = 0
uStart = copy.deepcopy(fine_propagator.u_full)
len_rhs = fine_propagator._last_step_model.len_rhs

# exact numerical solution
UFine = np.empty(N+1, dtype=object)
UFine_diff = np.empty(N+1, dtype=object)
TFine = np.zeros(N+1, dtype=float)
UFine[0] = uStart
UFine_diff[0] = uStart[0][:len_rhs]
for cycle in range(N):
    # fine propagator
    tFine, solFine = fine_propagator.propagate(
        t=TFine[cycle],
        u=UFine[cycle],
        cycle=cycle)    
    TFine[cycle+1] = tFine
    UFine[cycle+1] = solFine
    UFine_diff[cycle+1] = solFine[0][:len_rhs]

# solve parareal
parareal = PararealModified(fine=fine_propagator, coarse=coarse_propagator, coarse_s=coarse_propagator)
try:
    TT, U = parareal.solve(t0=tStart, u0=uStart, K=K, N=N)
except Exception as e:
    print(f"Parareal execution failed: {e}")

# extract differential state
Udiff = np.empty(U.shape, dtype=object)
# we don't analyze the derivatives of U: UP
# because they are recomputed with every solver step
# Tested when use same U but different UP, results are the same, so we can ignore UP
# UPdiff = np.empty(U.shape, dtype=object)
for k in range(U.shape[0]):
    for cycle in range(U.shape[1]):
        Udiff[k, cycle] = U[k, cycle][0][:len_rhs]

# error at the event points for the Parareal solver
errTT = compute_parareal_errors(TT,TFine)
errU_diff = compute_parareal_errors(Udiff,UFine_diff)
errU_diff_max = np.max(errU_diff, axis=1) # error in event times for the Parareal solver
errTT_max = np.max(errTT, axis=1) # error in event times for the Parareal solver

# plot error
# plot error per iteration
fig, (ax_left, ax_right) = plt.subplots(1,2, figsize=(12, 8), sharex=True)
ax_left, ax_right = plot_error_per_iteration(ax_left, ax_right, errTT_max, errU_diff_max, K, N)
ax_left.legend(labels=["time error"], loc="upper right")
ax_right.legend(labels=["state error"], loc="upper right") 

# plot error per cycle
fig, (ax_left, ax_right) = plt.subplots(1,2, figsize=(12, 8), sharex=True)
ax_left, ax_right = plot_error_per_cycle(ax_left, ax_right, errTT, errU_diff, K, N)
fig.legend(handles=ax_left.lines, loc="upper center", ncol=K+1)
plt.tight_layout()  
plt.show()