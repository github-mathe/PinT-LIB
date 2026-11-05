#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 12:25:18 2026

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
import json
from pathlib import Path
import copy

results_dir = Path.cwd().parent / "results" / "test_event"
results_dir.mkdir(parents=True, exist_ok=True)

#%%
# Script parameters
half_cell = True
exp = "CCCV" # GITT or CCCV
nCycles = 1
N = 5
K = N
# Space discretization
var_pts = {
            "x_n": 1,   # points in the negative electrode
            "x_s": 10,  # points in the separator
            "x_p": 10,  # points in the positive electrode
            "r_p": 30  # points in the radius of positive electrode
            }

# set tolerances
rtolFine = 1e-7
rtolCoarse = 1e-7

# define propagatoros
fine_propagator = PybammPropagator(
                    model=pybamm.lithium_ion.DFN(*({"working electrode": "positive"},) if half_cell else ()),
                    experiment=batteryExperiment(nCycles, expType=exp),
                    parameter_values=pybamm.ParameterValues(bpar.Li_half.PARAMS if half_cell else bpar.Li_full.PARAMS),
                    var_pts=var_pts,
                    solver=pybamm.IDAKLUSolver(rtol=rtolFine)
                    )


# set initial conditions
tStart = 0
uStart = copy.deepcopy(fine_propagator.u_full)
len_rhs = fine_propagator._last_step_model.len_rhs # check only differential equations variables 0:len_rhs

#%%
# exact numerical solution
exact_propagator = PybammPropagator(
                    model=pybamm.lithium_ion.DFN(*({"working electrode": "positive"},) if half_cell else ()),
                    experiment=batteryExperiment(nCycles, expType=exp),
                    parameter_values=pybamm.ParameterValues(bpar.Li_half.PARAMS if half_cell else bpar.Li_full.PARAMS),
                    var_pts=var_pts,
                    solver=pybamm.IDAKLUSolver(rtol=rtolFine)
                    )

solutionFine = np.empty(N+1, dtype=object)
tFine_cycles = np.zeros(N+1, dtype=float)
solutionFine_diff = np.empty(solutionFine.shape, dtype=object)
solutionFine[0] = uStart
solutionFine_diff[0] = uStart[0][:len_rhs]

for cycle in range(N):
    tFine, solFine = exact_propagator.propagate(
                                                t=tFine_cycles[cycle],
                                                u=solutionFine[cycle],
                                                cycle=cycle
                                                )    
    tFine_cycles[cycle+1] = tFine
    solutionFine[cycle+1] = solFine
    solutionFine_diff[cycle+1] = solFine[0][:len_rhs]


# define small perturbation in state and time solution
eps_times = [0]
eps_states = [1e-3] #1/10**np.arange(12, 11,-1)

# solve Parareal
try:
    for idxT, eps_time in enumerate(eps_times):
        for idxS, eps_state in enumerate(eps_states):
            
            coarse_propagator = PybammPropagator(
                                                model=pybamm.lithium_ion.DFN(*({"working electrode": "positive"},) if half_cell else ()),
                                                experiment=batteryExperiment(nCycles, expType=exp),
                                                parameter_values=pybamm.ParameterValues(bpar.Li_half.PARAMS if half_cell else bpar.Li_full.PARAMS),
                                                var_pts=var_pts,
                                                solver=pybamm.IDAKLUSolver(rtol=rtolCoarse),
                                                eps_state=eps_state,
                                                eps_time=eps_time
                                                )

            parareal = PararealModified(fine=fine_propagator, coarse=coarse_propagator, coarse_s=coarse_propagator)
            TT, U = parareal.solve(t0=tStart, u0=uStart, K=K, N=N)
            
            # extract differential state
            Udiff = np.empty(U.shape, dtype=object)
            for k in range(U.shape[0]):
                for cycle in range(U.shape[1]):
                    Udiff[k, cycle] = U[k, cycle][0][:len_rhs]
            
            # compute error 
            errTT = compute_parareal_errors(TT, tFine_cycles)
            errU_diff = compute_parareal_errors(Udiff, solutionFine_diff)
            errTT_max = np.max(errTT, axis=1) # error in event times for the Parareal solver
            errU_diff_max = np.max(errU_diff, axis=1)
except Exception as e:
        print(f"Parareal execution failed: {e}")

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