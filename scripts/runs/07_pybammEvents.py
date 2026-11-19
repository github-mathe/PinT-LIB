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
from matplotlib.legend_handler import HandlerTuple

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
rtolFine = 1e-5
rtolCoarse = 1e-5
solver_options = {
    #"linear_solver":"SUNLinSol_Dense", 
    #"jacobian":"dense", 
    "max_order_bdf":1}

# define propagatoros
fine_propagator = PybammPropagator(
                    model=pybamm.lithium_ion.DFN(*({"working electrode": "positive"},) if half_cell else ()),
                    experiment=batteryExperiment(nCycles, expType=exp),
                    parameter_values=pybamm.ParameterValues(bpar.Li_half.PARAMS if half_cell else bpar.Li_full.PARAMS),
                    var_pts=var_pts,
                    solver=pybamm.IDAKLUSolver(rtol=rtolFine, options=solver_options)
                    )


# set initial conditions
tStart = 0
uStart = copy.deepcopy(fine_propagator.u_full)
len_rhs = fine_propagator._last_step_model.len_rhs # check only differential equations variables 0:len_rhs
state_size = fine_propagator._state_size

#%%
# exact numerical solution
UFine = np.empty(N+1, dtype=object)
UUFine = np.empty(N+1, dtype=object)
tFine_cycles = np.zeros(N+1, dtype=float)
UFine_diff = np.empty(UFine.shape, dtype=object)
UFine_alg = np.empty(UFine.shape, dtype=object)
UFine[0] = uStart
UUFine[0] = uStart[:state_size]
UFine_diff[0] = uStart[:state_size][:len_rhs]
UFine_alg[0] = uStart[:state_size][len_rhs:]

for cycle in range(N):
    tFine, solFine = fine_propagator.propagate(
                                                t=tFine_cycles[cycle],
                                                u=UFine[cycle],
                                                cycle=cycle
                                                )    
    tFine_cycles[cycle+1] = tFine
    UFine[cycle+1] = solFine
    UUFine[cycle+1] = solFine[:state_size]
    UFine_diff[cycle+1] = solFine[:state_size][:len_rhs]
    UFine_alg[cycle+1] = solFine[:state_size][len_rhs:]

# define small perturbation in state and time solution
eps_times = [0]

eps_states = 1/10**np.arange(6, 1,-1)

# solve Parareal
try:
    fig, (ax_left, ax_right) = plt.subplots(1,2, sharex=True, layout="constrained")
    
    markers = ['o', 's', '^', 'v', 'D', '*', '+', 'x', 'p', '8']
    labels_state=[]
    for idxT, eps_time in enumerate(eps_times):
        for idxS, eps_state in enumerate(eps_states):
            
            coarse_propagator = PybammPropagator(
                                                model=pybamm.lithium_ion.DFN(*({"working electrode": "positive"},) if half_cell else ()),
                                                experiment=batteryExperiment(nCycles, expType=exp),
                                                parameter_values=pybamm.ParameterValues(bpar.Li_half.PARAMS if half_cell else bpar.Li_full.PARAMS),
                                                var_pts=var_pts,
                                                solver=pybamm.IDAKLUSolver(rtol=rtolCoarse, options=solver_options),
                                                eps_state=eps_state,
                                                eps_time=eps_time
                                                )

            parareal = PararealModified(fine=fine_propagator, coarse=coarse_propagator, coarse_s=coarse_propagator)
            TT, U = parareal.solve(t0=tStart, u0=uStart, K=K, N=N)
            
            # extract differential state
            UU = np.empty(U.shape, dtype=object)
            Udiff = np.empty(U.shape, dtype=object)
            Ualg = np.empty(U.shape, dtype=object)
            for k in range(U.shape[0]):
                for cycle in range(U.shape[1]):
                    UU[k, cycle] = U[k, cycle][:state_size]
                    Udiff[k, cycle] = U[k, cycle][:state_size][:len_rhs]
                    Ualg[k, cycle] = U[k, cycle][:state_size][len_rhs:]
            # compute error 
            errTT = compute_parareal_errors(TT, tFine_cycles)
            errUU = compute_parareal_errors(UU, UUFine)
            errU_diff = compute_parareal_errors(Udiff, UFine_diff)
            errU_alg = compute_parareal_errors(Ualg, UFine_alg)
            
            errTT_max = np.max(errTT, axis=1) # error in event times for the Parareal solver
            errUU_max = np.max(errUU, axis=1)
            errU_diff_max = np.max(errU_diff, axis=1)
            errU_alg_max = np.max(errU_alg, axis=1)
            
            # plot error per cycle
            labels_state.append(r"$\epsilon_{state, diff}=$" + f"{eps_state}"+"\n" 
                                + r"$\epsilon_{state, alg}=$" + f"{eps_state*1e-3}")
            plot_error_per_iteration(
                                    ax_left, 
                                    ax_right, 
                                    errTT_max, 
                                    errUU_max, 
                                    K, 
                                    N, 
                                    marker=markers[idxS], 
                                    linewidth=1, 
                                    left_style={"color":"red"},
                                    right_style={"color":"blue"})
            
    handles = list(zip(ax_left.lines, ax_right.lines))
    fig.legend(
                handles,
                labels_state, 
                loc='outside lower center', 
                ncols=5,
                handler_map={tuple: HandlerTuple(ndivide=None)},
                )
    ax_right.plot(errU_diff_max, '--', marker=markers[idxS], color="blue", alpha=0.3)
    ax_right.plot(errU_alg_max, ':', marker=markers[idxS], color="blue", alpha=0.3)
    fig.suptitle(f"rtol_coarse {rtolCoarse}, rtol_fine {rtolFine}")
    plt.draw()
except Exception as e:
        print(f"Parareal execution failed: {e}")
plt.show()
#%%
# plot error
# plot error per iteration
# fig, (ax_left, ax_right) = plt.subplots(1,2, figsize=(12, 8), sharex=True)
# ax_left, ax_right = plot_error_per_iteration(ax_left, ax_right, errTT_max, errU_diff_max, K, N)
# ax_left.legend(labels=["time error"], loc="upper right")
# ax_right.legend(labels=["state error"], loc="upper right") 

# # plot error per cycle
# fig, (ax_left, ax_right) = plt.subplots(1,2, figsize=(12, 8), sharex=True)
# ax_left, ax_right = plot_error_per_cycle(ax_left, ax_right, errTT, errU_diff, K, N)
# fig.legend(handles=ax_left.lines, loc="upper center", ncol=K+1)
# plt.tight_layout()  
# plt.show()