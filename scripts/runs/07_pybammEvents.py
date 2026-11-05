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

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import copy


# Script parameters
half_cell = True
exp = "CCCV" # GITT or CCCV
nCycles = 1
N = 3
K = N+1
# Space discretization
var_pts = {
    "x_n": 1,   # points in the negative electrode
    "x_s": 30,  # points in the separator
    "x_p": 30,  # points in the positive electrode
    "r_p": 30  # points in the radius of positive electrode
}

rtolCoarse = 1e-3
rtolFine = 1e-8

fine_propagator = PybammPropagator(
    model = pybamm.lithium_ion.DFN(*({"working electrode": "positive"},) if half_cell else ()),
    experiment = batteryExperiment(nCycles, expType=exp),
    parameter_values = pybamm.ParameterValues(bpar.Li_half.PARAMS if half_cell else bpar.Li_full.PARAMS),
    var_pts = var_pts,
    solver=pybamm.IDAKLUSolver(rtol=rtolFine))

tStart = 0
uStart = copy.deepcopy(fine_propagator.u_full)

len_rhs = fine_propagator._last_step_model.len_rhs

coarse_propagator = PybammPropagator(
        model = pybamm.lithium_ion.DFN(*({"working electrode": "positive"},) if half_cell else ()),
        experiment = batteryExperiment(nCycles, expType=exp),
        parameter_values = pybamm.ParameterValues(bpar.Li_half.PARAMS if half_cell else bpar.Li_full.PARAMS),
        var_pts = var_pts,
        solver=pybamm.IDAKLUSolver(rtol=rtolCoarse)
        )

coarse_s_propagator = PybammPropagator(
        model = pybamm.lithium_ion.DFN(*({"working electrode": "positive"},) if half_cell else ()),
        experiment = batteryExperiment(nCycles, expType=exp),
        parameter_values = pybamm.ParameterValues(bpar.Li_half.PARAMS if half_cell else bpar.Li_full.PARAMS),
        var_pts = var_pts,
        solver=pybamm.IDAKLUSolver(rtol=rtolCoarse)
        )

TT          =   np.zeros((K+1, N+1), dtype = float)
TT[:,0]     =   tStart

U           =   np.empty((K+1, N+1), dtype = object)  

# def parareal_state(k, n):
#     return self.make_state(TT[k, n], U[k, n], n)

for k in range(K+1):
    U[k, 0]     =   copy.deepcopy(uStart)

print("Initialization completed.")

# First coarse propagation
print("Starting first coarse propagation...")
for n in range(N):
    TPG, UPG    =  coarse_propagator.propagate(TT[0,n], U[0,n], cycle = n)
    TT[0,n+1]           =   TPG
    U[0, n+1]           =   copy.deepcopy(UPG)

print("First coarse propagation completed.")
print("Starting Parareal iterations...")
# Parareal iterations
for k in range(K):  
    print("Iteration", k)

    for n in range(N):

        TPF, UPF     =    fine_propagator.propagate(TT[k,n], copy.deepcopy(U[k,n]), cycle = n) 
        TPG, UPG     =    coarse_propagator.propagate(TT[k,n], copy.deepcopy(U[k,n]), cycle = n)
        TPG_s, UPG_s =    coarse_s_propagator.propagate(TT[k+1,n], copy.deepcopy(U[k+1,n]), cycle = n)
        
      
        print("Jacobi approximation", np.linalg.norm(UPF[0][:len_rhs] - UPG[0][:len_rhs], ord=np.inf))
        print("Diff coarse sequential fine", np.linalg.norm(UPF[0][:len_rhs] - UPG_s[0][:len_rhs], ord=np.inf))
        
        print("Duration fine:" , TPF - TT[k,n])
        print("Duration coarse:", TPG - TT[k,n])
        print("Difference in time fine coarse:", TPF - TPG)
        TT[k+1, n+1] =    TPF - TPG + TPG_s 
        U[k+1, n+1]  =    UPF - UPG + UPG_s   
        print("Difference between parareal and fine:", TPF-TT[k+1,n+1])
        print("Diff coarse parareal times: ", TPG - TT[k+1,n+1])
    print("\n")
#%%

# exact numerical solution
exact_propagator = PybammPropagator(
    model = pybamm.lithium_ion.DFN(*({"working electrode": "positive"},) if half_cell else ()),
    experiment = batteryExperiment(nCycles, expType=exp),
    parameter_values = pybamm.ParameterValues(bpar.Li_half.PARAMS if half_cell else bpar.Li_full.PARAMS),
    var_pts = var_pts,
    solver=pybamm.IDAKLUSolver(rtol=rtolFine))
solutionFine = np.empty(N+1, dtype=object)
tFine_cycles = np.zeros(N+1, dtype=float)
solutionFine[0] = uStart
for cycle in range(N):
    # fine propagator
    tFine, solFine = exact_propagator.propagate(
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
fig.legend(loc = "center")

ax_right.semilogy(range(K+1), err_uDiffPararealmax, label=r'$\|u_{Parareal} - u_{FineNum}\|_\infty$')
ax_twin = ax_right.twinx()
ax_right.set_yscale('symlog', linthresh=1e-14)
ax_twin.semilogy(range(K+1), err_upPararealmax,'--', label=r'$\|up_{Parareal} - up_{FineNum}\|_\infty$')
ax_right.set_xlabel("Parareal iteration k"), ax_right.set_ylabel("Error"), ax_right.grid();
fig.legend(loc = "center")  
plt.tight_layout()  
plt.show()