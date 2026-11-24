#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 12:25:18 2026

@author: yessima
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple

from batpint.parareal.errors import compute_parareal_errors
from batpint.parareal.parareal import PararealModified
from batpint.parareal.plot import plot_error_per_iteration
from batpint.problems.battery import make_propagator

results_dir = Path.cwd().parent / "results" / "test_event"
results_dir.mkdir(parents=True, exist_ok=True)

# %%
# Script parameters
half_cell = True
exp = "CCCV"  # GITT or CCCV
nCycles = 5  # number of Parareal windows, 1 battery cycle per window
nIterations = nCycles  # number of Parareal iterations
# Space discretization
var_pts = {
    "x_n": 1,  # points in the negative electrode
    "x_s": 10,  # points in the separator
    "x_p": 10,  # points in the positive electrode
    "r_p": 10,  # points in the radius of positive electrode
}

# set tolerances
rtolFine = 1e-5
rtolCoarse = 1e-5
solver_options = {
    # "linear_solver":"SUNLinSol_Dense",
    # "jacobian":"dense",
    "max_order_bdf": 5
}

# define propagators
fine_propagator = make_propagator(
    half_cell=half_cell,
    exp=exp,
    var_pts=var_pts,
    rtol=rtolFine,
    solver_options=solver_options,
)


# set initial conditions
tStart = 0
uStart = fine_propagator.u_full.copy()

# -- check only differential equations variables 0:len_rhs
len_rhs = fine_propagator._last_step_model.len_rhs

# %%
# exact numerical solution
UFine = np.empty((nCycles + 1, *uStart.shape), dtype=uStart.dtype)
tFine_cycles = np.zeros(nCycles + 1)
UFine[0] = uStart

for cycle in range(nCycles):
    tFine_cycles[cycle + 1], UFine[cycle + 1] = fine_propagator.propagate(t=tFine_cycles[cycle], u=UFine[cycle], cycle=cycle)
UUFine =  UFine[:,:,0]
UFine_diff = UFine[:, :len_rhs, 0]
UFine_alg = UFine[:, len_rhs:, 0]

# define small perturbation in state and time solution
eps_times = [0]
eps_states = 1 / 10 ** np.arange(6, 0, -1)

# solve Parareal
try:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, sharex=True, layout="constrained")

    markers = ["o", "s", "^", "v", "D", "*", "+", "x", "p", "8"]
    labels_state = []
    iteration_handles = []
    component_handles = None
    for idxT, eps_time in enumerate(eps_times):
        for idxS, eps_state in enumerate(eps_states):
            coarse_propagator = make_propagator(
                half_cell=half_cell,
                exp=exp,
                var_pts=var_pts,
                rtol=rtolCoarse,
                solver_options=solver_options,
                eps_state=eps_state,
                eps_time=eps_time,
            )

            parareal = PararealModified(fine=fine_propagator, coarse=coarse_propagator, coarse_s=coarse_propagator)
            TT, U = parareal.solve(t0=tStart, u0=uStart, K=nIterations, N=nCycles)

            # extract differential state
            UU = U[...,0]
            Udiff = U[:, :, :len_rhs, 0]
            Ualg = U[:, :, len_rhs:, 0]

            # compute error
            errTT = compute_parareal_errors(TT, tFine_cycles)
            errUU = compute_parareal_errors(UU, UUFine)
            errU_diff = compute_parareal_errors(Udiff, UFine_diff)
            errU_alg = compute_parareal_errors(Ualg, UFine_alg)

            errTT_max = np.max(errTT, axis=1)  # error in event times for the Parareal solver
            errUU_max = np.max(errUU, axis=1)
            errU_diff_max = np.max(errU_diff, axis=1)
            errU_alg_max = np.max(errU_alg, axis=1)
            # plot error per cycle
            labels_state.append(
                r"$\epsilon_{state, diff}=$"
                + f"{eps_state}"
                + "\n"
                + r"$\epsilon_{state, alg}=$"
                + f"{eps_state * 1e-3}"
            )
            plot_error_per_iteration(
                ax_left,
                ax_right,
                errTT_max,
                errUU_max,
                nIterations,
                nCycles,
                marker=markers[idxS],
                linewidth=1,
                left_style={"color": "red"},
                right_style={"color": "blue"},
            )
            iteration_handles.append((ax_left.lines[-1], ax_right.lines[-1]))
            diff_line, = ax_right.plot(
                errU_diff_max,
                "--",
                marker=markers[idxS],
                color="blue",
                alpha=0.5,
            )
            alg_line, = ax_right.plot(
                errU_alg_max,
                ":",
                marker=markers[idxS],
                color="blue",
                alpha=0.5,
            )
            if component_handles is None:
                component_handles = [diff_line, alg_line]

    fig.legend(
        iteration_handles,
        labels_state,
        loc="outside upper center",
        ncols=4,
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )
    ax_right.legend(component_handles, ["differential error", "algebraic error"], loc="best")
    fig.suptitle(f"rtol_coarse {rtolCoarse}, rtol_fine {rtolFine}")
    plt.draw()
except Exception as e:
    print(f"Parareal execution failed: {e}")
    raise
plt.show()
# %%
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
