#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:17:01 2026

@author: yessima
"""

# check sundials solvers on ODE for parareal

import matplotlib.pyplot as plt
import numpy as np
import pybamm

from batpint.parareal.base_propagator import Propagator
from batpint.parareal.errors import compute_parareal_errors
from batpint.parareal.parareal import PararealModified
from batpint.parareal.plot import plot_error_per_cycle, plot_error_per_iteration
from batpint.problems.dahlquist import Dahlquist


#%%
# construct pybamm Dahlquist model
def pybammDahlquist(uStart, lam, alpha):
    model = pybamm.BaseModel()

    # variables
    uIm = pybamm.Variable("uIm")
    uRe = pybamm.Variable("uRe")

    # parameters
    cycle = pybamm.InputParameter("cycle")
    lam = pybamm.FunctionParameter(name="lambda", inputs={"cycle":cycle})
    alpha = pybamm.Parameter("alpha")
    t = pybamm.t

    # equations
    model.rhs = {uIm: lam * uRe, uRe: -uIm*lam + pybamm.sin(alpha*t) } #λu + sin(αt)
    model.variables = {"uIm": uIm, "uRe": uRe}
    model.initial_conditions = {uIm: np.imag(uStart), uRe: np.real(uStart)}
    t_gate = pybamm.InputParameter("t_gate")

    model.events.append(
        pybamm.Event(
            "Second zero crossing",
            pybamm.maximum(t_gate - pybamm.t, -uIm),
            pybamm.EventType.TERMINATION,
        )
    )
    param = pybamm.ParameterValues({"lambda": lam_function,
                                    "alpha": alpha_value})
    disc = pybamm.Discretisation()
    param.process_model(model)
    disc.process_model(model)
    return model

# create a propagator class for the Dahlquist problem
class MyPropagator(Propagator):
    def __init__(
        self,
        model,
        solver,
        inputs=None,
    ):
        self.model = model
        self.solver = solver
        self.inputs = {} if inputs is None else dict(inputs)

    def propagate(self, t, u, cycle):

        starting_solution = pybamm.Solution(
            all_ts=np.array([t], copy=True),
            all_ys=np.array(u,copy=True),
            all_models = [self.model],
            all_inputs = [{"cycle": cycle-1, "t_gate": .5}],
            t_event=None,
            y_event=None,
            all_t_evals=None)
        self.model.set_initial_conditions_from(starting_solution, inplace=True)
        solution = self.solver.solve(
             self.model,
             t_eval=np.linspace(t, t+100),
             inputs={"cycle": cycle, "t_gate": t + 3.5,
                     })
        return solution.t_event[-1], solution.y_event

    def solve(self, t, u, cycle):

        starting_solution = pybamm.Solution(
                all_ts=np.array([t], copy=True),
                all_ys=np.array(u,copy=True),
                all_models = [self.model],
                all_inputs = [{"cycle": 0, "t_gate": .5}],
                t_event=None,
                y_event=None,
                all_t_evals=None)
        self.model.set_initial_conditions_from(starting_solution, inplace=True)
        solution = self.solver.solve(
             self.model,
             t_eval=np.linspace(t, t+100),
             inputs={"cycle": cycle, "t_gate": t + 3.5,
                     })
        return solution.t_event[-1], solution.y_event, solution

#%%
# define dahlquist problem
num_cycles = 2
t_start = 0
uStart = 1+0j
cycle_Start = 0 # event start cycle

# define lam and alpha
def lam_function(cycle):
    return (1 + 0.01*(cycle+cycle_Start))
alpha_value = 6.001   # to avoid resonance regime

# Dahlquist problem
def event(t, u, cycle, u_event):
    return np.imag(u) - np.imag(u_event)
# dahlquist problem
problem = Dahlquist(t_start=t_start, u_start=uStart, lam=lambda cycle: 1j*lam_function(cycle), alpha=alpha_value, event=event)

# exact solution
dtEx = 1e-3
tTh, uTh, tTh_events, uTh_events = problem.solve_dahlquist_exact_cycles(num_cycles, dt=dtEx)

# pybamm model
model = pybammDahlquist(uStart, lam_function, alpha_value)
# solver
ode_solver = pybamm.IDAKLUSolver(rtol=1e-7)

# compare MyPropagate.propagate solving with the exact solution
model = pybammDahlquist(1+0j, lam_function, alpha_value)
propagator = MyPropagator(model=model, solver=pybamm.IDAKLUSolver(rtol=1e-4))

t = 0
u = np.array([[0],[1.]])

fig,ax = plt.subplots()
for cycle in range(num_cycles):
    tEv, uEv, solution = propagator.solve(t, u, cycle)
    t=tEv.copy()
    u=uEv.copy()
    ax.plot(tTh, np.imag(uTh), 'g-',label=r"$\Im(u)$ exact" if cycle==0 else "")
    ax.plot(solution.t, solution.y[0,:], 'go', label=r"$\Im(u)$ PyBaMM" if cycle==0 else "", markersize=3)

    ax1 = ax.twinx()
    ax1.plot(tTh, np.real(uTh), 'r-', label=r"$\Re(u)$ exact" if cycle==0 else "")
    ax1.plot(solution.t, solution.y[1,:], 'ro', label=r"$\Re(u)$ PyBaMM" if cycle==0 else "", markersize=3)

ax.set_xlabel("t"), ax.set_ylabel(r"$\Im(u)$"), ax.grid()
ax1.set_ylabel(r"$\Re(u)$")
fig.legend(loc="upper center", ncol=2)

# parareal
N=5
K=N+1

# create parareal propagators
modelF = pybammDahlquist(uStart, lam_function, alpha_value)
modelC = pybammDahlquist(uStart, lam_function, alpha_value)
fine_propagator = MyPropagator(model=modelF, solver=pybamm.IDAKLUSolver(rtol=1e-6))
coarse_propagator = MyPropagator(model=modelC, solver=pybamm.IDAKLUSolver(rtol=1e-4))

# compute fine solution for error analysis
# exact solution
solutionFine = np.empty(N+1, dtype=object)
tFine_cycles = np.zeros(N+1, dtype=float)
uFine_cycles = np.zeros(N+1, dtype=complex)
uFine_cycles[0] = 1+0j
solutionFine[0] = np.array([[0],[1]])
for cycle in range(N):
    # fine propagator
    tFine, solFine = fine_propagator.propagate(
        t=tFine_cycles[cycle],
        u=solutionFine[cycle],
        cycle=cycle)
    tFine_cycles[cycle+1] = tFine
    solutionFine[cycle+1] = solFine
    uFine_cycles[cycle+1] = solFine[0][0]*1j + solFine[1][0]

# parareal solver
parareal = PararealModified(fine=fine_propagator, coarse=coarse_propagator, coarse_s=coarse_propagator)
try:
    TT, U = parareal.solve(t0=0, u0=np.array([[0],[1]]), K=K, N=N)
except Exception as e:
    print(f"Parareal execution failed: {e}")

# construct complex solution array
U_complex = np.empty(U.shape, dtype=complex)
for k in range(U.shape[0]):
    for cycle in range(U.shape[1]):
        U_complex[k, cycle] = U[k, cycle][1][0] + 1j*U[k, cycle][0][0]

# compute the time and state errors at the event points for the Parareal solver
errTT = compute_parareal_errors(TT, tFine_cycles)
errU = compute_parareal_errors(U_complex, uFine_cycles)

# plot the errors per iteration
fig, (ax_left, ax_right) = plt.subplots(1,2, figsize=(12, 8), sharex=True)
ax_left, ax_right = plot_error_per_iteration(ax_left, ax_right, errTT[:,-1], errU[:,-1], K=K, N=N)
ax_left.legend(labels=["time error"], loc="upper right")
ax_right.legend(labels=["state error"], loc="upper right")

# plot the errors per cycle
fig, (ax_left, ax_right) = plt.subplots(1,2, figsize=(12, 8), sharex=True)
ax_left, ax_right = plot_error_per_cycle(ax_left, ax_right, errTT, errU, K=K, N=N)
fig.legend(handles=ax_left.lines, loc="upper center", ncol=K+1)
plt.tight_layout()
plt.show()
