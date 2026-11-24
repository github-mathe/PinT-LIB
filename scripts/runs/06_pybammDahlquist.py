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


# %%
# construct pybamm Dahlquist model
def pybammDahlquist(uStart, lam_f, alpha_v):
    """Construct and discretise the Dahlquist ODE model."""
    model = pybamm.BaseModel()

    # variables
    uIm = pybamm.Variable("uIm")
    uRe = pybamm.Variable("uRe")

    # parameters
    cycle = pybamm.InputParameter("cycle")
    lam = pybamm.FunctionParameter(name="lambda", inputs={"cycle": cycle})
    alpha = pybamm.Parameter("alpha")
    t = pybamm.t

    # equations
    model.rhs = {uIm: lam * uRe, uRe: -uIm * lam + pybamm.sin(alpha * t)}  # λu + sin(αt)
    model.variables = {"uIm": uIm, "uRe": uRe}
    model.initial_conditions = {uIm: np.imag(uStart), uRe: np.real(uStart)}
    event_imag = pybamm.InputParameter("event_imag")
    event_direction = pybamm.InputParameter("event_direction")
    event_margin = pybamm.InputParameter("event_margin")

    model.events.append(
        pybamm.Event(
            "Upward imaginary-level crossing",
            event_direction * (uIm - event_imag) + event_margin,
            pybamm.EventType.TERMINATION,
        )
    )
    param = pybamm.ParameterValues({"lambda": lam_f, "alpha": alpha_v})
    disc = pybamm.Discretisation()
    param.process_model(model)
    disc.process_model(model)
    return model


# create a propagator class for the Dahlquist problem
class MyPropagator(Propagator):
    """Propagate to an upward crossing of an imaginary level.

    With event_imag=None, each solve returns to its starting imaginary value.
    Supply a real event_imag to use a fixed target for every solve instead.
    """

    def __init__(
        self,
        model,
        solver,
        inputs=None,
        event_imag=None,
    ):
        self.model = model
        self.solver = solver
        self.inputs = {} if inputs is None else dict(inputs)
        self.event_imag = event_imag

    def _solve_to_event(self, t, u, cycle, t_end, event_imag, direction, margin):
        inputs = {
            **self.inputs,
            "cycle": cycle,
            "event_imag": event_imag,
            "event_direction": direction,
            "event_margin": margin,
        }

        starting_solution = pybamm.Solution(
            all_ts=np.array([t]),
            all_ys=np.asarray(u).reshape(2, 1),
            all_models=[self.model],
            all_inputs=[inputs],
            t_event=None,
            y_event=None,
            all_t_evals=None,
        )

        self.model.set_initial_conditions_from(starting_solution, inplace=True)

        solution = self.solver.solve(self.model, t_eval=np.linspace(t, t_end), inputs=inputs)
        if not solution.termination.startswith("event") or solution.t_event is None:
            raise RuntimeError(
                f"Dahlquist propagation did not reach the requested event at imaginary level {event_imag} "
                f"before t={t_end}."
            )
        return solution

    def solve(self, t, u, cycle):
        """Reach the next upward crossing of the chosen imaginary level."""
        u = np.asarray(u).reshape(2)
        level = u[0]
        if self.event_imag is not None:
            level = self.event_imag

        t_end = t + 100
        # Use a buffer ten times the solver's absolute tolerance.
        margin = 10 * np.max(self.solver.atol)

        if u[0] - level <= -margin:
            # Already below the target: find the upward crossing directly.
            solution = self._solve_to_event(t, u, cycle, t_end, level, direction=-1, margin=0)
        else:
            # First move below the target to avoid stopping at the initial point.
            below_target = self._solve_to_event(t, u, cycle, t_end, level, direction=1, margin=margin)

            # Then find the upward crossing, starting from that endpoint.
            t_below = below_target.t[-1]
            u_below = below_target.y[:, -1]
            upward_crossing = self._solve_to_event(t_below, u_below, cycle, t_end, level, direction=-1, margin=0)
            solution = below_target + upward_crossing

        return solution.t[-1], solution.y[:, -1], solution

    def propagate(self, t, u, cycle):
        t_event, u_event, _ = self.solve(t, u, cycle)
        return t_event, u_event


def main():
    """Run the Dahlquist comparison and Parareal plots."""
    # %%
    # define dahlquist problem
    num_cycles = 5
    t_start = 0
    uStart = 1. + 0.j
    initial_state = np.array([np.imag(uStart), np.real(uStart)], dtype=float)
    cycle_Start = 0  # event start cycle

    # define lam and alpha
    def lam_function(cycle):
        return 1 + 0.01 * (cycle + cycle_Start)

    alpha_value = 6.001  # to avoid resonance regime

    # Dahlquist problem
    def event(t, u, cycle, u_event):
        return np.imag(u) - np.imag(u_event)

    problem = Dahlquist(
        t_start=t_start, u_start=uStart, lam=lambda cycle: 1j * lam_function(cycle), alpha=alpha_value, event=event
    )

    # exact solution
    dtEx = 1e-4
    tTh, uTh, tTh_events, uTh_events = problem.solve_dahlquist_exact_cycles(num_cycles, dt=dtEx)

    # pybamm model
    model = pybammDahlquist(uStart, lam_f=lam_function, alpha_v=alpha_value)
    propagator = MyPropagator(model=model, solver=pybamm.IDAKLUSolver(rtol=1e-4))

    t = 0
    u = initial_state.copy()

    fig, ax = plt.subplots()
    for cycle in range(num_cycles):
        tEv, uEv, solution = propagator.solve(t, u, cycle)
        t = tEv.copy()
        u = uEv.copy()
        ax.plot(tTh, np.imag(uTh), "g-", label=r"$\Im(u)$ exact" if cycle == 0 else "")
        ax.plot(solution.t, solution.y[0, :], "go", label=r"$\Im(u)$ PyBaMM" if cycle == 0 else "", markersize=3)

        ax1 = ax.twinx()
        ax1.plot(tTh, np.real(uTh), "r-", label=r"$\Re(u)$ exact" if cycle == 0 else "")
        ax1.plot(solution.t, solution.y[1, :], "ro", label=r"$\Re(u)$ PyBaMM" if cycle == 0 else "", markersize=3)

    ax.set_xlabel("t"), ax.set_ylabel(r"$\Im(u)$"), ax.grid()
    ax1.set_ylabel(r"$\Re(u)$")
    fig.legend(loc="upper center", ncol=2)

    # %%
    # parareal
    N = 5
    K = N + 1
    u = initial_state.copy()
    t0 = 0.0

    # create parareal propagators
    modelF = pybammDahlquist(uStart, lam_f=lam_function, alpha_v=alpha_value)
    modelC = pybammDahlquist(uStart, lam_f=lam_function, alpha_v=alpha_value)

    solver_options = {
        # "linear_solver":"SUNLinSol_Dense",
        # "jacobian":"dense",
        "max_order_bdf": 5
    }
    atol = 1e-6

    fine_propagator = MyPropagator(
        model=modelF, solver=pybamm.IDAKLUSolver(rtol=1e-6, atol=atol, options=solver_options)
    )
    coarse_propagator = MyPropagator(
        model=modelC, solver=pybamm.IDAKLUSolver(rtol=1e-4, atol=atol, options=solver_options)
    )

    # compute fine solution for error analysis
    # exact solution
    solutionFine = np.empty((N + 1, *u.shape), dtype=u.dtype)
    tFine_cycles = np.zeros(N + 1, dtype=float)
    solutionFine[0] = u

    for cycle in range(N):
        # fine propagator
        tFine_cycles[cycle + 1], solutionFine[cycle + 1] = fine_propagator.propagate(
            t=tFine_cycles[cycle], u=solutionFine[cycle], cycle=cycle
        )
    uFine_cycles = solutionFine[:, 1] + 1j * solutionFine[:, 0]

    # parareal solver
    parareal = PararealModified(fine=fine_propagator, coarse=coarse_propagator, coarse_s=coarse_propagator)
    try:
        TT, U = parareal.solve(t0=t0, u0=u, K=K, N=N)
    except Exception as e:
        print(f"Parareal execution failed: {e}")

    # construct complex solution array
    U_complex = np.empty((K + 1, N + 1), dtype=complex)
    U_complex = U[:, :, 1] + 1j * U[:, :, 0]

    # compute the time and state errors at the event points for the Parareal solver
    errTT = compute_parareal_errors(TT, tFine_cycles)
    errU_real = compute_parareal_errors(U_complex.real, uFine_cycles.real)
    errU_imag = compute_parareal_errors(U_complex.imag, uFine_cycles.imag)
    errU_complex = compute_parareal_errors(U_complex, uFine_cycles)

    # plot the errors per iteration
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 8), sharex=True)
    ax_left, ax_right = plot_error_per_iteration(
        ax_left,
        ax_right,
        errTT[:, -1],
        errU_complex[:, -1],
        K=K,
        N=N,
        left_style={"label": "Time error"},
        right_style={"label": "Complex state error"},)
    ax_right.plot(
    range(K + 1), errU_imag[:, -1],
    color="green", linestyle="--",
    label="Imaginary part error",
    )
    ax_right.plot(
        range(K + 1), errU_real[:, -1],
        color="red", linestyle="-.",
        label="Real part error",
    )

    ax_left.legend(loc="upper right")
    ax_right.legend(loc="upper right")
    # plot the errors per cycle
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 8), sharex=True)
    ax_left, ax_right = plot_error_per_cycle(ax_left, ax_right, errTT, errU_complex, K=K, N=N)
    fig.legend(handles=ax_left.lines, loc="upper center", ncol=K + 1)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
