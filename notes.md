## Default time-integration used by PyBaMM

- Based on the IDA solver from [Sundials](https://sundials.readthedocs.io/en/latest/ida/Mathematics_link.html) (adaptive multi-step based on BDF).
- Stage solves use the non-linear root-finding algorithm in PyBaMM's ["Algebraic Solver"](https://github.com/pybamm-team/PyBaMM/blob/main/src/pybamm/solvers/algebraic_solver.py).
