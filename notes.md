## Default time-integration used by pyBAMM

- based on IDA solver from [Sundials](https://sundials.readthedocs.io/en/latest/ida/Mathematics_link.html) (adaptive multi-step based on BDF)
  - ❓ chosen order
  - ❓ time-step selection (and possibility to choose a constant time-step)
- stage solves are based on non-linear root finding algorithm stored in the ["Algebraic Solver"](https://github.com/pybamm-team/PyBaMM/blob/main/src/pybamm/solvers/algebraic_solver.py)
    - ❓ generic definition
    - ❓ how to call a function