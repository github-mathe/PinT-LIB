"""Tests for the Parareal solver and error computation."""

import numpy as np
import pytest

from batpint.parareal.errors import compute_parareal_errors
from batpint.parareal.parareal import PararealModified
from batpint.parareal.propagation_state import PropagationState

# --- PropagationState ---


def test_propagation_state_deep_copies():
    u = np.array([1.0, 2.0])
    state = PropagationState(t=0.0, u=u, cycle=3, extra="hello")
    u[0] = 999.0
    assert state.u[0] == 1.0, "PropagationState should deep-copy u"
    assert state.cycle == 3
    assert state.extra == "hello"


# --- compute_parareal_errors ---


def test_compute_parareal_errors_shape():
    K, N = 2, 3
    U = np.empty((K + 1, N + 1), dtype=object)
    for k in range(K + 1):
        for n in range(N + 1):
            U[k, n] = np.array([float(k + n)])
    u_exact = np.empty(N + 1, dtype=object)
    for n in range(N + 1):
        u_exact[n] = np.array([0.0])

    errors = compute_parareal_errors(U, u_exact)
    assert errors.shape == (K + 1, N + 1)
    # Error at k=0, n=0 should be |0 - 0| = 0
    assert errors[0, 0] == 0.0
    # Error at k=1, n=2 should be |3 - 0| = 3
    assert errors[1, 2] == 3.0


def test_compute_parareal_errors_relative():
    K, N = 1, 1
    U = np.empty((K + 1, N + 1), dtype=object)
    U[0, 0] = np.array([1.0])
    U[0, 1] = np.array([2.0])
    U[1, 0] = np.array([1.0])
    U[1, 1] = np.array([4.0])
    u_exact = np.empty(N + 1, dtype=object)
    u_exact[0] = np.array([1.0])
    u_exact[1] = np.array([2.0])

    errors = compute_parareal_errors(U, u_exact, relative=True)
    assert errors[1, 1] == 1.0  # |4-2| / |2| = 1.0


def test_compute_parareal_errors_shape_mismatch():
    U = np.empty((2, 3), dtype=object)
    u_exact = np.empty(5, dtype=object)  # wrong shape
    with pytest.raises(ValueError, match="one entry per cycle"):
        compute_parareal_errors(U, u_exact)


def test_compute_parareal_errors_zero_reference():
    U = np.empty((1, 2), dtype=object)
    U[0, 0] = np.array([0.0])
    U[0, 1] = np.array([1.0])
    u_exact = np.empty(2, dtype=object)
    u_exact[0] = np.array([0.0])
    u_exact[1] = np.array([0.0])

    errors = compute_parareal_errors(U, u_exact, relative=True)
    assert errors[0, 0] == 0.0  # zero error on zero reference
    assert errors[0, 1] == np.inf  # nonzero error on zero reference


# --- PararealModified ---


class _MockPropagator:
    """Simple propagator: u_next = u + 1, t_next = t + 1."""

    def __init__(self, offset=0.0):
        self.offset = offset

    def propagate(self, t, u, cycle):
        return t + 1.0, np.asarray(u, dtype=np.longdouble) + 1.0 + self.offset


def test_parareal_modified_basic():
    """Parareal with mock propagators should produce correct shapes and values."""
    fine = _MockPropagator(offset=0.0)
    coarse = _MockPropagator(offset=0.5)
    parareal = PararealModified(fine=fine, coarse=coarse, coarse_s=coarse)

    N, K = 3, 2
    TT, U = parareal.solve(t0=0.0, u0=np.array([0.0]), K=K, N=N)

    assert TT.shape == (K + 1, N + 1)
    assert U.shape == (K + 1, N + 1)
    # First row: coarse propagation -> t = 0,1,2,3
    assert np.allclose(TT[0, :], [0, 1, 2, 3])
    # All rows should have t[*,0] = 0
    assert np.allclose(TT[:, 0], 0.0)
