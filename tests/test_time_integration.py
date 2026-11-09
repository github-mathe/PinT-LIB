"""Tests for time integration methods: BackwardEuler and Newton solver."""

import numpy as np
import pytest

from batpint.time_integration.backwardeuler import BackwardEuler
from batpint.time_integration.newton import Newton


def test_newton_scalar():
    """Newton should solve x^2 - 2 = 0."""
    newton = Newton(tol=1e-12)
    residual = lambda x: x**2 - 2
    jacobian = lambda x: 2 * x
    x = newton.solve(residual, x0=1.0, residual_jacobian=jacobian)
    assert x == pytest.approx(np.sqrt(2), rel=1e-10)


def test_newton_vector():
    """Newton should solve a 2D system."""
    newton = Newton(tol=1e-12)

    def residual(x):
        return np.array([x[0] ** 2 + x[1] ** 2 - 1, x[0] - x[1]])

    def jacobian(x):
        return np.array([[2 * x[0], 2 * x[1]], [1.0, -1.0]])

    x = newton.solve(residual, x0=np.array([1.0, 0.0]), residual_jacobian=jacobian)
    assert x[0] == pytest.approx(1 / np.sqrt(2), rel=1e-10)
    assert x[1] == pytest.approx(1 / np.sqrt(2), rel=1e-10)


def test_newton_requires_jacobian():
    newton = Newton()
    with pytest.raises(ValueError, match="requires a residual Jacobian"):
        newton.solve(lambda x: x, x0=1.0)


def test_newton_no_convergence():
    """Newton should raise RuntimeError if it doesn't converge."""
    newton = Newton(tol=1e-15, max_iter=3)
    # Oscillating residual that won't converge in 3 iterations
    residual = lambda x: np.array([np.sin(x[0] * 100) + 10])
    jacobian = lambda x: np.array([[100 * np.cos(x[0] * 100)]])
    with pytest.raises(RuntimeError, match="did not converge"):
        newton.solve(residual, x0=np.array([0.0]), residual_jacobian=jacobian)


def test_backward_euler_linear():
    """Backward Euler on u' = -u should give u_{n+1} = u_n / (1 + h)."""
    be = BackwardEuler(algebraic_solver=Newton(tol=1e-14))
    rhs = lambda t, u: -u
    jacobian = lambda t, u: -1.0
    u_next = be.step(t=0.0, u=1.0, h=0.1, rhs=rhs, jacobian=jacobian)
    assert u_next == pytest.approx(1.0 / 1.1, rel=1e-10)


def test_backward_euler_vector():
    """Backward Euler on a 2D linear system — first-order, so use small h."""
    be = BackwardEuler(algebraic_solver=Newton(tol=1e-14))
    A = np.array([[-1.0, 0.5], [0.0, -2.0]])
    rhs = lambda t, u: A @ u
    jacobian = lambda t, u: A
    u0 = np.array([1.0, 2.0])
    h = 0.001
    u_next = be.step(t=0.0, u=u0, h=h, rhs=rhs, jacobian=jacobian)
    # Exact: u(h) = exp(A*h) @ u0
    from scipy.linalg import expm

    u_exact = expm(A * h) @ u0
    assert np.allclose(u_next, u_exact, atol=1e-4)


def test_integration_method_param_validation():
    """IntegrationMethod.step must start with (t, u, h, rhs, jacobian)."""
    from batpint.time_integration.base_method import IntegrationMethod

    class BadMethod(IntegrationMethod):
        def step(self, x, u, h, rhs, jacobian=None):
            return u

    with pytest.raises(ValueError, match="must start with arguments"):
        BadMethod()
