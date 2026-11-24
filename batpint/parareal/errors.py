"""Error measures for Parareal solutions."""

from typing import Any

import numpy as np


def _norm(value: Any, ord: int | float | None):
    """Return a vector or scalar norm as a Python float."""
    array = np.asarray(value)
    if array.ndim == 0:
        return np.abs(array)
    return np.linalg.norm(array, ord=ord)


def _error(
    computed: Any,
    reference: Any,
    ord: int | float | None,
    relative: bool,
):
    absolute_error = _norm(np.asarray(computed) - np.asarray(reference), ord)
    if not relative:
        return absolute_error

    reference_norm = _norm(reference, ord)
    if reference_norm == 0:
        return 0.0 if absolute_error == 0 else np.inf
    return absolute_error / reference_norm


def compute_parareal_errors(
    U: np.ndarray,
    u_exact: np.ndarray,
    *,
    ord: int | float | None = np.inf,
    relative: bool = False,
) -> np.ndarray:
    """Compute solution errors for every Parareal iteration and cycle.

    Parameters
    ----------

    U : numpy.ndarray
        Object array of Parareal solutions with shape ``(K + 1, N + 1)``.
        Each entry can be one NumPy array, a tuple/list of component arrays, or
        a multidimensional array whose first axis indexes components.
    u_exact : numpy.ndarray
        Object array of exact solutions with shape ``(N + 1,)``
    ord : int, float, or None, default numpy.inf
        Order passed to :func:`numpy.linalg.norm`.
    relative : bool, default False
        If true, divide each error by the norm of its fine reference.
        A zero reference gives zero for a zero error and infinity otherwise.

    Returns
    -------
    numpy.ndarray
        Solution errors as an object array.
        For multi-component solutions, each object entry is a NumPy array
        containing one error per component. For an array state, components are
        taken along its first axis.
    """
    U = np.asarray(U, dtype=U.dtype)
    u_exact = np.asarray(u_exact, dtype=u_exact.dtype)

    if u_exact.shape != (U.shape[1:]):
        raise ValueError("u_exact must have one entry per cycle.")

    solution_errors = np.empty(U.shape[:2], dtype=np.float64)
    for iteration, cycle in np.ndindex(U.shape[0:2]):
        solution_errors[iteration, cycle] = _error(U[iteration, cycle], u_exact[cycle], ord, relative)

    return solution_errors
