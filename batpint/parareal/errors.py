"""Error measures for Parareal solutions."""

from collections.abc import Sequence
from typing import Any

import numpy as np


def _norm(value: Any, ord: int | float | None) -> float:
    """Return a vector or scalar norm as a Python float."""
    array = np.asarray(value)
    if array.ndim == 0:
        return float(np.abs(array))
    return float(np.linalg.norm(array, ord=ord))


def _error(
    computed: Any,
    reference: Any,
    ord: int | float | None,
    relative: bool,
) -> float:
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
    U = np.asarray(U, dtype=object)  
    u_exact = np.asarray(u_exact, dtype=object)
    
    if u_exact.shape != (U.shape[1],):
        raise ValueError("u_exact must have one entry per cycle.")

    solution_errors = np.empty(U.shape, dtype=object)
    for iteration, cycle in np.ndindex(U.shape):
        solution_errors[iteration, cycle] = _error(
            U[iteration, cycle], u_exact[cycle], ord, relative
        )

    return solution_errors
