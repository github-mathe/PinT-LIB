"""Half-cell lithium-ion parameter set based on Javid et al. (DOI: 10.1016/j.electacta.2024.144259).

In the half-cell configuration the negative electrode is a lithium metal plate,
so the negative electrode particle parameters are omitted (thickness = 0).
"""

from batpint.parameter.javid.common import BASE_PARAMS

PARAMS = {
    **BASE_PARAMS,
    # Anode (lithium metal plate — no particle domain)
    "Negative electrode thickness [m]": 0,
    "Negative electrode OCP [V]": 0.0,
    "Negative electrode conductivity [S.m-1]": 100,
}
