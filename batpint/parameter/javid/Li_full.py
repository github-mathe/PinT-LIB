"""Full-cell lithium-ion parameter set based on Javid et al. (DOI: 10.1016/j.electacta.2024.144259)."""

import pybamm

from batpint.parameter.javid.common import BASE_PARAMS, anode_OCP


def anode_exchange_current_density(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for the negative electrode [A.m-2]
    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    m_ref_Javid = 0.2  # kin rate(t)
    ce_ref = 1000
    c_e_avg = pybamm.XAverage(c_e)
    m_ref = 2 * m_ref_Javid / (c_s_max * ce_ref**0.5)  # (A/m2)(mol/m3) - includes ref concentrations
    return m_ref * c_e_avg**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


PARAMS = {
    **BASE_PARAMS,
    # Anode (full-cell specific)
    "Negative electrode thickness [m]": 80e-6,
    "Negative particle radius [m]": 12e-6,
    "Negative electrode active material volume fraction": 0.53,
    "Negative electrode porosity": 0.42,
    "Negative particle diffusivity [m2.s-1]": 1.5e-13,
    "Negative electrode Bruggeman coefficient (electrode)": 1.5,
    "Negative electrode exchange-current density [A.m-2]": anode_exchange_current_density,
    "Initial concentration in negative electrode [mol.m-3]": 26332.5,
    "Maximum concentration in negative electrode [mol.m-3]": 30000,
    "Negative electrode OCP [V]": anode_OCP,
    "Negative electrode OCP entropic change [V.K-1]": 0.0,
    "Negative electrode conductivity [S.m-1]": 100,
}
