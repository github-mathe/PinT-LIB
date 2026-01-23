import pybamm
from pathlib import Path 


# exchange current density of the anode - Li plate
def li_metal_exchange_current_density(c_e, c_Li, T): # i0_cat
    """
    Exchange-current density for lithium metal electrode [A.m-2]
    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_Li : :class:`pybamm.Symbol`
        Pure metal lithium concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    m_ref_Javid = 100
    c_e_ref = 1000
    m_ref = m_ref_Javid/c_e_ref**0.5  # (A/m2)(mol/m3) - includes ref concentrations
    c_e_avg = pybamm.XAverage(c_e)
    return m_ref * c_e_avg**0.5 # i0_an

# exchange current density of the cathode - Li half cell
def cathode_exchange_current_density(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for lithium metal electrode [A.m-2]
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
    m_ref_Javid = 0.75 #kin rate(t)
    ce_ref=1000
    c_e_avg = pybamm.XAverage(c_e)
    m_ref = 2*m_ref_Javid/(c_s_max*ce_ref**0.5)  # (A/m2)(mol/m3) - includes ref concentrations
    return m_ref * c_e_avg**0.5 * c_s_surf**0.5*(c_s_max-c_s_surf)**0.5

def anode_exchange_current_density(c_e, c_s_surf,c_s_max, T):
    """
    Exchange-current density for lithium metal electrode [A.m-2]
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
    m_ref_Javid = 0.2 #kin rate(t)
    ce_ref=1000
    c_e_avg = pybamm.XAverage(c_e)
    m_ref = 2*m_ref_Javid/(c_s_max*ce_ref**0.5)  # (A/m2)(mol/m3) - includes ref concentrations
    return m_ref * c_e_avg**0.5 * c_s_surf**0.5*(c_s_max-c_s_surf)**0.5


# Load data in the appropriate format
folder_path = Path(__file__).resolve().parent / "OCV"


OCV_cat_data = pybamm.parameters.process_1D_data("OCV_cat.csv", path=folder_path)
OCV_an_data = pybamm.parameters.process_1D_data("OCV_an.csv", path=folder_path)   

def cat_OCP(sto):
    name, (x,y) = OCV_cat_data
    return pybamm.Interpolant(x,y,sto,name=name)
def anode_OCP(sto):
    name, (x,y) = OCV_an_data
    return pybamm.Interpolant(x,y,sto,name=name)

#Scalars
# define cell parameters
PARAMS = {
        "Faraday constant": 96485,
        "Initial temperature": 298.15,
        "Ideal gas constant": 8.314462618,
        
    # Cathode parameters

    # Geometric and material properties
    "Positive electrode thickness [m]": 55e-6,       
    "Positive particle radius [m]": 8e-6,
    "Positive electrode active material volume fraction": 0.56,
    "Positive electrode porosity": 0.39,
    'Positive electrode conductivity [S.m-1]': 9.2,
    # Transport properties
    "Positive particle diffusivity [m2.s-1]": 4e-14,
    "Positive electrode Bruggeman coefficient (electrode)": 1.5,
    
    # kinetic properties    
    "Positive electrode exchange-current density [A.m-2]": cathode_exchange_current_density,

    # Concentrations
    "Initial concentration in positive electrode [mol.m-3]": 0.222*49200,
    "Maximum concentration in positive electrode [mol.m-3]": 49200,

    # Thermal properties (optional)
    "Positive electrode OCP [V]": cat_OCP,
    'Positive electrode OCP entropic change [V.K-1]':0,

    #electrolyte
    # Geometric and material properties
    "Separator thickness [m]": 15e-6,
    "Separator porosity": 0.7,

    # Transport properties
    "Separator Bruggeman coefficient (electrolyte)": 1.5,
    "Separator Bruggeman coefficient (electrode)": 1.5, 
    "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
    "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
    "Electrolyte diffusivity [m2.s-1]": 1.95e-10,
    "Electrolyte conductivity [S.m-1]": 0.927,
    "Cation transference number": 0.47,

    # Concentrations
    "Initial concentration in electrolyte [mol.m-3]": 1000,

    # Thermal properties
    "Thermodynamic factor": 3,


    # Anode parameters
    # Geometric and material properties
    "Negative electrode thickness [m]": 80e-6,
    "Negative particle radius [m]": 12e-6,
    "Negative electrode active material volume fraction": 0.53,
    "Negative electrode porosity": 0.42,

    # Transport properties
    "Negative particle diffusivity [m2.s-1]": 1.5e-13,
    "Negative electrode Bruggeman coefficient (electrode)": 1.5,
    
    # kinetic properties    
    "Negative electrode exchange-current density [A.m-2]": anode_exchange_current_density,

    # Concentrations
    "Initial concentration in negative electrode [mol.m-3]": 26332.5,
    "Maximum concentration in negative electrode [mol.m-3]": 30000,

    # Thermal properties (optional)
    "Negative electrode OCP [V]": anode_OCP,
    "Negative electrode OCP entropic change [V.K-1]": 0.0,
    "Exchange-current density for lithium metal electrode [A.m-2]":li_metal_exchange_current_density,
    "Negative electrode conductivity [S.m-1]": 100,
    
    "Nominal cell capacity [A.h]": 29.2425,
    "Current function [A]":  29.2425,
    'Lithium metal partial molar volume [m3.mol-1]':1,
        
    "Reference temperature [K]" : 298.15,
    "Ambient temperature [K]" : 298.15,
    "Initial temperature [K]" : 298.15,
    
    "Electrode height [m]" : 1,
    "Electrode width [m]": 1,
    "Upper voltage cut-off [V]" : 4.259,
    "Lower voltage cut-off [V]" : 3,
    "Open-circuit voltage at 0% SOC [V]": 4.259,
    "Open-circuit voltage at 100% SOC [V]": 3.5696,
    "Number of electrodes connected in parallel to make a cell": 1,
    'Number of cells connected in series to make a battery':1

}