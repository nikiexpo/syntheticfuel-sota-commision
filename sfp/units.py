"""Physical constants, molar masses and unit helpers.

Everything internal is SI unless the name says otherwise:
    time            s
    energy          J
    power           W
    amount          mol
    mass            kg
    temperature     K
    pressure        Pa

Report-facing code converts to kWh / kg / degC / bar at the boundary only.
"""

from __future__ import annotations

# --- universal constants ---------------------------------------------------
R_GAS = 8.314462618  # J/(mol K)      universal gas constant
F_FARADAY = 96485.332  # C/mol          Faraday constant
T0_C = 273.15  # K              0 degC in kelvin
P_ATM = 101325.0  # Pa             standard atmosphere
STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m^2 K^4)

# --- molar masses (kg/mol) -------------------------------------------------
M_CO2 = 44.0095e-3
M_H2 = 2.01588e-3
M_H2O = 18.01528e-3
M_CH4 = 16.0425e-3
M_O2 = 31.9988e-3
M_CAO = 56.0774e-3
M_CACO3 = 100.0869e-3
M_AIR = 28.9647e-3

# --- reaction enthalpies (J/mol, sign = as written) ------------------------
# CaCO3 -> CaO + CO2                              (endothermic, drives the kiln)
DH_CALCINATION = 178.0e3
# CaO + CO2 -> CaCO3                              (exothermic, the air contactor)
DH_CARBONATION = -178.0e3
# CO2 + 4 H2 -> CH4 + 2 H2O(g)                    (Sabatier, exothermic)
DH_SABATIER = -165.0e3
# H2O(l) -> H2 + 0.5 O2, higher heating value      (electrolysis, endothermic)
DH_WATER_SPLIT_HHV = 285.83e3

# --- fuel properties -------------------------------------------------------
LHV_CH4_MASS = 50.0e6  # J/kg   lower heating value of methane
HHV_CH4_MASS = 55.5e6  # J/kg
LHV_H2_MASS = 120.0e6  # J/kg
LHV_CH4_MOLAR = LHV_CH4_MASS * M_CH4  # J/mol

# --- atmosphere ------------------------------------------------------------
Y_CO2_AMBIENT = 420e-6  # mol/mol   ambient CO2 mole fraction (2024 global mean)
RHO_AIR_STD = 1.225  # kg/m^3    sea level, 15 degC

# --- time ------------------------------------------------------------------
SECONDS_PER_MINUTE = 60.0
SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86400.0
SECONDS_PER_YEAR = 31_536_000.0

# --- energy conversions ----------------------------------------------------
J_PER_KWH = 3.6e6
J_PER_MWH = 3.6e9


def c_to_k(t_celsius: float) -> float:
    """Celsius -> kelvin."""
    return t_celsius + T0_C


def k_to_c(t_kelvin: float) -> float:
    """Kelvin -> Celsius."""
    return t_kelvin - T0_C


def j_to_kwh(energy_j: float) -> float:
    """Joule -> kilowatt-hour."""
    return energy_j / J_PER_KWH


def kwh_to_j(energy_kwh: float) -> float:
    """Kilowatt-hour -> joule."""
    return energy_kwh * J_PER_KWH
