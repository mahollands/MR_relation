import numpy as np
from scipy.interpolate import griddata
from astropy.constants import G, c, sigma_sb
from .MR_utils import Grid, units_handling

MR_grid = Grid(grid_name="Bedard20")

#######################################
# M, R, logg conversions


@units_handling(x_kind="Mass", y_kind="Radius", z_kind="logg")
def logg_from_M_R(M, R):
    """
    Input mass and radius to get the logg.
    """
    return G * M / R**2


@units_handling(x_kind="logg", y_kind="Radius", z_kind="Mass")
def M_from_logg_R(logg, R):
    """
    Input logg and radius to get the mass.
    """
    g = logg.physical
    return g * R**2 / G


@units_handling(x_kind="Mass", y_kind="logg", z_kind="Radius")
def R_from_M_logg(M, logg):
    """
    Input mass and logg to get the radius.
    """
    g = logg.physical
    return np.sqrt(G * M / g)


#######################################
# Teff conversions


@units_handling(x_kind="Teff", y_kind="Radius", z_kind="logg")
def logg_from_Teff_R(Teff, R, thickness):
    """
    Input Teff and radius to get the WD logg.
    Thickness should be one of 'thin'/'thick'.
    """
    grid = MR_grid(thickness)
    xyi = np.log10(Teff.value), np.log10(R.value)
    return griddata((grid["logT"], grid["logR"]), grid["logg"], xyi)


@units_handling(x_kind="Teff", y_kind="logg", z_kind="Radius")
def R_from_Teff_logg(Teff, logg, thickness):
    """
    Input Teff and logg to get the WD radius.
    Thickness should be one of 'thin'/'thick'.
    """
    grid = MR_grid(thickness)
    xyi = np.log10(Teff.value), logg.value
    logR = griddata((grid["logT"], grid["logg"]), grid["logR"], xyi)
    return 10**logR


@units_handling(x_kind="Teff", y_kind="logg", z_kind="Mass")
def M_from_Teff_logg(Teff, logg, thickness):
    """
    Input Teff and logg to get the WD mass.
    Thickness should be one of 'thin'/'thick'.
    """
    R = R_from_Teff_logg(Teff, logg, thickness)
    return M_from_logg_R(logg, R)


@units_handling(x_kind="Teff", y_kind="Mass", z_kind="logg")
def logg_from_Teff_M(Teff, M, thickness):
    """
    Input Teff and mass to get the WD logg.
    Thickness should be one of 'thin'/'thick'.
    """
    grid = MR_grid(thickness)
    xyi = np.log10(Teff.value), np.log10(M.value)
    return griddata((grid["logT"], grid["logM"]), grid["logg"], xyi)


@units_handling(x_kind="Teff", y_kind="Radius", z_kind="Mass")
def M_from_Teff_R(Teff, R, thickness):
    """
    Input Teff and radius to get the WD mass.
    Thickness should be one of 'thin'/'thick'.
    """
    logg = logg_from_Teff_R(Teff, R, thickness)
    return M_from_logg_R(logg, R)


@units_handling(x_kind="Teff", y_kind="Mass", z_kind="Radius")
def R_from_Teff_M(Teff, M, thickness):
    """
    Input Teff and mass to get the WD radius.
    Thickness should be one of 'thin'/'thick'.
    """
    logg = logg_from_Teff_M(Teff, M, thickness)
    return R_from_M_logg(M, logg)


#######################################
# cooling age conversions


@units_handling(x_kind="Teff", y_kind="Radius", z_kind="Age")
def tau_from_Teff_R(Teff, R, thickness):
    """
    Input Teff and radius to get the WD cooling age.
    Thickness should be one of 'thin'/'thick'.
    """
    grid = MR_grid(thickness)
    xyi = np.log10(Teff.value), np.log10(R.value)
    logtau = griddata((grid["logT"], grid["logR"]), grid["logtau"], xyi)
    return 10 ** (logtau - 9)


@units_handling(x_kind="Teff", y_kind="logg", z_kind="Age")
def tau_from_Teff_logg(Teff, logg, thickness):
    """
    Input Teff and logg to get the WD cooling age.
    Thickness should be one of 'thin'/'thick'.
    """
    R = R_from_Teff_logg(Teff, logg, thickness)
    return tau_from_Teff_R(Teff, R, thickness)


@units_handling(x_kind="Teff", y_kind="Mass", z_kind="Age")
def tau_from_Teff_M(Teff, M, thickness):
    """
    Input Teff  and mass to get the WD cooling age.
    Thickness should be one of 'thin'/'thick'.
    """
    R = R_from_Teff_M(Teff, M, thickness)
    return tau_from_Teff_R(Teff, R, thickness)


@units_handling(x_kind="Age", y_kind="Mass", z_kind="Teff")
def Teff_from_tau_M(tau, M, thickness):
    """
    Input tau and mass to get the Teff.
    Thickness should be one of 'thin'/'thick'.
    Useful for simulation work.
    """
    grid = MR_grid(thickness)
    xyi = np.log10(tau.value) + 9, np.log10(M.value)
    logT = griddata((grid["logtau"], grid["logM"]), grid["logT"], xyi)
    return 10**logT


#######################################
# luminosity conversions


@units_handling(x_kind="Teff", y_kind="Radius", z_kind="Luminosity")
def L_from_Teff_R(Teff, R):
    """
    Input Teff and radius to get the luminosity.
    """
    return 4 * np.pi * sigma_sb * R**2 * Teff**4


@units_handling(x_kind="Teff", y_kind="Luminosity", z_kind="Radius")
def R_from_Teff_L(Teff, L):
    """
    Input Teff and luminosity to get the radius.
    """
    R2 = L / (4 * np.pi * sigma_sb * Teff**4)
    return np.sqrt(R2)


@units_handling(x_kind="Radius", y_kind="Luminosity", z_kind="Teff")
def Teff_from_R_L(R, L):
    """
    Input R and luminosity to get the Teff.
    """
    T4 = L / (4 * np.pi * sigma_sb * R**2)
    return T4 ** (1 / 4)


@units_handling(x_kind="Teff", y_kind="logg", z_kind="Luminosity")
def L_from_Teff_logg(Teff, logg, thickness):
    """
    Input Teff and logg to get the WD luminosity.
    Thickness should be one of 'thin'/'thick'.
    """
    R = R_from_Teff_logg(Teff, logg, thickness)
    return L_from_Teff_R(Teff, R)


@units_handling(x_kind="Teff", y_kind="Luminosity", z_kind="logg")
def logg_from_Teff_L(Teff, L, thickness):
    """
    Input Teff and luminosity to get the WD logg.
    Thickness should be one of 'thin'/'thick'.
    """
    R = R_from_Teff_L(Teff, L)
    return logg_from_Teff_R(Teff, R, thickness)


@units_handling(x_kind="Teff", y_kind="Mass", z_kind="Luminosity")
def L_from_Teff_M(Teff, M, thickness):
    """
    Input Teff and mass to get the WD luminosity.
    Thickness should be one of 'thin'/'thick'.
    """
    R = R_from_Teff_M(Teff, M, thickness)
    return L_from_Teff_R(Teff, R)


@units_handling(x_kind="Teff", y_kind="Luminosity", z_kind="Mass")
def M_from_Teff_L(Teff, L, thickness):
    """
    Input Teff and luminosity to get the WD mass.
    Thickness should be one of 'thin'/'thick'.
    """
    R = R_from_Teff_L(Teff, L)
    return M_from_Teff_R(Teff, R, thickness)


#######################################
# gravitational redshift


@units_handling(x_kind="Mass", y_kind="Radius", z_kind="Velocity")
def Grv_from_M_R(M, R):
    """
    Input mass and radius to get the gravitational redshift.
    """
    return G * M / (c * R)


@units_handling(x_kind="Velocity", y_kind="Mass", z_kind="Radius")
def R_from_Grv_M(Grv, M):
    """
    Input gravitational redshift and mass to get the radius.
    """
    return G * M / (c * Grv)


@units_handling(x_kind="Velocity", y_kind="Radius", z_kind="Mass")
def M_from_Grv_R(Grv, R):
    """
    Input gravitational redshift and radius to get the mass.
    """
    return c * Grv * R / G
