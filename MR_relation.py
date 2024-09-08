"""
Utilities for interpolating white dwarf Mass-Radius relations
and evolutionary models. The default models are from
Bedard et al. (2020) (adsabs.harvard.edu/abs/2020ApJ...901...93B)
using the CO_Hthick (mH=1e-4) and CO_Hthin (mH=1e-10) model grids.

Generic routines are also provided for simple conversions,
e.g. radius from mass and logg, luminosity from Teff and radius.
"""
import os.path
from functools import wraps
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from astropy import units as u
from astropy.constants import G, c, sigma_sb

__all__ = [
    "logg_from_M_R",
    "M_from_logg_R",
    "R_from_M_logg",
    "logg_from_Teff_R",
    "M_from_Teff_R",
    "R_from_Teff_logg",
    "M_from_Teff_logg",
    "logg_from_Teff_M",
    "R_from_Teff_M",
    "tau_from_Teff_R",
    "tau_from_Teff_logg",
    "tau_from_Teff_M",
    "Teff_from_tau_M",
    "L_from_Teff_R",
    "L_from_Teff_logg",
    "L_from_Teff_M",
    "logg_from_Teff_L",
    "R_from_Teff_L",
    "M_from_Teff_L",
    "Teff_from_R_L",
    "Grv_from_M_R",
]

default_units = {
    'Mass' : u.Msun,
    'Radius' : u.Rsun,
    'logg': u.dex(u.cm/u.s**2),
    'Teff' : u.K,
    'tau_cool' : u.Gyr,
    'Luminosity' : u.Lsun,
    'velocity' : u.km/u.s,
}

def set_models(models):
    t_opt = "thin", "thick"
    mr_dir = os.path.dirname(os.path.abspath(__file__))
    f_grids = {t : f"{mr_dir}/MR_grids/{models}_{t}.csv" for t in t_opt}
    GRIDS = {t : pd.read_csv(f_grids[t]) for t in t_opt}
    for t in t_opt:
        GRID = GRIDS[t]
        GRID['logM'] = np.log10(GRID['Mass'])
    return GRIDS
GRIDS = set_models('Bedard20')

def units_handling(x_kind, y_kind, z_kind):
    """
    Wraps a routine for interpolating z(x, y). If inputs x or y are astropy
    quantities then the output z is also an astropy quantity. If x or y are
    both scalars, then z is returned as a scalar assuming sensible default
    options for their units.
    """
    x_unit, y_unit, z_unit = map(default_units.get, (x_kind, y_kind, z_kind))
    def _decorator(func):
        @wraps(func)
        def _wrapper(x, y, *args, **kwargs):
            has_x_unit, has_y_unit = hasattr(x, 'unit'), hasattr(y, 'unit')
            x = x.to(x_unit) if has_x_unit else x << x_unit
            y = y.to(y_unit) if has_y_unit else y << y_unit
            z = func(x, y, *args, **kwargs)
            z = z.to(z_unit) if hasattr(z, 'unit') else z << z_unit
            return z if has_x_unit or has_y_unit else z.value
        return _wrapper
    return _decorator

#######################################
# M, R, logg conversions

@units_handling(x_kind='Mass', y_kind='Radius', z_kind='logg')
def logg_from_M_R(M, R):
    """
    Input mass and radius to get the WD logg.
    """
    return G*M/R**2

@units_handling(x_kind='logg', y_kind='Radius', z_kind='Mass')
def M_from_logg_R(logg, R):
    """
    Input logg and radius to get the WD mass.
    """
    g = logg.physical
    return g*R**2/G

@units_handling(x_kind='Mass', y_kind='logg', z_kind='Radius')
def R_from_M_logg(M, logg):
    """
    Input mass and logg to get the WD radius.
    """
    g = logg.physical
    return np.sqrt(G*M/g)

#######################################
# M, R, logg, Teff conversions

@units_handling(x_kind='Teff', y_kind='Radius', z_kind='logg')
def logg_from_Teff_R(Teff, R, thickness):
    """
    Input Teff and radius to get the WD logg.
    Thickness should be one of 'thin'/'thick'.
    """
    GRID = GRIDS[thickness]
    xyi = np.log10(Teff.value), np.log10(R.value)
    return griddata((GRID['logT'], GRID['logR']), GRID['logg'], xyi)

@units_handling(x_kind='Teff', y_kind='Radius', z_kind='Mass')
def M_from_Teff_R(Teff, R, thickness):
    """
    Input Teff and radius to get the WD mass.
    Thickness should be one of 'thin'/'thick'.
    """
    logg = logg_from_Teff_R(Teff, R, thickness)
    return M_from_logg_R(logg, R)

@units_handling(x_kind='Teff', y_kind='logg', z_kind='Radius')
def R_from_Teff_logg(Teff, logg, thickness):
    """
    Input Teff and logg to get the WD radius.
    Thickness should be one of 'thin'/'thick'.
    """
    GRID = GRIDS[thickness]
    xyi = np.log10(Teff.value), logg.value
    logR = griddata((GRID['logT'], GRID['logg']), GRID['logR'], xyi)
    return 10**logR

@units_handling(x_kind='Teff', y_kind='logg', z_kind='Mass')
def M_from_Teff_logg(Teff, logg, thickness):
    """
    Input Teff and logg to get the WD mass.
    Thickness should be one of 'thin'/'thick'.
    """
    R = R_from_Teff_logg(Teff, logg, thickness)
    return M_from_logg_R(logg, R)

@units_handling(x_kind='Teff', y_kind='Mass', z_kind='logg')
def logg_from_Teff_M(Teff, M, thickness):
    """
    Input Teff and mass to get the WD logg.
    Thickness should be one of 'thin'/'thick'.
    """
    GRID = GRIDS[thickness]
    xyi = np.log10(Teff.value), np.log10(M.value)
    return griddata((GRID['logT'], GRID['logM']), GRID['logg'], xyi)

@units_handling(x_kind='Teff', y_kind='Mass', z_kind='Radius')
def R_from_Teff_M(Teff, M, thickness):
    """
    Input Teff and mass to get the WD radius.
    Thickness should be one of 'thin'/'thick'.
    """
    logg = logg_from_Teff_M(Teff, M, thickness)
    return R_from_M_logg(M, logg)

#######################################
# cooling age conversions

@units_handling(x_kind='Teff', y_kind='Radius', z_kind='tau_cool')
def tau_from_Teff_R(Teff, R, thickness):
    """
    Input Teff and radius to get the WD cooling age (Gyr).
    Thickness should be one of 'thin'/'thick'.
    """
    GRID = GRIDS[thickness]
    xyi = np.log10(Teff.value), np.log10(R.value)
    logtau = griddata((GRID['logT'], GRID['logR']), GRID['logtau'], xyi)
    return 10**(logtau-9)

@units_handling(x_kind='Teff', y_kind='logg', z_kind='tau_cool')
def tau_from_Teff_logg(Teff, logg, thickness):
    """
    Input Teff and logg to get the WD cooling age.
    Thickness should be one of 'thin'/'thick'.
    """
    R = R_from_Teff_logg(Teff, logg, thickness)
    return tau_from_Teff_R(Teff, R, thickness)

@units_handling(x_kind='Teff', y_kind='Mass', z_kind='tau_cool')
def tau_from_Teff_M(Teff, M, thickness):
    """
    Input Teff  and mass to get the WD cooling age.
    Thickness should be one of 'thin'/'thick'.
    """
    R = R_from_Teff_M(Teff, M, thickness)
    return tau_from_Teff_R(Teff, R, thickness)

@units_handling(x_kind='tau_cool', y_kind='Mass', z_kind='Teff')
def Teff_from_tau_M(tau, M, thickness):
    """
    Input tau and mass to get the Teff.
    Thickness should be one of 'thin'/'thick'.
    Useful for simulation work.
    """
    GRID = GRIDS[thickness]
    xyi = np.log10(tau.value) + 9, np.log10(M.value)
    logT = griddata((GRID['logtau'], GRID['logM']), GRID['logT'], xyi)
    return 10**logT

#######################################
# luminosity conversions

@units_handling(x_kind='Teff', y_kind='Radius', z_kind='Luminosity')
def L_from_Teff_R(Teff, R):
    """
    Input Teff and radius to get the WD luminosity.
    """
    return 4*np.pi * sigma_sb * R**2 * Teff**4

@units_handling(x_kind='Teff', y_kind='logg', z_kind='Luminosity')
def L_from_Teff_logg(Teff, logg, thickness):
    """
    Input Teff and logg to get the WD luminosity.
    Thickness should be one of 'thin'/'thick'.
    """
    R = R_from_Teff_logg(Teff, logg, thickness)
    return L_from_Teff_R(Teff, R)

@units_handling(x_kind='Teff', y_kind='Mass', z_kind='Luminosity')
def L_from_Teff_M(Teff, M, thickness):
    """
    Input Teff and mass to get the WD luminosity.
    Thickness should be one of 'thin'/'thick'.
    """
    R = R_from_Teff_M(Teff, M, thickness)
    return L_from_Teff_R(Teff, R)

@units_handling(x_kind='Teff', y_kind='Luminosity', z_kind='Radius')
def R_from_Teff_L(Teff, L):
    """
    Input Teff and luminosity to get the WD radius.
    """
    R2 = L / (4*np.pi * sigma_sb * Teff**4)
    return np.sqrt(R2)

@units_handling(x_kind='Teff', y_kind='Luminosity', z_kind='logg')
def logg_from_Teff_L(Teff, L, thickness):
    """
    Input Teff (K) and luminosity (Lsun) to get the WD logg (cm s-2 dex).
    Thickness should be one of 'thin'/'thick'.
    """
    R = R_from_Teff_L(Teff, L)
    return logg_from_Teff_R(Teff, R, thickness)

@units_handling(x_kind='Teff', y_kind='Luminosity', z_kind='Mass')
def M_from_Teff_L(Teff, L, thickness):
    """
    Input Teff and luminosity to get the WD mass.
    Thickness should be one of 'thin'/'thick'.
    """
    R = R_from_Teff_L(Teff, L)
    return M_from_Teff_R(Teff, R, thickness)

@units_handling(x_kind='Radius', y_kind='Luminosity', z_kind='Teff')
def Teff_from_R_L(R, L):
    """
    Input R and luminosity to get the WD Teff.
    """
    T4 = L / (4*np.pi * sigma_sb * R**2)
    return T4**(1/4)

#######################################
# gravitational redshift

@units_handling(x_kind='Mass', y_kind='Radius', z_kind='velocity')
def Grv_from_M_R(M, R):
    """
    Input mass and radius to get the gravitational redshift.
    """
    return G*M/(c*R)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    GRID = GRIDS['thin']
    logg, logR = [np.array(GRID[x]) for x in ("logg", "logR")]
    M2 = M_from_logg_R(logg, 10**logR)
    for T in np.unique(np.array(GRID['logT'])):
        plt.plot(GRID['logM'], GRID['logR'], 'k.', ms=1)
    plt.show()
