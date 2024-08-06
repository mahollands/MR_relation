"""
Utilities for interpolating the MR-relations of Fontaine et al. 2001.
Uses the CO_Hthick (mH=1e-4) and CO_Hthin (mH=1e-10) models.
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
    def _decorator(func):
        @wraps(func)
        def _wrapper(x, y, *args, **kwargs):
            return_with_units = False
            if hasattr(x, 'unit'):
                x = x.to(default_units[x_kind])
                return_with_units = True
            else:
                x <<= default_units[x_kind]
            if hasattr(y, 'unit'):
                y = y.to(default_units[y_kind])
                return_with_units = True
            else:
                y <<= default_units[y_kind]

            z = func(x, y, *args, **kwargs).to(default_units[z_kind])
            return z if return_with_units else z.value
        return _wrapper
    return _decorator

@units_handling(x_kind='Mass', y_kind='Radius', z_kind='logg')
def logg_from_M_R(M, R):
    """
    Input mass (Msun) and radius (Rsun) to get the WD logg (cm s-2 dex).
    """
    g = G*M/R**2
    return g

@units_handling(x_kind='logg', y_kind='Radius', z_kind='Mass')
def M_from_logg_R(logg, R):
    """
    Input logg (cm s-2 dex) and radius (Rsun) to get the WD mass (Msun).
    """
    g = logg.physical
    M = g*R**2/G
    return M

@units_handling(x_kind='Mass', y_kind='logg', z_kind='Radius')
def R_from_M_logg(M, logg):
    """
    Input mass (Msun) and logg (cm s-2 dex) to get the WD radius (Rsun).
    """
    g = logg.physical
    R = np.sqrt(G*M/g)
    return R

@units_handling(x_kind='Teff', y_kind='Radius', z_kind='logg')
def logg_from_Teff_R(Teff, R, thickness):
    """
    Input Teff (K) and radius (Rsun) to get the WD logg (cm s-2 dex).
    Thickness should be one of 'thin'/'thick'.
    """
    GRID = GRIDS[thickness]
    logT = np.log10(Teff.value)
    logR = np.log10(R.value)
    logg = griddata((GRID['logT'], GRID['logR']), GRID['logg'], (logT, logR))
    return logg * u.dex(u.cm/u.s**2)

@units_handling(x_kind='Teff', y_kind='Radius', z_kind='Mass')
def M_from_Teff_R(Teff, R, thickness):
    """
    Input Teff (K) and radius (Rsun) to get the WD mass (Msun).
    Thickness should be one of 'thin'/'thick'.
    """
    logg = logg_from_Teff_R(Teff, R, thickness)
    return M_from_logg_R(logg, R)

@units_handling(x_kind='Teff', y_kind='logg', z_kind='Radius')
def R_from_Teff_logg(Teff, logg, thickness):
    """
    Input Teff (K) and logg (c ms-2 dex) to get the WD radius (Rsun).
    Thickness should be one of 'thin'/'thick'.
    """
    GRID = GRIDS[thickness]
    xyi = np.log10(Teff.value), logg.value
    logR = griddata((GRID['logT'], GRID['logg']), GRID['logR'], xyi)
    return 10**logR * u.Rsun

@units_handling(x_kind='Teff', y_kind='logg', z_kind='Mass')
def M_from_Teff_logg(Teff, logg, thickness):
    """
    Input Teff (K) and logg (cms-2 dex) to get the WD mass (Msun).
    Thickness should be one of 'thin'/'thick'.
    """
    R = R_from_Teff_logg(Teff, logg, thickness)
    return M_from_logg_R(logg, R)

@units_handling(x_kind='Teff', y_kind='Mass', z_kind='logg')
def logg_from_Teff_M(Teff, M, thickness):
    """
    Input Teff (K) and mass (Msun) to get the WD logg (cm s-2 dex).
    Thickness should be one of 'thin'/'thick'.
    """
    GRID = GRIDS[thickness]
    xyi = np.log10(Teff.value), np.log10(M.value)
    logg = griddata((GRID['logT'], GRID['logM']), GRID['logg'], xyi)
    return logg * u.dex(u.cm/u.s**2)

@units_handling(x_kind='Teff', y_kind='Mass', z_kind='Radius')
def R_from_Teff_M(Teff, M, thickness):
    """
    Input Teff (K) and mass (Msun) to get the WD radius.
    Thickness should be one of 'thin'/'thick'.
    """
    logg = logg_from_Teff_M(Teff, M, thickness)
    return R_from_M_logg(M, logg)

@units_handling(x_kind='Teff', y_kind='Radius', z_kind='tau_cool')
def tau_from_Teff_R(Teff, R, thickness):
    """
    Input Teff (K) and radius (Rsun) to get the WD cooling age (Gyr).
    Thickness should be one of 'thin'/'thick'.
    """
    GRID = GRIDS[thickness]
    xyi = np.log10(Teff.value), np.log10(R.value)
    logtau = griddata((GRID['logT'], GRID['logR']), GRID['logtau'], xyi)
    return 10**(logtau-9) * u.Gyr

@units_handling(x_kind='Teff', y_kind='logg', z_kind='tau_cool')
def tau_from_Teff_logg(Teff, logg, thickness):
    """
    Input Teff (K) and logg (cm s-2 dex) to get the WD cooling age (Gyr).
    Thickness should be one of 'thin'/'thick'.
    """
    R = R_from_Teff_logg(Teff, logg, thickness)
    return tau_from_Teff_R(Teff, R, thickness)

@units_handling(x_kind='Teff', y_kind='Mass', z_kind='tau_cool')
def tau_from_Teff_M(Teff, M, thickness):
    """
    Input Teff (K) and mass (Msun) to get the WD cooling age (Gyr).
    Thickness should be one of 'thin'/'thick'.
    """
    R = R_from_Teff_M(Teff, M, thickness)
    return tau_from_Teff_R(Teff, R, thickness)

@units_handling(x_kind='tau_cool', y_kind='Mass', z_kind='Teff')
def Teff_from_tau_M(tau, M, thickness):
    """
    Input tau (Gyr) and mass (Msun) to get the Teff [K].
    Thickness should be one of 'thin'/'thick'.
    Useful for simulation work.
    """
    GRID = GRIDS[thickness]
    xyi = np.log10(tau.value) + 9, np.log10(M.value)
    logT = griddata((GRID['logtau'], GRID['logM']), GRID['logT'], (logtau, logM))
    return 10**logT * u.K

@units_handling(x_kind='Teff', y_kind='Radius', z_kind='Luminosity')
def L_from_Teff_R(Teff, R):
    """
    Input Teff (K) and radius (Rsun) to get the WD luminosity (Lsun).
    """
    L = 4*np.pi * sigma_sb * R**2 * Teff**4
    return L

@units_handling(x_kind='Teff', y_kind='logg', z_kind='Luminosity')
def L_from_Teff_logg(Teff, logg, thickness):
    """
    Input Teff (K) and logg (cm s-2 dex) to get the WD luminosity (Lsun).
    Thickness should be one of 'thin'/'thick'.
    """
    R = R_from_Teff_logg(Teff, logg, thickness)
    return L_from_Teff_R(Teff, R)

@units_handling(x_kind='Teff', y_kind='Mass', z_kind='Luminosity')
def L_from_Teff_M(Teff, M, thickness):
    """
    Input Teff (K) and mass (Msun) to get the WD luminosity (Lsun).
    Thickness should be one of 'thin'/'thick'.
    """
    R = R_from_Teff_M(Teff, M, thickness)
    return L_from_Teff_R(Teff, R)

@units_handling(x_kind='Teff', y_kind='Luminosity', z_kind='Radius')
def R_from_Teff_L(Teff, L):
    """
    Input Teff (K) and luminosity (Lsun) to get the WD radius (Rsun).
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
    Input Teff (K) and luminosity (Lsun) to get the WD mass (Msun).
    Thickness should be one of 'thin'/'thick'.
    """
    R = R_from_Teff_L(Teff, L)
    return M_from_Teff_R(Teff, R, thickness)

@units_handling(x_kind='Radius', y_kind='Luminosity', z_kind='Teff')
def Teff_from_R_L(R, L):
    """
    Input R (Rsun) and luminosity (Lsun) to get the WD Teff (K).
    """
    T4 = L / (4*np.pi * sigma_sb * R**2)
    return T4**(1/4)

@units_handling(x_kind='Mass', y_kind='Radius', z_kind='velocity')
def Grv_from_M_R(M, R):
    """
    Input mass (Msun) and radius (Rsun) to get the WD Grv (km/s).
    """
    rv = G*M/(c*R)
    return rv

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    GRID = GRIDS['thin']
    logg, logR = [np.array(GRID[x]) for x in ("logg", "logR")]
    M2 = M_from_logg_R(logg, 10**logR)
    for T in np.unique(np.array(GRID['logT'])):
        plt.plot(GRID['logM'], GRID['logR'], 'k.', ms=1)
    plt.show()
