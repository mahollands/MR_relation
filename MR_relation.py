"""
Utilities for interpolating the MR-relations of Fontaine et al. 2001.
Uses the CO_Hthick (mH=1e-4) and CO_Hthin (mH=1e-10) models.
"""
import os.path
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from astropy import units as u
from astropy.constants import G, c

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
    "L_from_Teff_R",
    "L_from_Teff_logg",
    "L_from_Teff_M",
    "logg_from_Teff_L",
    "R_from_Teff_L",
    "M_from_Teff_L",
    "Teff_from_logg_L",
    "R_from_logg_L",
    "M_from_logg_L",
    "Grv_from_M_R",
]

t_opt = "thin", "thick"
mr_dir = os.path.dirname(os.path.abspath(__file__))
f_grids = {t : f"{mr_dir}/MR_grids/CO_{t}H_processed.csv" for t in t_opt}
GRIDS = {t : pd.read_csv(f_grids[t]) for t in t_opt}
for t in t_opt:
    GRID = GRIDS[t]
    GRID['logM'] = np.log10(GRID['Mass'])

def logg_from_M_R(M, R):
    """
    Input mass (Msun) and radius (Rsun) to get the WD logg (cm s-2 dex).
    """
    M <<= u.Msun
    R <<= u.Rsun
    g = G*M/R**2
    return np.log10(g.to(u.cm/u.s**2).value)

def M_from_logg_R(logg, R):
    """
    Input logg (cm s-2 dex) and radius (Rsun) to get the WD mass (Msun).
    """
    g = 10**logg * u.cm/u.s**2
    R <<= u.Rsun
    M = g*R**2/G
    return M.to(u.Msun).value

def R_from_M_logg(M, logg):
    """
    Input mass (Msun) and logg (cm s-2 dex) to get the WD radius (Rsun).
    """
    g = 10**logg * u.cm/u.s**2
    M <<= u.Msun
    R = np.sqrt(G*M/g)
    return R.to(u.Rsun).value

def logg_from_Teff_R(Teff, R, thickness):
    """
    Input Teff (K) and radius (Rsun) to get the WD logg (cm s-2 dex).
    Thickness should be one of 'thin'/'thick'.
    """
    GRID = GRIDS[thickness]
    logT = np.log10(Teff)
    logR = np.log10(R)
    return griddata((GRID['logT'], GRID['logR']), GRID['logg'], (logT, logR))

def M_from_Teff_R(Teff, R, thickness):
    """
    Input Teff (K) and radius (Rsun) to get the WD mass (Msun).
    Thickness should be one of 'thin'/'thick'.
    """
    logg = logg_from_Teff_R(Teff, R, thickness)
    return M_from_logg_R(logg, R)

def R_from_Teff_logg(Teff, logg, thickness):
    """
    Input Teff (K) and logg (c ms-2 dex) to get the WD radius (Rsun).
    Thickness should be one of 'thin'/'thick'.
    """
    GRID = GRIDS[thickness]
    logT = np.log10(Teff)
    logR = griddata((GRID['logT'], GRID['logg']), GRID['logR'], (logT, logg))
    return 10**logR

def M_from_Teff_logg(Teff, logg, thickness):
    """
    Input Teff (K) and logg (cms-2 dex) to get the WD mass (Msun).
    Thickness should be one of 'thin'/'thick'.
    """
    R = R_from_Teff_logg(Teff, logg, thickness)
    return M_from_logg_R(logg, R)

def logg_from_Teff_M(Teff, M, thickness):
    """
    Input Teff (K) and mass (Msun) to get the WD logg (cm s-2 dex).
    Thickness should be one of 'thin'/'thick'.
    """
    GRID = GRIDS[thickness]
    logT = np.log10(Teff)
    logM = np.log10(M)
    return griddata((GRID['logT'], GRID['logM']), GRID['logg'], (logT, logM))

def R_from_Teff_M(Teff, M, thickness):
    """
    Input Teff (K) and mass (Msun) to get the WD radius.
    Thickness should be one of 'thin'/'thick'.
    """
    logg = logg_from_Teff_M(Teff, M, thickness)
    return R_from_M_logg(M, logg)

def tau_from_Teff_R(Teff, R, thickness):
    """
    Input Teff (K) and radius (Rsun) to get the WD cooling age (Gyr).
    Thickness should be one of 'thin'/'thick'.
    """
    GRID = GRIDS[thickness]
    logT = np.log10(Teff)
    logR = np.log10(R)
    logtau = griddata((GRID['logT'], GRID['logR']), GRID['logtau'], (logT, logR))
    return 10**(logtau-9)

def tau_from_Teff_logg(Teff, logg, thickness):
    """
    Input Teff (K) and logg (cm s-2 dex) to get the WD cooling age (Gyr).
    Thickness should be one of 'thin'/'thick'.
    """
    R = R_from_Teff_logg(Teff, logg, thickness)
    return tau_from_Teff_R(Teff, R, thickness)

def tau_from_Teff_M(Teff, M, thickness):
    """
    Input Teff (K) and mass (Msun) to get the WD cooling age (Gyr).
    Thickness should be one of 'thin'/'thick'.
    """
    R = R_from_Teff_M(Teff, M, thickness)
    return tau_from_Teff_R(Teff, R, thickness)

def L_from_Teff_R(Teff, R, thickness):
    """
    Input Teff (K) and radius (Rsun) to get the WD luminosity (Lsun).
    Thickness should be one of 'thin'/'thick'.
    """
    GRID = GRIDS[thickness]
    logT = np.log10(Teff)
    logR = np.log10(R)
    logL = griddata((GRID['logT'], GRID['logR']), GRID['logL'], (logT, logR))
    return 10**logL

def L_from_Teff_logg(Teff, logg, thickness):
    """
    Input Teff (K) and logg (cm s-2 dex) to get the WD luminosity (Lsun).
    Thickness should be one of 'thin'/'thick'.
    """
    R = R_from_Teff_logg(Teff, logg, thickness)
    return L_from_Teff_R(Teff, R, thickness)

def L_from_Teff_M(Teff, M, thickness):
    """
    Input Teff (K) and mass (Msun) to get the WD luminosity (Lsun).
    Thickness should be one of 'thin'/'thick'.
    """
    R = R_from_Teff_M(Teff, M, thickness)
    return L_from_Teff_R(Teff, R, thickness)

def logg_from_Teff_L(Teff, L, thickness):
    """
    Input Teff (K) and luminosity (Lsun) to get the WD logg (cm s-2 dex).
    Thickness should be one of 'thin'/'thick'.
    """
    GRID = GRIDS[thickness]
    logT = np.log10(Teff)
    logL = np.log10(L)
    logg = griddata((GRID['logT'], GRID['logL']), GRID['logg'], (logT, logL))
    return logg

def R_from_Teff_L(Teff, L, thickness):
    """
    Input Teff (K) and luminosity (Lsun) to get the WD radius (Rsun).
    Thickness should be one of 'thin'/'thick'.
    """
    GRID = GRIDS[thickness]
    logT = np.log10(Teff)
    logL = np.log10(L)
    logR = griddata((GRID['logT'], GRID['logL']), GRID['logR'], (logT, logL))
    return 10**logR

def M_from_Teff_L(Teff, L, thickness):
    """
    Input Teff (K) and luminosity (Lsun) to get the WD mass (Msun).
    Thickness should be one of 'thin'/'thick'.
    """
    logg = logg_from_Teff_L(Teff, L, thickness)
    R = R_from_Teff_L(Teff, L, thickness)
    return M_from_logg_R(logg, R)

def Teff_from_logg_L(logg, L, thickness):
    """
    Input logg (cm s-2 dex) and luminosity (Lsun) to get the WD Teff (K).
    Thickness should be one of 'thin'/'thick'.
    """
    GRID = GRIDS[thickness]
    logL = np.log10(L)
    logT = griddata((GRID['logg'], GRID['logL']), GRID['logT'], (logg, logL))
    return 10**logT

def R_from_logg_L(logg, L, thickness):
    """
    Input logg (cm s-2 dex) and luminosity (Lsun) to get the WD radius (Lsun).
    Thickness should be one of 'thin'/'thick'.
    """
    GRID = GRIDS[thickness]
    logL = np.log10(L)
    logR = griddata((GRID['logg'], GRID['logL']), GRID['logR'], (logg, logL))
    return 10**logR

def M_from_logg_L(logg, L, thickness):
    """
    Input logg (cm s-2 dex) and luminosity (Lsun) to get the WD mass (Msun).
    Thickness should be one of 'thin'/'thick'.
    """
    R = R_from_logg_L(logg, L, thickness)
    return M_from_logg_R(logg, R)

def Grv_from_M_R(M, R):
    """
    Input mass (Msun) and radius (Rsun) to get the WD Grv (km/s).
    """
    M <<= u.Msun
    R <<= u.Rsun
    rv = G*M/(c*R)
    return rv.to(u.km/u.s).value

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    GRID = GRIDS['thin']
    logg, logR = [np.array(GRID[x]) for x in ("logg", "logR")]
    M2 = M_from_logg_R(logg, 10**logR)
    for T in np.unique(np.array(GRID['logT'])):
        plt.plot(GRID['logM'], GRID['logR'], 'k.', ms=1)
    plt.show()
