"""
3D corrections from Tremblay et al 2013
"""
from math import exp
import functools
from scipy.optimize import root

__all__ = [
    "correct_1D_to_3D",
    "decorrect_3D_to_1D",
]

c = (
    1.0947335E-03,
    -1.8716231E-01,
    1.9350009E-02,
    6.4821613E-01,
    -2.2863187E-01,
    5.8699232E-01,
    -1.0729871E-01,
    1.1009070E-01
)

d = (
    7.5209868E-04,
    -9.2086619E-01,
    3.1253746E-01,
    -1.0348176E+01,
    6.5854716E-01,
    4.2849862E-01,
    -8.8982873E-02,
    1.0199718E+01,
    4.9277883E-02,
    -8.6543477E-01,
    3.6232756E-03,
    -5.8729354E-02
)

@functools.cache
def correct_1D_to_3D(Teff, logg):
    """
    Correction to 3D parameters from 1D ML2/alpha=0.8
    """
    TX, gX = (Teff-10000)/1000, logg-8.0

    T_term1 = c[1] + c[6]*TX + c[7]*gX
    T_term2 = (c[2]+c[4]*TX+c[5]*gX)*(TX-c[3])
    dT = c[0] + T_term1 * exp(-T_term2**2)

    d_term1 = d[5]*(TX-d[6])**2
    d_term2 = (d[8] + d[10]*TX + d[11]*gX)*(TX-d[9])
    d_term3 = (TX - (d[3] + d[7]*exp(-d_term2**2)))
    dg = d[0] + d[4]*exp(-d_term1) + d[1]*exp(-d[2]*d_term3**2)

    return Teff+dT*1000, logg+dg

def target_3D1D(params, Teff, logg):
    T1D, g1D = params
    T3D, g3D = correct_1D_to_3D(T1D, g1D)
    return Teff-T3D, logg-g3D

@functools.cache
def decorrect_3D_to_1D(Teff, logg):
    """
    Remove correction from 3D parameters to 1D ML2/alpha=0.8. This is function
    is useful for fitting spectra and photometry simulateously with 1D models.
    I.e. use true (3D) parameters for photometric fit, and apply decorrection
    to get corresponding 1D parameters for fitting a normalised spectrum.
    """
    result = root(target_3D1D, x0=(Teff, logg), args=(Teff, logg))
    return tuple(result['x'])
