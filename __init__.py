"""
Utilities for interpolating white dwarf Mass-Radius relations and evolutionary
models. The default models are from Bedard et al. (2020)
(adsabs.harvard.edu/abs/2020ApJ...901...93B) using the CO_Hthick (mH=1e-4) and
CO_Hthin (mH=1e-10) model grids.

Generic routines are also provided for simple conversions, i.e. between
(radius, mass, logg), (luminosity, Teff, radius), and (mass, radius,
gravitational redshift).

This module also handles astropy units (optionally). If inputs are scalars,
then the output is also a scalar, but with sensible default units assumed (see
below). If any of the inputs are astropy quantities, then conversions are
performed and the output is given as a quantity (with the default unit).

Default units:
    * Mass: Msun
    * Radius: Rsun
    * logg: dex(cm s-2)
    * Teff: K
    * Cooling age: Gyr
    * Luminosity: Lsun
    * Gravitational redshift: km/s
"""

__author__ = "Mark Hollands"
__email__ = "M.Hollands.1@warwick.ac.uk"

from .MR_functions import *
from .correct_3D import *
