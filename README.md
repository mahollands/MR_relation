# MR relation

Module for interpolating white dwarf evolutionary models. For example
calculating the white dwarf radius from a known Teff and logg, or the white
dwarf cooling age from mass and radius.

By default the evolutionary models of Bedard et al. (2020) are used. Other
model grids are included (see references) with more to be added over time. Let
me know if you feel your evolutionary model grid should be included!

Generic routines are also included for calculting gravitational redshifts, and
conversions between M/R/logg or L/R/Teff.

This module is compatible with astropy units. If arguments are provided without
units, sensible defaults are assumed, e.g. Msun for mass, and the result is also
returned without units. If either of the two interpolating arguments are input
with units, the output is also returned with a unit attached.

A sub-module `correct_3D` is included for applying 3D corrections to parameters
from 1D models, or the inverse.

## Examples:
```python
#No units provided
>>> R_from_Teff_M(15000, 0.6, 'thick')
0.013157301496275331

#Units on one argument
>>> R_from_Teff_M(15000*u.K, 0.6, 'thick')
<Quantity 0.0131573 solRad>

#Units on both arguments
>>> R_from_Teff_M(15000*u.K, 0.6*u.Msun, 'thick')
<Quantity 0.0131573 solRad>

#Non default unit used on an argument
>>> R_from_Teff_M(15000*u.K, 1.2E30*u.kg, 'thick')
<Quantity 0.01310131 solRad>

#Inputs can mix and match linear and log scales if units are provided
>>> M_from_logg_R(1E8*u.cm/u.s**2, -1.9*u.dex(u.Rsun))
<Quantity 0.57800603 solMass>
>>> M_from_logg_R(9.81*u.m/u.s**2, 0*u.dex(u.Rearth)).to(u.Mearth)
<Quantity 1.00118406 earthMass>

#Switching to a different grid (see references below)
>>> tau_from_Teff_M(10000*u.K, 0.6*u.Msun, 'thick')
<Quantity 0.6328035 Gyr>
>>> MR_relation.CHOSEN_GRID = 'Fontaine01'
>>> tau_from_Teff_M(10000*u.K, 0.6*u.Msun, 'thick')
<Quantity 0.60274991 Gyr>
```

## Dependencies:
* numpy
* scipy
* astropy
* pandas

## References:
* `Bedard20`: [A. Bedard et al, ApJ 901, 26 (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...901...93B/abstract)
* `Fontaine01`: [G. Fontaine et al, PASP 113, 409 (2001)](https://ui.adsabs.harvard.edu/abs/2001PASP..113..409F/abstract)
* `Camisassa25`: [M. Camisassa, Astronomische Nachrichten 346, e20240118 (2025)](https://ui.adsabs.harvard.edu/abs/2025AN....34640118C/abstract)
