# MR relation

Module for interpolating the Bedard 2020 grid of evolutionary models to use as
a mass radius relation. For example calculating radius from a known Teff and
logg. There are other routines included for calculting gravitational redshifts,
and simple conversions between M, R, and logg. 

This module is compatible with astropy units. If arguments are provided without
units, sensible defaults are assumed, e.g. Msun for mass, and the result is also
returned without units. If either of the two interpolating arguments are input
with units, the output is also returned with a unit attached.

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
>>> R_from_Teff_M(15000*u.K, 1.2e30*u.kg, 'thick')
<Quantity 0.01310131 solRad>
```

# Dependencies:
* numpy
* scipy
* astropy
* pandas
