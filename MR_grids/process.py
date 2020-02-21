import numpy as np
import glob
from astropy import units as u
from astropy.constants import G

fnames = sorted(glob.iglob("CO_*0210"))
fnames = sorted(glob.iglob("CO_*0204"))

def process_models(fname):
  Mass = int(fname.split("_")[1][:3])/100
  with open(fname, 'r') as F:
    Nmod = int(F.readline())
    for i in range(Nmod):
      line = F.readline()
      imod, Teff, logg, rayon, age, lum = line.split()  
      line = F.readline()
      line = F.readline()
      Teff = float(Teff)
      logg = float(logg)
      age  = float(age)
      lum  = float(lum)

      if Teff <= 0. or age <= 0.:
        continue

      M = Mass * u.Msun
      g = 10**logg * u.cm / u.s**2
      R = np.sqrt(G * M / g).to(u.Rsun)
      L = lum * u.erg/u.s
      L = L.to(u.Lsun)

      logT   = np.log10(Teff)
      logtau = np.log10(age)
      logR   = np.log10(R.value)
      logL   = np.log10(L.value)

      mod = Mass, logT, logg, logR, logtau, logL
      print(",".join(str(param) for param in mod))
      
print("Mass,logT,logg,logR,logtau,logL")
for fname in fnames:
  process_models(fname)
