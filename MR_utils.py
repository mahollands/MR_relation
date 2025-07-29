import os.path
import functools
import numpy as np
import pandas as pd
from astropy import units as u


default_units = {
    "Mass": u.Msun,
    "Radius": u.Rsun,
    "logg": u.dex(u.cm / u.s**2),
    "Teff": u.K,
    "Age": u.Gyr,
    "Luminosity": u.Lsun,
    "Velocity": u.km / u.s,
}


class Grid:
    """
    Class for keeping track of which grid is loaded.
    """

    MR_DIR = os.path.dirname(os.path.abspath(__file__))
    valid_grids = {
        "Bedard20",
        "Fontaine01",
        "Camisassa25",
        "Althaus13ELM",
    }

    def __init__(self, grid_name="Bedard20"):
        self._GRID_NAME = grid_name

    def __call__(self, thickness):
        """
        Load a white dwarf evolutionary model grid. The results are
        cached for future use.
        """
        if thickness not in {"thick", "thin"}:
            raise ValueError("hydrogen thickness must be 'thick'/'thin'")
        if self._GRID_NAME == 'Althaus13ELM' and thickness == "thin":
            raise ValueError('Althaus ELM grid only for thick H')
        return self.get_grid(self._GRID_NAME, thickness)

    @staticmethod
    @functools.cache
    def get_grid(grid_name, thickness):
        """
        Load a white dwarf evolutionary model grid. The results are
        cached for future use.
        """
        f_grid = f"{Grid.MR_DIR}/MR_grids/{grid_name}_{thickness}.csv"
        grid = pd.read_csv(f_grid)
        grid["logM"] = np.log10(grid["Mass"])
        return grid

    def set_grid(self, grid_name):
        """
        Set the model grid to use. The default is 'Bedard20'
        """
        if grid_name not in Grid.valid_grids:
            raise ValueError(f"grid_name should be one of {Grid.valid_grids}")
        self._GRID_NAME = grid_name


def units_handling(x_kind, y_kind, z_kind):
    """
    Wraps a routine for interpolating z(x, y). If inputs x or y are astropy
    quantities then the output z is also an astropy quantity. If x or y are
    both scalars, then z is returned as a scalar assuming sensible default
    options for their units.
    """
    x_unit, y_unit, z_unit = map(default_units.get, (x_kind, y_kind, z_kind))

    def _decorator(func):
        @functools.wraps(func)
        def _wrapper(x, y, *args, **kwargs):
            x_has_unit = isinstance(x, u.Quantity)
            y_has_unit = isinstance(y, u.Quantity)
            x = x.to(x_unit) if x_has_unit else x << x_unit
            y = y.to(y_unit) if y_has_unit else y << y_unit
            z = func(x, y, *args, **kwargs)
            z = z.to(z_unit) if isinstance(z, u.Quantity) else z << z_unit
            return z if x_has_unit or y_has_unit else z.value

        return _wrapper

    return _decorator
