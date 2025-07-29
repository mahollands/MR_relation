import pytest
import numpy as np
from astropy import units as u
from mh.MR_relation import *


class Test_inputs:
    class Test_valid_inputs:
        def test_input1_type(self):
            with pytest.raises(TypeError):
                M_from_Teff_logg("test", 8.0, thickness="thick")

        def test_input2_type(self):
            with pytest.raises(TypeError):
                M_from_Teff_logg(10000, "test", thickness="thick")

        def test_input3_type(self):
            with pytest.raises(TypeError):
                M_from_Teff_logg(10000, 8.0, thickness=5)

        def test_input3_value(self):
            with pytest.raises(ValueError):
                M_from_Teff_logg(10000, 8.0, thickness="wrong")

    class Test_floats_vs_array:
        def test_float_float(self):
            logg = logg_from_M_R(0.6, 0.013)
            assert isinstance(logg, np.float64)

        def test_array_float(self):
            M = np.random.normal(0.6, 0.01, size=100)
            logg = logg_from_M_R(M, 0.013)
            assert isinstance(logg, np.ndarray)

        def test_float_array(self):
            R = np.random.normal(0.013, 0.001, size=100)
            logg = logg_from_M_R(0.6, R)
            assert isinstance(logg, np.ndarray)

        def test_array_array(self):
            M = np.random.normal(0.6, 0.01, size=100)
            R = np.random.normal(0.013, 0.001, size=100)
            logg = logg_from_M_R(M, R)
            assert isinstance(logg, np.ndarray)

    class Test_units:
        def test_no_units(self):
            logg = logg_from_M_R(0.6, 0.013)
            assert not isinstance(logg, u.Quantity)

        def test_units_on_1(self):
            logg = logg_from_M_R(0.6 * u.Msun, 0.013)
            assert isinstance(logg, u.Quantity)
            assert logg.unit == u.dex(u.cm / u.s**2)

        def test_units_on_2(self):
            logg = logg_from_M_R(0.6, 0.013 * u.Rsun)
            assert isinstance(logg, u.Quantity)
            assert logg.unit == u.dex(u.cm / u.s**2)

        def test_units_on_both(self):
            logg = logg_from_M_R(0.6 * u.Msun, 0.013 * u.Rsun)
            assert isinstance(logg, u.Quantity)
            assert logg.unit == u.dex(u.cm / u.s**2)

        def test_wrong_units_on_1(self):
            with pytest.raises(u.UnitsError):
                logg = logg_from_M_R(0.6 * u.K, 0.013)

        def test_wrong_units_on_2(self):
            with pytest.raises(u.UnitsError):
                logg = logg_from_M_R(0.6, 0.013 * u.erg)


class Test_model_grids:
    def test_switch_grid(self):
        # changing model grid should give different results
        tau1 = tau_from_Teff_M(10000, 0.6, "thick")
        MR_grid.set_grid("Fontaine01")
        tau2 = tau_from_Teff_M(10000, 0.6, "thick")
        assert tau1 != tau2

    def test_all_grids(self):
        MR_grid.set_grid("Bedard20")
        MR_grid.set_grid("Fontaine01")
        MR_grid.set_grid("Camisassa25")
        MR_grid.set_grid("Althaus13ELM")

        with pytest.raises(ValueError):
            MR_grid.set_grid("Something else")

    def test_Althaus_thin(self):
        MR_grid.set_grid("Althaus13ELM")
        with pytest.raises(ValueError):
            tau2 = tau_from_Teff_M(10000, 0.6, "thin")
