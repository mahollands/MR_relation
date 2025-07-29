import pytest
import numpy as np
from astropy import units as u
from mh.MR_relation import *


class Test_inputs:
    @pytest.mark.parametrize(
        "x, y, thickness, err",
        [
            ("abc", 8.00, "thick", TypeError),
            (10000, "ab", "thick", TypeError),
            (10000, 8.00, 1234567, TypeError),
            (10000, 8.00, "wrong", ValueError),
        ],
    )
    def test_input_exceptions(self, x, y, thickness, err):
        with pytest.raises(err):
            M_from_Teff_logg(x, y, thickness=thickness)

    @pytest.mark.parametrize(
        "x, y, expected_type",
        [
            (0.6, 0.013, np.float64),
            (0.6 * np.ones(100), 0.013, np.ndarray),
            (0.6, 0.013 * np.ones(100), np.ndarray),
            (0.6 * np.ones(100), 0.013 * np.ones(100), np.ndarray),
        ],
    )
    def test_floats_vs_array(self, x, y, expected_type):
        logg = logg_from_M_R(x, y)
        assert isinstance(logg, expected_type)

    class Test_units:
        def test_no_units(self):
            logg = logg_from_M_R(0.6, 0.013)
            assert not isinstance(logg, u.Quantity)

        @pytest.mark.parametrize(
            "x, y",
            [
                (0.6 * u.Msun, 0.013),
                (0.6, 0.013 * u.Rsun),
                (0.6 * u.Msun, 0.013 * u.Rsun),
            ],
        )
        def test_with_units(self, x, y):
            logg = logg_from_M_R(x, y)
            assert isinstance(logg, u.Quantity)
            assert logg.unit == u.dex(u.cm / u.s**2)

        @pytest.mark.parametrize(
            "x, y",
            [
                (0.6 * u.K, 0.013),
                (0.6, 0.013 * u.erg),
                (0.6 * u.K, 0.013 * u.erg),
            ],
        )
        def test_wrong_units(self, x, y):
            with pytest.raises(u.UnitsError):
                logg = logg_from_M_R(x, y)


class Test_model_grids:
    @pytest.mark.parametrize(
        "value",
        [
            "Bedard20",
            "Fontaine01",
            "Camisassa25",
            "Althaus13ELM",
        ],
    )
    def test_valid_grids(self, value):
        MR_grid.set_grid(value)

    def test_invalid_grid(self):
        with pytest.raises(ValueError):
            MR_grid.set_grid("Something else")

    def test_Althaus_thin(self):
        MR_grid.set_grid("Althaus13ELM")
        with pytest.raises(ValueError):
            tau2 = tau_from_Teff_M(10000, 0.6, "thin")

    def test_switch_grid(self):
        # changing model grid should give different results
        MR_grid.set_grid("Bedard20")
        tau1 = tau_from_Teff_M(10000, 0.6, "thick")
        MR_grid.set_grid("Fontaine01")
        tau2 = tau_from_Teff_M(10000, 0.6, "thick")
        assert tau1 != tau2
