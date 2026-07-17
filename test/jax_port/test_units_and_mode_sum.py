"""M10 tests: physical units and mode summation vs the reference model."""

import numpy as np
import pytest

from gwsurrogate.jax import NRSur7dq4JAX

TOL_OF_PEAK = 1e-10


@pytest.fixture(scope="module")
def jax_surrogate(h5_path):
    return NRSur7dq4JAX(h5_path)


def test_mks_units_match_reference(jax_surrogate, reference_surrogate):
    q = 2.2
    chiA = np.array([0.2, 0.3, -0.1])
    chiB = np.array([-0.15, 0.1, 0.25])
    total_mass = 65.0
    dist_mpc = 400.0
    dt_seconds = 1.0 / 4096

    t_ref, h_ref, _ = reference_surrogate(
        q=q, chiA0=np.copy(chiA), chiB0=np.copy(chiB), M=total_mass,
        dist_mpc=dist_mpc, f_low=0.0, dt=dt_seconds, units="mks")
    t_jax, h_jax, _ = jax_surrogate(
        q, chiA, chiB, M=total_mass, dist_mpc=dist_mpc, f_low=0.0,
        dt=dt_seconds, units="mks")

    np.testing.assert_allclose(t_jax, t_ref, rtol=1e-14, atol=0.0)
    peak22 = np.abs(h_ref[(2, 2)]).max()
    # Exclude the final sample: the reference C++ hunt bug mis-evaluates
    # it on dt grids (see test_end_to_end.py).
    for mode in h_ref:
        max_abs_diff = np.abs(h_jax[mode][:-1] - h_ref[mode][:-1]).max()
        assert max_abs_diff <= TOL_OF_PEAK * peak22, \
            "mode %s: max diff %.3e vs peak22 %.3e" % (mode, max_abs_diff,
                                                       peak22)


def test_mode_sum_matches_reference(jax_surrogate, reference_surrogate):
    q = 1.9
    chiA = np.array([0.1, -0.3, 0.2])
    chiB = np.array([0.05, 0.2, -0.1])
    inclination = 0.7
    phi_ref = 1.2

    t_ref, h_ref, _ = reference_surrogate(
        q=q, chiA0=np.copy(chiA), chiB0=np.copy(chiB), f_low=0.0,
        inclination=inclination, phi_ref=phi_ref)
    t_jax, h_jax, _ = jax_surrogate(
        q, chiA, chiB, f_low=0.0, inclination=inclination, phi_ref=phi_ref)

    assert not isinstance(h_jax, dict)
    peak = np.abs(h_ref).max()
    max_abs_diff = np.abs(h_jax - h_ref).max()
    assert max_abs_diff <= TOL_OF_PEAK * peak, \
        "summed strain: max diff %.3e vs peak %.3e" % (max_abs_diff, peak)


def test_unit_validation(jax_surrogate):
    with pytest.raises(ValueError):
        jax_surrogate(2.0, np.zeros(3), np.zeros(3), f_low=0.0, M=60.0)
    with pytest.raises(ValueError):
        jax_surrogate(2.0, np.zeros(3), np.zeros(3), f_low=0.0, M=60.0,
                      dist_mpc=100.0)  # units left dimensionless
    with pytest.raises(ValueError):
        jax_surrogate(2.0, np.zeros(3), np.zeros(3), f_low=0.0,
                      units="mks")
