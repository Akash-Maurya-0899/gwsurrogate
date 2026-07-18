"""M11 tests: start-frequency (f_low > 0) support vs the reference.

f_low > 0 maps to omega_low = pi*f_low, which the reference inverts to a
start time t_low (``DynamicsSurrogate._get_t_from_omega``) and then uses
to truncate the output grid (``PrecessingSurrogate.__call__``
:1082-1096). These tests check t_low itself, the truncated default-grid
and dt-grid waveforms, the f_ref != f_low combination, mks units, the
batched API, and the error paths.

Known reference quirks accounted for here (see CLAUDE.md):
- On the DEFAULT grid with f_low > 0 the reference returns a truncated
  time array but FULL-length mode arrays (upstream inconsistency). The
  JAX side truncates both consistently; mode values are compared against
  the tail of the reference arrays.
- On dt grids the final output sample can be mis-evaluated by the C++
  ``hunt`` off-by-one, so dt-grid comparisons exclude the final sample.
"""

import numpy as np
import pytest

from gwsurrogate.jax import NRSur7dq4JAX

# t_low is a smooth functional of omega-fit values that agree with the C
# fits to ~1e-13 relative; observed |t_low_jax - t_low_ref| ~ 7e-12 on
# t_low ~ -3400.
T_LOW_ATOL = 1e-9
END_TO_END_TOL_OF_PEAK22 = 1e-10
# dt-grid samples are evaluated on grids shifted by |t_low diff| <= 1e-9,
# adding |dh/dt|*1e-9 ~ 1e-10 absolute on O(0.1) mode amplitudes;
# observed max diff ~1e-12.
DT_GRID_TOL_OF_PEAK22 = 1e-10
BATCH_VS_SINGLE_ATOL = 1e-12

TEST_CASES = [
    # (q, chiA0, chiB0, f_low)
    (2.3, [0.3, -0.2, 0.4], [-0.1, 0.2, -0.3], 6e-3),
    (1.2, [0.1, 0.3, -0.5], [0.4, -0.2, 0.1], 8e-3),
    (3.7, [-0.2, 0.1, 0.6], [0.0, 0.0, -0.4], 7e-3),
]


@pytest.fixture(scope="module")
def jax_surrogate(h5_path):
    return NRSur7dq4JAX(h5_path)


def _reference_t_low(reference_surrogate, q, chiA, chiB, f_low):
    """t_low straight from the reference omega -> t map."""
    reference_dynamics = reference_surrogate._sur_dimless.dynamics_sur
    return reference_dynamics._get_t_from_omega(
        np.pi * f_low, q, np.asarray(chiA), np.asarray(chiB), 0.0, None)


@pytest.mark.parametrize("q, chiA, chiB, f_low", TEST_CASES)
def test_t_low_matches_reference(jax_surrogate, reference_surrogate, q,
                                 chiA, chiB, f_low):
    t_low_ref = _reference_t_low(reference_surrogate, q, chiA, chiB, f_low)
    t_low_jax = jax_surrogate.start_times([q], [chiA], [chiB], f_low)[0]
    assert abs(t_low_jax - t_low_ref) < T_LOW_ATOL


@pytest.mark.parametrize("q, chiA, chiB, f_low", TEST_CASES)
def test_default_grid_truncation_matches_reference(
        jax_surrogate, reference_surrogate, q, chiA, chiB, f_low):
    """Default (coorbital) grid: times match exactly; modes match the
    tail of the reference's (un-truncated, see module docstring) arrays."""
    t_ref_out, h_ref, _ = reference_surrogate(
        q=q, chiA0=np.array(chiA), chiB0=np.array(chiB), f_low=f_low)
    t_jax_out, h_jax, _ = jax_surrogate(q, chiA, chiB, f_low=f_low)

    num_kept = len(t_ref_out)
    assert len(t_jax_out) == num_kept
    assert len(t_jax_out) < len(jax_surrogate.t_coorb)  # truncation happened
    np.testing.assert_array_equal(t_jax_out, t_ref_out)

    peak22 = np.abs(h_ref[(2, 2)]).max()
    for mode in h_ref:
        assert len(h_jax[mode]) == num_kept  # consistent, unlike upstream
        max_abs_diff = np.abs(h_jax[mode] - h_ref[mode][-num_kept:]).max()
        assert max_abs_diff <= END_TO_END_TOL_OF_PEAK22 * peak22, \
            "mode %s: max diff %.3e" % (mode, max_abs_diff)


@pytest.mark.parametrize("q, chiA, chiB, f_low", TEST_CASES)
def test_dt_grid_matches_reference(jax_surrogate, reference_surrogate, q,
                                   chiA, chiB, f_low):
    dt = 0.5
    t_ref_out, h_ref, _ = reference_surrogate(
        q=q, chiA0=np.array(chiA), chiB0=np.array(chiB), f_low=f_low, dt=dt)
    t_jax_out, h_jax, _ = jax_surrogate(q, chiA, chiB, f_low=f_low, dt=dt)

    assert len(t_jax_out) == len(t_ref_out)
    np.testing.assert_allclose(t_jax_out, t_ref_out, rtol=0,
                               atol=T_LOW_ATOL)

    # Final sample excluded: C++ hunt() off-by-one (see module docstring).
    peak22 = np.abs(h_ref[(2, 2)]).max()
    for mode in h_ref:
        max_abs_diff = np.abs(h_jax[mode][:-1] - h_ref[mode][:-1]).max()
        assert max_abs_diff <= DT_GRID_TOL_OF_PEAK22 * peak22, \
            "mode %s: max diff %.3e" % (mode, max_abs_diff)


def test_f_ref_distinct_from_f_low_matches_reference(jax_surrogate,
                                                     reference_surrogate):
    """f_ref > f_low: t_low comes from omega_low, t_ref from omega_ref."""
    q, chiA, chiB = 2.3, [0.3, -0.2, 0.4], [-0.1, 0.2, -0.3]
    f_low, f_ref, dt = 6e-3, 8e-3, 0.5

    t_ref_out, h_ref, _ = reference_surrogate(
        q=q, chiA0=np.array(chiA), chiB0=np.array(chiB), f_low=f_low,
        f_ref=f_ref, dt=dt)
    t_jax_out, h_jax, _ = jax_surrogate(q, chiA, chiB, f_low=f_low,
                                        f_ref=f_ref, dt=dt)

    assert len(t_jax_out) == len(t_ref_out)
    np.testing.assert_allclose(t_jax_out, t_ref_out, rtol=0,
                               atol=T_LOW_ATOL)
    peak22 = np.abs(h_ref[(2, 2)]).max()
    for mode in h_ref:
        max_abs_diff = np.abs(h_jax[mode][:-1] - h_ref[mode][:-1]).max()
        assert max_abs_diff <= DT_GRID_TOL_OF_PEAK22 * peak22, \
            "mode %s: max diff %.3e" % (mode, max_abs_diff)


def test_mks_units_with_f_low_in_hz(jax_surrogate, reference_surrogate):
    """Physical units: f_low = 20 Hz for a 70 Msun binary at 400 Mpc."""
    q, chiA, chiB = 2.0, [0.2, 0.1, 0.3], [-0.1, 0.0, 0.2]
    total_mass, dist_mpc, dt = 70.0, 400.0, 1.0 / 4096

    t_ref_out, h_ref, _ = reference_surrogate(
        q=q, chiA0=np.array(chiA), chiB0=np.array(chiB), f_low=20.0,
        M=total_mass, dist_mpc=dist_mpc, units="mks", dt=dt)
    t_jax_out, h_jax, _ = jax_surrogate(
        q, chiA, chiB, f_low=20.0, M=total_mass, dist_mpc=dist_mpc,
        units="mks", dt=dt)

    assert len(t_jax_out) == len(t_ref_out)
    peak22 = np.abs(h_ref[(2, 2)]).max()
    for mode in h_ref:
        max_abs_diff = np.abs(h_jax[mode][:-1] - h_ref[mode][:-1]).max()
        assert max_abs_diff <= DT_GRID_TOL_OF_PEAK22 * peak22, \
            "mode %s: max diff %.3e" % (mode, max_abs_diff)


def test_return_dynamics_truncated_consistently(jax_surrogate):
    """Default grid + f_low + return_dynamics: all lengths match times."""
    q, chiA, chiB = 2.3, [0.3, -0.2, 0.4], [-0.1, 0.2, -0.3]
    t_out, h, dynamics = jax_surrogate(
        q, chiA, chiB, f_low=6e-3,
        precessing_opts={"return_dynamics": True})
    num_kept = len(t_out)
    assert num_kept < len(jax_surrogate.t_coorb)
    assert dynamics["orbphase"].shape == (num_kept,)
    assert dynamics["q_copr"].shape == (4, num_kept)
    assert dynamics["chiA"].shape == (num_kept, 3)
    assert dynamics["chiB_copr"].shape == (num_kept, 3)


def test_batched_f_low_matches_single(jax_surrogate):
    q_values = np.array([1.5, 2.3, 3.1])
    chiA = np.array([[0.3, -0.2, 0.4], [0.1, 0.2, 0.3], [-0.2, 0.1, 0.5]])
    chiB = np.array([[-0.1, 0.2, -0.3], [0.0, -0.1, 0.2], [0.2, 0.0, -0.1]])
    f_low = 6e-3

    t_lows = jax_surrogate.start_times(q_values, chiA, chiB, f_low)
    assert np.all(t_lows > jax_surrogate.t_coorb[0])
    assert np.all(t_lows < 0.0)

    shared_times = np.linspace(t_lows.max(), 90.0, 4000)
    h_batch = np.asarray(jax_surrogate.eval_modes_batch(
        q_values, chiA, chiB, f_low=f_low, times=shared_times))
    for i in range(len(q_values)):
        _, h_single, _ = jax_surrogate(q_values[i], chiA[i], chiB[i],
                                       f_low=f_low, times=shared_times)
        h_single_array = np.stack(
            [h_single[mode] for mode in sorted(h_single)])
        max_abs_diff = np.abs(h_batch[i] - h_single_array).max()
        assert max_abs_diff < BATCH_VS_SINGLE_ATOL, \
            "element %d: max diff %.3e" % (i, max_abs_diff)


def test_start_times_scalar_matches_batch(jax_surrogate):
    q, chiA, chiB, f_low = 2.3, [0.3, -0.2, 0.4], [-0.1, 0.2, -0.3], 6e-3
    t_low_one = jax_surrogate.start_times([q], [chiA], [chiB], f_low)
    t_low_two = jax_surrogate.start_times(
        [q, q], [chiA, chiA], [chiB, chiB], [f_low, f_low])
    assert t_low_one.shape == (1,)
    np.testing.assert_array_equal(t_low_two, [t_low_one[0], t_low_one[0]])


def test_f_low_error_paths(jax_surrogate):
    q, chiA, chiB = 2.0, [0.2, 0.1, 0.3], [-0.1, 0.0, 0.2]
    with pytest.raises(ValueError, match="f_ref cannot be lower"):
        jax_surrogate(q, chiA, chiB, f_low=6e-3, f_ref=5e-3)
    with pytest.raises(ValueError, match="too small"):
        jax_surrogate(q, chiA, chiB, f_low=1e-4)
    with pytest.raises(ValueError, match="too large"):
        jax_surrogate(q, chiA, chiB, f_low=0.2)
    with pytest.raises(ValueError, match="f_ref cannot be lower"):
        jax_surrogate.eval_modes_batch(
            np.array([q]), np.array([chiA]), np.array([chiB]),
            f_low=6e-3, f_ref=np.array([5e-3]))
