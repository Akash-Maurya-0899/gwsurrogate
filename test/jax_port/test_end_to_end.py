"""M7 tests: end-to-end NRSur7dq4JAX waveforms vs the reference model."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from gwsurrogate.jax import NRSur7dq4JAX

# End-to-end budget: per-mode max abs difference relative to the peak
# amplitude of the (2,2) mode. The dominant contributions are the AB4
# trajectory accumulation (~1e-12) amplified by the orbital phase entering
# mode phases.
END_TO_END_TOL_OF_PEAK22 = 1e-10


@pytest.fixture(scope="module")
def jax_surrogate(h5_path):
    return NRSur7dq4JAX(h5_path)


def _random_parameters(num_cases, seed, q_range=(1.0, 4.0), chi_max=0.8):
    rng = np.random.default_rng(seed)
    parameters = []
    for _ in range(num_cases):
        q = rng.uniform(*q_range)
        chiA = rng.uniform(-1.0, 1.0, 3)
        chiA *= rng.uniform(0.0, chi_max) / np.linalg.norm(chiA)
        chiB = rng.uniform(-1.0, 1.0, 3)
        chiB *= rng.uniform(0.0, chi_max) / np.linalg.norm(chiB)
        parameters.append((q, chiA, chiB))
    return parameters


def _compare_mode_dicts(h_jax, h_reference, tolerance_of_peak22,
                        context=""):
    peak22 = np.abs(h_reference[(2, 2)]).max()
    assert set(h_jax) == set(h_reference)
    for mode in h_reference:
        max_abs_diff = np.abs(h_jax[mode] - h_reference[mode]).max()
        assert max_abs_diff <= tolerance_of_peak22 * peak22, \
            "%smode %s: max diff %.3e exceeds %.1e of peak22 %.3e" % (
                context, mode, max_abs_diff, tolerance_of_peak22, peak22)


def test_waveforms_match_reference(jax_surrogate, reference_surrogate):
    """Random parameter sets, full grid, all modes, vs the C oracle."""
    cases = _random_parameters(12, seed=50) + [
        # Aligned spins (hits the Wigner-D edge case), zero spins, q=1.
        (2.0, np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.0, -0.3])),
        (3.0, np.zeros(3), np.zeros(3)),
        (1.0, np.array([0.3, -0.2, 0.1]), np.array([-0.1, 0.4, 0.2])),
    ]
    for q, chiA, chiB in cases:
        t_ref, h_ref, _ = reference_surrogate(
            q=q, chiA0=np.copy(chiA), chiB0=np.copy(chiB), f_low=0.0)
        t_jax, h_jax, _ = jax_surrogate(q, chiA, chiB, f_low=0.0)
        np.testing.assert_allclose(t_jax, t_ref, rtol=0.0, atol=0.0)
        _compare_mode_dicts(h_jax, h_ref, END_TO_END_TOL_OF_PEAK22,
                            context="q=%.3f: " % q)


def test_waveforms_match_reference_ellmax_and_options(
        jax_surrogate, reference_surrogate):
    """ellMax=2/3 and init_orbphase/init_quat options."""
    q = 2.4
    chiA = np.array([0.35, -0.1, 0.2])
    chiB = np.array([-0.2, 0.15, 0.3])
    init_quat = np.array([0.9, 0.1, -0.3, 0.2])
    init_quat /= np.linalg.norm(init_quat)
    opts = {"init_orbphase": 0.7, "init_quat": init_quat}

    for ell_max in (2, 3):
        _, h_ref, _ = reference_surrogate(
            q=q, chiA0=np.copy(chiA), chiB0=np.copy(chiB), f_low=0.0,
            ellMax=ell_max, precessing_opts=dict(opts))
        _, h_jax, _ = jax_surrogate(q, chiA, chiB, f_low=0.0,
                                    ellMax=ell_max,
                                    precessing_opts=dict(opts))
        _compare_mode_dicts(h_jax, h_ref, END_TO_END_TOL_OF_PEAK22,
                            context="ellMax=%d: " % ell_max)


def test_waveforms_match_reference_on_user_grids(jax_surrogate,
                                                 reference_surrogate):
    """dt and times output grids."""
    q = 1.7
    chiA = np.array([0.1, 0.4, -0.3])
    chiB = np.array([0.25, -0.05, 0.1])

    t_ref, h_ref, _ = reference_surrogate(
        q=q, chiA0=np.copy(chiA), chiB0=np.copy(chiB), f_low=0.0, dt=0.5)
    t_jax, h_jax, _ = jax_surrogate(q, chiA, chiB, f_low=0.0, dt=0.5)
    np.testing.assert_allclose(t_jax, t_ref, rtol=0.0, atol=1e-12)
    # The reference mis-evaluates the FINAL dt-grid sample when the
    # second-to-last sample is >= 2 knot intervals behind: the C++ hunt
    # interval-search bug (see CLAUDE.md / test_spline.py) picks the wrong
    # cubic segment there (verified: scipy's natural spline agrees with
    # the JAX value to 0.0 and differs from C++ by ~1e-6). Exclude it.
    h_ref_trimmed = {mode: series[:-1] for mode, series in h_ref.items()}
    h_jax_trimmed = {mode: series[:-1] for mode, series in h_jax.items()}
    _compare_mode_dicts(h_jax_trimmed, h_ref_trimmed,
                        END_TO_END_TOL_OF_PEAK22, context="dt grid: ")

    times = np.linspace(-4000.0, 90.0, 3000)
    t_ref, h_ref, _ = reference_surrogate(
        q=q, chiA0=np.copy(chiA), chiB0=np.copy(chiB), f_low=0.0,
        times=times)
    t_jax, h_jax, _ = jax_surrogate(q, chiA, chiB, f_low=0.0, times=times)
    np.testing.assert_allclose(t_jax, t_ref, rtol=0.0, atol=0.0)
    _compare_mode_dicts(h_jax, h_ref, END_TO_END_TOL_OF_PEAK22,
                        context="times grid: ")


def test_dynamics_match_reference(jax_surrogate, reference_surrogate):
    """return_dynamics output on the default and dt grids."""
    q = 2.1
    chiA = np.array([0.3, 0.2, -0.1])
    chiB = np.array([-0.15, 0.1, 0.35])
    opts = {"return_dynamics": True}

    for grid_kwargs in ({}, {"dt": 1.0}):
        _, _, dyn_ref = reference_surrogate(
            q=q, chiA0=np.copy(chiA), chiB0=np.copy(chiB), f_low=0.0,
            precessing_opts=dict(opts), **grid_kwargs)
        _, _, dyn_jax = jax_surrogate(q, chiA, chiB, f_low=0.0,
                                      precessing_opts=dict(opts),
                                      **grid_kwargs)
        for key in dyn_ref:
            max_abs_diff = np.abs(dyn_jax[key] - dyn_ref[key]).max()
            assert max_abs_diff < 1e-10, \
                "dynamics[%s] (%s): max diff %.3e" % (key, grid_kwargs,
                                                      max_abs_diff)


def test_batch_matches_single(jax_surrogate):
    """vmapped batch evaluation equals per-parameter evaluation."""
    parameters = _random_parameters(5, seed=51)
    q_batch = np.array([p[0] for p in parameters])
    chiA_batch = np.array([p[1] for p in parameters])
    chiB_batch = np.array([p[2] for p in parameters])

    h_batch = np.asarray(jax_surrogate.eval_modes_batch(
        q_batch, chiA_batch, chiB_batch))
    assert h_batch.shape == (5, 21, len(jax_surrogate.t_coorb))

    for i, (q, chiA, chiB) in enumerate(parameters):
        _, h_single, _ = jax_surrogate(q, chiA, chiB, f_low=0.0)
        h_single_array = np.stack(
            [h_single[(ell, m)] for ell in range(2, 5)
             for m in range(-ell, ell + 1)])
        max_abs_diff = np.abs(h_batch[i] - h_single_array).max()
        assert max_abs_diff < 1e-12, \
            "batch element %d: max diff %.3e vs single" % (i, max_abs_diff)


def test_unsupported_options_raise(jax_surrogate):
    with pytest.raises(ValueError):
        # f_low below the frequency at the first node (f_low > 0 itself
        # is supported since M11; range violations still raise)
        jax_surrogate(2.0, np.zeros(3), np.zeros(3), f_low=0.001)
    with pytest.raises(ValueError):
        # f_ref far above omega_ref_max/pi
        jax_surrogate(2.0, np.zeros(3), np.zeros(3), f_low=0.0, f_ref=0.5)
    with pytest.raises(ValueError):
        # f_ref below the frequency at the first node
        jax_surrogate(2.0, np.zeros(3), np.zeros(3), f_low=0.0,
                      f_ref=1e-4)
    with pytest.raises(ValueError):
        jax_surrogate(0.5, np.zeros(3), np.zeros(3), f_low=0.0)
    with pytest.raises(ValueError):
        jax_surrogate(2.0, np.array([0.0, 0.0, 1.2]), np.zeros(3),
                      f_low=0.0)


def test_gradients_are_finite_and_match_finite_differences(jax_surrogate):
    """Differentiability smoke test: d|h22(t_i)|/dq via jax.grad."""
    data = jax_surrogate.data
    from gwsurrogate.jax.surrogate import _evaluate_dimensionless_modes

    chiA0 = jnp.asarray([0.2, 0.1, -0.3])
    chiB0 = jnp.asarray([0.1, -0.2, 0.25])
    init_quat = jnp.asarray([1.0, 0.0, 0.0, 0.0])
    time_index = 1000  # mid-inspiral sample
    mode_index = 4  # (2, 2)

    def h22_amplitude(q):
        h, _, _ = _evaluate_dimensionless_modes(
            data, q, chiA0, chiB0, init_quat, 0.0, 4)
        return jnp.abs(h[mode_index, time_index])

    gradient = jax.grad(h22_amplitude)(2.0)
    assert np.isfinite(float(gradient))

    step = 1e-5
    finite_difference = (h22_amplitude(2.0 + step)
                         - h22_amplitude(2.0 - step)) / (2 * step)
    np.testing.assert_allclose(float(gradient), float(finite_difference),
                               rtol=1e-4)
