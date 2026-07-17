"""M6 tests: coorbital-frame waveform modes vs the reference implementation."""

import numpy as np
import jax.numpy as jnp
import pytest

from gwsurrogate.jax import coorb as jax_coorb

RTOL_OF_PEAK = 1e-12


def _random_coorbital_spins(num_times, seed):
    """Smooth, physically plausible coorbital-frame spin time series."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, num_times)
    spins = np.empty((num_times, 3))
    for component in range(3):
        amplitude = rng.uniform(0.05, 0.4)
        frequency = rng.uniform(0.5, 4.0)
        phase = rng.uniform(0.0, 2 * np.pi)
        offset = rng.uniform(-0.3, 0.3)
        spins[:, component] = offset + amplitude * np.sin(
            2 * np.pi * frequency * t + phase)
    return spins


@pytest.mark.parametrize("ell_max", [2, 3, 4])
def test_coorbital_modes_match_reference(jax_data, reference_surrogate,
                                         ell_max):
    reference_coorb = reference_surrogate._sur_dimless.coorb_sur
    num_times = len(np.asarray(jax_data.coorb.t_coorb))

    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        q = rng.uniform(1.0, 4.0)
        chiA_coorb = _random_coorbital_spins(num_times, seed=200 + seed)
        chiB_coorb = _random_coorbital_spins(num_times, seed=300 + seed)

        reference_modes = reference_coorb(q, np.copy(chiA_coorb),
                                          np.copy(chiB_coorb), ellMax=ell_max)
        jax_modes = np.asarray(jax_coorb.coorbital_waveform_modes(
            jax_data.coorb, q, jnp.asarray(chiA_coorb),
            jnp.asarray(chiB_coorb), ell_max=ell_max))

        assert jax_modes.shape == reference_modes.shape
        for mode_index in range(reference_modes.shape[0]):
            peak = np.abs(reference_modes[mode_index]).max()
            max_abs_diff = np.abs(jax_modes[mode_index]
                                  - reference_modes[mode_index]).max()
            assert max_abs_diff <= RTOL_OF_PEAK * peak + 1e-16, \
                "mode %d: max diff %.3e exceeds %.1e of peak %.3e" % (
                    mode_index, max_abs_diff, RTOL_OF_PEAK, peak)
