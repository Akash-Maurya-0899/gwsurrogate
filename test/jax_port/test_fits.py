"""M1 tests: padded fit kernel and fit-coordinate map vs the C oracle."""

import numpy as np
import jax
import jax.numpy as jnp

from gwsurrogate.jax import fits as jax_fits
from gwsurrogate.precessing_utils import _utils

# Kernel comparisons: identical arithmetic up to reduction order. The tiny
# atol floor covers XLA's pairwise summation vs C's sequential summation
# (differences of a few 1e-17 on values whose terms are O(0.1)).
RTOL_KERNEL = 1e-13
ATOL_KERNEL = 1e-15

FIT_SETTINGS = (
    jax_fits.NRSUR7DQ4_Q_FIT_OFFSET,
    jax_fits.NRSUR7DQ4_Q_FIT_SLOPE,
    jax_fits.NRSUR7DQ4_Q_MAX_BF_ORDER,
    jax_fits.NRSUR7DQ4_CHI_MAX_BF_ORDER,
)


def _eval_fit_with_c(coefs, bf_orders, fit_params):
    """Reference evaluation via the C extension."""
    q_fit_offset, q_fit_slope, q_max_bf_order, chi_max_bf_order = FIT_SETTINGS
    return _utils.eval_fit(bf_orders, coefs, fit_params, q_fit_offset,
                           q_fit_slope, q_max_bf_order, chi_max_bf_order)


def test_dynamics_fits_match_c_oracle(jax_data, raw_dynamics_fit_tables,
                                      random_fit_param_points):
    """All 233 nodes x 9 dynamics fits at 50 random points, vs C."""
    evaluate_all = jax.jit(jax_fits.evaluate_fits)
    dynamics = jax_data.dynamics

    for fit_params in random_fit_param_points:
        jax_values = np.asarray(evaluate_all(
            dynamics.fit_coefs, dynamics.fit_bf_orders,
            jnp.asarray(fit_params)))  # (233, 9)

        c_values = np.array([
            [_eval_fit_with_c(coefs, bf_orders, fit_params)
             for coefs, bf_orders in node_fits]
            for node_fits in raw_dynamics_fit_tables])

        np.testing.assert_allclose(jax_values, c_values, rtol=RTOL_KERNEL,
                                   atol=ATOL_KERNEL)


def test_coorbital_fits_match_c_oracle(jax_data, raw_coorbital_fit_tables,
                                       random_fit_param_points):
    """All 42 components x EI-node fits at 50 random points, vs C.

    Padded EI-node slots (beyond a component's real node count) are checked
    to evaluate to exactly 0.
    """
    evaluate_all = jax.jit(jax_fits.evaluate_fits)
    coorb = jax_data.coorb

    for fit_params in random_fit_param_points[:10]:
        jax_values = np.asarray(evaluate_all(
            coorb.node_fit_coefs, coorb.node_fit_bf_orders,
            jnp.asarray(fit_params)))  # (42, N_max)

        for c, component_fits in enumerate(raw_coorbital_fit_tables):
            c_values = np.array([
                _eval_fit_with_c(coefs, bf_orders, fit_params)
                for coefs, bf_orders in component_fits])
            num_real_nodes = len(component_fits)
            np.testing.assert_allclose(jax_values[c, :num_real_nodes],
                                       c_values, rtol=RTOL_KERNEL,
                                       atol=ATOL_KERNEL)
            assert (jax_values[c, num_real_nodes:] == 0.0).all(), \
                "Padded EI-node fits must evaluate to exactly zero."


def test_extra_padding_is_exact(jax_data, random_fit_param_points):
    """Growing the zero-padding must not change fit values at all."""
    dynamics = jax_data.dynamics
    extra_coefs = jnp.pad(dynamics.fit_coefs, ((0, 0), (0, 0), (0, 5)))
    extra_orders = jnp.pad(dynamics.fit_bf_orders,
                           ((0, 0), (0, 0), (0, 5), (0, 0)))

    fit_params = jnp.asarray(random_fit_param_points[0])
    baseline = jax_fits.evaluate_fits(dynamics.fit_coefs,
                                      dynamics.fit_bf_orders, fit_params)
    padded = jax_fits.evaluate_fits(extra_coefs, extra_orders, fit_params)
    assert (np.asarray(baseline) == np.asarray(padded)).all()


def test_get_fit_params_matches_reference(reference_surrogate):
    """jnp fit-coordinate map vs the closure inside the loaded NRSur7dq4."""
    reference_get_fit_params = \
        reference_surrogate._sur_dimless.dynamics_sur._get_fit_params

    rng = np.random.default_rng(seed=7)
    for _ in range(50):
        q = rng.uniform(1.0, 6.0)
        chiA = rng.uniform(-0.8, 0.8, 3)
        chiB = rng.uniform(-0.8, 0.8, 3)
        x = np.concatenate([[q], chiA, chiB])

        reference_values = reference_get_fit_params(np.copy(x))
        jax_values = np.asarray(jax_fits.get_fit_params(jnp.asarray(x)))
        np.testing.assert_allclose(jax_values, reference_values,
                                   rtol=1e-15, atol=1e-15)


def test_get_fit_params_batched():
    """Batched application equals per-row application."""
    rng = np.random.default_rng(seed=11)
    x_batch = np.column_stack(
        [rng.uniform(1.0, 6.0, 8)] + [rng.uniform(-0.8, 0.8, 8)
                                      for _ in range(6)])

    batched = np.asarray(jax_fits.get_fit_params(jnp.asarray(x_batch)))
    rowwise = np.array([
        np.asarray(jax_fits.get_fit_params(jnp.asarray(row)))
        for row in x_batch])
    assert (batched == rowwise).all()
