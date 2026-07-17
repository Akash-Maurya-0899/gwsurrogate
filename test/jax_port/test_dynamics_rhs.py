"""M3 tests: dynamics RHS building blocks vs the C oracle."""

import numpy as np
import jax.numpy as jnp

from gwsurrogate.jax import dynamics as jax_dynamics
from gwsurrogate.precessing_utils import _utils

RTOL = 1e-13
ATOL = 1e-15


def _random_states(num_states, seed):
    """Realistic random dynamics states: unit quat, phase, bounded spins."""
    rng = np.random.default_rng(seed)
    states = np.empty((num_states, 11))
    quats = rng.standard_normal((num_states, 4))
    states[:, 0:4] = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    states[:, 4] = rng.uniform(0.0, 60.0, num_states)
    for start in (5, 8):
        spins = rng.uniform(-0.5, 0.5, (num_states, 3))
        states[:, start:start + 3] = spins
    return states


def test_get_ds_fit_x_matches_c():
    for i, y in enumerate(_random_states(30, seed=20)):
        q = np.random.default_rng(i).uniform(1.0, 6.0)
        reference_x = _utils.get_ds_fit_x(y, q)
        jax_x = np.asarray(jax_dynamics.get_ds_fit_x(jnp.asarray(y), q))
        np.testing.assert_allclose(jax_x, reference_x, rtol=RTOL, atol=ATOL)


def test_assemble_dydt_matches_c():
    rng = np.random.default_rng(21)
    for y in _random_states(30, seed=22):
        omega_orb_xy = rng.standard_normal(2)
        omega = rng.standard_normal()
        chiA_dot = rng.standard_normal(3)
        chiB_dot = rng.standard_normal(3)

        reference_dydt = _utils.assemble_dydt(y, omega_orb_xy, omega,
                                              chiA_dot, chiB_dot)
        fit_values = np.concatenate([omega_orb_xy, [omega], chiA_dot,
                                     chiB_dot])
        jax_dydt = np.asarray(jax_dynamics.assemble_dydt(
            jnp.asarray(y), jnp.asarray(fit_values)))
        np.testing.assert_allclose(jax_dydt, reference_dydt, rtol=RTOL,
                                   atol=ATOL)


def test_normalize_y_matches_c():
    rng = np.random.default_rng(23)
    for y in _random_states(30, seed=24):
        norm_chiA = rng.uniform(0.01, 0.99)
        norm_chiB = rng.uniform(0.01, 0.99)
        reference_y = _utils.normalize_y(y, norm_chiA, norm_chiB)
        jax_y = np.asarray(jax_dynamics.normalize_y(jnp.asarray(y),
                                                    norm_chiA, norm_chiB))
        np.testing.assert_allclose(jax_y, reference_y, rtol=RTOL, atol=ATOL)


def test_normalize_y_zero_spin_is_nan_free():
    """Zero-magnitude spins must stay exactly zero (no NaN), unlike raw C."""
    y = _random_states(1, seed=25)[0]
    y[5:8] = 0.0
    jax_y = np.asarray(jax_dynamics.normalize_y(jnp.asarray(y), 0.0, 0.3))
    assert np.isfinite(jax_y).all()
    assert (jax_y[5:8] == 0.0).all()


def test_ab4_dy_matches_c():
    rng = np.random.default_rng(26)
    for _ in range(30):
        k1, k2, k3, k4 = rng.standard_normal((4, 11))
        dt1, dt2, dt3, dt4 = rng.uniform(0.1, 10.0, 4)
        reference_dy = _utils.ab4_dy(k1, k2, k3, k4, dt1, dt2, dt3, dt4)
        jax_dy = np.asarray(jax_dynamics.ab4_dy(
            jnp.asarray(k1), jnp.asarray(k2), jnp.asarray(k3),
            jnp.asarray(k4), dt1, dt2, dt3, dt4))
        np.testing.assert_allclose(jax_dy, reference_dy, rtol=RTOL, atol=ATOL)


def test_rhs_matches_reference_at_random_nodes(jax_data, reference_surrogate):
    """Full dydt vs DynamicsSurrogate.get_time_deriv_from_index."""
    reference_dynamics = reference_surrogate._sur_dimless.dynamics_sur
    rng = np.random.default_rng(27)
    states = _random_states(20, seed=28)
    node_indices = rng.integers(0, len(reference_dynamics.t), 20)

    for y, node_index in zip(states, node_indices):
        q = rng.uniform(1.0, 6.0)
        reference_dydt = reference_dynamics.get_time_deriv_from_index(
            int(node_index), q, np.copy(y))
        jax_dydt = np.asarray(jax_dynamics.dynamics_rhs_at_node(
            jax_data.dynamics, int(node_index), q, jnp.asarray(y)))
        np.testing.assert_allclose(jax_dydt, reference_dydt, rtol=RTOL,
                                   atol=ATOL)
