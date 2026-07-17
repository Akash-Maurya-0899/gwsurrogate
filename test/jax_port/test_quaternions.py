"""M2 tests: quaternion utilities and Wigner-D matrices vs the C oracle."""

import numpy as np
import jax.numpy as jnp

from gwsurrogate.jax import quaternions as jax_quaternions
from gwsurrogate.new import precessing_surrogate as reference

RTOL = 1e-13
ATOL = 1e-15


def _random_unit_quaternions(num_times, seed):
    rng = np.random.default_rng(seed)
    quat = rng.standard_normal((4, num_times))
    return quat / np.sqrt(np.sum(quat**2, axis=0))


def _edge_case_quaternions():
    """Quaternions hitting both Wigner-D edge cases and the general case."""
    phases = np.linspace(0.0, 4 * np.pi, 17)
    # |rb| = 0 exactly (aligned-spin-like: rotation about z only)
    rb_zero = np.stack([np.cos(phases), 0 * phases, 0 * phases,
                        np.sin(phases)])
    # |ra| = 0 exactly
    ra_zero = np.stack([0 * phases, np.cos(phases), np.sin(phases),
                        0 * phases])
    general = _random_unit_quaternions(17, seed=3)
    return np.concatenate([rb_zero, ra_zero, general], axis=1)


def test_multiply_quats_and_inverse():
    q1 = _random_unit_quaternions(40, seed=1)
    q2 = _random_unit_quaternions(40, seed=2)

    product_reference = reference.multiplyQuats(q1, q2)
    product_jax = np.asarray(
        jax_quaternions.multiply_quats(jnp.asarray(q1), jnp.asarray(q2)))
    np.testing.assert_allclose(product_jax, product_reference, rtol=RTOL,
                               atol=ATOL)

    inverse_reference = reference.quatInv(np.copy(q1))
    inverse_jax = np.asarray(jax_quaternions.quat_inv(jnp.asarray(q1)))
    np.testing.assert_allclose(inverse_jax, inverse_reference, rtol=RTOL,
                               atol=ATOL)


def test_transform_time_dependent_vector():
    quat = _random_unit_quaternions(60, seed=4)
    vec = np.random.default_rng(5).standard_normal((3, 60))

    transformed_reference = reference.transformTimeDependentVector(
        np.copy(quat), np.copy(vec))
    transformed_jax = np.asarray(jax_quaternions.transform_time_dependent_vector(
        jnp.asarray(quat), jnp.asarray(vec)))
    np.testing.assert_allclose(transformed_jax, transformed_reference,
                               rtol=RTOL, atol=ATOL)


def test_wigner_d_matrices_general_case():
    quat = _random_unit_quaternions(50, seed=6)
    matrices_reference = reference._wignerD_matrices(np.copy(quat), 4)
    matrices_jax = jax_quaternions.wigner_d_matrices(jnp.asarray(quat), 4)

    for ell_index in range(3):
        np.testing.assert_allclose(np.asarray(matrices_jax[ell_index]),
                                   matrices_reference[ell_index],
                                   rtol=RTOL, atol=ATOL)


def test_wigner_d_matrices_edge_cases():
    """Time series mixing |rb|=0, |ra|=0 and general quaternions."""
    quat = _edge_case_quaternions()
    matrices_reference = reference._wignerD_matrices(np.copy(quat), 4)
    matrices_jax = jax_quaternions.wigner_d_matrices(jnp.asarray(quat), 4)

    for ell_index in range(3):
        np.testing.assert_allclose(np.asarray(matrices_jax[ell_index]),
                                   matrices_reference[ell_index],
                                   rtol=RTOL, atol=ATOL)


def test_rotate_waveform():
    rng = np.random.default_rng(8)
    num_times = 40
    quat = np.concatenate(
        [_edge_case_quaternions(), _random_unit_quaternions(num_times, 9)],
        axis=1)
    num_modes = 21  # ell_max = 4
    h_modes = rng.standard_normal((num_modes, quat.shape[1])) \
        + 1j * rng.standard_normal((num_modes, quat.shape[1]))

    rotated_reference = reference.rotateWaveform(np.copy(quat),
                                                 np.copy(h_modes))
    rotated_jax = np.asarray(jax_quaternions.rotate_waveform(
        jnp.asarray(quat), jnp.asarray(h_modes), 4))
    np.testing.assert_allclose(rotated_jax, rotated_reference, rtol=RTOL,
                               atol=1e-13)
