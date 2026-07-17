"""M4 tests: dynamics integrator (i0 = 0 path) vs the reference AB4/RK4."""

import numpy as np
import jax
import jax.numpy as jnp

from gwsurrogate.jax import dynamics as jax_dynamics

# Trajectory tolerance: identical scheme and arithmetic, but summation-order
# differences accumulate over the 226 AB4 steps.
TRAJECTORY_ATOL = 1e-11


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


def _integrate_both(jax_data, reference_surrogate, q, chiA, chiB,
                    init_orbphase=0.0):
    reference_dynamics = reference_surrogate._sur_dimless.dynamics_sur
    quat_ref, orbphase_ref, chiA_ref, chiB_ref, _ = reference_dynamics(
        q, np.copy(chiA), np.copy(chiB), init_orbphase=init_orbphase)

    y_of_t = jax_dynamics.integrate_dynamics_from_start(
        jax_data.dynamics, q, jnp.asarray(chiA), jnp.asarray(chiB),
        init_orbphase=init_orbphase)
    quat_jax, orbphase_jax, chiA_jax, chiB_jax = \
        jax_dynamics.unpack_dynamics_state(np.asarray(y_of_t))
    return ((quat_ref, orbphase_ref, chiA_ref, chiB_ref),
            (quat_jax, orbphase_jax, chiA_jax, chiB_jax))


def _assert_trajectories_close(reference_values, jax_values):
    labels = ("quat", "orbphase", "chiA_copr", "chiB_copr")
    for label, ref, jaxv in zip(labels, reference_values, jax_values):
        max_abs_diff = np.max(np.abs(np.asarray(jaxv) - ref))
        assert max_abs_diff < TRAJECTORY_ATOL, \
            "%s trajectory max abs diff %.3e exceeds %.1e" % (
                label, max_abs_diff, TRAJECTORY_ATOL)


def test_trajectories_match_reference_training_range(
        jax_data, reference_surrogate):
    """20 random parameter sets in the training range q in [1,4]."""
    for q, chiA, chiB in _random_parameters(20, seed=30):
        reference_values, jax_values = _integrate_both(
            jax_data, reference_surrogate, q, chiA, chiB)
        _assert_trajectories_close(reference_values, jax_values)


def test_trajectories_match_reference_extrapolated_range(
        jax_data, reference_surrogate):
    """A few cases in the extrapolated range q up to 6, |chi| up to 0.99."""
    for q, chiA, chiB in _random_parameters(4, seed=31, q_range=(4.0, 6.0),
                                            chi_max=0.99):
        reference_values, jax_values = _integrate_both(
            jax_data, reference_surrogate, q, chiA, chiB)
        _assert_trajectories_close(reference_values, jax_values)


def test_trajectories_match_reference_special_cases(
        jax_data, reference_surrogate):
    """Aligned spins, zero spins, equal mass, and nonzero init_orbphase."""
    special_cases = [
        (2.0, np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.0, -0.3]), 0.0),
        (3.0, np.zeros(3), np.zeros(3), 0.0),
        (1.0, np.array([0.3, -0.2, 0.1]), np.array([-0.1, 0.4, 0.2]), 0.0),
        (2.5, np.array([0.2, 0.5, -0.3]), np.array([0.4, -0.1, 0.2]), 1.3),
    ]
    for q, chiA, chiB, init_orbphase in special_cases:
        reference_values, jax_values = _integrate_both(
            jax_data, reference_surrogate, q, chiA, chiB,
            init_orbphase=init_orbphase)
        _assert_trajectories_close(reference_values, jax_values)


def test_jit_matches_eager(jax_data):
    """jit-compiled integration must agree with eager to float64 precision."""
    q, chiA, chiB = 2.2, np.array([0.3, 0.1, -0.4]), np.array([0.1, 0.2, 0.3])

    eager = jax_dynamics.integrate_dynamics_from_start(
        jax_data.dynamics, q, jnp.asarray(chiA), jnp.asarray(chiB))
    jitted = jax.jit(jax_dynamics.integrate_dynamics_from_start)(
        jax_data.dynamics, q, jnp.asarray(chiA), jnp.asarray(chiB))
    # jit-vs-eager differ only by XLA fusion reordering: a couple ULP of the
    # orbital phase (which grows to ~60), i.e. ~1e-14 absolute.
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager),
                               rtol=1e-14, atol=1e-14)


def test_vmap_matches_loop(jax_data):
    """vmap over a parameter batch equals a Python loop of single evals.

    The batched and single programs compile to differently-fused XLA code,
    so ULP-level reorderings accumulate over the 226 sequential AB4 steps;
    a few 1e-14 absolute is the observed scale. We require agreement an
    order of magnitude below the 1e-11 oracle tolerance. Relative
    comparison is meaningless here (spin components pass through zero).
    """
    parameters = _random_parameters(6, seed=32)
    q_batch = jnp.asarray([p[0] for p in parameters])
    chiA_batch = jnp.asarray(np.array([p[1] for p in parameters]))
    chiB_batch = jnp.asarray(np.array([p[2] for p in parameters]))

    batched = jax.vmap(
        jax_dynamics.integrate_dynamics_from_start,
        in_axes=(None, 0, 0, 0))(jax_data.dynamics, q_batch, chiA_batch,
                                 chiB_batch)
    for i, (q, chiA, chiB) in enumerate(parameters):
        single = jax_dynamics.integrate_dynamics_from_start(
            jax_data.dynamics, q, jnp.asarray(chiA), jnp.asarray(chiB))
        np.testing.assert_allclose(np.asarray(batched[i]),
                                   np.asarray(single), rtol=0.0, atol=1e-12)
