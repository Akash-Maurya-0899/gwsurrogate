"""M9 tests: reference-frequency (f_ref/omega_ref) support vs the reference.

The reference integrator has three branches depending on where the
reference time lands on the output grid (i0 == 0 / 0 < i0 <= 2 / i0 > 2);
omega_ref values are chosen via the omega fit at specific nodes so every
branch is exercised, and the test asserts that coverage.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from gwsurrogate.jax import NRSur7dq4JAX, dynamics as jax_dynamics

TRAJECTORY_ATOL = 1e-11
END_TO_END_TOL_OF_PEAK22 = 1e-10


@pytest.fixture(scope="module")
def jax_surrogate(h5_path):
    return NRSur7dq4JAX(h5_path)


def _omega_ref_hitting_node(reference_surrogate, target_full_node, q, chiA,
                            chiB):
    """omega_ref that makes t_ref land (nearly) on a chosen output node."""
    reference_dynamics = reference_surrogate._sur_dimless.dynamics_sur
    full_nodes = [0, 2, 4] + list(range(6, len(reference_dynamics.t)))
    y0 = np.concatenate([[1.0, 0.0, 0.0, 0.0, 0.0], chiA, chiB])
    return reference_dynamics.get_omega(full_nodes[target_full_node], q, y0)


def _branch_of(reference_surrogate, omega_ref, q, chiA, chiB):
    reference_dynamics = reference_surrogate._sur_dimless.dynamics_sur
    t_ref = reference_dynamics._get_t_from_omega(omega_ref, q, chiA, chiB,
                                                 0.0, None)
    times = np.append(reference_dynamics.t[:6:2], reference_dynamics.t[6:])
    i0 = int(np.argmin(np.abs(times - t_ref)))
    return "A" if i0 == 0 else ("C" if i0 <= 2 else "B")


def test_dynamics_at_reference_match_all_branches(jax_data,
                                                  reference_surrogate):
    q = 2.3
    chiA = np.array([0.25, -0.1, 0.3])
    chiB = np.array([0.1, 0.2, -0.15])
    reference_dynamics = reference_surrogate._sur_dimless.dynamics_sur

    covered_branches = set()
    for target_node in (0, 1, 2, 3, 10, 60, 150):
        omega_ref = _omega_ref_hitting_node(reference_surrogate,
                                            target_node, q, chiA, chiB)
        covered_branches.add(_branch_of(reference_surrogate, omega_ref, q,
                                        chiA, chiB))

        quat_ref, orbphase_ref, chiA_ref, chiB_ref, _ = reference_dynamics(
            q, np.copy(chiA), np.copy(chiB), omega_ref=omega_ref)
        y_of_t = np.asarray(jax_dynamics.integrate_dynamics_at_reference(
            jax_data.dynamics, q, jnp.asarray(chiA), jnp.asarray(chiB),
            omega_ref))
        quat_jax, orbphase_jax, chiA_jax, chiB_jax = \
            jax_dynamics.unpack_dynamics_state(y_of_t)

        for label, ref, jaxv in (("quat", quat_ref, quat_jax),
                                 ("orbphase", orbphase_ref, orbphase_jax),
                                 ("chiA", chiA_ref, chiA_jax),
                                 ("chiB", chiB_ref, chiB_jax)):
            max_abs_diff = np.abs(np.asarray(jaxv) - ref).max()
            assert max_abs_diff < TRAJECTORY_ATOL, \
                "node %d %s: max diff %.3e" % (target_node, label,
                                               max_abs_diff)

    assert covered_branches == {"A", "B", "C"}, \
        "Expected all three reference-index branches, got %s" \
        % covered_branches


def test_waveforms_at_reference_match(jax_surrogate, reference_surrogate):
    """End-to-end f_ref waveforms vs the reference across branches."""
    q = 1.8
    chiA = np.array([0.3, 0.15, -0.2])
    chiB = np.array([-0.1, 0.25, 0.1])

    covered_branches = set()
    for target_node in (0, 1, 8, 120):
        omega_ref = _omega_ref_hitting_node(reference_surrogate,
                                            target_node, q, chiA, chiB)
        covered_branches.add(_branch_of(reference_surrogate, omega_ref, q,
                                        chiA, chiB))
        f_ref = omega_ref / np.pi

        _, h_ref, _ = reference_surrogate(
            q=q, chiA0=np.copy(chiA), chiB0=np.copy(chiB), f_low=0.0,
            f_ref=f_ref)
        _, h_jax, _ = jax_surrogate(q, chiA, chiB, f_low=0.0, f_ref=f_ref)

        peak22 = np.abs(h_ref[(2, 2)]).max()
        for mode in h_ref:
            max_abs_diff = np.abs(h_jax[mode] - h_ref[mode]).max()
            assert max_abs_diff <= END_TO_END_TOL_OF_PEAK22 * peak22, \
                "node %d mode %s: max diff %.3e" % (target_node, mode,
                                                    max_abs_diff)

    assert covered_branches >= {"A", "B"}


def test_batched_f_ref_matches_single(jax_surrogate, reference_surrogate):
    q_values = np.array([1.5, 2.5, 3.5])
    chiA = np.array([[0.2, 0.1, -0.1], [0.0, 0.3, 0.2], [0.1, -0.2, 0.3]])
    chiB = np.array([[0.1, -0.1, 0.2], [0.2, 0.0, -0.3], [0.0, 0.1, 0.1]])
    f_ref = np.array([
        _omega_ref_hitting_node(reference_surrogate, node, q, a, b) / np.pi
        for node, q, a, b in zip((5, 40, 100), q_values, chiA, chiB)])

    h_batch = np.asarray(jax_surrogate.eval_modes_batch(
        q_values, chiA, chiB, f_ref=f_ref))
    for i in range(3):
        _, h_single, _ = jax_surrogate(q_values[i], chiA[i], chiB[i],
                                       f_low=0.0, f_ref=f_ref[i])
        h_single_array = np.stack(
            [h_single[(ell, m)] for ell in range(2, 5)
             for m in range(-ell, ell + 1)])
        max_abs_diff = np.abs(h_batch[i] - h_single_array).max()
        assert max_abs_diff < 1e-12, \
            "batch element %d: max diff %.3e" % (i, max_abs_diff)
