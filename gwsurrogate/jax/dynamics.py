"""Precession dynamics for the NRSur7dq4 JAX port.

Ports the ODE right-hand side and integration helpers of the reference
implementation:

- ``get_ds_fit_x``, ``assemble_dydt``, ``normalize_y``, ``ab4_dy`` from
  gwsurrogate/precessing_utils/src/precessing_utils.c,
- the RHS orchestration ``DynamicsSurrogate.get_time_deriv_from_index`` from
  gwsurrogate/new/precessing_surrogate.py.

The dynamics state vector y has length 11:
    y[0:4]  coprecessing-frame quaternion
    y[4]    orbital phase in the coprecessing frame
    y[5:8]  chiA in the coprecessing frame
    y[8:11] chiB in the coprecessing frame
"""

import jax
import jax.numpy as jnp

from . import fits as jax_fits
from . import quaternions as jax_quaternions


def get_ds_fit_x(y, q):
    """Fit input for the dynamics surrogate at state y.

    Port of ``get_ds_fit_x`` (precessing_utils.c:417): rotates the
    coprecessing-frame spins into the coorbital frame by the orbital phase.
    Returns x = [q, chiA_coorb (3), chiB_coorb (3)].
    """
    sin_phase = jnp.sin(y[4])
    cos_phase = jnp.cos(y[4])
    return jnp.stack([
        q,
        y[5] * cos_phase + y[6] * sin_phase,
        -1 * y[5] * sin_phase + y[6] * cos_phase,
        y[7],
        y[8] * cos_phase + y[9] * sin_phase,
        -1 * y[8] * sin_phase + y[9] * cos_phase,
        y[10]])


def assemble_dydt(y, fit_values):
    """Assemble the ODE right-hand side from the 9 dynamics fit values.

    Port of the dydt assembly in ``eval_fit_batch_dydt``
    (precessing_utils.c:462, identical arithmetic to ``assemble_dydt``
    :547). fit_values is ordered [omega_orb_x, omega_orb_y, omega,
    chiAdot (3), chiBdot (3)], all with vector components in the coorbital
    frame.
    """
    cos_phase = jnp.cos(y[4])
    sin_phase = jnp.sin(y[4])

    # Rotate Omega^coorb (x, y) into the coprecessing frame.
    omega_orb_x = fit_values[0] * cos_phase - fit_values[1] * sin_phase
    omega_orb_y = fit_values[0] * sin_phase + fit_values[1] * cos_phase

    return jnp.stack([
        # Quaternion derivative: dqdt = 0.5 * quat * [0, ooxy_x, ooxy_y, 0]
        -0.5 * y[1] * omega_orb_x - 0.5 * y[2] * omega_orb_y,
        -0.5 * y[3] * omega_orb_y + 0.5 * y[0] * omega_orb_x,
        0.5 * y[3] * omega_orb_x + 0.5 * y[0] * omega_orb_y,
        0.5 * y[1] * omega_orb_y - 0.5 * y[2] * omega_orb_x,
        # Orbital phase derivative
        fit_values[2],
        # Spin derivatives, rotated from the coorbital to the coprecessing
        # frame
        fit_values[3] * cos_phase - fit_values[4] * sin_phase,
        fit_values[3] * sin_phase + fit_values[4] * cos_phase,
        fit_values[5],
        fit_values[6] * cos_phase - fit_values[7] * sin_phase,
        fit_values[6] * sin_phase + fit_values[7] * cos_phase,
        fit_values[8]])


def normalize_y(y, norm_chiA, norm_chiB):
    """Renormalize the quaternion and rescale spins to fixed magnitudes.

    Port of ``normalize_y`` (precessing_utils.c:351), with double-``where``
    guards so zero-magnitude spins stay exactly zero instead of producing
    NaN (and gradients stay finite).
    """
    quat_norm = jnp.sqrt(y[0]**2 + y[1]**2 + y[2]**2 + y[3]**2)

    current_norm_chiA = jnp.sqrt(y[5]**2 + y[6]**2 + y[7]**2)
    current_norm_chiB = jnp.sqrt(y[8]**2 + y[9]**2 + y[10]**2)
    safe_norm_chiA = jnp.where(current_norm_chiA > 0.0,
                               current_norm_chiA, 1.0)
    safe_norm_chiB = jnp.where(current_norm_chiB > 0.0,
                               current_norm_chiB, 1.0)
    scale_chiA = jnp.where(current_norm_chiA > 0.0,
                           norm_chiA / safe_norm_chiA, 0.0)
    scale_chiB = jnp.where(current_norm_chiB > 0.0,
                           norm_chiB / safe_norm_chiB, 0.0)

    return jnp.concatenate([
        y[0:4] / quat_norm,
        y[4:5],
        y[5:8] * scale_chiA,
        y[8:11] * scale_chiB])


def ab4_dy(k1, k2, k3, k4, dt1, dt2, dt3, dt4):
    """Variable-step 4th-order Adams-Bashforth update.

    Port of ``ab4_dy`` (precessing_utils.c:603): given the RHS evaluations
    k1..k4 at the four previous nodes (k4 most recent) and the three
    preceding step sizes dt1..dt3, returns the increment of y over the new
    step dt4.
    """
    dt12 = dt1 + dt2
    dt123 = dt12 + dt3
    dt23 = dt2 + dt3

    D1 = dt1 * dt12 * dt123
    D2 = dt1 * dt2 * dt23
    D3 = dt2 * dt12 * dt3

    B41 = dt3 * dt23 / D1
    B42 = -1 * dt3 * dt123 / D2
    B43 = dt23 * dt123 / D3
    B4 = B41 + B42 + B43

    C41 = (dt23 + dt3) / D1
    C42 = -1 * (dt123 + dt3) / D2
    C43 = (dt123 + dt23) / D3
    C4 = C41 + C42 + C43

    A = k4
    B = k4 * B4 - k1 * B41 - k2 * B42 - k3 * B43
    C = k4 * C4 - k1 * C41 - k2 * C42 - k3 * C43
    D = (k4 - k1) / D1 - (k4 - k2) / D2 + (k4 - k3) / D3

    return dt4 * (A + dt4 * (0.5 * B + dt4 * (C / 3.0 + dt4 * 0.25 * D)))


def dynamics_rhs_at_node(dynamics_data, node_index, q, y):
    """dydt at a dynamics time node.

    Port of ``DynamicsSurrogate.get_time_deriv_from_index``
    (precessing_surrogate.py:273): build the fit input from the current
    state, map to fit coordinates, evaluate the 9 stacked fits of the node
    (a gather into the padded tables, so node_index may be traced), and
    assemble dydt.
    """
    fit_x = get_ds_fit_x(y, q)
    fit_params = jax_fits.get_fit_params(fit_x)
    fit_values = jax_fits.evaluate_fits(
        dynamics_data.fit_coefs[node_index],
        dynamics_data.fit_bf_orders[node_index],
        fit_params)  # (9,)
    return assemble_dydt(y, fit_values)


def integrate_dynamics_from_start(dynamics_data, q, chiA0, chiB0,
                                  init_quat=None, init_orbphase=0.0):
    """Integrate the precession dynamics from the first time node (i0 = 0).

    Port of the ``i0 == 0`` branch of ``DynamicsSurrogate.__call__``
    (precessing_surrogate.py:461): three RK4 bootstrap steps over the
    paired half-step nodes (``_initial_RK4`` :542) followed by variable-step
    AB4 over the remaining nodes (``_integrate_forward`` :607), expressed as
    a ``lax.scan``.

    Arguments:
        dynamics_data: DynamicsData pytree (padded fit tables and grids).
        q: mass ratio (scalar, may be traced).
        chiA0, chiB0: reference-time spins, shape (3,), in the coorbital
            frame convention of the reference implementation.
        init_quat: initial coprecessing-frame quaternion, shape (4,);
            identity if None.
        init_orbphase: initial orbital phase in the coprecessing frame.

    Returns y_of_t with shape (n_dyn, 11) on the ``t_dynamics`` grid
    (n_dyn = len(t_ds) - 3).
    """
    # Rotate spins from the lalsimulation source frame into the surrogate
    # frame (DynamicsSurrogate.__call__ :431).
    chiA0 = jax_quaternions.rotate_spin(chiA0, -1 * init_orbphase)
    chiB0 = jax_quaternions.rotate_spin(chiB0, -1 * init_orbphase)
    norm_chiA = jnp.sqrt(jnp.sum(chiA0**2))
    norm_chiB = jnp.sqrt(jnp.sum(chiB0**2))

    if init_quat is None:
        init_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
    y0 = jnp.concatenate([init_quat,
                          jnp.asarray(init_orbphase, dtype=jnp.float64)[None],
                          chiA0, chiB0])

    t_ds = dynamics_data.t_ds

    # --- RK4 bootstrap: 3 steps over the paired half-step nodes ---
    bootstrap_states = [y0]
    bootstrap_k1 = []
    bootstrap_full_dts = []
    y = y0
    for i in range(3):
        half_dt = t_ds[2 * i + 1] - t_ds[2 * i]  # diff_t[2*i]
        k1 = dynamics_rhs_at_node(dynamics_data, 2 * i, q, y)
        k2 = dynamics_rhs_at_node(dynamics_data, 2 * i + 1, q,
                                  y + half_dt * k1)
        k3 = dynamics_rhs_at_node(dynamics_data, 2 * i + 1, q,
                                  y + half_dt * k2)
        k4 = dynamics_rhs_at_node(dynamics_data, 2 * i + 2, q,
                                  y + 2 * half_dt * k3)
        y_next = y + (half_dt / 3.) * (k1 + 2 * k2 + 2 * k3 + k4)
        y = normalize_y(y_next, norm_chiA, norm_chiB)
        bootstrap_states.append(y)
        bootstrap_k1.append(k1)
        bootstrap_full_dts.append(2 * half_dt)

    # --- AB4 main loop as a scan over the remaining nodes ---
    # At output index i (starting from 3), k4 is the RHS at t_ds node i + 3
    # and the step is diff_t[i + 3] (_integrate_forward :625).
    k4_node_indices = jnp.arange(6, len(dynamics_data.t_ds) - 1)
    dt4_values = t_ds[7:] - t_ds[6:-1]  # diff_t[6:]

    def ab4_step(carry, step_inputs):
        y, k1, k2, k3, dt1, dt2, dt3 = carry
        node_index, dt4 = step_inputs
        k4 = dynamics_rhs_at_node(dynamics_data, node_index, q, y)
        y_next = y + ab4_dy(k1, k2, k3, k4, dt1, dt2, dt3, dt4)
        y_new = normalize_y(y_next, norm_chiA, norm_chiB)
        return (y_new, k2, k3, k4, dt2, dt3, dt4), y_new

    initial_carry = (y, bootstrap_k1[0], bootstrap_k1[1], bootstrap_k1[2],
                     bootstrap_full_dts[0], bootstrap_full_dts[1],
                     bootstrap_full_dts[2])
    _, scanned_states = jax.lax.scan(
        ab4_step, initial_carry, (k4_node_indices, dt4_values))

    return jnp.concatenate(
        [jnp.stack(bootstrap_states), scanned_states], axis=0)


def unpack_dynamics_state(y_of_t):
    """Split y_of_t (n_dyn, 11) into the reference return layout:
    quat (4, n_dyn), orbphase (n_dyn,), chiA_copr / chiB_copr (n_dyn, 3)."""
    return (y_of_t[:, 0:4].T, y_of_t[:, 4], y_of_t[:, 5:8], y_of_t[:, 8:11])
