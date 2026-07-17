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

import numpy as np
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
    NaN (and gradients stay finite). The quaternion norm is also guarded:
    the masked scans of the reference-frequency path evaluate this on
    not-yet-initialized (all-zero) states whose results are discarded.
    """
    quat_norm = jnp.sqrt(y[0]**2 + y[1]**2 + y[2]**2 + y[3]**2)
    quat_norm = jnp.where(quat_norm > 0.0, quat_norm, 1.0)

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


###############################################################################
# Reference-frequency (omega_ref / t_ref) support.
#
# Ports the t_ref != None machinery of DynamicsSurrogate
# (precessing_surrogate.py): _get_t_from_omega (:326), _initialize (:508),
# the off-node RHS get_time_deriv (:297), the off-node RK4 steps
# (_one_forward_RK4_step :560, _one_backward_RK4_step :583), and the three
# integration branches of __call__ (:461-499) expressed as lax.switch over
# masked full-length AB4 scans.
#
# NOTE on two reference quirks replicated verbatim (they matter for exact
# agreement): the forward-restart seed RHS of the i0 > 2 branch is evaluated
# with t_ds node index i0 for the state y_of_t[i0-3] (:479), and the
# backward-restart seed of the 0 < i0 <= 2 branch with node index i0+3 for
# y_of_t[i0+3] (:495) — even where the state's true t_ds node differs.


def _natural_spline_four_points(knot_times, values, t):
    """Natural cubic spline through 4 knots, evaluated at one time.

    Same arithmetic as _spline_interp.cpp specialized to n = 4 (interior
    system is 2x2). values has shape (4, 11); returns shape (11,).
    """
    h = knot_times[1:] - knot_times[:-1]        # (3,)
    inv_h = 1.0 / h

    diag0 = 2.0 * (h[0] + h[1])
    factor = h[1] / diag0
    diag1 = 2.0 * (h[1] + h[2]) - factor * h[1]

    rhs0 = 6.0 * (values[2] * inv_h[1]
                  - values[1] * (inv_h[1] + inv_h[0])
                  + values[0] * inv_h[0])
    raw1 = 6.0 * (values[3] * inv_h[2]
                  - values[2] * (inv_h[2] + inv_h[1])
                  + values[1] * inv_h[1])
    rhs1 = raw1 - factor * rhs0

    c2 = rhs1 / diag1
    c1 = (rhs0 - h[1] * c2) / diag0
    second_derivs = jnp.stack(
        [jnp.zeros_like(c1), c1, c2, jnp.zeros_like(c1)])  # (4, 11)

    idx = jnp.clip(jnp.searchsorted(knot_times, t, side="right") - 1, 0, 2)
    dt_local = t - knot_times[idx]
    ci = second_derivs[idx]
    ci1 = second_derivs[idx + 1]
    d_coeff = (ci1 - ci) * inv_h[idx] * (1.0 / 6.0)
    b = -h[idx] * (1.0 / 6.0) * (2.0 * ci + ci1) \
        + (values[idx + 1] - values[idx]) * inv_h[idx]
    return values[idx] + dt_local * (b + dt_local * (0.5 * ci
                                                     + dt_local * d_coeff))


def dynamics_rhs_at_time(dynamics_data, t, q, y):
    """dydt at an arbitrary time via 4-node cubic interpolation of the RHS.

    Port of ``DynamicsSurrogate.get_time_deriv`` (precessing_surrogate.py:297).
    Callers must ensure t lies inside the dynamics time range (the reference
    raises; under jit we cannot).
    """
    t_ds = dynamics_data.t_ds
    nearest = jnp.argmin(jnp.abs(t_ds - t))
    imin = jnp.where(t > t_ds[nearest], nearest - 1, nearest - 2)
    imin = jnp.clip(imin, 0, t_ds.shape[0] - 4)

    dydt_at_nodes = jnp.stack([
        dynamics_rhs_at_node(dynamics_data, imin + offset, q, y)
        for offset in range(4)])  # (4, 11)
    knot_times = jax.lax.dynamic_slice(t_ds, (imin,), (4,))
    return _natural_spline_four_points(knot_times, dydt_at_nodes, t)


def reference_time_from_omega(dynamics_data, omega_ref, q, chiA0, chiB0,
                              init_orbphase, init_quat):
    """Map a reference orbital frequency to a reference time.

    Port of ``DynamicsSurrogate._get_t_from_omega``
    (precessing_surrogate.py:326): evaluate the omega fit at every full
    node (skipping the three half-step nodes) with the fixed initial state
    y0, find the first node where omega exceeds omega_ref (scanning from
    node 1, matching the reference while-loop even for non-monotonic
    omega), and linearly interpolate in omega. Range validation is the
    caller's job (host-side).

    chiA0/chiB0 here are already rotated into the surrogate frame.
    """
    y0 = jnp.concatenate([
        init_quat, jnp.asarray(init_orbphase, dtype=jnp.float64)[None],
        chiA0, chiB0])
    fit_params = jax_fits.get_fit_params(get_ds_fit_x(y0, q))

    full_node_indices = _full_node_indices(dynamics_data)
    omega_at_nodes = jax_fits.evaluate_fits(
        dynamics_data.fit_coefs[full_node_indices, 2],
        dynamics_data.fit_bf_orders[full_node_indices, 2],
        fit_params)  # (n_dyn,)

    crossed = omega_at_nodes > omega_ref
    crossed = crossed.at[0].set(False)  # the search starts at index 1
    imax = jnp.argmax(crossed)
    omega_min = omega_at_nodes[imax - 1]
    omega_max = omega_at_nodes[imax]
    t_min = dynamics_data.t_dynamics[imax - 1]
    t_max = dynamics_data.t_dynamics[imax]
    return (t_min * (omega_max - omega_ref)
            + t_max * (omega_ref - omega_min)) / (omega_max - omega_min)


def _full_node_indices(dynamics_data):
    """Static t_ds indices of the full nodes (skips half-nodes 1, 3, 5)."""
    num_nodes = dynamics_data.t_ds.shape[0]
    return np.concatenate([[0, 2, 4], np.arange(6, num_nodes)])


def omega_at_first_node(dynamics_data, q, chiA0, chiB0, init_orbphase,
                        init_quat):
    """omega fit at node 0 with the initial state (for range validation)."""
    y0 = jnp.concatenate([
        init_quat, jnp.asarray(init_orbphase, dtype=jnp.float64)[None],
        chiA0, chiB0])
    fit_params = jax_fits.get_fit_params(get_ds_fit_x(y0, q))
    return jax_fits.evaluate_fits(dynamics_data.fit_coefs[0, 2],
                                  dynamics_data.fit_bf_orders[0, 2],
                                  fit_params)


def _forward_rk4_step_at_index(dynamics_data, q, y, output_index,
                               norm_chiA, norm_chiB):
    """One forward RK4 step from a (traced) output-grid index.

    Port of ``_one_forward_RK4_step`` (precessing_surrogate.py:560),
    including the half-node index mapping for output_index < 3.
    """
    t_ds = dynamics_data.t_ds
    i_t = jnp.where(output_index < 3, 2 * output_index, output_index + 3)
    t1 = t_ds[i_t]
    t2 = jnp.where(output_index < 3, t_ds[i_t + 2], t_ds[i_t + 1])
    half_dt = 0.5 * (t2 - t1)

    k1 = dynamics_rhs_at_time(dynamics_data, t1, q, y)
    k2 = dynamics_rhs_at_time(dynamics_data, t1 + half_dt, q,
                              y + half_dt * k1)
    k3 = dynamics_rhs_at_time(dynamics_data, t1 + half_dt, q,
                              y + half_dt * k2)
    k4 = dynamics_rhs_at_time(dynamics_data, t2, q, y + 2 * half_dt * k3)
    y_next = y + (half_dt / 3.) * (k1 + 2 * k2 + 2 * k3 + k4)
    return normalize_y(y_next, norm_chiA, norm_chiB), k1


def _backward_rk4_step_at_index(dynamics_data, q, y, output_index,
                                norm_chiA, norm_chiB):
    """One backward RK4 step from a (traced) output-grid index.

    Port of ``_one_backward_RK4_step`` (precessing_surrogate.py:583). Note
    the reference uses ``i0 <= 3`` (not < 3) for the half-node handling of
    the target time t2.
    """
    t_ds = dynamics_data.t_ds
    i_t = jnp.where(output_index < 3, 2 * output_index, output_index + 3)
    t1 = t_ds[i_t]
    t2 = jnp.where(output_index <= 3, t_ds[i_t - 2], t_ds[i_t - 1])
    half_dt = 0.5 * (t2 - t1)

    k1 = dynamics_rhs_at_time(dynamics_data, t1, q, y)
    k2 = dynamics_rhs_at_time(dynamics_data, t1 + half_dt, q,
                              y + half_dt * k1)
    k3 = dynamics_rhs_at_time(dynamics_data, t1 + half_dt, q,
                              y + half_dt * k2)
    k4 = dynamics_rhs_at_time(dynamics_data, t2, q, y + 2 * half_dt * k3)
    y_next = y + (half_dt / 3.) * (k1 + 2 * k2 + 2 * k3 + k4)
    return normalize_y(y_next, norm_chiA, norm_chiB), k1


def _masked_forward_ab4(dynamics_data, q, y_of_t, norm_chiA, norm_chiB,
                        start_output_index, k_init, dt_init):
    """Forward AB4 over output steps 3..n_dyn-2, active from start index.

    Port of ``_integrate_forward`` (precessing_surrogate.py:607) as a
    full-length scan carrying the whole y_of_t array; steps below
    start_output_index leave the carry untouched, so the k/dt history
    equals (k_init, dt_init) at activation.
    """
    num_output = dynamics_data.t_dynamics.shape[0]
    dt_dynamics = jnp.asarray(dynamics_data.dt_dynamics)

    def step(carry, i_output):
        y_grid, k1, k2, k3, dt1, dt2, dt3 = carry
        active = i_output >= start_output_index
        y_current = y_grid[i_output]
        k4 = dynamics_rhs_at_node(dynamics_data, i_output + 3, q, y_current)
        dt4 = dt_dynamics[i_output]
        y_next = normalize_y(
            y_current + ab4_dy(k1, k2, k3, k4, dt1, dt2, dt3, dt4),
            norm_chiA, norm_chiB)
        y_grid = jnp.where(active, y_grid.at[i_output + 1].set(y_next),
                           y_grid)
        k1, k2, k3 = [jnp.where(active, new, old) for new, old in
                      ((k2, k1), (k3, k2), (k4, k3))]
        dt1, dt2, dt3 = [jnp.where(active, new, old) for new, old in
                         ((dt2, dt1), (dt3, dt2), (dt4, dt3))]
        return (y_grid, k1, k2, k3, dt1, dt2, dt3), None

    carry = (y_of_t, k_init[0], k_init[1], k_init[2],
             dt_init[0], dt_init[1], dt_init[2])
    (y_of_t, *_), _ = jax.lax.scan(step, carry,
                                   jnp.arange(3, num_output - 1))
    return y_of_t


def _masked_backward_ab4(dynamics_data, q, y_of_t, norm_chiA, norm_chiB,
                         first_backward_index, k_init, dt_init):
    """Backward AB4 filling output indices first_backward_index-1 .. 0.

    Port of ``_integrate_backward`` (precessing_surrogate.py:641) as a
    full-length reversed scan; steps at or above first_backward_index are
    inactive.
    """
    num_output = dynamics_data.t_dynamics.shape[0]
    dt_dynamics = jnp.asarray(dynamics_data.dt_dynamics)

    def step(carry, i_output):
        y_grid, k1, k2, k3, dt1, dt2, dt3 = carry
        active = i_output < first_backward_index
        node_index = jnp.where(i_output < 2, 2 + 2 * i_output, i_output + 4)
        y_current = y_grid[i_output + 1]
        k4 = dynamics_rhs_at_node(dynamics_data, node_index, q, y_current)
        dt4 = dt_dynamics[i_output]
        y_next = normalize_y(
            y_current - ab4_dy(k1, k2, k3, k4, dt1, dt2, dt3, dt4),
            norm_chiA, norm_chiB)
        y_grid = jnp.where(active, y_grid.at[i_output].set(y_next), y_grid)
        k1, k2, k3 = [jnp.where(active, new, old) for new, old in
                      ((k2, k1), (k3, k2), (k4, k3))]
        dt1, dt2, dt3 = [jnp.where(active, new, old) for new, old in
                         ((dt2, dt1), (dt3, dt2), (dt4, dt3))]
        return (y_grid, k1, k2, k3, dt1, dt2, dt3), None

    carry = (y_of_t, k_init[0], k_init[1], k_init[2],
             dt_init[0], dt_init[1], dt_init[2])
    (y_of_t, *_), _ = jax.lax.scan(step, carry,
                                   jnp.arange(num_output - 1)[::-1])
    return y_of_t


def _bootstrap_rk4_and_forward(dynamics_data, q, y_start, norm_chiA,
                               norm_chiB):
    """The i0 == 0 integration: on-node RK4 bootstrap + forward AB4 scan.

    Same scheme as integrate_dynamics_from_start but writing into a full
    y_of_t array (for shape compatibility with the lax.switch branches).
    """
    t_ds = dynamics_data.t_ds
    num_output = dynamics_data.t_dynamics.shape[0]

    y_of_t = jnp.zeros((num_output, 11))
    y_of_t = y_of_t.at[0].set(y_start)
    y = y_start
    k_init = []
    dt_init = []
    for i in range(3):
        half_dt = t_ds[2 * i + 1] - t_ds[2 * i]
        k1 = dynamics_rhs_at_node(dynamics_data, 2 * i, q, y)
        k2 = dynamics_rhs_at_node(dynamics_data, 2 * i + 1, q,
                                  y + half_dt * k1)
        k3 = dynamics_rhs_at_node(dynamics_data, 2 * i + 1, q,
                                  y + half_dt * k2)
        k4 = dynamics_rhs_at_node(dynamics_data, 2 * i + 2, q,
                                  y + 2 * half_dt * k3)
        y = normalize_y(y + (half_dt / 3.) * (k1 + 2 * k2 + 2 * k3 + k4),
                        norm_chiA, norm_chiB)
        y_of_t = y_of_t.at[i + 1].set(y)
        k_init.append(k1)
        dt_init.append(2 * half_dt)

    return _masked_forward_ab4(dynamics_data, q, y_of_t, norm_chiA,
                               norm_chiB, 3, k_init, jnp.stack(dt_init))


def integrate_dynamics_at_reference(dynamics_data, q, chiA0, chiB0,
                                    omega_ref, init_quat=None,
                                    init_orbphase=0.0):
    """Integrate the dynamics with a reference frequency omega_ref.

    Port of the t_ref != None path of ``DynamicsSurrogate.__call__``
    (precessing_surrogate.py:440-499) plus ``_initialize`` (:508): map
    omega_ref to t_ref, Euler-step the initial state to the nearest output
    node i0, then integrate forward and backward from i0 via the three
    reference branches expressed as lax.switch. Fully traceable and
    vmap-able (under vmap all three branches are evaluated).

    Returns y_of_t with shape (n_dyn, 11).
    """
    chiA0 = jax_quaternions.rotate_spin(chiA0, -1 * init_orbphase)
    chiB0 = jax_quaternions.rotate_spin(chiB0, -1 * init_orbphase)
    norm_chiA = jnp.sqrt(jnp.sum(chiA0**2))
    norm_chiB = jnp.sqrt(jnp.sum(chiB0**2))

    if init_quat is None:
        init_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
    y0 = jnp.concatenate([
        init_quat, jnp.asarray(init_orbphase, dtype=jnp.float64)[None],
        chiA0, chiB0])

    t_ref = reference_time_from_omega(dynamics_data, omega_ref, q, chiA0,
                                      chiB0, init_orbphase, init_quat)

    # _initialize (:508): Euler step to the nearest output node.
    times = jnp.asarray(dynamics_data.t_dynamics)
    i0 = jnp.argmin(jnp.abs(times - t_ref))
    t0 = times[i0]
    dydt0 = dynamics_rhs_at_time(dynamics_data, t_ref, q, y0)
    y_node = normalize_y(y0 + (t0 - t_ref) * dydt0, norm_chiA, norm_chiB)

    num_output = times.shape[0]
    dt_dynamics = jnp.asarray(dynamics_data.dt_dynamics)

    def branch_start(operands):
        (y_node, i0) = operands
        return _bootstrap_rk4_and_forward(dynamics_data, q, y_node,
                                          norm_chiA, norm_chiB)

    def branch_low(operands):
        # 0 < i0 <= 2 (__call__ :483): 3 forward RK4 steps, forward AB4
        # from i0+3, then backward AB4 from i0.
        (y_node, i0) = operands
        y_of_t = jnp.zeros((num_output, 11)).at[i0].set(y_node)
        seeds = []
        y = y_node
        for step_index in range(3):
            output_index = i0 + step_index
            y, k1 = _forward_rk4_step_at_index(
                dynamics_data, q, y, output_index, norm_chiA, norm_chiB)
            y_of_t = y_of_t.at[output_index + 1].set(y)
            seeds.append(k1)
        dt_forward = jax.lax.dynamic_slice(dt_dynamics, (i0,), (3,))
        y_of_t = _masked_forward_ab4(dynamics_data, q, y_of_t, norm_chiA,
                                     norm_chiB, i0 + 3, seeds, dt_forward)
        # Reference quirk (:495): the backward seed RHS uses t_ds node
        # index i0+3 for the state at output node i0+3.
        seed_back = dynamics_rhs_at_node(dynamics_data, i0 + 3, q,
                                         y_of_t[i0 + 3])
        k_backward = [seed_back, seeds[2], seeds[1]]
        dt_backward = dt_forward[::-1]
        return _masked_backward_ab4(dynamics_data, q, y_of_t, norm_chiA,
                                    norm_chiB, i0, k_backward, dt_backward)

    def branch_high(operands):
        # i0 > 2 (__call__ :467): 3 backward RK4 steps, backward AB4 to 0,
        # then forward AB4 from i0.
        (y_node, i0) = operands
        y_of_t = jnp.zeros((num_output, 11)).at[i0].set(y_node)
        seeds = []
        y = y_node
        for step_index in range(3):
            output_index = i0 - step_index
            y, k1 = _backward_rk4_step_at_index(
                dynamics_data, q, y, output_index, norm_chiA, norm_chiB)
            y_of_t = y_of_t.at[output_index - 1].set(y)
            seeds.append(k1)
        dt_backward = jax.lax.dynamic_slice(dt_dynamics, (i0 - 3,),
                                            (3,))[::-1]
        y_of_t = _masked_backward_ab4(dynamics_data, q, y_of_t, norm_chiA,
                                      norm_chiB, i0 - 3, seeds, dt_backward)
        # Reference quirk (:479): the forward seed RHS uses t_ds node
        # index i0 for the state at output node i0-3.
        seed_forward = dynamics_rhs_at_node(dynamics_data, i0, q,
                                            y_of_t[i0 - 3])
        k_forward = [seed_forward, seeds[2], seeds[1]]
        dt_forward = dt_backward[::-1]
        return _masked_forward_ab4(dynamics_data, q, y_of_t, norm_chiA,
                                   norm_chiB, i0, k_forward, dt_forward)

    branch_index = jnp.where(i0 == 0, 0, jnp.where(i0 > 2, 2, 1))
    return jax.lax.switch(branch_index, [branch_start, branch_low,
                                         branch_high], (y_node, i0))
