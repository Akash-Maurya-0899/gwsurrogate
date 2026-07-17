"""Coorbital-frame waveform modes for the NRSur7dq4 JAX port.

Port of ``CoorbitalWaveformSurrogate`` (gwsurrogate/new/precessing_surrogate.py:703):
for every waveform component, evaluate one parametric fit per empirical-
interpolation (EI) node at x = [q, chiA_coorb(t_node), chiB_coorb(t_node)],
then contract the node values with the EI basis to get the component time
series, and assemble the components into complex modes ordered
(2,-2)..(2,2), (3,-3)..(3,3), (4,-4)..(4,4).

The padded component tables (see gwsurrogate.jax.data) are stacked in the
canonical order of ``coorbital_component_names``: per ell, the m=0
real/imag components, then Re+/Re-/Im+/Im- for each m = 1..ell. Padded EI
nodes have zero coefficients and zero EI-basis rows, so they contribute
exactly nothing.
"""

import jax
import jax.numpy as jnp

from . import fits as jax_fits

# Number of components per ell block: 2 for m=0, then 4 per m = 1..ell.
_COMPONENTS_PER_ELL = {ell: 2 + 4 * ell for ell in (2, 3, 4)}

# vmap the shared fit kernel over (component, EI node) axes; each
# (component, node) pair has its own fit-parameter vector.
_evaluate_fits_per_node = jax.vmap(jax.vmap(jax_fits.evaluate_fits))


def coorbital_waveform_modes(coorb_data, q, chiA_coorb, chiB_coorb,
                             ell_max=4):
    """Evaluate the coorbital-frame waveform modes.

    Port of ``CoorbitalWaveformSurrogate.__call__``
    (precessing_surrogate.py:754) and ``_eval_comp`` (:791).

    Arguments:
        coorb_data: CoorbitalData pytree (padded fit tables and EI bases).
        q: mass ratio (scalar, may be traced).
        chiA_coorb, chiB_coorb: coorbital-frame spins on the t_coorb grid,
            shape (n_coorb, 3).
        ell_max: largest ell to evaluate (2..4, static).

    Returns complex modes with shape (ell_max^2 + 2*ell_max - 3, n_coorb).
    """
    if ell_max < 2 or ell_max > 4:
        raise ValueError("coorbital_waveform_modes supports ell_max in 2..4.")
    num_components = sum(_COMPONENTS_PER_ELL[ell]
                         for ell in range(2, ell_max + 1))

    node_fit_coefs = coorb_data.node_fit_coefs[:num_components]
    node_fit_bf_orders = coorb_data.node_fit_bf_orders[:num_components]
    node_indices = coorb_data.node_indices[:num_components]
    ei_basis = coorb_data.ei_basis[:num_components]

    # Fit input per (component, EI node): x = [q, chiA(t_node), chiB(t_node)]
    # (_eval_comp :800). Padded nodes gather index 0 — harmless, their
    # coefficients and EI rows are zero.
    chiA_at_nodes = chiA_coorb[node_indices]  # (n_comp, N, 3)
    chiB_at_nodes = chiB_coorb[node_indices]
    q_at_nodes = jnp.broadcast_to(q, node_indices.shape)[..., None]
    fit_x = jnp.concatenate([q_at_nodes, chiA_at_nodes, chiB_at_nodes],
                            axis=-1)  # (n_comp, N, 7)
    fit_params = jax_fits.get_fit_params(fit_x)

    node_values = _evaluate_fits_per_node(
        node_fit_coefs, node_fit_bf_orders, fit_params)  # (n_comp, N)

    # nodes . EI_basis for every component (_eval_comp :804).
    component_series = jnp.einsum("cn,cnt->ct", node_values, ei_basis)

    # Assemble components into complex modes (__call__ :765-787 and
    # _assemble_mode_pair :689).
    num_modes = ell_max * ell_max + 2 * ell_max - 3
    mode_rows = [None] * num_modes
    component_index = 0
    for ell in range(2, ell_max + 1):
        center = ell * (ell + 1) - 4  # index of the (ell, 0) mode
        real_part = component_series[component_index]
        imag_part = component_series[component_index + 1]
        component_index += 2
        mode_rows[center] = real_part + 1j * imag_part

        for m in range(1, ell + 1):
            re_plus = component_series[component_index]
            re_minus = component_series[component_index + 1]
            im_plus = component_series[component_index + 2]
            im_minus = component_series[component_index + 3]
            component_index += 4
            # hplus = re_plus + i*im_plus, hminus = re_minus + i*im_minus;
            # h_{+m} = (hplus - hminus).conj(), h_{-m} = hplus + hminus.
            mode_rows[center + m] = (re_plus - re_minus) \
                + 1j * (im_minus - im_plus)
            mode_rows[center - m] = (re_plus + re_minus) \
                + 1j * (im_plus + im_minus)

    return jnp.stack(mode_rows)
