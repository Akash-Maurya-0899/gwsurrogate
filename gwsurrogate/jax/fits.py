"""Parametric fit evaluation for the NRSur7dq4 JAX port.

Ports the polynomial fit evaluation from
``gwsurrogate/precessing_utils/src/precessing_utils.c`` (``compute_x_powers``
and ``eval_one_fit``) and the fit-coordinate map ``get_fit_params`` defined in
``NRSur7dq4._load_dimless_surrogate`` (gwsurrogate/surrogate.py).

A fit value is

    val = sum_i coefs[i] * prod_j basis_j(x_j)^bfOrders[i, j]

over 7 fit parameters, where the basis for parameter 0 (mass ratio) is the
monomial in the affinely rescaled coordinate
``q_fit_offset + q_fit_slope * x_0`` and the basis for parameters 1..6 (spin
components) are plain monomials.

Ragged fit tables are zero-padded to a common length: a padded entry has
``coefs[i] == 0`` so its term contributes exactly 0 regardless of the (also
zero-padded) basis-function orders.
"""

import jax.numpy as jnp

# Model constants for NRSur7dq4, from get_fit_settings() in
# NRSur7dq4._load_dimless_surrogate (gwsurrogate/surrogate.py).
# The offset/slope rescale log(q) from [-0.01, log(4.01)] to [-1, 1].
NRSUR7DQ4_Q_FIT_OFFSET = -0.9857019407834238
NRSUR7DQ4_Q_FIT_SLOPE = 1.4298059216576398
NRSUR7DQ4_Q_MAX_BF_ORDER = 3  # max monomial order for the q parameter
NRSUR7DQ4_CHI_MAX_BF_ORDER = 2  # max monomial order for each chi parameter

NUM_FIT_PARAMS = 7


def get_fit_params(x):
    """Map surrogate parameters to fit coordinates.

    Port of ``get_fit_params`` in ``NRSur7dq4._load_dimless_surrogate``
    (gwsurrogate/surrogate.py). Converts
    ``x = [q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z]`` to
    ``[log(q), chi1x, chi1y, chiHat, chi2x, chi2y, chi_a]`` where chiHat is
    defined in Eq.(3) of arXiv:1508.07253 and ``chi_a = (chi1z - chi2z)/2``.

    Supports arbitrary leading batch dimensions: x has shape (..., 7).
    """
    q = x[..., 0]
    chi1z = x[..., 3]
    chi2z = x[..., 6]
    eta = q / (1. + q)**2
    chi_weighted_average = (q * chi1z + chi2z) / (1 + q)
    chi_hat = (chi_weighted_average - 38. * eta / 113. * (chi1z + chi2z)) \
        / (1. - 76. * eta / 113.)
    chi_a = (chi1z - chi2z) / 2.

    return jnp.stack(
        [jnp.log(q), x[..., 1], x[..., 2], chi_hat,
         x[..., 4], x[..., 5], chi_a],
        axis=-1)


def compute_basis_power_table(fit_params, q_fit_offset=NRSUR7DQ4_Q_FIT_OFFSET,
                              q_fit_slope=NRSUR7DQ4_Q_FIT_SLOPE,
                              q_max_bf_order=NRSUR7DQ4_Q_MAX_BF_ORDER,
                              chi_max_bf_order=NRSUR7DQ4_CHI_MAX_BF_ORDER):
    """Build the (7, max_order+1) table of basis-function powers.

    Port of ``compute_x_powers`` (precessing_utils.c:159). Row j holds the
    powers 0..max_order of the j-th basis coordinate; powers are built by
    incremental multiplication exactly as in the C code. Rows for chi
    parameters are padded past ``chi_max_bf_order`` (those columns are never
    indexed because chi orders never exceed ``chi_max_bf_order``).

    fit_params has shape (7,); the returned table has shape
    (7, q_max_bf_order + 1).
    """
    q_rescaled = q_fit_offset + q_fit_slope * fit_params[0]
    bases = jnp.concatenate([q_rescaled[None], fit_params[1:]])  # (7,)

    max_order = max(q_max_bf_order, chi_max_bf_order)
    columns = [jnp.ones_like(bases)]
    for _ in range(max_order):
        columns.append(columns[-1] * bases)  # incremental multiply, as in C
    return jnp.stack(columns, axis=-1)  # (7, max_order+1)


def evaluate_fits(padded_coefs, padded_bf_orders, fit_params, **fit_settings):
    """Evaluate zero-padded polynomial fits at a single fit-parameter point.

    Port of ``eval_one_fit`` (precessing_utils.c:187), vectorized over any
    number of stacked fits.

    Arguments:
        padded_coefs: float array of shape (..., K) — fit coefficients,
            zero-padded along the last axis.
        padded_bf_orders: int array of shape (..., K, 7) — basis-function
            orders per coefficient and fit parameter, zero-padded.
        fit_params: float array of shape (7,) — the (already transformed)
            fit coordinates.
        fit_settings: optional overrides forwarded to
            ``compute_basis_power_table``.

    Returns an array of shape (...,) with one value per stacked fit.
    """
    power_table = compute_basis_power_table(fit_params, **fit_settings)

    # Gather the power of each fit parameter for every coefficient:
    # gathered[..., i, j] = power_table[j, padded_bf_orders[..., i, j]]
    gathered_powers = power_table[jnp.arange(NUM_FIT_PARAMS),
                                  padded_bf_orders]  # (..., K, 7)

    # Multiply in the same fixed order as the C loop (j = 0..6).
    basis_product = gathered_powers[..., 0]
    for j in range(1, NUM_FIT_PARAMS):
        basis_product = basis_product * gathered_powers[..., j]

    return jnp.sum(padded_coefs * basis_product, axis=-1)
