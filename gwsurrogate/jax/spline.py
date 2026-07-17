"""Natural cubic spline interpolation for the NRSur7dq4 JAX port.

Port of gwsurrogate/spline_interp_Cwrapper/_spline_interp.cpp: a natural
cubic spline (zero second derivative at both ends — NOT scipy's default
not-a-knot), solved with the Thomas algorithm and evaluated in Horner form.

Two paths are provided:

1. A NumPy, load-time path that builds a dense interpolation matrix between
   two fixed grids. Spline interpolation is linear in the data, so
   interpolating from the dynamics grid (230 nodes) onto the coorbital grid
   (2000 samples) is a precomputable (2000, 230) matrix applied with a
   single matmul — exact (up to summation order) and trivially
   vmap/GPU-friendly.

2. A jnp path (``interpolate_natural_spline``) for resampling from a fixed
   source grid onto an arbitrary (traced) output grid, used for user time
   arrays. The tridiagonal sweeps are ``lax.scan``s; interval lookup uses
   ``searchsorted``. Out-of-range output points are clamped to the source
   interval ends (callers validate ranges host-side; the C code errors
   instead).
"""

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp


def _thomas_factors(data_x):
    """Interval widths, reciprocals, and factored interior diagonal.

    Port of ``spline_prepare`` (_spline_interp.cpp:214). data_x is a 1D
    NumPy array of strictly monotonic knots. Returns (h, inv_h, diag) with
    shapes (n-1,), (n-1,), (n-2,).
    """
    data_x = np.asarray(data_x, dtype=np.float64)
    h = np.diff(data_x)
    if (h == 0.0).any():
        raise ValueError("Zero-length interval in spline knots.")
    inv_h = 1.0 / h

    num_interior = len(data_x) - 2
    diag = np.empty(num_interior)
    diag[0] = 2.0 * (h[0] + h[1])
    for k in range(1, num_interior):
        factor = h[k] / diag[k - 1]
        diag[k] = 2.0 * (h[k] + h[k + 1]) - factor * h[k]
    return h, inv_h, diag


def _append_axes(array, num_axes):
    """Reshape a 1D array to broadcast over ``num_axes`` trailing axes."""
    return np.reshape(array, array.shape + (1,) * num_axes)


def _second_derivatives_numpy(data_x, data_y):
    """Natural-spline second derivatives c with c[0] = c[n-1] = 0 (NumPy).

    Port of Phase 1 of ``spline_interp_multi_tmpl`` (_spline_interp.cpp:359).
    data_y may have shape (n,) or (n, D) — the Thomas sweeps vectorize over
    trailing axes.
    """
    h, inv_h, diag = _thomas_factors(data_x)
    y = np.asarray(data_y, dtype=np.float64)
    n = len(data_x)
    num_interior = n - 2
    trailing = y.ndim - 1

    raw = 6.0 * (y[2:] * _append_axes(inv_h[1:], trailing)
                 - y[1:-1] * _append_axes(inv_h[1:] + inv_h[:-1], trailing)
                 + y[:-2] * _append_axes(inv_h[:-1], trailing))

    rhs = np.empty_like(raw)
    rhs[0] = raw[0]
    for k in range(1, num_interior):
        factor = h[k] / diag[k - 1]
        rhs[k] = raw[k] - factor * rhs[k - 1]

    c = np.zeros((n,) + y.shape[1:])
    c[n - 2] = rhs[num_interior - 1] / diag[num_interior - 1]
    for k in range(num_interior - 2, -1, -1):
        c[k + 1] = (rhs[k] - h[k + 1] * c[k + 2]) / diag[k]
    return c


def _evaluate_numpy(data_x, data_y, second_derivs, out_x):
    """Horner-form piecewise cubic evaluation (NumPy).

    Port of Phase 2b of ``spline_interp_multi_tmpl``
    (_spline_interp.cpp:414). data_y/second_derivs may have trailing axes.
    """
    data_x = np.asarray(data_x)
    h, inv_h, _ = _thomas_factors(data_x)
    idx = np.clip(np.searchsorted(data_x, out_x, side="right") - 1,
                  0, len(data_x) - 2)
    trailing = data_y.ndim - 1

    t = _append_axes(out_x - data_x[idx], trailing)
    inv_hi = _append_axes(inv_h[idx], trailing)
    hi = _append_axes(h[idx], trailing)

    ci = second_derivs[idx]
    ci1 = second_derivs[idx + 1]
    d_coeff = (ci1 - ci) * inv_hi * (1.0 / 6.0)
    b = -hi * (1.0 / 6.0) * (2.0 * ci + ci1) \
        + (data_y[idx + 1] - data_y[idx]) * inv_hi
    return data_y[idx] + t * (b + t * (0.5 * ci + t * d_coeff))


def interpolate_natural_spline_numpy(out_x, data_x, data_y):
    """NumPy natural-spline interpolation (reference-equivalent)."""
    second_derivs = _second_derivatives_numpy(data_x, data_y)
    return _evaluate_numpy(np.asarray(data_x), np.asarray(data_y,
                                                          dtype=np.float64),
                           second_derivs, np.asarray(out_x))


def build_natural_spline_interpolation_matrix(data_x, out_x):
    """Dense operator M with (M @ y) == natural-spline interp of y.

    Exploits linearity: columns are the spline interpolants of the cardinal
    basis vectors. Built once at load time with NumPy; both grids fixed.
    Returns shape (len(out_x), len(data_x)).
    """
    identity = np.eye(len(data_x))
    return _evaluate_numpy(
        np.asarray(data_x), identity,
        _second_derivatives_numpy(data_x, identity), np.asarray(out_x))


class SplineGridData(NamedTuple):
    """Precomputed knot geometry of a fixed source grid (a JAX pytree).

    Per-call jnp work is then only the two Thomas sweeps (in the data) and
    the pointwise evaluation.
    """
    knots: np.ndarray             # (n,)
    h: np.ndarray                 # (n-1,) interval widths
    inv_h: np.ndarray             # (n-1,) reciprocals
    diag: np.ndarray              # (n-2,) factored interior diagonal
    forward_factors: np.ndarray   # (n-3,) h[k]/diag[k-1] for the sweep


def make_spline_grid_data(data_x):
    """Build SplineGridData for a fixed knot grid (NumPy, load time)."""
    data_x = np.asarray(data_x, dtype=np.float64)
    h, inv_h, diag = _thomas_factors(data_x)
    return SplineGridData(
        knots=data_x, h=h, inv_h=inv_h, diag=diag,
        forward_factors=h[1:len(diag)] / diag[:-1])


def spline_second_derivatives(grid, data_y):
    """Natural-spline second derivatives of data_y (jnp; scans).

    data_y has shape (n,) or (D, n) (datasets leading, matching the
    multi-dataset C API); the scans carry all datasets at once.
    """
    y = jnp.asarray(data_y)
    squeeze = (y.ndim == 1)
    if squeeze:
        y = y[None, :]

    inv_h = jnp.asarray(grid.inv_h)
    h = jnp.asarray(grid.h)
    diag = jnp.asarray(grid.diag)

    raw = 6.0 * (y[:, 2:] * inv_h[1:]
                 - y[:, 1:-1] * (inv_h[1:] + inv_h[:-1])
                 + y[:, :-2] * inv_h[:-1])  # (D, n-2)

    def forward_sweep(rhs_prev, inputs):
        raw_k, factor_k = inputs
        rhs_k = raw_k - factor_k * rhs_prev
        return rhs_k, rhs_k

    _, rhs_rest = jax.lax.scan(
        forward_sweep, raw[:, 0],
        (raw[:, 1:].T, jnp.asarray(grid.forward_factors)))
    rhs = jnp.concatenate([raw[:, :1], rhs_rest.T], axis=1)  # (D, n-2)

    def backward_sweep(c_next, inputs):
        rhs_k, h_k1, diag_k = inputs
        c_k = (rhs_k - h_k1 * c_next) / diag_k
        return c_k, c_k

    c_last_interior = rhs[:, -1] / diag[-1]
    num_interior = rhs.shape[1]
    _, c_rest = jax.lax.scan(
        backward_sweep, c_last_interior,
        (rhs[:, :-1].T[::-1], h[1:num_interior][::-1],
         diag[:-1][::-1]))
    c_interior = jnp.concatenate(
        [c_rest[::-1].T, c_last_interior[:, None]], axis=1)

    zeros = jnp.zeros_like(c_interior[:, :1])
    c = jnp.concatenate([zeros, c_interior, zeros], axis=1)  # (D, n)
    return c[0] if squeeze else c


def spline_evaluate(grid, data_y, second_derivs, out_x):
    """Piecewise-cubic Horner evaluation at out_x (jnp).

    data_y/second_derivs have shape (n,) or (D, n); out_x is 1D (may be
    traced). Out-of-range points are clamped to the end intervals.
    """
    y = jnp.asarray(data_y)
    c = jnp.asarray(second_derivs)
    squeeze = (y.ndim == 1)
    if squeeze:
        y, c = y[None, :], c[None, :]

    knots = jnp.asarray(grid.knots)
    idx = jnp.clip(jnp.searchsorted(knots, out_x, side="right") - 1,
                   0, knots.shape[0] - 2)
    t = out_x - knots[idx]
    inv_hi = jnp.asarray(grid.inv_h)[idx]
    hi = jnp.asarray(grid.h)[idx]

    ci = c[:, idx]
    ci1 = c[:, idx + 1]
    d_coeff = (ci1 - ci) * inv_hi * (1.0 / 6.0)
    b = -hi * (1.0 / 6.0) * (2.0 * ci + ci1) \
        + (y[:, idx + 1] - y[:, idx]) * inv_hi
    result = y[:, idx] + t * (b + t * (0.5 * ci + t * d_coeff))
    return result[0] if squeeze else result


def spline_interpolate(grid, data_y, out_x):
    """Full natural-spline resample of data_y onto out_x (jnp)."""
    return spline_evaluate(grid, data_y,
                           spline_second_derivatives(grid, data_y), out_x)
