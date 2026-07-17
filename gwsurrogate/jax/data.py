"""Loading and preprocessing of NRSur7dq4 surrogate data for the JAX port.

Reads the gwsurrogate-format ``NRSur7dq4.h5`` file (the same file used by the
reference implementation) and converts the ragged per-node fit tables into
rectangular, zero-padded NumPy arrays stacked into ``NamedTuple`` containers
(which JAX treats as pytrees). This module uses only NumPy/h5py — JAX arrays
enter via ``jax.device_put`` on the finished containers.

Zero-padding is exact: a padded coefficient is 0, so its fit term contributes
exactly 0 regardless of its (zero-padded) basis-function orders, and padded
empirical-interpolation nodes have all-zero EI-basis rows, so they contribute
exactly 0 to the reconstructed time series.

Reference data layout (see ``DynamicsSurrogate.__init__`` and
``CoorbitalWaveformSurrogate.__init__`` in
gwsurrogate/new/precessing_surrogate.py):

- ``t_ds``: dynamics time nodes; per node a group ``ds_node_<i>`` with 9
  scalar fits, ordered here as in the reference ``fit_data_batch``:
  [omega_orb_0, omega_orb_1, omega, chiA_0, chiA_1, chiA_2,
   chiB_0, chiB_1, chiB_2].
- ``t_coorb``: waveform time grid; per waveform component a group with
  ``EIBasis``, ``nodeIndices`` and per-EI-node fits under ``nodeModelers``.
"""

import os
from typing import NamedTuple

import h5py
import numpy as np

# Order of the 9 dynamics fits per time node, matching the reference
# fit_data_batch list (precessing_surrogate.py, DynamicsSurrogate.__init__).
DYNAMICS_FIT_NAMES = (
    "omega_orb_0", "omega_orb_1", "omega",
    "chiA_0", "chiA_1", "chiA_2",
    "chiB_0", "chiB_1", "chiB_2",
)


def coorbital_component_names(ell_max=4):
    """Canonical ordering of the coorbital waveform components.

    Mirrors the loading order in ``CoorbitalWaveformSurrogate.__init__``:
    for each ell, the m=0 real/imag components, then for each m=1..ell the
    Re+/Re-/Im+/Im- components. For NRSur7dq4 (ell_max=4) this gives 42
    components.
    """
    names = []
    for ell in range(2, ell_max + 1):
        names.append("hCoorb_%s_0_real" % ell)
        names.append("hCoorb_%s_0_imag" % ell)
        for m in range(1, ell + 1):
            for reim in ("Re", "Im"):
                for pm in ("+", "-"):
                    names.append("hCoorb_%s_%s_%s%s" % (ell, m, reim, pm))
    return names


class DynamicsData(NamedTuple):
    """Stacked, zero-padded dynamics-surrogate data (a JAX pytree)."""
    t_ds: np.ndarray            # (n_ds,) dynamics time nodes (233)
    t_dynamics: np.ndarray      # (n_dyn,) output grid of the integrator (230)
    dt_dynamics: np.ndarray     # (n_dyn-1,) diffs of t_dynamics
    fit_coefs: np.ndarray       # (n_ds, 9, K) zero-padded coefficients
    fit_bf_orders: np.ndarray   # (n_ds, 9, K, 7) int32 basis-function orders


class CoorbitalData(NamedTuple):
    """Stacked, zero-padded coorbital-waveform data (a JAX pytree)."""
    t_coorb: np.ndarray            # (n_coorb,) waveform time grid (2000)
    node_fit_coefs: np.ndarray     # (n_comp, N, K) zero-padded coefficients
    node_fit_bf_orders: np.ndarray  # (n_comp, N, K, 7) int32 orders
    node_indices: np.ndarray       # (n_comp, N) int32, zero-padded
    ei_basis: np.ndarray           # (n_comp, N, n_coorb), zero-padded rows


class NRSur7dq4Data(NamedTuple):
    """All NRSur7dq4 model data needed by the JAX evaluation pipeline."""
    dynamics: DynamicsData
    coorb: CoorbitalData


def default_nrsur7dq4_h5_path():
    """Path where gwsurrogate stores the downloaded NRSur7dq4.h5."""
    import gwsurrogate
    return os.path.join(gwsurrogate.__path__[0], "surrogate_downloads",
                        "NRSur7dq4.h5")


def _pad_and_stack_fits(fit_tables):
    """Stack a ragged list of (coefs, bf_orders) fits into padded arrays.

    fit_tables is a list of tuples (coefs (n_i,), bf_orders (n_i, 7)).
    Returns (padded_coefs (n_fits, K), padded_bf_orders (n_fits, K, 7))
    with K = max n_i, zero-padded.
    """
    num_fits = len(fit_tables)
    max_len = max(coefs.shape[0] for coefs, _ in fit_tables)
    padded_coefs = np.zeros((num_fits, max_len), dtype=np.float64)
    padded_bf_orders = np.zeros((num_fits, max_len, 7), dtype=np.int32)
    for i, (coefs, bf_orders) in enumerate(fit_tables):
        n = coefs.shape[0]
        assert bf_orders.shape == (n, 7), \
            "bfOrders shape %s inconsistent with %d coefs" % (bf_orders.shape, n)
        padded_coefs[i, :n] = coefs
        padded_bf_orders[i, :n, :] = bf_orders
    return padded_coefs, padded_bf_orders


def _load_dynamics_data(h5file):
    """Load and stack the dynamics section (t_ds and per-node fit tables)."""
    t_ds = h5file["t_ds"][()]
    num_nodes = len(t_ds)

    per_node_fits = []
    for i in range(num_nodes):
        group = h5file["ds_node_%s" % i]
        per_node_fits.append([
            (group["%s_coefs" % name][()], group["%s_bfOrders" % name][()])
            for name in DYNAMICS_FIT_NAMES
        ])

    max_len = max(coefs.shape[0]
                  for node_fits in per_node_fits
                  for coefs, _ in node_fits)
    fit_coefs = np.zeros((num_nodes, len(DYNAMICS_FIT_NAMES), max_len),
                         dtype=np.float64)
    fit_bf_orders = np.zeros(
        (num_nodes, len(DYNAMICS_FIT_NAMES), max_len, 7), dtype=np.int32)
    for i, node_fits in enumerate(per_node_fits):
        coefs_i, orders_i = _pad_and_stack_fits(node_fits)
        fit_coefs[i, :, :coefs_i.shape[1]] = coefs_i
        fit_bf_orders[i, :, :orders_i.shape[1], :] = orders_i

    # The first 3 pairs of t_ds steps are half-steps used by the RK4
    # bootstrap; the integrator output grid skips the half-step nodes
    # (see DynamicsSurrogate._initialize, precessing_surrogate.py).
    diff_t = np.diff(t_ds)
    for i in range(3):
        if diff_t[2 * i] != diff_t[2 * i + 1]:
            raise ValueError("t_ds does not start with paired half-steps; "
                             "AB4 bootstrap assumptions are violated.")
    t_dynamics = np.append(t_ds[:6:2], t_ds[6:])

    return DynamicsData(
        t_ds=t_ds,
        t_dynamics=t_dynamics,
        dt_dynamics=np.diff(t_dynamics),
        fit_coefs=fit_coefs,
        fit_bf_orders=fit_bf_orders,
    )


def _load_coorbital_data(h5file, ell_max=4):
    """Load and stack the coorbital section (per-component EI + fit data)."""
    t_coorb = h5file["t_coorb"][()]
    component_names = coorbital_component_names(ell_max)
    for name in component_names:
        if name not in h5file:
            raise ValueError(
                "Component %s missing from h5 file; the JAX port assumes "
                "all NRSur7dq4 components are present." % name)

    num_components = len(component_names)
    components = []
    for name in component_names:
        group = h5file[name]
        ei_basis = group["EIBasis"][()]
        node_indices = group["nodeIndices"][()]
        node_fits = [
            (group["nodeModelers"]["coefs_%s" % i][()],
             group["nodeModelers"]["bfOrders_%s" % i][()])
            for i in range(len(node_indices))
        ]
        components.append((ei_basis, node_indices, node_fits))

    max_nodes = max(len(node_indices) for _, node_indices, _ in components)
    max_len = max(coefs.shape[0]
                  for _, _, node_fits in components
                  for coefs, _ in node_fits)

    node_fit_coefs = np.zeros((num_components, max_nodes, max_len),
                              dtype=np.float64)
    node_fit_bf_orders = np.zeros((num_components, max_nodes, max_len, 7),
                                  dtype=np.int32)
    node_indices_padded = np.zeros((num_components, max_nodes),
                                   dtype=np.int32)
    ei_basis_padded = np.zeros((num_components, max_nodes, len(t_coorb)),
                               dtype=np.float64)
    for c, (ei_basis, node_indices, node_fits) in enumerate(components):
        num_nodes = len(node_indices)
        assert ei_basis.shape == (num_nodes, len(t_coorb))
        coefs_c, orders_c = _pad_and_stack_fits(node_fits)
        node_fit_coefs[c, :num_nodes, :coefs_c.shape[1]] = coefs_c
        node_fit_bf_orders[c, :num_nodes, :orders_c.shape[1], :] = orders_c
        node_indices_padded[c, :num_nodes] = node_indices
        ei_basis_padded[c, :num_nodes, :] = ei_basis

    return CoorbitalData(
        t_coorb=t_coorb,
        node_fit_coefs=node_fit_coefs,
        node_fit_bf_orders=node_fit_bf_orders,
        node_indices=node_indices_padded,
        ei_basis=ei_basis_padded,
    )


def load_nrsur7dq4_jax_data(h5_path=None, ell_max=4, to_device=True):
    """Load NRSur7dq4.h5 into padded, stacked arrays ready for JAX.

    Arguments:
        h5_path: path to the gwsurrogate-format NRSur7dq4.h5 file; defaults
            to the standard gwsurrogate download location.
        ell_max: largest ell mode in the model (4 for NRSur7dq4).
        to_device: if True, move all arrays onto the default JAX device.

    Returns an ``NRSur7dq4Data`` pytree.
    """
    if h5_path is None:
        h5_path = default_nrsur7dq4_h5_path()
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(
            "NRSur7dq4.h5 not found at %s. Download it via "
            "gwsurrogate.catalog.pull('NRSur7dq4') or pass h5_path." % h5_path)

    with h5py.File(h5_path, "r") as h5file:
        data = NRSur7dq4Data(
            dynamics=_load_dynamics_data(h5file),
            coorb=_load_coorbital_data(h5file, ell_max),
        )

    _validate_padding_invariants(data)

    if to_device:
        import jax
        data = jax.device_put(data)
    return data


def _validate_padding_invariants(data):
    """Sanity checks on the padded tables (exactness of zero-padding)."""
    max_orders = np.array([3, 2, 2, 2, 2, 2, 2], dtype=np.int32)
    for bf_orders in (data.dynamics.fit_bf_orders,
                      data.coorb.node_fit_bf_orders):
        if bf_orders.min() < 0 or (bf_orders > max_orders).any():
            raise ValueError("Basis-function orders outside the expected "
                             "ranges (q <= 3, chi <= 2).")
    if not np.isfinite(data.dynamics.fit_coefs).all():
        raise ValueError("Non-finite dynamics fit coefficients.")
    if not np.isfinite(data.coorb.node_fit_coefs).all():
        raise ValueError("Non-finite coorbital fit coefficients.")
