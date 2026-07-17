"""Shared fixtures for the JAX-port accuracy tests.

Every test in this directory compares the JAX implementation in
``gwsurrogate.jax`` against the reference NumPy/C implementation (the oracle).
All fixtures are session-scoped: the model data is loaded once.
"""

import os
import sys

import h5py
import numpy as np
import pytest

# Make the repo checkout importable when pytest is run from anywhere.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# If a CUDA device is visible, don't let XLA preallocate most of the VRAM —
# on this WSL2 setup the preallocation can fail and abort the test run.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Enables jax_enable_x64 before any jnp array is created (see CLAUDE.md).
import gwsurrogate.jax  # noqa: E402,F401
from gwsurrogate.jax import data as jax_data_module  # noqa: E402

H5_PATH = os.path.join(REPO_ROOT, "gwsurrogate", "surrogate_downloads",
                       "NRSur7dq4.h5")

requires_nrsur7dq4_h5 = pytest.mark.skipif(
    not os.path.isfile(H5_PATH),
    reason="NRSur7dq4.h5 not found in gwsurrogate/surrogate_downloads/")


@pytest.fixture(scope="session")
def h5_path():
    if not os.path.isfile(H5_PATH):
        pytest.skip("NRSur7dq4.h5 not found at %s" % H5_PATH)
    return H5_PATH


@pytest.fixture(scope="session")
def jax_data(h5_path):
    """The padded/stacked NRSur7dq4 model data on the JAX device."""
    return jax_data_module.load_nrsur7dq4_jax_data(h5_path)


@pytest.fixture(scope="session")
def raw_dynamics_fit_tables(h5_path):
    """Ragged per-node dynamics fit tables straight from the h5 file.

    Returns a list (over the t_ds nodes) of lists (over the 9 fits, in
    DYNAMICS_FIT_NAMES order) of (coefs, bfOrders) numpy arrays.
    """
    tables = []
    with h5py.File(h5_path, "r") as h5file:
        num_nodes = len(h5file["t_ds"])
        for i in range(num_nodes):
            group = h5file["ds_node_%s" % i]
            tables.append([
                (group["%s_coefs" % name][()],
                 group["%s_bfOrders" % name][()])
                for name in jax_data_module.DYNAMICS_FIT_NAMES
            ])
    return tables


@pytest.fixture(scope="session")
def raw_coorbital_fit_tables(h5_path):
    """Ragged per-component coorbital node fits straight from the h5 file.

    Returns a list (over the 42 components, in coorbital_component_names
    order) of lists (over that component's EI nodes) of (coefs, bfOrders).
    """
    tables = []
    with h5py.File(h5_path, "r") as h5file:
        for name in jax_data_module.coorbital_component_names():
            group = h5file[name]
            num_nodes = len(group["nodeIndices"])
            tables.append([
                (group["nodeModelers"]["coefs_%s" % i][()],
                 group["nodeModelers"]["bfOrders_%s" % i][()])
                for i in range(num_nodes)
            ])
    return tables


@pytest.fixture(scope="session")
def reference_surrogate(h5_path):
    """The oracle: the C-backed NRSur7dq4 loaded through the standard API."""
    import gwsurrogate
    return gwsurrogate.LoadSurrogate("NRSur7dq4")


@pytest.fixture(scope="session")
def random_fit_param_points():
    """50 random fit-coordinate points spanning the training range.

    Fit coordinates are [log q, chi1x, chi1y, chiHat, chi2x, chi2y, chi_a];
    all chi-like entries lie in [-0.8, 0.8], q in [1, 4].
    """
    rng = np.random.default_rng(seed=42)
    num_points = 50
    points = np.empty((num_points, 7))
    points[:, 0] = np.log(rng.uniform(1.0, 4.0, num_points))
    points[:, 1:] = rng.uniform(-0.8, 0.8, (num_points, 6))
    return points
