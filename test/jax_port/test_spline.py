"""M5 tests: natural cubic spline vs the C++ _spline_interp oracle.

NOTE on the C++ oracle: its ``hunt`` interval search has an off-by-one bug
for evaluation points in the LAST knot interval when the previous point was
two or more intervals behind (the geometric expansion overshoots the array
end and the clamped bracket breaks the binary-search invariant), causing
those points to be evaluated from the second-to-last interval. This never
triggers for output grids denser than the knot grid (the surrogate's use
case), but random sparse out_x arrays can hit it. C-comparison tests here
therefore keep out_x below the last knot interval; the last interval is
covered by the independent scipy natural-spline oracle instead.
"""

import numpy as np
import jax.numpy as jnp
from scipy.interpolate import CubicSpline

from gwsurrogate.jax import spline as jax_spline
from gwsurrogate.new.surrogate import _splinterp_Cwrapper, \
    _splinterp_Cwrapper_many, _splinterp_Cwrapper_many_complex

RTOL = 1e-13
ATOL = 1e-14
# The C++ real-valued path uses hardware fused-multiply-add (std::fma),
# which rounds differently from separate mul+add; the difference propagates
# through the sequential Thomas sweeps. ~1e-13 absolute on O(1) data is the
# observed scale (the complex C++ path has no FMA and agrees to 1e-14).
ATOL_REAL_FMA = 1e-12


def _random_grid_and_data(num_knots, num_out, seed, complex_data=False):
    rng = np.random.default_rng(seed)
    data_x = np.sort(rng.uniform(-100.0, 100.0, num_knots))
    data_y = rng.standard_normal(num_knots)
    if complex_data:
        data_y = data_y + 1j * rng.standard_normal(num_knots)
    # Stay below the last knot interval: see the C hunt bug note above.
    out_x = np.sort(rng.uniform(data_x[0], data_x[-2], num_out))
    return data_x, data_y, out_x


def test_numpy_path_matches_c():
    for seed in range(5):
        data_x, data_y, out_x = _random_grid_and_data(37, 200, seed)
        reference = _splinterp_Cwrapper(out_x, data_x, data_y)
        ours = jax_spline.interpolate_natural_spline_numpy(out_x, data_x,
                                                           data_y)
        np.testing.assert_allclose(ours, reference, rtol=RTOL,
                                   atol=ATOL_REAL_FMA)


def test_dense_interpolation_matrix_matches_c():
    """The precomputed cardinal-basis matrix equals direct interpolation."""
    data_x, _, out_x = _random_grid_and_data(230, 2000, seed=40)
    matrix = jax_spline.build_natural_spline_interpolation_matrix(data_x,
                                                                  out_x)
    rng = np.random.default_rng(41)
    for _ in range(5):
        data_y = rng.standard_normal(len(data_x))
        reference = _splinterp_Cwrapper(out_x, data_x, data_y)
        np.testing.assert_allclose(matrix @ data_y, reference, rtol=RTOL,
                                   atol=ATOL_REAL_FMA)


def test_jnp_fixed_grid_real_matches_c():
    data_x, _, out_x = _random_grid_and_data(101, 500, seed=42)
    fixed_grid = jax_spline.make_spline_grid_data(data_x)
    rng = np.random.default_rng(43)
    many_y = rng.standard_normal((7, len(data_x)))

    reference = _splinterp_Cwrapper_many(out_x, data_x, many_y)
    ours = np.asarray(jax_spline.spline_interpolate(fixed_grid, jnp.asarray(many_y),
                                             jnp.asarray(out_x)))
    np.testing.assert_allclose(ours, reference, rtol=RTOL,
                               atol=ATOL_REAL_FMA)


def test_jnp_fixed_grid_complex_matches_c():
    data_x, _, out_x = _random_grid_and_data(101, 500, seed=44)
    fixed_grid = jax_spline.make_spline_grid_data(data_x)
    rng = np.random.default_rng(45)
    many_y = rng.standard_normal((5, len(data_x))) \
        + 1j * rng.standard_normal((5, len(data_x)))

    reference = _splinterp_Cwrapper_many_complex(out_x, data_x, many_y)
    ours = np.asarray(jax_spline.spline_interpolate(fixed_grid, jnp.asarray(many_y),
                                             jnp.asarray(out_x)))
    np.testing.assert_allclose(ours, reference, rtol=RTOL, atol=ATOL)


def test_jnp_single_dataset_shape():
    data_x, data_y, out_x = _random_grid_and_data(50, 120, seed=46)
    fixed_grid = jax_spline.make_spline_grid_data(data_x)
    reference = _splinterp_Cwrapper(out_x, data_x, data_y)
    ours = np.asarray(jax_spline.spline_interpolate(fixed_grid, jnp.asarray(data_y),
                                             jnp.asarray(out_x)))
    assert ours.shape == (len(out_x),)
    np.testing.assert_allclose(ours, reference, rtol=RTOL, atol=ATOL)


def test_full_range_matches_scipy_natural_spline():
    """Independent oracle covering the full range incl. the last interval."""
    rng = np.random.default_rng(48)
    data_x = np.sort(rng.uniform(-100.0, 100.0, 101))
    data_y = rng.standard_normal(101)
    out_x = np.sort(rng.uniform(data_x[0], data_x[-1], 500))

    scipy_values = CubicSpline(data_x, data_y, bc_type="natural")(out_x)
    fixed_grid = jax_spline.make_spline_grid_data(data_x)
    ours = np.asarray(jax_spline.spline_interpolate(fixed_grid, jnp.asarray(data_y),
                                             jnp.asarray(out_x)))
    np.testing.assert_allclose(ours, scipy_values, rtol=1e-12, atol=1e-12)

    matrix = jax_spline.build_natural_spline_interpolation_matrix(data_x,
                                                                  out_x)
    np.testing.assert_allclose(matrix @ data_y, scipy_values, rtol=1e-12,
                               atol=1e-12)


def test_evaluation_at_knots_is_exact():
    """Interpolating onto the knots must reproduce the data (near-exactly)."""
    data_x, data_y, _ = _random_grid_and_data(80, 10, seed=47)
    fixed_grid = jax_spline.make_spline_grid_data(data_x)
    ours = np.asarray(jax_spline.spline_interpolate(fixed_grid, jnp.asarray(data_y),
                                             jnp.asarray(data_x)))
    np.testing.assert_allclose(ours, data_y, rtol=0.0, atol=1e-13)
