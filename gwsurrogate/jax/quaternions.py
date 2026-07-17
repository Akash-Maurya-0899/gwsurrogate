"""Quaternion algebra, frame transformations and Wigner-D matrices in JAX.

Ports of:
- the quaternion utilities ``multiplyQuats``/``quatInv`` and the frame
  transformation helpers ``rotateWaveform``/``transformTimeDependentVector``
  from gwsurrogate/new/precessing_surrogate.py, and
- the Wigner-D matrix computation ``wignerD_matrices`` from
  gwsurrogate/precessing_utils/src/precessing_utils.c, using the hardcoded
  closed-form d-matrix polynomials for ell = 2, 3, 4 (``hardcode_ell2/3/4``)
  and the four-position symmetry writes of the C implementation.

Everything is vectorized over the trailing time axis: quaternions have shape
(4, N) and waveform mode arrays have shape (n_modes, N).
"""

import math

import jax.numpy as jnp

# Same threshold as the C implementation (precessing_utils.c, wignerD_matrices):
# |ra|^2 or |rb|^2 below this is treated as exactly zero (edge case).
_EDGE_CASE_THRESHOLD = 1e-24


def multiply_quats(q1, q2):
    """Port of ``multiplyQuats`` (precessing_surrogate.py:22)."""
    return jnp.stack([
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[2] * q2[3] - q2[2] * q1[3] + q1[0] * q2[1] + q2[0] * q1[1],
        q1[3] * q2[1] - q2[3] * q1[1] + q1[0] * q2[2] + q2[0] * q1[2],
        q1[1] * q2[2] - q2[1] * q1[2] + q1[0] * q2[3] + q2[0] * q1[3]])


def quat_inv(q):
    """Port of ``quatInv`` (precessing_surrogate.py:29)."""
    norm_sqr = q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2
    return jnp.stack([q[0], -q[1], -q[2], -q[3]]) / norm_sqr


def transform_time_dependent_vector(quat, vec):
    """Port of ``transformTimeDependentVector`` (precessing_surrogate.py:112).

    quat has shape (4, N); vec has shape (3, N). Rotates vec from the
    coprecessing frame to the inertial frame.
    """
    q_inv = quat_inv(quat)
    vec_as_quat = jnp.concatenate([jnp.zeros_like(vec[:1]), vec], axis=0)
    return multiply_quats(quat, multiply_quats(vec_as_quat, q_inv))[1:]


def rotate_spin(chi, phase):
    """Rotate spin vectors about the z axis by ``phase``.

    Port of ``rotate_spin`` (precessing_surrogate.py:817), used for
    transforming spins between the coprecessing and coorbital frames.
    chi has shape (..., 3); phase is a scalar or matches chi's leading
    dimensions.
    """
    sin_phase = jnp.sin(phase)
    cos_phase = jnp.cos(phase)
    return jnp.stack([
        chi[..., 0] * cos_phase + chi[..., 1] * sin_phase,
        chi[..., 1] * cos_phase - chi[..., 0] * sin_phase,
        chi[..., 2]], axis=-1)


def _complex_unit_power_table(unit_phase, max_power):
    """Powers p = -max_power..max_power of a unit-modulus complex array.

    Mirrors ``build_cpows`` (precessing_utils.c:896): positive powers by
    incremental multiplication with the base, negative powers by incremental
    multiplication with the conjugate. Returns a dict p -> array.
    """
    powers = {0: jnp.ones_like(unit_phase)}
    conj_phase = jnp.conj(unit_phase)
    for p in range(1, max_power + 1):
        powers[p] = powers[p - 1] * unit_phase
        powers[-p] = powers[-(p - 1)] * conj_phase
    return powers


def _complex_reciprocal_power_table(base, max_power):
    """Like ``_complex_unit_power_table`` but for a general complex base,
    with negative powers via the exact reciprocal (as the C edge-case path
    does with ``cinv = 1.0 / base``)."""
    powers = {0: jnp.ones_like(base)}
    reciprocal = 1.0 / base
    for p in range(1, max_power + 1):
        powers[p] = powers[p - 1] * base
        powers[-p] = powers[-(p - 1)] * reciprocal
    return powers


def _wigner_d_fundamental_domain(c, s, ell):
    """Real Wigner-d values on the fundamental domain for one ell.

    Direct transcription of ``hardcode_ell2/3/4`` (precessing_utils.c:988,
    :1020, :1062) with c = |ra| and s = |rb|. Returns a dict
    (m, mp) -> array over the same fundamental domain the C code fills
    (mp = 0..ell; m = -mp..ell for mp > 0, m = 0..ell for mp = 0).
    """
    c2, s2 = c * c, s * s
    c3, s3 = c2 * c, s2 * s
    c4, s4 = c2 * c2, s2 * s2
    if ell == 2:
        sqrt6 = math.sqrt(6.0)
        return {
            (-2, 2): s4,
            (-1, 1): -s4 + 3.0 * c2 * s2,
            (-1, 2): -2.0 * c * s3,
            (0, 0): s4 - 4.0 * c2 * s2 + c4,
            (0, 1): sqrt6 * c * s3 - sqrt6 * c3 * s,
            (0, 2): sqrt6 * c2 * s2,
            (1, 0): -sqrt6 * c * s3 + sqrt6 * c3 * s,
            (1, 1): -3.0 * c2 * s2 + c4,
            (1, 2): -2.0 * c3 * s,
            (2, 0): sqrt6 * c2 * s2,
            (2, 1): 2.0 * c3 * s,
            (2, 2): c4,
        }

    c5, s5 = c4 * c, s4 * s
    c6, s6 = c3 * c3, s3 * s3
    if ell == 3:
        sqrt3 = math.sqrt(3.0)
        sqrt5 = math.sqrt(5.0)
        sqrt6 = math.sqrt(6.0)
        sqrt10 = math.sqrt(10.0)
        sqrt15 = math.sqrt(15.0)
        sqrt30 = math.sqrt(30.0)
        return {
            (-3, 3): s6,
            (-2, 2): -s6 + 5.0 * c2 * s4,
            (-2, 3): -sqrt6 * c * s5,
            (-1, 1): s6 - 8.0 * c2 * s4 + 6.0 * c4 * s2,
            (-1, 2): sqrt10 * c * s5 - 2.0 * sqrt10 * c3 * s3,
            (-1, 3): sqrt15 * c2 * s4,
            (0, 0): -s6 + 9.0 * c2 * s4 - 9.0 * c4 * s2 + c6,
            (0, 1): -2.0 * sqrt3 * c * s5 + 6.0 * sqrt3 * c3 * s3
                    - 2.0 * sqrt3 * c5 * s,
            (0, 2): -sqrt30 * c2 * s4 + sqrt30 * c4 * s2,
            (0, 3): -2.0 * sqrt5 * c3 * s3,
            (1, 0): 2.0 * sqrt3 * c * s5 - 6.0 * sqrt3 * c3 * s3
                    + 2.0 * sqrt3 * c5 * s,
            (1, 1): 6.0 * c2 * s4 - 8.0 * c4 * s2 + c6,
            (1, 2): 2.0 * sqrt10 * c3 * s3 - sqrt10 * c5 * s,
            (1, 3): sqrt15 * c4 * s2,
            (2, 0): -sqrt30 * c2 * s4 + sqrt30 * c4 * s2,
            (2, 1): -2.0 * sqrt10 * c3 * s3 + sqrt10 * c5 * s,
            (2, 2): -5.0 * c4 * s2 + c6,
            (2, 3): -sqrt6 * c5 * s,
            (3, 0): 2.0 * sqrt5 * c3 * s3,
            (3, 1): sqrt15 * c4 * s2,
            (3, 2): sqrt6 * c5 * s,
            (3, 3): c6,
        }

    c7, s7 = c6 * c, s6 * s
    c8, s8 = c4 * c4, s4 * s4
    if ell == 4:
        sqrt2 = math.sqrt(2.0)
        sqrt5 = math.sqrt(5.0)
        sqrt7 = math.sqrt(7.0)
        sqrt10 = math.sqrt(10.0)
        sqrt14 = math.sqrt(14.0)
        sqrt35 = math.sqrt(35.0)
        sqrt70 = math.sqrt(70.0)
        return {
            (-4, 4): s8,
            (-3, 3): -s8 + 7.0 * c2 * s6,
            (-3, 4): -2.0 * sqrt2 * c * s7,
            (-2, 2): s8 - 12.0 * c2 * s6 + 15.0 * c4 * s4,
            (-2, 3): sqrt14 * c * s7 - 3.0 * sqrt14 * c3 * s5,
            (-2, 4): 2.0 * sqrt7 * c2 * s6,
            (-1, 1): -s8 + 15.0 * c2 * s6 - 30.0 * c4 * s4 + 10.0 * c6 * s2,
            (-1, 2): -3.0 * sqrt2 * c * s7 + 15.0 * sqrt2 * c3 * s5
                     - 10.0 * sqrt2 * c5 * s3,
            (-1, 3): -3.0 * sqrt7 * c2 * s6 + 5.0 * sqrt7 * c4 * s4,
            (-1, 4): -2.0 * sqrt14 * c3 * s5,
            (0, 0): s8 - 16.0 * c2 * s6 + 36.0 * c4 * s4 - 16.0 * c6 * s2 + c8,
            (0, 1): 2.0 * sqrt5 * c * s7 - 12.0 * sqrt5 * c3 * s5
                    + 12.0 * sqrt5 * c5 * s3 - 2.0 * sqrt5 * c7 * s,
            (0, 2): 3.0 * sqrt10 * c2 * s6 - 8.0 * sqrt10 * c4 * s4
                    + 3.0 * sqrt10 * c6 * s2,
            (0, 3): 2.0 * sqrt35 * c3 * s5 - 2.0 * sqrt35 * c5 * s3,
            (0, 4): sqrt70 * c4 * s4,
            (1, 0): -2.0 * sqrt5 * c * s7 + 12.0 * sqrt5 * c3 * s5
                    - 12.0 * sqrt5 * c5 * s3 + 2.0 * sqrt5 * c7 * s,
            (1, 1): -10.0 * c2 * s6 + 30.0 * c4 * s4 - 15.0 * c6 * s2 + c8,
            (1, 2): -10.0 * sqrt2 * c3 * s5 + 15.0 * sqrt2 * c5 * s3
                    - 3.0 * sqrt2 * c7 * s,
            (1, 3): -5.0 * sqrt7 * c4 * s4 + 3.0 * sqrt7 * c6 * s2,
            (1, 4): -2.0 * sqrt14 * c5 * s3,
            (2, 0): 3.0 * sqrt10 * c2 * s6 - 8.0 * sqrt10 * c4 * s4
                    + 3.0 * sqrt10 * c6 * s2,
            (2, 1): 10.0 * sqrt2 * c3 * s5 - 15.0 * sqrt2 * c5 * s3
                    + 3.0 * sqrt2 * c7 * s,
            (2, 2): 15.0 * c4 * s4 - 12.0 * c6 * s2 + c8,
            (2, 3): 3.0 * sqrt14 * c5 * s3 - sqrt14 * c7 * s,
            (2, 4): 2.0 * sqrt7 * c6 * s2,
            (3, 0): -2.0 * sqrt35 * c3 * s5 + 2.0 * sqrt35 * c5 * s3,
            (3, 1): -5.0 * sqrt7 * c4 * s4 + 3.0 * sqrt7 * c6 * s2,
            (3, 2): -3.0 * sqrt14 * c5 * s3 + sqrt14 * c7 * s,
            (3, 3): -7.0 * c6 * s2 + c8,
            (3, 4): -2.0 * sqrt2 * c7 * s,
            (4, 0): sqrt70 * c4 * s4,
            (4, 1): 2.0 * sqrt14 * c5 * s3,
            (4, 2): 2.0 * sqrt7 * c6 * s2,
            (4, 3): 2.0 * sqrt2 * c7 * s,
            (4, 4): c8,
        }

    raise ValueError("Hardcoded Wigner-d formulas only cover ell = 2, 3, 4.")


def wigner_d_matrices(quat, ell_max=4):
    """Wigner-D matrices for a quaternion time series.

    Port of ``wignerD_matrices`` (precessing_utils.c:1119) restricted to the
    hardcoded ell <= 4 path, with the two edge-case branches (|ra| ~ 0 or
    |rb| ~ 0) handled by masking instead of index scatter.

    quat has shape (4, N) and is assumed unit-normalized. Returns a list of
    complex arrays [(5, 5, N), ...] for ell = 2..ell_max, indexed
    [ell + m, ell + mp, time].
    """
    if ell_max < 2 or ell_max > 4:
        raise ValueError("wigner_d_matrices supports 2 <= ell_max <= 4.")

    ra = quat[0] + 1j * quat[3]
    rb = quat[2] + 1j * quat[1]
    abs_ra_sqr = quat[0]**2 + quat[3]**2
    abs_rb_sqr = quat[2]**2 + quat[1]**2

    ra_is_zero = abs_ra_sqr < _EDGE_CASE_THRESHOLD
    rb_is_zero = abs_rb_sqr < _EDGE_CASE_THRESHOLD
    is_general = ~(ra_is_zero | rb_is_zero)

    # --- General case, with masked-out points made numerically safe ---
    # The squared magnitudes are guarded BEFORE the sqrt: edge-case points
    # (|ra| ~ 0 or |rb| ~ 0) would otherwise poison gradients through the
    # masked branch (d/dx sqrt(0) = inf, and inf * 0 = NaN in the vjp).
    safe_c = jnp.sqrt(jnp.where(is_general, abs_ra_sqr, 1.0))  # |ra|
    safe_s = jnp.sqrt(jnp.where(is_general, abs_rb_sqr, 1.0))  # |rb|
    unit_phase_a = ra / safe_c
    unit_phase_b = rb / safe_s

    max_phase_power = 2 * ell_max
    phase_powers_a = _complex_unit_power_table(unit_phase_a, max_phase_power)
    phase_powers_b = _complex_unit_power_table(unit_phase_b, max_phase_power)

    # --- Edge cases: only one (anti)diagonal is nonzero ---
    # |ra| ~ 0: D[m, -m] = sign * rb^(2m), sign = +1 iff (ell + m) is odd.
    # |rb| ~ 0: D[m, m] = ra^(2m).
    safe_rb = jnp.where(ra_is_zero, rb, 1.0)
    safe_ra = jnp.where(rb_is_zero, ra, 1.0)
    rb_powers = _complex_reciprocal_power_table(safe_rb, max_phase_power)
    ra_powers = _complex_reciprocal_power_table(safe_ra, max_phase_power)

    matrices = []
    for ell in range(2, ell_max + 1):
        dim = 2 * ell + 1
        num_times = quat.shape[1]
        d_general = _wigner_d_fundamental_domain(safe_c, safe_s, ell)

        matrix = jnp.zeros((dim, dim, num_times), dtype=jnp.complex128)

        # General case: four-position symmetry writes, as in the C hot loop.
        for mp in range(0, ell + 1):
            m_start = -mp if mp > 0 else 0
            for m in range(m_start, ell + 1):
                d_value = d_general[(m, mp)]
                symmetry_sign = -1.0 if (m - mp) % 2 else 1.0

                # Position 1: (m, mp)
                value = d_value * phase_powers_a[m + mp] \
                    * phase_powers_b[m - mp]
                matrix = matrix.at[ell + m, ell + mp].set(value)
                # Position 2: (mp, m)
                if mp != m:
                    value = symmetry_sign * d_value \
                        * phase_powers_a[m + mp] * phase_powers_b[mp - m]
                    matrix = matrix.at[ell + mp, ell + m].set(value)
                # Position 3: (-m, -mp)
                if m > 0 or mp > 0:
                    value = symmetry_sign * d_value \
                        * phase_powers_a[-(m + mp)] * phase_powers_b[mp - m]
                    matrix = matrix.at[ell - m, ell - mp].set(value)
                # Position 4: (-mp, -m)
                if mp != m and (m > 0 or mp > 0):
                    value = d_value * phase_powers_a[-(m + mp)] \
                        * phase_powers_b[m - mp]
                    matrix = matrix.at[ell - mp, ell - m].set(value)

        # Edge cases overwrite the whole matrix (all other entries zero).
        edge_ra_zero = jnp.zeros_like(matrix)
        edge_rb_zero = jnp.zeros_like(matrix)
        for m in range(-ell, ell + 1):
            sign = 1.0 if (ell + m) % 2 else -1.0
            edge_ra_zero = edge_ra_zero.at[ell + m, ell - m].set(
                sign * rb_powers[2 * m])
            edge_rb_zero = edge_rb_zero.at[ell + m, ell + m].set(
                ra_powers[2 * m])

        matrix = jnp.where(is_general, matrix,
                           jnp.where(ra_is_zero, edge_ra_zero, edge_rb_zero))
        matrices.append(matrix)

    return matrices


def rotate_waveform(quat, h_modes, ell_max=4):
    """Rotate waveform modes from the coprecessing to the inertial frame.

    Port of ``rotateWaveform`` (precessing_surrogate.py:78). quat has shape
    (4, N); h_modes has shape (n_modes, N) with modes ordered
    (2,-2)..(2,2), (3,-3)..(3,3), ... and n_modes = ell_max^2 + 2*ell_max - 3.
    """
    matrices = wigner_d_matrices(quat_inv(quat), ell_max)

    rotated_blocks = []
    mode_offset = 0
    for ell in range(2, ell_max + 1):
        dim = 2 * ell + 1
        h_block = h_modes[mode_offset:mode_offset + dim]  # (dim, N)
        # res[m] = sum_mp D[ell+m, ell+mp] * h[mp]
        rotated_blocks.append(
            jnp.einsum("abt,bt->at", matrices[ell - 2], h_block))
        mode_offset += dim
    return jnp.concatenate(rotated_blocks, axis=0)
