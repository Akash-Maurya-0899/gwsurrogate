"""User-facing JAX evaluator for the NRSur7dq4 precessing surrogate.

Port of ``PrecessingSurrogate.__call__``
(gwsurrogate/new/precessing_surrogate.py:922) restricted, for now, to
fM_ref = None/0 and fM_low = None/0 (dynamics start at the first surrogate
time node; full reference-frequency support is a later milestone).

The numerical pipeline is fully jitted and vmap-able:

    dynamics (RK4 bootstrap + AB4 scan on the t_ds grid)
      -> natural-spline resample onto the coorbital grid (dense matmul)
      -> quaternion/spin renormalization, coprecessing -> coorbital spins
      -> coorbital-frame modes (EI-node fits + EI-basis contraction)
      -> inertial-frame rotation (Wigner-D)
      -> optional natural-spline resample onto a user time grid

Usage::

    from gwsurrogate.jax import NRSur7dq4JAX
    surrogate = NRSur7dq4JAX()
    t, h, dynamics = surrogate(q, chiA0, chiB0, f_low=0)
    h_batch = surrogate.eval_modes_batch(q_array, chiA0_array, chiB0_array)

Note: each distinct user time-grid length triggers one jit recompilation
(static output shapes); reuse grids of the same length to avoid it.
"""

import functools

import numpy as np
import jax
import jax.numpy as jnp

from . import coorb as jax_coorb
from . import data as jax_data
from . import dynamics as jax_dynamics
from . import quaternions as jax_quaternions
from . import spline as jax_spline

_IDENTITY_QUATERNION = np.array([1.0, 0.0, 0.0, 0.0])


def _normalize_spin_series(chi, chi_norm):
    """Rescale each spin vector of a time series to magnitude chi_norm.

    Port of ``normalize_spin`` (precessing_surrogate.py:849), with the
    host-side ``if chi_norm > 0`` guard replaced by double-``where`` so a
    traced zero norm stays exactly zero without NaNs.
    """
    current_norm = jnp.sqrt(jnp.sum(chi**2, axis=1))
    safe_norm = jnp.where(current_norm > 0.0, current_norm, 1.0)
    scale = jnp.where(current_norm > 0.0, chi_norm / safe_norm, 1.0)
    return chi * scale[:, None]


def _modes_from_dynamics(data, q, chiA_norm, chiB_norm, y_of_t, ell_max):
    """The post-dynamics pipeline shared by all reference-epoch variants.

    Port of ``PrecessingSurrogate.__call__`` :1046-1067.
    """
    # Natural-spline resample of all dynamics series onto t_coorb via the
    # precomputed dense operator (linear in the data; __call__ :1049-1056).
    interpolation_matrix = data.spline.dynamics_to_coorb_matrix
    chiA_copr = interpolation_matrix @ y_of_t[:, 5:8]    # (n_coorb, 3)
    chiB_copr = interpolation_matrix @ y_of_t[:, 8:11]
    orbphase = interpolation_matrix @ y_of_t[:, 4]
    quat = (interpolation_matrix @ y_of_t[:, 0:4]).T     # (4, n_coorb)

    chiA_copr = _normalize_spin_series(chiA_copr, chiA_norm)
    chiB_copr = _normalize_spin_series(chiB_copr, chiB_norm)
    quat = quat / jnp.sqrt(jnp.sum(quat * quat, axis=0))

    # Coorbital-frame spins (coorb_spins_from_copr_spins :827).
    chiA_coorb = jax_quaternions.rotate_spin(chiA_copr, orbphase)
    chiB_coorb = jax_quaternions.rotate_spin(chiB_copr, orbphase)

    h_coorb = jax_coorb.coorbital_waveform_modes(
        data.coorb, q, chiA_coorb, chiB_coorb, ell_max=ell_max)

    # Coorbital -> inertial frame (inertial_waveform_modes :832).
    orbphase_quat = jnp.stack([
        jnp.cos(orbphase / 2.), jnp.zeros_like(orbphase),
        jnp.zeros_like(orbphase), jnp.sin(orbphase / 2.)])
    full_quat = jax_quaternions.multiply_quats(quat, orbphase_quat)
    h_inertial = jax_quaternions.rotate_waveform(full_quat, h_coorb,
                                                 ell_max=ell_max)

    return h_inertial, (quat, orbphase, chiA_copr, chiB_copr), y_of_t


def _evaluate_dimensionless_modes(data, q, chiA0, chiB0, init_quat,
                                  init_orbphase, ell_max):
    """Modes on the coorbital time grid for f_ref=None (start of data).

    Port of the fM_ref=None, fM_low=None path of
    ``PrecessingSurrogate.__call__`` (precessing_surrogate.py:1016-1067).

    Returns (h_inertial (n_modes, n_coorb) complex,
             dynamics_on_coorb = (quat (4, n_coorb), orbphase (n_coorb,),
                                  chiA_copr, chiB_copr (n_coorb, 3)),
             y_of_t (n_dyn, 11) on the dynamics grid).
    """
    chiA_norm = jnp.sqrt(jnp.sum(chiA0**2))
    chiB_norm = jnp.sqrt(jnp.sum(chiB0**2))
    y_of_t = jax_dynamics.integrate_dynamics_from_start(
        data.dynamics, q, chiA0, chiB0, init_quat=init_quat,
        init_orbphase=init_orbphase)
    return _modes_from_dynamics(data, q, chiA_norm, chiB_norm, y_of_t,
                                ell_max)


def _evaluate_dimensionless_modes_at_reference(data, q, chiA0, chiB0,
                                               init_quat, init_orbphase,
                                               omega_ref, ell_max):
    """Modes on the coorbital time grid for a given reference frequency."""
    chiA_norm = jnp.sqrt(jnp.sum(chiA0**2))
    chiB_norm = jnp.sqrt(jnp.sum(chiB0**2))
    y_of_t = jax_dynamics.integrate_dynamics_at_reference(
        data.dynamics, q, chiA0, chiB0, omega_ref, init_quat=init_quat,
        init_orbphase=init_orbphase)
    return _modes_from_dynamics(data, q, chiA_norm, chiB_norm, y_of_t,
                                ell_max)


def _resample_modes(data, h_inertial, timesM):
    """Natural-spline resample of the mode array onto a user time grid."""
    return jax_spline.spline_interpolate(data.spline.t_coorb_grid,
                                         h_inertial, timesM)


def _dynamics_on_times(data, y_of_t, chiA_norm, chiB_norm, timesM):
    """Dynamics quantities resampled from the dynamics grid onto timesM.

    Mirrors the return_dynamics interpolation of ``__call__`` :1113-1124
    (interpolate from the sparse dynamics grid, then renormalize).
    """
    grid = data.spline.t_dynamics_grid
    series = jnp.concatenate([y_of_t[:, 0:4], y_of_t[:, 4:5],
                              y_of_t[:, 5:8], y_of_t[:, 8:11]], axis=1).T
    resampled = jax_spline.spline_interpolate(grid, series, timesM)
    quat = resampled[0:4]
    orbphase = resampled[4]
    chiA_copr = _normalize_spin_series(resampled[5:8].T, chiA_norm)
    chiB_copr = _normalize_spin_series(resampled[8:11].T, chiB_norm)
    quat = quat / jnp.sqrt(jnp.sum(quat * quat, axis=0))
    return quat, orbphase, chiA_copr, chiB_copr


def _inertial_spins(quat, chiA_copr, chiB_copr):
    """Inertial-frame spins from coprecessing spins (__call__ :1126)."""
    chiA_inertial = jax_quaternions.transform_time_dependent_vector(
        quat, chiA_copr.T).T
    chiB_inertial = jax_quaternions.transform_time_dependent_vector(
        quat, chiB_copr.T).T
    return chiA_inertial, chiB_inertial


class NRSur7dq4JAX:
    """JAX implementation of the NRSur7dq4 precessing surrogate.

    Dimensionless evaluation only (physical units are a later milestone).
    Supports f_low = 0/None (full surrogate length) and any valid f_ref
    (reference frequency; the three reference-index branches of the
    integrator are expressed with lax.switch and masked scans).

    The heavy pipeline is jit-compiled on first use (per ellMax and per
    distinct user time-grid length) and evaluated in float64.
    """

    def __init__(self, h5_path=None):
        self.data = jax_data.load_nrsur7dq4_jax_data(h5_path)
        self.t_coorb = np.asarray(self.data.coorb.t_coorb)
        self.ell_max_model = 4

        self._eval_modes = jax.jit(
            _evaluate_dimensionless_modes, static_argnames=("ell_max",))
        self._eval_modes_at_reference = jax.jit(
            _evaluate_dimensionless_modes_at_reference,
            static_argnames=("ell_max",))
        self._omega_at_first_node = jax.jit(
            jax_dynamics.omega_at_first_node)
        # vmap maps keyword arguments over axis 0 (in_axes only covers
        # positionals), so ell_max must be bound statically BEFORE vmap;
        # one compiled batch function is cached per (ell_max, with_f_ref).
        self._eval_modes_batch_cache = {}
        self._resample_modes = jax.jit(_resample_modes)
        self._resample_modes_batch = jax.jit(
            jax.vmap(_resample_modes, in_axes=(None, 0, None)))
        self._dynamics_on_times = jax.jit(_dynamics_on_times)
        self._inertial_spins = jax.jit(_inertial_spins)

    def _batched_mode_evaluator(self, ell_max, with_f_ref=False):
        key = (ell_max, with_f_ref)
        if key not in self._eval_modes_batch_cache:
            if with_f_ref:
                bound = functools.partial(
                    _evaluate_dimensionless_modes_at_reference,
                    ell_max=ell_max)
                in_axes = (None, 0, 0, 0, 0, 0, 0)
            else:
                bound = functools.partial(_evaluate_dimensionless_modes,
                                          ell_max=ell_max)
                in_axes = (None, 0, 0, 0, 0, 0)
            self._eval_modes_batch_cache[key] = jax.jit(
                jax.vmap(bound, in_axes=in_axes))
        return self._eval_modes_batch_cache[key]

    def _validate_omega_ref(self, omega_ref, q, chiA0, chiB0,
                            init_orbphase, init_quat):
        """Host-side range checks mirroring _get_t_from_omega (:329-342)."""
        omega_ref_max_model = 0.201  # NRSur7dq4 (surrogate.py:2492)
        if np.any(omega_ref > omega_ref_max_model):
            raise ValueError(
                "Got omega_ref = %s > %s, too large for the model!"
                % (omega_ref, omega_ref_max_model))
        chiA_rotated = np.asarray(jax_quaternions.rotate_spin(
            jnp.asarray(chiA0), -1 * init_orbphase))
        chiB_rotated = np.asarray(jax_quaternions.rotate_spin(
            jnp.asarray(chiB0), -1 * init_orbphase))
        omega0 = float(self._omega_at_first_node(
            self.data.dynamics, q, jnp.asarray(chiA_rotated),
            jnp.asarray(chiB_rotated), init_orbphase,
            jnp.asarray(init_quat)))
        # Tiny relative slack: omega0 here is computed by the JAX kernel
        # and may differ from a C-computed omega_ref == omega_0 by a ULP.
        if omega_ref < omega0 * (1.0 - 1e-10):
            raise ValueError(
                "Got omega_ref = %.4f < %.4f = omega_0, too small!"
                % (omega_ref, omega0))

    def _validate_inputs(self, q, chiA0, chiB0, f_low, f_ref, ellMax):
        if f_low not in (None, 0, 0.0):
            raise NotImplementedError(
                "NRSur7dq4JAX currently supports only f_low=0 (full "
                "surrogate length); use the NumPy/C implementation for "
                "f_low > 0.")
        if f_ref is not None and not np.isscalar(f_ref):
            raise ValueError("f_ref must be a scalar (or None).")
        if ellMax is not None and not 2 <= ellMax <= self.ell_max_model:
            raise ValueError("ellMax must be in 2..%d." % self.ell_max_model)

        q = np.atleast_1d(np.asarray(q, dtype=np.float64))
        chiA0 = np.asarray(chiA0, dtype=np.float64)
        chiB0 = np.asarray(chiB0, dtype=np.float64)
        if np.any(q < 0.99):
            raise ValueError("Mass ratio q must be >= 1.")
        for name, chi in (("chiA0", chiA0), ("chiB0", chiB0)):
            norms = np.sqrt(np.sum(np.atleast_2d(chi)**2, axis=-1))
            if np.any(norms > 1.001):
                raise ValueError("%s has magnitude > 1." % name)
        if np.any(q > 6.01):
            raise ValueError("q=%s outside the hard limit q <= 6." % q)

    def _prepare_times(self, dt, times):
        """Uniform grid from dt, or validated user times (host-side).

        Mirrors ``__call__`` :1069-1096 with t0 = t_coorb[0] (f_low=0).
        """
        if dt is not None and times is not None:
            raise ValueError("Specify at most one of dt and times.")
        if dt is not None:
            t0 = self.t_coorb[0]
            tf = self.t_coorb[-1]
            num_times = int(np.ceil((tf - t0) / dt))
            return t0 + dt * np.arange(num_times)
        if times is not None:
            times = np.asarray(times, dtype=np.float64)
            if times[-1] > self.t_coorb[-1] + 0.01:
                raise ValueError("'times' includes times larger than the "
                                 "maximum time value in domain.")
            if times[0] < self.t_coorb[0]:
                raise ValueError("'times' starts before start of domain.")
            return times
        return None

    def __call__(self, q, chiA0, chiB0, M=None, dist_mpc=None, f_low=None,
                 f_ref=None, dt=None, times=None, ellMax=None,
                 inclination=None, phi_ref=0, precessing_opts=None,
                 units="dimensionless"):
        """Evaluate inertial-frame modes h_lm(t) (or the summed strain).

        Arguments mirror ``SurrogateEvaluator.__call__``
        (gwsurrogate/surrogate.py:1721): q, chiA0, chiB0 are the mass
        ratio and reference-epoch spins (lalsimulation conventions);
        f_low must be 0/None for now; f_ref is the reference frequency at
        which the frame and spins are defined (None/0 = start of the
        surrogate data); dt or times select an output grid (default: the
        coorbital grid); ellMax limits the modes; precessing_opts supports
        'init_orbphase', 'init_quat' and 'return_dynamics'.

        With units='mks', specify both M (solar masses) and dist_mpc;
        f_low/f_ref are then in Hz and dt/times in seconds, and the
        returned strain is physically scaled. With inclination not None,
        the modes are summed at (inclination, pi/2 - phi_ref) following
        the LAL convention and a single complex strain is returned in
        place of the mode dict.

        Returns (times, h, dynamics) where h maps (ell, m) -> complex
        time series (or is the summed strain) and dynamics is None unless
        return_dynamics is set.
        """
        # Unit scalings (SurrogateEvaluator.__call__ :2039-2062).
        if (M is None) ^ (dist_mpc is None):
            raise ValueError("Either specify both M and dist_mpc, or "
                             "neither")
        if (M is not None) ^ (units == "mks"):
            raise ValueError("M/dist_mpc must be specified if and only if "
                             "units='mks'")
        if units == "dimensionless":
            amp_scale = 1.0
            t_scale = 1.0
        elif units == "mks":
            import gwtools as _gwtools
            amp_scale = M * _gwtools.Msuninsec * _gwtools.c \
                / (1e6 * dist_mpc * _gwtools.PC_SI)
            t_scale = _gwtools.Msuninsec * M
        else:
            raise ValueError("Invalid units")

        f_low = None if f_low is None else f_low * t_scale
        f_ref = None if f_ref is None else f_ref * t_scale
        dt = None if dt is None else dt / t_scale
        times = None if times is None else np.asarray(times) / t_scale

        precessing_opts = dict(precessing_opts or {})
        init_orbphase = precessing_opts.pop("init_orbphase", 0.0) or 0.0
        init_quat = precessing_opts.pop("init_quat", None)
        return_dynamics = precessing_opts.pop("return_dynamics", False)
        if precessing_opts:
            raise ValueError("Unused keys in precessing_opts: %s"
                             % sorted(precessing_opts))

        self._validate_inputs(q, chiA0, chiB0, f_low, f_ref, ellMax)
        ell_max = self.ell_max_model if ellMax is None else ellMax
        if init_quat is None:
            init_quat = _IDENTITY_QUATERNION

        q = float(np.asarray(q).reshape(()))
        chiA0 = jnp.asarray(chiA0, dtype=jnp.float64)
        chiB0 = jnp.asarray(chiB0, dtype=jnp.float64)
        init_quat = jnp.asarray(init_quat, dtype=jnp.float64)
        init_orbphase = jnp.asarray(init_orbphase, dtype=jnp.float64)

        if f_ref is None or f_ref == 0:
            h_inertial, dynamics_on_coorb, y_of_t = self._eval_modes(
                self.data, q, chiA0, chiB0, init_quat, init_orbphase,
                ell_max=ell_max)
        else:
            omega_ref = float(f_ref) * np.pi
            self._validate_omega_ref(omega_ref, q, np.asarray(chiA0),
                                     np.asarray(chiB0),
                                     float(init_orbphase),
                                     np.asarray(init_quat))
            h_inertial, dynamics_on_coorb, y_of_t = \
                self._eval_modes_at_reference(
                    self.data, q, chiA0, chiB0, init_quat, init_orbphase,
                    jnp.asarray(omega_ref), ell_max=ell_max)

        output_times = self._prepare_times(dt, times)
        if output_times is None:
            output_times = np.copy(self.t_coorb)
        else:
            h_inertial = self._resample_modes(
                self.data, h_inertial, jnp.asarray(output_times))

        h_array = np.asarray(h_inertial)
        h = {}
        mode_index = 0
        for ell in range(2, ell_max + 1):
            for m in range(-ell, ell + 1):
                h[(ell, m)] = h_array[mode_index]
                mode_index += 1

        dynamics = None
        if return_dynamics:
            if times is None and dt is None:
                quat, orbphase, chiA_copr, chiB_copr = dynamics_on_coorb
            else:
                chiA_norm = jnp.sqrt(jnp.sum(chiA0**2))
                chiB_norm = jnp.sqrt(jnp.sum(chiB0**2))
                quat, orbphase, chiA_copr, chiB_copr = \
                    self._dynamics_on_times(self.data, y_of_t, chiA_norm,
                                            chiB_norm,
                                            jnp.asarray(output_times))
            chiA_inertial, chiB_inertial = self._inertial_spins(
                quat, chiA_copr, chiB_copr)
            dynamics = {
                "chiA": np.asarray(chiA_inertial),
                "chiB": np.asarray(chiB_inertial),
                "chiA_copr": np.asarray(chiA_copr),
                "chiB_copr": np.asarray(chiB_copr),
                "q_copr": np.asarray(quat),
                "orbphase": np.asarray(orbphase),
            }

        # Mode sum at (inclination, pi/2 - phi_ref), LAL convention
        # (SurrogateEvaluator.__call__ :2085; _mode_sum :1704 — precessing
        # models carry all m modes, no faked negative modes).
        if inclination is not None:
            from gwtools.harmonics import sYlm as _sYlm
            theta = inclination
            phi = np.pi / 2 - phi_ref
            h = sum(_sYlm(-2, ell, m, theta, phi) * h_mode
                    for (ell, m), h_mode in h.items())

        # Rescale to physical units (:2102, :2124).
        output_times = output_times * t_scale
        if amp_scale != 1:
            if isinstance(h, dict):
                h = {mode: series * amp_scale for mode, series in h.items()}
            else:
                h = h * amp_scale

        return output_times, h, dynamics

    def eval_modes_batch(self, q, chiA0, chiB0, ellMax=None, times=None,
                         f_ref=None):
        """Batched dimensionless mode evaluation via jit(vmap(...)).

        Arguments:
            q: (B,) mass ratios.
            chiA0, chiB0: (B, 3) reference-epoch spins.
            ellMax: largest ell (static; default 4).
            times: optional shared output time grid, 1D.
            f_ref: optional (B,) reference frequencies (dimensionless,
                cycles/M). Under vmap all three reference-index branches
                are evaluated per element, so this path is slower than
                f_ref=None. Values must be in range (validated per
                element host-side).

        Returns a complex array with shape (B, n_modes, T) where modes are
        ordered (2,-2)..(2,2), (3,-3)..(3,3), ... and T is len(times) or
        the coorbital grid length.
        """
        self._validate_inputs(q, chiA0, chiB0, None, None, ellMax)
        ell_max = self.ell_max_model if ellMax is None else ellMax

        q = jnp.asarray(q, dtype=jnp.float64)
        chiA0 = jnp.asarray(chiA0, dtype=jnp.float64)
        chiB0 = jnp.asarray(chiB0, dtype=jnp.float64)
        batch_size = q.shape[0]
        init_quat = jnp.broadcast_to(jnp.asarray(_IDENTITY_QUATERNION),
                                     (batch_size, 4))
        init_orbphase = jnp.zeros(batch_size)

        if f_ref is None:
            h_inertial, _, _ = self._batched_mode_evaluator(ell_max)(
                self.data, q, chiA0, chiB0, init_quat, init_orbphase)
        else:
            omega_ref = np.asarray(f_ref, dtype=np.float64) * np.pi
            for i in range(batch_size):
                self._validate_omega_ref(
                    omega_ref[i], float(q[i]), np.asarray(chiA0[i]),
                    np.asarray(chiB0[i]), 0.0, _IDENTITY_QUATERNION)
            h_inertial, _, _ = self._batched_mode_evaluator(
                ell_max, with_f_ref=True)(
                    self.data, q, chiA0, chiB0, init_quat, init_orbphase,
                    jnp.asarray(omega_ref))
        if times is not None:
            output_times = self._prepare_times(None, times)
            h_inertial = self._resample_modes_batch(
                self.data, h_inertial, jnp.asarray(output_times))
        return h_inertial
