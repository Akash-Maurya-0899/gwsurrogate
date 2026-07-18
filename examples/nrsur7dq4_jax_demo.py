# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # The JAX backend of the NRSur7dq4 surrogate
#
# `gwsurrogate.jax.NRSur7dq4JAX` is a pure-JAX re-implementation of the
# NRSur7dq4 precessing numerical-relativity surrogate, validated against the
# NumPy/C reference implementation to better than `1e-10` of the waveform
# peak (typically `~1e-13`). Being JAX end-to-end buys features the C
# backend cannot offer:
#
# * **JIT compilation** — the whole pipeline (precession dynamics
#   integration → spline resampling → coorbital mode fits → inertial-frame
#   rotation) compiles to a single XLA program.
# * **Batched evaluation** — `jit(vmap(...))` evaluates whole parameter-space
#   batches at once, amortizing dispatch overhead and saturating the device.
# * **CPU *and* GPU** — the same code runs on either backend, unchanged.
# * **Automatic differentiation** — exact gradients of any waveform summary
#   with respect to the physical parameters (q, spins), and even along time
#   (through the interpolating spline).
# * Full feature parity with the reference: `f_ref`, `f_low > 0`
#   (low-frequency truncation), physical (mks) units, mode summation.
#
# This notebook tours those features and ends with benchmarks against the
# NumPy/C backend and an accuracy (mismatch) histogram.
#
# Convert to a notebook with `jupytext --to ipynb nrsur7dq4_jax_demo.py`,
# or run top-to-bottom as a plain script (figures are saved to
# `examples/figures/`). Requires the `NRSur7dq4.h5` model data in
# `gwsurrogate/surrogate_downloads/`.

# %%
import os
import sys
import time

# Do not let XLA grab most of the GPU memory up front (flaky on WSL2).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Make the repo checkout importable when running from examples/.
try:
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:  # __file__ is undefined inside a notebook kernel
    REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
if os.path.isdir(os.path.join(REPO_ROOT, "gwsurrogate")):
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import gwsurrogate.jax  # sets jax_enable_x64 BEFORE any array exists
import jax
import jax.numpy as jnp
import gwtools

from gwsurrogate.jax import NRSur7dq4JAX

assert jax.config.read("jax_enable_x64"), "float64 must be enabled"
print("JAX", jax.__version__, "— devices:", jax.devices())

FIG_DIR = os.path.join(REPO_ROOT, "examples", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# %% [markdown]
# ### Plot style
#
# A small, consistent style: recessive grid and axes, thin marks, and a
# fixed, colorblind-validated categorical palette. Color follows the
# *backend/device* (blue = JAX CPU, aqua = JAX GPU, yellow = NumPy/C);
# line style distinguishes single (solid) from batched (dashed) so that
# identity is never carried by color alone.

# %%
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"  # validated order

matplotlib.rcParams.update({
    "figure.figsize": (8.0, 5.0),
    "figure.dpi": 110,
    "savefig.dpi": 170,
    "savefig.bbox": "tight",
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "legend.frameon": False,
    "font.size": 11,
})


def finish(fig, filename):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, filename))
    plt.show()
    print("saved", os.path.join("figures", filename))

# %% [markdown]
# ## Loading the surrogate on CPU (and, if present, GPU)
#
# One `NRSur7dq4JAX` instance is created per device, inside a
# `jax.default_device(...)` context so its model-data arrays (and every
# computation run inside the same context) live on that device. The GPU is
# optional — everything below degrades gracefully to CPU-only.

# %%
CPU = jax.devices("cpu")[0]
try:
    GPU = jax.devices("gpu")[0]
except RuntimeError:
    GPU = None
print("CPU:", CPU, "| GPU:", GPU if GPU is not None else "not available")

with jax.default_device(CPU):
    sur_cpu = NRSur7dq4JAX()

sur_gpu = None
if GPU is not None:
    with jax.default_device(GPU):
        sur_gpu = NRSur7dq4JAX()

# The NumPy/C reference implementation, for comparison and benchmarks.
import gwsurrogate
sur_c = gwsurrogate.LoadSurrogate("NRSur7dq4")

# %% [markdown]
# ## Basic evaluation, and truncating with `f_low`
#
# The call signature mirrors the standard `gwsurrogate` evaluator: mass
# ratio `q`, dimensionless spin vectors at the reference epoch, and either
# dimensionless or physical (mks) units. With `f_low=0` the full surrogate
# length (~4300 M before merger) is returned; with `f_low > 0` the output
# starts at the time `t_low` where the (2,2)-mode instantaneous frequency
# reaches `f_low`, exactly as in the reference implementation.

# %%
q = 2.3
chiA0 = np.array([0.3, -0.2, 0.4])   # spin of the heavier black hole
chiB0 = np.array([-0.1, 0.2, -0.3])  # spin of the lighter black hole

with jax.default_device(CPU):
    t_full, h_full, _ = sur_cpu(q, chiA0, chiB0, f_low=0)
    t_cut, h_cut, _ = sur_cpu(q, chiA0, chiB0, f_low=8e-3)  # cycles/M

print("full grid: %d samples from t=%.0f M" % (len(t_full), t_full[0]))
print("f_low=8e-3: %d samples from t=%.0f M" % (len(t_cut), t_cut[0]))

fig, ax = plt.subplots()
ax.plot(t_full, h_full[(2, 2)].real, color=GRID, lw=2,
        label="full surrogate (f_low = 0)")
ax.plot(t_cut, h_cut[(2, 2)].real, color=BLUE, lw=2,
        label="truncated (f_low = 8e-3 cycles/M)")
ax.set_xlabel("time [M]")
ax.set_ylabel(r"Re $h_{22}$")
ax.set_title("NRSur7dq4JAX: (2,2) mode, with and without f_low truncation")
ax.legend(loc="upper left")
finish(fig, "demo_flow_truncation.png")

# %% [markdown]
# ## Physical units — and a first look at JAX/C agreement
#
# With `units='mks'` you give the total mass in solar masses and the
# distance in Mpc; `f_low`/`f_ref` are then in Hz, `dt`/`times` in
# seconds. Below, both backends generate the same 20 Hz, 70 M☉ system and
# the curves lie exactly on top of each other.

# %%
M_TOTAL, DIST_MPC, DT_PHYS = 70.0, 400.0, 1.0 / 4096

with jax.default_device(CPU):
    t_jax, h_jax, _ = sur_cpu(q, chiA0, chiB0, f_low=20.0, M=M_TOTAL,
                              dist_mpc=DIST_MPC, units="mks", dt=DT_PHYS)
t_ref, h_ref, _ = sur_c(q, chiA0, chiB0, f_low=20.0, M=M_TOTAL,
                        dist_mpc=DIST_MPC, units="mks", dt=DT_PHYS)

peak = np.abs(h_ref[(2, 2)]).max()
# the final dt-grid sample is excluded: the C++ spline's interval search
# ("hunt") has a known off-by-one there (see CLAUDE.md)
worst = max(np.abs(h_jax[m][:-1] - h_ref[m][:-1]).max() for m in h_ref)
print("max |JAX - C| over all 21 modes: %.2e  (%.1e of the peak)"
      % (worst, worst / peak))

fig, ax = plt.subplots()
ax.plot(t_jax, h_jax[(2, 2)].real, color=BLUE, lw=2, label="JAX")
ax.plot(t_ref, h_ref[(2, 2)].real, color=YELLOW, lw=1.2, ls="--",
        label="NumPy/C")
ax.set_xlabel("time [s]")
ax.set_ylabel(r"Re $h_{22}$ (strain)")
ax.set_title("70 $M_\\odot$ at 400 Mpc from $f_{low}$ = 20 Hz — "
             "JAX vs NumPy/C")
ax.legend(loc="upper left")
finish(fig, "demo_mks_overlay.png")

# %% [markdown]
# ## JIT: compile once, then fly
#
# The first call at a given output-grid length traces and compiles the
# XLA program; every later call reuses it. (A new grid *length* triggers
# one recompilation — reuse grids of the same length in hot loops.)

# %%
q2, chiA2, chiB2 = 1.7, np.array([0.1, 0.4, -0.2]), np.array([0.3, 0.0, 0.1])

with jax.default_device(CPU):
    # a fresh instance AND a cleared global jit cache, so the earlier
    # cells' cached compilations don't hide the one-off compile cost
    sur_fresh = NRSur7dq4JAX()
    jax.clear_caches()
    start = time.perf_counter()
    sur_fresh(q2, chiA2, chiB2, f_low=0)        # traces + compiles
    first_call = time.perf_counter() - start

    times = []
    for _ in range(10):                          # cached program
        start = time.perf_counter()
        sur_fresh(q2, chiA2, chiB2, f_low=0)
        times.append(time.perf_counter() - start)
del sur_fresh

print("first call (with compile): %6.2f s" % first_call)
print("steady state:              %6.2f ms  (median of 10)"
      % (1e3 * np.median(times)))

# %% [markdown]
# ## Batched evaluation with `vmap`
#
# `eval_modes_batch` maps the whole pipeline over a batch of parameters in
# a single compiled program — no Python loop, one device dispatch. On CPU
# this roughly matches the single-call latency per waveform (the work is
# already compute-bound); on a GPU — where a single call is dominated by
# dispatch latency — it is what unlocks the hardware (see the benchmarks
# below).

# %%
def random_parameters(batch_size, seed=0, q_range=(1.2, 3.8)):
    """Random points in the training region (|chi| <= 0.8)."""
    rng = np.random.default_rng(seed)
    qs = rng.uniform(*q_range, batch_size)
    def spins():
        v = rng.normal(size=(batch_size, 3))
        v /= np.linalg.norm(v, axis=1)[:, None]
        return v * rng.uniform(0.0, 0.8, batch_size)[:, None]
    return qs, spins(), spins()


BATCH = 16
qs, chiAs, chiBs = random_parameters(BATCH, seed=1)

with jax.default_device(CPU):
    h_batch = sur_cpu.eval_modes_batch(qs, chiAs, chiBs)  # compile
    np.asarray(h_batch)

    start = time.perf_counter()
    h_batch = sur_cpu.eval_modes_batch(qs, chiAs, chiBs)
    h_batch.block_until_ready()
    batch_seconds = time.perf_counter() - start

print("batch of %d waveforms, all 21 modes: shape %s" % (BATCH, h_batch.shape))
print("CPU: %.1f ms total = %.2f ms per waveform"
      % (1e3 * batch_seconds, 1e3 * batch_seconds / BATCH))

# %% [markdown]
# ## The same code on the GPU
#
# Nothing changes but the device. A single evaluation is latency-bound
# (the dynamics integration is a ~230-step sequential scan), so the GPU
# only pulls ahead on batches — see the benchmarks below.

# %%
if sur_gpu is not None:
    with jax.default_device(GPU):
        sur_gpu.eval_modes_batch(qs, chiAs, chiBs).block_until_ready()
        start = time.perf_counter()
        h_batch_gpu = sur_gpu.eval_modes_batch(qs, chiAs, chiBs)
        h_batch_gpu.block_until_ready()
        gpu_seconds = time.perf_counter() - start
    print("GPU: %.1f ms total = %.2f ms per waveform (batch of %d)"
          % (1e3 * gpu_seconds, 1e3 * gpu_seconds / BATCH, BATCH))
    print("max |CPU - GPU|: %.2e"
          % np.abs(np.asarray(h_batch) - np.asarray(h_batch_gpu)).max())
else:
    print("No GPU available in this session — skipping.")

# %% [markdown]
# ## Gradients in parameter space
#
# The pipeline is differentiable end-to-end, so `jax.grad` gives exact
# derivatives of any scalar built from the waveform with respect to the
# physical parameters — the workhorse of gradient-based sampling
# (HMC/NUTS), Fisher forecasts, and template-placement studies. Here:
# the derivative of an SNR-like quantity, the time-integrated (2,2)
# power, with respect to all 7 intrinsic parameters at once.

# %%
from gwsurrogate.jax.surrogate import (_evaluate_dimensionless_modes,
                                       _resample_modes)

IDENTITY_QUAT = jnp.array([1.0, 0.0, 0.0, 0.0])
MODE_22 = 4  # modes are ordered (2,-2) ... (2,2), (3,-3) ...


def h22_power(params, data, times):
    """sum |h22(t)|^2 over the output grid; params = (q, chiA, chiB)."""
    h_modes, _, _ = _evaluate_dimensionless_modes(
        data, params[0], params[1:4], params[4:7], IDENTITY_QUAT, 0.0,
        ell_max=2)
    h22 = _resample_modes(data, h_modes, times)[MODE_22]
    return jnp.sum(jnp.real(h22 * jnp.conj(h22)))


params0 = jnp.concatenate([jnp.array([q]), jnp.asarray(chiA0),
                           jnp.asarray(chiB0)])
demo_times = jnp.asarray(sur_cpu.t_coorb[200:-1:2])

with jax.default_device(CPU):
    power_fn = jax.jit(h22_power)
    grad_fn = jax.jit(jax.grad(h22_power))
    gradient = np.asarray(grad_fn(params0, sur_cpu.data, demo_times))

    # cross-check the q-derivative against a central finite difference
    eps = 1e-6
    fd_q = (power_fn(params0.at[0].add(eps), sur_cpu.data, demo_times)
            - power_fn(params0.at[0].add(-eps), sur_cpu.data, demo_times)
            ) / (2 * eps)

labels = ["q", "chiA_x", "chiA_y", "chiA_z", "chiB_x", "chiB_y", "chiB_z"]
for name, value in zip(labels, gradient):
    print("  d(power)/d%-7s = %+.6e" % (name, value))
print("finite-difference check on q: %+.6e  (rel. diff %.1e)"
      % (fd_q, abs(gradient[0] - fd_q) / abs(fd_q)))

# %% [markdown]
# ## Gradients along time
#
# The output grid enters through a (natural cubic) spline, so the strain
# is differentiable with respect to *time* too — no finite-difference
# noise, exact at every sample. Below, `jax.grad` through the spline
# reproduces the numerical derivative of Re $h_{22}$(t).

# %%
with jax.default_device(CPU):
    h_modes_sparse, _, _ = jax.jit(
        _evaluate_dimensionless_modes, static_argnames=("ell_max",))(
            sur_cpu.data, q, jnp.asarray(chiA0), jnp.asarray(chiB0),
            IDENTITY_QUAT, 0.0, ell_max=2)

    def re_h22_at(t):
        """Re h22 at a single (traced) time, via the spline."""
        return jnp.real(_resample_modes(
            sur_cpu.data, h_modes_sparse, t[None])[MODE_22, 0])

    t_window = jnp.linspace(-500.0, 80.0, 600)
    dh_dt = np.asarray(jax.jit(jax.vmap(jax.grad(re_h22_at)))(t_window))
    h_window = np.asarray(jax.jit(jax.vmap(re_h22_at))(t_window))

numerical = np.gradient(h_window, np.asarray(t_window))

fig, ax = plt.subplots()
ax.plot(t_window, dh_dt, color=BLUE, lw=2, label="jax.grad (exact)")
ax.plot(t_window, numerical, color=YELLOW, lw=1.2, ls="--",
        label="np.gradient (finite diff.)")
ax.set_xlabel("time [M]")
ax.set_ylabel(r"$\partial_t\,$Re $h_{22}$")
ax.set_title("Time derivative of the strain through the spline")
ax.legend(loc="upper left")
finish(fig, "demo_time_gradient.png")

# %% [markdown]
# # Benchmarks
#
# Cost of generating a waveform **from $f_{low}$ = 20 Hz** as a function of
# the total mass, at `dt = 1/2048 s` (ample for total masses ≥ 60 M☉;
# spins/frame defined at 20 Hz, i.e. `f_ref = f_low`, for both backends). Below a minimum total mass the
# surrogate is too short to reach down to 20 Hz; the mass axis starts
# just above that minimum and runs to 200 M☉. Curves show the **median**
# over repeated evaluations with a ±1 standard deviation band; compile
# time is excluded (each configuration is warmed up first — the JIT cost
# is a one-off per output-grid length, the ~seconds shown in the JIT
# section above).
#
# * single-point evaluation: JAX CPU, JAX GPU, and the NumPy/C backend;
# * batched evaluation (batch of 16): per-waveform cost, JAX CPU and GPU.

# %%
F_LOW_HZ = 20.0
DT_BENCH = 1.0 / 2048
BENCH_Q, BENCH_CHIA, BENCH_CHIB = 2.0, chiA0, chiB0
N_REP_SINGLE, N_REP_BATCH = 15, 5
BENCH_BATCH = 16
bqs, bchiAs, bchiBs = random_parameters(BENCH_BATCH, seed=2,
                                        q_range=(1.5, 2.5))


def timed(fn, n_rep, n_warmup=2):
    """Median/std of n_rep timings (seconds), after n_warmup warmups."""
    for _ in range(n_warmup):
        fn()
    samples = []
    for _ in range(n_rep):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return np.array(samples)


# --- the minimum total mass reachable from 20 Hz ---------------------------
# omega_low = pi * f_low * (G M / c^3) must exceed the orbital frequency at
# the first surrogate node, which depends (weakly) on q and the spins; take
# the largest over every configuration used below.
def omega_first_node(sur, q_value, chiA, chiB):
    return float(sur._omega_at_first_node(
        sur.data.dynamics, float(q_value), jnp.asarray(chiA),
        jnp.asarray(chiB), 0.0, IDENTITY_QUAT))


with jax.default_device(CPU):
    omega0_max = max(
        [omega_first_node(sur_cpu, BENCH_Q, BENCH_CHIA, BENCH_CHIB)]
        + [omega_first_node(sur_cpu, bqs[i], bchiAs[i], bchiBs[i])
           for i in range(BENCH_BATCH)])

M_MIN = omega0_max / (np.pi * F_LOW_HZ * gwtools.Msuninsec)
MASS_GRID = np.round(np.geomspace(1.04 * M_MIN, 200.0, 7), 1)
print("minimum total mass from 20 Hz: %.1f Msun" % M_MIN)
print("benchmark masses:", MASS_GRID)


def shared_times_dimless(sur, total_mass):
    """A shared dimensionless output grid: t_low(20 Hz) to merger at
    dt = DT_BENCH, matching what the mks dt-grid uses for this mass."""
    t_scale = gwtools.Msuninsec * total_mass
    f_low_dimless = F_LOW_HZ * t_scale
    t_lows = sur.start_times(bqs, bchiAs, bchiBs, f_low_dimless)
    dt_dimless = DT_BENCH / t_scale
    t0, tf = t_lows.max(), sur.t_coorb[-1]
    return t0 + dt_dimless * np.arange(int(np.ceil((tf - t0) / dt_dimless))), \
        f_low_dimless

# %%
# --- run the waveform-cost benchmarks --------------------------------------
bench = {}  # label -> (medians_ms, stds_ms)


def record(label, samples_per_mass):
    arr = 1e3 * np.array(samples_per_mass)  # (n_mass, n_rep) in ms
    bench[label] = (np.median(arr, axis=1), arr.std(axis=1))
    print("  %-22s %s ms" % (label, np.round(np.median(arr, axis=1), 2)))


print("NumPy/C single...")
record("C single", [
    timed(lambda: sur_c(BENCH_Q, BENCH_CHIA, BENCH_CHIB, f_low=F_LOW_HZ,
                        M=mass, dist_mpc=DIST_MPC, units="mks",
                        dt=DT_BENCH), N_REP_SINGLE)
    for mass in MASS_GRID])

for name, sur, device in [("CPU", sur_cpu, CPU), ("GPU", sur_gpu, GPU)]:
    if sur is None:
        continue
    with jax.default_device(device):
        print("JAX %s single..." % name)
        record("JAX %s single" % name, [
            timed(lambda: sur(BENCH_Q, BENCH_CHIA, BENCH_CHIB,
                              f_low=F_LOW_HZ, M=mass, dist_mpc=DIST_MPC,
                              units="mks", dt=DT_BENCH), N_REP_SINGLE)
            for mass in MASS_GRID])

        print("JAX %s batched..." % name)
        samples = []
        for mass in MASS_GRID:
            times_dimless, f_low_dimless = shared_times_dimless(sur, mass)
            samples.append(timed(
                lambda: sur.eval_modes_batch(
                    bqs, bchiAs, bchiBs, f_low=f_low_dimless,
                    times=times_dimless).block_until_ready(),
                N_REP_BATCH) / BENCH_BATCH)
        record("JAX %s batched" % name, samples)

# %%
# --- plot: waveform generation cost vs total mass --------------------------
SERIES_STYLE = {
    "JAX CPU single":  (BLUE, "-", "JAX CPU, single"),
    "JAX GPU single":  (AQUA, "-", "JAX GPU, single"),
    "C single":        (YELLOW, "-", "NumPy/C, single"),
    "JAX CPU batched": (BLUE, "--", "JAX CPU, batched (per wf)"),
    "JAX GPU batched": (AQUA, "--", "JAX GPU, batched (per wf)"),
}

def benchmark_axes(ax, results, title, ylabel):
    """Shared layout: log-log, direct labels in the right margin, legend
    above the axes (so neither can collide with the flat curves)."""
    for key, (color, style, label) in SERIES_STYLE.items():
        if key not in results:
            continue
        med, std = results[key]
        ax.plot(MASS_GRID, med, color=color, ls=style, lw=2, marker="o",
                markersize=4, label=label)
        ax.fill_between(MASS_GRID, np.maximum(med - std, 1e-3), med + std,
                        color=color, alpha=0.15, lw=0)
        ax.annotate(label, (MASS_GRID[-1], med[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color=INK_2, annotation_clip=False)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(MASS_GRID)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel(r"total mass $M$ [$M_\odot$]")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=64)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0, 1, 0.25),
              mode="expand", ncols=2, fontsize=9)


fig, ax = plt.subplots()
benchmark_axes(ax, bench,
               "Waveform cost from $f_{low}$ = 20 Hz (median ± std, "
               "dt = 1/2048 s)",
               "evaluation time per waveform [ms]")
finish(fig, "benchmark_waveform_cost.png")

# %% [markdown]
# ## Cost of parameter-space gradients
#
# The same comparison for the 7-parameter gradient of an SNR-like scalar
# (time-integrated power over all 21 modes: full pipeline + resampling
# onto the 20 Hz-truncated physical grid, reverse-mode). There is no
# NumPy/C curve here — the C backend simply cannot produce gradients;
# finite differences would need ≥ 8 full evaluations *per point* and
# still be noisy.
#
# Reverse-mode stores the forward pass's intermediates, so the batched
# gradient is memory-hungry at low masses (long grids); the batch is 8
# here, and jit caches are cleared between mass points to keep the
# footprint flat.

# %%
import gc
def total_power_all_modes(params, data, times):
    """Time-integrated power over ALL 21 modes: keeps the whole pipeline
    live in the backward pass (a single-mode scalar would let XLA prune
    the other modes' fits and understate the cost)."""
    h_modes, _, _ = _evaluate_dimensionless_modes(
        data, params[0], params[1:4], params[4:7], IDENTITY_QUAT, 0.0,
        ell_max=4)
    h_all = _resample_modes(data, h_modes, times)
    return jnp.sum(jnp.real(h_all * jnp.conj(h_all)))


GRAD_BATCH = 8  # reverse-mode residuals for long grids are memory-hungry
params_batch = jnp.column_stack([jnp.asarray(bqs[:GRAD_BATCH]),
                                 jnp.asarray(bchiAs[:GRAD_BATCH]),
                                 jnp.asarray(bchiBs[:GRAD_BATCH])])
N_REP_GRAD_SINGLE, N_REP_GRAD_BATCH = 10, 4

grad_bench = {}
for name, sur, device in [("CPU", sur_cpu, CPU), ("GPU", sur_gpu, GPU)]:
    if sur is None:
        continue
    with jax.default_device(device):
        singles, batches = [], []
        for mass in MASS_GRID:
            # drop the previous mass's compiled programs and residual
            # buffers — without this the run's memory grows past what a
            # 14 GB machine can hold
            jax.clear_caches()
            gc.collect()
            grad_single = jax.jit(jax.grad(total_power_all_modes))
            grad_batch = jax.jit(jax.vmap(jax.grad(total_power_all_modes),
                                          in_axes=(0, None, None)))
            params_dev = jax.device_put(params0, device)
            params_batch_dev = jax.device_put(params_batch, device)
            times_dimless, _ = shared_times_dimless(sur, mass)
            times_dev = jax.device_put(jnp.asarray(times_dimless), device)
            print("gradients, %s, M = %s..." % (name, mass))
            singles.append(timed(
                lambda: grad_single(params_dev, sur.data,
                                    times_dev).block_until_ready(),
                N_REP_GRAD_SINGLE))
            batches.append(timed(
                lambda: grad_batch(params_batch_dev, sur.data,
                                   times_dev).block_until_ready(),
                N_REP_GRAD_BATCH) / GRAD_BATCH)

    arr = 1e3 * np.array(singles)
    grad_bench["JAX %s single" % name] = (np.median(arr, 1), arr.std(1))
    arr = 1e3 * np.array(batches)
    grad_bench["JAX %s batched" % name] = (np.median(arr, 1), arr.std(1))

# %%
fig, ax = plt.subplots()
benchmark_axes(ax, grad_bench,
               "7-parameter gradient cost from $f_{low}$ = 20 Hz "
               "(median ± std)",
               "gradient time per waveform [ms]")
finish(fig, "benchmark_gradient_cost.png")

# %% [markdown]
# ## Accuracy: mismatch-type L2 error against the NumPy/C backend
#
# For random points across the parameter space, both backends are
# evaluated on the surrogate's native time grid and compared through a
# mismatch-style normalized L2 error over all 21 modes,
#
# $$ \varepsilon \;=\; \sqrt{ \frac{\sum_{\ell m} \sum_i
#    |h^{JAX}_{\ell m}(t_i) - h^{C}_{\ell m}(t_i)|^2 }
#    {\sum_{\ell m} \sum_i |h^{C}_{\ell m}(t_i)|^2 } } $$
#
# (an overlap-based mismatch would be $\mathcal{O}(\varepsilon^2)
# \sim 10^{-26}$ — beneath double precision, so the L2 form is the
# resolvable, conservative measure). Every point sits around
# $10^{-14}$–$10^{-13}$: the backends are numerically interchangeable.

# %%
N_MISMATCH = 40
mqs, mchiAs, mchiBs = random_parameters(N_MISMATCH, seed=3, q_range=(1.0, 4.0))

errors = np.empty(N_MISMATCH)
for i in range(N_MISMATCH):
    _, h_c, _ = sur_c(mqs[i], mchiAs[i], mchiBs[i], f_low=0.0)
    with jax.default_device(CPU):
        _, h_j, _ = sur_cpu(mqs[i], mchiAs[i], mchiBs[i], f_low=0)
    num = sum(np.sum(np.abs(h_j[m] - h_c[m])**2) for m in h_c)
    den = sum(np.sum(np.abs(h_c[m])**2) for m in h_c)
    errors[i] = np.sqrt(num / den)

print("normalized L2 error: median %.2e, max %.2e"
      % (np.median(errors), errors.max()))

fig, ax = plt.subplots()
log_err = np.log10(errors)
bins = np.arange(np.floor(log_err.min() * 4) / 4,
                 np.ceil(log_err.max() * 4) / 4 + 0.25, 0.25)
ax.hist(log_err, bins=bins, color=BLUE, edgecolor="white", linewidth=1.5)
ax.axvline(np.median(log_err), color=INK, lw=1.2, ls=":")
ax.annotate("median = %.1e" % np.median(errors),
            (np.median(log_err), ax.get_ylim()[1] * 0.95),
            xytext=(8, 0), textcoords="offset points", fontsize=9,
            color=INK_2, va="top")
ax.set_xlabel(r"$\log_{10}$ normalized L2 error (all 21 modes)")
ax.set_ylabel("number of parameter-space points")
ax.set_title("JAX vs NumPy/C agreement over %d random points" % N_MISMATCH)
finish(fig, "mismatch_histogram.png")

# %% [markdown]
# ## Summary (numbers from this machine: WSL2, CPU + a 6 GB CUDA GPU)
#
# * **Accuracy**: the JAX backend reproduces the NumPy/C backend to
#   $\sim 10^{-14}$–$10^{-13}$ (normalized L2, all 21 modes) across the
#   parameter space — the backends are numerically interchangeable.
# * **Single evaluations** are comparable on CPU: NumPy/C ~9 ms, JAX ~9 ms
#   on the plain `f_low=0` path and ~2x that when a 20 Hz start frequency
#   engages the reference-frequency (`lax.switch`) integrator. A *single*
#   call on the GPU is dominated by dispatch latency (the dynamics
#   integration is a ~230-step sequential scan) — don't use a GPU for one
#   waveform at a time.
# * **Batched** evaluation is where JAX pays off: ~3 ms/waveform on this
#   GPU at batch 16 on the `f_low=0` path (larger batches push this well
#   below 1 ms — see `test/benchmark_jax_vs_c.py`), and the gap over the
#   single-call GPU latency is ~6x even on the heavier 20 Hz path.
# * **Gradients** — impossible with the C backend — cost ~8x a forward
#   evaluation for all 7 intrinsic parameters at once (finite differences
#   would need ≥ 8 evaluations for *one-sided* derivatives, ~15 for
#   central, and still be noisy). Batched on the GPU they run at nearly
#   the same per-waveform cost as batched forward evaluations.
