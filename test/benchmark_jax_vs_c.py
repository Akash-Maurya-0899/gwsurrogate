"""Benchmark: NRSur7dq4 JAX implementation vs the NumPy/C reference.

Run from the repository root inside the esigmapy_jax environment:

    python test/benchmark_jax_vs_c.py [--batch-sizes 1 8 64 256] [--repeat 5]

Measures, with jit warm-up excluded and .block_until_ready() on all JAX
timings:
  - jit compile time (first call) for the single and batched evaluators,
  - single-evaluation latency (full dimensionless waveform, all modes),
  - batched throughput: jit(vmap) batches vs a Python loop over the C
    implementation.

Force CPU with JAX_PLATFORMS=cpu if the CUDA setup is flaky.
"""

import argparse
import os
import sys
import time
import timeit

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import gwsurrogate  # noqa: E402
from gwsurrogate.jax import NRSur7dq4JAX  # noqa: E402


def random_parameters(batch_size, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.uniform(1.0, 4.0, batch_size)
    chiA = rng.uniform(-1.0, 1.0, (batch_size, 3))
    chiA *= (rng.uniform(0.0, 0.8, batch_size)
             / np.linalg.norm(chiA, axis=1))[:, None]
    chiB = rng.uniform(-1.0, 1.0, (batch_size, 3))
    chiB *= (rng.uniform(0.0, 0.8, batch_size)
             / np.linalg.norm(chiB, axis=1))[:, None]
    return q, chiA, chiB


def time_call(fn, repeat, number=1):
    times = timeit.repeat(fn, repeat=repeat, number=number)
    return min(times) / number


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=[1, 8, 64, 256])
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    print("JAX devices: %s" % jax.devices())
    print("Loading models...")
    reference = gwsurrogate.LoadSurrogate("NRSur7dq4")
    jax_surrogate = NRSur7dq4JAX()

    q, chiA, chiB = random_parameters(1)
    q0, chiA0, chiB0 = float(q[0]), chiA[0], chiB[0]

    # --- Reference (C-backed) single evaluation ---
    def run_reference():
        reference(q=q0, chiA0=chiA0, chiB0=chiB0, f_low=0.0)

    reference_latency = time_call(run_reference, args.repeat)
    print("\nSingle evaluation (all modes, dimensionless, full grid):")
    print("  reference (NumPy/C): %8.2f ms" % (reference_latency * 1e3))

    # --- JAX single evaluation ---
    def run_jax_single():
        _, h, _ = jax_surrogate(q0, chiA0, chiB0, f_low=0.0)
        return h

    compile_start = time.perf_counter()
    run_jax_single()
    single_compile_time = time.perf_counter() - compile_start
    jax_latency = time_call(run_jax_single, args.repeat)
    print("  JAX (jit, warm):     %8.2f ms   (compile: %.1f s)"
          % (jax_latency * 1e3, single_compile_time))
    print("  speedup vs reference: %.2fx"
          % (reference_latency / jax_latency))

    # --- Batched evaluation ---
    print("\nBatched evaluation (JAX jit(vmap) vs Python loop over C):")
    print("  %6s %14s %14s %12s %14s" % (
        "B", "C loop [ms]", "JAX [ms]", "speedup", "JAX ms/wf"))
    for batch_size in args.batch_sizes:
        qb, chiAb, chiBb = random_parameters(batch_size, seed=batch_size)

        def run_reference_loop():
            for i in range(batch_size):
                reference(q=float(qb[i]), chiA0=chiAb[i], chiB0=chiBb[i],
                          f_low=0.0)

        def run_jax_batch():
            jax_surrogate.eval_modes_batch(
                qb, chiAb, chiBb).block_until_ready()

        compile_start = time.perf_counter()
        run_jax_batch()
        batch_compile_time = time.perf_counter() - compile_start

        loop_time = time_call(run_reference_loop,
                              max(2, args.repeat // 2))
        jax_time = time_call(run_jax_batch, args.repeat)
        print("  %6d %14.2f %14.2f %11.2fx %14.3f   (compile: %.1f s)"
              % (batch_size, loop_time * 1e3, jax_time * 1e3,
                 loop_time / jax_time, jax_time * 1e3 / batch_size,
                 batch_compile_time))

    # --- Stage breakdown (JAX, warm) ---
    from gwsurrogate.jax import dynamics as jax_dynamics
    data = jax_surrogate.data
    dynamics_jit = jax.jit(jax_dynamics.integrate_dynamics_from_start)

    def run_dynamics_only():
        dynamics_jit(data.dynamics, q0, jnp.asarray(chiA0),
                     jnp.asarray(chiB0)).block_until_ready()

    run_dynamics_only()
    dynamics_latency = time_call(run_dynamics_only, args.repeat)
    print("\nJAX stage breakdown (single evaluation, warm):")
    print("  dynamics ODE scan:   %8.2f ms (%.0f%% of total)"
          % (dynamics_latency * 1e3,
             100 * dynamics_latency / jax_latency))
    print("  everything else:     %8.2f ms"
          % ((jax_latency - dynamics_latency) * 1e3))


if __name__ == "__main__":
    main()
