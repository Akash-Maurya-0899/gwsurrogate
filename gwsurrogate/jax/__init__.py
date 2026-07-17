"""JAX implementation of the NRSur7dq4 precessing surrogate model.

This subpackage provides a JAX port of the NumPy/C implementation in
``gwsurrogate.new.precessing_surrogate``, gaining JIT compilation, ``vmap``
over batches of intrinsic parameters, and hardware-agnostic execution.

JAX is an optional dependency::

    pip install gwsurrogate[jax]

Note: importing this subpackage enables float64 computation globally in JAX
(``jax_enable_x64``). The surrogate arithmetic requires double precision to
match the reference implementation, and JAX's x64 flag is process-wide.
"""

try:
    import jax
except ImportError as e:
    raise ImportError(
        "gwsurrogate.jax requires the optional dependency 'jax'. "
        "Install it with: pip install gwsurrogate[jax]"
    ) from e

# Must be set before any JAX arrays are created anywhere in this subpackage.
jax.config.update("jax_enable_x64", True)
