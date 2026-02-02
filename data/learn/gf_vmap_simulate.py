"""Vectorise GenJAX simulation over inputs with jax.vmap.

Run with:
    source venv/bin/activate && python examples/learn/examples/gf_vmap_simulate.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import genjax
from genjax import gen, seed


@gen
def quadratic_with_outlier(
    x: float,
    p: float,
    a: jnp.ndarray,
    sigma_inlier: float,
    sigma_outlier: float,
) -> jnp.ndarray:
    z = genjax.flip(p) @ "z"
    mu = a[0] + a[1] * x + a[2] * (x**2)
    sigma = jnp.where(z, sigma_outlier, sigma_inlier)
    y = genjax.normal(mu, sigma) @ "y"
    return y


def simulate_retval(key: jax.Array, x: float) -> jnp.ndarray:
    trace = seed(quadratic_with_outlier.simulate)(
        key,
        x,
        0.1,
        jnp.array([0.5, -0.25, 0.1]),
        0.3,
        8.0,
    )
    return jnp.asarray(trace.get_retval())


def main() -> None:
    xs = jnp.linspace(-2.0, 2.0, 32)
    key = jax.random.key(0)
    keys = jax.random.split(key, xs.shape[0])

    vmapped = jax.jit(jax.vmap(simulate_retval, in_axes=(0, 0)))
    ys = vmapped(keys, xs)

    print("samples shape:", ys.shape)
    print("mean / std:", float(jnp.mean(ys)), float(jnp.std(ys)))


if __name__ == "__main__":
    main()
