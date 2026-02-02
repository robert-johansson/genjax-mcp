"""Use GenerativeFunction.repeat to draw IID samples in one call.

Run with:
    source venv/bin/activate && python examples/learn/examples/gf_repeat.py
"""

from __future__ import annotations

import jax.numpy as jnp

import genjax
from genjax import gen


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


def main() -> None:
    repeated = quadratic_with_outlier.repeat(n=10)
    trace = repeated.simulate(
        0.0,
        0.2,
        jnp.array([0.5, 0.0, 0.0]),
        0.5,
        5.0,
    )

    choices = trace.get_choices()
    ys = jnp.asarray(choices["y"], dtype=jnp.float32)

    print("repeated choices shape:", ys.shape)
    print("samples:", ys.tolist())


if __name__ == "__main__":
    main()
