"""Simulate a single draw from an addressed GenJAX model.

Run with:
    source venv/bin/activate && python examples/learn/examples/gf_simulate_once.py
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
    trace = quadratic_with_outlier.simulate(
        1.25, 0.1, jnp.array([0.5, -0.25, 0.1]), 0.3, 8.0
    )

    print("return value:", float(trace.get_retval()))
    print("choice z:", bool(trace.get_choices()["z"]))
    print("choice y:", float(trace.get_choices()["y"]))
    print("score:", float(trace.get_score()))


if __name__ == "__main__":
    main()
