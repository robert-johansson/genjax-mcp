"""Assess requires all addressed choices; marginalise manually when needed.

Run with:
    source venv/bin/activate && python examples/learn/examples/choice_maps_assess_all.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import genjax
from genjax import gen


@gen
def quad_with_outlier(
    x: float,
    p: float,
    a: jnp.ndarray,
    sigma_in: float,
    sigma_out: float,
) -> jnp.ndarray:
    z = genjax.flip(p) @ "z"
    mu = a[0] + a[1] * x + a[2] * (x**2)
    sigma = jnp.where(z, sigma_out, sigma_in)
    y = genjax.normal(mu, sigma) @ "y"
    return y


def main() -> None:
    y_obs = 1.0

    logp_true, _ = quad_with_outlier.assess(
        {"y": y_obs, "z": True}, 0.5, 0.25, jnp.array([0.2, 0.3, -0.1]), 0.5, 7.0
    )
    logp_false, _ = quad_with_outlier.assess(
        {"y": y_obs, "z": False}, 0.5, 0.25, jnp.array([0.2, 0.3, -0.1]), 0.5, 7.0
    )
    logp_y = jax.scipy.special.logsumexp(jnp.array([logp_true, logp_false]))

    print("log p(y=1.0, z=True ):", float(logp_true))
    print("log p(y=1.0, z=False):", float(logp_false))
    print("log p(y=1.0):", float(logp_y))


if __name__ == "__main__":
    main()
