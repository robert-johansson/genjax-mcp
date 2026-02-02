"""Assess observations for addressed choices using plain dicts.

Run with:
    source venv/bin/activate && python examples/learn/examples/gf_assess_observation.py
"""

from __future__ import annotations

import jax
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
    y_obs = 2.0

    logp_false, _ = quadratic_with_outlier.assess(
        {"y": y_obs, "z": False}, 1.25, 0.1, jnp.array([0.5, -0.25, 0.1]), 0.3, 8.0
    )
    logp_true, _ = quadratic_with_outlier.assess(
        {"y": y_obs, "z": True}, 1.25, 0.1, jnp.array([0.5, -0.25, 0.1]), 0.3, 8.0
    )

    logp_y = jax.scipy.special.logsumexp(jnp.array([logp_false, logp_true]))

    print("log p(y=2.0, z=False):", float(logp_false))
    print("log p(y=2.0, z=True ):", float(logp_true))
    print("log p(y=2.0):", float(logp_y))


if __name__ == "__main__":
    main()
