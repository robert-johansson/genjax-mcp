"""Build and merge choice maps (dicts) before assessing a generative function.

Run with:
    source venv/bin/activate && python examples/learn/examples/choice_maps_build_merge.py
"""

from __future__ import annotations

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
    obs_y = {"y": 2.0}
    obs_z = {"z": False}
    obs = {**obs_y, **obs_z}

    logp, _ = quad_with_outlier.assess(
        obs,
        1.25,
        0.1,
        jnp.array([0.5, -0.25, 0.1]),
        0.3,
        8.0,
    )

    print("obs_y:", obs_y)
    print("obs_z:", obs_z)
    print("merged obs:", obs)
    print("log p(y=2.0, z=False | args):", float(logp))


if __name__ == "__main__":
    main()
