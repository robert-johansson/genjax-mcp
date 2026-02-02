"""Vectorize the branching Gaussian model with jax.vmap and seed.

Run with:
    source venv/bin/activate && python examples/learn/examples/branching_gaussian_batched.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from genjax import flip, gen, normal, seed


@gen
def branching_gaussian(prob_branch: float, mu_false: float, mu_true: float, sigma: float):
    z = flip(prob_branch) @ "z"
    mu = jnp.where(z, mu_true, mu_false)
    obs = normal(mu, sigma) @ "obs"
    return obs


@gen
def noisy_branching(prob_branch: float, mu_false: float, mu_true: float, sigma: float, tau: float):
    base = branching_gaussian(prob_branch, mu_false, mu_true, sigma) @ "core"
    y = normal(base, tau) @ "y"
    return y


def main() -> None:
    key = jax.random.key(0)
    n = 20000
    keys = jax.random.split(key, n)

    def one_ret(k):
        return seed(noisy_branching.simulate)(k, 0.3, -1.0, 2.0, 0.6, 0.1).get_retval()

    vmapped = jax.jit(jax.vmap(one_ret))
    _ = vmapped(keys[:1024])  # warmup compile
    ys = vmapped(keys)

    print("mean/std:", float(jnp.mean(ys)), float(jnp.std(ys)))
    print("P(y>0):", float(jnp.mean(ys > 0)))


if __name__ == "__main__":
    main()
