"""Metropolis–Hastings sampler for the Beta–Bernoulli model.

Run with:
    python learn/examples/beta_bernoulli_mh.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def simulate_data(key: jax.Array, N: int, p_true: float) -> tuple[jax.Array, jnp.ndarray]:
    key, sub = jax.random.split(key)
    ys = jax.random.bernoulli(sub, p_true, (N,)).astype(jnp.float64)
    return key, ys


def log_beta_pdf(theta: jnp.ndarray, a: float, b: float) -> jnp.ndarray:
    theta = jnp.clip(theta, 1e-12, 1 - 1e-12)
    return (a - 1.0) * jnp.log(theta) + (b - 1.0) * jnp.log1p(-theta) - jax.scipy.special.betaln(a, b)


def bernoulli_loglik(theta: jnp.ndarray, ys: jnp.ndarray) -> jnp.ndarray:
    theta = jnp.clip(theta, 1e-12, 1 - 1e-12)
    return jnp.sum(ys * jnp.log(theta) + (1.0 - ys) * jnp.log1p(-theta))


def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / (1.0 + jnp.exp(-x))


def mh_random_walk(
    key: jax.Array,
    ys: jnp.ndarray,
    a: float = 2.0,
    b: float = 2.0,
    iters: int = 8000,
    burn: int = 2000,
    step: float = 0.08,
) -> tuple[jax.Array, jnp.ndarray, float, float]:
    z = jnp.array(0.0, dtype=jnp.float64)
    theta = sigmoid(z)
    logp = log_beta_pdf(theta, a, b) + bernoulli_loglik(theta, ys)
    samples = []

    for _ in range(iters):
        key, k_prop, k_u = jax.random.split(key, 3)
        z_prop = z + step * jax.random.normal(k_prop, ())
        theta_prop = sigmoid(z_prop)
        logp_prop = log_beta_pdf(theta_prop, a, b) + bernoulli_loglik(theta_prop, ys)
        acc_prob = jnp.minimum(1.0, jnp.exp(logp_prop - logp))
        accept = jax.random.uniform(k_u, ()) < acc_prob
        z = jnp.where(accept, z_prop, z)
        theta = jnp.where(accept, theta_prop, theta)
        logp = jnp.where(accept, logp_prop, logp)
        samples.append(theta)

    draws = jnp.array(samples[burn:], dtype=jnp.float64)
    mean = float(jnp.mean(draws))
    var = float(jnp.var(draws))
    return key, draws, mean, var


def main() -> None:
    key = jax.random.key(0)
    key, ys = simulate_data(key, N=50, p_true=0.6)
    key, draws, mean, var = mh_random_walk(key, ys)

    print("data mean:", float(jnp.mean(ys)))
    print("posterior mean (MH):", mean)
    print("posterior variance (MH):", var)
    print("accepted draws:", draws.shape[0])

    assert draws.ndim == 1
    assert draws.shape[0] == 8000 - 2000
    assert 0.0 <= mean <= 1.0


if __name__ == "__main__":
    main()
