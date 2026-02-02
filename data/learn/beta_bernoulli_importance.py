"""Prior-importance sampler for the Beta–Bernoulli model.

Run with:
    python learn/examples/beta_bernoulli_importance.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def simulate_data(key: jax.Array, N: int, p_true: float) -> tuple[jax.Array, jnp.ndarray]:
    key, sub = jax.random.split(key)
    ys = jax.random.bernoulli(sub, p_true, (N,)).astype(jnp.float64)
    return key, ys


def bernoulli_loglik(theta: jnp.ndarray, ys: jnp.ndarray) -> jnp.ndarray:
    theta = jnp.clip(theta, 1e-12, 1 - 1e-12)
    return jnp.sum(ys * jnp.log(theta) + (1.0 - ys) * jnp.log1p(-theta))


def softmax_normalised(logw: jnp.ndarray) -> jnp.ndarray:
    m = jnp.max(logw)
    w = jnp.exp(logw - m)
    return w / jnp.sum(w)


def importance_sampling(
    key: jax.Array,
    ys: jnp.ndarray,
    a: float = 2.0,
    b: float = 2.0,
    num_particles: int = 4096,
) -> tuple[jax.Array, jnp.ndarray, jnp.ndarray, float, float]:
    key, sub = jax.random.split(key)
    thetas = jax.random.beta(sub, a, b, (num_particles,))
    logw = jax.vmap(lambda th: bernoulli_loglik(th, ys))(thetas)
    weights = softmax_normalised(logw)
    mean = float(jnp.sum(weights * thetas))
    var = float(jnp.sum(weights * (thetas - mean) ** 2))
    return key, thetas, weights, mean, var


def main() -> None:
    key = jax.random.key(0)
    key, ys = simulate_data(key, N=50, p_true=0.6)
    key, thetas, weights, mean, var = importance_sampling(key, ys)

    print("data mean:", float(jnp.mean(ys)))
    print("posterior mean (IS):", mean)
    print("posterior variance (IS):", var)
    print("effective sample size:", float(1.0 / jnp.sum(weights**2)))

    assert thetas.shape == (4096,)
    assert weights.shape == (4096,)
    assert 0.0 <= mean <= 1.0


if __name__ == "__main__":
    main()
