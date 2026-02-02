"""Sequential Monte Carlo for the Beta–Bernoulli model.

Run with:
    python learn/examples/beta_bernoulli_smc.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def simulate_data(key: jax.Array, N: int, p_true: float) -> tuple[jax.Array, jnp.ndarray]:
    key, sub = jax.random.split(key)
    ys = jax.random.bernoulli(sub, p_true, (N,)).astype(jnp.float64)
    return key, ys


def ess(weights: jnp.ndarray) -> jnp.ndarray:
    w = weights / jnp.sum(weights)
    return 1.0 / jnp.sum(w * w)


def smc(
    key: jax.Array,
    ys: jnp.ndarray,
    a: float = 2.0,
    b: float = 2.0,
    num_particles: int = 2048,
    ess_threshold: float = 0.5,
) -> tuple[jax.Array, jnp.ndarray, jnp.ndarray, float, float]:
    key, sub = jax.random.split(key)
    thetas = jax.random.beta(sub, a, b, (num_particles,))
    weights = jnp.ones((num_particles,), dtype=jnp.float64) / num_particles

    def smc_step(carry, y):
        key, thetas, weights = carry
        log_inc = jnp.log(jnp.where(y == 1.0, thetas, 1.0 - thetas))
        max_inc = jnp.max(log_inc)
        weights = weights * jnp.exp(log_inc - max_inc)
        weights = weights / jnp.sum(weights)

        def do_resample(state):
            key, thetas, weights = state
            key, k_resample = jax.random.split(key)
            idx = jax.random.choice(k_resample, thetas.shape[0], (thetas.shape[0],), p=weights)
            thetas = thetas[idx]
            weights = jnp.ones_like(weights) / weights.shape[0]
            return key, thetas, weights

        state = (key, thetas, weights)
        state = jax.lax.cond(ess(weights) < ess_threshold * num_particles, do_resample, lambda x: x, state)
        return state, None

    (key, thetas, weights), _ = jax.lax.scan(smc_step, (key, thetas, weights), ys)
    mean = float(jnp.sum(weights * thetas))
    var = float(jnp.sum(weights * (thetas - mean) ** 2))
    return key, thetas, weights, mean, var


def main() -> None:
    key = jax.random.key(0)
    key, ys = simulate_data(key, N=50, p_true=0.6)
    key, thetas, weights, mean, var = smc(key, ys)

    print("data mean:", float(jnp.mean(ys)))
    print("posterior mean (SMC):", mean)
    print("posterior variance (SMC):", var)
    print("ESS after final update:", float(ess(weights)))

    assert thetas.shape == (2048,)
    assert weights.shape == (2048,)
    assert 0.0 <= mean <= 1.0


if __name__ == "__main__":
    main()
