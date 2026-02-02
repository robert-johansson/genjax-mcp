"""Vectorized GenJAX sampling with explicit random keys.

Run with:
    source venv/bin/activate && python examples/learn/examples/random_keys_and_vmap.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import genjax
from genjax import beta, gen, seed


@gen
def beta_bernoulli_process(alpha: float, beta_param: float) -> jnp.ndarray:
    p = beta(alpha, beta_param) @ "p"
    v = genjax.bernoulli(probs=p) @ "v"
    return v


def main() -> None:
    key = jax.random.key(0)
    keys = jax.random.split(key, 20)

    jitted_simulate = jax.jit(seed(beta_bernoulli_process.simulate))
    traces = jax.vmap(jitted_simulate, in_axes=(0, None, None))(keys, 2.0, 2.0)

    choices = traces.get_choices()
    samples = jnp.asarray(choices["v"], dtype=jnp.float32)
    probs = jnp.asarray(choices["p"], dtype=jnp.float32)

    print("first 5 draws:", choices["v"][:5])
    print("p samples mean:", float(jnp.mean(probs)))
    print("v mean (~ 0.5 expected):", float(jnp.mean(samples)))

    assert jnp.all((samples == 0) | (samples == 1))
    assert 0.2 < float(jnp.mean(samples)) < 0.8


if __name__ == "__main__":
    main()
