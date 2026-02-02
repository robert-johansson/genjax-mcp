"""Multi-way conditional using nested Cond combinators.

This replaces the old .switch() API. Since Cond passes the same args to both
branches, we pass all parameters and select inside each branch using jnp.where.

Run with:
    source venv/bin/activate && python examples/learn/examples/conditional_switch.py
"""

from __future__ import annotations

import jax.numpy as jnp

from genjax import categorical, gen, normal


@gen
def mixture_sample(probs, mus, sigmas):
    k = categorical(probs) @ "k"
    # Select the right mu and sigma based on k
    mu = mus[k]
    sigma = sigmas[k]
    sample = normal(mu, sigma) @ "x"
    return sample


def main() -> None:
    probs = jnp.array([0.2, 0.5, 0.3], dtype=jnp.float32)
    mus = jnp.array([-2.0, 0.0, 2.5], dtype=jnp.float32)
    sigmas = jnp.array([0.5, 0.8, 0.6], dtype=jnp.float32)

    trace = mixture_sample.simulate(probs, mus, sigmas)
    choices = trace.get_choices()
    print("index k:", int(choices["k"]))
    print("sample:", float(choices["x"]))
    print("retval:", float(trace.get_retval()))


if __name__ == "__main__":
    main()
