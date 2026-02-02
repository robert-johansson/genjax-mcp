"""Sample datasets from a Beta-Bernoulli generative function using repeat.

Run with:
    source venv/bin/activate && python examples/learn/examples/beta_bernoulli_generative.py
"""

from __future__ import annotations

import jax.numpy as jnp

import genjax
from genjax import gen


@gen
def beta_bernoulli_model(a: float, b: float, N: int) -> jnp.ndarray:
    theta = genjax.beta(a, b) @ "theta"
    ys = genjax.flip.repeat(n=N)(theta) @ "ys"
    return ys


def main() -> None:
    N = 10
    trace = beta_bernoulli_model.simulate(2.0, 2.0, N)

    choices = trace.get_choices()
    theta = float(choices["theta"])
    ys = jnp.asarray(choices["ys"])

    print("theta sample:", theta)
    print("first draws:", ys[:5])
    print("mean of draws:", float(jnp.mean(ys)))

    assert ys.shape == (N,)


if __name__ == "__main__":
    main()
