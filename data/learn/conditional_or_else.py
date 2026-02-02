"""Demonstrate Cond combinator for branching logic.

This replaces the old .or_else() API with the current Cond combinator.
Cond(A, B)(check, *args): when check=True -> A, when check=False -> B.
Both branches receive the same *args.

Run with:
    source venv/bin/activate && python examples/learn/examples/conditional_or_else.py
"""

from __future__ import annotations

import jax.numpy as jnp

from genjax import Cond, gen, normal


@gen
def high_branch(mu: float, sigma: float):
    return normal(mu, sigma) @ "x"


@gen
def low_branch(mu: float, sigma: float):
    return normal(mu, sigma) @ "x"


# Cond(A, B): check=True -> A (high), check=False -> B (low)
cond_fn = Cond(high_branch, low_branch)


@gen
def temperature_model(use_high: bool, mu_low: float, mu_high: float, sigma: float):
    # When use_high=True: high_branch runs with (mu_high, sigma)
    # When use_high=False: low_branch runs with (mu_high, sigma)
    # Both branches get the same args, so we pass the relevant mu
    mu = jnp.where(use_high, mu_high, mu_low)
    sample = cond_fn(jnp.bool_(use_high), mu, sigma) @ "temp"
    return sample


def main() -> None:
    cold = temperature_model.simulate(jnp.bool_(False), -5.0, +10.0, 1.0)
    print("use_high=False ->", float(cold.get_retval()))
    print("choices:", cold.get_choices())

    warm = temperature_model.simulate(jnp.bool_(True), -5.0, +10.0, 1.0)
    print("use_high=True  ->", float(warm.get_retval()))
    print("choices:", warm.get_choices())


if __name__ == "__main__":
    main()
