"""Conditioned run using model.generate (importance sampling).

Run with:
    source venv/bin/activate && python examples/learn/examples/importance_conditioning.py
"""

import math

import jax.numpy as jnp

from genjax import gen, normal

SCALE = jnp.float32(1.0)


@gen
def latent_state(x: jnp.ndarray) -> jnp.ndarray:
    y = normal(x, SCALE) @ "y"
    z = normal(y, SCALE) @ "z"
    return y + z


def main() -> None:
    constraints = {"z": jnp.float32(4.0)}

    trace, log_weight = latent_state.generate(constraints, jnp.float32(0.0))
    full_score, _ = latent_state.assess(trace.get_choices(), jnp.float32(0.0))

    print("choices:", trace.get_choices())
    print("log weight:", float(log_weight))
    print("full assess score:", float(full_score))

    assert trace.get_choices()["z"] == jnp.float32(4.0)
    assert math.isclose(float(full_score), -float(trace.get_score()), abs_tol=1e-5)


if __name__ == "__main__":
    main()
