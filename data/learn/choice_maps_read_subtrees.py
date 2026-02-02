"""Read subtrees of a Trace choice map using nested dict access.

Run with:
    source venv/bin/activate && python examples/learn/examples/choice_maps_read_subtrees.py
"""

from __future__ import annotations

import jax.numpy as jnp

import genjax
from genjax import gen


@gen
def scalar(x: float) -> jnp.ndarray:
    z = genjax.flip(0.3) @ "z"
    y = genjax.normal(0.2 + 0.8 * x, 0.6) @ "y"
    return y


@gen
def batched(xs: jnp.ndarray) -> jnp.ndarray:
    return scalar.vmap(in_axes=(0,))(xs) @ "ys"


def main() -> None:
    xs = jnp.linspace(0.0, 1.0, 8)
    trace = batched.simulate(xs)
    choices = trace.get_choices()

    ys_vals = jnp.asarray(choices["ys"]["y"])
    z_vals = jnp.asarray(choices["ys"]["z"])

    print("ys values:", ys_vals.tolist())
    print("z mask   :", [bool(z) for z in z_vals])


if __name__ == "__main__":
    main()
