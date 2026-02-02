"""Work with vmapped GenJAX models and read choice arrays.

Run with:
    source venv/bin/activate && python examples/learn/examples/choice_maps_tuple_addresses.py
"""

from __future__ import annotations

import jax.numpy as jnp

import genjax
from genjax import gen


@gen
def scalar_model(x: float) -> jnp.ndarray:
    p = 0.2
    a = jnp.array([0.7, -0.4, 0.08])
    sigma_in, sigma_out = 0.4, 6.0
    z = genjax.flip(p) @ "z"
    mu = a[0] + a[1] * x + a[2] * (x**2)
    sigma = jnp.where(z, sigma_out, sigma_in)
    y = genjax.normal(mu, sigma) @ "y"
    return y


@gen
def vmapped(xs: jnp.ndarray) -> jnp.ndarray:
    return scalar_model.vmap(in_axes=(0,))(xs) @ "ys"


def main() -> None:
    xs = jnp.linspace(-2.0, 2.0, 10)
    trace = vmapped.simulate(xs)
    choices = trace.get_choices()

    ys_all = jnp.asarray(choices["ys"]["y"])
    z_all = jnp.asarray(choices["ys"]["z"])

    print("ys_all:", ys_all.tolist())
    print("z_all:", [bool(z) for z in z_all])

    # Building constraints for vmapped models uses arrays:
    # To constrain specific indices, use the full array
    print("constraint example: {'ys': {'y': jnp.array([...])}}")


if __name__ == "__main__":
    main()
