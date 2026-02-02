"""Chain a stateful generative function with Scan combinator.

Run with:
    source venv/bin/activate && python examples/learn/examples/gf_scan.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import genjax
from genjax import Const, Scan, gen, seed


@gen
def linear_state_step(x: float, noise_std: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    xp = genjax.normal(0.9 * x, noise_std) @ "x_next"
    return xp, xp


def main() -> None:
    key = jax.random.key(0)
    n_steps = 20
    noise = 0.1

    scanned = Scan(linear_state_step, length=Const(n_steps))
    noise_seq = jnp.full((n_steps,), noise)

    trace = seed(scanned.simulate)(key, 1.0, noise_seq)
    carried, history = trace.get_retval()

    history = jnp.asarray(history)

    print("first states:", history[:5].tolist())
    print("final state:", float(history[-1]))
    print("carried out:", float(jnp.asarray(carried)))


if __name__ == "__main__":
    main()
