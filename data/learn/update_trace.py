"""Trace update example mirroring the GenJAX tests.

Run with:
    source venv/bin/activate && python examples/learn/examples/update_trace.py
"""

import math

import jax.numpy as jnp

from genjax import gen, normal


@gen
def simple_normal() -> float:
    y1 = normal(0.0, 1.0) @ "y1"
    y2 = normal(0.0, 1.0) @ "y2"
    return y1 + y2


def main() -> None:
    trace = simple_normal.simulate()

    new_choices = {"y1": jnp.float32(2.0)}
    updated, log_delta, discarded = simple_normal.update(trace, new_choices)

    print("original choices:", trace.get_choices())
    print("updated choices:", updated.get_choices())
    print("log delta:", float(log_delta))
    print("discarded choices:", discarded)

    assert updated.get_choices()["y1"] == 2.0
    assert discarded["y1"] == trace.get_choices()["y1"]
    # weight = log P(new) - log P(old) = -new_score + old_score
    assert math.isclose(
        float(trace.get_score()) - float(log_delta),
        float(updated.get_score()),
        rel_tol=0.0,
        abs_tol=1e-5,
    )


if __name__ == "__main__":
    main()
