"""Programmatically build nested choice maps (dicts) from data.

Run with:
    source venv/bin/activate && python examples/learn/examples/choice_maps_programmatic_merge.py
"""

from __future__ import annotations


def main() -> None:
    targets = {0: -0.1, 2: 0.3, 5: 1.5}

    # Build nested dict constraints programmatically
    # For vmapped models, choices are accessed as {"address": array}
    # For individual indices, build nested dicts:
    pieces = {f"obs_{i}": {"y": val} for i, val in targets.items()}
    merged = {}
    for piece in pieces.values():
        merged.update(piece)

    print("pieces:", pieces)
    print("merged:", merged)

    # More practical: for a vmapped model with address "ys",
    # you'd typically constrain with an array:
    import jax.numpy as jnp

    ys_constraint = {"ys": {"y": jnp.array([-0.1, 0.0, 0.3, 0.0, 0.0, 1.5])}}
    print("array constraint:", ys_constraint)


if __name__ == "__main__":
    main()
