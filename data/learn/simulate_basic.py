"""Minimal GenJAX simulate example.

Run with:
    source venv/bin/activate && python examples/learn/examples/simulate_basic.py
"""

from genjax import gen, bernoulli, normal


@gen
def model(x: float) -> float:
    """Normal + Bernoulli toy model."""
    y = normal(x, 1.0) @ "y"
    z = bernoulli(probs=0.7) @ "z"
    return y + z


def main() -> None:
    trace = model.simulate(0.0)

    print("return value:", trace.get_retval())
    print("choices:", trace.get_choices())
    print("score:", float(trace.get_score()))

    assert trace.get_retval() == trace.get_choices()["y"] + trace.get_choices()["z"]


if __name__ == "__main__":
    main()
