"""Demonstrate that GenJAX generative functions now accept unpacked arguments.

In the current GenJAX API, arguments are passed directly (not in a tuple).
This is a change from the older API where args had to be wrapped in a tuple.

Run with:
    source venv/bin/activate && python examples/learn/examples/tuple_args_required.py
"""

from __future__ import annotations

import genjax
from genjax import gen


@gen
def bernoulli_flag(prob):
    return genjax.bernoulli(probs=prob) @ "flag"


def main() -> None:
    # Current API: pass arguments directly (no tuple wrapping needed)
    trace = bernoulli_flag.simulate(0.5)
    print("simulate succeeded, retval:", trace.get_retval())

    # Multiple calls to show it works
    for p in [0.1, 0.5, 0.9]:
        trace = bernoulli_flag.simulate(p)
        print(f"  p={p} -> flag={bool(trace.get_retval())}")


if __name__ == "__main__":
    main()
