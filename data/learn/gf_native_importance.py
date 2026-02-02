"""Use model.generate for importance-weighted constrained sampling.

Run with:
    source venv/bin/activate && python examples/learn/examples/gf_native_importance.py
"""

from __future__ import annotations

import genjax
from genjax import gen


@gen
def toy_coin(prob: float) -> bool:
    return genjax.flip(prob) @ "y"


def main() -> None:
    obs = {"y": True}
    trace, logw = toy_coin.generate(obs, 0.4)

    print("log weight:", float(logw))
    print("trace return value:", bool(trace.get_retval()))


if __name__ == "__main__":
    main()
