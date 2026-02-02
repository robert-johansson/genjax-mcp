"""Two-stage branching model: latent Bernoulli + noisy observation.

Run with:
    source venv/bin/activate && python examples/learn/examples/branching_gaussian.py
"""

from __future__ import annotations

import jax.numpy as jnp

from genjax import flip, gen, normal


@gen
def branching_gaussian(prob_branch: float, mu_false: float, mu_true: float, sigma: float):
    z = flip(prob_branch) @ "z"
    mu = jnp.where(z, mu_true, mu_false)
    obs = normal(mu, sigma) @ "obs"
    return obs


@gen
def noisy_branching(prob_branch: float, mu_false: float, mu_true: float, sigma: float, tau: float):
    base = branching_gaussian(prob_branch, mu_false, mu_true, sigma) @ "core"
    y = normal(base, tau) @ "y"
    return y


def simulate_once() -> None:
    trace = noisy_branching.simulate(0.3, -1.0, 2.0, 0.6, 0.1)
    choices = trace.get_choices()
    print("retval y:", float(trace.get_retval()))
    print({
        "z": bool(choices["core"]["z"]),
        "core.obs": float(choices["core"]["obs"]),
        "y": float(choices["y"]),
    })


def assess_given_obs() -> None:
    obs = {"y": 1.0, "core": {"obs": 0.8, "z": True}}
    logp = noisy_branching.assess(obs, 0.3, -1.0, 2.0, 0.6, 0.1)[0]
    print("log p(y=1.0, core.obs=0.8, z=True | ...):", float(logp))


def main() -> None:
    simulate_once()
    assess_given_obs()


if __name__ == "__main__":
    main()
