# GenJAX Current API Reference (as of 2026-02)

This document captures the working API patterns for the current GenJAX codebase
at `/Users/robert/claude/genjax/`. Written from hands-on experience building
and running models, not from outdated documentation.

## Critical: This Is NOT the Old API

The `genjax_official/` directory contains an older version with a different API.
The examples in `genjax_official/learn/` do NOT run against the current codebase.
Key differences are documented below.

---

## 1. Imports

```python
from genjax import (
    # Core
    gen, Const, const, sel, Selection, seed, modular_vmap,
    # Combinators
    Scan, Cond,
    # Distributions
    normal, beta, flip, categorical, uniform, bernoulli,
    exponential, gamma, poisson, dirichlet, multivariate_normal,
    binomial, log_normal, student_t, laplace, half_normal,
    # Inference
    mh, mala, chain,
    rejuvenation_smc, init, change, extend, rejuvenate, resample,
    # State
    state, save, namespace,
)
```

## 2. Defining Generative Functions

```python
@gen
def model(x, n: Const[int]):
    """Arguments can be JAX arrays or Const[T] for static values."""
    mu = normal(0.0, 1.0) @ "mu"          # sample and address
    obs = normal(mu, 0.5) @ "obs"          # another addressed sample
    return obs
```

**Rules:**
- Use `@ "name"` to address random choices
- Addresses must be static strings (no f-strings, no dynamic names)
- No Python control flow inside `@gen` (use JAX: `jax.lax.cond`, `jnp.where`)
- Use `Const[int]` for values that must remain static through JAX tracing

## 3. Running Generative Functions

### Simulate (forward sampling)

```python
trace = seed(model.simulate)(key, x, const(n))
retval = trace.get_retval()
choices = trace.get_choices()       # {"mu": ..., "obs": ...}
score = trace.get_score()           # log(1/P(choices))
```

**Key pattern:** `seed` wraps the call and provides the PRNG key as first arg.
The remaining args are passed to the model unpacked (NOT as a tuple).

### Generate (constrained sampling / importance sampling)

```python
trace, weight = seed(model.generate)(key, constraints, x, const(n))
# weight = log[P(all_choices) / Q(unconstrained | constrained)]
```

**Constraints are plain dicts:**
```python
constraints = {"obs": jnp.float32(2.5)}                    # scalar
constraints = {"trials": {"action": observed_array}}        # nested (Scan)
constraints = {"mu": jnp.float32(0.3), "obs": jnp.float32(2.5)}  # multiple
```

### Assess (density evaluation)

```python
log_density, retval = model.assess(choices, x, const(n))
# choices must contain ALL addressed random variables
```

**No `seed` needed** — assess is deterministic (no sampling).

### Update (edit a trace)

```python
new_trace, weight, discard = seed(model.update)(key, trace, new_choices, *args)
# weight = log[P(new) / P(old)]
```

### Regenerate (resample selected addresses)

```python
new_trace, weight, discard = seed(model.regenerate)(key, trace, sel("mu"))
```

## 4. Constraints (Choice Maps)

**Current API uses plain Python dicts.** No `ChoiceMapBuilder`.

```python
# Simple
{"x": jnp.float32(1.0)}

# Nested (for hierarchical models)
{"sub_model": {"x": jnp.float32(1.0)}}

# For Scan (array of choices across timesteps)
{"trials": {"action": jnp.array([True, False, True, ...])}}

# Multiple addresses
{"mu": jnp.float32(0.0), "sigma": jnp.float32(1.0)}
```

**Old API (does NOT work):**
```python
# ChoiceMapBuilder -- NOT available in current GenJAX
C["x"].set(1.0)        # DOES NOT EXIST
C["x"] + C["y"]        # DOES NOT EXIST
```

## 5. Selections

Used for `regenerate` and MCMC to target specific addresses.

```python
sel("x")                    # select address "x"
sel("x") | sel("y")        # OR (either)
sel("x") & sel("y")        # AND (both) -- untested in examples
~sel("x")                  # NOT (complement) -- untested in examples
sel()                       # empty selection
Selection(AllSel())         # select everything
```

## 6. Combinators

### Scan (sequential iteration)

```python
@gen
def step(carry, x):
    noise = normal(0.0, 0.1) @ "noise"
    new_carry = carry + x + noise
    return new_carry, new_carry     # (new_carry, output)

@gen
def model(xs, n: Const[int]):
    scan_fn = Scan(step, length=n)
    final_carry, outputs = scan_fn(init_carry, xs) @ "steps"
    return outputs
```

**Length must be `Const[int]`** (static).

**Constraint structure for Scan:**
```python
# Choices inside scan are indexed automatically
{"steps": {"noise": jnp.array([0.1, 0.2, 0.3])}}
```

**Carry can be any pytree** (tuple, dict, nested arrays):
```python
carry = (V_table, learning_rate, temperature)  # tuple
carry = {"state": x, "count": n}                # dict
```

**Scanned input can be a pytree too:**
```python
# Tuple of arrays — each gets sliced along axis 0
trial_inputs = (samples, left_colors, right_colors, feedback_on)
scan_fn(carry, trial_inputs)  # step receives one element from each
```

### Cond (conditional branching)

```python
@gen
def branch_true():
    return normal(0.0, 1.0) @ "x"

@gen
def branch_false():
    return normal(10.0, 1.0) @ "x"

cond_fn = Cond(branch_true, branch_false)

@gen
def model(condition):
    result = cond_fn(condition) @ "branch"
    return result
```

**First arg to Cond result must be boolean condition.**
Both branches can use the same addresses (enhanced June 2025 merge API).

**Old API (does NOT work):**
```python
# or_else, switch -- NOT available in current GenJAX
branch1.or_else(branch2)(condition, args1, args2)    # DOES NOT EXIST
branch1.switch(branch2, branch3)(index, *args)       # DOES NOT EXIST
```

### Vmap (vectorization) and Repeat

```python
# Repeat: IID samples
obs = normal.repeat(n=20)(mu, sigma) @ "observations"

# Vmap for generative functions
vmapped_model = model.vmap(in_axes=(0, None))
```

## 7. Inference

### Importance Sampling (manual via vmap)

```python
def importance_sample(_):
    trace, weight = model.generate(constraints, *args)
    params = trace.get_choices()["param"]
    return params, weight

sampler = seed(modular_vmap(importance_sample, axis_size=5000))
samples, log_weights = sampler(key, jnp.arange(5000))

# Normalize
log_weights = log_weights - jax.scipy.special.logsumexp(log_weights)
weights = jnp.exp(log_weights)
posterior_mean = jnp.sum(samples * weights)
```

### MCMC (Metropolis-Hastings)

```python
from genjax import mh, sel, seed

# Single step: mh(trace, selection) -> new_trace
trace = seed(mh)(key, trace, sel("x"))

# Chain
from genjax import chain
traces = seed(chain)(key, mh, trace, sel("x"), const(1000))
```

**MH signature:** `mh(current_trace, selection) -> new_trace`
**MALA signature:** `mala(current_trace, selection, step_size) -> new_trace`

Both require `seed` wrapping for the PRNG key.

### SMC (Sequential Monte Carlo)

```python
from genjax import rejuvenation_smc, const, seed

result = seed(rejuvenation_smc)(
    key,
    model,              # step generative function
    proposal,           # proposal generative function (optional)
    const(mcmc_kernel), # rejuvenation kernel (optional)
    obs_sequence,       # {"obs": observations_array}
    initial_args,       # tuple of initial arguments
    const(n_particles),
)

log_ml = result.log_marginal_likelihood()
ess = result.effective_sample_size()
```

## 8. The seed Transformation

**`seed` is required** whenever a function contains PJAX probabilistic primitives
and you need to provide a PRNG key.

```python
# Wrapping GFI methods
trace = seed(model.simulate)(key, *args)
trace, w = seed(model.generate)(key, constraints, *args)

# Wrapping inference
new_trace = seed(mh)(key, trace, sel("x"))

# Wrapping vmap of probabilistic functions
result = seed(modular_vmap(fn, axis_size=N))(key, inputs)
```

**When NOT needed:**
- `model.assess(choices, *args)` — deterministic, no sampling
- `model.log_density(choices, *args)` — deterministic
- Pure JAX computations

## 9. Const Pattern

```python
from genjax import Const, const

# Creating
n = const(10)           # Const[int] with value 10

# Type annotation in @gen functions
@gen
def model(data, n_trials: Const[int]):
    scan_fn = Scan(step, length=n_trials)    # length must be Const
    ...

# Calling
model(data, const(20))

# Accessing value
n_trials.value  # -> 20
```

**Use `Const` for:**
- Scan lengths
- Any value that must remain static during JAX tracing
- Configuration that shouldn't become a JAX tracer

## 10. State Inspector

```python
from genjax import state, save, namespace

@state
def computation(x):
    intermediate = x ** 2
    save(squared=intermediate)          # named save
    result = intermediate + 1
    save(result)                        # leaf save
    return result

result, state_dict = computation(3.0)
# state_dict = {"squared": 9.0, 0: 10.0}

# Namespaces for organization
@state
def outer(x):
    y = namespace(inner_fn, "inner")(x)
    save(final=y)
    return y
# state_dict = {"inner": {...}, "final": ...}
```

Compatible with `jit`, `vmap`, `grad`, `scan`.

## 11. Trace Interface

```python
trace.get_retval()          # return value of the generative function
trace.get_choices()         # dict-like access to all random choices
trace.get_score()           # log(1/P(choices)) — negative log probability
trace.get_args()            # arguments the function was called with
trace.get_gen_fn()          # the generative function that produced this trace
trace.get_fixed_choices()   # choices preserving Fixed wrappers
```

## 12. Distribution API

All distributions follow the same pattern:

```python
# In @gen functions — sample and address
x = normal(mu, sigma) @ "x"

# GFI methods available on each distribution
normal.assess(sample, mu, sigma)        # -> (log_density, sample)
seed(normal.simulate)(key, mu, sigma)   # -> trace
normal.repeat(n=10)                     # -> Vmap for IID samples
```

## 13. Common Patterns

### Simulation = Model with Parameter Constraints

```python
# Same model for simulation AND inference
param_constraints = {"alpha": jnp.float32(0.2), "beta": jnp.float32(6.0)}
trace, _ = seed(model.generate)(key, param_constraints, *data_args)
simulated_data = trace.get_retval()
```

### Vectorized Schedule Generation

```python
# Use jax.vmap instead of Python loops
all_blocks = jax.vmap(generate_one_block)(keys)   # (n_blocks, 4, 3)
trials = all_blocks.reshape(-1, 3)                  # (n_trials, 3)
```

### Feedback Gating in Scan

```python
# Use jnp.where for conditional updates (no Python if/else)
V_updated = V.at[s, c].set(v_old + alpha * delta)
V_new = jnp.where(feedback_on, V_updated, V)       # gate by feedback flag
```

---

## API Migration from genjax_official

| Old Pattern | Current Pattern |
|---|---|
| `model.simulate(key, (args,))` | `seed(model.simulate)(key, *args)` |
| `model.importance(key, cm, (args,))` | `seed(model.generate)(key, constraints, *args)` |
| `ChoiceMapBuilder: C["x"].set(v)` | `{"x": v}` |
| `branch1.or_else(branch2)` | `Cond(branch1, branch2)` |
| `branch1.switch(b2, b3)` | Not available (chain Cond or use jax.lax.switch) |
| `trace.get_subtrace("x")` | `trace.get_choices()["x"]` |
| `ChoiceMap merge: cm1 + cm2` | `{**dict1, **dict2}` |

---

## Files That Run Successfully

**Current repo examples:**
- `examples/simple_intro/main.py` — coin flip IS
- `examples/simple_intro/match_to_sample.py` — MTS with RW learning
- `examples/faircoin/core.py` — framework comparison (core only, figs need seaborn)
- `examples/gol/core.py`, `localization/core.py`, `state_space/core.py` — import OK

**genjax_official (pure JAX only):**
- `learn/examples/beta_bernoulli_importance.py` — hand-coded IS
- `learn/examples/beta_bernoulli_smc.py` — hand-coded SMC
- `learn/examples/beta_bernoulli_mh.py` — hand-coded MH

**genjax_official (GenJAX-using) — ALL FAIL** due to old API calling convention.

---

## Verified Working Features (Smoke Tested)

1. `@gen` function definition with `@` addressing
2. `seed(model.simulate)` — forward sampling
3. `seed(model.generate)` — constrained generation with importance weights
4. `model.assess` — density evaluation
5. `Scan` — sequential iteration with carry
6. `Cond` — conditional branching
7. `seed(modular_vmap(...))` — vectorized importance sampling
8. `seed(model.regenerate)` — selective resampling
9. `seed(mh)` — Metropolis-Hastings MCMC
10. `sel()` and `sel("x") | sel("y")` — selections

## Known Gaps (Not Tested in Any Example)

- `filter` method on GFI
- `merge` method on GFI
- `log_density` convenience method
- `sel() & sel()` (AND combinator)
- `~sel()` (NOT combinator)
- ADEV gradient estimators in practice
- Variational inference in practice
