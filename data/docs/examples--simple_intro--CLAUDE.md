# CLAUDE.md - Simple Introduction Example

This file provides guidance to Claude Code when working with the simple coin flip introduction example.

## Overview

The simple_intro example is designed as a gentle introduction to GenJAX, demonstrating:
- Basic `@gen` function syntax
- Sampling from probability distributions
- Running importance sampling inference
- Comparing with analytical solutions
- Publication-quality visualization

This example uses a Beta-Bernoulli model for coin flip inference - one of the simplest possible probabilistic models with a known analytical solution.

## Directory Structure

```
examples/simple_intro/
├── CLAUDE.md           # This file - guidance for Claude Code
├── __init__.py         # Python package marker
├── core.py             # Model definitions and inference
├── data.py             # Data generation utilities
├── figs.py             # Visualization code
├── main.py             # Command-line interface
└── figs/               # Output directory for plots
```

## Code Organization

### `core.py` - Model and Inference

**Key Functions**:
- **`coin_model()`**: Single coin flip Beta-Bernoulli model
- **`coin_model_multiple(n_flips: Const[int])`**: Multiple flips model
- **`run_importance_sampling()`**: Importance sampling inference
- **`compute_analytical_posterior()`**: Exact Beta posterior computation
- **`posterior_statistics()`**: Weighted statistics from samples

**Model Specification**:
```
Prior: Beta(1, 1) = Uniform(0, 1)
Likelihood: Bernoulli(fairness)
Posterior: Beta(1 + n_heads, 1 + n_tails)
```

### `data.py` - Data Generation

**Key Functions**:
- **`generate_coin_flips()`**: Generate random flips with true fairness
- **`create_biased_dataset()`**: Create dataset with exact proportion
- **`summarize_flips()`**: Compute summary statistics

### `figs.py` - Visualization

Uses the shared `examples.viz` module for GenJAX Research Visualization Standards (GRVS).

**Key Functions**:
- **`plot_posterior_comparison()`**: IS samples vs analytical posterior
- **`plot_convergence()`**: Show convergence with sample size
- **`plot_prior_posterior()`**: Visualize Bayesian updating

### `main.py` - CLI Interface

**Default Behavior**: Generates posterior comparison plot
**Command Line Arguments**:
- `--n-flips`: Number of observations (default: 20)
- `--true-fairness`: True coin fairness (default: 0.7)
- `--n-samples`: Number of importance samples (default: 5000)
- `--posterior`: Generate posterior comparison
- `--convergence`: Generate convergence plot
- `--prior-posterior`: Generate prior vs posterior
- `--all`: Generate all plots

## Usage Examples

### Basic Usage
```bash
# Run with defaults (20 flips, 0.7 fairness, posterior plot)
pixi run python -m examples.simple_intro.main

# Generate all plots
pixi run python -m examples.simple_intro.main --all
```

### Custom Parameters
```bash
# More data, different fairness
pixi run python -m examples.simple_intro.main --n-flips 50 --true-fairness 0.3

# Test convergence with different sample sizes
pixi run python -m examples.simple_intro.main --convergence --n-flips 100
```

### Understanding the Output
```
=== Generating Data ===
Number of flips: 20
True fairness: 0.7
Observed heads: 14
Observed tails: 6
Empirical fairness: 0.700

=== Running Inference ===
Number of importance samples: 5000
Posterior mean: 0.682
Posterior std: 0.093
95% CI: [0.500, 0.841]
Analytical posterior: Beta(15, 7)
Analytical mean: 0.682
```

## Key Concepts Demonstrated

### 1. GenJAX Model Definition
```python
@gen
def coin_model():
    fairness = beta(1.0, 1.0) @ "fairness"  # Prior
    outcome = flip(fairness) @ "flip"        # Likelihood
    return outcome
```

### 2. Importance Sampling
- Samples from prior (proposal)
- Weights by likelihood of observations
- Approximates posterior distribution

### 3. Vectorized Inference
```python
imp_vmap = seed(vmap(importance_sample, axis_size=n_samples))
```
Uses JAX vectorization for efficient parallel sampling

### 4. Analytical Validation
Beta-Bernoulli has closed-form posterior:
- Prior: Beta(α, β)
- Data: h heads, t tails
- Posterior: Beta(α + h, β + t)

## Common Modifications

### Different Priors
```python
# Informative prior (believes coin is fair)
fairness = beta(10.0, 10.0) @ "fairness"

# Skeptical prior (believes coin is biased)
fairness = beta(0.5, 0.5) @ "fairness"
```

### Multiple Coins
```python
@gen
def two_coins_model():
    fair1 = beta(1.0, 1.0) @ "fairness1"
    fair2 = beta(1.0, 1.0) @ "fairness2"
    
    flip1 = flip(fair1) @ "flip1"
    flip2 = flip(fair2) @ "flip2"
    
    return (flip1, flip2)
```

### Hierarchical Model
```python
@gen
def hierarchical_coin():
    # Population-level fairness
    pop_mean = beta(2.0, 2.0) @ "population_mean"
    
    # Individual coin fairness
    fairness = beta(pop_mean * 10, (1 - pop_mean) * 10) @ "fairness"
    
    # Observations
    flips = flip.repeat(n=20)(fairness) @ "flips"
    
    return flips
```

## Performance Notes

- Importance sampling is exact in the limit
- For this simple model, 1000 samples give good approximation
- JAX compilation makes repeated runs very fast
- Vectorization enables efficient GPU usage

## Learning Path

After this example, explore:
1. **faircoin**: More complex Beta-Bernoulli with framework comparisons
2. **curvefit**: Regression with normal distributions
3. **localization**: Sequential Monte Carlo for state estimation
4. **gol**: Inverse graphics with discrete models

## Troubleshooting

### Import Errors
Ensure you're in the GenJAX environment:
```bash
pixi shell
python -m examples.simple_intro.main
```

### No Plots Generated
Check the output directory exists:
```bash
mkdir -p examples/simple_intro/figs
```

### Numerical Issues
If weights are all zero, check:
- Data matches model support
- No impossible observations
- Proper use of log-space arithmetic