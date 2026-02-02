"""Core model and inference for simple coin flip example."""

import jax
import jax.numpy as jnp
import jax.random as jrand
from genjax import beta, flip, gen, Const, const
from genjax import seed, modular_vmap as vmap
from scipy.stats import beta as scipy_beta


@gen
def coin_model():
    """Simple Beta-Bernoulli model for coin flips.
    
    This model represents our belief about a coin's fairness:
    - Prior: Beta(1, 1) - uniform belief over fairness
    - Likelihood: Bernoulli - coin flips
    """
    # Sample the coin's fairness from a uniform prior
    fairness = beta(1.0, 1.0) @ "fairness"
    
    # Flip the coin
    outcome = flip(fairness) @ "flip"
    
    return outcome


@gen
def coin_model_multiple(n_flips: Const[int]):
    """Beta-Bernoulli model with multiple coin flips.
    
    Args:
        n_flips: Number of coin flips to model
        
    Returns:
        Array of coin flip outcomes
    """
    # Sample the coin's fairness
    fairness = beta(1.0, 1.0) @ "fairness"
    
    # Flip the coin multiple times
    outcomes = flip.repeat(n=n_flips.value)(fairness) @ "flips"
    
    return outcomes


def run_importance_sampling(observed_flips, n_samples=1000):
    """Run importance sampling to infer coin fairness.
    
    Args:
        observed_flips: Array of observed coin flips (0 or 1)
        n_samples: Number of importance samples
        
    Returns:
        samples: Array of posterior samples for fairness
        weights: Importance weights (log space)
    """
    n_flips = len(observed_flips)
    constraints = {"flips": observed_flips}
    
    # Define importance sampling function
    def importance_sample(_):
        trace, weight = coin_model_multiple.generate(
            constraints, const(n_flips)
        )
        fairness = trace.get_choices()["fairness"]
        return fairness, weight
    
    # Run importance sampling with vectorization
    imp_vmap = seed(vmap(importance_sample, axis_size=n_samples))
    key = jrand.key(42)
    samples, log_weights = imp_vmap(key, jnp.arange(n_samples))
    
    # Normalize weights
    log_weights = log_weights - jax.scipy.special.logsumexp(log_weights)
    weights = jnp.exp(log_weights)
    
    return samples, weights


def compute_analytical_posterior(observed_flips, prior_alpha=1.0, prior_beta=1.0):
    """Compute the analytical Beta posterior.
    
    For Beta-Bernoulli model, the posterior is:
    Beta(alpha + sum(flips), beta + n - sum(flips))
    
    Args:
        observed_flips: Array of observed coin flips
        prior_alpha: Alpha parameter of Beta prior
        prior_beta: Beta parameter of Beta prior
        
    Returns:
        posterior_alpha: Alpha parameter of posterior
        posterior_beta: Beta parameter of posterior
    """
    n_heads = jnp.sum(observed_flips)
    n_tails = len(observed_flips) - n_heads
    
    posterior_alpha = prior_alpha + n_heads
    posterior_beta = prior_beta + n_tails
    
    return posterior_alpha, posterior_beta


def posterior_statistics(samples, weights):
    """Compute weighted statistics from importance samples.
    
    Args:
        samples: Posterior samples
        weights: Importance weights (normalized)
        
    Returns:
        Dictionary with mean, std, and credible intervals
    """
    # Weighted mean
    mean = jnp.sum(samples * weights)
    
    # Weighted variance
    variance = jnp.sum(weights * (samples - mean)**2)
    std = jnp.sqrt(variance)
    
    # Weighted quantiles (approximate)
    sorted_idx = jnp.argsort(samples)
    sorted_samples = samples[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cumsum_weights = jnp.cumsum(sorted_weights)
    
    # Find 2.5% and 97.5% quantiles
    idx_025 = jnp.searchsorted(cumsum_weights, 0.025)
    idx_975 = jnp.searchsorted(cumsum_weights, 0.975)
    
    ci_lower = sorted_samples[idx_025]
    ci_upper = sorted_samples[idx_975]
    
    return {
        "mean": mean,
        "std": std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }