"""Data generation utilities for coin flip example."""

import jax.numpy as jnp
import jax.random as jrand


def generate_coin_flips(key, n_flips, true_fairness=0.5):
    """Generate synthetic coin flip data.
    
    Args:
        key: JAX random key
        n_flips: Number of flips to generate
        true_fairness: True probability of heads
        
    Returns:
        flips: Array of coin flips (0 or 1)
    """
    flips = jrand.bernoulli(key, p=true_fairness, shape=(n_flips,))
    return flips.astype(jnp.float32)


def create_biased_dataset(n_flips, bias_towards_heads=0.7):
    """Create a dataset with known bias.
    
    Args:
        n_flips: Total number of flips
        bias_towards_heads: Proportion of heads
        
    Returns:
        flips: Array with specified proportion of heads/tails
    """
    n_heads = int(n_flips * bias_towards_heads)
    n_tails = n_flips - n_heads
    
    # Create array with exact proportion
    flips = jnp.concatenate([
        jnp.ones(n_heads),
        jnp.zeros(n_tails)
    ])
    
    # Shuffle for realism (using fixed seed for reproducibility)
    key = jrand.key(42)
    shuffled_idx = jrand.permutation(key, len(flips))
    
    return flips[shuffled_idx]


def summarize_flips(flips):
    """Summarize coin flip data.
    
    Args:
        flips: Array of coin flips
        
    Returns:
        Dictionary with summary statistics
    """
    n_flips = len(flips)
    n_heads = int(jnp.sum(flips))
    n_tails = n_flips - n_heads
    empirical_fairness = n_heads / n_flips
    
    return {
        "n_flips": n_flips,
        "n_heads": n_heads,
        "n_tails": n_tails,
        "empirical_fairness": empirical_fairness,
        "flips": flips
    }