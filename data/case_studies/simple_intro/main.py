"""Command-line interface for simple coin flip example."""

import argparse
import os
from .core import (
    run_importance_sampling, 
    compute_analytical_posterior,
    posterior_statistics
)
from .data import create_biased_dataset, summarize_flips
from .figs import (
    plot_posterior_comparison,
    plot_convergence,
    plot_prior_posterior
)


def main():
    """Run the simple coin flip example."""
    parser = argparse.ArgumentParser(
        description="GenJAX Simple Introduction: Coin Flip Inference"
    )
    
    # Data generation arguments
    parser.add_argument(
        "--n-flips", type=int, default=20,
        help="Number of coin flips to observe (default: 20)"
    )
    parser.add_argument(
        "--true-fairness", type=float, default=0.7,
        help="True coin fairness for data generation (default: 0.7)"
    )
    
    # Inference arguments
    parser.add_argument(
        "--n-samples", type=int, default=5000,
        help="Number of importance samples (default: 5000)"
    )
    
    # Figure selection
    parser.add_argument(
        "--posterior", action="store_true",
        help="Generate posterior comparison plot"
    )
    parser.add_argument(
        "--convergence", action="store_true",
        help="Generate convergence plot"
    )
    parser.add_argument(
        "--prior-posterior", action="store_true",
        help="Generate prior vs posterior plot"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate all plots"
    )
    
    # Output directory
    parser.add_argument(
        "--output-dir", type=str, default="examples/simple_intro/figs",
        help="Output directory for figures"
    )
    
    args = parser.parse_args()
    
    # Default to posterior plot if nothing specified
    if not any([args.posterior, args.convergence, args.prior_posterior, args.all]):
        args.posterior = True
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate observed data
    print(f"\n=== Generating Data ===")
    print(f"Number of flips: {args.n_flips}")
    print(f"True fairness: {args.true_fairness}")
    
    observed_flips = create_biased_dataset(args.n_flips, args.true_fairness)
    data_summary = summarize_flips(observed_flips)
    
    print(f"Observed heads: {data_summary['n_heads']}")
    print(f"Observed tails: {data_summary['n_tails']}")
    print(f"Empirical fairness: {data_summary['empirical_fairness']:.3f}")
    
    # Run importance sampling
    print(f"\n=== Running Inference ===")
    print(f"Number of importance samples: {args.n_samples}")
    
    samples, weights = run_importance_sampling(observed_flips, args.n_samples)
    stats = posterior_statistics(samples, weights)
    
    print(f"Posterior mean: {stats['mean']:.3f}")
    print(f"Posterior std: {stats['std']:.3f}")
    print(f"95% CI: [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]")
    
    # Compute analytical posterior for comparison
    post_alpha, post_beta = compute_analytical_posterior(observed_flips)
    analytical_mean = post_alpha / (post_alpha + post_beta)
    print(f"Analytical posterior: Beta({post_alpha:.0f}, {post_beta:.0f})")
    print(f"Analytical mean: {analytical_mean:.3f}")
    
    # Generate plots
    print(f"\n=== Generating Plots ===")
    
    if args.posterior or args.all:
        output_path = os.path.join(args.output_dir, 
            f"posterior_n{args.n_flips}_samples{args.n_samples}.pdf")
        plot_posterior_comparison(
            samples, weights, post_alpha, post_beta, 
            data_summary, output_path
        )
    
    if args.convergence or args.all:
        output_path = os.path.join(args.output_dir, 
            f"convergence_n{args.n_flips}.pdf")
        plot_convergence(observed_flips, output_path=output_path)
    
    if args.prior_posterior or args.all:
        output_path = os.path.join(args.output_dir,
            f"prior_posterior_n{args.n_flips}.pdf")
        plot_prior_posterior(observed_flips, output_path=output_path)
    
    print("\nDone! Check the output directory for generated plots.")
    print("\nTo learn more about GenJAX:")
    print("- Explore other examples in examples/")
    print("- Read the documentation in src/genjax/CLAUDE.md")
    print("- Try modifying the model in core.py")


if __name__ == "__main__":
    main()