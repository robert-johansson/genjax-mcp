"""Match-to-sample task with Rescorla-Wagner learning.

A cognitive science model of a 2AFC matching-to-sample task where a participant
learns stimulus-response associations through trial-and-error feedback.

Protocol (baseline -> training -> test):
    - Baseline: trials without feedback (no learning)
    - Training: trials with corrective feedback (RW updates)
    - Test: trials without feedback (no learning)

Stimuli:
    - Sample colors: yellow, blue
    - Comparison colors: red, green (presented left/right, randomized)
    - Correct rule: yellow -> red, blue -> green

Agent model (Rescorla-Wagner):
    - alpha ~ Beta(2, 2)           -- learning rate
    - inv_temp ~ Uniform(1, 10)    -- inverse temperature
    - eps ~ Beta(1, 19)            -- lapse rate (~5% prior mean)
    - V[sample, color] table tracks association strengths
    - Each trial: action ~ Bernoulli(eps * 0.5 + (1-eps) * sigmoid(inv_temp * (vR - vL)))
    - V update on feedback trials: V[s,c] += alpha * (reward - V[s,c])

Inference recovers alpha, inv_temp, and eps from observed actions
using importance sampling.
"""

import jax
import jax.numpy as jnp
import jax.random as jrand
from genjax import (
    beta,
    flip,
    gen,
    Scan,
    Const,
    const,
    seed,
    modular_vmap as vmap,
    uniform,
)

# Color encoding
YELLOW, BLUE, RED, GREEN = 0, 1, 2, 3
NUM_COLORS = 4
COLOR_NAMES = {0: "yellow", 1: "blue", 2: "red", 3: "green"}


@gen
def mts_step(carry, trial_input):
    """Single trial of match-to-sample.

    The agent sees a sample color and two comparison colors (left/right),
    then chooses left or right. On feedback trials, V-table is updated.
    """
    V, alpha, inv_temp, eps = carry
    sample, left_color, right_color, fb_on = trial_input

    # Compute value-based probability of choosing right
    vL = V[sample, left_color]
    vR = V[sample, right_color]
    p_right_greedy = jax.nn.sigmoid(inv_temp * (vR - vL))

    # Incorporate lapse: with prob eps choose randomly, else use softmax
    p_right = eps * 0.5 + (1.0 - eps) * p_right_greedy

    # Sample action (True = choose right)
    choose_right = flip(p_right) @ "action"

    # Determine chosen color and correctness
    chosen_color = jax.lax.select(choose_right, right_color, left_color)
    target = jax.lax.select(sample == YELLOW, RED, GREEN)
    correct = chosen_color == target
    reward = correct.astype(jnp.float32)

    # RW update (only on feedback trials)
    v_old = V[sample, chosen_color]
    delta = reward - v_old
    V_updated = V.at[sample, chosen_color].set(v_old + alpha * delta)
    V_new = jnp.where(fb_on, V_updated, V)

    return (V_new, alpha, inv_temp, eps), choose_right


@gen
def mts_model(trial_inputs, n_trials: Const[int]):
    """Full MTS session with latent learning parameters.

    Args:
        trial_inputs: Tuple of (samples, left_colors, right_colors, feedback_on),
                      each of shape (T,).
        n_trials: Static number of trials.
    """
    alpha = beta(2.0, 2.0) @ "alpha"
    inv_temp = uniform(1.0, 10.0) @ "inv_temp"
    eps = beta(1.0, 19.0) @ "eps"

    V0 = jnp.zeros((NUM_COLORS, NUM_COLORS), dtype=jnp.float32)
    scan_fn = Scan(mts_step, length=n_trials)
    _, actions = scan_fn((V0, alpha, inv_temp, eps), trial_inputs) @ "trials"
    return actions


def _shuffled_block(key):
    """Return one balanced block: the 4 trial types in shuffled order.

    Trial types:
        (yellow, red,   green)  -- correct = left
        (yellow, green, red)    -- correct = right
        (blue,   red,   green)  -- correct = right
        (blue,   green, red)    -- correct = left
    """
    block = jnp.array([
        [YELLOW, RED, GREEN],
        [YELLOW, GREEN, RED],
        [BLUE, RED, GREEN],
        [BLUE, GREEN, RED],
    ], dtype=jnp.int32)
    perm = jrand.permutation(key, 4)
    return block[perm]


def generate_schedule(key, n_baseline_blocks=1, n_training_blocks=3, n_test_blocks=1):
    """Generate a balanced experimenter-controlled trial schedule.

    Each block contains all 4 trial types (2 samples x 2 side arrangements)
    in shuffled order. This ensures balanced exposure within every block.

    Fully functional — uses jax.vmap over keys instead of Python loops.

    Args:
        key: JAX random key.
        n_baseline_blocks: Number of 4-trial blocks without feedback.
        n_training_blocks: Number of 4-trial blocks with feedback.
        n_test_blocks: Number of 4-trial blocks without feedback.

    Returns:
        Tuple of (samples, left_colors, right_colors, feedback_on).
    """
    n_total_blocks = n_baseline_blocks + n_training_blocks + n_test_blocks
    keys = jrand.split(key, n_total_blocks)

    # Vectorized block generation: (n_blocks, 4, 3) -> (n_blocks * 4, 3)
    all_blocks = jax.vmap(_shuffled_block)(keys)
    trials = all_blocks.reshape(-1, 3)
    samples = trials[:, 0]
    left_colors = trials[:, 1]
    right_colors = trials[:, 2]

    # Feedback schedule: 0 for baseline/test blocks, 1 for training blocks
    block_feedback = jnp.concatenate([
        jnp.zeros(n_baseline_blocks, dtype=jnp.int32),
        jnp.ones(n_training_blocks, dtype=jnp.int32),
        jnp.zeros(n_test_blocks, dtype=jnp.int32),
    ])
    feedback_on = jnp.repeat(block_feedback, 4)

    return samples, left_colors, right_colors, feedback_on


def simulate_participant(key, schedule, true_alpha, true_inv_temp, true_eps):
    """Simulate a participant by running the generative model with fixed parameters.

    Uses mts_model.generate with parameter constraints so the simulation
    uses exactly the same code path as inference.

    Returns:
        Boolean array of actions (True = chose right).
    """
    n_trials = len(schedule[0])
    param_constraints = {
        "alpha": jnp.float32(true_alpha),
        "inv_temp": jnp.float32(true_inv_temp),
        "eps": jnp.float32(true_eps),
    }
    trace, _ = seed(mts_model.generate)(
        key, param_constraints, schedule, const(n_trials)
    )
    return trace.get_retval()


def run_importance_sampling(schedule, observed_actions, n_samples=5000):
    """Run importance sampling to infer learning parameters."""
    n_trials = len(observed_actions)
    constraints = {"trials": {"action": observed_actions}}

    def importance_sample(_):
        trace, weight = mts_model.generate(
            constraints, schedule, const(n_trials)
        )
        choices = trace.get_choices()
        return choices["alpha"], choices["inv_temp"], choices["eps"], weight

    imp_vmap = seed(vmap(importance_sample, axis_size=n_samples))
    key = jrand.key(42)
    alpha_samples, inv_temp_samples, eps_samples, log_weights = imp_vmap(
        key, jnp.arange(n_samples)
    )

    # Normalize weights
    log_weights = log_weights - jax.scipy.special.logsumexp(log_weights)
    weights = jnp.exp(log_weights)

    return alpha_samples, inv_temp_samples, eps_samples, weights


def plot_block_accuracy(schedule, observed_actions, n_baseline_blocks, n_training_blocks,
                        n_test_blocks, output_path="examples/simple_intro/figs/block_accuracy.png"):
    """Plot percent correct per block, with phases color-coded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples, left_colors, right_colors, _ = schedule
    n_total_blocks = n_baseline_blocks + n_training_blocks + n_test_blocks

    # Compute accuracy per block
    block_acc = []
    for b in range(n_total_blocks):
        start = b * 4
        n_correct = 0
        for t in range(start, start + 4):
            chose_right = bool(observed_actions[t])
            chosen_color = int(right_colors[t]) if chose_right else int(left_colors[t])
            target = RED if int(samples[t]) == YELLOW else GREEN
            n_correct += int(chosen_color == target)
        block_acc.append(n_correct / 4 * 100)

    # Assign phase colors
    colors = []
    for b in range(n_total_blocks):
        if b < n_baseline_blocks:
            colors.append("#7f7f7f")      # gray
        elif b < n_baseline_blocks + n_training_blocks:
            colors.append("#1f77b4")      # blue
        else:
            colors.append("#2ca02c")      # green

    # Phase labels for legend
    phase_labels = []
    for b in range(n_total_blocks):
        if b < n_baseline_blocks:
            phase_labels.append("baseline")
        elif b < n_baseline_blocks + n_training_blocks:
            phase_labels.append("training")
        else:
            phase_labels.append("test")

    fig, ax = plt.subplots(figsize=(7, 4))

    # Plot bars
    x = range(1, n_total_blocks + 1)
    bars = ax.bar(x, block_acc, color=colors, edgecolor="white", linewidth=1.5)

    # Chance line
    ax.axhline(50, color="black", linestyle="--", linewidth=1, alpha=0.5, label="chance (50%)")

    # Phase boundary lines
    if n_baseline_blocks > 0:
        ax.axvline(n_baseline_blocks + 0.5, color="black", linestyle=":", linewidth=1, alpha=0.4)
    if n_test_blocks > 0:
        boundary = n_baseline_blocks + n_training_blocks + 0.5
        ax.axvline(boundary, color="black", linestyle=":", linewidth=1, alpha=0.4)

    # Legend (one entry per phase)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#7f7f7f", label="baseline (no feedback)"),
        Patch(facecolor="#1f77b4", label="training (feedback)"),
        Patch(facecolor="#2ca02c", label="test (no feedback)"),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1, alpha=0.5, label="chance"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="lower right")

    ax.set_xlabel("Block", fontsize=13, fontweight="bold")
    ax.set_ylabel("% Correct", fontsize=13, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {output_path}")


def replay_v_tables(schedule, observed_actions, alpha):
    """Replay the V-table evolution given observed actions.

    Purely functional — uses jax.lax.scan to reconstruct the V-table
    after every trial. Returns V-tables of shape (T+1, NUM_COLORS, NUM_COLORS),
    where index 0 is the initial state and index t is the state after trial t.
    """
    samples, left_colors, right_colors, feedback_on = schedule

    def step(V, t):
        s = samples[t]
        lc = left_colors[t]
        rc = right_colors[t]
        fb_on = feedback_on[t]
        chose_right = observed_actions[t]

        chosen_color = jax.lax.select(chose_right, rc, lc)
        target = jax.lax.select(s == YELLOW, RED, GREEN)
        reward = (chosen_color == target).astype(jnp.float32)

        v_old = V[s, chosen_color]
        delta = reward - v_old
        V_updated = V.at[s, chosen_color].set(v_old + alpha * delta)
        V_new = jnp.where(fb_on, V_updated, V)
        return V_new, V_new

    V0 = jnp.zeros((NUM_COLORS, NUM_COLORS), dtype=jnp.float32)
    _, all_V = jax.lax.scan(step, V0, jnp.arange(len(observed_actions)))

    # Prepend V0 so index 0 = initial, index t = after trial t
    return jnp.concatenate([V0[None], all_V], axis=0)


def print_v_table(V, label):
    """Print the relevant 2x2 portion of the V-table."""
    print(f"\n  {label}")
    print(f"  {'':>8} {'red':>8} {'green':>8}")
    print(f"  {'yellow':>8} {float(V[YELLOW, RED]):>8.3f} {float(V[YELLOW, GREEN]):>8.3f}")
    print(f"  {'blue':>8} {float(V[BLUE, RED]):>8.3f} {float(V[BLUE, GREEN]):>8.3f}")


def weighted_stats(samples, weights):
    """Compute weighted mean and std."""
    mean = jnp.sum(samples * weights)
    variance = jnp.sum(weights * (samples - mean) ** 2)
    std = jnp.sqrt(variance)
    return float(mean), float(std)


def main():
    # True parameters
    true_alpha = 0.2
    true_inv_temp = 6.0
    true_eps = 0.05

    # Phase sizes (in balanced blocks of 4 trials each)
    n_baseline_blocks = 1
    n_training_blocks = 3
    n_test_blocks = 1
    n_baseline = n_baseline_blocks * 4
    n_training = n_training_blocks * 4
    n_test = n_test_blocks * 4
    n_total = n_baseline + n_training + n_test

    # Generate balanced schedule
    key = jrand.key(0)
    key, subkey = jrand.split(key)
    schedule = generate_schedule(subkey, n_baseline_blocks, n_training_blocks, n_test_blocks)
    samples, left_colors, right_colors, _ = schedule

    # Simulate participant using the generative model itself
    key, subkey = jrand.split(key)
    observed_actions = simulate_participant(
        subkey, schedule,
        true_alpha=true_alpha, true_inv_temp=true_inv_temp, true_eps=true_eps,
    )

    # Print header
    print("=" * 70)
    print("Match-to-Sample with Rescorla-Wagner Learning")
    print("=" * 70)
    print(f"\nCorrect rule: yellow -> red, blue -> green")
    print(f"True parameters: alpha={true_alpha}, inv_temp={true_inv_temp}, eps={true_eps}")
    print(f"Schedule: {n_baseline_blocks}x4 baseline + {n_training_blocks}x4 training + {n_test_blocks}x4 test = {n_total} trials")
    print()

    # Print trial-by-trial data
    header = f"{'Trial':>5} {'Phase':>10} {'Sample':>8} {'Left':>6} {'Right':>6} {'Choice':>8} {'Correct':>8}"
    print(header)
    print("-" * len(header))

    phase_correct = {"baseline": 0, "training": 0, "test": 0}
    phase_total = {"baseline": 0, "training": 0, "test": 0}

    for t in range(n_total):
        s = int(samples[t])
        lc = int(left_colors[t])
        rc = int(right_colors[t])
        chose_right = bool(observed_actions[t])
        chosen_color = rc if chose_right else lc
        target = RED if s == YELLOW else GREEN
        correct = chosen_color == target
        choice_side = "right" if chose_right else "left"
        choice_str = f"{choice_side}({COLOR_NAMES[chosen_color]})"

        if t < n_baseline:
            phase = "baseline"
        elif t < n_baseline + n_training:
            phase = "training"
        else:
            phase = "test"

        phase_correct[phase] += int(correct)
        phase_total[phase] += 1

        print(
            f"{t + 1:>5} {phase:>10} {COLOR_NAMES[s]:>8} "
            f"{COLOR_NAMES[lc]:>6} {COLOR_NAMES[rc]:>6} "
            f"{choice_str:>8} {'yes' if correct else 'no':>8}"
        )

    # Phase summary
    print(f"\nPhase summary:")
    for phase in ["baseline", "training", "test"]:
        n_correct = phase_correct[phase]
        n_total_phase = phase_total[phase]
        pct = n_correct / n_total_phase if n_total_phase > 0 else 0
        print(f"  {phase:>10}: {n_correct}/{n_total_phase} correct ({pct:.0%})")

    # Replay V-table evolution and print at block boundaries
    n_total_blocks = n_baseline_blocks + n_training_blocks + n_test_blocks
    all_V = replay_v_tables(schedule, observed_actions, true_alpha)

    print(f"\nV-table evolution (V[sample, comparison]):")
    print(f"  Correct associations: V[yellow,red] and V[blue,green] should grow.")
    for b in range(n_total_blocks + 1):
        trial_idx = b * 4
        if b == 0:
            label = "Initial state (before any trials)"
        elif b <= n_baseline_blocks:
            label = f"After baseline block {b}"
        elif b <= n_baseline_blocks + n_training_blocks:
            label = f"After training block {b - n_baseline_blocks}"
        else:
            label = f"After test block {b - n_baseline_blocks - n_training_blocks}"
        print_v_table(all_V[trial_idx], label)

    # Plot block accuracy
    plot_block_accuracy(schedule, observed_actions,
                        n_baseline_blocks, n_training_blocks, n_test_blocks)

    # Run inference
    print("\n" + "=" * 70)
    print("Importance Sampling Inference")
    print("=" * 70)

    n_samples = 5000
    print(f"\nRunning importance sampling with {n_samples} samples...")
    alpha_samples, inv_temp_samples, eps_samples, weights = run_importance_sampling(
        schedule, observed_actions, n_samples=n_samples
    )

    ess = 1.0 / jnp.sum(weights ** 2)
    print(f"Effective sample size: {float(ess):.1f} / {n_samples}")

    alpha_mean, alpha_std = weighted_stats(alpha_samples, weights)
    inv_temp_mean, inv_temp_std = weighted_stats(inv_temp_samples, weights)
    eps_mean, eps_std = weighted_stats(eps_samples, weights)

    print(f"\n{'Parameter':<15} {'True':>8} {'Post. Mean':>12} {'Post. Std':>12}")
    print("-" * 50)
    print(f"{'alpha':<15} {true_alpha:>8.3f} {alpha_mean:>12.3f} {alpha_std:>12.3f}")
    print(f"{'inv_temp':<15} {true_inv_temp:>8.3f} {inv_temp_mean:>12.3f} {inv_temp_std:>12.3f}")
    print(f"{'eps':<15} {true_eps:>8.3f} {eps_mean:>12.3f} {eps_std:>12.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
