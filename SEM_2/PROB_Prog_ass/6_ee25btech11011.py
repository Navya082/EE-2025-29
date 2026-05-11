# Checking Central Limit Theorem using simulations
# We'll test this idea using a few different random variables:
#   Mode 0 - Bernoulli(p)
#   Mode 1 - Geometric(p)
#   Mode 2 - Exponential(lambda)
#   Mode 3 - A slightly tricky case: correlated Bernoulli samples (Markov chain)

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# set seed so output doesn't change every time
np.random.seed(42)


# SECTION 1: Sample Generators
# Each function returns an (N x n) matrix of samples

def generate_bernoulli(n, N, p):
    # Bernoulli(p): P(X=1) = p, P(X=0) = 1-p
    # Mean  = p
    # Variance = p*(1-p)
    return np.random.binomial(1, p, size=(N, n))


def generate_geometric(n, N, p):
    # Geometric(p): number of trials until first success
    # Mean  = 1/p
    # Variance = (1-p) / p^2
    return np.random.geometric(p, size=(N, n))


def generate_exponential(n, N, lam):
    # Exponential(lambda): inter-arrival time distribution
    # Mean  = 1/lambda
    # Variance = 1/lambda^2
    # Note: numpy uses 'scale = 1/lambda'
    return np.random.exponential(scale=1.0 / lam, size=(N, n))


def generate_correlated(n, N, p, q):
    # Mode 3: Correlated Bernoulli samples using a Markov chain
    # Instead of independent samples, each value now depends on the previous one.

    # The chain has two possible states: 0 and 1.
    # In the long run, probability of being in state 1 is p/(p+q)  [stationary dist]

    # Transition:
    #   - If current state is 0, next state is 1 with probability p
    #   - If current state is 1, it stays 1 with probability (1 - q)

    # First sample is drawn from the stationary distribution to avoid startup bias.

    stationary_p = p / (p + q)
    samples = np.zeros((N, n), dtype=int)

    # First column: draw from stationary distribution
    samples[:, 0] = np.random.binomial(1, stationary_p, size=N)

    # Subsequent columns: each depends on the previous
    for j in range(1, n):
        prev = samples[:, j - 1]
        transition_prob = np.where(prev == 0, p, 1 - q)
        samples[:, j] = np.random.binomial(1, transition_prob)

    return samples


# SECTION 2: Row Averaging
def row_averages(samples):
    # Yi = (1/n) * sum of n samples in each row
    # Returns N row averages
    return samples.mean(axis=1)


# SECTION 3: Plotting
def plot_clt(row_means, mu, sigma2, n, N, mode, label):
    # Plot histogram of sample means and compare with CLT prediction (Normal curve)
    # As n increases:
    #   mean stays around mu
    #   spread shrinks ~ sqrt(sigma^2 / n)

    num_bins = 50

    fig, ax = plt.subplots(figsize=(8, 5))

    # Histogram
    ax.hist(row_means, bins=num_bins, density=True,
            color='steelblue', alpha=0.6, label='Sample means histogram')

    # Gaussian overlay only for independent cases (modes 0, 1, 2)
    # Mode 3 is excluded as per the assignment requirement
    if mode in (0, 1, 2):
        std_clt = np.sqrt(sigma2 / n)
        x = np.linspace(row_means.min(), row_means.max(), 500)
        gaussian_pdf = norm.pdf(x, loc=mu, scale=std_clt)
        ax.plot(x, gaussian_pdf, 'r-', linewidth=2,
                label=f'N(μ={mu:.3f}, σ²/n={sigma2/n:.4f})')

    ax.set_title(f'CLT Demo — Mode {mode} ({label}), n={n}, N={N}')
    ax.set_xlabel('Sample Mean')
    ax.set_ylabel('Density')
    ax.legend()
    plt.tight_layout()

    fname = f'clt_mode{mode}_n{n}_N{N}.png'
    plt.savefig(fname, dpi=120)
    print(f"[Saved] {fname}")
    plt.show()


# SECTION 4: Main part
def main():
    if len(sys.argv) < 4:
        print("Usage: python 6_ee25btech11011.py <mode> <n> <N> <param> [<param2>]")
        sys.exit(1)

    mode = int(sys.argv[1])
    n    = int(sys.argv[2])
    N    = int(sys.argv[3])

    # Mode 0: Bernoulli
    if mode == 0:
        p = float(sys.argv[4])
        mu     = p              # E[X] = p
        sigma2 = p * (1 - p)   # Var[X] = p(1-p)

        print(f"Mode 0 | Bernoulli(p={p}) | n={n}, N={N}")
        print(f"  Theoretical mean = {mu:.4f}, variance = {sigma2:.4f}")

        samples   = generate_bernoulli(n, N, p)
        row_means = row_averages(samples)

        print(f"  Sample grand mean = {row_means.mean():.4f}")
        plot_clt(row_means, mu, sigma2, n, N, mode, f'Bernoulli(p={p})')

    # Mode 1: Geometric
    elif mode == 1:
        p = float(sys.argv[4])
        mu     = 1.0 / p            # E[X] = 1/p
        sigma2 = (1 - p) / p**2     # Var[X] = (1-p)/p^2

        print(f"Mode 1 | Geometric(p={p}) | n={n}, N={N}")
        print(f"  Theoretical mean = {mu:.4f}, variance = {sigma2:.4f}")

        samples   = generate_geometric(n, N, p)
        row_means = row_averages(samples)

        print(f"  Sample grand mean = {row_means.mean():.4f}")
        plot_clt(row_means, mu, sigma2, n, N, mode, f'Geometric(p={p})')

    # Mode 2: Exponential
    elif mode == 2:
        lam    = float(sys.argv[4])
        mu     = 1.0 / lam          # E[X] = 1/lambda
        sigma2 = 1.0 / lam**2       # Var[X] = 1/lambda^2

        print(f"Mode 2 | Exponential(λ={lam}) | n={n}, N={N}")
        print(f"  Theoretical mean = {mu:.4f}, variance = {sigma2:.4f}")

        samples   = generate_exponential(n, N, lam)
        row_means = row_averages(samples)

        print(f"  Sample grand mean = {row_means.mean():.4f}")
        plot_clt(row_means, mu, sigma2, n, N, mode, f'Exponential(λ={lam})')

    # Mode 3: Correlated Bernoulli
    elif mode == 3:
        p = float(sys.argv[4])
        q = float(sys.argv[5])

        # Stationary mean = p/(p+q), marginal variance = pq/(p+q)^2
        pi1    = p / (p + q)
        mu     = pi1
        sigma2 = p * q / (p + q)**2

        print(f"Mode 3 | Correlated Bernoulli(p={p}, q={q}) | n={n}, N={N}")
        print(f"  Stationary P(X=1) = {pi1:.4f}")
        print(f"  Theoretical mean = {mu:.4f}, marginal variance = {sigma2:.4f}")
        print("  Note: No Gaussian overlay for correlated mode")

        samples   = generate_correlated(n, N, p, q)
        row_means = row_averages(samples)

        print(f"  Sample grand mean = {row_means.mean():.4f}")
        plot_clt(row_means, mu, sigma2, n, N, mode,
                 f'Correlated Bernoulli(p={p}, q={q})')
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()