import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from scipy.stats import norm, geom

# get roll number from file name
def get_name_from_script():
    script = os.path.basename(sys.argv[0])
    parts = script.replace('.py', '').split('_')
    if len(parts) >= 2:
        return '_'.join(parts[1:])
    
    # fallback if format is unexpected
    return 'ee25btech11011'


# MODE 0: rejection sampling
# idea:
# generate half-normal using exponential proposal
# accept/reject using condition
# then randomly assign sign to get full normal
def rejection_sampling(M):
    # constant from theory
    c = np.sqrt(2 * np.e / np.pi)

    samples = np.zeros(M)
    iterations = np.zeros(M, dtype=int)

    for i in range(M):
        n_iter = 0

        while True:
            n_iter += 1

            # sample from exponential
            Y = np.random.exponential(scale=1.0)

            # uniform random number
            U = np.random.uniform(0, 1)

            # acceptance condition
            if U <= np.exp(-(Y - 1)**2 / 2):
                X = Y
                break

        # randomly flip sign
        sign = np.random.choice([-1, 1])
        samples[i] = sign * X

        # store iterations count
        iterations[i] = n_iter

    return samples, iterations, c

# MODE 1: polar transform
# generate R^2 and theta
# convert to cartesian to get normal samples
def polar_transform(M):
    # generate enough pairs
    n_pairs = math.ceil(M / 2)

    # R^2 ~ Exp(1/2)
    R2 = np.random.exponential(scale=2.0, size=n_pairs)

    # theta ~ uniform(0, 2pi)
    theta = np.random.uniform(0, 2 * np.pi, size=n_pairs)

    R = np.sqrt(R2)

    # convert to x,y
    Z1 = R * np.cos(theta)
    Z2 = R * np.sin(theta)

    # interleave values
    samples = np.empty(2 * n_pairs)
    samples[0::2] = Z1
    samples[1::2] = Z2

    # trim extra values
    samples = samples[:M]

    return samples

# plot histogram + actual gaussian curve
def plot_gaussian_histogram(samples, mean, variance, mode, name):
    M = len(samples)
    n_bins = int(math.sqrt(M))

    fig, ax = plt.subplots(figsize=(8, 5))

    # histogram
    ax.hist(samples, bins=n_bins, density=True,
            color='steelblue', edgecolor='white',
            alpha=0.8)

    # true gaussian
    sigma = math.sqrt(variance)
    x_vals = np.linspace(mean - 5*sigma, mean + 5*sigma, 500)
    pdf_vals = norm.pdf(x_vals, loc=mean, scale=sigma)

    ax.plot(x_vals, pdf_vals, 'r-', linewidth=2)

    mode_name = "Rejection Sampling" if mode == 0 else "Polar Transform"

    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.set_title(f'Gaussian via {mode_name} (M={M})')

    fig.tight_layout()

    fname = f"gauss_sampling_hist_{mode}_{name}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)

    print(f"Saved: {fname}")

# plot how many iterations rejection sampling took
def plot_iteration_histogram(iterations, c, mode, name):
    M = len(iterations)
    n_bins = int(math.sqrt(M))

    fig, ax = plt.subplots(figsize=(8, 5))

    # histogram of iterations
    ax.hist(iterations, bins=n_bins, density=True,
            color='mediumseagreen', edgecolor='white',
            alpha=0.8)

    # geometric distribution (expected)
    p_geom = 1.0 / c

    k_vals = np.arange(1, iterations.max() + 1)
    pmf_vals = geom.pmf(k_vals, p_geom)

    ax.plot(k_vals, pmf_vals, 'r-o',
            markersize=3, linewidth=1.5)

    ax.set_xlabel('Number of tries')
    ax.set_ylabel('Probability')
    ax.set_title(f'Iteration count (M={M})')
    fig.tight_layout()
    fname = f"gauss_iterpdf_hist_{mode}_{name}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")

def main():
    # fix seed so results stay same each run
    np.random.seed(42)

    if len(sys.argv) < 5:
        print("Usage: python script.py <mode> <mean> <variance> <M>")
        print("mode 0 → rejection sampling")
        print("mode 1 → polar transform")
        sys.exit(1)

    mode     = int(sys.argv[1])
    mean     = float(sys.argv[2])
    variance = float(sys.argv[3])
    M        = int(sys.argv[4])
    name     = get_name_from_script()

    print(f"Mode={mode}, mean={mean}, variance={variance}, M={M}")

    # mode 0
    if mode == 0:
        print("Using Rejection Sampling...")

        std_samples, iterations, c = rejection_sampling(M)

        # scale to required mean/variance
        sigma = math.sqrt(variance)
        final_samples = mean + sigma * std_samples

        print(f"Sample mean     : {final_samples.mean():.4f}")
        print(f"Sample variance : {final_samples.var():.4f}")
        print(f"Expected c      : {c:.4f}")
        print(f"Avg iterations  : {iterations.mean():.4f}")

        plot_gaussian_histogram(final_samples, mean, variance, mode, name)
        plot_iteration_histogram(iterations, c, mode, name)

    # mode 1
    elif mode == 1:
        print("Using Polar Transform...")

        std_samples = polar_transform(M)

        sigma = math.sqrt(variance)
        final_samples = mean + sigma * std_samples

        print(f"Sample mean     : {final_samples.mean():.4f}")
        print(f"Sample variance : {final_samples.var():.4f}")

        plot_gaussian_histogram(final_samples, mean, variance, mode, name)

    else:
        print("Invalid mode. Use 0 or 1.")
        sys.exit(1)


if __name__ == "__main__":
    main()