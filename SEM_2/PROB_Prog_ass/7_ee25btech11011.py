import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# setup:
#   Yi = hi * X + Wi
#
#   X  ~ N(mu0, sigma0^2)   → this is what we want to estimate
#   Wi ~ N(0, sigma_i^2)    → noise (independent)
#
# given Y1, Y2, ..., Yn → want E[X | all observations]
#
# since everything is Gaussian, the posterior is also Gaussian
# so MMSE = posterior mean (nice and simple here)
#
# formulas:
#   precision (1/variance):
#       1/sigma_n^2 = 1/sigma0^2 + sum(hi^2 / sigma_i^2)
#   estimate:
#       x_hat = sigma_n^2 * (mu0/sigma0^2 + sum(hi*Yi / sigma_i^2))
#
# basically:
#   - start from prior (mu0)
#   - each observation pulls the estimate depending on how noisy it is
#   - less noise → more weight
#   - as we add more data → estimate should stabilize

def mmse_estimate(mu0, sigma0_sq, Y_vals, sigma_sq_vals, h_vals):
    # start with prior info
    precision = 1.0 / sigma0_sq
    weighted_sum = mu0 / sigma0_sq

    estimates = []

    # add observations one by one
    for i in range(len(Y_vals)):
        # update using new observation
        precision    += (h_vals[i]**2) / sigma_sq_vals[i]
        weighted_sum += (h_vals[i] * Y_vals[i]) / sigma_sq_vals[i]

        # current estimate
        x_hat = weighted_sum / precision
        estimates.append(x_hat)

    return np.array(estimates)

def get_name_from_script():
    # trying to get roll number from filename
    script = os.path.basename(sys.argv[0])
    parts = script.replace('.py', '').split('_')
    if len(parts) >= 2:
        return '_'.join(parts[1:])
    return 'ee25btech11011'

def main():
    if len(sys.argv) < 4:
        print("Usage: python 7_ee25btech11011.py <mu0> <sigma0_sq> <samples.csv>")
        sys.exit(1)

    mu0       = float(sys.argv[1])
    sigma0_sq = float(sys.argv[2])
    csv_file  = sys.argv[3]

    # reading csv → columns: Yi, sigma_i^2, hi
    Y_vals        = []
    sigma_sq_vals = []
    h_vals        = []

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            Y_vals.append(float(row[0]))
            sigma_sq_vals.append(float(row[1]))
            h_vals.append(float(row[2]))

    Y_vals        = np.array(Y_vals)
    sigma_sq_vals = np.array(sigma_sq_vals)
    h_vals        = np.array(h_vals)

    N = len(Y_vals)
    print(f"Total samples N = {N}")

    # compute estimates for n = 1 to N
    estimates = mmse_estimate(mu0, sigma0_sq, Y_vals, sigma_sq_vals, h_vals)

    # plot how estimate changes as we add more observations
    # expecting it to settle after some time
    name = get_name_from_script()
    plt.figure(figsize=(10, 5))
    plt.scatter(np.arange(1, N+1), estimates, s=5, color='steelblue')

    plt.xlabel('n (number of observations used)')
    plt.ylabel('estimate x_hat')
    plt.title(f'MMSE estimate as n increases (mu0={mu0}, sigma0^2={sigma0_sq})')
    plt.tight_layout()

    fname = f'mmse_estimates_mu{mu0}_sigma{sigma0_sq}_{name}.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Plot saved: {fname}")

    # final estimate after all observations
    print(f"Final estimate after {N} samples: {estimates[-1]:.4f}")

if __name__ == "__main__":
    main()