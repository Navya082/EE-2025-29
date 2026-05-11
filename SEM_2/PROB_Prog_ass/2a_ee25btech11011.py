import sys
import numpy as np
import matplotlib.pyplot as plt
import math

np.random.seed(42)

# read inputs
mode = int(sys.argv[1])
samples_file = sys.argv[2]

# read the uniform samples from csv using numpy - skip_header=1 skips the column name row
# no need for pandas here, numpy handles this just fine
U = np.genfromtxt(samples_file, delimiter=',', skip_header=1)
N = len(U)

# clip U slightly away from 0 and 1 to avoid log(0) blowing up
# this can happen if someone gives us exactly 0 or 1 in the file
U = np.clip(U, 1e-10, 1 - 1e-10)

# output text file - we'll append stuff here depending on the mode
out_txt = "invsampling_out_ee25btech11011.txt"

# helper function to save samples to csv without pandas
# just write header + values manually
def save_csv(filename, data, colname):
    with open(filename, 'w') as f:
        f.write(colname + '\n')
        for val in data:
            f.write(str(val) + '\n')

# MODE 0: Geometric(p) using inverse CDF 
# CDF of Geometric(p): F(k) = 1 - (1-p)^k for k = 1, 2, 3, ...
# inverting this: k = ceil(log(1-u) / log(1-p))
# this is the standard formula for inverse transform sampling on geometric

if mode == 0:
    p = float(sys.argv[3])

    # apply the inverse CDF formula to each uniform sample
    # we use log(1 - U) / log(1 - p) and take ceiling to get integer values
    geom_samples = np.ceil(np.log(1 - U) / np.log(1 - p)).astype(int)

    # sample mean - should be close to 1/p theoretically
    sample_mean = np.mean(geom_samples)

    # save the samples to csv
    p_str = str(p).replace(".", "p")  # e.g. 0.25 -> 0p25
    out_csv = f"invsampling_mode0_{p_str}_ee25btech11011.csv"
    save_csv(out_csv, geom_samples, "Geometric_samples")

    # write sample mean to text file
    with open(out_txt, "a") as f:
        f.write(f"Mode 0 (Geometric, p={p}): Sample mean = {sample_mean:.4f}, Theoretical mean = {1/p:.4f}\n")

    print(f"Mode 0 done. Sample mean = {sample_mean:.4f}, Expected = {1/p:.4f}")
    print(f"Samples saved to {out_csv}")


# MODE 1: Exponential(lambda) using inverse CDF 
# CDF of Exp(lambda): F(x) = 1 - e^(-lambda*x)
# inverting: x = -log(1 - u) / lambda

elif mode == 1:
    lam = float(sys.argv[3])

    # inverse CDF trick - pretty standard stuff
    exp_samples = -np.log(1 - U) / lam

    # number of bins = sqrt(N) as asked
    num_bins = int(math.sqrt(N))

    # plot the histogram
    plt.figure()
    plt.hist(exp_samples, bins=num_bins, density=True, label="Histogram")

    # overlay the theoretical pdf: f(x) = lambda * e^(-lambda * x)
    x_vals = np.linspace(0, np.max(exp_samples), 500)
    theoretical_pdf = lam * np.exp(-lam * x_vals)
    plt.plot(x_vals, theoretical_pdf, 'r-', label=f"Exp(lambda={lam}) PDF")

    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title(f"Exponential(lambda={lam}) samples via Inverse Sampling")
    plt.legend()

    # file naming: replace dot with 'p' in lambda value
    lam_str = str(lam).replace(".", "p")
    hist_file = f"invsampling_mode1_hist_{lam_str}_ee25btech11011.png"
    plt.savefig(hist_file)
    plt.close()

    # save samples
    out_csv = f"invsampling_mode1_{lam_str}_ee25btech11011.csv"
    save_csv(out_csv, exp_samples, "Exponential_samples")

    print(f"Mode 1 done. Histogram saved to {hist_file}")
    print(f"Samples saved to {out_csv}")


# MODE 2: Custom distribution X with piecewise CDF
# The CDF is:
#   F(x) = x^2 / 3      for x in [0, 1]
#   F(x) = 1/3          for 1 < x < 2      (flat region = no density here)
#   F(x) = (x+2)/6      for 2 <= x <= 4
#
# There's a jump at x=2: F(2-) = 1/3, F(2) = (2+2)/6 = 2/3
# So there's a point mass of 1/3 at x=2 (that's the discrete part!)
#
# Inverting:
#   if U in [0, 1/3]        -> from F(x) = x^2/3, so x = sqrt(3*U)    (x in [0,1])
#   if U in (1/3, 2/3)      -> x = 2 (the discrete mass)
#   if U in [2/3, 1]        -> from F(x) = (x+2)/6, so x = 6*U - 2    (x in [2,4])

elif mode == 2:
    # apply the piecewise inverse CDF using numpy masks (faster than a for loop)
    X_samples = np.zeros(N)

    mask0 = U <= 1/3                    # region 1: continuous part on [0,1]
    mask1 = (U > 1/3) & (U <= 2/3)     # region 2: discrete mass at x=2
    mask2 = U > 2/3                     # region 3: continuous part on [2,4]

    X_samples[mask0] = np.sqrt(3 * U[mask0])
    X_samples[mask1] = 2
    X_samples[mask2] = 6 * U[mask2] - 2

    # count how many times x = 2 appears (since U in (1/3, 2/3))
    count_2 = np.sum(X_samples == 2)

    # histogram with sqrt(N) bins
    num_bins = int(math.sqrt(N))

    plt.figure()
    plt.hist(X_samples, bins=num_bins, density=True, label="Histogram of X")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title("Samples of X with custom CDF (Inverse Sampling)")
    plt.legend()

    hist_file = "invsampling_mode2_hist_ee25btech11011.png"
    plt.savefig(hist_file)
    plt.close()

    # save samples to csv
    out_csv = "invsampling_mode2_ee25btech11011.csv"
    save_csv(out_csv, X_samples, "X_samples")

    # append outputs to text file
    with open(out_txt, "a") as f:
        f.write(f"Mode 2: Number of times X=2 appeared = {count_2}\n")
    print(f"Mode 2 done.")
    print(f"Number of times X=2 appeared: {count_2}")
    print(f"Samples saved to {out_csv}, Histogram saved to {hist_file}")